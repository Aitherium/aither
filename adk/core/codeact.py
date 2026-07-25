"""CodeAct strategy: LLM writes Python code, executes in persistent namespace.

This module implements an iterative code execution loop where:
1. The model emits Python code cells (extracted from markdown code blocks)
2. Each cell is validated with CodeValidator before execution
3. Validation errors are fed back to the model (not executed)
4. Code executes in a persistent namespace (variables survive across cells)
5. Agent tools are available as callables within the code
6. Execution terminates when return_result(value) is called or max_steps is hit

The loop follows the AgentLoop protocol and supports:
- Per-loop model override (like ReActLoop and PredictLoop)
- Tracing and span generation
- Optional memory recall/remember hooks
- Bounded output previews via pformat if available

**SECURITY NOTICE:** In-process code execution is BEST-EFFORT validation only,
not a security boundary. The CodeValidator catches common unsafe patterns but
requires OS-level sandboxing (container/VM) for real isolation.
"""

from __future__ import annotations

import asyncio
import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from adk.core.agent import Agent, AgentResult
from adk.core.logging import get_logger
from adk.core.model import Message, ModelBackend
from adk.core.tool import Tool, ToolResult
from adk.core.validator import CodeValidator

_log = get_logger("codeact")


# =============================================================================
# Code Extraction
# =============================================================================

def _extract_code_block(text: str) -> str | None:
    """Extract Python code from a markdown code block.

    Looks for ```python ... ``` or just ``` ... ```.
    Returns the first code block found, or None if none found.
    """
    # Try ```python variant first
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic ``` variant
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try without code fence (plain Python text)
    lines = text.strip().split("\n")
    # Only treat as bare code if it looks like Python (has = or def or import, etc.)
    if lines and any(
        any(keyword in line for keyword in ["=", "def ", "class ", "import ", "from "])
        for line in lines[:3]  # Check first 3 lines
    ):
        return text.strip()
    return None


# =============================================================================
# Return Result Signal
# =============================================================================

class _ReturnResultSignal(Exception):
    """Signal raised when return_result() is called in user code.

    This is an exception-based mechanism that allows return_result() to work
    anywhere in the code and immediately terminate execution.
    """

    def __init__(self, result: Any) -> None:
        self.result = result
        super().__init__(f"return_result() called with: {result!r}")


def _make_return_result(ns: dict[str, Any]) -> Any:
    """Create a return_result() function that raises _ReturnResultSignal."""

    def return_result(value: Any) -> None:
        """Terminate the cell and return a value to the model."""
        raise _ReturnResultSignal(value)

    return return_result


# =============================================================================
# Code Execution Result
# =============================================================================

@dataclass
class _ExecutionResult:
    """Result of executing a single code cell."""

    stdout: str = ""
    stderr: str = ""
    exception: Exception | None = None
    return_signal: _ReturnResultSignal | None = None
    # Captured locals after execution (for debugging)
    captured_locals: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if the cell executed without error or return signal."""
        return self.exception is None and self.return_signal is None


# =============================================================================
# CodeActLoop Strategy
# =============================================================================

class CodeActLoop:
    """Iterative code execution loop (CodeAct strategy).

    Args:
        max_steps: Maximum reasoning steps before terminating (default 8).
        model: Optional per-loop model override (None → use agent.model).
        max_preview_chars: If set, bound observations to this many chars using
            pformat(). None (default) = unbounded str() behavior.
        cell_timeout: Wall-clock timeout per cell in seconds (default 5.0).
        max_output_bytes: Maximum observation output size in bytes (default 10000).
    """

    def __init__(
        self,
        *,
        max_steps: int = 8,
        model: ModelBackend | None = None,
        max_preview_chars: int | None = None,
        cell_timeout: float = 5.0,
        max_output_bytes: int = 10000,
    ) -> None:
        self.max_steps = max_steps
        self.model = model  # Per-loop model override (NOOA pattern)
        self.max_preview_chars = max_preview_chars
        self.cell_timeout = cell_timeout
        self.max_output_bytes = max_output_bytes
        self._validator = CodeValidator()

    async def run(self, agent: Agent, prompt: str) -> AgentResult:
        """Execute the CodeAct loop.

        Args:
            agent: The agent to run
            prompt: The user prompt / task

        Returns:
            AgentResult with output, messages, tool_calls, steps, and finish_reason
        """
        tracer = agent.tracer
        model = self.model or agent.model
        tools_by_name = {t.name: t for t in agent.tools}

        # Build system prompt
        system = self._build_system_prompt(agent, tools_by_name)

        # Optionally inject memory recall
        if getattr(agent, "recall_memory", False):
            mem = agent.memory
            if hasattr(mem, "context_block") and hasattr(mem, "constraints_block"):
                try:
                    constr = await mem.constraints_block()
                    recalled = await mem.context_block(prompt)
                    extra = "\n\n".join(b for b in (constr, recalled) if b)
                    if extra:
                        system = f"{system}\n\n{extra}"
                except Exception as e:  # noqa: BLE001 — memory is best-effort
                    _log.warning("agent.memory.recall_failed", extra={"err": str(e)})

        messages: list[Message] = [
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ]
        tool_calls: list[dict[str, Any]] = []
        output = ""
        finish = "max_steps"

        # Persistent namespace for code execution
        namespace: dict[str, Any] = self._build_namespace(agent, tools_by_name)

        with tracer.span("agent.run", agent=agent.name, prompt_len=len(prompt)) as run_span:
            for step in range(1, self.max_steps + 1):
                with tracer.span("agent.llm", step=step) as llm_span:
                    resp = await model.generate(messages)
                    llm_span.set_attr("model", resp.model)
                    llm_span.set_attr("finish_reason", resp.finish_reason or "")
                messages.append(Message(role="assistant", content=resp.text))

                # Try to extract and execute a code cell
                code = _extract_code_block(resp.text)

                if code is None:
                    # No code block found; treat as final answer
                    output = resp.text.strip()
                    finish = "stop"
                    break

                # Validate the code
                validation_issues = self._validator.validate(code)
                if validation_issues:
                    # Feed validation errors back to the model
                    error_msg = self._format_validation_errors(validation_issues)
                    messages.append(Message(role="user", content=error_msg))
                    continue

                # Execute the code
                with tracer.span("agent.code_exec", step=step, code_len=len(code)) as exec_span:
                    try:
                        result = await self._execute_code(code, namespace)
                        exec_span.set_attr("success", result.success)
                    except Exception as e:  # noqa: BLE001 — surface to model
                        _log.exception("codeact.code_exec_failed")
                        result = _ExecutionResult(
                            exception=RuntimeError(f"Execution timeout or critical error: {e}")
                        )
                        exec_span.set_attr("success", False)

                # Check for return_result signal
                if result.return_signal is not None:
                    output = str(result.return_signal.result)
                    finish = "stop"
                    tool_calls.append(
                        {
                            "step": step,
                            "name": "return_result",
                            "args": {"value": result.return_signal.result},
                            "result": ToolResult(ok=True, value=output),
                        }
                    )
                    break

                # Format observation (stdout + stderr + exception)
                observation = self._format_observation(result)
                messages.append(Message(role="user", content=observation))

                # Record tool call (for debugging/tracing)
                tool_calls.append(
                    {
                        "step": step,
                        "name": "execute_python",
                        "args": {"code": code},
                        "result": ToolResult(ok=result.success, value=observation),
                    }
                )

            run_span.set_attr("steps", step)
            run_span.set_attr("finish_reason", finish)

        # Optionally persist interaction to memory
        if getattr(agent, "remember_interactions", False) and output:
            mem = agent.memory
            if hasattr(mem, "remember"):
                try:
                    await mem.remember(
                        f"Q: {prompt}\nA: {output}", role="interaction",
                    )
                except Exception as e:  # noqa: BLE001 — best-effort
                    _log.warning("agent.memory.remember_failed", extra={"err": str(e)})

        return AgentResult(
            output=output,
            messages=messages,
            tool_calls=tool_calls,
            steps=step,
            finish_reason=finish,
        )

    def _build_system_prompt(self, agent: Agent, tools_by_name: dict[str, Tool]) -> str:
        """Build the system prompt for the CodeAct loop."""
        tool_list = (
            "\n".join(f"- {t.name}: {t.description}" for t in agent.tools)
            or "  (no tools)"
        )
        return f"""\
You are {agent.name}.
{agent.instructions or ""}

You have access to these tools:
{tool_list}

You are in a Jupyter-like Python REPL. Your task is to:
1. Write Python code cells to solve the problem
2. Code executes in a persistent namespace (variables persist across cells)
3. Emit code in markdown code blocks: ```python ... ```
4. Print results with print() or return them at the end of a cell
5. When done, call return_result(value) from within your code

Available in every cell:
- self: the agent instance
- All tools (as functions: {', '.join(tools_by_name.keys()) if tools_by_name else 'none'})
- Standard Python builtins
- Libraries: asyncio, json, re, collections

Example:
```python
# Analyze data
data = [1, 2, 3, 4, 5]
result = sum(data) / len(data)
print(f"Average: {{result}}")

# Call a tool
tool_result = await tool_name(arg1=value1, arg2=value2)
print(tool_result)

# When done, return the final answer
return_result(result)
```

Do NOT:
- Use eval(), exec(), compile(), __import__(), input()
- Access __builtins__, globals(), locals(), __dict__, __class__
- Use from X import *
- Attach functions to self (self.foo = fn)
"""

    def _build_namespace(
        self, agent: Agent, tools_by_name: dict[str, Tool]
    ) -> dict[str, Any]:
        """Build the namespace for code execution."""
        namespace: dict[str, Any] = {
            "self": agent,
            "asyncio": asyncio,
            "return_result": _make_return_result({}),  # Populated below
        }
        # Add all tools as callables
        namespace.update(tools_by_name)
        return namespace

    def _format_validation_errors(self, issues: list[Any]) -> str:
        """Format validation issues as a user-facing error message."""
        lines = ["ERROR: Code validation failed. Please fix the following issues:\n"]
        for issue in issues:
            lines.append(f"  Line {issue.line}: {issue.message}")
            if issue.fix_hint:
                lines.append(f"    Hint: {issue.fix_hint}")
        return "\n".join(lines)

    async def _execute_code(
        self, code: str, namespace: dict[str, Any]
    ) -> _ExecutionResult:
        """Execute a code cell in the persistent namespace.

        Captures stdout/stderr and exceptions. Handles return_result() signal.
        Executes with a wall-clock timeout.

        The code is wrapped in an async function to support await. Local variables
        defined in the code are captured and updated back to the namespace after
        execution, so they persist across cells.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            # Run code with timeout
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    # Inject return_result and locals capture dict into namespace
                    namespace["return_result"] = _make_return_result(namespace)
                    namespace["__repl_captured_locals__"] = {}

                    # Wrap user code in an async function so await works
                    wrapped_code = self._wrap_user_code(code)

                    # Compile the wrapper function
                    compiled = compile(wrapped_code, "<cell>", "exec")

                    # Define the async function in the namespace
                    exec(compiled, namespace)

                    # Call the async wrapper
                    await asyncio.wait_for(
                        namespace["__user_code__"](),
                        timeout=self.cell_timeout,
                    )

                    # Capture locals from the function into persistent namespace
                    # This makes variables defined in the cell persist across cells
                    captured = namespace.pop("__repl_captured_locals__", {})
                    namespace.update(captured)

                except _ReturnResultSignal as sig:
                    # Capture locals before returning
                    captured = namespace.pop("__repl_captured_locals__", {})
                    namespace.update(captured)
                    return _ExecutionResult(
                        stdout=stdout_buf.getvalue(),
                        stderr=stderr_buf.getvalue(),
                        return_signal=sig,
                    )
                except asyncio.TimeoutError:
                    return _ExecutionResult(
                        stderr=f"Execution timeout (>{self.cell_timeout}s)",
                        exception=TimeoutError("Cell execution exceeded timeout"),
                    )
                except SyntaxError as e:
                    return _ExecutionResult(
                        stderr=f"SyntaxError: {e}",
                        exception=e,
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    return _ExecutionResult(
                        stderr=tb,
                        exception=e,
                    )

        except Exception as e:
            return _ExecutionResult(
                stderr=f"Outer error: {e}",
                exception=e,
            )

        return _ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
        )

    def _wrap_user_code(self, code: str) -> str:
        """Wrap user code in an async function with locals capture.

        This allows the user code to:
        - Use await for async tools
        - Access variables from the persistent namespace (via globals)
        - Define new variables that persist via __repl_captured_locals__
        - Call return_result() to terminate

        The wrapper:
        1. Wraps code in an async function to enable await
        2. Captures all local variables after execution
        3. Updates __repl_captured_locals__ so they persist

        Returns Python code that defines an async __user_code__() function.
        """
        # Indent user code by 4 spaces
        lines = code.split("\n")
        indented_lines = [f"    {line}" if line.strip() else "" for line in lines]
        indented_code = "\n".join(indented_lines)

        # Wrap with finally block to capture locals
        wrapper = f"""\
async def __user_code__():
{indented_code}
    # Capture all local variables into namespace for persistence
    __repl_captured_locals__.update({{
        k: v for k, v in locals().items()
        if not k.startswith('_') and k != 'return_result'
    }})
"""
        return wrapper

    def _format_observation(self, result: _ExecutionResult) -> str:
        """Format execution result as a model-facing observation."""
        parts: list[str] = []

        if result.stdout.strip():
            parts.append("Output:\n" + result.stdout)

        if result.stderr.strip():
            parts.append("Errors/Traceback:\n" + result.stderr)

        if result.exception and not result.stderr.strip():
            # Exception without stderr (shouldn't happen, but be safe)
            parts.append(f"Exception: {type(result.exception).__name__}: {result.exception}")

        if not parts:
            parts.append("(No output)")

        observation = "\n".join(parts)

        # Truncate if too large
        if len(observation) > self.max_output_bytes:
            observation = (
                observation[: self.max_output_bytes]
                + f"\n... (truncated, {len(observation)} bytes total)"
            )

        # Use pformat if available and requested
        if self.max_preview_chars is not None:
            try:
                from adk.agentdoc import truncating_pformat
                observation = truncating_pformat(
                    observation,
                    max_chars=self.max_preview_chars,
                )
            except ImportError:
                pass

        return observation
