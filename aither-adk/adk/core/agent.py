"""Agent + reasoning loop.

The :class:`Agent` is intentionally tiny. The reasoning loop is pluggable
via the :class:`AgentLoop` protocol; the default is :class:`ReActLoop` which
implements the standard ReAct pattern (Thought -> Action -> Observation).

Full tool-call orchestration with model-native function calling lands in
slice B once we have a real backend. The slice-A default loop is text-based
and deliberately simple, so any backend works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from adk.core.capability import (
    Capability,
    CapabilityContext,
    use_context,
)
from adk.core.logging import get_logger
from adk.core.memory import InMemoryStore, Memory
from adk.core.model import Message, ModelBackend
from adk.core.tool import Tool, ToolResult
from adk.core.trace import Tracer, get_tracer

_log = get_logger("agent")


@dataclass(slots=True)
class AgentResult:
    """One agent run."""

    output: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    finish_reason: str = "stop"


class AgentLoop(Protocol):
    """A pluggable reasoning loop."""

    async def run(self, agent: Agent, prompt: str) -> AgentResult: ...


# ---------------------------------------------------------------------------
# Default ReAct loop
# ---------------------------------------------------------------------------

_REACT_SYS_TEMPLATE = """\
You are {name}.
{instructions}

You have access to these tools:
{tool_list}

Use the ReAct pattern. Each turn, emit EXACTLY ONE of:

  Thought: <your reasoning>
  Action: <tool_name>
  Action Input: <json arguments>

...then wait for the Observation. When you have the final answer, emit:

  Final Answer: <your answer>
"""

_ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_\-]+)\s*\n\s*Action Input:\s*(.+)")
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S)


class ReActLoop:
    """Textual ReAct loop. Works with any model backend."""

    def __init__(self, *, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    async def run(self, agent: Agent, prompt: str) -> AgentResult:
        tracer = agent.tracer
        tools_by_name = {t.name: t for t in agent.tools}
        tool_list = (
            "\n".join(f"- {t.name}: {t.description}" for t in agent.tools)
            or "  (no tools)"
        )
        system = _REACT_SYS_TEMPLATE.format(
            name=agent.name,
            instructions=agent.instructions or "",
            tool_list=tool_list,
        )
        # Inject authority-ranked memory + active decisions/corrections so the
        # model SEES authority. No-op for plain stores (which lack these hooks)
        # and never fatal — memory must not break the reasoning loop.
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
        finish = "max_steps"
        output = ""

        with tracer.span("agent.run", agent=agent.name, prompt_len=len(prompt)) as run_span:
            for step in range(1, self.max_steps + 1):
                with tracer.span("agent.llm", step=step) as llm_span:
                    resp = await agent.model.generate(messages)
                    llm_span.set_attr("model", resp.model)
                    llm_span.set_attr("finish_reason", resp.finish_reason or "")
                messages.append(Message(role="assistant", content=resp.text))

                final = _FINAL_RE.search(resp.text)
                if final:
                    output = final.group(1).strip()
                    finish = "stop"
                    break

                m = _ACTION_RE.search(resp.text)
                if not m:
                    # Model didn't follow the protocol; treat the raw text as the answer.
                    output = resp.text.strip()
                    finish = "stop_no_protocol"
                    break

                tool_name = m.group(1).strip()
                raw_args = m.group(2).strip()
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("args must be a JSON object")
                except (ValueError, json.JSONDecodeError) as e:
                    observation = f"ERROR: invalid Action Input JSON: {e}"
                    messages.append(Message(role="tool", content=observation, name=tool_name))
                    continue

                tool = tools_by_name.get(tool_name)
                if tool is None:
                    observation = f"ERROR: unknown tool {tool_name!r}"
                    messages.append(Message(role="tool", content=observation, name=tool_name))
                    continue

                with tracer.span("agent.tool", tool=tool_name, step=step) as tool_span:
                    try:
                        result: ToolResult = await tool(**args)
                    except Exception as e:  # noqa: BLE001 — surface to model
                        result = ToolResult.failure(f"{type(e).__name__}: {e}")
                    tool_span.set_attr("ok", result.ok)

                tool_calls.append(
                    {"step": step, "name": tool_name, "args": args, "result": result}
                )
                observation = (
                    str(result.value) if result.ok else f"ERROR: {result.error}"
                )
                messages.append(Message(role="tool", content=observation, name=tool_name))
            run_span.set_attr("steps", step)
            run_span.set_attr("finish_reason", finish)

        # Optionally persist the turn as a typed interaction memory so future
        # recalls can surface it. Off by default to avoid noise.
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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """A minimal agent. Compose; don't subclass.

    Example::

        agent = Agent(
            name="research",
            instructions="Find facts and cite sources.",
            model=auto_backend(),
            tools=[fetch, search],
            capabilities={Capability.NET_HTTP, Capability.LLM_INFERENCE},
        )
        result = await agent.run("What is the boiling point of mercury?")
    """

    def __init__(
        self,
        *,
        name: str,
        model: ModelBackend,
        instructions: str = "",
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        capabilities: set[Capability] | None = None,
        loop: AgentLoop | None = None,
        tracer: Tracer | None = None,
        recall_memory: bool = True,
        remember_interactions: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = list(tools or [])
        self.memory = memory if memory is not None else InMemoryStore()
        # When True (and memory is a TypedMemory), each run injects
        # authority-ranked recall + active decisions/corrections into the
        # system prompt. No-op for plain stores.
        self.recall_memory = recall_memory
        # When True, each completed run is stored back as an interaction memory.
        self.remember_interactions = remember_interactions
        # Always grant LLM_INFERENCE by default; user can revoke if they want
        # to enforce dry-run mode.
        self._caps = CapabilityContext(capabilities or set()).grant(
            Capability.LLM_INFERENCE
        )
        self.loop = loop or ReActLoop()
        self.tracer = tracer or get_tracer()

    @property
    def capabilities(self) -> CapabilityContext:
        return self._caps

    def grant(self, *caps: Capability) -> Agent:
        self._caps.grant(*caps)
        return self

    async def run(self, prompt: str) -> AgentResult:
        _log.info(
            "agent.run.start",
            extra={"agent": self.name, "prompt_len": len(prompt)},
        )
        with use_context(self._caps):
            result = await self.loop.run(self, prompt)
        _log.info(
            "agent.run.end",
            extra={
                "agent": self.name,
                "steps": result.steps,
                "finish": result.finish_reason,
            },
        )
        return result
