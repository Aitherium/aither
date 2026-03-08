"""AitherAgent — the core agent class."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from adk.config import Config
from adk.identity import Identity, load_identity
from adk.llm import LLMRouter, Message, LLMResponse
from adk.loop_guard import LoopGuard, LoopAction
from adk.memory import Memory
from adk.metering import AgentMeter, QuotaAction, get_meter
from adk.metrics import get_metrics
from adk.tools import ToolDef, ToolRegistry
from adk.trace import get_trace_id

logger = logging.getLogger("adk.agent")

_MAX_TOOL_LOOPS = 10
_conversations_store = None


def _get_conversations():
    """Lazy-load the global ConversationStore."""
    global _conversations_store
    if _conversations_store is None:
        from adk.conversations import get_conversation_store
        _conversations_store = get_conversation_store()
    return _conversations_store


@dataclass
class AgentResponse:
    """Response from an agent interaction."""
    content: str
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_calls_made: list[str] = field(default_factory=list)
    session_id: str = ""


class AitherAgent:
    """An AI agent with identity, tools, memory, and LLM access.

    Usage:
        agent = AitherAgent("atlas")
        response = await agent.chat("What's the project status?")

        # With custom tools
        agent = AitherAgent("demiurge", tools=[my_tool_registry])

        # With specific LLM
        agent = AitherAgent("lyra", llm=LLMRouter(provider="openai", api_key="sk-..."))
    """

    def __init__(
        self,
        name: str | None = None,
        identity: str | Identity | None = None,
        llm: LLMRouter | None = None,
        tools: list[ToolRegistry] | ToolRegistry | None = None,
        memory: Memory | None = None,
        config: Config | None = None,
        system_prompt: str | None = None,
        phonehome: bool = False,
    ):
        self.config = config or Config.from_env()

        # Identity
        if isinstance(identity, Identity):
            self._identity = identity
        elif isinstance(identity, str):
            self._identity = load_identity(identity)
        elif name:
            self._identity = load_identity(name)
        else:
            self._identity = Identity(name="assistant")

        self.name = name or self._identity.name
        self._system_prompt = system_prompt

        # LLM
        self.llm = llm or LLMRouter(config=self.config)

        # Tools
        self._tools = ToolRegistry()
        if tools:
            registries = tools if isinstance(tools, list) else [tools]
            for reg in registries:
                for td in reg.list_tools():
                    self._tools._tools[td.name] = td

        # Memory
        self.memory = memory or Memory(agent_name=self.name)

        # Metering (per-agent token & cost tracking)
        self.meter = get_meter(self.name)

        # Session
        self._session_id = str(uuid.uuid4())[:8]

        # Phonehome
        self._phonehome = phonehome or self.config.phonehome_enabled

    @property
    def system_prompt(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        return self._identity.build_system_prompt()

    def tool(self, fn=None, *, name=None, description=None):
        """Decorator to register a tool function on this agent.

        Usage:
            @agent.tool
            def search(query: str) -> str:
                '''Search the web.'''
                return "results..."
        """
        def decorator(f):
            self._tools.register(f, name=name, description=description)
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Send a message and get a response. Uses tools if available."""
        sid = session_id or self._session_id

        # Build messages
        messages = [Message(role="system", content=self.system_prompt)]

        # Add history
        if history:
            for h in history:
                messages.append(Message(role=h["role"], content=h["content"]))
        else:
            stored = await self.memory.get_history(sid, limit=20)
            for h in stored:
                messages.append(Message(role=h["role"], content=h["content"]))

        messages.append(Message(role="user", content=message))

        # Store user message (in-memory + persistent JSON)
        await self.memory.add_message(sid, "user", message)
        try:
            store = _get_conversations()
            await store.append_message(sid, "user", message, agent_name=self.name)
        except Exception:
            pass  # Non-fatal — persistent store is best-effort

        # Call LLM (with tool loop if tools registered)
        tools_schema = self._tools.to_openai_format() if self._tools.list_tools() else None
        tool_calls_made = []
        loop_guard = LoopGuard(
            warn_threshold=2,
            block_threshold=3,
            circuit_break_total=_MAX_TOOL_LOOPS + 5,
        )

        for _loop_idx in range(_MAX_TOOL_LOOPS):
            # Check if circuit breaker tripped from previous iteration
            if loop_guard.tripped:
                logger.info("Loop guard circuit breaker tripped — forcing synthesis")
                break

            resp = await self.llm.chat(messages, tools=tools_schema, **kwargs)

            if not resp.tool_calls:
                # No tool calls — we have the final answer
                await self.memory.add_message(sid, "assistant", resp.content)
                try:
                    store = _get_conversations()
                    await store.append_message(sid, "assistant", resp.content, agent_name=self.name)
                except Exception:
                    pass
                # Record metering
                self.meter.record_usage(
                    tokens=resp.tokens_used,
                    model=resp.model,
                    latency_ms=resp.latency_ms,
                )
                return AgentResponse(
                    content=resp.content,
                    model=resp.model,
                    tokens_used=resp.tokens_used,
                    latency_ms=resp.latency_ms,
                    tool_calls_made=tool_calls_made,
                    session_id=sid,
                )

            # Execute tool calls with loop guard checks
            messages.append(Message(
                role="assistant",
                content=resp.content or "",
                tool_calls=[
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls
                ],
            ))

            for tc in resp.tool_calls:
                verdict = loop_guard.check(tc.name, tc.arguments)

                if verdict.action == LoopAction.CIRCUIT_BREAK:
                    logger.warning("Loop guard CIRCUIT BREAK: %s", verdict.reason)
                    messages.append(Message(role="system", content=verdict.nudge_message))
                    tool_calls_made.append(f"{tc.name}[circuit_break]")
                    messages.append(Message(
                        role="tool",
                        content=json.dumps({"error": "circuit_break", "message": verdict.reason}),
                        tool_call_id=tc.id,
                    ))
                    # Fire metrics + Pulse alert
                    get_metrics().record_loop_guard_break()
                    _fire_pulse_loop_break(self.name, tc.name, loop_guard.stats.total_checks)
                    continue

                if verdict.action == LoopAction.BLOCK:
                    logger.info("Loop guard BLOCKED: %s", verdict.reason)
                    messages.append(Message(role="system", content=verdict.nudge_message))
                    tool_calls_made.append(f"{tc.name}[blocked]")
                    messages.append(Message(
                        role="tool",
                        content=json.dumps({"error": "blocked_duplicate", "message": verdict.reason}),
                        tool_call_id=tc.id,
                    ))
                    continue

                if verdict.action == LoopAction.WARN:
                    logger.debug("Loop guard WARN: %s", verdict.reason)
                    messages.append(Message(role="system", content=verdict.nudge_message))

                # ALLOW or WARN — execute the tool
                tool_calls_made.append(tc.name)
                _tool_start = time.perf_counter()
                result = await self._tools.execute(tc.name, tc.arguments)
                _tool_ms = (time.perf_counter() - _tool_start) * 1000
                get_metrics().record_tool_call(tool=tc.name, latency_ms=_tool_ms)
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                ))

        # Exhausted tool loops — return last response
        final = await self.llm.chat(messages, **kwargs)
        await self.memory.add_message(sid, "assistant", final.content)
        try:
            store = _get_conversations()
            await store.append_message(sid, "assistant", final.content, agent_name=self.name)
        except Exception:
            pass
        # Record metering for final response
        self.meter.record_usage(
            tokens=final.tokens_used,
            model=final.model,
            latency_ms=final.latency_ms,
        )
        return AgentResponse(
            content=final.content,
            model=final.model,
            tokens_used=final.tokens_used,
            latency_ms=final.latency_ms,
            tool_calls_made=tool_calls_made,
            session_id=sid,
        )

    async def run(self, task: str, **kwargs) -> AgentResponse:
        """Execute a task with ReAct-style reasoning.

        Same as chat() but with a task-oriented system prompt wrapper.
        """
        task_prompt = (
            f"Complete the following task. Use available tools as needed. "
            f"Think step by step.\n\nTask: {task}"
        )
        return await self.chat(task_prompt, **kwargs)

    async def remember(self, key: str, value: str, category: str = "general"):
        """Store a value in the agent's persistent memory."""
        await self.memory.remember(key, value, category=category)

    async def recall(self, key: str) -> str | None:
        """Retrieve a value from the agent's persistent memory."""
        return await self.memory.recall(key)

    def new_session(self) -> str:
        """Start a new conversation session."""
        self._session_id = str(uuid.uuid4())[:8]
        return self._session_id

    async def report_bug(self, description: str, include_logs: bool = True) -> dict:
        """Report a bug programmatically."""
        from adk.bugreport import submit_bug_report
        return await submit_bug_report(
            description=description,
            agent_name=self.name,
            llm_backend=self.llm.provider_name,
            include_logs=include_logs,
        )


def _fire_pulse_loop_break(agent: str, tool: str, total_calls: int):
    """Fire-and-forget Pulse pain signal for loop guard circuit break."""
    async def _send():
        try:
            from adk.pulse import get_pulse
            pulse = get_pulse()
            await pulse.send_loop_break(
                agent=agent, tool=tool,
                total_calls=total_calls,
                request_id=get_trace_id(),
            )
        except Exception:
            pass
    try:
        asyncio.ensure_future(_send())
    except RuntimeError:
        pass  # No event loop — skip
