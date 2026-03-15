"""AitherAgent — the core agent class."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


def _get_session_artifacts(session_id: str) -> list:
    """Get artifacts collected for a session."""
    try:
        from adk.artifacts import get_registry
        return get_registry().get(session_id)
    except Exception:
        return []


@dataclass
class AgentResponse:
    """Response from an agent interaction."""
    content: str
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_calls_made: list[str] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
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
        builtin_tools: bool = True,
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

        # LLM — auto-detect Elysium if no local backend and API key present
        self.llm = llm or LLMRouter(config=self.config)
        self._elysium_connected = False
        if not llm:
            self._try_elysium_fallback()

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

        # Safety (IntakeGuard) — non-fatal
        self._safety = None
        try:
            from adk.safety import IntakeGuard
            self._safety = IntakeGuard()
        except Exception:
            pass

        # Context manager (token-aware truncation) — non-fatal
        self._context_mgr = None
        try:
            from adk.context import ContextManager
            max_tokens = self.config.max_context or 8000
            self._context_mgr = ContextManager(max_tokens=max_tokens)
        except Exception:
            pass

        # Event emitter — non-fatal
        self._events = None
        try:
            from adk.events import get_emitter
            self._events = get_emitter()
        except Exception:
            pass

        # Graph memory (knowledge graph with embeddings) — non-fatal
        self._graph = None
        try:
            from adk.graph_memory import GraphMemory
            self._graph = GraphMemory(agent_name=self.name)
        except Exception:
            pass

        # Neuron auto-fire (context gathering before LLM) — non-fatal
        self._auto_neurons = None
        try:
            from adk.neurons import AutoNeuronFire
            self._auto_neurons = AutoNeuronFire(agent=self)
        except Exception:
            pass

        # Strata unified storage — lazy init via property
        self._strata = None

        # Built-in tools — non-fatal
        if builtin_tools:
            try:
                from adk.builtin_tools import register_builtin_tools
                register_builtin_tools(self)
            except Exception:
                pass

    def _try_elysium_fallback(self):
        """If no local LLM is available but AITHER_API_KEY is set, use Elysium."""
        api_key = os.environ.get("AITHER_API_KEY", "")
        if not api_key:
            return
        # Check if the LLM router has a working local backend
        if self.llm.provider_name in ("ollama", "vllm"):
            return  # Local backend detected, no need for Elysium
        # Wire up Elysium inference
        try:
            from adk.llm import LLMRouter
            self.llm = LLMRouter(
                provider="gateway",
                base_url="https://mcp.aitherium.com/v1",
                api_key=api_key,
                model="aither-orchestrator",
            )
            self._elysium_connected = True
            logger.info(
                "Agent '%s' using Elysium cloud inference (AITHER_API_KEY set). "
                "Run 'aither connect' for details.",
                self.name,
            )
        except Exception as exc:
            logger.debug("Elysium fallback failed: %s", exc)

    @property
    def system_prompt(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        return self._identity.build_system_prompt()

    @property
    def strata(self):
        """Lazy-initialized Strata unified storage.

        Returns the global Strata instance. Agents can use this to
        read/write data through a single API that resolves to local
        filesystem, S3, or full AitherOS Strata transparently.

        Usage:
            data = await agent.strata.read("codegraph/index.json")
            await agent.strata.write("models/config.json", payload)
        """
        if self._strata is None:
            from adk.strata import get_strata
            self._strata = get_strata()
        return self._strata

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
        _chat_start = time.perf_counter()

        # Emit chat start event
        if self._events:
            try:
                await self._events.emit(
                    "chat_request", agent=self.name,
                    message=message[:200], session_id=sid,
                )
            except Exception:
                pass

        # Input safety check
        if self._safety:
            try:
                safety_result = self._safety.check(message)
                if safety_result.blocked:
                    logger.warning("Safety blocked input for agent %s", self.name)
                    return AgentResponse(
                        content="I can't process that request — it was flagged by the safety filter.",
                        session_id=sid,
                    )
            except Exception as exc:
                logger.warning("Safety check failed (non-fatal): %s", exc)

        # Build messages with context-aware truncation
        messages = None
        if self._context_mgr:
            try:
                self._context_mgr.clear()
                self._context_mgr.add_system(self.system_prompt)
                if history:
                    for h in history:
                        self._context_mgr.add(h["role"], h["content"])
                else:
                    stored = await self.memory.get_history(sid, limit=20)
                    for h in stored:
                        self._context_mgr.add(h["role"], h["content"])
                self._context_mgr.add_user(message)
                msg_dicts = self._context_mgr.build()
                messages = [Message(role=d["role"], content=d["content"]) for d in msg_dicts]
            except Exception:
                messages = None  # Fall back to manual

        if messages is None:
            messages = [Message(role="system", content=self.system_prompt)]
            if history:
                for h in history:
                    messages.append(Message(role=h["role"], content=h["content"]))
            else:
                stored = await self.memory.get_history(sid, limit=20)
                for h in stored:
                    messages.append(Message(role=h["role"], content=h["content"]))
            messages.append(Message(role="user", content=message))

        # Inject graph memory context (non-fatal)
        if self._graph:
            try:
                relevant = await self._graph.search(message, limit=3)
                if relevant:
                    graph_lines = [f"- {n.label}: {n.content[:200]}" for n in relevant if n.content]
                    if graph_lines:
                        graph_context = "[MEMORY GRAPH]\n" + "\n".join(graph_lines)
                        messages.insert(1, Message(role="system", content=graph_context))
            except Exception:
                pass

        # Auto-fire neurons for additional context (non-fatal)
        if self._auto_neurons:
            try:
                neuron_context = await self._auto_neurons.gather_context(message)
                if neuron_context:
                    messages.insert(1, Message(role="system", content=neuron_context))
            except Exception:
                pass

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
                content = resp.content
                # Output safety check
                if self._safety:
                    try:
                        from adk.safety import check_output
                        out_result = check_output(content)
                        if not out_result.safe:
                            content = out_result.sanitized_content
                    except Exception:
                        pass
                await self.memory.add_message(sid, "assistant", content)
                try:
                    store = _get_conversations()
                    await store.append_message(sid, "assistant", content, agent_name=self.name)
                except Exception:
                    pass
                # Record metering
                self.meter.record_usage(
                    tokens=resp.tokens_used,
                    model=resp.model,
                    latency_ms=resp.latency_ms,
                )
                _total_ms = (time.perf_counter() - _chat_start) * 1000
                if self._events:
                    try:
                        await self._events.emit(
                            "chat_response", agent=self.name,
                            tokens_used=resp.tokens_used, model=resp.model,
                            latency_ms=_total_ms, session_id=sid,
                        )
                    except Exception:
                        pass
                # Auto-ingest conversation into graph memory (fire-and-forget)
                if self._graph:
                    try:
                        await self._graph.ingest_conversation(sid, [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": content},
                        ])
                    except Exception:
                        pass
                return AgentResponse(
                    content=content,
                    model=resp.model,
                    tokens_used=resp.tokens_used,
                    latency_ms=resp.latency_ms,
                    tool_calls_made=tool_calls_made,
                    artifacts=[a.to_dict() for a in _get_session_artifacts(sid)],
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
                if self._events:
                    try:
                        await self._events.emit(
                            "tool_call", agent=self.name,
                            tool=tc.name, arguments=tc.arguments,
                        )
                    except Exception:
                        pass
                _tool_start = time.perf_counter()
                result = await self._tools.execute(tc.name, tc.arguments)
                _tool_ms = (time.perf_counter() - _tool_start) * 1000
                get_metrics().record_tool_call(tool=tc.name, latency_ms=_tool_ms)
                # Detect artifacts in tool output
                try:
                    from adk.artifacts import detect_artifact, get_registry
                    _art = detect_artifact(tc.name, result)
                    if _art:
                        _art.tool = tc.name
                        get_registry().add(sid, _art)
                except Exception:
                    pass
                if self._events:
                    try:
                        await self._events.emit(
                            "tool_result", agent=self.name,
                            tool=tc.name, latency_ms=_tool_ms,
                        )
                    except Exception:
                        pass
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                ))

        # Exhausted tool loops — return last response
        final = await self.llm.chat(messages, **kwargs)
        content = final.content
        # Output safety check
        if self._safety:
            try:
                from adk.safety import check_output
                out_result = check_output(content)
                if not out_result.safe:
                    content = out_result.sanitized_content
            except Exception:
                pass
        await self.memory.add_message(sid, "assistant", content)
        try:
            store = _get_conversations()
            await store.append_message(sid, "assistant", content, agent_name=self.name)
        except Exception:
            pass
        # Record metering for final response
        self.meter.record_usage(
            tokens=final.tokens_used,
            model=final.model,
            latency_ms=final.latency_ms,
        )
        _total_ms = (time.perf_counter() - _chat_start) * 1000
        if self._events:
            try:
                await self._events.emit(
                    "chat_response", agent=self.name,
                    tokens_used=final.tokens_used, model=final.model,
                    latency_ms=_total_ms, session_id=sid,
                )
            except Exception:
                pass
        # Auto-ingest conversation into graph memory (fire-and-forget)
        if self._graph:
            try:
                await self._graph.ingest_conversation(sid, [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": content},
                ])
            except Exception:
                pass
        return AgentResponse(
            content=content,
            model=final.model,
            tokens_used=final.tokens_used,
            latency_ms=final.latency_ms,
            tool_calls_made=tool_calls_made,
            artifacts=[a.to_dict() for a in _get_session_artifacts(sid)],
            session_id=sid,
        )

    async def chat_stream(
        self,
        message: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        """Stream a response. Yields string chunks.

        If the agent has tools and the LLM requests tool use, falls back
        to non-streaming chat() (tool loops can't stream mid-execution).
        """
        sid = session_id or self._session_id

        # Input safety check
        if self._safety:
            try:
                safety_result = self._safety.check(message)
                if safety_result.blocked:
                    yield "I can't process that request — it was flagged by the safety filter."
                    return
            except Exception:
                pass

        # Emit chat start event
        if self._events:
            try:
                await self._events.emit(
                    "chat_request", agent=self.name,
                    message=message[:200], session_id=sid, streaming=True,
                )
            except Exception:
                pass

        # If agent has tools, fall back to sync (tool loops can't stream)
        if self._tools.list_tools():
            resp = await self.chat(message, history=history, session_id=sid, **kwargs)
            yield resp.content
            return

        # Build messages
        messages = [Message(role="system", content=self.system_prompt)]
        if history:
            for h in history:
                messages.append(Message(role=h["role"], content=h["content"]))
        else:
            stored = await self.memory.get_history(sid, limit=20)
            for h in stored:
                messages.append(Message(role=h["role"], content=h["content"]))
        messages.append(Message(role="user", content=message))

        # Stream
        full_content = ""
        async for chunk in self.llm.chat_stream(messages, **kwargs):
            if chunk.content:
                full_content += chunk.content
                yield chunk.content

        # Output safety check on full response
        if self._safety and full_content:
            try:
                from adk.safety import check_output
                out_result = check_output(full_content)
                if not out_result.safe:
                    logger.warning("Streaming output flagged by safety check")
            except Exception:
                pass

        # Store in memory
        await self.memory.add_message(sid, "user", message)
        await self.memory.add_message(sid, "assistant", full_content)

        # Emit completion event
        if self._events:
            try:
                await self._events.emit(
                    "chat_response", agent=self.name,
                    session_id=sid, streaming=True,
                )
            except Exception:
                pass

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

    async def graph_remember(self, subject: str, relation: str, object_: str):
        """Store a knowledge triple in the agent's graph memory."""
        if not self._graph:
            return
        await self._graph.remember(subject, relation, object_)

    async def graph_query(self, question: str, limit: int = 5) -> list:
        """Query the agent's graph memory. Returns list of GraphNode."""
        if not self._graph:
            return []
        return await self._graph.query(question, limit=limit)

    async def graph_stats(self) -> dict:
        """Get graph memory statistics."""
        if not self._graph:
            return {"enabled": False}
        stats = await self._graph.get_stats()
        stats["enabled"] = True
        return stats

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
