"""AitherAgent — the core agent class."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from adk.config import Config
from adk.identity import Identity, load_identity
from adk.llm import DegenerationDetector, LLMRouter, Message, LLMResponse, strip_internal_tags
from adk.loop_guard import LoopGuard, LoopAction
from adk.memory import Memory
from adk.metering import AgentMeter, QuotaAction, get_meter
from adk.metrics import get_metrics
from adk.tools import ToolDef, ToolRegistry
from adk.trace import get_trace_id

logger = logging.getLogger("adk.agent")

_MAX_TOOL_LOOPS = 10

# Essential tools that must always be available when agent has tools
_ESSENTIAL_TOOL_NAMES = frozenset({
    "read_file", "write_file", "replace_text", "search_files", "list_directory",
})

# Steering message injected when LLM returns text-only on turn 1 with tools available
_TOOL_STEERING_MSG = (
    "You have tools available. You MUST use them to complete the user's request. "
    "Do not describe what you would do — actually call the appropriate tool function. "
    "Re-read the user's message and select the right tool."
)

# Patterns that indicate the user wants an action (not just conversation)
_ACTION_PATTERNS = re.compile(
    r'(?i)(?:read|write|edit|create|delete|search|find|list|run|execute|check|fix|'
    r'refactor|deploy|build|test|commit|push|pull|install|update|remove|send|fetch|'
    r'open|close|show\s+me\s+(?:the\s+)?(?:file|code|log|error))',
)


def _should_steer_tool_use(message: str, tool_choice: str | dict | None) -> bool:
    """Determine if we should retry with tool-steering on turn 1.

    Only steers when: tool_choice was explicitly set, OR the message
    contains action verbs that strongly suggest tool use is needed.
    Simple greetings like "Hello" should NOT trigger steering.
    """
    if tool_choice and tool_choice != "auto":
        return True
    return bool(_ACTION_PATTERNS.search(message))
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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls_made: list[str] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    session_id: str = ""
    finish_reason: str = "stop"
    effort_level: int = 0
    cache_status: str = ""


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
        # Extract inference controls from kwargs (null=auto pattern)
        _tool_choice = kwargs.pop("tool_choice", None)
        _top_p = kwargs.pop("top_p", None)
        _repetition_penalty = kwargs.pop("repetition_penalty", None)
        _effort = kwargs.pop("effort", None)
        _effort_int = _effort if isinstance(_effort, int) else 5
        loop_guard = LoopGuard(
            warn_threshold=2,
            block_threshold=4,
            circuit_break_total=_MAX_TOOL_LOOPS + 5,
            effort_level=_effort_int,
        )
        _steered_once = False  # Track turn-1 tool-call steering
        _token_counts_per_iter: list[int] = []  # Gap H: diminishing returns tracking
        _max_output_escalated = False  # Gap 3: track if we already escalated

        for _loop_idx in range(_MAX_TOOL_LOOPS):
            # Check if circuit breaker tripped from previous iteration
            if loop_guard.tripped and _effort_int < 4:
                logger.info("Loop guard circuit breaker tripped — forcing synthesis")
                break

            # ── Gap K: Message normalization ──
            # Merge consecutive same-role messages and strip empties
            _normalized: list[Message] = []
            for _msg in messages:
                if not _msg.content and _msg.role not in ("assistant",) and not _msg.tool_calls and not _msg.tool_call_id:
                    continue  # Strip empty non-assistant messages with no tool data
                if (_normalized and _msg.role == _normalized[-1].role
                        and _msg.role in ("system", "user")
                        and not _msg.tool_call_id and not _normalized[-1].tool_calls):
                    _normalized[-1] = Message(
                        role=_msg.role,
                        content=(_normalized[-1].content or "") + "\n" + (_msg.content or ""),
                    )
                else:
                    _normalized.append(_msg)
            messages = _normalized

            # ── Gap 6: Micro-compaction of old tool results ──
            # If there are more than 5 tool result messages, clear old ones
            _tool_msg_indices = [
                i for i, m in enumerate(messages) if m.role == "tool"
            ]
            if len(_tool_msg_indices) > 5:
                _stale = _tool_msg_indices[:-5]
                for _idx in _stale:
                    messages[_idx] = Message(
                        role="tool",
                        content="[Prior result cleared]",
                        tool_call_id=messages[_idx].tool_call_id,
                    )

            # ── Gap 4: Tool result pairing guarantee ──
            # Scan for assistant messages with tool_calls that lack matching tool results
            _seen_tool_ids: set[str] = set()
            for _msg in messages:
                if _msg.role == "tool" and _msg.tool_call_id:
                    _seen_tool_ids.add(_msg.tool_call_id)
            _orphan_patches: list[Message] = []
            for _i, _msg in enumerate(messages):
                if _msg.role == "assistant" and _msg.tool_calls:
                    for _tc in _msg.tool_calls:
                        _tc_id = _tc.get("id", "") if isinstance(_tc, dict) else getattr(_tc, "id", "")
                        if _tc_id and _tc_id not in _seen_tool_ids:
                            _orphan_patches.append(Message(
                                role="tool",
                                content=json.dumps({"error": "orphaned_tool_call", "message": "No result was returned for this tool call."}),
                                tool_call_id=_tc_id,
                            ))
                            _seen_tool_ids.add(_tc_id)
            if _orphan_patches:
                logger.debug("[REACT] Injecting %d synthetic tool results for orphaned calls", len(_orphan_patches))
                messages.extend(_orphan_patches)

            resp = await self.llm.chat(
                messages, tools=tools_schema, effort=_effort,
                tool_choice=_tool_choice, top_p=_top_p,
                repetition_penalty=_repetition_penalty, **kwargs,
            )

            # ── Gap 3: max_output_tokens escalation ──
            # If response was truncated (finish_reason == "length"), retry with doubled tokens
            if resp.finish_reason == "length" and not _max_output_escalated:
                _max_output_escalated = True
                _current_max = kwargs.get("max_tokens", 4096)
                logger.debug("[REACT] Response truncated — retrying with max_tokens=%d", _current_max * 2)
                # Inject partial response + continuation prompt
                messages.append(Message(role="assistant", content=resp.content or ""))
                messages.append(Message(
                    role="user",
                    content="Your previous response was truncated. Continue exactly where you left off.",
                ))
                _escalation_kwargs = {**kwargs, "max_tokens": _current_max * 2}
                resp = await self.llm.chat(
                    messages, tools=tools_schema, effort=_effort,
                    tool_choice=_tool_choice, top_p=_top_p,
                    repetition_penalty=_repetition_penalty, **_escalation_kwargs,
                )
                # If still truncated, try one more time with 3x
                if resp.finish_reason == "length":
                    messages.append(Message(role="assistant", content=resp.content or ""))
                    messages.append(Message(
                        role="user",
                        content="Your previous response was truncated. Continue exactly where you left off.",
                    ))
                    _escalation_kwargs["max_tokens"] = _current_max * 3
                    resp = await self.llm.chat(
                        messages, tools=tools_schema, effort=_effort,
                        tool_choice=_tool_choice, top_p=_top_p,
                        repetition_penalty=_repetition_penalty, **_escalation_kwargs,
                    )

            # ── Gap H: Diminishing returns detection ──
            _iter_tokens = resp.completion_tokens or len((resp.content or "").split())
            _token_counts_per_iter.append(_iter_tokens)
            if len(_token_counts_per_iter) >= 3:
                _recent = _token_counts_per_iter[-3:]
                if all(t < 500 for t in _recent):
                    logger.debug("[REACT] Diminishing returns — 3+ iterations with < 500 tokens each")
                    messages.append(Message(
                        role="system",
                        content=(
                            "[DIMINISHING RETURNS] The last 3 iterations produced very little output. "
                            "Consider concluding with a synthesis of what you have found so far."
                        ),
                    ))

            if not resp.tool_calls:
                # Turn-1 tool-call steering: if LLM didn't use tools on first
                # turn and tools are available AND the user explicitly requested
                # an action (not just chatting), inject steering and retry once.
                # Heuristic: steer if tool_choice was explicitly set, or if the
                # message contains action verbs typical of tool-requiring tasks.
                if (_loop_idx == 0 and tools_schema and not _steered_once
                        and _effort_int >= 6
                        and _should_steer_tool_use(message, _tool_choice)):
                    _steered_once = True
                    logger.debug("[REACT] Turn-1 no tool call — injecting steering retry")
                    messages.append(Message(role="assistant", content=resp.content or ""))
                    messages.append(Message(role="system", content=_TOOL_STEERING_MSG))
                    continue

                # No tool calls — we have the final answer
                content = strip_internal_tags(resp.content)
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
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    latency_ms=resp.latency_ms,
                    tool_calls_made=tool_calls_made,
                    artifacts=[a.to_dict() for a in _get_session_artifacts(sid)],
                    session_id=sid,
                    finish_reason=resp.finish_reason,
                    effort_level=resp.effort_level,
                    cache_status=resp.cache_status,
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
        final = await self.llm.chat(
            messages, effort=_effort, top_p=_top_p,
            repetition_penalty=_repetition_penalty, **kwargs,
        )
        content = strip_internal_tags(final.content)
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
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            latency_ms=final.latency_ms,
            tool_calls_made=tool_calls_made,
            artifacts=[a.to_dict() for a in _get_session_artifacts(sid)],
            session_id=sid,
            finish_reason=final.finish_reason,
            effort_level=final.effort_level,
            cache_status=final.cache_status,
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

        Includes degeneration detection — if the model starts repeating,
        the stream is killed and trimmed to the last clean sentence.
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

        # Extract inference controls
        _effort = kwargs.pop("effort", None)
        _top_p = kwargs.pop("top_p", None)
        _repetition_penalty = kwargs.pop("repetition_penalty", None)

        # Stream with degeneration detection
        full_content = ""
        _degenerated = False
        async for chunk in self.llm.chat_stream(
            messages, effort=_effort, top_p=_top_p,
            repetition_penalty=_repetition_penalty, **kwargs,
        ):
            if chunk.finish_reason == "degeneration":
                _degenerated = True
                logger.warning("Degeneration detected in stream for agent %s", self.name)
                break
            if chunk.content:
                full_content += chunk.content
                yield chunk.content

        # If degenerated, trim to clean content
        if _degenerated and full_content:
            detector = DegenerationDetector()
            full_content = detector.trim_clean(full_content)

        # Strip internal tags from full response
        full_content = strip_internal_tags(full_content)

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
                    degenerated=_degenerated,
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
