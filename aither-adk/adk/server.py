"""FastAPI server wrapping an AitherAgent — OpenAI-compatible + Genesis-compatible.

Supports two modes:
- Single agent: `aither-serve --identity aither`
- Fleet mode:   `aither-serve --fleet fleet.yaml` or `aither-serve --agents aither,lyra,demiurge`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from adk import __version__
from adk.agent import AitherAgent, AgentResponse
from adk.config import Config
from adk.identity import list_identities, load_identity
from adk.llm import LLMRouter, Message
from adk.metrics import get_metrics
from adk.trace import TraceMiddleware, get_trace_id, new_trace

logger = logging.getLogger("adk.server")


def create_app(
    agent: AitherAgent | None = None,
    identity: str = "aither",
    config: Config | None = None,
    fleet_path: str | None = None,
    fleet_agents: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app wrapping an AitherAgent or a fleet of agents.

    Returns a fully configured app with both OpenAI-compatible and Genesis-compatible endpoints.
    """
    config = config or Config.from_env()

    is_fleet = bool(fleet_path or fleet_agents)

    # ─── Lifespan (replaces deprecated @app.on_event("startup")) ───

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Run startup tasks, yield for request handling, then cleanup."""
        # Configure structured logging
        from adk.chronicle import configure_logging
        configure_logging(
            level=os.getenv("AITHER_LOG_LEVEL", "INFO"),
            json_output=config.json_logging,
        )

        await _register_with_gateway()
        await _join_aithernet()
        await _init_chat_relay()
        await _init_mail_relay()
        await _init_mcp_server()
        await _init_a2a_server()
        await _connect_service_bridge()
        await _flush_strata_queue()
        await _flush_chronicle_queue()
        await _start_watch_reporter()
        await _flush_pulse_queue()
        # Eagerly detect LLM backend so /health shows the right provider
        try:
            a = await get_agent()
            await a.llm.get_provider()
        except Exception:
            pass
        yield
        # ── Shutdown cleanup ──
        bridge = _state.get("aither_bridge")
        if bridge:
            await bridge.stop()
        chat = _state.get("chat_relay")
        if chat:
            await chat.stop_irc_server()

    app = FastAPI(
        title=f"AitherADK — {'Fleet' if is_fleet else identity}",
        version=__version__,
        docs_url="/docs",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Trace ID middleware — generates/propagates X-Request-ID on every request
    app.add_middleware(TraceMiddleware)

    # State shared across endpoints
    _state: dict[str, Any] = {
        "agent": agent,
        "identity": identity,
        "config": config,
        "fleet": None,
        "is_fleet": is_fleet,
        "service_bridge": None,
    }

    # ─── Auth middleware (optional, enabled via AITHER_API_KEY or --api-key) ───

    _server_api_key = os.getenv("AITHER_SERVER_API_KEY", "")
    _SKIP_AUTH_PATHS = {"/health", "/docs", "/openapi.json", "/metrics", "/demo", "/redoc"}

    # Valid caller types for header validation (prevents spoofing)
    _VALID_CALLER_TYPES = {"PLATFORM", "PUBLIC", "DEMO", "TENANT", "ANONYMOUS"}

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        """Bearer token auth + caller-type header validation.

        Validates X-Caller-Type header to prevent header-spoofing attacks.
        External requests cannot claim PLATFORM caller type.
        """
        if request.url.path in _SKIP_AUTH_PATHS:
            return await call_next(request)

        # Validate X-Caller-Type if present (prevent spoofing)
        caller_type = request.headers.get("x-caller-type", "")
        if caller_type:
            if caller_type not in _VALID_CALLER_TYPES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid X-Caller-Type: {caller_type}"},
                )
            # External requests cannot claim PLATFORM — that's internal-only
            if caller_type == "PLATFORM" and _server_api_key:
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer ") or auth_header[7:] != _server_api_key:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "PLATFORM caller type requires valid API key"},
                    )

        if not _server_api_key:
            return await call_next(request)
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header"})
        token = auth_header[7:]
        if token != _server_api_key:
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
        return await call_next(request)

    async def _init_fleet():
        """Initialize fleet mode (lazy, on first request)."""
        if _state["fleet"] is not None:
            return _state["fleet"]
        from adk.fleet import load_fleet
        fleet = load_fleet(
            path=fleet_path,
            agent_names=fleet_agents,
            config=config,
        )
        _state["fleet"] = fleet
        return fleet

    async def get_agent(name: str | None = None) -> AitherAgent:
        """Get agent by name. In fleet mode, routes to the right agent."""
        if is_fleet:
            fleet = await _init_fleet()
            if name and name in fleet.registry:
                return fleet.registry.get(name)
            # Default to orchestrator
            orch = fleet.get_orchestrator()
            if orch:
                return orch
            # Fallback to first agent
            if fleet.agents:
                return fleet.agents[0]

        if _state["agent"] is None:
            _state["agent"] = AitherAgent(
                name=_state["identity"],
                identity=_state["identity"],
                config=_state["config"],
            )
        agent = _state["agent"]

        # If a different agent is requested in single mode, create it
        if name and name != agent.name:
            return AitherAgent(name=name, identity=name, config=_state["config"])

        return agent

    # ─── Metrics (Prometheus) ───

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus-compatible metrics export."""
        return PlainTextResponse(get_metrics().export(), media_type="text/plain; version=0.0.4")

    # ─── Health ───

    @app.get("/health")
    async def health():
        try:
            a = await get_agent()
            provider = a.llm.provider_name or "detecting..."
            agent_name = a.name
        except ConnectionError:
            provider = "none"
            agent_name = _state["identity"]

        result = {
            "status": "healthy",
            "agent": agent_name,
            "llm_backend": provider,
            "version": __version__,
            "gateway_connected": _state.get("gateway_connected", False),
        }

        if is_fleet and _state["fleet"]:
            fleet = _state["fleet"]
            result["fleet"] = {
                "name": fleet.name,
                "agents": fleet.registry.agent_names,
                "orchestrator": fleet.orchestrator_name,
            }

        return result

    # ─── No-backend handler ───

    @app.get("/demo")
    async def demo_redirect():
        """Redirect to demo.aitherium.com when no local backend is available."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse("https://demo.aitherium.com")

    @app.exception_handler(ConnectionError)
    async def _no_backend_handler(request: Request, exc: ConnectionError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_backend",
                "message": "No LLM backend available. Set AITHER_API_KEY to use the gateway, or install Ollama locally.",
                "demo": "https://demo.aitherium.com",
                "gateway": "https://gateway.aitherium.com",
                "docs": "https://github.com/Aitherium/aither/blob/main/docs/GETTING_STARTED.md",
            },
        )

    # ─── Fleet endpoints ───

    @app.get("/agents")
    async def list_agents_endpoint():
        """List all agents in the fleet (or the single agent)."""
        if is_fleet:
            fleet = await _init_fleet()
            return {
                "fleet": fleet.name,
                "orchestrator": fleet.orchestrator_name,
                "agents": fleet.registry.list(),
            }
        a = await get_agent()
        return {
            "fleet": None,
            "orchestrator": a.name,
            "agents": [{
                "name": a.name,
                "identity": a._identity.name,
                "description": a._identity.description,
                "skills": a._identity.skills,
                "tools": [t.name for t in a._tools.list_tools()],
                "status": "running",
            }],
        }

    @app.post("/agents/{agent_name}/chat")
    async def agent_chat(agent_name: str, request: Request):
        """Chat with a specific agent in the fleet."""
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id")
        request_id = get_trace_id()

        a = await get_agent(agent_name)
        start = time.time()
        resp = await a.chat(message, session_id=session_id)
        latency_ms = (time.time() - start) * 1000

        # Record metrics (safe — latency_ms may be MagicMock in tests)
        try:
            _metrics = get_metrics()
            _metrics.record_request(latency_ms=latency_ms, status_code=200)
            _metrics.record_llm_call(
                model=str(resp.model or ""), latency_ms=float(resp.latency_ms or 0),
                tokens=int(resp.tokens_used or 0),
            )
        except (TypeError, ValueError):
            pass

        # Fire-and-forget Strata ingest
        asyncio.ensure_future(_strata_ingest(
            agent=a.name, session_id=resp.session_id,
            user_message=message, assistant_response=resp.content,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
        ))

        # Fire-and-forget Chronicle log
        asyncio.ensure_future(_chronicle_log_chat(
            agent=a.name, session_id=resp.session_id,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, request_id=request_id,
        ))

        return {
            "response": resp.content,
            "agent": a.name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
            "artifacts": resp.artifacts,
            "request_id": request_id,
        }

    @app.get("/agents/{agent_name}/sessions")
    async def agent_sessions(agent_name: str):
        """List conversation sessions for an agent."""
        from adk.conversations import get_conversation_store
        store = get_conversation_store()
        sessions = await store.list_sessions(agent_name=agent_name)
        return {"agent": agent_name, "sessions": sessions}

    @app.post("/forge/dispatch")
    async def forge_dispatch(request: Request):
        """Dispatch a task via AgentForge."""
        from adk.forge import ForgeSpec, get_forge
        body = await request.json()
        spec = ForgeSpec(
            agent_type=body.get("agent", body.get("agent_type", "auto")),
            task=body.get("task", body.get("message", "")),
            timeout=body.get("timeout", 120.0),
            effort=body.get("effort", 5),
            context=body.get("context", ""),
        )
        forge = get_forge()
        result = await forge.dispatch(spec)
        return {
            "content": result.content,
            "agent": result.agent,
            "tokens_used": result.tokens_used,
            "tool_calls": result.tool_calls,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }

    # ─── Genesis-compatible chat ───

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id")
        agent_name = body.get("agent")
        request_id = get_trace_id()

        # Inference controls (null=auto pattern — only pass if explicitly set)
        chat_kwargs: dict[str, Any] = {}
        if body.get("effort") is not None:
            chat_kwargs["effort"] = int(body["effort"])
        if body.get("temperature") is not None:
            chat_kwargs["temperature"] = float(body["temperature"])
        if body.get("top_p") is not None:
            chat_kwargs["top_p"] = float(body["top_p"])
        if body.get("repetition_penalty") is not None:
            chat_kwargs["repetition_penalty"] = float(body["repetition_penalty"])
        if body.get("max_tokens") is not None:
            chat_kwargs["max_tokens"] = int(body["max_tokens"])
        if body.get("model") is not None:
            chat_kwargs["model"] = body["model"]
        if body.get("tool_choice") is not None:
            chat_kwargs["tool_choice"] = body["tool_choice"]

        a = await get_agent(agent_name)
        start = time.time()
        resp = await a.chat(message, session_id=session_id, **chat_kwargs)
        latency_ms = (time.time() - start) * 1000

        # Record metrics (safe — latency_ms may be MagicMock in tests)
        try:
            _metrics = get_metrics()
            _metrics.record_request(latency_ms=latency_ms, status_code=200)
            _metrics.record_llm_call(
                model=str(resp.model or ""), latency_ms=float(resp.latency_ms or 0),
                tokens=int(resp.tokens_used or 0),
            )
        except (TypeError, ValueError):
            pass

        # Fire-and-forget Strata ingest (training loop)
        asyncio.ensure_future(_strata_ingest(
            agent=a.name, session_id=resp.session_id,
            user_message=message, assistant_response=resp.content,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
        ))

        # Fire-and-forget Chronicle log
        asyncio.ensure_future(_chronicle_log_chat(
            agent=a.name, session_id=resp.session_id,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, request_id=request_id,
        ))

        return {
            "response": resp.content,
            "agent": a.name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
            "artifacts": resp.artifacts,
            "request_id": request_id,
            "finish_reason": resp.finish_reason,
            "effort_level": resp.effort_level,
            "cache_status": resp.cache_status,
        }

    # ─── AitherOS-typed SSE streaming ───

    @app.post("/stream")
    async def stream_chat(request: Request):
        """SSE stream using AitherOS event protocol.

        Emits typed events: session_start, thinking, tool_call, tool_result,
        token, answer, complete — matching the Genesis/MicroScheduler protocol
        so shell-core's useAitherStream works identically against ADK and Genesis.
        """
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id") or f"adk-{uuid.uuid4().hex[:8]}"
        agent_name = body.get("agent")
        reasoning = body.get("reasoning", False)

        return StreamingResponse(
            _aitheros_stream(get_agent, message, session_id, agent_name, reasoning),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ─── Artifact endpoints ───

    @app.get("/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        """Get artifact metadata by ID."""
        from adk.artifacts import get_registry
        art = get_registry().get_by_id(artifact_id)
        if not art:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return art.to_dict()

    @app.get("/sessions/{session_id}/artifacts")
    async def get_session_artifacts(session_id: str):
        """List artifacts produced in a session."""
        from adk.artifacts import get_registry
        arts = get_registry().get(session_id)
        return {"session_id": session_id, "artifacts": [a.to_dict() for a in arts]}

    # ─── OpenAI-compatible endpoints ───

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages_raw = body.get("messages", [])
        model = body.get("model")
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 4096)
        stream = body.get("stream", False)

        a = await get_agent()

        # Convert to Message objects
        messages = [Message(role=m["role"], content=m.get("content", "")) for m in messages_raw]

        if stream:
            # Extract last user message for agent.chat_stream()
            last_user_msg = ""
            history_for_stream = []
            for m in messages_raw:
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                if m.get("role") in ("user", "assistant"):
                    history_for_stream.append({"role": m["role"], "content": m.get("content", "")})
            # Remove last user message from history (chat_stream takes it separately)
            if history_for_stream and history_for_stream[-1]["role"] == "user":
                history_for_stream = history_for_stream[:-1]

            return StreamingResponse(
                _stream_agent_response(a, last_user_msg, history_for_stream, model),
                media_type="text/event-stream",
            )

        resp = await a.llm.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.content},
                    "finish_reason": resp.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.tokens_used,
            },
        }

    @app.get("/v1/models")
    async def list_models_endpoint():
        a = await get_agent()
        try:
            models = await a.llm.list_models()
        except Exception:
            models = []

        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
                for m in models
            ],
        }

    @app.get("/v1/identities")
    async def list_identities_endpoint():
        """List available agent identities."""
        return {"identities": list_identities()}

    # ─── Strata ingest helper (fire-and-forget) ───

    async def _strata_ingest(**kwargs):
        """Send chat data to Strata for training/analytics. Never blocks or raises."""
        try:
            from adk.strata import get_strata_ingest
            strata = get_strata_ingest()
            await strata.ingest_chat(**kwargs)
        except Exception:
            pass  # Truly fire-and-forget

    # ─── MCP server (every node is also an MCP server) ───

    async def _init_mcp_server():
        """Initialize the MCP server so this node SERVES tools, not just consumes them."""
        try:
            from adk.mcp_server import MCPServer

            a = _state.get("agent")
            if a is None and not is_fleet:
                a = AitherAgent(
                    name=_state["identity"],
                    identity=_state["identity"],
                    config=_state["config"],
                )
                _state["agent"] = a

            # Build a merged registry: agent tools + fleet tools
            if is_fleet and _state.get("fleet"):
                from adk.tools import ToolRegistry
                merged = ToolRegistry()
                for fleet_agent in _state["fleet"].agents:
                    for td in fleet_agent._tools.list_tools():
                        # Prefix with agent name to avoid collisions
                        prefixed_name = f"{fleet_agent.name}__{td.name}"
                        merged._tools[prefixed_name] = td._replace(name=prefixed_name) if hasattr(td, '_replace') else td
                        # Also keep original name from first agent that has it
                        if td.name not in merged._tools:
                            merged._tools[td.name] = td
                mcp = MCPServer(tool_registry=merged, server_name=_state["fleet"].name)
            elif a:
                mcp = MCPServer(tool_registry=a._tools, server_name=a.name)
            else:
                mcp = MCPServer(server_name=_state["identity"])

            mcp.mount(app)
            _state["mcp_server"] = mcp

            # Wire relay → MCP server so inbound mesh tool calls are handled locally
            relay_obj = _state.get("relay")
            if relay_obj:
                relay_obj.set_local_mcp_server(mcp)

            logger.info("MCP server initialized (%d tools)", len(mcp.registry.list_tools()))
        except Exception as exc:
            logger.debug("MCP server init failed (non-fatal): %s", exc)

    # ─── A2A protocol server (Google A2A v0.3.0) ───

    async def _init_a2a_server():
        """Initialize the A2A protocol server for cross-agent interop."""
        try:
            from adk.a2a import A2AServer

            a = _state.get("agent")
            base_url = f"http://localhost:{port}"

            a2a = A2AServer(
                agent=a,
                base_url=base_url,
                server_name=a.name if a else _state.get("identity", "adk-agent"),
            )
            a2a.mount(app)
            _state["a2a_server"] = a2a
            logger.info("A2A server initialized (protocol v0.3.0)")
        except Exception as exc:
            logger.debug("A2A server init failed (non-fatal): %s", exc)

    async def _connect_service_bridge():
        """Connect ServiceBridge to discover AitherOS services (non-fatal)."""
        try:
            from adk.services import ServiceBridge
            bridge = ServiceBridge()
            status = await bridge.connect()
            _state["service_bridge"] = bridge

            if status.mode != "standalone":
                # Register MCP tools on fleet agents or single agent
                if is_fleet and _state["fleet"]:
                    for a in _state["fleet"].agents:
                        await bridge.register_on_agent(a)
                elif _state["agent"]:
                    await bridge.register_on_agent(_state["agent"])
            else:
                # Visible warning when AitherOS is not detected
                import sys
                agent = _state.get("agent")
                builtin_count = len(agent._tools.list_tools()) if agent else 0
                print(
                    "\n\033[33m\u26a0  STANDALONE MODE \u2014 AitherOS not detected\033[0m\n"
                    "   AitherNode (localhost:8080) and Genesis (localhost:8001) "
                    "are unreachable.\n"
                    f"   Only {builtin_count} built-in tools available "
                    f"(vs 449+ with AitherOS).\n"
                    "   Start AitherOS or set AITHER_NODE_URL to connect.\n",
                    file=sys.stderr,
                )
                # Start background reconnect so we auto-upgrade when
                # AitherOS services come online
                await bridge.start_background_reconnect()

            logger.info("ServiceBridge mode: %s (tools: %d)",
                        status.mode, status.tools_count)
        except Exception as exc:
            logger.debug("ServiceBridge startup failed (non-fatal): %s", exc)

    async def _flush_strata_queue():
        """Flush any queued Strata entries from previous sessions."""
        try:
            from adk.strata import get_strata_ingest
            strata = get_strata_ingest()
            flushed = await strata.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Strata entries", flushed)
        except Exception:
            pass

    async def _chronicle_log_chat(**kwargs):
        """Send chat event to Chronicle. Never blocks or raises."""
        try:
            from adk.chronicle import get_chronicle
            chronicle = get_chronicle()
            await chronicle.log_llm_call(**kwargs)
        except Exception:
            pass  # Truly fire-and-forget

    async def _flush_chronicle_queue():
        """Flush any queued Chronicle entries from previous sessions."""
        try:
            from adk.chronicle import get_chronicle
            chronicle = get_chronicle()
            flushed = await chronicle.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Chronicle entries", flushed)
        except Exception:
            pass

    async def _start_watch_reporter():
        """Start the background Watch health reporter."""
        try:
            from adk.watch import get_watch_reporter
            reporter = get_watch_reporter()

            # Register a collector that reports fleet/agent state
            def _collect_health():
                data = {"version": __version__}
                try:
                    if is_fleet and _state["fleet"]:
                        fleet = _state["fleet"]
                        data["agents"] = fleet.registry.agent_names
                        data["agent_count"] = len(fleet.agents)
                    elif _state["agent"]:
                        data["agents"] = [_state["agent"].name]
                        data["agent_count"] = 1
                except Exception:
                    pass
                return data

            reporter.register_collector(_collect_health)
            await reporter.start()
        except Exception as exc:
            logger.debug("Watch reporter startup failed (non-fatal): %s", exc)

    async def _flush_pulse_queue():
        """Flush any queued Pulse pain signals from previous sessions."""
        try:
            from adk.pulse import get_pulse
            pulse = get_pulse()
            flushed = await pulse.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Pulse pain signals", flushed)
        except Exception:
            pass

    async def _register_with_gateway():
        if not config.gateway_url or not config.aither_api_key:
            logger.debug("Gateway auto-registration skipped (not configured)")
            return
        if not config.register_agent:
            logger.debug("Gateway auto-registration skipped (AITHER_REGISTER_AGENT not set)")
            _state["gateway_connected"] = False
            return
        try:
            from adk.gateway import GatewayClient
            gw = GatewayClient(gateway_url=config.gateway_url, api_key=config.aither_api_key)
            ident = load_identity(identity)
            result = await gw.register_agent(
                agent_name=ident.name,
                capabilities=ident.skills,
                description=ident.description,
            )
            _state["gateway_connected"] = True
            logger.info("Registered with gateway %s: %s", config.gateway_url, result)
        except Exception as exc:
            _state["gateway_connected"] = False
            logger.warning("Gateway registration failed (non-fatal): %s", exc)

    async def _join_aithernet():
        """Auto-join the AitherNet mesh relay if API key is configured."""
        if not config.aither_api_key:
            return
        try:
            from adk.relay import get_relay
            agent_names = []
            if is_fleet and _state.get("fleet"):
                agent_names = [a.name for a in _state["fleet"].agents]
            elif _state.get("agent"):
                agent_names = [_state["agent"].name]

            relay = get_relay(
                api_key=config.aither_api_key,
                gateway_url=config.gateway_url or "",
                node_name=os.getenv("AITHER_NODE_NAME", ""),
                agents=agent_names,
                capabilities=_detect_node_capabilities(),
                port=port,
            )
            result = await relay.register()
            if result.get("ok") is not False:
                _state["relay"] = relay
                await relay.start_heartbeat(interval=60)
                logger.info(
                    "Joined AitherNet mesh as %s (node_id=%s, agents=%s)",
                    relay.node_name, relay.node_id[:12], agent_names,
                )
        except Exception as exc:
            logger.debug("AitherNet join failed (non-fatal): %s", exc)

    def _detect_node_capabilities() -> list[str]:
        """Detect what this node can do."""
        caps = ["chat", "tools", "mcp", "a2a", "irc", "smtp"]
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                caps.append("inference")
                caps.append("gpu")
        except Exception:
            pass
        if is_fleet:
            caps.append("fleet")
        return caps

    # ─── Chat relay startup + endpoints ───

    async def _init_chat_relay():
        """Initialize the chat relay and wire federation handlers."""
        try:
            from adk.chat import get_chat_relay
            relay_obj = _state.get("relay")
            node_id = relay_obj.node_id if relay_obj else ""
            chat = get_chat_relay(node_id=node_id)
            _state["chat_relay"] = chat

            # Register agents as chat participants
            if is_fleet and _state.get("fleet"):
                for a in _state["fleet"].agents:
                    chat.register_agent(a.name)
            elif _state.get("agent"):
                chat.register_agent(_state["agent"].name)

            # Wire federation: relay mesh "chat" messages → local chat
            if relay_obj:
                relay_obj.on("chat", chat.handle_federated_message)
                relay_obj.on("mail", lambda data: _handle_mesh_mail(data))

            # Start raw IRC protocol server (non-fatal)
            try:
                irc_port = int(os.getenv("AITHER_IRC_PORT", "6667"))
                await chat.start_irc_server(port=irc_port)
                logger.info("IRC server listening on port %d", irc_port)
            except Exception as irc_exc:
                logger.debug("IRC server startup failed (non-fatal): %s", irc_exc)

            # Start Aither ↔ IRC bridge (AT Protocol social feed in IRC)
            try:
                from adk.aither_bridge import init_aither_bridge
                bridge = await init_aither_bridge(chat)
                if bridge:
                    _state["aither_bridge"] = bridge
                    logger.info("Aither ↔ IRC bridge active")
            except Exception as bridge_exc:
                logger.debug("Aither bridge startup failed (non-fatal): %s", bridge_exc)

            logger.info("Chat relay initialized (channels=%d)", len(chat._channels))
        except Exception as exc:
            logger.debug("Chat relay init failed (non-fatal): %s", exc)

    async def _init_mail_relay():
        """Initialize the mail relay."""
        try:
            from adk.smtp import get_mail_relay
            relay_obj = _state.get("relay")
            node_id = relay_obj.node_id if relay_obj else ""
            mail = get_mail_relay(node_id=node_id)
            _state["mail_relay"] = mail

            # Auto-provision mailboxes for agents
            if is_fleet and _state.get("fleet"):
                for a in _state["fleet"].agents:
                    mail.provision_mailbox(a.name)
            elif _state.get("agent"):
                mail.provision_mailbox(_state["agent"].name)

            # Start inbound SMTP listener (non-fatal)
            smtp_port = int(os.getenv("AITHER_SMTP_PORT", "2525"))
            try:
                started = await mail.start_inbound_server(port=smtp_port)
                if started:
                    logger.info("Inbound SMTP server started on port %d", smtp_port)
            except Exception as smtp_exc:
                logger.debug("Inbound SMTP server startup failed (non-fatal): %s", smtp_exc)

            logger.info("Mail relay initialized (configured=%s)", mail.is_configured)
        except Exception as exc:
            logger.debug("Mail relay init failed (non-fatal): %s", exc)

    def _handle_mesh_mail(data: dict):
        """Handle incoming mail from the mesh relay."""
        try:
            mail = _state.get("mail_relay")
            if mail:
                mail.receive_mesh_mail(data)
        except Exception as exc:
            logger.debug("Mesh mail handler error: %s", exc)

    # ── Chat WebSocket ──

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        """WebSocket endpoint for real-time chat (IRC-compatible)."""
        chat = _state.get("chat_relay")
        if not chat:
            await websocket.close(code=4000, reason="Chat relay not initialized")
            return

        await websocket.accept()
        nick = f"user_{uuid.uuid4().hex[:6]}"

        try:
            # Wait for join message with nick
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            if init_data.get("type") == "join":
                nick = init_data.get("nick", nick)
                channel = init_data.get("channel", "#general")
            else:
                channel = "#general"

            chat.connect_ws(nick, websocket)
            chat.join(channel, nick)

            # Send channel history
            history = chat.history(channel, limit=50)
            await websocket.send_json({"type": "history", "channel": channel, "messages": history})

            # Message loop
            while True:
                data = await websocket.receive_json()
                await chat.handle_ws_message(nick, data)

        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            pass
        except Exception as exc:
            logger.debug("WebSocket chat error: %s", exc)
        finally:
            chat.disconnect_ws(nick)
            # Part all channels
            user = chat._users.get(nick)
            if user:
                for ch in list(user.channels):
                    chat.part(ch, nick)

    # ── Chat REST endpoints ──

    @app.get("/chat/channels")
    async def chat_channels():
        """List available chat channels."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"channels": []}
        return {"channels": chat.list_channels()}

    @app.get("/chat/channels/{channel}/history")
    async def chat_channel_history(channel: str, limit: int = 50):
        """Get message history for a channel."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"messages": []}
        ch = f"#{channel}" if not channel.startswith("#") else channel
        return {"channel": ch, "messages": chat.history(ch, limit=limit)}

    @app.get("/chat/channels/{channel}/users")
    async def chat_channel_users(channel: str):
        """List users in a channel."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"users": []}
        ch = f"#{channel}" if not channel.startswith("#") else channel
        return {"channel": ch, "users": chat.who(ch)}

    @app.post("/chat/channels/{channel}/message")
    async def chat_post_message(channel: str, request: Request):
        """Post a message to a channel (REST alternative to WebSocket)."""
        chat = _state.get("chat_relay")
        if not chat:
            return JSONResponse({"error": "Chat relay not initialized"}, status_code=503)
        body = await request.json()
        nick = body.get("nick", body.get("from", "api"))
        content = body.get("content", body.get("message", ""))
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        ch = f"#{channel}" if not channel.startswith("#") else channel
        msg = chat.post(ch, nick, content)

        # Federate to mesh
        if msg and _state.get("relay"):
            asyncio.ensure_future(chat.federate_message(msg))

        return {"ok": bool(msg), "msg_id": msg.msg_id if msg else None}

    @app.get("/chat/users")
    async def chat_online_users():
        """List all online users across channels."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"users": []}
        return {"users": chat.online_users()}

    @app.get("/chat/status")
    async def chat_status():
        """Chat relay status."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"active": False, "message": "Chat relay not initialized"}
        return {**chat.status(), "active": True}

    @app.get("/bridge/status")
    async def bridge_status():
        """Aither ↔ IRC bridge status."""
        bridge = _state.get("aither_bridge")
        if not bridge:
            return {"active": False, "message": "Aither bridge not running"}
        return {**bridge.status(), "active": True}

    # ── Mail REST endpoints ──

    @app.post("/mail/send")
    async def mail_send(request: Request):
        """Send an email (queued for delivery)."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        result = await mail.send(
            to=body.get("to", ""),
            subject=body.get("subject", ""),
            body=body.get("body", ""),
            html=body.get("html", ""),
            from_addr=body.get("from", ""),
            agent=body.get("agent", ""),
            attachments=body.get("attachments"),
        )
        return result

    @app.get("/mail/inbox")
    async def mail_inbox(agent: str = "", limit: int = 50):
        """Get received emails."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.inbox(agent=agent, limit=limit)}

    @app.get("/mail/sent")
    async def mail_sent(agent: str = "", limit: int = 50):
        """Get sent/queued emails."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.sent(agent=agent, limit=limit)}

    @app.get("/mail/email/{email_id}")
    async def mail_get_email(email_id: str):
        """Get email by ID."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "not_found"}, status_code=404)
        email_obj = mail.get_email(email_id)
        if not email_obj:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return email_obj

    @app.post("/mail/config")
    async def mail_configure(request: Request):
        """Configure SMTP settings."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        mail.configure(**body)
        return {"ok": True, "config": mail.get_config()}

    @app.get("/mail/config")
    async def mail_get_config():
        """Get SMTP configuration (password redacted)."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"configured": False}
        return mail.get_config()

    @app.get("/mail/providers")
    async def mail_providers():
        """List available SMTP provider presets."""
        from adk.smtp import PROVIDER_PRESETS
        return {"providers": PROVIDER_PRESETS}

    @app.post("/mail/mailbox/provision")
    async def mail_provision_mailbox(request: Request):
        """Provision a mailbox for a user or agent."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        return mail.provision_mailbox(
            username=body.get("username", ""),
            email_address=body.get("email_address", ""),
            display_name=body.get("display_name", ""),
            domain=body.get("domain", ""),
        )

    @app.get("/mail/mailboxes")
    async def mail_list_mailboxes():
        """List all provisioned mailboxes."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"mailboxes": []}
        return {"mailboxes": mail.list_mailboxes()}

    @app.get("/mail/mailbox/{username}/inbox")
    async def mail_user_inbox(username: str, limit: int = 50):
        """Get inbox for a specific user/agent."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.inbox(agent=username, limit=limit)}

    @app.get("/mail/status")
    async def mail_status():
        """Mail relay status."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"active": False, "message": "Mail relay not initialized"}
        return {**mail.status(), "active": True}

    # ─── Mesh relay endpoints ───

    @app.get("/mesh/status")
    async def mesh_status():
        """AitherNet mesh relay status."""
        relay = _state.get("relay")
        if not relay:
            return {"joined": False, "message": "Set AITHER_API_KEY to join AitherNet"}
        return relay.status()

    @app.get("/mesh/nodes")
    async def mesh_nodes(capability: str = "", limit: int = 50):
        """Discover other nodes on the mesh."""
        relay = _state.get("relay")
        if not relay:
            return {"nodes": [], "message": "Not connected to AitherNet"}
        nodes = await relay.discover(capability=capability, limit=limit)
        return {"nodes": [n.__dict__ for n in nodes], "total": len(nodes)}

    @app.post("/mesh/relay")
    async def mesh_relay(request: Request):
        """Relay a message to another node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        to_node = body.get("to_node", "")
        msg_type = body.get("msg_type", "chat")
        payload = body.get("payload", {})
        if not to_node:
            return JSONResponse({"error": "to_node is required"}, status_code=400)
        ok = await relay.send(to_node, msg_type, payload)
        return {"ok": ok, "relayed_to": to_node, "msg_type": msg_type}

    @app.get("/mesh/messages")
    async def mesh_messages():
        """Poll for relay messages addressed to this node."""
        relay = _state.get("relay")
        if not relay:
            return {"messages": []}
        messages = await relay.poll_messages()
        return {"messages": [m.__dict__ for m in messages], "count": len(messages)}

    @app.post("/mesh/tools/call")
    async def mesh_tool_call(request: Request):
        """Call an MCP tool on a remote mesh node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        node_id = body.get("node_id", "")
        tool_name = body.get("name", body.get("tool", ""))
        arguments = body.get("arguments", {})
        if not node_id or not tool_name:
            return JSONResponse({"error": "node_id and name are required"}, status_code=400)
        result = await relay.call_remote_tool(node_id, tool_name, arguments)
        return result

    @app.get("/mesh/tools")
    async def mesh_discover_tools(node_id: str = ""):
        """Discover MCP tools on mesh nodes."""
        relay = _state.get("relay")
        if not relay:
            return {"tools": [], "message": "Not connected to AitherNet"}
        if node_id:
            tools = await relay.list_remote_tools(node_id)
            return {"node_id": node_id, "tools": tools}
        all_tools = await relay.discover_mesh_tools()
        return {"tools": all_tools, "total": len(all_tools)}

    @app.post("/mesh/agent/call")
    async def mesh_agent_call(request: Request):
        """Call an agent on a remote mesh node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        agent_name = body.get("agent", "")
        message = body.get("message", body.get("content", ""))
        target_node = body.get("node_id", "")
        if not agent_name or not message:
            return JSONResponse({"error": "agent and message are required"}, status_code=400)
        result = await relay.call_remote_agent(agent_name, message, target_node=target_node)
        return result

    # ─── MCP server status ───

    @app.get("/mcp/status")
    async def mcp_status():
        """MCP server status."""
        mcp = _state.get("mcp_server")
        if not mcp:
            return {"active": False, "message": "MCP server not initialized"}
        return {**mcp.status(), "active": True}

    # ─── Aeon — Multi-Agent Group Chat ───

    _state["aeon_sessions"] = {}

    @app.post("/aeon/chat")
    async def aeon_chat(request: Request):
        """Multi-agent group chat. Creates or reuses an AeonSession."""
        from adk.aeon import AeonSession, AEON_PRESETS

        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        session_id = body.get("session_id")
        preset = body.get("preset", "balanced")
        participants = body.get("participants")
        rounds = body.get("rounds", 1)
        synthesize = body.get("synthesize", True)

        # Reuse existing session or create new
        sessions = _state["aeon_sessions"]
        if session_id and session_id in sessions:
            session = sessions[session_id]
        else:
            session = AeonSession(
                participants=participants,
                preset=preset,
                rounds=rounds,
                synthesize=synthesize,
                config=config,
            )
            sessions[session.session_id] = session

        response = await session.chat(message)
        return {
            "session_id": response.session_id,
            "messages": [m.to_dict() for m in response.messages],
            "synthesis": response.synthesis.to_dict() if response.synthesis else None,
            "total_tokens": response.total_tokens,
            "total_latency_ms": response.total_latency_ms,
            "round_number": response.round_number,
            "participants": session.participants,
        }

    @app.get("/aeon/presets")
    async def aeon_presets():
        """List available group chat presets."""
        from adk.aeon import AEON_PRESETS
        return {"presets": AEON_PRESETS}

    @app.get("/aeon/sessions/{session_id}")
    async def aeon_session_detail(session_id: str):
        """Get history and stats for an Aeon session."""
        sessions = _state["aeon_sessions"]
        if session_id not in sessions:
            return JSONResponse({"error": "session not found"}, status_code=404)
        session = sessions[session_id]
        return {
            "session_id": session.session_id,
            "participants": session.participants,
            "history": [m.to_dict() for m in session.history],
            "rounds": session._round_counter,
            "total_messages": len(session.history),
        }

    return app


async def _aitheros_stream(get_agent_fn, message: str, session_id: str, agent_name: str | None, reasoning: bool):
    """SSE generator emitting AitherOS-typed events.

    Uses the app-level get_agent() for shared memory/tools/fleet support.
    Emits heartbeat during sync tool execution to prevent frontend timeout.

    Protocol:
      event: session_start  -> {session_id, agent, model}
      event: heartbeat      -> {} (every 2s during tool execution)
      event: tool_call      -> {tools: [{name, args}]}
      event: tool_result    -> {results: [{tool, success, output}]}
      event: token          -> {t: "chunk", n: count}
      event: answer         -> {answer: "full response"}
      event: complete       -> {duration_ms, tokens_used}
    """
    start = time.time()
    try:
        agent = await get_agent_fn(agent_name)
        model_name = getattr(agent.llm, "provider_name", "unknown")

        # session_start
        yield f"event: session_start\ndata: {json.dumps({'type': 'session_start', 'session_id': session_id, 'agent': agent.name, 'model': model_name})}\n\n"

        # If agent has tools, use sync chat with background heartbeat
        if agent._tools.list_tools():
            # Run chat in a task with heartbeat to keep SSE alive
            chat_task = asyncio.ensure_future(agent.chat(message, session_id=session_id))
            while not chat_task.done():
                yield f"event: heartbeat\ndata: {json.dumps({'type': 'heartbeat'})}\n\n"
                await asyncio.sleep(2)
            resp = chat_task.result()

            # Emit tool calls if any
            if resp.tool_calls_made:
                for tc in resp.tool_calls_made:
                    yield f"event: tool_call\ndata: {json.dumps({'type': 'tool_call', 'tools': [{'name': tc.get('name', '?'), 'args': tc.get('args', {})}]})}\n\n"
                    yield f"event: tool_result\ndata: {json.dumps({'type': 'tool_result', 'results': [{'tool': tc.get('name', '?'), 'success': True, 'output': str(tc.get('output', ''))[:500]}]})}\n\n"

            yield f"event: answer\ndata: {json.dumps({'type': 'answer', 'answer': resp.content})}\n\n"

            duration_ms = int((time.time() - start) * 1000)
            yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms, 'tokens_used': resp.tokens_used, 'model': resp.model, 'session_id': session_id})}\n\n"

            # Fire-and-forget Strata + Chronicle
            asyncio.ensure_future(_strata_ingest(
                agent=agent.name, session_id=session_id,
                user_message=message, assistant_response=resp.content,
                model=resp.model, tokens_used=resp.tokens_used,
                latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
            ))
            return

        # Streaming path — no tools
        full_content = ""
        token_count = 0
        async for chunk in agent.chat_stream(message, session_id=session_id):
            if chunk:
                full_content += chunk
                token_count += 1
                yield f"event: token\ndata: {json.dumps({'type': 'token', 't': chunk, 'n': token_count})}\n\n"

        yield f"event: answer\ndata: {json.dumps({'type': 'answer', 'answer': full_content})}\n\n"

        duration_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms, 'tokens_used': token_count, 'model': model_name, 'session_id': session_id})}\n\n"

        asyncio.ensure_future(_strata_ingest(
            agent=agent.name, session_id=session_id,
            user_message=message, assistant_response=full_content,
            model=model_name, tokens_used=token_count,
            latency_ms=duration_ms, tool_calls=[],
        ))

    except Exception as exc:
        logger.error("AitherOS stream error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        duration_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms})}\n\n"


async def _stream_agent_response(agent, message: str, history: list[dict], model: str | None):
    """SSE stream generator using agent.chat_stream() — full pipeline.

    Routes through the agent's tool loop, safety, context manager, memory,
    and events — NOT a raw LLM stream bypass.

    If the agent has tools and the LLM requests a tool call, chat_stream()
    falls back to sync chat() and yields the full response as one chunk.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        async for text_chunk in agent.chat_stream(
            message, history=history or None, model=model,
        ):
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model or getattr(agent.llm, "provider_name", ""),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as exc:
        logger.error("Streaming error: %s", exc)
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model or "",
            "choices": [{"index": 0, "delta": {"content": f"Error: {exc}"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Final stop chunk
    stop_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_data)}\n\n"
    yield "data: [DONE]\n\n"


def main():
    """CLI entry point: aither-serve"""
    parser = argparse.ArgumentParser(description="AitherADK Agent Server")
    parser.add_argument("--identity", "-i", default="aither", help="Agent identity to load (single-agent mode)")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port (default: 8080)")
    parser.add_argument("--host", default=None, help="Host (default: 0.0.0.0)")
    parser.add_argument("--backend", "-b", help="LLM backend: ollama, openai, anthropic")
    parser.add_argument("--model", "-m", help="Model name override")
    parser.add_argument("--fleet", "-f", default=None, help="Fleet YAML config file for multi-agent mode")
    parser.add_argument("--agents", "-a", default=None, help="Comma-separated agent identities for fleet mode (e.g. aither,lyra,demiurge)")
    args = parser.parse_args()

    config = Config.from_env()
    if args.backend:
        config.llm_backend = args.backend
    if args.model:
        config.model = args.model

    port = args.port or config.server_port
    host = args.host or config.server_host

    # Determine mode
    fleet_path = args.fleet
    fleet_agents = args.agents.split(",") if args.agents else None
    is_fleet = bool(fleet_path or fleet_agents)

    app = create_app(
        identity=args.identity,
        config=config,
        fleet_path=fleet_path,
        fleet_agents=fleet_agents,
    )

    import uvicorn

    if config.gateway_url and config.aither_api_key:
        gateway_line = f"  Gateway: {config.gateway_url} (will register on startup)"
    else:
        gateway_line = (
            "  Gateway: not configured — set AITHER_API_KEY to connect\n"
            "  Demo:    https://demo.aitherium.com"
        )

    if is_fleet:
        agents_str = fleet_agents if fleet_agents else f"from {fleet_path}"
        print(f"Starting AitherADK fleet server — agents: {agents_str}, port: {port}")
        print(f"  Fleet:  GET  http://localhost:{port}/agents")
        print(f"  Chat:   POST http://localhost:{port}/agents/<name>/chat")
        print(f"  Forge:  POST http://localhost:{port}/forge/dispatch")
    else:
        print(f"Starting AitherADK server — identity: {args.identity}, port: {port}")

    irc_port = int(os.getenv("AITHER_IRC_PORT", "6667"))
    print(f"  Chat:   POST http://localhost:{port}/chat")
    print(f"  OpenAI: POST http://localhost:{port}/v1/chat/completions")
    print(f"  WS:     WS   ws://localhost:{port}/ws/chat")
    print(f"  IRC:    TCP  localhost:{irc_port} (mIRC, WeeChat, HexChat, irssi)")
    print(f"  MCP:    POST http://localhost:{port}/mcp (JSON-RPC 2.0)")
    print(f"  A2A:    POST http://localhost:{port}/a2a (Google A2A v0.3.0)")
    print(f"  Card:   GET  http://localhost:{port}/.well-known/agent.json")
    print(f"  Mail:   POST http://localhost:{port}/mail/send")
    print(f"  Health: GET  http://localhost:{port}/health")
    print(f"  Docs:   GET  http://localhost:{port}/docs")
    print(gateway_line)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
