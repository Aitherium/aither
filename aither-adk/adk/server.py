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
from typing import Any

from fastapi import FastAPI, Request
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

    app = FastAPI(
        title=f"AitherADK — {'Fleet' if is_fleet else identity}",
        version=__version__,
        docs_url="/docs",
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
    }

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
                "docs": "https://github.com/Aitherium/AitherOS-Alpha/blob/main/docs/GETTING_STARTED.md",
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
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
            "request_id": request_id,
        }

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
            return StreamingResponse(
                _stream_response(a, messages, model, temperature, max_tokens),
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

    # ─── Gateway auto-registration (fire-and-forget) ───

    @app.on_event("startup")
    async def _startup_tasks():
        """Run all startup tasks."""
        # Configure structured logging
        from adk.chronicle import configure_logging
        configure_logging(
            level=os.getenv("AITHER_LOG_LEVEL", "INFO"),
            json_output=config.json_logging,
        )

        await _register_with_gateway()
        await _flush_strata_queue()
        await _flush_chronicle_queue()
        await _start_watch_reporter()
        await _flush_pulse_queue()

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

    return app


async def _stream_response(agent, messages, model, temperature, max_tokens):
    """SSE stream generator for OpenAI-compatible streaming."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async for chunk in agent.llm.chat_stream(
        messages, model=model, temperature=temperature, max_tokens=max_tokens
    ):
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chunk.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.content} if chunk.content else {},
                    "finish_reason": "stop" if chunk.done else None,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"

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

    print(f"  Chat:   POST http://localhost:{port}/chat")
    print(f"  OpenAI: POST http://localhost:{port}/v1/chat/completions")
    print(f"  Health: GET  http://localhost:{port}/health")
    print(f"  Docs:   GET  http://localhost:{port}/docs")
    print(gateway_line)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
