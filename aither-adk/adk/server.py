"""FastAPI server wrapping an AitherAgent — OpenAI-compatible + Genesis-compatible."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from adk import __version__
from adk.agent import AitherAgent, AgentResponse
from adk.config import Config
from adk.identity import list_identities, load_identity
from adk.llm import LLMRouter, Message

logger = logging.getLogger("adk.server")


def create_app(
    agent: AitherAgent | None = None,
    identity: str = "aither",
    config: Config | None = None,
) -> FastAPI:
    """Create a FastAPI app wrapping an AitherAgent.

    Returns a fully configured app with both OpenAI-compatible and Genesis-compatible endpoints.
    """
    config = config or Config.from_env()

    app = FastAPI(
        title=f"AitherADK — {identity}",
        version=__version__,
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy-init agent on first request
    _state: dict[str, Any] = {"agent": agent, "identity": identity, "config": config}

    async def get_agent() -> AitherAgent:
        if _state["agent"] is None:
            _state["agent"] = AitherAgent(
                name=_state["identity"],
                identity=_state["identity"],
                config=_state["config"],
            )
        return _state["agent"]

    # ─── Health ───

    @app.get("/health")
    async def health():
        try:
            a = await get_agent()
            provider = a.llm.provider_name or "detecting..."
        except ConnectionError:
            provider = "none"
        return {
            "status": "healthy",
            "agent": _state["identity"],
            "llm_backend": provider,
            "version": __version__,
            "gateway_connected": _state.get("gateway_connected", False),
        }

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

    # ─── Genesis-compatible chat ───

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id")
        agent_name = body.get("agent")

        a = await get_agent()

        # If a different agent is requested, load that identity
        if agent_name and agent_name != a.name:
            a = AitherAgent(name=agent_name, identity=agent_name, config=_state["config"])

        resp = await a.chat(message, session_id=session_id)
        return {
            "response": resp.content,
            "agent": a.name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
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

    # ─── Gateway auto-registration (fire-and-forget) ───

    @app.on_event("startup")
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
    parser.add_argument("--identity", "-i", default="aither", help="Agent identity to load")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port (default: 8080)")
    parser.add_argument("--host", default=None, help="Host (default: 0.0.0.0)")
    parser.add_argument("--backend", "-b", help="LLM backend: ollama, openai, anthropic")
    parser.add_argument("--model", "-m", help="Model name override")
    args = parser.parse_args()

    config = Config.from_env()
    if args.backend:
        config.llm_backend = args.backend
    if args.model:
        config.model = args.model

    port = args.port or config.server_port
    host = args.host or config.server_host

    app = create_app(identity=args.identity, config=config)

    import uvicorn

    if config.gateway_url and config.aither_api_key:
        gateway_line = f"  Gateway: {config.gateway_url} (will register on startup)"
    else:
        gateway_line = (
            "  Gateway: not configured — set AITHER_API_KEY to connect\n"
            "  Demo:    https://demo.aitherium.com"
        )

    print(f"Starting AitherADK server — identity: {args.identity}, port: {port}")
    print(f"  Chat:   POST http://localhost:{port}/chat")
    print(f"  OpenAI: POST http://localhost:{port}/v1/chat/completions")
    print(f"  Health: GET  http://localhost:{port}/health")
    print(f"  Docs:   GET  http://localhost:{port}/docs")
    print(gateway_line)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
