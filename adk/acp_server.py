"""ACP (Agent Client Protocol) SERVER — expose an adk Agent to any ACP client.

The mirror of :mod:`adk.acp` (which DRIVES other agents). This module makes an
``adk`` agent drivable BY editors and hosts that speak ACP — Zed, VS Code,
JetBrains — over stdio JSON-RPC 2.0, with no editor-specific code.

Wire format follows the ACP spec exactly (verified against Zed's reference
``agent-client-protocol`` package): camelCase fields, ``session/new``,
``session/prompt``, and ``session/update`` notifications whose payload is nested
under ``update`` with a ``sessionUpdate`` discriminator.

Usage::

    from adk import AitherAgent
    from adk.acp_server import serve_stdio

    asyncio.run(serve_stdio(my_agent))       # speaks ACP on stdin/stdout

Only stdlib + the adk agent contract — no dependency on the reference library.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ACPServer",
    "ACPSession",
    "serve_stdio",
    "PROTOCOL_VERSION",
]

PROTOCOL_VERSION = 1

# JSON-RPC error codes (spec).
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class ACPSession:
    """One ACP conversation bound to a working directory."""

    def __init__(self, session_id: str, cwd: str, model: Optional[str] = None) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.model = model
        self.history: list[dict[str, Any]] = []


class ACPServer:
    """Serve an adk agent over the Agent Client Protocol.

    Args:
        agent: Anything exposing ``async run(prompt) -> result``. The result's
            ``output`` (or its ``str``) becomes the streamed agent message; a
            ``tool_calls`` list, when present, is streamed as ACP tool calls.
        name: Agent name reported in ``initialize`` (``agentInfo.name``).
        version: Agent version reported in ``initialize``.
    """

    def __init__(
        self,
        agent: Any,
        *,
        name: Optional[str] = None,
        version: str = "1.0.0",
    ) -> None:
        self.agent = agent
        self.name = name or getattr(agent, "name", None) or "aither-adk-agent"
        self.version = version
        self.sessions: dict[str, ACPSession] = {}
        self._writer: Any = None
        self._write_lock = asyncio.Lock()

    # ---- wire I/O ---------------------------------------------------------

    async def _write(self, obj: dict[str, Any]) -> None:
        """Write one JSON-RPC frame (newline-delimited), serialized."""
        if self._writer is None:
            return
        data = (json.dumps(obj) + "\n").encode("utf-8")
        async with self._write_lock:
            self._writer.write(data)
            drain = getattr(self._writer, "drain", None)
            if drain is not None:
                await drain()

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def session_update(self, session_id: str, update: dict[str, Any]) -> None:
        """Emit a ``session/update`` notification in ACP's nested shape."""
        await self.notify("session/update", {"sessionId": session_id, "update": update})

    # ---- protocol methods -------------------------------------------------

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        # protocolVersion is required by the spec; echo a supported version.
        client_version = params.get("protocolVersion", PROTOCOL_VERSION)
        return {
            "protocolVersion": min(int(client_version or PROTOCOL_VERSION), PROTOCOL_VERSION),
            "agentCapabilities": {
                "loadSession": True,
                "promptCapabilities": {"image": False, "audio": False},
            },
            "agentInfo": {"name": self.name, "version": self.version},
            "authMethods": [],
        }

    async def handle_new_session(self, params: dict[str, Any]) -> dict[str, Any]:
        cwd = params.get("cwd") or os.getcwd()
        session_id = f"adk-{uuid.uuid4().hex[:12]}"
        self.sessions[session_id] = ACPSession(session_id, cwd, params.get("model"))
        return {"sessionId": session_id}

    async def handle_load_session(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId") or ""
        if session_id not in self.sessions:
            # Loading an unknown session recreates it rather than failing the editor.
            self.sessions[session_id] = ACPSession(
                session_id, params.get("cwd") or os.getcwd()
            )
        return {}

    async def handle_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId") or ""
        session = self.sessions.get(session_id)
        if session is None:
            raise _RpcError(_INVALID_PARAMS, f"unknown sessionId {session_id!r}")

        text = _extract_text(params.get("prompt") or [])
        session.history.append({"role": "user", "content": text})

        try:
            result = await self.agent.run(text)
        except Exception as e:  # noqa: BLE001 — report to the editor, don't die
            logger.exception("acp_server: agent run failed")
            await self.session_update(
                session_id,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": f"[agent error] {e}"},
                },
            )
            return {"stopReason": "refusal"}

        output = getattr(result, "output", None)
        if output is None:
            output = result if isinstance(result, str) else str(result)

        # Stream any tool calls the agent made, in real ACP shape.
        for i, call in enumerate(getattr(result, "tool_calls", None) or []):
            await self._stream_tool_call(session_id, i, call)

        await self.session_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": output},
            },
        )
        session.history.append({"role": "assistant", "content": output})

        stop = getattr(result, "finish_reason", None) or "end_turn"
        # ACP stopReason vocabulary; anything else maps to end_turn.
        if stop not in ("end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"):
            stop = "end_turn"
        return {"stopReason": stop}

    async def _stream_tool_call(self, session_id: str, idx: int, call: Any) -> None:
        """Emit tool_call (start) then tool_call_update (terminal) for one call."""
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool_name") or f"tool_{idx}"
            tool_id = str(call.get("id") or call.get("tool_call_id") or f"tc-{idx}")
            raw_result = call.get("result")
        else:
            name = getattr(call, "tool_name", None) or f"tool_{idx}"
            tool_id = str(getattr(call, "tool_call_id", None) or f"tc-{idx}")
            raw_result = getattr(call, "result", None)

        await self.session_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_id,
                "title": str(name),
                "kind": "other",
                "status": "pending",
            },
        )
        await self.session_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": tool_id,
                "status": "completed",
                "content": [
                    {
                        "type": "content",
                        "content": {"type": "text", "text": _stringify(raw_result)},
                    }
                ],
            },
        )

    async def handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    # ---- dispatch ---------------------------------------------------------

    @property
    def _routes(self) -> dict[str, Callable[[dict], Awaitable[dict]]]:
        return {
            "initialize": self.handle_initialize,
            "authenticate": lambda p: _ok(),
            "session/new": self.handle_new_session,
            "session/load": self.handle_load_session,
            "session/prompt": self.handle_prompt,
            "session/cancel": self.handle_cancel,
        }

    async def _dispatch(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        msg_id = msg.get("id")
        method = msg.get("method")
        if method is None:
            return None  # a response/notification we don't act on
        handler = self._routes.get(method)
        if handler is None:
            if msg_id is None:
                return None  # unknown notification: ignore
            return _err(msg_id, _METHOD_NOT_FOUND, f"Method not found: {method}")
        try:
            result = await handler(msg.get("params") or {})
        except _RpcError as e:
            return _err(msg_id, e.code, e.message)
        except Exception as e:  # noqa: BLE001
            logger.exception("acp_server: handler %s failed", method)
            return _err(msg_id, _INTERNAL_ERROR, f"Internal error: {e}")
        if msg_id is None:
            return None  # it was a notification
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def serve(self, reader: Any, writer: Any) -> None:
        """Read JSON-RPC frames from *reader*, answer on *writer*, until EOF."""
        self._writer = writer
        while True:
            line = await reader.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await self._write(_err(None, _PARSE_ERROR, f"Parse error: {e}"))
                continue
            if not isinstance(msg, dict):
                await self._write(_err(None, _INVALID_REQUEST, "Invalid Request"))
                continue
            response = await self._dispatch(msg)
            if response is not None:
                await self._write(response)


# ---- helpers ---------------------------------------------------------------


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def _ok() -> dict[str, Any]:
    return {}


def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _extract_text(blocks: Any) -> str:
    """Flatten ACP prompt content blocks into plain text."""
    if isinstance(blocks, str):
        return blocks
    parts: list[str] = []
    for b in blocks or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


class _ThreadStdinReader:
    """Cross-platform async line reader over a blocking binary stream.

    ``loop.connect_read_pipe`` cannot attach to stdin under Windows' Proactor
    event loop (it raises in ``_loop_reading``), so reads are delegated to the
    default executor instead. Works identically on POSIX and Windows.
    """

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stdin.buffer

    async def readline(self) -> bytes:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._stream.readline
        )


class _ThreadStdoutWriter:
    """Blocking binary writer with the ``write``/``drain`` shape ACPServer expects."""

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stdout.buffer

    def write(self, data: bytes) -> None:
        self._stream.write(data)

    async def drain(self) -> None:
        self._stream.flush()


async def serve_stdio(agent: Any, *, name: Optional[str] = None, version: str = "1.0.0") -> None:
    """Serve *agent* over ACP on stdin/stdout (the editor-facing entrypoint)."""
    server = ACPServer(agent, name=name, version=version)
    await server.serve(_ThreadStdinReader(), _ThreadStdoutWriter())
