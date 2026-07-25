"""ACP (Agent Client Protocol) client for aither-adk.

Implements a JSON-RPC 2.0 stdio-based client to connect to ACP-compliant agents
(e.g., hermes, other ACP servers). Supports session management, prompting, and
tool-call streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _pick(d: dict, *names: str, default: Any = None) -> Any:
    """First present key among *names* (ACP wire is camelCase; snake = legacy)."""
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return default


# Real ACP session/update discriminator values (agent-client-protocol).
_START_KINDS = ("tool_call", "tool_call_start")
_UPDATE_KINDS = ("tool_call_update", "tool_call_complete")
_DONE_STATUSES = ("completed", "failed")


def _render_tool_result(update: dict) -> Optional[str]:
    """Extract a tool result string from a real-ACP or legacy update.

    Real ACP carries results in a ``content`` LIST of content blocks; legacy
    mocks use a flat ``result`` string. Returns None when the update carries no
    result yet (e.g. a pending/in_progress tool_call).
    """
    legacy = update.get("result")
    if isinstance(legacy, str):
        return legacy
    content = update.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            inner = block.get("content")
            if isinstance(inner, dict) and inner.get("type") == "text":
                parts.append(inner.get("text", ""))
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "".join(parts)
    return None

__all__ = [
    "ACPClient",
    "ACPCapabilities",
    "ACPUsage",
    "ACPToolCall",
    "ACPPromptResult",
    "ACPApprovalRequest",
]


class ACPCapabilities(BaseModel):
    """Agent capabilities returned from initialize."""

    protocol_version: int
    agent_name: str
    agent_version: str
    load_session: bool = False
    prompt_capabilities: dict[str, Any] = Field(default_factory=dict)
    session_capabilities: dict[str, Any] = Field(default_factory=dict)


class ACPUsage(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0


class ACPToolCall(BaseModel):
    """A tool call within a prompt response."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None


class ACPPromptResult(BaseModel):
    """Result from a prompt request."""

    text: str
    tool_calls: list[ACPToolCall] = Field(default_factory=list)
    usage: ACPUsage = Field(default_factory=ACPUsage)
    stop_reason: str = "end_turn"


class ACPApprovalRequest(BaseModel):
    """Request for user approval (e.g., for file edits, tool execution)."""

    request_id: str
    kind: str  # e.g., "edit", "execute", "tool"
    description: str


class ACPClient:
    """Async JSON-RPC 2.0 ACP client over stdio.

    Spawns/attaches to an ACP-compliant server and manages JSON-RPC framing.
    Supports initialize, create_session, prompt (with streaming), and other
    session operations.
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        approval_callback: Optional[
            Callable[[ACPApprovalRequest], bool]
        ] = None,
        subprocess_instance: Optional[Any] = None,
    ):
        """Initialize an ACP client.

        Args:
            command: Path to ACP server executable. If None, use subprocess_instance.
            args: Command-line arguments for the server.
            approval_callback: Async or sync callback for approval requests.
                               Returns True to approve, False to deny. Default: deny all.
            subprocess_instance: Existing subprocess instance to use instead of spawning.
        """
        self.command = command
        self.args = args or []
        self.approval_callback = approval_callback or (lambda _: False)
        self.subprocess = subprocess_instance
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_id_counter = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._capabilities: Optional[ACPCapabilities] = None
        # Handshake config (ACP requires protocolVersion on initialize).
        self.protocol_version = 1
        self.client_capabilities: dict[str, Any] = {"fs": {}, "terminal": False}

    async def connect(self) -> None:
        """Spawn/attach to the ACP server and start the read loop."""
        if self.subprocess is None:
            if not self.command:
                raise ValueError("Either command or subprocess_instance must be provided")
            self.subprocess = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        self._reader = self.subprocess.stdout
        self._writer = self.subprocess.stdin
        if not self._reader or not self._writer:
            raise RuntimeError("Failed to open stdio pipes")
        self._read_task = asyncio.create_task(self._read_loop())

    async def disconnect(self) -> None:
        """Close the connection and cleanup."""
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        if self.subprocess:
            try:
                await asyncio.wait_for(self.subprocess.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.subprocess.kill()

    def _get_next_request_id(self) -> int:
        """Generate the next unique JSON-RPC request ID."""
        self._request_id_counter += 1
        return self._request_id_counter

    async def _send_request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Send a JSON-RPC 2.0 request and wait for the response.

        Args:
            method: RPC method name.
            params: Method parameters (keyword args).

        Returns:
            The result field from the JSON-RPC response.

        Raises:
            RuntimeError: If the response is an error or missing result.
        """
        if not self._writer or not self._reader:
            raise RuntimeError("Not connected")

        request_id = self._get_next_request_id()
        request_obj = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request_obj["params"] = params

        # Serialize and send
        line = json.dumps(request_obj)
        self._writer.write(line.encode("utf-8") + b"\n")
        await self._writer.drain()

        # Wait for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future
        try:
            result = await asyncio.wait_for(future, timeout=60.0)
            return result
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            raise RuntimeError(f"RPC request {method} timed out")

    async def _read_loop(self) -> None:
        """Read and dispatch JSON-RPC messages from the server."""
        if not self._reader:
            return
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to decode JSON-RPC message: %s", e)
                    continue
                await self._process_message(obj)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Read loop error: %s", e, exc_info=True)

    async def _process_message(self, obj: dict[str, Any]) -> None:
        """Process an inbound JSON-RPC message."""
        # Check if it's a response to a pending request
        if "id" in obj:
            request_id = obj["id"]
            if request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if "error" in obj:
                    error_obj = obj.get("error", {})
                    error_msg = error_obj.get("message", "Unknown error")
                    future.set_exception(RuntimeError(f"RPC error: {error_msg}"))
                elif "result" in obj:
                    future.set_result(obj["result"])
                else:
                    future.set_exception(
                        RuntimeError("Invalid JSON-RPC response: no result or error")
                    )
                return

        # Check if it's a notification (method without id)
        if "method" in obj and "id" not in obj:
            await self._handle_notification(obj)

    async def _handle_notification(self, obj: dict[str, Any]) -> None:
        """Handle server-side notifications."""
        method = obj.get("method")
        if method in ("session/request_permission", "approval_request"):
            params = obj.get("params", {})
            request_id = params.get("request_id")
            kind = params.get("kind", "unknown")
            description = params.get("description", "")

            # Call approval callback
            approval_req = ACPApprovalRequest(
                request_id=request_id, kind=kind, description=description
            )
            try:
                approved = await self._call_approval_callback(approval_req)
            except Exception as e:
                logger.error("Approval callback raised: %s", e, exc_info=True)
                approved = False

            # Send approval response (notification, no id)
            if self._writer:
                response = {
                    "jsonrpc": "2.0",
                    "method": "approval_response",
                    "params": {"request_id": request_id, "approved": approved},
                }
                line = json.dumps(response)
                self._writer.write(line.encode("utf-8") + b"\n")
                await self._writer.drain()

    async def _call_approval_callback(self, req: ACPApprovalRequest) -> bool:
        """Call the approval callback, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(self.approval_callback):
            return await self.approval_callback(req)
        else:
            return self.approval_callback(req)

    async def initialize(self) -> ACPCapabilities:
        """Initialize the ACP connection.

        Returns:
            Agent capabilities.
        """
        result = await self._send_request(
            "initialize",
            {
                # protocolVersion is REQUIRED by the ACP spec; omitting it makes a
                # real agent reject the handshake with "Invalid params".
                "protocolVersion": self.protocol_version,
                "clientCapabilities": self.client_capabilities,
                "clientInfo": {"name": "aither-adk", "version": "1"},
            },
        )
        # ACP wire format is camelCase (verified against agent-client-protocol,
        # the Zed reference lib). snake_case is accepted as a lenient fallback.
        info = _pick(result, "agentInfo", "agent_info") or {}
        agent_caps = _pick(result, "agentCapabilities", "agent_capabilities") or {}
        caps = ACPCapabilities(
            protocol_version=_pick(result, "protocolVersion", "protocol_version") or 1,
            agent_name=info.get("name", "unknown"),
            agent_version=info.get("version", "0.0.0"),
            load_session=bool(_pick(agent_caps, "loadSession", "load_session") or False),
            prompt_capabilities=_pick(
                agent_caps, "promptCapabilities", "prompt_capabilities"
            ) or {},
            session_capabilities=_pick(
                agent_caps, "sessionCapabilities", "session_capabilities"
            ) or {},
        )
        self._capabilities = caps
        return caps

    async def create_session(
        self,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        mcp_servers: Optional[list] = None,
    ) -> str:
        """Create a new ACP session.

        Args:
            cwd: Working directory for the session.
            model: Model to use for the session.

        Returns:
            Session ID.
        """
        # cwd and mcpServers are REQUIRED by the ACP spec on session/new.
        params: dict[str, Any] = {
            "cwd": cwd or os.getcwd(),
            "mcpServers": mcp_servers if mcp_servers is not None else [],
        }
        if model:
            params["model"] = model
        result = await self._send_request("session/new", params)
        return _pick(result, "sessionId", "session_id") or ""

    async def prompt(
        self,
        session_id: str,
        text: str,
        *,
        drain_timeout: float = 2.0,
    ) -> ACPPromptResult:
        """Send a prompt to the agent and consume the streamed response.

        The server may stream ToolCallStart, ToolCallProgress, ToolCallComplete,
        AgentMessageChunk, and UsageUpdate messages via session/update notifications.

        Args:
            session_id: Session ID from create_session.
            text: Prompt text.
            drain_timeout: Max seconds to wait for trailing notifications after the
                prompt response, so tool calls that complete late are not dropped.
                Returns as soon as every started tool call has completed.

        Returns:
            Aggregated result with text, tool calls, and usage.
        """
        # Temporarily collect session updates during this prompt
        pending_updates: list[dict[str, Any]] = []

        original_handle = self._handle_notification
        self._handle_notification = self._make_update_collector(pending_updates)

        try:
            params = {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": text}],
            }
            result = await self._send_request("session/prompt", params)

            # Drain trailing session/update notifications (bounded, never silent).
            await self._drain_updates(pending_updates, timeout=drain_timeout)

            # Parse the aggregated response
            return self._parse_prompt_result(result, pending_updates)
        finally:
            self._handle_notification = original_handle

    def _make_update_collector(
        self, pending_updates: list[dict[str, Any]]
    ) -> Callable:
        """Return a handler that collects session/update notifications."""

        async def collector(obj: dict[str, Any]) -> None:
            method = obj.get("method")
            if method == "session/update":
                params = obj.get("params", {}) or {}
                # Real ACP nests the payload under "update"; legacy mocks send it flat.
                pending_updates.append(params.get("update", params))
            else:
                await self._handle_notification(obj)

        return collector

    @staticmethod
    def _outstanding_tool_calls(updates: list[dict[str, Any]]) -> set[str]:
        """Tool call ids that have started but not yet reported completion."""
        started: set[str] = set()
        done: set[str] = set()
        for update in updates:
            kind = _pick(update, "sessionUpdate", "session_update")
            tid = _pick(update, "toolCallId", "tool_call_id", default="")
            if kind in _START_KINDS:
                started.add(tid)
                # a start may already carry a terminal status
                if _pick(update, "status") in _DONE_STATUSES:
                    done.add(tid)
            elif kind in _UPDATE_KINDS:
                status = _pick(update, "status")
                # legacy tool_call_complete has no status but IS terminal
                if status in _DONE_STATUSES or kind == "tool_call_complete":
                    done.add(tid)
        return started - done

    async def _drain_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        timeout: float = 2.0,
        settle: float = 0.1,
        poll: float = 0.02,
    ) -> None:
        """Wait (bounded) for trailing ``session/update`` notifications.

        Replaces a fixed sleep, which silently dropped tool-call completions that
        arrived later than the sleep. Returns as soon as no tool call is
        outstanding (after a short settle window for trailing text/usage chunks),
        and at the deadline logs a WARNING naming anything still outstanding —
        so late data is never lost silently.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        deadline = start + max(timeout, settle)
        while True:
            outstanding = self._outstanding_tool_calls(updates)
            if not outstanding and (loop.time() - start) >= settle:
                return
            if loop.time() >= deadline:
                if outstanding:
                    logger.warning(
                        "acp.prompt.drain_timeout: %d tool call(s) did not complete "
                        "within %.2fs: %s",
                        len(outstanding),
                        timeout,
                        ", ".join(sorted(outstanding)),
                    )
                return
            await asyncio.sleep(poll)

    def _parse_prompt_result(
        self,
        response: dict[str, Any],
        updates: list[dict[str, Any]],
    ) -> ACPPromptResult:
        """Aggregate streamed updates into a single prompt result.

        Args:
            response: The prompt request response.
            updates: List of session/update notifications received.

        Returns:
            Parsed ACPPromptResult.
        """
        result_text = ""
        usage = ACPUsage()
        # Key by tool_call_id (NOT a single "active" slot) so interleaved tool
        # calls cannot overwrite each other, and a call that never completes is
        # still REPORTED (result=None) rather than silently dropped.
        started: dict[str, ACPToolCall] = {}

        # Process streamed updates
        for update in updates:
            update_type = _pick(update, "sessionUpdate", "session_update")
            tool_call_id = _pick(update, "toolCallId", "tool_call_id", default="")

            if update_type in ("agent_message_chunk", "agent_thought_chunk"):
                content = update.get("content") or {}
                if isinstance(content, dict) and content.get("type") == "text":
                    if update_type == "agent_message_chunk":
                        result_text += content.get("text", "")

            elif update_type in _START_KINDS:
                started[tool_call_id] = ACPToolCall(
                    tool_call_id=tool_call_id,
                    # real ACP uses `title`; legacy mocks use `tool_name`
                    tool_name=_pick(update, "title", "tool_name", default=""),
                    arguments=_pick(update, "rawInput", "function_args", default={}) or {},
                    result=_render_tool_result(update),
                )

            elif update_type in _UPDATE_KINDS:
                res = _render_tool_result(update)
                call = started.get(tool_call_id)
                if call is not None:
                    if res is not None:
                        call.result = res
                    if not call.tool_name:
                        call.tool_name = _pick(update, "title", "tool_name", default="")
                else:
                    # Update without a start — record it rather than drop it.
                    started[tool_call_id] = ACPToolCall(
                        tool_call_id=tool_call_id,
                        tool_name=_pick(update, "title", "tool_name", default=""),
                        arguments={},
                        result=res,
                    )

            elif update_type == "usage_update":
                usage = ACPUsage(
                    input_tokens=_pick(update, "inputTokens", "input_tokens", default=0),
                    output_tokens=_pick(update, "outputTokens", "output_tokens", default=0),
                )

        # Real ACP returns usage on the PromptResponse, not as a notification.
        resp_usage = _pick(response, "usage") or {}
        if resp_usage:
            usage = ACPUsage(
                input_tokens=_pick(resp_usage, "inputTokens", "input_tokens", default=0),
                output_tokens=_pick(resp_usage, "outputTokens", "output_tokens", default=0),
            )

        incomplete = [c.tool_call_id for c in started.values() if c.result is None]
        if incomplete:
            logger.warning(
                "acp.prompt.tool_calls_incomplete: %d tool call(s) reported no "
                "result: %s (returned with result=None)",
                len(incomplete),
                ", ".join(incomplete),
            )

        return ACPPromptResult(
            text=result_text,
            tool_calls=list(started.values()),
            usage=usage,
            stop_reason=_pick(response, "stopReason", "stop_reason") or "end_turn",
        )

    async def set_session_model(
        self,
        session_id: str,
        model: str,
    ) -> None:
        """Change the model for an active session.

        Args:
            session_id: Session ID.
            model: Model name/identifier.
        """
        params = {"session_id": session_id, "model": model}
        await self._send_request("session/set_model", params)

    async def load_session(
        self,
        session_id: str,
        cwd: Optional[str] = None,
    ) -> None:
        """Load/restore a persisted session.

        Args:
            session_id: Session ID to load.
            cwd: Working directory.
        """
        params = {"session_id": session_id}
        if cwd:
            params["cwd"] = cwd
        await self._send_request("session/load", params)

    async def resume_session(
        self,
        session_id: str,
        cwd: Optional[str] = None,
    ) -> None:
        """Resume a paused or stopped session.

        Args:
            session_id: Session ID to resume.
            cwd: Working directory.
        """
        params = {"session_id": session_id}
        if cwd:
            params["cwd"] = cwd
        await self._send_request("session/resume", params)

    async def fork_session(
        self,
        session_id: str,
        cwd: Optional[str] = None,
    ) -> str:
        """Fork an existing session into a new one.

        Args:
            session_id: Session to fork.
            cwd: Working directory for the new session.

        Returns:
            New session ID.
        """
        params = {"session_id": session_id}
        if cwd:
            params["cwd"] = cwd
        result = await self._send_request("session/fork", params)
        return result.get("session_id", "")

    async def list_sessions(
        self,
        cwd: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> dict[str, Any]:
        """List available sessions.

        Args:
            cwd: Filter by working directory.
            cursor: Pagination cursor.

        Returns:
            Session list response with pagination info.
        """
        params = {}
        if cwd:
            params["cwd"] = cwd
        if cursor:
            params["cursor"] = cursor
        return await self._send_request("session/list", params)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
