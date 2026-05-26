"""A2A (Agent-to-Agent) protocol server — Google A2A v0.3.0 compatible.

Exposes any AitherAgent as a compliant A2A service with:
- Agent Card discovery at ``/.well-known/agent.json``
- Task lifecycle (submitted → working → completed/failed)
- Message send/receive via JSON-RPC 2.0
- SSE streaming for long-running tasks
- Skill auto-detection from @tool decorated functions

Every ``aither-serve`` node becomes an A2A-compatible agent that can
interoperate with any other A2A agent (Google, LangGraph, CrewAI, etc.).

Usage::

    from adk.a2a import A2AServer
    a2a = A2AServer(agent=my_agent, base_url="http://localhost:8080")
    a2a.mount(app)  # adds /.well-known/agent.json + /a2a (JSON-RPC)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

logger = logging.getLogger("adk.a2a")

_PROTOCOL_VERSION = "0.3.0"


# ─────────────────────────────────────────────────────────────────────────────
# Data models (Google A2A spec)
# ─────────────────────────────────────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TextPart:
    text: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class DataPart:
    data: dict
    type: str = "data"

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data}


@dataclass
class A2AMessage:
    role: str  # "user" or "agent"
    parts: list[dict] = field(default_factory=list)
    messageId: str = ""
    taskId: str = ""

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "parts": self.parts}
        if self.messageId:
            d["messageId"] = self.messageId
        if self.taskId:
            d["taskId"] = self.taskId
        return d


@dataclass
class Artifact:
    artifactId: str = ""
    parts: list[dict] = field(default_factory=list)
    name: str = ""

    def to_dict(self) -> dict:
        d: dict = {"parts": self.parts}
        if self.artifactId:
            d["artifactId"] = self.artifactId
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class TaskStatus:
    state: TaskState = TaskState.SUBMITTED
    message: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "message": self.message,
            "timestamp": self.timestamp or _now(),
        }


@dataclass
class Task:
    id: str = ""
    contextId: str = ""
    status: TaskStatus = field(default_factory=TaskStatus)
    history: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contextId": self.contextId,
            "status": self.status.to_dict(),
            "history": self.history,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task Manager
# ─────────────────────────────────────────────────────────────────────────────

class TaskManager:
    """In-memory task lifecycle manager."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def create_task(self, context_id: str = "", metadata: dict | None = None) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            contextId=context_id or str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.SUBMITTED, timestamp=_now()),
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, state: TaskState, message: str = ""):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus(state=state, message=message, timestamp=_now())
        self._notify(task_id, {"type": "status", "task": task.to_dict()})

    def add_artifact(self, task_id: str, artifact: Artifact):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.artifacts.append(artifact.to_dict())
        self._notify(task_id, {"type": "artifact", "artifact": artifact.to_dict()})

    def add_message(self, task_id: str, message: A2AMessage):
        task = self._tasks.get(task_id)
        if not task:
            return
        task.history.append(message.to_dict())

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED):
            return False
        self.update_status(task_id, TaskState.CANCELED)
        return True

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(q)
        return q

    def _notify(self, task_id: str, event: dict):
        for q in self._subscribers.get(task_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# A2A Server
# ─────────────────────────────────────────────────────────────────────────────

class A2AServer:
    """Google A2A v0.3.0 protocol server wrapping an AitherAgent.

    Handles:
      - /.well-known/agent.json — Agent Card discovery
      - POST /a2a — JSON-RPC 2.0 (message/send, tasks/get, tasks/cancel)
      - GET /a2a/tasks/{id}/subscribe — SSE streaming
    """

    def __init__(
        self,
        agent=None,
        base_url: str = "http://localhost:8080",
        server_name: str = "",
    ):
        self._agent = agent
        self._base_url = base_url.rstrip("/")
        self._server_name = server_name
        self._tasks = TaskManager()
        self._agent_card: dict | None = None

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, value):
        self._agent = value
        self._agent_card = None  # Rebuild on next request

    def build_agent_card(self) -> dict:
        """Build the A2A agent card from the agent's identity and tools."""
        if self._agent_card:
            return self._agent_card

        # Try identity-based card first
        if self._agent and hasattr(self._agent, "_identity"):
            card = self._agent._identity.to_a2a_card(base_url=self._base_url)
        else:
            name = self._server_name or (self._agent.name if self._agent else "adk-agent")
            card = {
                "name": name,
                "description": f"AitherADK agent: {name}",
                "url": self._base_url,
                "version": _get_version(),
                "provider": {"organization": "Aitherium", "url": "https://aitherium.com"},
                "capabilities": {"streaming": True, "pushNotifications": False,
                                 "stateTransitionHistory": True},
                "authentication": {"schemes": ["bearer"]},
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["text/plain"],
                "skills": [],
            }

        # Enrich with tool-derived skills
        if self._agent and hasattr(self._agent, "_tools"):
            tool_skills = []
            for td in self._agent._tools.list_tools():
                tool_skills.append({
                    "id": td.name,
                    "name": td.name,
                    "description": td.description,
                    "tags": ["tool"],
                    "inputModes": ["application/json"],
                    "outputModes": ["text/plain"],
                })
            # Merge: keep identity skills, add tool skills
            existing_ids = {s.get("id") for s in card.get("skills", [])}
            for ts in tool_skills:
                if ts["id"] not in existing_ids:
                    card.setdefault("skills", []).append(ts)

        # Ensure streaming capability
        card.setdefault("capabilities", {})["streaming"] = True
        card.setdefault("capabilities", {})["stateTransitionHistory"] = True

        # Protocol version
        card["protocolVersion"] = _PROTOCOL_VERSION

        # Interfaces
        card["interfaces"] = [
            {"url": f"{self._base_url}/a2a", "transport": "JSONRPC"},
            {"url": f"{self._base_url}/mcp", "transport": "JSONRPC"},
        ]

        self._agent_card = card
        return card

    # ── JSON-RPC handler ──────────────────────────────────────────────────

    async def handle_jsonrpc(self, body: dict) -> dict:
        """Handle a JSON-RPC 2.0 A2A request."""
        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id")

        if method == "message/send":
            return await self._handle_message_send(req_id, params)
        elif method == "tasks/get":
            return self._handle_tasks_get(req_id, params)
        elif method == "tasks/cancel":
            return self._handle_tasks_cancel(req_id, params)
        else:
            return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")

    async def _handle_message_send(self, req_id, params: dict) -> dict:
        """Handle message/send — create or continue a task."""
        message = params.get("message", {})
        task_id = message.get("taskId", "")
        context_id = message.get("contextId", params.get("contextId", ""))

        # Extract text from message parts
        text = ""
        for part in message.get("parts", []):
            if isinstance(part, str):
                text += part
            elif isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")

        if not text:
            return _jsonrpc_error(req_id, -32602, "No text content in message")

        # Create or get task
        if task_id:
            task = self._tasks.get_task(task_id)
            if not task:
                return _jsonrpc_error(req_id, -32602, f"Task not found: {task_id}")
        else:
            task = self._tasks.create_task(context_id=context_id)

        # Record user message
        user_msg = A2AMessage(
            role="user",
            parts=message.get("parts", [{"type": "text", "text": text}]),
            messageId=str(uuid.uuid4()),
            taskId=task.id,
        )
        self._tasks.add_message(task.id, user_msg)
        self._tasks.update_status(task.id, TaskState.WORKING)

        # Execute agent chat
        try:
            if not self._agent:
                raise RuntimeError("No agent configured")

            # Build history from task context
            history = []
            for hist_msg in task.history[:-1]:  # Exclude the message we just added
                role = hist_msg.get("role", "user")
                parts = hist_msg.get("parts", [])
                content = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
                if content:
                    history.append({
                        "role": "user" if role == "user" else "assistant",
                        "content": content,
                    })

            resp = await self._agent.chat(text, history=history or None)

            # Record agent response
            agent_msg = A2AMessage(
                role="agent",
                parts=[{"type": "text", "text": resp.content}],
                messageId=str(uuid.uuid4()),
                taskId=task.id,
            )
            self._tasks.add_message(task.id, agent_msg)

            # Add artifacts if any
            if resp.artifacts:
                for art_data in resp.artifacts:
                    art = Artifact(
                        artifactId=art_data.get("id", str(uuid.uuid4())),
                        parts=[{"type": "data", "data": art_data}],
                        name=art_data.get("type", "artifact"),
                    )
                    self._tasks.add_artifact(task.id, art)

            self._tasks.update_status(task.id, TaskState.COMPLETED)

            return _jsonrpc_success(req_id, {
                "task": task.to_dict(),
                "message": agent_msg.to_dict(),
            })

        except Exception as exc:
            logger.error("A2A message/send failed: %s", exc)
            self._tasks.update_status(task.id, TaskState.FAILED, message=str(exc))
            return _jsonrpc_success(req_id, {"task": task.to_dict()})

    def _handle_tasks_get(self, req_id, params: dict) -> dict:
        task_id = params.get("id", params.get("taskId", ""))
        task = self._tasks.get_task(task_id)
        if not task:
            return _jsonrpc_error(req_id, -32602, f"Task not found: {task_id}")
        return _jsonrpc_success(req_id, {"task": task.to_dict()})

    def _handle_tasks_cancel(self, req_id, params: dict) -> dict:
        task_id = params.get("id", params.get("taskId", ""))
        ok = self._tasks.cancel_task(task_id)
        if not ok:
            return _jsonrpc_error(req_id, -32602, f"Cannot cancel task: {task_id}")
        task = self._tasks.get_task(task_id)
        return _jsonrpc_success(req_id, {"task": task.to_dict() if task else {}})

    # ── SSE streaming ─────────────────────────────────────────────────────

    async def stream_task(self, task_id: str) -> AsyncIterator[str]:
        """Yield SSE events for a task."""
        task = self._tasks.get_task(task_id)
        if not task:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        q = self._tasks.subscribe(task_id)

        # Send current state first
        yield f"data: {json.dumps({'type': 'status', 'task': task.to_dict()})}\n\n"

        # Stream updates
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                # Check if terminal
                task = self._tasks.get_task(task_id)
                if task and task.status.state in (
                    TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED
                ):
                    break
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

    # ── FastAPI mount ─────────────────────────────────────────────────────

    def mount(self, app):
        """Mount A2A endpoints on a FastAPI app.

        Adds:
          - GET  /.well-known/agent.json — Agent Card
          - POST /a2a — JSON-RPC 2.0
          - GET  /a2a/tasks/{task_id}/subscribe — SSE stream
        """
        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse

        @app.get("/.well-known/agent.json")
        async def agent_card():
            return self.build_agent_card()

        @app.post("/a2a")
        async def a2a_endpoint(request: Request):
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    _jsonrpc_error(None, -32700, "Parse error"),
                    status_code=200,
                )
            result = await self.handle_jsonrpc(body)
            return JSONResponse(result)

        @app.get("/a2a/tasks/{task_id}/subscribe")
        async def a2a_subscribe(task_id: str):
            return StreamingResponse(
                self.stream_task(task_id),
                media_type="text/event-stream",
            )

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "protocol": "a2a",
            "protocolVersion": _PROTOCOL_VERSION,
            "agent": self._agent.name if self._agent else None,
            "tasks_total": len(self._tasks._tasks),
            "tasks_active": sum(
                1 for t in self._tasks._tasks.values()
                if t.status.state in (TaskState.SUBMITTED, TaskState.WORKING)
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jsonrpc_success(req_id, result) -> dict:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _jsonrpc_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _get_version() -> str:
    try:
        from adk import __version__
        return __version__
    except Exception:
        return "0.0.0"
