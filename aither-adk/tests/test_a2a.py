"""Tests for the A2A (Agent-to-Agent) protocol server."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adk.a2a import (
    A2AMessage,
    A2AServer,
    Artifact,
    DataPart,
    Task,
    TaskManager,
    TaskState,
    TaskStatus,
    TextPart,
)


# ── Data Model Tests ────────────────────────────────────────────────────


class TestDataModels:
    def test_task_state_values(self):
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"
        assert TaskState.INPUT_REQUIRED == "input-required"

    def test_text_part(self):
        p = TextPart(text="hello")
        d = p.to_dict()
        assert d == {"type": "text", "text": "hello"}

    def test_data_part(self):
        p = DataPart(data={"key": "value"})
        d = p.to_dict()
        assert d == {"type": "data", "data": {"key": "value"}}

    def test_message_to_dict(self):
        msg = A2AMessage(role="user", parts=[{"type": "text", "text": "hi"}])
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["parts"] == [{"type": "text", "text": "hi"}]
        assert "messageId" not in d  # empty strings are excluded

    def test_message_with_ids(self):
        msg = A2AMessage(role="agent", parts=[], messageId="m1", taskId="t1")
        d = msg.to_dict()
        assert d["messageId"] == "m1"
        assert d["taskId"] == "t1"

    def test_artifact_to_dict(self):
        art = Artifact(artifactId="a1", parts=[{"type": "text", "text": "data"}], name="test")
        d = art.to_dict()
        assert d["artifactId"] == "a1"
        assert d["name"] == "test"
        assert len(d["parts"]) == 1

    def test_artifact_minimal(self):
        art = Artifact(parts=[{"type": "text", "text": "x"}])
        d = art.to_dict()
        assert "artifactId" not in d
        assert "name" not in d

    def test_task_status(self):
        ts = TaskStatus(state=TaskState.WORKING, message="processing")
        d = ts.to_dict()
        assert d["state"] == "working"
        assert d["message"] == "processing"
        assert "timestamp" in d

    def test_task_to_dict(self):
        task = Task(id="t1", contextId="c1")
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["contextId"] == "c1"
        assert d["status"]["state"] == "submitted"
        assert d["history"] == []
        assert d["artifacts"] == []


# ── Task Manager Tests ───────────────────────────────────────────────────


class TestTaskManager:
    def test_create_task(self):
        tm = TaskManager()
        task = tm.create_task(context_id="ctx1")
        assert task.id
        assert task.contextId == "ctx1"
        assert task.status.state == TaskState.SUBMITTED

    def test_create_task_auto_context(self):
        tm = TaskManager()
        task = tm.create_task()
        assert task.contextId  # Auto-generated UUID

    def test_get_task(self):
        tm = TaskManager()
        task = tm.create_task()
        retrieved = tm.get_task(task.id)
        assert retrieved is task

    def test_get_task_nonexistent(self):
        tm = TaskManager()
        assert tm.get_task("nonexistent") is None

    def test_update_status(self):
        tm = TaskManager()
        task = tm.create_task()
        tm.update_status(task.id, TaskState.WORKING, "doing stuff")
        assert task.status.state == TaskState.WORKING
        assert task.status.message == "doing stuff"

    def test_update_status_nonexistent(self):
        tm = TaskManager()
        tm.update_status("nope", TaskState.WORKING)  # Should not raise

    def test_add_message(self):
        tm = TaskManager()
        task = tm.create_task()
        msg = A2AMessage(role="user", parts=[{"type": "text", "text": "hi"}])
        tm.add_message(task.id, msg)
        assert len(task.history) == 1
        assert task.history[0]["role"] == "user"

    def test_add_artifact(self):
        tm = TaskManager()
        task = tm.create_task()
        art = Artifact(artifactId="a1", parts=[{"type": "text", "text": "data"}])
        tm.add_artifact(task.id, art)
        assert len(task.artifacts) == 1
        assert task.artifacts[0]["artifactId"] == "a1"

    def test_cancel_task(self):
        tm = TaskManager()
        task = tm.create_task()
        assert tm.cancel_task(task.id) is True
        assert task.status.state == TaskState.CANCELED

    def test_cancel_completed_task(self):
        tm = TaskManager()
        task = tm.create_task()
        tm.update_status(task.id, TaskState.COMPLETED)
        assert tm.cancel_task(task.id) is False  # Can't cancel completed

    def test_cancel_nonexistent(self):
        tm = TaskManager()
        assert tm.cancel_task("nope") is False

    def test_subscribe_gets_updates(self):
        tm = TaskManager()
        task = tm.create_task()
        q = tm.subscribe(task.id)
        tm.update_status(task.id, TaskState.WORKING)
        assert not q.empty()
        event = q.get_nowait()
        assert event["type"] == "status"

    def test_subscribe_artifact_notification(self):
        tm = TaskManager()
        task = tm.create_task()
        q = tm.subscribe(task.id)
        art = Artifact(artifactId="a1", parts=[])
        tm.add_artifact(task.id, art)
        event = q.get_nowait()
        assert event["type"] == "artifact"


# ── A2A Server Tests ─────────────────────────────────────────────────────


def _make_mock_agent(name="test-agent"):
    agent = MagicMock()
    agent.name = name
    agent._identity = MagicMock()
    agent._identity.to_a2a_card.return_value = {
        "name": name,
        "description": f"Test agent: {name}",
        "url": "http://localhost:8080",
        "skills": [{"id": "chat", "name": "chat", "description": "Chat skill"}],
    }
    agent._tools = MagicMock()
    agent._tools.list_tools.return_value = []
    resp = MagicMock()
    resp.content = "Hello from agent"
    resp.artifacts = []
    agent.chat = AsyncMock(return_value=resp)
    return agent


class TestA2AServer:
    def test_init(self):
        server = A2AServer()
        assert server._base_url == "http://localhost:8080"
        assert server._agent is None

    def test_agent_property(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        assert server.agent is agent
        # Setting clears cached card
        server.build_agent_card()
        assert server._agent_card is not None
        server.agent = _make_mock_agent("other")
        assert server._agent_card is None

    def test_build_agent_card_from_identity(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent, base_url="http://myhost:9000")
        card = server.build_agent_card()
        assert card["name"] == "test-agent"
        assert "protocolVersion" in card
        assert "interfaces" in card
        assert card["capabilities"]["streaming"] is True

    def test_build_agent_card_no_identity(self):
        agent = MagicMock()
        agent.name = "bare-agent"
        del agent._identity  # No identity attribute
        agent._tools = MagicMock()
        agent._tools.list_tools.return_value = []
        server = A2AServer(agent=agent, base_url="http://localhost:8080")
        card = server.build_agent_card()
        assert card["name"] == "bare-agent"
        assert card["protocolVersion"] == "0.3.0"

    def test_build_agent_card_no_agent(self):
        server = A2AServer(server_name="standalone")
        card = server.build_agent_card()
        assert card["name"] == "standalone"

    def test_build_agent_card_with_tools(self):
        agent = _make_mock_agent()
        td1 = MagicMock()
        td1.name = "search_web"
        td1.description = "Search the web"
        td2 = MagicMock()
        td2.name = "file_read"
        td2.description = "Read a file"
        agent._tools.list_tools.return_value = [td1, td2]
        server = A2AServer(agent=agent)
        card = server.build_agent_card()
        skill_ids = {s["id"] for s in card.get("skills", [])}
        assert "search_web" in skill_ids
        assert "file_read" in skill_ids

    def test_card_caching(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        card1 = server.build_agent_card()
        card2 = server.build_agent_card()
        assert card1 is card2  # Same object = cached

    @pytest.mark.asyncio
    async def test_message_send_basic(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}],
                },
            },
            "id": 1,
        })
        assert "result" in resp
        task = resp["result"]["task"]
        assert task["status"]["state"] == "completed"
        assert resp["result"]["message"]["role"] == "agent"
        agent.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_message_send_with_existing_task(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        # First message creates task
        resp1 = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}],
                },
            },
            "id": 1,
        })
        task_id = resp1["result"]["task"]["id"]
        # Second message continues task
        resp2 = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Follow up"}],
                    "taskId": task_id,
                },
            },
            "id": 2,
        })
        assert resp2["result"]["task"]["id"] == task_id

    @pytest.mark.asyncio
    async def test_message_send_no_text(self):
        server = A2AServer(agent=_make_mock_agent())
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {"message": {"role": "user", "parts": []}},
            "id": 1,
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_message_send_bad_task_id(self):
        server = A2AServer(agent=_make_mock_agent())
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "x"}],
                    "taskId": "nonexistent",
                },
            },
            "id": 1,
        })
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_message_send_no_agent(self):
        server = A2AServer()  # No agent
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}],
                },
            },
            "id": 1,
        })
        # Task created but failed
        assert resp["result"]["task"]["status"]["state"] == "failed"

    @pytest.mark.asyncio
    async def test_message_send_agent_error(self):
        agent = _make_mock_agent()
        agent.chat = AsyncMock(side_effect=RuntimeError("boom"))
        server = A2AServer(agent=agent)
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}],
                },
            },
            "id": 1,
        })
        assert resp["result"]["task"]["status"]["state"] == "failed"

    @pytest.mark.asyncio
    async def test_message_send_with_artifacts(self):
        agent = _make_mock_agent()
        resp_mock = MagicMock()
        resp_mock.content = "Done"
        resp_mock.artifacts = [{"id": "file1", "type": "code", "data": "x=1"}]
        agent.chat = AsyncMock(return_value=resp_mock)
        server = A2AServer(agent=agent)
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Generate code"}],
                },
            },
            "id": 1,
        })
        task = resp["result"]["task"]
        assert len(task["artifacts"]) == 1

    @pytest.mark.asyncio
    async def test_tasks_get(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        # Create a task via message/send
        send_resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}],
                },
            },
            "id": 1,
        })
        task_id = send_resp["result"]["task"]["id"]
        # Get it
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": 2,
        })
        assert resp["result"]["task"]["id"] == task_id

    @pytest.mark.asyncio
    async def test_tasks_get_not_found(self):
        server = A2AServer()
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": "nope"},
            "id": 1,
        })
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_tasks_cancel(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        # Create task
        task = server._tasks.create_task()
        server._tasks.update_status(task.id, TaskState.WORKING)
        # Cancel it
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": task.id},
            "id": 1,
        })
        assert resp["result"]["task"]["status"]["state"] == "canceled"

    @pytest.mark.asyncio
    async def test_tasks_cancel_not_found(self):
        server = A2AServer()
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": "nope"},
            "id": 1,
        })
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        server = A2AServer()
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "nonexistent",
            "params": {},
            "id": 1,
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_status(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        status = server.status()
        assert status["protocol"] == "a2a"
        assert status["protocolVersion"] == "0.3.0"
        assert status["agent"] == "test-agent"
        assert status["tasks_total"] == 0

    @pytest.mark.asyncio
    async def test_status_after_task(self):
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        server._tasks.create_task()
        server._tasks.create_task()
        status = server.status()
        assert status["tasks_total"] == 2
        assert status["tasks_active"] == 2  # Both submitted

    def test_mount_registers_routes(self):
        app = MagicMock()
        app.get = MagicMock(return_value=lambda f: f)
        app.post = MagicMock(return_value=lambda f: f)
        server = A2AServer()
        server.mount(app)
        # Should register: GET /.well-known/agent.json, POST /a2a, GET /a2a/tasks/{task_id}/subscribe
        get_calls = [c[0][0] for c in app.get.call_args_list]
        post_calls = [c[0][0] for c in app.post.call_args_list]
        assert "/.well-known/agent.json" in get_calls
        assert "/a2a" in post_calls

    @pytest.mark.asyncio
    async def test_stream_task_not_found(self):
        server = A2AServer()
        chunks = []
        async for chunk in server.stream_task("nonexistent"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "error" in chunks[0]

    @pytest.mark.asyncio
    async def test_message_send_string_part(self):
        """Parts can be plain strings (not just dicts)."""
        agent = _make_mock_agent()
        server = A2AServer(agent=agent)
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": ["Hello plain string"],
                },
            },
            "id": 1,
        })
        assert "result" in resp
        assert resp["result"]["task"]["status"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_tasks_get_by_taskId_key(self):
        """tasks/get accepts both 'id' and 'taskId' params."""
        server = A2AServer()
        task = server._tasks.create_task()
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"taskId": task.id},
            "id": 1,
        })
        assert resp["result"]["task"]["id"] == task.id


# ── Agent Card Interfaces ────────────────────────────────────────────────


class TestAgentCardInterfaces:
    def test_interfaces_include_a2a_and_mcp(self):
        server = A2AServer(base_url="http://myhost:9000", server_name="node1")
        card = server.build_agent_card()
        urls = [i["url"] for i in card["interfaces"]]
        assert "http://myhost:9000/a2a" in urls
        assert "http://myhost:9000/mcp" in urls

    def test_card_has_required_fields(self):
        server = A2AServer(server_name="node1")
        card = server.build_agent_card()
        assert "name" in card
        assert "protocolVersion" in card
        assert "capabilities" in card
        assert "interfaces" in card

    def test_card_default_modes(self):
        server = A2AServer(server_name="node1")
        card = server.build_agent_card()
        assert "text/plain" in card.get("defaultInputModes", [])
        assert "text/plain" in card.get("defaultOutputModes", [])


# ── Custom Agent A2A Compatibility ───────────────────────────────────────


class TestCustomAgentCompat:
    @pytest.mark.asyncio
    async def test_custom_agent_works_with_a2a(self):
        """Any agent implementing chat(msg, history=...) works with A2A."""
        class MyAgent:
            name = "custom-agent"
            async def chat(self, message, history=None):
                resp = MagicMock()
                resp.content = f"Echo: {message}"
                resp.artifacts = []
                return resp

        agent = MyAgent()
        server = A2AServer(agent=agent, server_name="custom")
        resp = await server.handle_jsonrpc({
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Test"}],
                },
            },
            "id": 1,
        })
        assert resp["result"]["task"]["status"]["state"] == "completed"
        assert "Echo: Test" in resp["result"]["message"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_custom_agent_card_without_identity(self):
        """Agent without _identity still gets a valid card."""
        class MinimalAgent:
            name = "minimal"

        agent = MinimalAgent()
        server = A2AServer(agent=agent)
        card = server.build_agent_card()
        assert card["name"] == "minimal"
        assert card["protocolVersion"] == "0.3.0"
