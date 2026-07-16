"""Tests for the FastAPI server."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from adk.server import create_app
from adk.agent import AitherAgent
from adk.llm.base import LLMResponse, Message
from adk.config import Config


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=AitherAgent)
    agent.name = "test-agent"
    agent.llm = MagicMock()
    agent.llm.provider_name = "mock"
    agent.llm.chat = AsyncMock(return_value=LLMResponse(
        content="Hello!", model="mock-model", tokens_used=10,
        prompt_tokens=5, completion_tokens=5, finish_reason="stop",
    ))
    agent.llm.list_models = AsyncMock(return_value=["model-a", "model-b"])
    agent.chat = AsyncMock(return_value=MagicMock(
        content="Agent response",
        model="mock-model",
        tokens_used=10,
        session_id="s1",
        tool_calls_made=[],
    ))
    return agent


@pytest.fixture
def client(mock_agent):
    app = create_app(agent=mock_agent)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent"] == "test-agent"


class TestChatEndpoint:
    def test_chat(self, client, mock_agent):
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Agent response"
        assert data["agent"] == "test-agent"

    def test_chat_with_session(self, client, mock_agent):
        resp = client.post("/chat", json={"message": "Hello", "session_id": "my-session"})
        assert resp.status_code == 200
        mock_agent.chat.assert_called_once()
        call_kwargs = mock_agent.chat.call_args
        assert call_kwargs.kwargs.get("session_id") == "my-session"


class TestOpenAICompatEndpoints:
    def test_chat_completions(self, client, mock_agent):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 10

    def test_list_models(self, client, mock_agent):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "model-a"


class TestIdentitiesEndpoint:
    def test_list_identities(self, client):
        resp = client.get("/v1/identities")
        assert resp.status_code == 200
        data = resp.json()
        assert "identities" in data
        assert "aither" in data["identities"]


class TestCORS:
    def test_cors_headers(self, client):
        resp = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        # CORS middleware should not block
        assert resp.status_code in (200, 405)


# ── Central-console admin proxy allowlist (fail-closed) ──────────────────

from adk.server import _mesh_admin_allowed  # noqa: E402


class TestMeshAdminAllowlist:
    """The mesh console proxy must be fail-closed: cli/exec + credential writes
    are never reachable; only observe routes + narrow safe controls pass."""

    def test_denies_remote_code_exec(self):
        assert _mesh_admin_allowed("POST", "cli/exec") is False

    def test_denies_credential_writes(self):
        assert _mesh_admin_allowed("POST", "llm/keys") is False
        assert _mesh_admin_allowed("POST", "llm/switch") is False

    def test_denies_config_write_and_unknown(self):
        assert _mesh_admin_allowed("POST", "config") is False
        assert _mesh_admin_allowed("POST", "packs/x/tools/y/invoke") is False
        assert _mesh_admin_allowed("GET", "cli/commands") is False

    def test_denies_path_traversal(self):
        assert _mesh_admin_allowed("GET", "../secrets") is False
        assert _mesh_admin_allowed("GET", "") is False

    def test_allows_observe_routes(self):
        for p in ("config", "meta", "sessions", "sessions/abc", "packs",
                  "packs/mypack", "logs/tail", "mcp/servers", "llm/status"):
            assert _mesh_admin_allowed("GET", p) is True, p

    def test_allows_narrow_safe_controls(self):
        assert _mesh_admin_allowed("POST", "packs/enable") is True
        assert _mesh_admin_allowed("POST", "packs/reload") is True
        assert _mesh_admin_allowed("DELETE", "sessions/abc") is True

    def test_delete_limited_to_sessions(self):
        assert _mesh_admin_allowed("DELETE", "packs/x") is False
        assert _mesh_admin_allowed("DELETE", "config") is False


class TestMeshAdminProxyHttpGate:
    """The console admin proxy must deny dangerous routes at the REAL HTTP route
    (before any forwarding) and stay bearer-gated."""

    def _app(self, api_key="consolekey"):
        import os
        with patch.dict(os.environ, {"AITHER_SERVER_API_KEY": api_key}, clear=False):
            cfg = Config()
            cfg.gateway_url = ""
            cfg.aither_api_key = ""
            agent = MagicMock()
            agent.name = "local"
            agent.llm = MagicMock()
            agent.llm.provider_name = "mock"
            agent._identity = MagicMock()
            agent._identity.name = "local"
            agent._identity.description = ""
            agent._identity.skills = []
            agent._tools = MagicMock()
            agent._tools.list_tools = MagicMock(return_value=[])
            agent._safety = None
            return create_app(agent=agent, identity="local", config=cfg)

    def test_cli_exec_denied_before_forward(self):
        c = TestClient(self._app())
        r = c.post("/mesh/agents/anyagent/admin/cli/exec",
                   headers={"Authorization": "Bearer consolekey"}, json={"cmd": "whoami"})
        assert r.status_code == 403
        assert "not permitted" in r.text

    def test_credential_write_denied_before_forward(self):
        c = TestClient(self._app())
        r = c.post("/mesh/agents/anyagent/admin/llm/keys",
                   headers={"Authorization": "Bearer consolekey"}, json={"key": "x"})
        assert r.status_code == 403

    def test_admin_proxy_requires_bearer(self):
        c = TestClient(self._app())
        r = c.get("/mesh/agents/anyagent/admin/config")  # no Authorization header
        assert r.status_code == 401
