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
