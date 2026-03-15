"""Tests for LLM provider layer."""

import json
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, MagicMock, patch

from adk.llm.base import Message, LLMResponse, ToolCall, messages_to_dicts, LLMProvider
from adk.llm.ollama import OllamaProvider
from adk.llm.openai_compat import OpenAIProvider
from adk.llm.anthropic import AnthropicProvider
from adk.llm import LLMRouter


# ─── Base types ───

class TestMessage:
    def test_create(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_with_tool_call_id(self):
        m = Message(role="tool", content="result", tool_call_id="tc_1")
        assert m.tool_call_id == "tc_1"

    def test_messages_to_dicts(self):
        msgs = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        dicts = messages_to_dicts(msgs)
        assert len(dicts) == 2
        assert dicts[0] == {"role": "system", "content": "sys"}

    def test_messages_to_dicts_with_name(self):
        msgs = [Message(role="user", content="hi", name="bob")]
        dicts = messages_to_dicts(msgs)
        assert dicts[0]["name"] == "bob"


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.tokens_used == 0
        assert r.tool_calls == []

    def test_with_tool_calls(self):
        tc = ToolCall(id="1", name="search", arguments={"q": "test"})
        r = LLMResponse(content="", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"


# ─── Ollama ───

class TestOllamaProvider:
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat(self):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={
                "message": {"role": "assistant", "content": "Hello there!"},
                "model": "llama3.2",
                "eval_count": 10,
                "prompt_eval_count": 5,
            })
        )
        provider = OllamaProvider()
        resp = await provider.chat([Message(role="user", content="hi")])
        assert resp.content == "Hello there!"
        assert resp.model == "llama3.2"
        assert resp.tokens_used == 15

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2"}, {"name": "deepseek-r1:14b"}]
            })
        )
        provider = OllamaProvider()
        models = await provider.list_models()
        assert "llama3.2" in models
        assert len(models) == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "search", "arguments": {"q": "test"}}}],
                },
                "model": "llama3.2",
            })
        )
        provider = OllamaProvider()
        resp = await provider.chat(
            [Message(role="user", content="search for test")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3.2"}]})
        )
        provider = OllamaProvider()
        assert await provider.health_check() is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        provider = OllamaProvider()
        assert await provider.health_check() is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_non_200(self):
        """health_check returns False on non-200 status (e.g. 500)."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(500, json={"error": "internal"})
        )
        provider = OllamaProvider()
        assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_uses_fast_timeout(self):
        """health_check uses a 5s timeout, not the default 120s."""
        provider = OllamaProvider(timeout=300.0)  # Deliberately large default
        with patch("adk.llm.ollama.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            await provider.health_check()
            # Verify the client was created with 5s, not 300s
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args
            assert call_kwargs.kwargs.get("timeout") == 5.0 or call_kwargs[1].get("timeout") == 5.0

    def test_custom_host(self):
        provider = OllamaProvider(host="http://myhost:11434")
        assert provider.host == "http://myhost:11434"


# ─── OpenAI-compatible ───

class TestOpenAIProvider:
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })
        )
        provider = OpenAIProvider(api_key="sk-test")
        resp = await provider.chat([Message(role="user", content="hi")])
        assert resp.content == "Hi!"
        assert resp.tokens_used == 7

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
        )
        provider = OpenAIProvider(api_key="sk-test")
        resp = await provider.chat([Message(role="user", content="calc 2+2")])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].arguments == {"expr": "2+2"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_models(self):
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={
                "data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]
            })
        )
        provider = OpenAIProvider(api_key="sk-test")
        models = await provider.list_models()
        assert "gpt-4o" in models

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})
        )
        provider = OpenAIProvider(api_key="sk-test")
        assert await provider.health_check() is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        respx.get("https://api.openai.com/v1/models").mock(
            side_effect=httpx.ConnectError("refused")
        )
        provider = OpenAIProvider(api_key="sk-test")
        assert await provider.health_check() is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_non_200(self):
        """health_check returns False on 401/500/etc."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        provider = OpenAIProvider(api_key="sk-bad")
        assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_uses_fast_timeout(self):
        """health_check uses a 5s timeout, not the default 120s."""
        provider = OpenAIProvider(api_key="sk-test", timeout=300.0)
        with patch("adk.llm.openai_compat.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                return_value=MagicMock(status_code=200)
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            await provider.health_check()
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args
            assert call_kwargs.kwargs.get("timeout") == 5.0 or call_kwargs[1].get("timeout") == 5.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_health_check_vllm_local(self):
        """health_check works for vLLM on localhost (OpenAI-compat format)."""
        respx.get("http://localhost:8200/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "meta-llama/Llama-3.2-8B"}]})
        )
        provider = OpenAIProvider(base_url="http://localhost:8200/v1", api_key="not-needed")
        assert await provider.health_check() is True

    def test_custom_base_url(self):
        provider = OpenAIProvider(base_url="http://localhost:8000/v1")
        assert provider.base_url == "http://localhost:8000/v1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_null_content_handled(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}],
                "model": "gpt-4o-mini",
                "usage": {"total_tokens": 0},
            })
        )
        provider = OpenAIProvider(api_key="sk-test")
        resp = await provider.chat([Message(role="user", content="hi")])
        assert resp.content == ""


# ─── Anthropic ───

class TestAnthropicProvider:
    @respx.mock
    @pytest.mark.asyncio
    async def test_chat(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 5, "output_tokens": 3},
                "stop_reason": "end_turn",
            })
        )
        provider = AnthropicProvider(api_key="sk-ant-test")
        resp = await provider.chat([Message(role="user", content="hi")])
        assert resp.content == "Hello!"
        assert resp.tokens_used == 8

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_message_extraction(self):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "OK"}],
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
        )
        provider = AnthropicProvider(api_key="sk-ant-test")
        await provider.chat([
            Message(role="system", content="You are helpful"),
            Message(role="user", content="hi"),
        ])
        body = json.loads(route.calls[0].request.content)
        assert body["system"] == "You are helpful"
        assert len(body["messages"]) == 1  # system extracted

    @respx.mock
    @pytest.mark.asyncio
    async def test_tool_use_response(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}},
                ],
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            })
        )
        provider = AnthropicProvider(api_key="sk-ant-test")
        resp = await provider.chat([Message(role="user", content="search test")])
        assert "Let me search" in resp.content
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"

    async def test_list_models(self):
        provider = AnthropicProvider()
        models = await provider.list_models()
        assert "claude-sonnet-4-6" in models


# ─── LLMRouter ───

class TestLLMRouter:
    def test_explicit_provider(self):
        router = LLMRouter(provider="ollama")
        assert router.provider_name == "ollama"

    def test_model_for_effort(self):
        router = LLMRouter(provider="openai", api_key="sk-test")
        assert router.model_for_effort(1) == "gpt-4o-mini"    # small
        assert router.model_for_effort(5) == "gpt-4o"          # medium
        assert router.model_for_effort(9) == "o1"              # large

    def test_model_override(self):
        router = LLMRouter(provider="ollama", model="custom-model")
        assert router.model_for_effort(5) == "custom-model"

    @respx.mock
    @pytest.mark.asyncio
    async def test_chat_delegates(self):
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={
                "message": {"content": "hi"},
                "model": "llama3.2",
            })
        )
        router = LLMRouter(provider="ollama")
        resp = await router.chat([Message(role="user", content="hi")])
        assert resp.content == "hi"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMRouter(provider="nonexistent")

    @respx.mock
    @pytest.mark.asyncio
    async def test_auto_detect_ollama(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3.2"}]})
        )
        router = LLMRouter()
        provider = await router.get_provider()
        assert router.provider_name == "ollama"

    @respx.mock
    @pytest.mark.asyncio
    async def test_auto_detect_falls_back_to_env(self, monkeypatch):
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        router = LLMRouter()
        provider = await router.get_provider()
        assert router.provider_name == "openai"
