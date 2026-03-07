"""Ollama LLM provider — direct httpx client for local Ollama instances."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    _timer,
    messages_to_dicts,
)


class OllamaProvider(LLMProvider):
    """Talk to a local or remote Ollama instance."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "llama3.2",
        timeout: float = 120.0,
    ):
        self.host = host.rstrip("/")
        self.default_model = default_model
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.host, timeout=self._timeout)

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        model = model or self.default_model
        payload: dict = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        start = _timer()
        async with self._client() as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = _timer() - start
        msg = data.get("message", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=fn.get("name", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", {}),
            ))

        return LLMResponse(
            content=msg.get("content", ""),
            model=data.get("model", model),
            tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency,
            tool_calls=tool_calls,
        )

    async def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.default_model
        payload: dict = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        async with self._client() as client:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    msg = data.get("message", {})
                    done = data.get("done", False)
                    yield StreamChunk(
                        content=msg.get("content", ""),
                        done=done,
                        model=data.get("model", model),
                    )

    async def list_models(self) -> list[str]:
        async with self._client() as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def pull_model(self, model: str) -> None:
        """Pull a model from the Ollama library."""
        async with self._client() as client:
            resp = await client.post(
                "/api/pull",
                json={"name": model, "stream": False},
                timeout=600.0,
            )
            resp.raise_for_status()
