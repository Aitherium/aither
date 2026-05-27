"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    _timer,
)

# Anthropic models that are always available
_ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API client."""

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self._timeout = timeout
        self._base_url = "https://api.anthropic.com"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, headers=self._headers())

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Split system message from conversation messages for Anthropic format."""
        system = ""
        converted = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                converted.append({"role": m.role, "content": m.content})
        return system, converted

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        model = model or self.default_model
        system, conv_messages = self._convert_messages(messages)

        payload: dict = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if top_p is not None:
            payload["top_p"] = top_p
        # Anthropic doesn't have repetition_penalty — skip silently

        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", t)
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools
            # Map tool_choice for Anthropic format
            if tool_choice is not None:
                if tool_choice == "auto":
                    payload["tool_choice"] = {"type": "auto"}
                elif tool_choice == "required":
                    payload["tool_choice"] = {"type": "any"}
                elif tool_choice == "none":
                    pass  # Don't send tools
                elif isinstance(tool_choice, dict):
                    payload["tool_choice"] = tool_choice

        start = _timer()
        async with self._client() as client:
            resp = await client.post(f"{self._base_url}/v1/messages", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = _timer() - start
        usage = data.get("usage", {})

        content_parts = []
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        return LLMResponse(
            content="\n".join(content_parts),
            model=data.get("model", model),
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", "end_turn"),
        )

    async def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.default_model
        system, conv_messages = self._convert_messages(messages)

        payload: dict = {
            "model": model,
            "messages": conv_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self._client() as client:
            async with client.stream(
                "POST", f"{self._base_url}/v1/messages", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    import json
                    data = json.loads(line[6:])
                    event_type = data.get("type", "")
                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        yield StreamChunk(
                            content=delta.get("text", ""),
                            model=model,
                        )
                    elif event_type == "message_stop":
                        yield StreamChunk(done=True, model=model)

    async def list_models(self) -> list[str]:
        return list(_ANTHROPIC_MODELS)
