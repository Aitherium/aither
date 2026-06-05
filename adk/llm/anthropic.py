"""Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger("adk.llm.anthropic")

_MAX_RETRIES = 3
# Transient statuses worth retrying: rate limit (429), Anthropic "overloaded"
# (529), and gateway/server blips (500/502/503/504, 408). Anthropic 502s are
# common transient gateway hiccups — a single one shouldn't kill a run.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


def _ensure_ok(resp: httpx.Response) -> None:
    """Raise on 4xx/5xx but INCLUDE the provider's body (e.g. the overloaded /
    invalid-request message) — ``raise_for_status()`` discards it."""
    if resp.status_code < 400:
        return
    body = ""
    try:
        body = (resp.text or "").strip()
    except Exception:  # noqa: BLE001
        body = ""
    msg = f"{resp.status_code} {resp.reason_phrase} for {resp.request.url}"
    if body:
        msg += f"\nProvider response: {body[:1200]}"
    raise httpx.HTTPStatusError(msg, request=resp.request, response=resp)


async def _post_with_retry(
    client: httpx.AsyncClient, url: str, payload: dict
) -> httpx.Response:
    """POST with retry on transient errors (429 + 5xx). Honors Retry-After else
    exponential backoff; surfaces the provider body on the final failure."""
    resp = None
    for attempt in range(1, _MAX_RETRIES + 1):
        resp = await client.post(url, json=payload)
        if resp.status_code not in _RETRYABLE_STATUS:
            _ensure_ok(resp)
            return resp
        if attempt == _MAX_RETRIES:
            break
        retry_after = resp.headers.get("retry-after", "")
        try:
            wait = int(retry_after) if retry_after else min(2 ** attempt, 30)
        except (ValueError, TypeError):
            wait = min(2 ** attempt, 30)
        wait = min(wait, 120)
        logger.warning(
            "Transient %s from %s — retrying in %ds (%d/%d)...",
            resp.status_code, url, wait, attempt, _MAX_RETRIES,
        )
        await asyncio.sleep(wait)
    _ensure_ok(resp)
    return resp

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
            resp = await _post_with_retry(client, f"{self._base_url}/v1/messages", payload)
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
            for attempt in range(1, _MAX_RETRIES + 1):
                async with client.stream(
                    "POST", f"{self._base_url}/v1/messages", json=payload
                ) as resp:
                    if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                        await resp.aread()  # body not loaded in stream mode until read
                        wait = min(2 ** attempt, 30)
                        logger.warning(
                            "Transient %s on Anthropic stream — retrying in %ds (%d/%d)...",
                            resp.status_code, wait, attempt, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code >= 400:
                        await resp.aread()
                        _ensure_ok(resp)
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
                    return

    async def list_models(self) -> list[str]:
        return list(_ANTHROPIC_MODELS)
