"""OpenAI-compatible LLM provider — works with OpenAI, vLLM, LM Studio, llama.cpp, Groq, Together."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger("adk.llm.openai")

_MAX_RETRIES = 3

from .base import (
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    _timer,
    messages_to_dicts,
)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API client. Works with any endpoint that speaks the OpenAI format."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        default_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, headers=self._headers())

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
        payload: dict = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None and tools:
            payload["tool_choice"] = tool_choice
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            # OpenAI uses frequency_penalty; vLLM accepts repetition_penalty
            payload["frequency_penalty"] = repetition_penalty - 1.0  # normalize: 1.3 -> 0.3

        start = _timer()
        async with self._client() as client:
            resp = await self._post_with_retry(
                client, f"{self.base_url}/chat/completions", payload
            )
            data = resp.json()

        latency = _timer() - start
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        content = msg.get("content", "") or ""

        # Hermes XML fallback: if model emitted <tool_call> tags in content
        # but no structured tool_calls, parse them from text
        if not tool_calls and content and "<tool_call>" in content:
            from .base import extract_tool_calls_from_text
            fallback_calls, content = extract_tool_calls_from_text(content)
            if fallback_calls:
                tool_calls = fallback_calls

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
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
        payload: dict = {
            "model": model,
            "messages": messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None and tools:
            payload["tool_choice"] = tool_choice
        if top_p is not None:
            payload["top_p"] = top_p
        if repetition_penalty is not None:
            payload["frequency_penalty"] = repetition_penalty - 1.0

        async with self._client() as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        yield StreamChunk(done=True, model=model)
                        return
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    yield StreamChunk(
                        content=delta.get("content", "") or "",
                        done=choice.get("finish_reason") is not None,
                        model=data.get("model", model),
                    )

    async def _post_with_retry(
        self, client: httpx.AsyncClient, url: str, payload: dict
    ) -> httpx.Response:
        """POST with retry on 429 (rate limit). Reads Retry-After header."""
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = await client.post(url, json=payload)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            # 429 — read retry-after and wait
            retry_after = resp.headers.get("retry-after", "")
            try:
                wait = int(retry_after) if retry_after else 5
            except (ValueError, TypeError):
                wait = 5
            wait = min(wait, 120)  # cap at 2 minutes
            logger.warning(
                "Rate limited (429). Waiting %ds before retry (%d/%d)...",
                wait, attempt, _MAX_RETRIES,
            )
            await asyncio.sleep(wait)
        # Final attempt — let it raise if still 429
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp

    async def list_models(self) -> list[str]:
        async with self._client() as client:
            try:
                resp = await client.get(f"{self.base_url}/models")
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPStatusError, json.JSONDecodeError):
                return []
        return [m["id"] for m in data.get("data", [])]

    async def health_check(self) -> bool:
        """Fast connectivity check — uses a 5s timeout instead of the default 120s."""
        try:
            async with httpx.AsyncClient(
                timeout=5.0, headers=self._headers()
            ) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False
