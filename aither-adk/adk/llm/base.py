"""Abstract LLM provider and shared data types."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list | None = None


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str = ""
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str = ""
    done: bool = False
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send messages and get a response."""
        ...

    async def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response. Default implementation wraps chat()."""
        resp = await self.chat(messages, model, temperature, max_tokens, tools, **kwargs)
        yield StreamChunk(content=resp.content, done=True, model=resp.model)

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models from this provider."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable."""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False


def _timer() -> float:
    """Return current time in ms for latency tracking."""
    return time.monotonic() * 1000


def messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert Message objects to plain dicts for API calls."""
    result = []
    for m in messages:
        d: dict = {"role": m.role, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        result.append(d)
    return result
