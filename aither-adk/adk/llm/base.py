"""Abstract LLM provider and shared data types."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import AsyncIterator

logger = logging.getLogger("adk.llm.base")


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
    effort_level: int = 0
    cache_status: str = ""


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""
    content: str = ""
    done: bool = False
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Internal tag stripping (ported from monorepo UnifiedChatBackend)
# ─────────────────────────────────────────────────────────────────────────────

# Matches <tool_call>...</tool_call> and unclosed (truncated) <tool_call>...
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_call>.*",
    re.DOTALL,
)
# Matches leaked system prompt markers
_INTERNAL_TAG_RE = re.compile(
    r"\[(?:SYSTEM|AXIOMS|RULES|IDENTITY|CAPABILITIES|CONTEXT|MEMORIES|AFFECT|RESPONSE FORMAT)\]"
)


def strip_internal_tags(text: str) -> str:
    """Strip leaked <tool_call> XML and internal prompt markers from LLM output."""
    if not text:
        return text
    cleaned = _TOOL_CALL_TAG_RE.sub("", text)
    cleaned = _INTERNAL_TAG_RE.sub("", cleaned)
    # Collapse runs of blank lines left by removal
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Degeneration detection (ported from monorepo AitherLLMQueue)
# ─────────────────────────────────────────────────────────────────────────────

class DegenerationDetector:
    """Monitors streaming tokens for repetition loops and low diversity.

    Detects degenerate output (e.g., "I want I want I want...") by tracking
    n-gram frequencies and word diversity in a sliding window.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        window_size: int = 50,
        repetition_threshold: float = 0.4,
        diversity_threshold: float = 0.15,
    ):
        self._ngram_size = ngram_size
        self._window_size = window_size
        self._repetition_threshold = repetition_threshold
        self._diversity_threshold = diversity_threshold
        self._tokens: list[str] = []
        self._degenerate = False

    @property
    def degenerate(self) -> bool:
        return self._degenerate

    def feed(self, text: str) -> bool:
        """Feed new text. Returns True if degeneration detected."""
        if self._degenerate:
            return True
        words = text.split()
        if not words:
            return False
        self._tokens.extend(words)
        # Only check when we have enough tokens
        if len(self._tokens) < self._window_size:
            return False
        window = self._tokens[-self._window_size:]
        # N-gram repetition check
        ngrams: dict[tuple, int] = {}
        for i in range(len(window) - self._ngram_size + 1):
            ng = tuple(w.lower() for w in window[i:i + self._ngram_size])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        total_ngrams = len(window) - self._ngram_size + 1
        if total_ngrams > 0:
            max_freq = max(ngrams.values())
            if max_freq / total_ngrams > self._repetition_threshold:
                self._degenerate = True
                return True
        # Word diversity check
        unique = len(set(w.lower() for w in window))
        if unique / len(window) < self._diversity_threshold:
            self._degenerate = True
            return True
        return False

    def trim_clean(self, full_text: str) -> str:
        """Trim degenerate tail from accumulated text.

        Walks backward to find the last non-repeating sentence boundary.
        """
        if not full_text:
            return full_text
        # Find sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        if len(sentences) <= 2:
            return full_text
        # Walk backward, drop sentences that look like repeats
        seen: set[str] = set()
        keep = []
        for s in sentences:
            normalized = s.lower().strip()
            if normalized in seen:
                break
            seen.add(normalized)
            keep.append(s)
        return " ".join(keep) if keep else sentences[0]


# ─────────────────────────────────────────────────────────────────────────────
# Retry with exponential backoff (connection/timeout errors only)
# ─────────────────────────────────────────────────────────────────────────────

# Exception types that warrant a retry (connection and timeout errors)
_RETRYABLE_EXCEPTIONS = (
    ConnectionError, TimeoutError, OSError,
)

# Also catch httpx-specific errors if httpx is available
try:
    import httpx
    _RETRYABLE_EXCEPTIONS = (
        ConnectionError, TimeoutError, OSError,
        httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
        httpx.PoolTimeout, httpx.RemoteProtocolError,
    )
except ImportError:
    pass


def llm_retry(
    max_retries: int = 5,
    base_delay_ms: int = 500,
    max_delay_ms: int = 16000,
):
    """Decorator that adds retry with exponential backoff to async LLM calls.

    Only retries on connection/timeout errors. API errors (auth, rate limit,
    bad request) are NOT retried — those indicate a real problem.

    Args:
        max_retries: Maximum number of retry attempts (default: 5).
        base_delay_ms: Base delay in milliseconds (default: 500).
        max_delay_ms: Maximum delay cap in milliseconds (default: 16000).
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except _RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        logger.warning(
                            "LLM call failed after %d retries: %s",
                            max_retries, exc,
                        )
                        raise
                    # Exponential backoff with jitter
                    delay_ms = min(base_delay_ms * (2 ** attempt), max_delay_ms)
                    jitter_ms = random.uniform(0, delay_ms * 0.25)
                    total_delay = (delay_ms + jitter_ms) / 1000.0
                    logger.debug(
                        "LLM call attempt %d/%d failed (%s), retrying in %.1fs",
                        attempt + 1, max_retries, type(exc).__name__, total_delay,
                    )
                    await asyncio.sleep(total_delay)
            raise last_exc  # Should not reach here, but satisfy type checker
        return wrapper
    return decorator


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
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
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
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response. Default implementation wraps chat()."""
        resp = await self.chat(
            messages, model, temperature, max_tokens, tools,
            tool_choice=tool_choice, top_p=top_p,
            repetition_penalty=repetition_penalty, **kwargs,
        )
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
