"""LLM provider layer — auto-detecting router across backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import LLMProvider, LLMResponse, Message, StreamChunk, ToolCall

if TYPE_CHECKING:
    from adk.config import Config

logger = logging.getLogger("adk.llm")

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "Message",
    "StreamChunk",
    "ToolCall",
]

# Effort-based model selection defaults
_EFFORT_MODELS = {
    "gateway": {
        "small": "aither-small",
        "medium": "aither-orchestrator",
        "large": "aither-reasoning",
    },
    "ollama": {
        "small": "llama3.2:3b",
        "medium": "llama3.2",
        "large": "deepseek-r1:14b",
    },
    "openai": {
        "small": "gpt-4o-mini",
        "medium": "gpt-4o",
        "large": "o1",
    },
    "anthropic": {
        "small": "claude-haiku-4-5-20251001",
        "medium": "claude-sonnet-4-6",
        "large": "claude-opus-4-6",
    },
}

# Default gateway URL for cloud inference
_GATEWAY_INFERENCE_URL = "https://gateway.aitherium.com/v1"
_DEMO_URL = "https://demo.aitherium.com"


class LLMRouter:
    """Multi-backend LLM router with auto-detection and effort-based model selection.

    Usage:
        # Auto-detect (tries Ollama localhost first)
        router = LLMRouter()

        # Explicit backend
        router = LLMRouter(provider="openai", api_key="sk-...")

        # Explicit with custom URL (vLLM, LM Studio, etc.)
        router = LLMRouter(provider="openai", base_url="http://localhost:8000/v1")
    """

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        config: Config | None = None,
    ):
        self._provider_name: str = ""
        self._provider: LLMProvider | None = None
        self._model = model
        self._config = config

        if provider:
            self._provider = self._create_provider(provider, base_url, api_key)
            self._provider_name = provider
        else:
            self._deferred_base_url = base_url
            self._deferred_api_key = api_key

    def _create_provider(
        self, name: str, base_url: str | None = None, api_key: str | None = None
    ) -> LLMProvider:
        if name == "gateway":
            from .openai_compat import OpenAIProvider
            gateway_url = base_url or _GATEWAY_INFERENCE_URL
            return OpenAIProvider(
                base_url=gateway_url,
                api_key=api_key or "",
                default_model=self._model or "aither-orchestrator",
            )
        elif name == "ollama":
            from .ollama import OllamaProvider
            return OllamaProvider(
                host=base_url or "http://localhost:11434",
                default_model=self._model or "llama3.2",
            )
        elif name in ("openai", "vllm", "lmstudio", "llamacpp", "groq", "together"):
            from .openai_compat import OpenAIProvider
            return OpenAIProvider(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key or "",
                default_model=self._model or "gpt-4o-mini",
            )
        elif name == "anthropic":
            from .anthropic import AnthropicProvider
            return AnthropicProvider(
                api_key=api_key or "",
                default_model=self._model or "claude-sonnet-4-6",
            )
        else:
            raise ValueError(f"Unknown provider: {name}. Use 'gateway', 'ollama', 'openai', or 'anthropic'.")

    async def _try_ollama(self) -> LLMProvider | None:
        """Try Ollama on localhost. Returns provider or None."""
        try:
            from .ollama import OllamaProvider
            host = (self._config.ollama_host if self._config else None) or "http://localhost:11434"
            p = OllamaProvider(host=host, default_model=self._model or "llama3.2")
            if await p.health_check():
                self._provider_name = "ollama"
                logger.info("Auto-detected Ollama at %s", host)
                return p
        except Exception:
            pass
        return None

    async def _auto_detect(self) -> LLMProvider:
        """Try backends in priority order. Default: gateway → Ollama → cloud APIs → demo.

        Set AITHER_PREFER_LOCAL=true to try Ollama first, then gateway as fallback.
        """
        import os

        prefer_local = (
            (self._config.prefer_local if self._config else False)
            or os.getenv("AITHER_PREFER_LOCAL", "").lower() in ("true", "1", "yes")
        )

        gateway_key = (
            (self._config.aither_api_key if self._config else "")
            or os.getenv("AITHER_API_KEY", "")
        )

        # When prefer_local is set, try Ollama BEFORE gateway
        if prefer_local:
            ollama = await self._try_ollama()
            if ollama:
                return ollama

        # Try gateway.aitherium.com (default for most users — no local GPU needed)
        if gateway_key:
            gateway_url = (
                (self._config.gateway_url if self._config else "")
                or os.getenv("AITHER_GATEWAY_URL", _GATEWAY_INFERENCE_URL)
            )
            # Ensure we hit the /v1 path for OpenAI-compat
            if not gateway_url.endswith("/v1"):
                gateway_url = gateway_url.rstrip("/") + "/v1"
            try:
                from .openai_compat import OpenAIProvider
                p = OpenAIProvider(
                    base_url=gateway_url,
                    api_key=gateway_key,
                    default_model=self._model or "aither-orchestrator",
                )
                if await p.health_check():
                    self._provider_name = "gateway"
                    logger.info("Connected to AitherOS gateway at %s", gateway_url)
                    return p
            except Exception:
                logger.debug("Gateway not reachable, trying local backends")

        # Try Ollama on localhost (skip if already tried via prefer_local)
        if not prefer_local:
            ollama = await self._try_ollama()
            if ollama:
                return ollama

        # Try config-based cloud API keys
        if self._config:
            if self._config.anthropic_api_key:
                self._provider_name = "anthropic"
                return self._create_provider(
                    "anthropic", api_key=self._config.anthropic_api_key
                )
            if self._config.openai_api_key:
                self._provider_name = "openai"
                return self._create_provider(
                    "openai",
                    base_url=self._config.openai_base_url,
                    api_key=self._config.openai_api_key,
                )

        # 4. Check env vars
        if os.getenv("ANTHROPIC_API_KEY"):
            self._provider_name = "anthropic"
            return self._create_provider("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("OPENAI_API_KEY"):
            self._provider_name = "openai"
            return self._create_provider(
                "openai",
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        # 5. No backend available — point user to the demo
        raise ConnectionError(
            "No LLM backend available.\n\n"
            f"  Try the demo:     {_DEMO_URL}\n"
            "  Get an API key:   https://gateway.aitherium.com\n"
            "  Or install local: https://ollama.com\n\n"
            "Set AITHER_API_KEY to connect to the AitherOS gateway for inference."
        )

    async def get_provider(self) -> LLMProvider:
        """Return the active provider, auto-detecting if needed."""
        if self._provider is None:
            if hasattr(self, "_deferred_base_url"):
                self._provider = await self._auto_detect()
            else:
                self._provider = await self._auto_detect()
        return self._provider

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def model_for_effort(self, effort: int) -> str:
        """Select model based on effort level (1-10)."""
        if self._model:
            return self._model
        tier = "small" if effort <= 3 else "medium" if effort <= 6 else "large"
        models = _EFFORT_MODELS.get(self._provider_name, {})
        return models.get(tier, self._model or "")

    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        effort: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Route a chat request to the active provider."""
        provider = await self.get_provider()
        if model is None and effort is not None:
            model = self.model_for_effort(effort)
        return await provider.chat(messages, model=model, **kwargs)

    async def chat_stream(self, messages: list[Message], model: str | None = None, **kwargs):
        """Stream a chat response."""
        provider = await self.get_provider()
        async for chunk in provider.chat_stream(messages, model=model, **kwargs):
            yield chunk

    async def list_models(self) -> list[str]:
        provider = await self.get_provider()
        return await provider.list_models()
