"""LLM provider layer — auto-detecting router across backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import (
    DegenerationDetector,
    LLMProvider,
    LLMResponse,
    Message,
    StreamChunk,
    ToolCall,
    llm_retry,
    strip_internal_tags,
)

if TYPE_CHECKING:
    from adk.config import Config

logger = logging.getLogger("adk.llm")

__all__ = [
    "DegenerationDetector",
    "LLMProvider",
    "LLMResponse",
    "LLMRouter",
    "Message",
    "StreamChunk",
    "ToolCall",
    "llm_retry",
    "strip_internal_tags",
]

# Effort-based model selection defaults (static fallback when llmfit unavailable)
_EFFORT_MODELS = {
    "gateway": {
        "small": "aither-small",
        "medium": "aither-orchestrator",
        "large": "aither-reasoning",
    },
    "ollama": {
        "small": "llama3.2:3b",
        "medium": "nemotron-orchestrator-8b",
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
    "picolm": {
        "small": "picolm",
        "medium": "picolm",
        "large": "picolm",
    },
}

# Default inference URL — mcp.aitherium.com hosts the OpenAI-compatible
# /v1/chat/completions endpoint with ACTA auth and tenant scoping.
_GATEWAY_INFERENCE_URL = "https://mcp.aitherium.com/v1"
_DEMO_URL = "https://demo.aitherium.com"

# llmfit-derived model cache (populated lazily)
_llmfit_models: dict[str, str] | None = None
_llmfit_checked: bool = False


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
                default_model=self._model or "nemotron-orchestrator-8b",
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
        elif name == "picolm":
            from .picolm import PicoLMProvider
            return PicoLMProvider(
                binary=base_url or "",  # Overload base_url as binary path
                model=self._model or "",
            )
        else:
            raise ValueError(f"Unknown provider: {name}. Use 'gateway', 'ollama', 'openai', 'anthropic', or 'picolm'.")

    async def _try_ollama(self) -> LLMProvider | None:
        """Try Ollama on localhost. Returns provider or None."""
        try:
            from .ollama import OllamaProvider
            host = (self._config.ollama_host if self._config else None) or "http://localhost:11434"
            p = OllamaProvider(host=host, default_model=self._model or "nemotron-orchestrator-8b")
            if await p.health_check():
                self._provider_name = "ollama"
                logger.info("Auto-detected Ollama at %s", host)
                return p
        except Exception:
            pass
        return None

    async def _try_vllm(self) -> LLMProvider | None:
        """Try vLLM on configured URL, standard AitherOS ports, and vLLM default.

        vLLM is the PRIMARY local inference backend — it runs optimized containers
        on the user's GPU with proper batching, paged attention, and tensor parallelism.

        Priority: AITHER_VLLM_URL env → VLLM_URL env → port scan (8120, 8200-8203, 8000)
        """
        import os
        from .openai_compat import OpenAIProvider

        # Check explicit env var first
        vllm_env = os.environ.get("AITHER_VLLM_URL") or os.environ.get("VLLM_URL", "")
        if vllm_env:
            try:
                url = vllm_env.rstrip("/")
                if not url.endswith("/v1"):
                    url = f"{url}/v1"
                p = OpenAIProvider(base_url=url, api_key="not-needed", default_model=self._model or "")
                if await p.health_check():
                    try:
                        models = await p.list_models()
                        if models and not self._model:
                            p.default_model = models[0]
                    except Exception:
                        pass
                    self._provider_name = "vllm"
                    logger.info("vLLM from env var at %s (model: %s)", url, p.default_model)
                    return p
            except Exception:
                pass

        for port in (8120, 8200, 8201, 8202, 8203, 8000):
            try:
                url = f"http://localhost:{port}/v1"
                p = OpenAIProvider(
                    base_url=url,
                    api_key="not-needed",
                    default_model=self._model or "",
                )
                if await p.health_check():
                    # Discover what model is loaded
                    try:
                        models = await p.list_models()
                        if models and not self._model:
                            p.default_model = models[0]
                    except Exception:
                        pass
                    self._provider_name = "vllm"
                    logger.info("Auto-detected vLLM at localhost:%d (model: %s)", port, p.default_model)
                    return p
            except Exception:
                continue
        return None

    async def _auto_detect(self) -> LLMProvider:
        """Try backends in priority order: vLLM → Ollama → gateway → cloud APIs → demo.

        LOCAL GPU FIRST. vLLM containers are the primary backend — they use the GPU
        efficiently with batching and paged attention. Ollama is the fallback for
        AMD/Apple/no-Docker. Gateway is cloud fallback when no local GPU.
        """
        import os

        gateway_key = (
            (self._config.aither_api_key if self._config else "")
            or os.getenv("AITHER_API_KEY", "")
        )

        # 1. vLLM containers — PRIMARY local backend (best GPU utilization)
        vllm = await self._try_vllm()
        if vllm:
            return vllm

        # 2. Ollama — fallback local backend (AMD, Apple Silicon, no Docker)
        ollama = await self._try_ollama()
        if ollama:
            return ollama

        # 3. Gateway — cloud inference via gateway.aitherium.com
        if gateway_key:
            gateway_url = (
                (self._config.gateway_url if self._config else "")
                or os.getenv("AITHER_GATEWAY_URL", _GATEWAY_INFERENCE_URL)
            )
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
                logger.debug("Gateway not reachable, trying cloud API keys")

        # 4. Cloud API keys (Anthropic/OpenAI direct)
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

        # 5. PicoLM — edge inference (pure C, zero dependencies)
        picolm_binary = os.getenv("PICOLM_BINARY", "")
        picolm_model = os.getenv("PICOLM_MODEL", "")
        if picolm_binary and picolm_model:
            try:
                from .picolm import PicoLMProvider
                p = PicoLMProvider(binary=picolm_binary, model=picolm_model)
                if await p.health_check():
                    self._provider_name = "picolm"
                    logger.info("Auto-detected PicoLM at %s (model: %s)", picolm_binary, p.default_model)
                    return p
            except Exception:
                pass

        # 6. No backend available
        raise ConnectionError(
            "No LLM backend available.\n\n"
            "  Run setup:        python -m adk.setup\n"
            f"  Try the demo:     {_DEMO_URL}\n"
            "  Get an API key:   https://gateway.aitherium.com\n\n"
            "AitherOS Alpha uses vLLM containers for GPU inference.\n"
            "Run auto_setup() to detect your GPU and start the right containers."
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
        """Select model based on effort level (1-10).

        Priority:
        1. Explicit model (from constructor or env)
        2. Config profile models (from hardware profile YAML)
        3. llmfit hardware-scored recommendations (if sidecar available)
        4. Static provider defaults (fallback)

        llmfit provides real hardware-scored model selection instead of
        static lookup tables. When available, it accounts for actual VRAM,
        dynamic quantization, MoE architectures, and speed estimation.
        """
        if self._model:
            return self._model

        tier = "small" if effort <= 3 else "medium" if effort <= 6 else "large"

        # Check config profile models first (from hardware profile YAML)
        if self._config and getattr(self._config, "profile_models", None):
            pm = self._config.profile_models
            profile_map = {
                "small": pm.get("small", ""),
                "medium": pm.get("default", pm.get("chat", "")),
                "large": pm.get("large", pm.get("reasoning", "")),
            }
            if profile_map.get(tier):
                return profile_map[tier]

        # Try llmfit hardware-scored recommendations (cached after first call)
        llmfit_model = self._llmfit_model_for_tier(tier)
        if llmfit_model:
            return llmfit_model

        # Fall back to static provider defaults
        models = _EFFORT_MODELS.get(self._provider_name, {})
        return models.get(tier, self._model or "")

    @staticmethod
    def _llmfit_model_for_tier(tier: str) -> str | None:
        """Query llmfit for hardware-optimal model for a tier (cached).

        Maps ADK tiers to llmfit use_case categories:
        - small → chat (fast, low-latency)
        - medium → general (balanced)
        - large → reasoning (quality over speed)

        Returns Ollama-compatible model name or None if llmfit unavailable.
        """
        global _llmfit_models, _llmfit_checked

        if _llmfit_checked and _llmfit_models is not None:
            return _llmfit_models.get(tier)

        if _llmfit_checked:
            # Already tried, llmfit not available
            return None

        _llmfit_checked = True

        try:
            import asyncio
            from adk.llmfit import get_llmfit

            async def _fetch():
                fit = get_llmfit()
                if not await fit.is_available():
                    return None

                config = await fit.recommend_config()
                if "error" in config:
                    return None

                result = {}
                # Map: fast → small, balanced → medium, reasoning → large
                if config.get("fast") and config["fast"].get("model"):
                    result["small"] = config["fast"]["model"]
                if config.get("balanced") and config["balanced"].get("model"):
                    result["medium"] = config["balanced"]["model"]
                if config.get("reasoning") and config["reasoning"].get("model"):
                    result["large"] = config["reasoning"]["model"]
                return result if result else None

            # Try to run in existing event loop or create one
            try:
                loop = asyncio.get_running_loop()
                # Already in async context — can't await synchronously
                # Schedule as a task and return None for now;
                # the cache will populate on the next call from an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _fetch())
                    _llmfit_models = future.result(timeout=8)
            except RuntimeError:
                # No event loop running — safe to asyncio.run()
                _llmfit_models = asyncio.run(_fetch())

            if _llmfit_models:
                logger.info(
                    "llmfit models loaded: small=%s, medium=%s, large=%s",
                    _llmfit_models.get("small", "?"),
                    _llmfit_models.get("medium", "?"),
                    _llmfit_models.get("large", "?"),
                )
                return _llmfit_models.get(tier)

        except Exception as e:
            logger.debug("llmfit model selection unavailable: %s", e)

        return None

    @llm_retry(max_retries=5, base_delay_ms=500, max_delay_ms=16000)
    async def chat(
        self,
        messages: list[Message],
        model: str | None = None,
        effort: int | None = None,
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Route a chat request to the active provider."""
        provider = await self.get_provider()
        if model is None and effort is not None:
            model = self.model_for_effort(effort)
        resp = await provider.chat(
            messages, model=model, tool_choice=tool_choice,
            top_p=top_p, repetition_penalty=repetition_penalty, **kwargs,
        )
        if effort is not None:
            resp.effort_level = effort
        return resp

    async def chat_stream(
        self,
        messages: list[Message],
        model: str | None = None,
        effort: int | None = None,
        tool_choice: str | dict | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs,
    ):
        """Stream a chat response with degeneration detection."""
        provider = await self.get_provider()
        if model is None and effort is not None:
            model = self.model_for_effort(effort)
        detector = DegenerationDetector()
        async for chunk in provider.chat_stream(
            messages, model=model, tool_choice=tool_choice,
            top_p=top_p, repetition_penalty=repetition_penalty, **kwargs,
        ):
            if chunk.content and detector.feed(chunk.content):
                # Degeneration detected — signal done with special finish_reason
                yield StreamChunk(
                    content="", done=True, model=chunk.model,
                    finish_reason="degeneration",
                )
                return
            yield chunk

    async def list_models(self) -> list[str]:
        provider = await self.get_provider()
        return await provider.list_models()
