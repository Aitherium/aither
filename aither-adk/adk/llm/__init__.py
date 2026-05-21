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
        "small": "gemma4:4b",
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
    "deepseek": {
        "small": "deepseek-chat",
        "medium": "deepseek-chat",
        "large": "deepseek-reasoner",
    },
    "picolm": {
        "small": "picolm",
        "medium": "picolm",
        "large": "picolm",
    },
    "desktop": {
        "small": "",          # let MicroScheduler choose
        "medium": "",
        "large": "",
    },
    "dual": {
        "small": "gemma4:4b",               # local Ollama
        "medium": "aither-orchestrator",     # remote desktop
        "large": "aither-reasoning",         # remote desktop
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
        # Dual-mode: local provider for low effort, remote for high effort
        self._local_provider: LLMProvider | None = None
        self._remote_provider: LLMProvider | None = None
        self._remote_provider_name: str = ""
        self._remote_healthy: bool = True
        self._remote_health_checked: float = 0.0
        # Hybrid reasoning backend (set via set_reasoning_backend or config)
        self._reasoning_provider: LLMProvider | None = None
        self._reasoning_provider_name: str = ""
        self._reasoning_model: str | None = None

        if provider:
            self._provider = self._create_provider(provider, base_url, api_key)
            self._provider_name = provider
        else:
            self._deferred_base_url = base_url
            self._deferred_api_key = api_key

    # Well-known OpenAI-compatible base URLs for named providers
    _COMPAT_URLS: dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
    }

    # Default models per named provider
    _COMPAT_MODELS: dict[str, str] = {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "groq": "llama-3.3-70b-versatile",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }

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
                default_model=self._model or "gemma4:4b",
            )
        elif name in ("openai", "vllm", "lmstudio", "llamacpp", "groq", "together", "deepseek"):
            from .openai_compat import OpenAIProvider
            default_url = self._COMPAT_URLS.get(name, "https://api.openai.com/v1")
            default_model = self._COMPAT_MODELS.get(name, "gpt-4o-mini")
            return OpenAIProvider(
                base_url=base_url or default_url,
                api_key=api_key or "",
                default_model=self._model or default_model,
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
            raise ValueError(
                f"Unknown provider: {name}. "
                "Use 'gateway', 'ollama', 'openai', 'anthropic', 'deepseek', "
                "'groq', 'together', 'vllm', 'lmstudio', 'llamacpp', or 'picolm'."
            )

    async def _try_ollama(self) -> LLMProvider | None:
        """Try Ollama on localhost. Returns provider or None."""
        try:
            from .ollama import OllamaProvider
            host = (self._config.ollama_host if self._config else None) or "http://localhost:11434"
            p = OllamaProvider(host=host, default_model=self._model or "gemma4:4b")
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

        # Build port list: standard ports + user-configured extras
        ports = [8120, 8200, 8201, 8202, 8203, 8000]
        extra_ports = (self._config.vllm_extra_ports if self._config else "") or os.environ.get("AITHER_VLLM_PORTS", "")
        if extra_ports:
            for p_str in extra_ports.split(","):
                p_str = p_str.strip()
                if p_str.isdigit() and int(p_str) not in ports:
                    ports.insert(0, int(p_str))  # User ports checked first

        for port in ports:
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

        # Check DGX Spark / remote vLLM endpoint
        dgx_url = (self._config.dgx_url if self._config else "") or os.environ.get("AITHER_DGX_URL", "")
        if dgx_url:
            try:
                dgx_base = dgx_url.rstrip("/")
                if not dgx_base.endswith("/v1"):
                    dgx_base = f"{dgx_base}/v1"
                p = OpenAIProvider(base_url=dgx_base, api_key="not-needed", default_model=self._model or "")
                if await p.health_check():
                    try:
                        models = await p.list_models()
                        if models and not self._model:
                            p.default_model = models[0]
                    except Exception:
                        pass
                    self._provider_name = "vllm"
                    logger.info("vLLM from DGX/remote at %s (model: %s)", dgx_base, p.default_model)
                    return p
            except Exception:
                pass

        return None

    async def _try_desktop(self) -> LLMProvider | None:
        """Try connecting to a desktop AitherOS MicroScheduler for remote inference.

        Reads AITHER_CORE_LLM_URL from env or ~/.aither/config.json.
        Returns an OpenAI-compatible provider pointing at the desktop's MicroScheduler.
        """
        import os
        from .openai_compat import OpenAIProvider
        from adk.config import load_saved_config

        # Check env var first, then saved config
        desktop_url = os.environ.get("AITHER_CORE_LLM_URL", "")
        if not desktop_url:
            try:
                saved = load_saved_config()
                desktop_url = saved.get("core_llm_url", "")
            except (OSError, ValueError):
                pass

        if not desktop_url:
            return None

        desktop_url = desktop_url.rstrip("/")
        if not desktop_url.endswith("/v1"):
            desktop_url = f"{desktop_url}/v1"

        token = os.environ.get("AITHER_NODE_TOKEN", "")
        if not token:
            try:
                saved = load_saved_config()
                token = saved.get("node_token", "")
            except (OSError, ValueError):
                pass

        try:
            p = OpenAIProvider(
                base_url=desktop_url,
                api_key=token or "not-needed",
                default_model=self._model or "",
            )
            if await p.health_check():
                try:
                    models = await p.list_models()
                    if models and not self._model:
                        p.default_model = models[0]
                except Exception:
                    pass
                logger.info("Connected to desktop MicroScheduler at %s (model: %s)", desktop_url, p.default_model)
                return p
        except Exception:
            pass
        return None

    async def _check_remote_health(self) -> bool:
        """Check if the remote desktop provider is healthy (30s cache)."""
        import time
        now = time.time()
        if now - self._remote_health_checked < 30.0:
            return self._remote_healthy
        self._remote_health_checked = now
        if self._remote_provider is None:
            self._remote_healthy = False
            return False
        try:
            self._remote_healthy = await self._remote_provider.health_check()
        except Exception:
            self._remote_healthy = False
        return self._remote_healthy

    def _setup_reasoning_backend(self) -> None:
        """Configure a separate provider for reasoning (effort 7+) if config specifies one.

        Hybrid mode: local Nemotron for effort 1-6, cloud API for effort 7-10.
        """
        cfg = self._config
        if not cfg:
            return
        backend = cfg.reasoning_backend
        if not backend:
            return

        # Resolve API key for the reasoning backend
        api_key = cfg.reasoning_api_key
        if not api_key:
            if backend == "anthropic":
                api_key = cfg.anthropic_api_key
            elif backend == "openai":
                api_key = cfg.openai_api_key
            elif backend == "deepseek":
                api_key = cfg.deepseek_api_key
            elif backend == "gateway":
                api_key = cfg.aither_api_key

        if not api_key and backend not in ("vllm", "ollama"):
            logger.debug("Reasoning backend '%s' configured but no API key available", backend)
            return

        try:
            self._reasoning_provider = self._create_provider(
                backend,
                base_url=cfg.reasoning_base_url or None,
                api_key=api_key,
            )
            self._reasoning_provider_name = backend
            self._reasoning_model = cfg.reasoning_model or None
            logger.info(
                "Hybrid reasoning: %s (model=%s) for effort 7+",
                backend, self._reasoning_model or "default",
            )
        except Exception as e:
            logger.warning("Failed to set up reasoning backend '%s': %s", backend, e)

    async def _auto_detect(self) -> LLMProvider:
        """Try backends in priority order: vLLM → desktop → Ollama → gateway → cloud APIs → demo.

        LOCAL GPU FIRST. vLLM containers are the primary backend — they use the GPU
        efficiently with batching and paged attention. Desktop MicroScheduler is tried
        before Ollama for dual-mode setups. Ollama is the fallback for AMD/Apple/no-Docker.
        Gateway is cloud fallback when no local GPU.
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

        # 1.5. Desktop MicroScheduler — remote inference from connected desktop
        desktop = await self._try_desktop()
        if desktop:
            self._remote_provider = desktop
            self._remote_provider_name = "desktop"
            self._remote_healthy = True
            self._provider_name = "desktop"
            # Also try local Ollama for dual-mode (low-effort local, high-effort remote)
            ollama = await self._try_ollama()
            if ollama:
                self._local_provider = ollama
                self._provider_name = "dual"
                logger.info("Dual-mode: local Ollama (effort 1-3) + desktop MicroScheduler (effort 4+)")
                return ollama  # Default to local; chat() handles routing
            return desktop

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
        if os.getenv("DEEPSEEK_API_KEY"):
            self._provider_name = "deepseek"
            return self._create_provider("deepseek", api_key=os.getenv("DEEPSEEK_API_KEY"))
        if self._config and self._config.deepseek_api_key:
            self._provider_name = "deepseek"
            return self._create_provider("deepseek", api_key=self._config.deepseek_api_key)

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
            self._provider = await self._auto_detect()
            # Set up hybrid reasoning backend after primary detection
            if self._reasoning_provider is None:
                self._setup_reasoning_backend()
        return self._provider

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def switch_backend(
        self,
        provider: str,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Switch the LLM backend at runtime without recreating the agent.

        Usage:
            agent.llm.switch_backend("anthropic", api_key="sk-ant-...")
            agent.llm.switch_backend("deepseek")
            agent.llm.switch_backend("vllm", base_url="http://dgx-spark:8000/v1")
        """
        if model:
            self._model = model
        self._provider = self._create_provider(provider, base_url, api_key)
        self._provider_name = provider
        logger.info("Switched backend to %s (model=%s)", provider, model or "default")

    def set_reasoning_backend(
        self,
        provider: str,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Set a separate backend for high-effort (7+) reasoning tasks.

        Usage:
            agent.llm.set_reasoning_backend("anthropic", api_key="sk-ant-...")
            agent.llm.set_reasoning_backend("deepseek")
        """
        self._reasoning_provider = self._create_provider(provider, base_url, api_key)
        self._reasoning_provider_name = provider
        self._reasoning_model = model
        logger.info("Set reasoning backend to %s (model=%s)", provider, model or "default")

    def get_backends(self) -> dict:
        """Return info about all configured backends."""
        info = {
            "primary": self._provider_name or "not initialized",
            "model": self._model or "auto",
        }
        if self._local_provider is not None:
            info["local"] = "ollama"
        if self._remote_provider is not None:
            info["remote"] = self._remote_provider_name
        if hasattr(self, "_reasoning_provider") and self._reasoning_provider is not None:
            info["reasoning"] = self._reasoning_provider_name
            if self._reasoning_model:
                info["reasoning_model"] = self._reasoning_model
        return info

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
        """Route a chat request to the active provider.

        In dual-mode (local + desktop), routes based on effort:
        - effort 1-3: local provider (fast, small models)
        - effort 4+: remote desktop provider (large models, GPU)
        Falls back to local if remote is unreachable.
        """
        provider = await self.get_provider()
        if model is None and effort is not None:
            model = self.model_for_effort(effort)

        # Hybrid reasoning: effort 7+ → dedicated reasoning provider (cloud API)
        reasoning_prov = getattr(self, "_reasoning_provider", None)
        if (
            effort is not None
            and effort >= 7
            and reasoning_prov is not None
        ):
            reasoning_model = getattr(self, "_reasoning_model", None) or model
            try:
                resp = await reasoning_prov.chat(
                    messages, model=reasoning_model, tool_choice=tool_choice,
                    top_p=top_p, repetition_penalty=repetition_penalty, **kwargs,
                )
                resp.effort_level = effort
                return resp
            except Exception as e:
                logger.warning("Reasoning backend failed, falling back to primary: %s", e)

        # Dual-mode routing: high effort → remote desktop, low effort → local
        if (
            effort is not None
            and effort > 3
            and self._remote_provider is not None
            and await self._check_remote_health()
        ):
            try:
                resp = await self._remote_provider.chat(
                    messages, model=model, tool_choice=tool_choice,
                    top_p=top_p, repetition_penalty=repetition_penalty, **kwargs,
                )
                resp.effort_level = effort
                return resp
            except Exception as e:
                logger.warning("Remote desktop inference failed, falling back to local: %s", e)
                self._remote_healthy = False

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
