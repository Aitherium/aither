"""LLMFit integration — hardware-aware model recommendations for ADK agents.

This module provides a lightweight async client for the llmfit REST API
sidecar. llmfit is a Rust CLI/TUI + REST server that detects GPU/CPU/RAM
specs and scores 200+ LLM models across quality, speed, fit, and context
dimensions.

Upstream:
    Repository: https://github.com/wizzense/llmfit
    Author:     Alex Jones (@AlexsJones)
    License:    MIT

Instead of using static profile YAMLs to pick models, agents can query
llmfit for *actual* hardware-scored recommendations that account for:
- Dynamic quantization selection (best quality that fits in VRAM)
- MoE architecture support (active experts ≠ total params)
- Multi-GPU aggregation
- Speed estimation from GPU memory bandwidth

Usage:
    from adk.llmfit import get_llmfit, LLMFitClient

    # Singleton client (auto-resolves URL)
    fit = get_llmfit()

    # Check if llmfit sidecar is available
    if await fit.is_available():
        # Get top models for coding tasks
        models = await fit.top_models(use_case="coding", limit=5)
        for m in models:
            print(f"{m['name']} — score={m['score']}, tps={m['estimated_tps']}")

        # Get hardware info
        hw = await fit.system_info()
        print(f"GPU: {hw['gpu_name']} ({hw['gpu_vram_gb']}GB)")

        # Get recommended model for each ADK tier
        config = await fit.recommend_config()
        print(f"Fast model: {config['fast']['model']}")
        print(f"Reasoning: {config['reasoning']['model']}")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("adk.llmfit")

# Port 8793 is AitherOS convention; upstream default is 8787
_DEFAULT_PORT = 8793
_HEALTH_TTL = 30.0
_SYSTEM_TTL = 300.0
_MODELS_TTL = 60.0


@dataclass
class ModelFit:
    """A single model fit result from llmfit scoring."""
    name: str = ""
    provider: str = ""
    params_b: float = 0.0
    context_length: int = 0
    use_case: str = ""
    is_moe: bool = False
    fit_level: str = "too_tight"
    run_mode: str = "cpu_only"
    score: float = 0.0
    estimated_tps: float = 0.0
    best_quant: str = ""
    score_quality: float = 0.0
    score_speed: float = 0.0
    score_fit: float = 0.0
    score_context: float = 0.0
    vram_used_pct: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> ModelFit:
        sc = data.get("score_components", {})
        return cls(
            name=data.get("name", ""),
            provider=data.get("provider", ""),
            params_b=data.get("params_b", 0.0),
            context_length=data.get("context_length", 0),
            use_case=data.get("use_case", ""),
            is_moe=data.get("is_moe", False),
            fit_level=data.get("fit_level", "too_tight"),
            run_mode=data.get("run_mode", "cpu_only"),
            score=data.get("score", 0.0),
            estimated_tps=data.get("estimated_tps", 0.0),
            best_quant=data.get("best_quant", ""),
            score_quality=sc.get("quality", 0.0),
            score_speed=sc.get("speed", 0.0),
            score_fit=sc.get("fit", 0.0),
            score_context=sc.get("context", 0.0),
            vram_used_pct=data.get("utilization_pct", data.get("mem_pct", 0.0)),
            raw=data,
        )

    @property
    def runnable(self) -> bool:
        return self.fit_level in ("perfect", "good", "marginal")


class LLMFitClient:
    """Async client for the llmfit REST API sidecar.

    Resilient by design — all methods return sensible defaults when llmfit
    is unavailable so callers can degrade gracefully.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 5.0):
        self._base_url = (base_url or self._resolve_url()).rstrip("/")
        self._timeout = timeout
        self._client = None
        self._available: bool | None = None
        self._last_health: float = 0.0
        self._system_cache: dict | None = None
        self._system_cache_time: float = 0.0
        self._models_cache: dict[str, list[ModelFit]] = {}
        self._models_cache_time: float = 0.0

    @staticmethod
    def _resolve_url() -> str:
        """Resolve llmfit URL from env or convention."""
        url = os.environ.get("AITHER_LLMFIT_URL")
        if url:
            return url.rstrip("/")
        if os.environ.get("AITHER_DOCKER_MODE", "").lower() in ("1", "true"):
            return f"http://aither-llmfit:{_DEFAULT_PORT}"
        return f"http://localhost:{_DEFAULT_PORT}"

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    follow_redirects=True,
                )
            except ImportError:
                raise ImportError(
                    "httpx is required for llmfit integration. "
                    "Install with: pip install httpx"
                )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Health ──────────────────────────────────────────────────────────────

    async def is_available(self, force: bool = False) -> bool:
        """Check if llmfit sidecar is reachable. Cached for 30s."""
        now = time.monotonic()
        if not force and self._available is not None and (now - self._last_health) < _HEALTH_TTL:
            return self._available
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        self._last_health = now
        return self._available

    # ── System Info ─────────────────────────────────────────────────────────

    async def system_info(self) -> dict | None:
        """Get detected hardware specs. Cached for 5 minutes."""
        now = time.monotonic()
        if self._system_cache and (now - self._system_cache_time) < _SYSTEM_TTL:
            return self._system_cache
        try:
            client = await self._get_client()
            resp = await client.get("/api/v1/system")
            if resp.status_code == 200:
                data = resp.json()
                sys_data = data.get("system", data)
                self._system_cache = {
                    "cpu_cores": sys_data.get("cpu_cores", 0),
                    "cpu_name": sys_data.get("cpu_name", ""),
                    "total_ram_gb": sys_data.get("total_ram_gb", 0.0),
                    "available_ram_gb": sys_data.get("available_ram_gb", 0.0),
                    "has_gpu": sys_data.get("has_gpu", False),
                    "gpu_name": sys_data.get("gpu_name", ""),
                    "gpu_vram_gb": sys_data.get("gpu_vram_gb", 0.0),
                    "backend": sys_data.get("backend", "cpu_x86"),
                    "unified_memory": sys_data.get("unified_memory", False),
                    "gpu_count": sys_data.get("gpu_count", 0),
                    "raw": data,
                }
                self._system_cache_time = now
                return self._system_cache
        except Exception as e:
            logger.debug("llmfit system info unavailable: %s", e)
        return None

    # ── Model Queries ───────────────────────────────────────────────────────

    async def top_models(
        self,
        use_case: str | None = None,
        min_fit: str = "good",
        limit: int = 5,
        sort: str = "score",
    ) -> list[ModelFit]:
        """Get top-fitting models for this hardware."""
        cache_key = f"top:{use_case}:{min_fit}:{limit}:{sort}"
        now = time.monotonic()
        if cache_key in self._models_cache and (now - self._models_cache_time) < _MODELS_TTL:
            return self._models_cache[cache_key]

        try:
            client = await self._get_client()
            params: dict[str, Any] = {"min_fit": min_fit, "limit": limit, "sort": sort}
            if use_case:
                params["use_case"] = use_case

            resp = await client.get("/api/v1/models/top", params=params)
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("models", data if isinstance(data, list) else [])
                results = [ModelFit.from_json(m) for m in models_list]
                self._models_cache[cache_key] = results
                self._models_cache_time = now
                return results
        except Exception as e:
            logger.debug("llmfit top_models unavailable: %s", e)
        return []

    async def search_model(self, query: str) -> list[ModelFit]:
        """Search for a model by name."""
        try:
            client = await self._get_client()
            resp = await client.get(f"/api/v1/models/{query}")
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("models", data if isinstance(data, list) else [])
                return [ModelFit.from_json(m) for m in models_list]
        except Exception as e:
            logger.debug("llmfit search unavailable: %s", e)
        return []

    async def best_for_task(
        self,
        use_case: str = "general",
        min_tps: float = 0.0,
        min_fit: str = "good",
    ) -> ModelFit | None:
        """Get the single best model for a task type."""
        models = await self.top_models(use_case=use_case, min_fit=min_fit, limit=10)
        if min_tps > 0:
            models = [m for m in models if m.estimated_tps >= min_tps]
        return models[0] if models else None

    async def recommend_config(self) -> dict[str, Any]:
        """Generate hardware-optimized model configuration for ADK tiers.

        Returns:
            {
                "hardware": {"gpu": ..., "vram_gb": ..., "backend": ...},
                "fast": {"model": ..., "score": ..., "tps": ...},
                "balanced": {"model": ..., ...},
                "reasoning": {"model": ..., ...},
                "coding": {"model": ..., ...},
                "embedding": {"model": ..., ...},
            }
        """
        hw = await self.system_info()
        if not hw:
            return {"error": "llmfit unavailable"}

        tier_map = {
            "fast": ("chat", 30.0),
            "balanced": ("general", 10.0),
            "reasoning": ("reasoning", 5.0),
            "coding": ("coding", 10.0),
            "embedding": ("embedding", 0.0),
        }

        config: dict[str, Any] = {
            "hardware": {
                "gpu": hw.get("gpu_name", ""),
                "vram_gb": hw.get("gpu_vram_gb", 0),
                "ram_gb": hw.get("total_ram_gb", 0),
                "cpu_cores": hw.get("cpu_cores", 0),
                "backend": hw.get("backend", ""),
            },
        }

        for tier, (use_case, min_tps) in tier_map.items():
            best = await self.best_for_task(use_case=use_case, min_tps=min_tps)
            if best:
                config[tier] = {
                    "model": best.name,
                    "provider": best.provider,
                    "score": best.score,
                    "estimated_tps": best.estimated_tps,
                    "fit_level": best.fit_level,
                    "best_quant": best.best_quant,
                    "params_b": best.params_b,
                }
            else:
                config[tier] = None

        return config

    async def recommended_ollama_models(self, limit: int = 5) -> list[str]:
        """Get Ollama-compatible model names recommended for this hardware.

        Returns model names in Ollama format (e.g. 'deepseek-r1:14b'),
        useful for auto-pulling during setup.
        """
        models = await self.top_models(min_fit="good", limit=limit)
        # Filter for models that have Ollama-compatible names
        ollama_names = []
        for m in models:
            name = m.name.lower()
            # llmfit uses HuggingFace names — convert common patterns to Ollama
            if "/" in name:
                # e.g. "meta-llama/Llama-3.2-3B" → approximate
                continue
            # Already Ollama format (e.g. from Ollama provider detection)
            ollama_names.append(m.name)
        return ollama_names


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_instance: LLMFitClient | None = None


def get_llmfit(base_url: str | None = None) -> LLMFitClient:
    """Get or create the singleton LLMFitClient."""
    global _instance
    if _instance is None:
        _instance = LLMFitClient(base_url=base_url)
    return _instance
