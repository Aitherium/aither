"""Tests for the ADK llmfit integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from adk.llmfit import LLMFitClient, ModelFit, get_llmfit


# ── ModelFit dataclass ────────────────────────────────────────────────────


class TestModelFit:
    def test_from_json_full(self):
        data = {
            "name": "deepseek-r1:14b",
            "provider": "ollama",
            "params_b": 14.0,
            "context_length": 16384,
            "use_case": "reasoning",
            "is_moe": False,
            "fit_level": "perfect",
            "run_mode": "gpu_full",
            "score": 0.87,
            "estimated_tps": 42.0,
            "best_quant": "Q4_K_M",
            "utilization_pct": 65.2,
            "score_components": {
                "quality": 0.85,
                "speed": 0.90,
                "fit": 0.88,
                "context": 0.82,
            },
        }
        m = ModelFit.from_json(data)
        assert m.name == "deepseek-r1:14b"
        assert m.provider == "ollama"
        assert m.params_b == 14.0
        assert m.context_length == 16384
        assert m.score == 0.87
        assert m.estimated_tps == 42.0
        assert m.best_quant == "Q4_K_M"
        assert m.score_quality == 0.85
        assert m.score_speed == 0.90
        assert m.score_fit == 0.88
        assert m.score_context == 0.82
        assert m.vram_used_pct == 65.2
        assert m.runnable is True

    def test_from_json_minimal(self):
        m = ModelFit.from_json({"name": "tiny"})
        assert m.name == "tiny"
        assert m.score == 0.0
        assert m.runnable is False  # fit_level defaults to "too_tight"

    def test_from_json_empty(self):
        m = ModelFit.from_json({})
        assert m.name == ""
        assert m.provider == ""
        assert m.runnable is False

    def test_runnable_levels(self):
        for level in ("perfect", "good", "marginal"):
            m = ModelFit.from_json({"fit_level": level})
            assert m.runnable is True, f"fit_level={level} should be runnable"

        for level in ("too_tight", "impossible", ""):
            m = ModelFit.from_json({"fit_level": level})
            assert m.runnable is False, f"fit_level={level} should NOT be runnable"


# ── LLMFitClient ─────────────────────────────────────────────────────────


class TestLLMFitClientResolveUrl:
    def test_default_port(self):
        """Default URL uses AitherOS convention port 8793."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMFitClient()
            assert "8793" in client._base_url

    def test_env_override(self):
        with patch.dict("os.environ", {"AITHER_LLMFIT_URL": "http://custom:9999"}):
            client = LLMFitClient()
            assert client._base_url == "http://custom:9999"

    def test_docker_mode(self):
        with patch.dict("os.environ", {"AITHER_DOCKER_MODE": "true"}, clear=True):
            client = LLMFitClient()
            assert "aither-llmfit" in client._base_url
            assert "8793" in client._base_url

    def test_explicit_url(self):
        client = LLMFitClient(base_url="http://test:1234")
        assert client._base_url == "http://test:1234"


class TestLLMFitClientHealth:
    @pytest.mark.asyncio
    async def test_available_when_healthy(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        assert await client.is_available(force=True) is True

    @pytest.mark.asyncio
    async def test_unavailable_on_error(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        client._client = mock_httpx_client

        assert await client.is_available(force=True) is False

    @pytest.mark.asyncio
    async def test_health_caching(self):
        """Health check result is cached for TTL period."""
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        await client.is_available(force=True)
        await client.is_available()  # should use cache
        # Only one actual call due to caching
        assert mock_httpx_client.get.call_count == 1


class TestLLMFitClientSystemInfo:
    @pytest.mark.asyncio
    async def test_system_info_success(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "system": {
                "cpu_cores": 16,
                "cpu_name": "AMD Ryzen 9",
                "total_ram_gb": 64.0,
                "available_ram_gb": 48.0,
                "has_gpu": True,
                "gpu_name": "RTX 4090",
                "gpu_vram_gb": 24.0,
                "backend": "cuda_12",
                "unified_memory": False,
                "gpu_count": 1,
            }
        }
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        info = await client.system_info()
        assert info is not None
        assert info["gpu_name"] == "RTX 4090"
        assert info["gpu_vram_gb"] == 24.0
        assert info["has_gpu"] is True
        assert info["cpu_cores"] == 16

    @pytest.mark.asyncio
    async def test_system_info_unavailable(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(side_effect=ConnectionError())
        client._client = mock_httpx_client

        info = await client.system_info()
        assert info is None


class TestLLMFitClientModels:
    @pytest.mark.asyncio
    async def test_top_models(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "deepseek-r1:14b", "score": 0.9, "fit_level": "perfect",
                 "estimated_tps": 40.0, "score_components": {}},
                {"name": "llama3.2:3b", "score": 0.8, "fit_level": "good",
                 "estimated_tps": 80.0, "score_components": {}},
            ]
        }
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        models = await client.top_models(use_case="coding", limit=5)
        assert len(models) == 2
        assert models[0].name == "deepseek-r1:14b"
        assert models[0].score == 0.9
        assert models[1].name == "llama3.2:3b"

    @pytest.mark.asyncio
    async def test_top_models_unavailable(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(side_effect=ConnectionError())
        client._client = mock_httpx_client

        models = await client.top_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_best_for_task(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "best-model", "score": 0.95, "fit_level": "perfect",
                 "estimated_tps": 50.0, "score_components": {}},
            ]
        }
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        best = await client.best_for_task(use_case="reasoning")
        assert best is not None
        assert best.name == "best-model"

    @pytest.mark.asyncio
    async def test_best_for_task_none_available(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(side_effect=ConnectionError())
        client._client = mock_httpx_client

        best = await client.best_for_task()
        assert best is None

    @pytest.mark.asyncio
    async def test_search_model(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "deepseek-r1:14b", "score": 0.85, "fit_level": "good",
                 "estimated_tps": 35.0, "score_components": {}},
            ]
        }
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        results = await client.search_model("deepseek")
        assert len(results) == 1
        assert results[0].name == "deepseek-r1:14b"


class TestLLMFitClientRecommendConfig:
    @pytest.mark.asyncio
    async def test_recommend_config_full(self):
        """Test full config recommendation across all tiers."""
        client = LLMFitClient(base_url="http://localhost:8793")

        system_resp = MagicMock()
        system_resp.status_code = 200
        system_resp.json.return_value = {
            "system": {
                "cpu_cores": 16,
                "total_ram_gb": 64.0,
                "available_ram_gb": 48.0,
                "has_gpu": True,
                "gpu_name": "RTX 4090",
                "gpu_vram_gb": 24.0,
                "backend": "cuda_12",
            }
        }

        # Each tier query returns a different top model
        tier_models = {
            "chat": {"name": "llama3.2:3b", "score": 0.92, "estimated_tps": 80.0},
            "general": {"name": "nemotron-orchestrator-8b", "score": 0.88, "estimated_tps": 40.0},
            "reasoning": {"name": "deepseek-r1:14b", "score": 0.85, "estimated_tps": 25.0},
            "coding": {"name": "qwen2.5-coder:14b", "score": 0.86, "estimated_tps": 30.0},
            "embedding": {"name": "nomic-embed-text", "score": 0.90, "estimated_tps": 100.0},
        }

        async def mock_get(path, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if "/system" in path:
                resp.json.return_value = system_resp.json.return_value
            else:
                # Determine use_case from params
                params = kwargs.get("params", {})
                use_case = params.get("use_case", "general")
                model = tier_models.get(use_case, tier_models["general"])
                resp.json.return_value = {
                    "models": [{
                        **model,
                        "fit_level": "perfect",
                        "provider": "ollama",
                        "params_b": 14.0,
                        "best_quant": "Q4_K_M",
                        "score_components": {},
                    }]
                }
            return resp

        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = mock_get
        client._client = mock_httpx_client

        config = await client.recommend_config()

        assert "hardware" in config
        assert config["hardware"]["gpu"] == "RTX 4090"
        assert config["hardware"]["vram_gb"] == 24.0

        assert config["fast"]["model"] == "llama3.2:3b"
        assert config["balanced"]["model"] == "nemotron-orchestrator-8b"
        assert config["reasoning"]["model"] == "deepseek-r1:14b"
        assert config["coding"]["model"] == "qwen2.5-coder:14b"
        assert config["embedding"]["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_recommend_config_unavailable(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(side_effect=ConnectionError())
        client._client = mock_httpx_client

        config = await client.recommend_config()
        assert "error" in config


class TestLLMFitClientOllamaModels:
    @pytest.mark.asyncio
    async def test_recommended_ollama_models(self):
        client = LLMFitClient(base_url="http://localhost:8793")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "deepseek-r1:14b", "score": 0.9, "fit_level": "perfect",
                 "score_components": {}, "estimated_tps": 40.0},
                {"name": "llama3.2:3b", "score": 0.85, "fit_level": "good",
                 "score_components": {}, "estimated_tps": 80.0},
                # HF-style names are filtered out
                {"name": "meta-llama/Llama-3.2-3B", "score": 0.80, "fit_level": "good",
                 "score_components": {}, "estimated_tps": 60.0},
            ]
        }
        mock_httpx_client = AsyncMock()
        mock_httpx_client.get = AsyncMock(return_value=mock_resp)
        client._client = mock_httpx_client

        names = await client.recommended_ollama_models()
        assert "deepseek-r1:14b" in names
        assert "llama3.2:3b" in names
        # HF-style names should be excluded
        assert "meta-llama/Llama-3.2-3B" not in names


# ── Singleton ─────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_llmfit_returns_same_instance(self):
        """get_llmfit() should return the same client each time."""
        import adk.llmfit as llmfit_mod
        llmfit_mod._instance = None  # reset singleton

        c1 = get_llmfit()
        c2 = get_llmfit()
        assert c1 is c2

        llmfit_mod._instance = None  # cleanup

    def test_get_llmfit_with_custom_url(self):
        import adk.llmfit as llmfit_mod
        llmfit_mod._instance = None

        c = get_llmfit(base_url="http://custom:9999")
        assert c._base_url == "http://custom:9999"

        llmfit_mod._instance = None


# ── Integration with setup.py ─────────────────────────────────────────────


class TestSetupIntegration:
    @pytest.mark.asyncio
    async def test_recommended_models_llmfit_success(self):
        """_recommended_models_llmfit returns model names when llmfit is up."""
        from adk.setup import _recommended_models_llmfit

        mock_config = {
            "hardware": {"gpu": "RTX 4090", "vram_gb": 24},
            "fast": {"model": "llama3.2:3b"},
            "balanced": {"model": "nemotron-orchestrator-8b"},
            "reasoning": {"model": "deepseek-r1:14b"},
            "coding": {"model": "qwen2.5-coder:14b"},
            "embedding": None,
        }

        mock_client = AsyncMock()
        mock_client.is_available = AsyncMock(return_value=True)
        mock_client.recommend_config = AsyncMock(return_value=mock_config)

        with patch("adk.llmfit.get_llmfit", return_value=mock_client):
            models = await _recommended_models_llmfit()

        assert models is not None
        assert "nemotron-orchestrator-8b" in models
        assert "llama3.2:3b" in models
        assert "deepseek-r1:14b" in models
        assert "qwen2.5-coder:14b" in models
        assert "nomic-embed-text" in models  # always appended

    @pytest.mark.asyncio
    async def test_recommended_models_llmfit_unavailable(self):
        """Returns None when llmfit is not available."""
        from adk.setup import _recommended_models_llmfit

        mock_client = AsyncMock()
        mock_client.is_available = AsyncMock(return_value=False)

        with patch("adk.llmfit.get_llmfit", return_value=mock_client):
            result = await _recommended_models_llmfit()

        assert result is None

    @pytest.mark.asyncio
    async def test_recommended_models_llmfit_filters_hf_names(self):
        """HuggingFace-style names (with /) are excluded."""
        from adk.setup import _recommended_models_llmfit

        mock_config = {
            "hardware": {},
            "fast": {"model": "meta-llama/Llama-3.2-3B"},  # HF name
            "balanced": {"model": "nemotron-orchestrator-8b"},  # Ollama name
            "reasoning": None,
            "coding": None,
        }

        mock_client = AsyncMock()
        mock_client.is_available = AsyncMock(return_value=True)
        mock_client.recommend_config = AsyncMock(return_value=mock_config)

        with patch("adk.llmfit.get_llmfit", return_value=mock_client):
            models = await _recommended_models_llmfit()

        assert models is not None
        assert "nemotron-orchestrator-8b" in models
        # HF names should be filtered
        for m in models:
            assert "/" not in m

    def test_static_recommended_models_still_works(self):
        """Static fallback still returns models per profile."""
        from adk.setup import _recommended_models

        models = _recommended_models("nvidia_high")
        assert "nemotron-orchestrator-8b" in models
        assert "nomic-embed-text" in models

        models = _recommended_models("cpu_only")
        assert "gemma4:4b" in models

        models = _recommended_models("unknown_profile")
        assert models == _recommended_models("cpu_only")


# ── Config integration ────────────────────────────────────────────────────


class TestConfigIntegration:
    def test_config_has_llmfit_url(self):
        from adk.config import Config
        config = Config()
        assert hasattr(config, "llmfit_url")
        assert config.llmfit_url == ""  # default empty

    def test_config_llmfit_url_from_env(self):
        from adk.config import Config
        with patch.dict("os.environ", {"AITHER_LLMFIT_URL": "http://custom:9999"}):
            config = Config()
        assert config.llmfit_url == "http://custom:9999"

    def test_get_llmfit_client(self):
        from adk.config import Config
        config = Config()
        client = config.get_llmfit_client()
        assert client is not None
        # Cleanup singleton
        import adk.llmfit as llmfit_mod
        llmfit_mod._instance = None

    def test_get_llmfit_client_with_url(self):
        from adk.config import Config
        import adk.llmfit as llmfit_mod
        llmfit_mod._instance = None

        config = Config(llmfit_url="http://test:1234")
        client = config.get_llmfit_client()
        assert client is not None
        assert client._base_url == "http://test:1234"

        llmfit_mod._instance = None
