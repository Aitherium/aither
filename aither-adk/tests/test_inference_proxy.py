"""Tests for the inference proxy (gateway/inference.py) — model routing, tier gating, cost calculation."""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List

# Minimal TenantContext for tests (avoids importing AitherOS internals)
@dataclass
class MockTenantContext:
    user_id: str = "u_test"
    tenant_id: str = "t_test"
    tenant_slug: str = "test"
    tier: str = "free"
    roles: list = field(default_factory=list)
    session_id: str = "sess_1"
    token_balance: int = 10000
    plan: str = "explorer"
    api_key: str = "aither_sk_live_test"
    ip_allowlist: list = field(default_factory=list)
    allowed_models: list = field(default_factory=list)
    data_residency: str = "global"


# We test the inference module at the unit level — the actual proxy
# functions, model routing, tier checks, and cost calculation.
# These don't require the full MCP gateway ASGI stack.

# Load the inference module via importlib to avoid full AitherOS import chain
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFERENCE_PATH = os.path.join(
    PROJECT_ROOT, "AitherOS", "apps", "AitherNode", "gateway", "inference.py"
)

# Stub the auth module dependency
_auth_mock = MagicMock()
_auth_mock.TenantContext = MockTenantContext
sys.modules["apps"] = MagicMock()
sys.modules["apps.AitherNode"] = MagicMock()
sys.modules["apps.AitherNode.gateway"] = MagicMock()
sys.modules["apps.AitherNode.gateway.auth"] = _auth_mock

spec = importlib.util.spec_from_file_location("inference", INFERENCE_PATH)
inference_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_mod)

ModelRoute = inference_mod.ModelRoute
InferenceResult = inference_mod.InferenceResult
get_model_routes = inference_mod.get_model_routes
list_available_models = inference_mod.list_available_models
proxy_chat_completion = inference_mod.proxy_chat_completion
tier_allows_model = inference_mod.tier_allows_model
_proxy_vllm = inference_mod._proxy_vllm
_proxy_ollama = inference_mod._proxy_ollama


# ── Model Routes ──────────────────────────────────────────────────────────


class TestModelRoutes:
    def test_all_models_defined(self):
        routes = get_model_routes()
        assert "aither-small" in routes
        assert "aither-orchestrator" in routes
        assert "aither-reasoning" in routes
        assert "aither-vision" in routes
        assert "aither-coding" in routes

    def test_model_tiers_assigned(self):
        routes = get_model_routes()
        assert routes["aither-small"].tier == "free"
        assert routes["aither-orchestrator"].tier == "pro"
        assert routes["aither-reasoning"].tier == "enterprise"

    def test_model_costs_positive(self):
        routes = get_model_routes()
        for name, route in routes.items():
            assert route.cost_per_1k > 0, f"{name} has zero cost"

    def test_model_backends(self):
        routes = get_model_routes()
        assert routes["aither-small"].backend == "ollama"
        assert routes["aither-orchestrator"].backend == "vllm"


# ── Tier Access ───────────────────────────────────────────────────────────


class TestTierAccess:
    def test_free_allows_free(self):
        assert tier_allows_model("free", "free") is True

    def test_free_denies_pro(self):
        assert tier_allows_model("free", "pro") is False

    def test_pro_allows_free(self):
        assert tier_allows_model("pro", "free") is True

    def test_pro_allows_pro(self):
        assert tier_allows_model("pro", "pro") is True

    def test_pro_denies_enterprise(self):
        assert tier_allows_model("pro", "enterprise") is False

    def test_enterprise_allows_all(self):
        assert tier_allows_model("enterprise", "free") is True
        assert tier_allows_model("enterprise", "pro") is True
        assert tier_allows_model("enterprise", "enterprise") is True

    def test_unknown_tier_denied_from_pro(self):
        assert tier_allows_model("unknown", "pro") is False


# ── Model Listing ─────────────────────────────────────────────────────────


class TestModelListing:
    def test_list_models_free_tier(self):
        models = list_available_models("free")
        assert len(models) == 5  # All models listed
        accessible = [m for m in models if m["accessible"]]
        assert len(accessible) == 1  # Only aither-small
        assert accessible[0]["id"] == "aither-small"

    def test_list_models_pro_tier(self):
        models = list_available_models("pro")
        accessible = [m for m in models if m["accessible"]]
        assert len(accessible) == 2  # small + orchestrator

    def test_list_models_enterprise_tier(self):
        models = list_available_models("enterprise")
        accessible = [m for m in models if m["accessible"]]
        assert len(accessible) == 5  # All models

    def test_model_format_openai_compatible(self):
        models = list_available_models("free")
        for m in models:
            assert "id" in m
            assert "object" in m
            assert m["object"] == "model"
            assert "owned_by" in m
            assert "tier_required" in m
            assert "cost_per_1k_tokens" in m


# ── Proxy Chat Completion ──────────────────────────────────────────────────


class TestProxyChatCompletion:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_400(self):
        tenant = MockTenantContext(tier="enterprise")
        result = await proxy_chat_completion(
            {"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]},
            tenant,
        )
        assert result.success is False
        assert result.status_code == 400
        assert "Unknown model" in result.error

    @pytest.mark.asyncio
    async def test_tier_too_low_returns_403(self):
        tenant = MockTenantContext(tier="free")
        result = await proxy_chat_completion(
            {"model": "aither-orchestrator", "messages": [{"role": "user", "content": "hi"}]},
            tenant,
        )
        assert result.success is False
        assert result.status_code == 403
        assert "requires pro tier" in result.error

    @pytest.mark.asyncio
    async def test_insufficient_balance_returns_402(self):
        tenant = MockTenantContext(tier="pro", token_balance=0)
        result = await proxy_chat_completion(
            {"model": "aither-orchestrator", "messages": [{"role": "user", "content": "hi"}]},
            tenant,
        )
        assert result.success is False
        assert result.status_code == 402
        assert "Insufficient" in result.error

    @pytest.mark.asyncio
    async def test_model_allowlist_enforced(self):
        tenant = MockTenantContext(
            tier="enterprise",
            allowed_models=["aither-small"],
        )
        result = await proxy_chat_completion(
            {"model": "aither-orchestrator", "messages": [{"role": "user", "content": "hi"}]},
            tenant,
        )
        assert result.success is False
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_default_model_by_tier_free(self):
        """Free tier defaults to aither-small when model is empty."""
        tenant = MockTenantContext(tier="free", token_balance=10000)

        with patch.object(
            inference_mod, "_proxy_ollama",
            new=AsyncMock(return_value=InferenceResult(success=True, response={"model": "aither-small"})),
        ):
            result = await proxy_chat_completion(
                {"model": "", "messages": [{"role": "user", "content": "hi"}]},
                tenant,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_default_model_by_tier_pro(self):
        """Pro tier defaults to aither-orchestrator when model is empty."""
        tenant = MockTenantContext(tier="pro", token_balance=10000)

        with patch.object(
            inference_mod, "_proxy_vllm",
            new=AsyncMock(return_value=InferenceResult(success=True, response={"model": "aither-orchestrator"})),
        ):
            result = await proxy_chat_completion(
                {"model": "", "messages": [{"role": "user", "content": "hi"}]},
                tenant,
            )
            assert result.success is True


# ── vLLM Proxy ─────────────────────────────────────────────────────────────


class TestVLLMProxy:
    @pytest.mark.asyncio
    async def test_successful_vllm_call(self):
        route = ModelRoute(
            name="aither-orchestrator",
            display_name="Orchestrator",
            backend="vllm",
            url="http://localhost:8200/v1",
            internal_model="",
            tier="pro",
            cost_per_1k=3,
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "model": "some-internal-model",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as MockSession:
            session = MagicMock()
            session.post = MagicMock(return_value=mock_resp)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session

            result = await _proxy_vllm(route, {
                "model": "aither-orchestrator",
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert result.success is True
        assert result.response["model"] == "aither-orchestrator"  # Normalized
        assert result.tokens_used == 30
        assert result.aitherium_cost >= 1

    @pytest.mark.asyncio
    async def test_vllm_backend_error(self):
        route = ModelRoute(
            name="aither-orchestrator",
            display_name="Orchestrator",
            backend="vllm",
            url="http://localhost:8200/v1",
            internal_model="",
            tier="pro",
            cost_per_1k=3,
        )

        import aiohttp
        with patch("aiohttp.ClientSession") as MockSession:
            session = MagicMock()
            session.post = MagicMock(side_effect=aiohttp.ClientError("connection refused"))
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session

            result = await _proxy_vllm(route, {"messages": [{"role": "user", "content": "hi"}]})

        assert result.success is False
        assert result.status_code == 503


# ── Ollama Proxy ──────────────────────────────────────────────────────────


class TestOllamaProxy:
    @pytest.mark.asyncio
    async def test_successful_ollama_call(self):
        route = ModelRoute(
            name="aither-small",
            display_name="Small",
            backend="ollama",
            url="http://localhost:11434",
            internal_model="llama3.2:3b",
            tier="free",
            cost_per_1k=1,
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Hello from Ollama!"},
            "prompt_eval_count": 15,
            "eval_count": 25,
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as MockSession:
            session = MagicMock()
            session.post = MagicMock(return_value=mock_resp)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session

            result = await _proxy_ollama(route, {
                "model": "aither-small",
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert result.success is True
        assert result.response["model"] == "aither-small"
        assert result.response["choices"][0]["message"]["content"] == "Hello from Ollama!"
        assert result.response["usage"]["total_tokens"] == 40
        assert result.tokens_used == 40

    @pytest.mark.asyncio
    async def test_ollama_response_openai_format(self):
        """Ollama responses are converted to OpenAI format."""
        route = ModelRoute(
            name="aither-small",
            display_name="Small",
            backend="ollama",
            url="http://localhost:11434",
            internal_model="llama3.2:3b",
            tier="free",
            cost_per_1k=1,
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Test"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as MockSession:
            session = MagicMock()
            session.post = MagicMock(return_value=mock_resp)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            MockSession.return_value = session

            result = await _proxy_ollama(route, {"messages": [{"role": "user", "content": "hi"}]})

        resp = result.response
        assert resp["object"] == "chat.completion"
        assert "choices" in resp
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert "usage" in resp
        assert resp["usage"]["prompt_tokens"] == 5
        assert resp["usage"]["completion_tokens"] == 10


# ── Cost Calculation ──────────────────────────────────────────────────────


class TestCostCalculation:
    @pytest.mark.asyncio
    async def test_cost_scales_with_tokens(self):
        """Higher token usage = higher Aitherium cost."""
        route = ModelRoute(
            name="aither-orchestrator",
            display_name="Orchestrator",
            backend="vllm",
            url="http://localhost:8200/v1",
            internal_model="",
            tier="pro",
            cost_per_1k=3,
        )

        for total_tokens, expected_min_cost in [(100, 1), (1000, 3), (5000, 15)]:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "model": "m",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": total_tokens, "prompt_tokens": total_tokens // 2, "completion_tokens": total_tokens // 2},
            })
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            with patch("aiohttp.ClientSession") as MockSession:
                session = MagicMock()
                session.post = MagicMock(return_value=mock_resp)
                session.__aenter__ = AsyncMock(return_value=session)
                session.__aexit__ = AsyncMock(return_value=False)
                MockSession.return_value = session

                result = await _proxy_vllm(route, {"messages": [{"role": "user", "content": "hi"}]})

            assert result.aitherium_cost >= expected_min_cost, (
                f"tokens={total_tokens}, cost={result.aitherium_cost}, expected>={expected_min_cost}"
            )
