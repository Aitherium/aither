"""Cloud Deploy client — programmatic access to AitherOS GPU model deployment.

Provides a thin async client for the Genesis /deploy/cloud-model/* API surface.
Handles model browsing, GPU search, cost estimation, one-click deployment,
status polling, and teardown.

Works with local AitherOS instances (Genesis on port 8001) and the public
Aitherium gateway (mcp.aitherium.com), which proxies to Genesis.

All paths target Genesis's deploy router (prefix=/deploy, routes at /cloud-model/*).
The PortalGateway (port 8206) is NOT used — it has an incompatible API surface.

Usage:
    from adk.cloud_deploy import CloudDeployClient, get_cloud_deploy_client

    client = get_cloud_deploy_client()

    # Browse available profiles
    profiles = await client.list_profiles()

    # Search GPU marketplace
    offers = await client.search_gpus(min_vram_gb=24)

    # Estimate cost (no commitment)
    estimate = await client.estimate_cost("reasoning")

    # Deploy
    session = await client.deploy("reasoning")

    # Poll until ready
    result = await client.wait_for_deployment(session["session_id"])

    # Use the endpoint
    print(result["session"]["vllm_url"])  # → http://x.x.x.x:8000

    # Teardown
    await client.teardown(session["session_id"])

Author: Aitherium
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("adk.cloud_deploy")

# ─── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_GENESIS_URL = "http://localhost:8001"
_DEFAULT_GATEWAY_URL = "https://mcp.aitherium.com"

_POLL_INTERVAL = 10  # seconds
_POLL_TIMEOUT = 600  # 10 minutes

# ─── Route prefix ────────────────────────────────────────────────────────────
# Genesis deploy router uses prefix="/deploy", so cloud model routes live at
# /deploy/cloud-model/*. This matches what MCP tools, Shell, and AitherZero use.

_P = "/deploy/cloud-model"


# ─── Client ──────────────────────────────────────────────────────────────────

class CloudDeployClient:
    """Async client for AitherOS cloud GPU model deployment.

    Targets Genesis (port 8001) or the Aitherium gateway. Does NOT target
    PortalGateway — its /deploy/* routes have incompatible schemas.
    """

    def __init__(
        self,
        api_key: str = "",
        genesis_url: str = "",
        gateway_url: str = "",
    ):
        self._api_key = api_key or os.environ.get("AITHER_API_KEY", "")
        self._genesis_url = genesis_url or os.environ.get("GENESIS_URL", _DEFAULT_GENESIS_URL)
        self._gateway_url = gateway_url or os.environ.get("AITHER_GATEWAY_URL", _DEFAULT_GATEWAY_URL)
        self._base_url: Optional[str] = None
        self._http = None

    @property
    def _client(self):
        if self._http is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx required for cloud deploy. Install: pip install httpx")
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._http = httpx.AsyncClient(timeout=60.0, headers=headers)
        return self._http

    async def _resolve_base(self) -> str:
        """Find a reachable Genesis instance."""
        if self._base_url:
            return self._base_url

        # Probe Genesis health (the main /health endpoint, not deploy-specific)
        try:
            resp = await self._client.get(f"{self._genesis_url}/health", timeout=5.0)
            if resp.status_code == 200:
                self._base_url = self._genesis_url
                logger.info(f"Cloud deploy using Genesis: {self._genesis_url}")
                return self._genesis_url
        except Exception:
            pass

        # Fall back to public gateway
        self._base_url = self._gateway_url
        logger.info(f"Cloud deploy using gateway: {self._gateway_url}")
        return self._gateway_url

    async def _get(self, path: str) -> Dict[str, Any]:
        base = await self._resolve_base()
        resp = await self._client.get(f"{base}{path}")
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        base = await self._resolve_base()
        resp = await self._client.post(f"{base}{path}", json=data or {})
        resp.raise_for_status()
        return resp.json()

    # ── Model & Profile Browsing ─────────────────────────────────────────

    async def list_profiles(self) -> Dict[str, Any]:
        """List cloud_node_profiles.yaml deployment profiles.

        Returns profile names with model, VRAM requirements, and pricing.
        Use profile names (e.g. 'reasoning', 'orchestrator') with deploy().
        """
        result = await self._get(f"{_P}/profiles")
        return result.get("profiles", {})

    async def search_models(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search HuggingFace for deployable models.

        Returns models with VRAM estimates and deploy-ready flags.
        """
        result = await self._get(f"{_P}/search-models?query={query}&limit={limit}")
        return result.get("models", [])

    async def list_registry_models(self) -> List[Dict[str, Any]]:
        """List locally-known models from ModelRegistry."""
        result = await self._get(f"{_P}/registry")
        return result.get("models", [])

    # ── GPU Marketplace ──────────────────────────────────────────────────

    async def search_gpus(
        self,
        min_vram_gb: int = 24,
        max_price_per_hour: float = 1.0,
        gpu_model: str = "",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search GPU marketplace for available instances."""
        params = f"min_vram_gb={min_vram_gb}&max_price_per_hour={max_price_per_hour}&limit={limit}"
        if gpu_model:
            params += f"&gpu_model={gpu_model}"
        result = await self._get(f"{_P}/marketplace?{params}")
        return result.get("offers", [])

    # ── Cost Estimation ──────────────────────────────────────────────────

    async def estimate_cost(
        self,
        model: str,
        profile: str = "",
        min_vram_gb: float = 0,
    ) -> Dict[str, Any]:
        """Estimate deployment cost. No resources provisioned.

        Args:
            model: Profile name, model name, or HuggingFace model ID.
            profile: Force a specific cloud_node_profiles.yaml profile.
            min_vram_gb: Override minimum VRAM (0 = auto).

        Returns:
            Dict with VRAM estimate, matching GPU offers, hourly/daily cost.
        """
        return await self._post(f"{_P}/estimate", {
            "model": model,
            "profile": profile,
            "min_vram_gb": min_vram_gb,
        })

    # ── Deploy ───────────────────────────────────────────────────────────

    async def deploy(
        self,
        model: str,
        served_name: str = "",
        profile: str = "",
        max_price_per_hour: float = 0.0,
        min_vram_gb: float = 0,
        max_model_len: int = 0,
        offer_id: str = "",
        register_with_queue: bool = True,
    ) -> Dict[str, Any]:
        """Deploy a model to a cloud GPU.

        Returns immediately with session_id. Use wait_for_deployment() to
        block until the model is serving, or poll with get_status().

        Args:
            model: Profile name (e.g. 'reasoning'), model name, or HF model ID.
            served_name: Name for /v1/models (auto-derived if empty).
            profile: Force a cloud_node_profiles.yaml profile.
            max_price_per_hour: Max GPU cost (0 = auto, typically $0.20).
            min_vram_gb: Min VRAM (0 = auto from model).
            max_model_len: Max context length (0 = 32768).
            offer_id: Specific GPU offer ID (skips marketplace search).
            register_with_queue: Register with LLMQueue for routing.
        """
        return await self._post(f"{_P}", {
            "model": model,
            "served_name": served_name,
            "profile": profile,
            "max_price_per_hour": max_price_per_hour,
            "min_vram_gb": min_vram_gb,
            "max_model_len": max_model_len,
            "offer_id": offer_id,
            "register_with_queue": register_with_queue,
        })

    async def deploy_sync(
        self,
        model: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Deploy a model and block until complete (up to 10 min).

        Same args as deploy(). Returns the full session with vllm_url.
        """
        return await self._post(f"{_P}/sync", {
            "model": model,
            **kwargs,
        })

    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get deployment status and progress."""
        return await self._get(f"{_P}/status/{session_id}")

    async def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List deployment sessions."""
        suffix = "?active_only=true" if active_only else ""
        result = await self._get(f"{_P}/sessions{suffix}")
        return result.get("sessions", [])

    async def get_billing(self) -> Dict[str, Any]:
        """Get GPU billing summary with live costs."""
        return await self._get(f"{_P}/billing")

    async def get_pool_status(self) -> Dict[str, Any]:
        """Get unified compute pool status (local + cloud + sovereign)."""
        return await self._get(f"{_P}/pool")

    async def teardown(self, session_id: str) -> Dict[str, Any]:
        """Tear down a deployment and stop billing."""
        return await self._post(f"{_P}/teardown/{session_id}")

    # ── Stack (multi-GPU) ────────────────────────────────────────────────

    async def deploy_stack(
        self,
        profiles: Optional[List[str]] = None,
        replicas: int = 1,
        max_price_per_hour: float = 0.0,
    ) -> Dict[str, Any]:
        """Deploy the full inference stack to N cloud GPU nodes."""
        return await self._post(f"{_P}/stack", {
            "profiles": profiles,
            "replicas": replicas,
            "max_price_per_hour": max_price_per_hour,
        })

    async def teardown_stack(self, profiles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tear down all cloud stack deployments."""
        return await self._post(f"{_P}/stack/teardown", {
            "profiles": profiles,
        })

    # ── Convenience ──────────────────────────────────────────────────────

    async def get_endpoint(self, session_id: str) -> Dict[str, Any]:
        """Get connection info for a completed deployment.

        Extracts vllm_url from the session status and returns a
        ready-to-use endpoint dict.
        """
        status = await self.get_status(session_id)
        session = status.get("session", {})
        vllm_url = session.get("vllm_url", "")
        return {
            "base_url": vllm_url,
            "chat_url": f"{vllm_url}/v1/chat/completions" if vllm_url else "",
            "completions_url": f"{vllm_url}/v1/completions" if vllm_url else "",
            "models_url": f"{vllm_url}/v1/models" if vllm_url else "",
            "served_name": session.get("served_name", ""),
            "backend_name": session.get("backend_name", ""),
            "gpu_model": session.get("gpu_model", ""),
            "price_per_hour": session.get("price_per_hour", 0),
        }

    async def wait_for_deployment(
        self,
        session_id: str,
        timeout: float = _POLL_TIMEOUT,
        poll_interval: float = _POLL_INTERVAL,
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Block until deployment completes or fails.

        Args:
            session_id: Deployment session ID.
            timeout: Max seconds to wait (default 600).
            poll_interval: Seconds between polls (default 10).
            on_progress: Optional callback(phase, percent, message).

        Returns:
            Final session status dict.

        Raises:
            TimeoutError: If deployment doesn't complete in time.
            RuntimeError: If deployment fails.
        """
        start = time.time()
        last_phase = ""

        while time.time() - start < timeout:
            status = await self.get_status(session_id)
            session = status.get("session", {})
            phase = session.get("phase", "")
            percent = session.get("percent", 0)

            if phase != last_phase:
                if on_progress:
                    on_progress(phase, percent, session.get("message", ""))
                last_phase = phase

            if phase in ("complete", "completed"):
                return status
            if phase == "failed":
                raise RuntimeError(
                    f"Deployment failed: {session.get('error', 'Unknown error')}"
                )

            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Deployment {session_id} did not complete within {timeout}s"
        )

    async def deploy_and_wait(
        self,
        model: str,
        on_progress: Optional[callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Deploy a model and wait for it to be ready.

        Convenience method combining deploy() + wait_for_deployment().
        Returns the endpoint info dict.
        """
        result = await self.deploy(model, **kwargs)
        session_id = result.get("session_id", "")
        if not session_id:
            raise RuntimeError(f"Deploy failed: {result}")

        await self.wait_for_deployment(session_id, on_progress=on_progress)
        return await self.get_endpoint(session_id)

    async def close(self):
        """Close the HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None


# ─── Singleton ───────────────────────────────────────────────────────────────

_client: Optional[CloudDeployClient] = None


def get_cloud_deploy_client(**kwargs) -> CloudDeployClient:
    """Get or create the global cloud deploy client."""
    global _client
    if _client is None:
        _client = CloudDeployClient(**kwargs)
    return _client
