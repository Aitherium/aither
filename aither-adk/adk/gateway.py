"""Client for gateway.aitherium.com — auth, agent registration, discovery, remote inference."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("adk.gateway")


class GatewayClient:
    """Client for the AitherOS gateway at gateway.aitherium.com.

    All features are optional — agents work fully offline without the gateway.

    Features:
    - User registration and authentication
    - Agent capability advertisement
    - Agent discovery (find other agents on the network)
    - Remote inference proxy (future paid tier)
    """

    def __init__(
        self,
        gateway_url: str = "https://gateway.aitherium.com",
        api_key: str = "",
        timeout: float = 15.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, headers=self._headers())

    # ─── Auth ───

    async def register(self, email: str, password: str) -> dict:
        """Register a new user account. Returns {"api_key": "...", "user_id": "..."}."""
        async with self._client() as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/auth/register",
                json={"email": email, "password": password},
            )
            resp.raise_for_status()
            return resp.json()

    async def verify_email(self, token: str) -> dict:
        """Verify email with token from verification email."""
        async with self._client() as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/auth/verify",
                json={"token": token},
            )
            resp.raise_for_status()
            return resp.json()

    # ─── Agent Registry ───

    async def register_agent(
        self,
        agent_name: str,
        capabilities: list[str] | None = None,
        description: str = "",
        tools: list[str] | None = None,
    ) -> dict:
        """Register an agent with the network. Makes it discoverable by others."""
        async with self._client() as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/agents/register",
                json={
                    "name": agent_name,
                    "capabilities": capabilities or [],
                    "description": description,
                    "tools": tools or [],
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def discover_agents(
        self,
        capability: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Discover agents on the network, optionally filtered by capability."""
        params: dict = {"limit": limit}
        if capability:
            params["capability"] = capability
        async with self._client() as client:
            resp = await client.get(
                f"{self.gateway_url}/v1/agents/discover",
                params=params,
            )
            resp.raise_for_status()
            return resp.json().get("agents", [])

    # ─── Remote Inference ───

    async def inference(self, messages: list[dict], model: str = "", **kwargs) -> dict:
        """Send inference request to AitherOS cloud (paid tier, future)."""
        async with self._client() as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/inference",
                json={"messages": messages, "model": model, **kwargs},
            )
            resp.raise_for_status()
            return resp.json()

    # ─── Health ───

    async def health(self) -> bool:
        """Check if the gateway is reachable."""
        try:
            async with self._client() as client:
                resp = await client.get(f"{self.gateway_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
