"""
AitherClient — The canonical Python client for AitherOS.

This is the ONE client that all AitherOS tools should use.
AitherShell, AitherDesktop, AitherConnect, ADK — all import from here.
"""

import os
import uuid
import json
from typing import Optional, AsyncIterator, Dict, Any, List

import httpx

from aithersdk.models import (
    ChatRequest, ChatResponse, ChatMetadata, ToolCall,
    WillInfo, AgentInfo, ServiceHealth,
)


class AitherResponse:
    """Wrapper around ChatResponse with convenience methods."""

    def __init__(self, data: dict):
        self._data = data
        self._parsed: Optional[ChatResponse] = None

    @property
    def text(self) -> str:
        return self._data.get("response", "")

    @property
    def error(self) -> Optional[str]:
        return self._data.get("error") or None

    @property
    def success(self) -> bool:
        return not self.error

    @property
    def model(self) -> str:
        return self._data.get("model_used", "")

    @property
    def elapsed_ms(self) -> int:
        return (self._data.get("metadata") or {}).get("elapsed_ms", 0)

    @property
    def effort(self) -> int:
        return (self._data.get("metadata") or {}).get("effort_level", 0)

    @property
    def parsed(self) -> ChatResponse:
        if self._parsed is None:
            self._parsed = ChatResponse(**self._data)
        return self._parsed

    @property
    def raw(self) -> dict:
        return self._data

    def json(self) -> str:
        return json.dumps(self._data, default=str, ensure_ascii=False)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"AitherResponse(success={self.success}, model={self.model}, len={len(self.text)})"


class AitherClient:
    """
    Async client for AitherOS Genesis API.

    Usage:
        client = AitherClient()
        response = await client.chat("Hello!")
        print(response.text)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        will_url: Optional[str] = None,
        context_url: Optional[str] = None,
        a2a_url: Optional[str] = None,
        strata_url: Optional[str] = None,
        expedition_url: Optional[str] = None,
        voice_url: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        self.url = (url or os.environ.get("AITHER_URL")
                    or os.environ.get("AITHER_ORCHESTRATOR_URL")
                    or "http://localhost:8001").rstrip("/")
        self.will_url = (will_url or os.environ.get("AITHER_WILL_URL")
                         or "http://localhost:8097").rstrip("/")
        self._context_url = (context_url or os.environ.get("AITHER_LIVECONTEXT_URL")
                             or "http://localhost:8098").rstrip("/")
        self._a2a_url = (a2a_url or os.environ.get("AITHER_A2A_URL")
                         or "http://localhost:8766").rstrip("/")
        self._strata_url = (strata_url or os.environ.get("AITHER_STRATA_URL")
                            or "http://localhost:8136").rstrip("/")
        self._expedition_url = (expedition_url or os.environ.get("AITHEREXPEDITION_URL")
                                or "http://localhost:8785").rstrip("/")
        self._voice_url = (voice_url or os.environ.get("AITHER_VOICE_URL")
                           or "http://localhost:8083").rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("AITHER_API_KEY", "")
        self.tenant_id = tenant_id or os.environ.get("AITHER_TENANT_ID", "")
        self._client: Optional[httpx.AsyncClient] = None

        # Lazy-initialized service sub-clients
        self._context: Optional["ContextClient"] = None
        self._a2a: Optional["A2AClient"] = None
        self._strata: Optional["StrataClient"] = None
        self._expeditions: Optional["ExpeditionClient"] = None
        self._voice: Optional["VoiceClient"] = None
        self._conversations: Optional["ConversationClient"] = None

    def _auth_headers(self) -> Dict[str, str]:
        """Build auth headers for requests."""
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            # ACTA keys also set X-API-Key for billing middleware
            if self.api_key.startswith("aither_sk_live_"):
                headers["X-API-Key"] = self.api_key
            # PATs also set X-API-Key for Veil middleware
            elif self.api_key.startswith("aither_pat_"):
                headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._auth_headers(),
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Chat ────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        model: Optional[str] = None,
        persona: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        effort: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> AitherResponse:
        """Send a chat message and get a response."""
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "message": message,
            "conversation_id": conversation_id or self.session_id,
        }
        if model:
            payload["model"] = model
        if persona:
            payload["persona"] = persona
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if effort:
            payload["effort"] = effort

        resp = await client.post(f"{self.url}/chat", json=payload)
        if resp.status_code != 200:
            return AitherResponse({"error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
        return AitherResponse(resp.json())

    async def stream(
        self,
        message: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response token by token."""
        client = await self._get_client()
        payload = {
            "message": message,
            "conversation_id": self.session_id,
            "stream": True,
        }
        if model:
            payload["model"] = model

        async with client.stream("POST", f"{self.url}/chat/stream", json=payload) as resp:
            async for chunk in resp.aiter_text():
                yield chunk

    # ── Agents ──────────────────────────────────────────────────────

    async def delegate(self, agent: str, task: str, **kwargs) -> AitherResponse:
        """Delegate a task to a specific agent."""
        client = await self._get_client()
        payload = {"agent": agent, "task": task, "session_id": self.session_id, **kwargs}
        resp = await client.post(f"{self.url}/forge/dispatch/sync", json=payload)
        if resp.status_code != 200:
            return AitherResponse({"error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
        return AitherResponse(resp.json())

    async def list_agents(self) -> List[AgentInfo]:
        """List available agents."""
        client = await self._get_client()
        resp = await client.get(f"{self.url}/agents")
        if resp.status_code == 200:
            data = resp.json()
            return [AgentInfo(**a) for a in data.get("agents", [])]
        return []

    # ── Wills ───────────────────────────────────────────────────────

    async def list_wills(self) -> List[WillInfo]:
        """List available wills."""
        client = await self._get_client()
        resp = await client.get(f"{self.will_url}/wills")
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("wills", data if isinstance(data, list) else [])
            return [WillInfo(**w) for w in raw]
        return []

    async def set_will(self, will_id: str) -> bool:
        """Switch to a different will."""
        client = await self._get_client()
        resp = await client.post(f"{self.will_url}/will/set/{will_id}")
        return resp.status_code == 200

    async def get_will_prompt(self) -> str:
        """Get the current will's prompt text."""
        client = await self._get_client()
        resp = await client.get(f"{self.will_url}/will/prompt")
        if resp.status_code == 200:
            return resp.json().get("prompt", "")
        return ""

    # ── Safety ──────────────────────────────────────────────────────

    async def set_safety(self, level: str) -> bool:
        """Set safety level (professional, casual, unrestricted)."""
        client = await self._get_client()
        resp = await client.post(
            f"{self.will_url}/safety/level",
            json={"level": level}
        )
        return resp.status_code == 200

    # ── System ──────────────────────────────────────────────────────

    async def health(self) -> ServiceHealth:
        """Check Genesis health."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.url}/health", timeout=5.0)
            if resp.status_code == 200:
                return ServiceHealth(**resp.json())
        except Exception:
            pass
        return ServiceHealth(status="unreachable")

    async def is_available(self) -> bool:
        """Quick check if Genesis is reachable."""
        h = await self.health()
        return h.status == "healthy"

    async def list_models(self) -> List[str]:
        """List available LLM models."""
        client = await self._get_client()
        resp = await client.get(f"{self.url}/models")
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for tier_models in data.get("tiers", {}).values():
                models.extend(tier_models)
            return models
        return []

    # ── Service sub-clients (lazy) ───────────────────────────────────

    @property
    def context(self) -> "ContextClient":
        """LiveContext service client."""
        if self._context is None:
            from aithersdk.services.context import ContextClient
            self._context = ContextClient(self._context_url, self._get_client)
        return self._context

    @property
    def a2a(self) -> "A2AClient":
        """A2A Gateway service client."""
        if self._a2a is None:
            from aithersdk.services.a2a import A2AClient
            self._a2a = A2AClient(self._a2a_url, self._get_client)
        return self._a2a

    @property
    def strata(self) -> "StrataClient":
        """Strata virtual filesystem client."""
        if self._strata is None:
            from aithersdk.services.strata import StrataClient
            self._strata = StrataClient(self._strata_url, self._get_client)
        return self._strata

    @property
    def expeditions(self) -> "ExpeditionClient":
        """Expeditions service client."""
        if self._expeditions is None:
            from aithersdk.services.expeditions import ExpeditionClient
            self._expeditions = ExpeditionClient(self._expedition_url, self._get_client)
        return self._expeditions

    @property
    def voice(self) -> "VoiceClient":
        """AitherVoice service client."""
        if self._voice is None:
            from aithersdk.services.voice import VoiceClient
            self._voice = VoiceClient(self._voice_url, self._get_client)
        return self._voice

    @property
    def conversations(self) -> "ConversationClient":
        """Conversation management client."""
        if self._conversations is None:
            from aithersdk.services.conversations import ConversationClient
            self._conversations = ConversationClient(self._context_url, self._get_client)
        return self._conversations

    async def multi_health(self) -> Dict[str, str]:
        """Check health of all configured services. Returns {name: 'ok'|'down'}."""
        results: Dict[str, str] = {}
        services = {
            "genesis": self.health,
            "context": self.context.health,
            "a2a": self.a2a.health,
            "strata": self.strata.health,
            "expeditions": self.expeditions.health,
            "voice": self.voice.health,
        }
        for name, check_fn in services.items():
            try:
                h = await check_fn()
                status = h.status if hasattr(h, "status") else h.get("status", "unknown")
                results[name] = "ok" if status not in ("unreachable", "error") else "down"
            except Exception:
                results[name] = "down"
        return results

    # ── Context manager ─────────────────────────────────────────────

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
