"""Tiny client for portal.aitherium.com — pull agent specs, push telemetry.

Designed for the binary-install path: a single ``aither`` binary on a
laptop can fetch an agent spec from the portal, run it locally with a
local orchestrator model, and stream back telemetry/usage events to the
portal so the user sees their fleet in one place.

Credentials are resolved by :mod:`aither_adk.auth` — env var first, then
the shared ``~/.aither/auth.json`` store (same file ``aithershell`` uses),
then a no-op local-root profile. If the resolved credentials are local-only
the methods quietly no-op so offline use stays offline.

Environment:

- ``AITHERIUM_API_KEY``  — ACTA bearer or session token. Overrides the file.
- ``AITHERIUM_BASE_URL`` — defaults to ``https://api.aitheros.ai``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from adk.auth import Credentials, resolve_credentials
from adk.core.logging import get_logger

_log = get_logger("aither_adk.portal")

DEFAULT_BASE_URL = "https://api.aitheros.ai"


@dataclass(slots=True)
class PortalConfig:
    base_url: str = DEFAULT_BASE_URL
    credentials: Credentials | None = None
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "PortalConfig":
        creds = resolve_credentials()
        base = os.environ.get("AITHERIUM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        # If creds came from a real OAuth/ACTA login, prefer that endpoint.
        if creds.endpoint and creds.endpoint != "local":
            base = creds.endpoint
        return cls(
            base_url=base,
            credentials=creds,
            timeout=float(os.environ.get("AITHERIUM_TIMEOUT", "30")),
        )

    @property
    def configured(self) -> bool:
        c = self.credentials
        return bool(c and c.access_token and not c.is_local and not c.is_expired)

    # Legacy compat — older callers built PortalConfig with api_key=
    @property
    def api_key(self) -> str | None:
        return self.credentials.access_token if self.credentials else None


class PortalClient:
    """Minimal portal HTTPS client. Offline-safe: methods no-op when not
    configured (local-root profile, no API key)."""

    def __init__(
        self,
        config: PortalConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        if config is None:
            config = PortalConfig.from_env()
        if api_key:
            # Explicit override; treat as a bearer token.
            config = PortalConfig(
                base_url=config.base_url,
                credentials=Credentials(access_token=api_key, token_type="bearer"),
                timeout=config.timeout,
            )
        self.config = config

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.config.configured:
            _log.debug("portal.skip", extra={"reason": "no credentials"})
            return None
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "PortalClient requires httpx. "
                "Install with: pip install 'aither-adk[full]'"
            ) from e
        url = f"{self.config.base_url}{path}"
        headers = kwargs.pop("headers", {}) or {}
        assert self.config.credentials is not None
        headers.setdefault(
            "Authorization", self.config.credentials.authorization_header()
        )
        headers.setdefault("Accept", "application/json")
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.request(method, url, headers=headers, **kwargs)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return resp.text

    # ---- read paths ------------------------------------------------------

    async def whoami(self) -> dict[str, Any] | None:
        """Confirm credentials by calling AitherIdentity's ``/auth/me``."""
        return await self._request("GET", "/auth/me")

    async def get_agent_spec(self, name: str) -> dict[str, Any] | None:
        """Pull a published agent card from the portal by name."""
        return await self._request("GET", f"/v1/agents/{name}/spec")

    async def list_agents(self) -> list[dict[str, Any]] | None:
        return await self._request("GET", "/v1/agents")

    async def list_orchestrator_models(self) -> list[dict[str, Any]] | None:
        """List orchestrator models the portal offers (e.g. nemotron)."""
        return await self._request("GET", "/v1/orchestrator/models")

    # ---- write paths -----------------------------------------------------

    async def report_run(
        self,
        *,
        agent_name: str,
        prompt: str,
        output: str,
        steps: int,
        finish_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Push a single agent run as telemetry to portal.aitherium.com."""
        payload = {
            "agent": agent_name,
            "prompt": prompt,
            "output": output,
            "steps": steps,
            "finish_reason": finish_reason,
            "extra": extra or {},
        }
        await self._request("POST", "/v1/telemetry/agent_run", json=payload)

    async def register_local_agent(
        self, spec: dict[str, Any], *, endpoint: str | None = None
    ) -> dict[str, Any] | None:
        """Tell the portal we are running this agent locally.

        Mirrors the auto-onboard flow described in ``AGENTS.md`` — gives
        the user a unified view of cloud + local agents in the portal UI.
        """
        body = {"spec": spec, "endpoint": endpoint}
        return await self._request("POST", "/v1/agents/register-local", json=body)

