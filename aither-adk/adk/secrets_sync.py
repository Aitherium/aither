"""Secrets sync — pull secrets from AitherOS vault into local ADK store.

When running standalone, ADK agents store secrets locally in ~/.aither/secrets.json.
This module syncs those secrets from the platform vault so sovereign/standalone agents
can access credentials configured via the portal without manual env var setup.

Sync sources (in priority order):
1. AitherSecrets service (http://secrets:8111) — when sovereign stack is running
2. Gateway vault (https://gateway.aitherium.com/v1/secrets/pull) — when online
3. Local cache (~/.aither/secrets.json) — always available as fallback

Usage:
    from adk.secrets_sync import sync_secrets

    # Pull secrets from platform vault into local store
    synced = await sync_secrets()
    # {"OPENAI_API_KEY": "sk-...", "STRIPE_SECRET_KEY": "sk_live_..."}

    # Or use the client directly
    client = SecretsSync(api_key="...", secrets_url="http://secrets:8111")
    await client.pull()
"""

from __future__ import annotations
from adk._tls import tls_verify

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("adk.secrets_sync")

_AITHER_DIR = Path(os.environ.get("AITHER_DATA_DIR", os.path.expanduser("~/.aither")))
_SECRETS_FILE = _AITHER_DIR / "secrets.json"


def _load_local_secrets() -> Dict[str, str]:
    """Load secrets from local store."""
    if _SECRETS_FILE.exists():
        try:
            return json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_local_secrets(secrets: Dict[str, str]) -> None:
    """Save secrets to local store."""
    _AITHER_DIR.mkdir(parents=True, exist_ok=True)
    _SECRETS_FILE.write_text(json.dumps(secrets, indent=2), encoding="utf-8")
    try:
        os.chmod(_SECRETS_FILE, 0o600)
    except (OSError, AttributeError):
        pass


class SecretsSync:
    """Pull secrets from AitherOS vault into local ADK store."""

    def __init__(
        self,
        api_key: str = "",
        secrets_url: str = "",
        gateway_url: str = "",
        tenant: str = "",
    ):
        self.api_key = api_key or os.environ.get("AITHER_API_KEY", "")
        self.secrets_url = (
            secrets_url
            or os.environ.get("AITHER_SECRETS_URL", "")
        ).rstrip("/")
        self.gateway_url = (
            gateway_url
            or os.environ.get("AITHER_GATEWAY_URL", "")
            or os.environ.get("AITHER_PORTAL_URL", "https://portal.aitherium.com")
        ).rstrip("/")
        self.tenant = tenant or os.environ.get("AITHER_TENANT", "default")

    async def pull(self) -> Dict[str, str]:
        """Pull secrets from vault and merge into local store.

        Returns dict of synced secrets (keys only, values not logged).
        """
        import httpx

        synced: Dict[str, str] = {}

        # Try AitherSecrets service directly (sovereign stack)
        if self.secrets_url:
            try:
                async with httpx.AsyncClient(timeout=10, verify=tls_verify()) as client:
                    resp = await client.get(
                        f"{self.secrets_url}/secrets/all",
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        secrets = data if isinstance(data, dict) else data.get("secrets", {})
                        synced.update(secrets)
                        logger.info("Pulled %d secrets from AitherSecrets", len(secrets))
            except (httpx.HTTPError, OSError) as exc:
                logger.debug("AitherSecrets pull failed: %s", exc)

        # Try gateway vault (online)
        if not synced and self.api_key and self.gateway_url:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"{self.gateway_url}/v1/secrets/pull",
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        secrets = data.get("secrets", {}) if isinstance(data, dict) else {}
                        synced.update(secrets)
                        logger.info("Pulled %d secrets from gateway vault", len(secrets))
            except (httpx.HTTPError, OSError) as exc:
                logger.debug("Gateway secrets pull failed: %s", exc)

        if synced:
            # Merge with local store (remote wins on conflict)
            local = _load_local_secrets()
            local.update(synced)
            _save_local_secrets(local)
            logger.info("Synced %d secrets to local store", len(synced))

        return synced

    async def push(self, key: str, value: str) -> bool:
        """Push a secret to the platform vault.

        Returns True on success.
        """
        import httpx

        if self.secrets_url:
            try:
                async with httpx.AsyncClient(timeout=10, verify=tls_verify()) as client:
                    resp = await client.post(
                        f"{self.secrets_url}/secrets/set",
                        json={"key": key, "value": value},
                        headers=self._headers(),
                    )
                    return resp.status_code in (200, 201)
            except (httpx.HTTPError, OSError):
                pass

        if self.api_key and self.gateway_url:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        f"{self.gateway_url}/v1/secrets/set",
                        json={"key": key, "value": value},
                        headers=self._headers(),
                    )
                    return resp.status_code in (200, 201)
            except (httpx.HTTPError, OSError):
                pass

        return False

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant:
            headers["X-Tenant-ID"] = self.tenant
        return headers


async def sync_secrets(
    api_key: str = "",
    secrets_url: str = "",
    gateway_url: str = "",
) -> Dict[str, str]:
    """Convenience function — pull secrets from platform vault.

    Returns dict of synced secret keys.
    """
    client = SecretsSync(
        api_key=api_key,
        secrets_url=secrets_url,
        gateway_url=gateway_url,
    )
    return await client.pull()
