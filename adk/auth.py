"""Shared auth — interoperates with ``aithershell`` and AitherIdentity.

This module reads and writes the same ``~/.aither/auth.json`` file that
``aithershell`` uses, so ``aither login`` from any tool authenticates the
whole CLI surface (chat, agents, scheduler, adapters, portal sync).

Supported credential types:

- **Local root** — auto-provisioned for offline dev. Token = ``aither_root_local``.
  Mirrors ``aithershell``'s ``ROOT_PROFILE``.
- **ACTA API key** — ``aither_sk_live_*`` bearer tokens issued by
  ``portal.aitherium.com`` (verified by AitherACTA / Identity). Used for
  cloud inference billing and per-tenant agent isolation.
- **OAuth device code** — RFC 8628 flow against AitherIdentity. The user
  visits a URL, types a code, and the CLI polls for the access_token.
  Backed by AitherIdentity's ``/oauth/device/code`` and ``/oauth/token``.

Environment overrides:

- ``AITHERIUM_BASE_URL``   — portal API root (default ``https://api.aitheros.ai``)
- ``AITHERIDENTITY_URL``   — identity service URL (default uses base URL)
- ``AITHER_OIDC_CLIENT_ID`` — OAuth client_id (default ``aither-cli``)
- ``AITHERIUM_API_KEY``    — short-circuit: skip the file store entirely
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adk.core.logging import get_logger

_log = get_logger("aither_adk.auth")

AUTH_FILE = Path.home() / ".aither" / "auth.json"
AUTH_VERSION = 1

DEFAULT_PORTAL_URL = "https://api.aitheros.ai"
DEFAULT_CLIENT_ID = "aither-cli"

# Same root profile aithershell uses — keeps the two tools in lock-step.
ROOT_PROFILE: dict[str, Any] = {
    "endpoint": "local",
    "genesis_url": os.environ.get("AITHER_GENESIS_URL", "http://localhost:8001"),
    "token_type": "local",
    "access_token": "aither_root_local",
    "expires_at": "",
    "user": {
        "id": "root",
        "username": "root",
        "display_name": "root",
        "email": "",
        "roles": ["admin"],
        "tenant_id": "",
        "tenant_slug": "",
    },
}


class AuthError(RuntimeError):
    """Raised on credential lookup / OAuth failures."""


# ---------------------------------------------------------------------------
# Profile + on-disk store
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Credentials:
    """Resolved credentials for a portal/identity round-trip."""

    access_token: str
    token_type: str = "bearer"  # bearer | acta | local
    endpoint: str = ""
    user: dict[str, Any] = field(default_factory=dict)
    expires_at: str = ""

    @property
    def is_local(self) -> bool:
        return self.token_type == "local"

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return False
        return exp < datetime.now(timezone.utc)

    def authorization_header(self) -> str:
        return f"Bearer {self.access_token}"


class AuthStore:
    """File-backed credential store at ``~/.aither/auth.json``.

    Format matches ``aithershell.auth`` so the two share state.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or AUTH_FILE

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != AUTH_VERSION:
                return None
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data.setdefault("version", AUTH_VERSION)
        self.path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        try:
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def set_profile(self, name: str, profile: dict[str, Any]) -> None:
        store = self.load() or {
            "version": AUTH_VERSION,
            "active_profile": name,
            "profiles": {},
        }
        store.setdefault("profiles", {})[name] = profile
        store["active_profile"] = name
        self.save(store)

    def get_active_profile(self) -> dict[str, Any] | None:
        store = self.load()
        if not store:
            return None
        active = store.get("active_profile", "local")
        return store.get("profiles", {}).get(active)

    def clear_profile(self, name: str) -> None:
        store = self.load()
        if not store:
            return
        store.get("profiles", {}).pop(name, None)
        if store.get("active_profile") == name:
            remaining = list(store.get("profiles", {}).keys())
            store["active_profile"] = remaining[0] if remaining else ""
        self.save(store)

    def ensure_root(self) -> dict[str, Any]:
        """Provision the local-root profile if no valid session exists.

        Mirrors :func:`aithershell.auth.ensure_root_profile` so a brand-new
        install just works — same behaviour as Linux auto-login as root.
        """
        existing = self.get_active_profile()
        if existing and existing.get("access_token"):
            return existing
        self.set_profile("local", ROOT_PROFILE.copy())
        return ROOT_PROFILE


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def resolve_credentials(
    *,
    store: AuthStore | None = None,
    env_key: str = "AITHERIUM_API_KEY",
) -> Credentials:
    """Resolve credentials in order: env var → auth.json → local root."""
    api_key = os.environ.get(env_key, "").strip()
    if api_key:
        token_type = "acta" if api_key.startswith("aither_sk_") else "bearer"
        return Credentials(
            access_token=api_key,
            token_type=token_type,
            endpoint=os.environ.get(
                "AITHERIUM_BASE_URL", DEFAULT_PORTAL_URL
            ).rstrip("/"),
        )

    s = store or AuthStore()
    profile = s.get_active_profile() or s.ensure_root()
    return Credentials(
        access_token=profile.get("access_token", ""),
        token_type=profile.get("token_type", "bearer"),
        endpoint=profile.get("endpoint", "local"),
        user=profile.get("user", {}),
        expires_at=profile.get("expires_at", ""),
    )


# ---------------------------------------------------------------------------
# OAuth 2.0 Device Authorization Grant (RFC 8628)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceCodeChallenge:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int = 5


async def begin_device_login(
    *,
    identity_url: str | None = None,
    client_id: str | None = None,
    scope: str = "openid profile email agents",
) -> DeviceCodeChallenge:
    """Kick off RFC 8628 device-code flow against AitherIdentity.

    Returns the challenge — the caller shows the user ``user_code`` and
    ``verification_uri``, then calls :func:`finish_device_login`.
    """
    import httpx

    url = (
        identity_url
        or os.environ.get(
            "AITHERIDENTITY_URL",
            os.environ.get("AITHERIUM_BASE_URL", DEFAULT_PORTAL_URL),
        )
    ).rstrip("/")
    cid = client_id or os.environ.get("AITHER_OIDC_CLIENT_ID", DEFAULT_CLIENT_ID)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{url}/oauth/device/code",
            data={"client_id": cid, "scope": scope},
        )
        if resp.status_code >= 400:
            raise AuthError(
                f"device-code start failed: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()

    return DeviceCodeChallenge(
        device_code=body["device_code"],
        user_code=body["user_code"],
        verification_uri=body["verification_uri"],
        verification_uri_complete=body.get(
            "verification_uri_complete", body["verification_uri"]
        ),
        expires_in=int(body.get("expires_in", 600)),
        interval=int(body.get("interval", 5)),
    )


async def finish_device_login(
    challenge: DeviceCodeChallenge,
    *,
    identity_url: str | None = None,
    client_id: str | None = None,
    store: AuthStore | None = None,
    profile_name: str = "portal",
) -> Credentials:
    """Poll the token endpoint until the user finishes the device flow.

    On success the resulting tokens are persisted to ``~/.aither/auth.json``
    under ``profile_name`` and made the active profile.
    """
    import httpx

    url = (
        identity_url
        or os.environ.get(
            "AITHERIDENTITY_URL",
            os.environ.get("AITHERIUM_BASE_URL", DEFAULT_PORTAL_URL),
        )
    ).rstrip("/")
    cid = client_id or os.environ.get("AITHER_OIDC_CLIENT_ID", DEFAULT_CLIENT_ID)
    deadline = asyncio.get_event_loop().time() + challenge.expires_in
    interval = max(1, challenge.interval)

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise AuthError("device login expired before user approval")
            await asyncio.sleep(interval)
            resp = await client.post(
                f"{url}/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": challenge.device_code,
                    "client_id": cid,
                },
            )
            body = resp.json() if resp.content else {}
            if resp.status_code == 200 and body.get("access_token"):
                break
            err = body.get("error", "")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += 5
                continue
            raise AuthError(f"device login failed: {err or resp.status_code}")

    creds = Credentials(
        access_token=body["access_token"],
        token_type=body.get("token_type", "bearer"),
        endpoint=url,
        expires_at=_expires_in_to_iso(body.get("expires_in")),
    )

    # Fetch /auth/me so the persisted profile mirrors aithershell layout.
    user: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            me = await client.get(
                f"{url}/auth/me",
                headers={"Authorization": creds.authorization_header()},
            )
            if me.status_code == 200:
                user = me.json()
        except Exception as e:  # noqa: BLE001
            _log.warning("auth.me.failed", extra={"err": str(e)})
    creds.user = user

    profile = {
        "endpoint": url,
        "genesis_url": user.get("genesis_url", ""),
        "token_type": creds.token_type,
        "access_token": creds.access_token,
        "expires_at": creds.expires_at,
        "user": user,
    }
    (store or AuthStore()).set_profile(profile_name, profile)
    return creds


def _expires_in_to_iso(expires_in: Any) -> str:
    try:
        secs = int(expires_in)
    except (TypeError, ValueError):
        return ""
    from datetime import timedelta

    return (datetime.now(timezone.utc) + timedelta(seconds=secs)).isoformat()


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


def logout(profile_name: str | None = None, store: AuthStore | None = None) -> None:
    s = store or AuthStore()
    if profile_name is None:
        data = s.load()
        if not data:
            return
        profile_name = data.get("active_profile") or "local"
    s.clear_profile(profile_name)
