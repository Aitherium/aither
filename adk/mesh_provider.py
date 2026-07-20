"""adk.mesh_provider — one-command setup for AitherNet community inference provider.

A community node self-services into the AitherNet compute market via:
  1. Advertise an inference endpoint to the peer record
  2. Grant participation consent (GDPR-style, revocable)
  3. Poll for operator trust tier (fail-closed gate)
  4. Print the operator command for final registration

This wraps the manual runbook into one idempotent command: adk mesh provide
--inference-url http://10.77.x.x:8000/v1 --model NAME
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("adk.mesh_provider")

# Public federation entry point. Internal fleet callers override these via
# AITHER_STRATA_URL / AITHER_CONDUCTOR_URL (set in placement_policy.yaml + compose)
# with the in-cluster service addresses; a standalone/remote node reaches the same
# services through the authenticated public gateway.
DEFAULT_STRATA_URL = os.getenv("AITHER_STRATA_URL", "https://gateway.aitherium.com")
DEFAULT_CONDUCTOR_URL = os.getenv("AITHER_CONDUCTOR_URL", "https://gateway.aitherium.com")


def _tls_verify() -> Any:
    """Return TLS verify policy (CA bundle path or True). Never False."""
    from adk._tls import tls_verify
    return tls_verify()


def _get_auth_token() -> str | None:
    """Resolve an auth token from adk config or environment.

    Returns the token or None if not set. This token is used to authenticate
    API calls to Strata and Conductor endpoints.
    """
    # Try environment first
    token = os.getenv("AITHER_AUTH_TOKEN", "").strip()
    if token:
        return token

    # Try adk config file
    try:
        from pathlib import Path
        config_file = Path.home() / ".aither" / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            return config.get("auth_token", "").strip() or None
    except Exception:
        pass

    return None


async def _resolve_peer_id() -> str:
    """Get the peer_id from the node's onboarded state.

    After a node runs `adk mesh onboard`, its peer_id is stored in the
    Strata peer record (the node's mesh identity). Real peer IDs have the format
    'peer-<12 hex characters>'.

    Resolution order:
      1. AITHER_PEER_ID environment variable (if set)
      2. ~/.aither/node_auth.json peer_id (set by onboard flow)
      3. Raise a clear error instructing the user to onboard or set AITHER_PEER_ID

    Never fabricates a peer ID from overlay IP or other sources.
    """
    # Try environment first
    peer_id = os.getenv("AITHER_PEER_ID", "").strip()
    if peer_id:
        return peer_id

    # Try adk node_auth.json (set by onboard flow)
    try:
        from pathlib import Path
        node_auth_file = Path.home() / ".aither" / "node_auth.json"
        if node_auth_file.exists():
            auth_data = json.loads(node_auth_file.read_text(encoding="utf-8"))
            peer_id = auth_data.get("peer_id", "").strip()
            if peer_id:
                return peer_id
    except Exception:
        pass

    # Fail-closed: do not fabricate. Real peer IDs look like 'peer-<12 hex>'.
    raise RuntimeError(
        "Cannot resolve peer_id. Run 'adk mesh onboard' first to register this node, "
        "or set AITHER_PEER_ID environment variable to the node's authenticated peer ID "
        "(format: peer-<12 hex characters>)."
    )


async def advertise(
    peer_id: str,
    inference_url: str,
    inference_model: str,
    strata_url: str = DEFAULT_STRATA_URL,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """Register the inference endpoint on the peer record.

    POST /strata/mesh/peers/inference with:
      - peer_id: the node's authenticated peer identity
      - inference_url: the OpenAI-compatible server URL
      - inference_model: model name (e.g., "gemma4-12b")

    Fail-closed: if the peer doesn't exist, the request 404s and we return error.
    Never trusts caller input for scope — the peer_id is authenticated at
    onboard time.
    """
    import httpx

    if not peer_id:
        return {
            "ok": False,
            "error": "peer_id required",
            "step": "advertise",
        }
    if not inference_url or not inference_model:
        return {
            "ok": False,
            "error": "inference_url and inference_model required",
            "step": "advertise",
        }

    url = strata_url.rstrip("/") + "/strata/mesh/peers/inference"
    payload = {
        "peer_id": peer_id,
        "inference_url": inference_url,
        "inference_model": inference_model,
    }
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=_tls_verify()) as c:
            r = await c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return {
                "ok": True,
                "step": "advertise",
                "peer_id": peer_id,
                "inference_url": inference_url,
                "inference_model": inference_model,
            }
    except httpx.HTTPStatusError as e:
        logger.error("advertise failed: %s %s", e.response.status_code, e.response.text)
        return {
            "ok": False,
            "error": f"advertise failed: {e.response.status_code}",
            "step": "advertise",
        }
    except Exception as e:
        logger.error("advertise error: %s", e)
        return {
            "ok": False,
            "error": f"advertise error: {e}",
            "step": "advertise",
        }


async def grant_consent(
    peer_id: str,
    tenant_id: str | None = None,
    conductor_url: str = DEFAULT_CONDUCTOR_URL,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """Grant participation consent for the node to serve community inference.

    POST /v1/mesh/peers/{peer}/consent with:
      - tenant_id: the owner's authenticated tenant (required for fail-closed gate)
      - granted: true

    Default-deny: without an explicit grant, the routing gate refuses.
    Tenant must be derived from the authenticated caller, never from the request.

    In the CLI context, the authenticated caller is the user providing a valid
    auth_token. The tenant_id is the user's authenticated owner identity, and
    Conductor validates the binding server-side. No token = deny.
    """
    import httpx

    if not peer_id:
        return {
            "ok": False,
            "error": "peer_id required",
            "step": "consent",
        }

    # Fail-closed: tenant_id is required and must match the caller's identity.
    # If not provided, we cannot proceed.
    if not tenant_id:
        return {
            "ok": False,
            "error": "tenant_id required (your authenticated owner identity)",
            "step": "consent",
        }

    # Fail-closed: auth_token is required for authentication.
    # Unauthenticated consent grants are denied.
    if not auth_token:
        return {
            "ok": False,
            "error": "auth_token required (authentication mandatory for consent grant)",
            "step": "consent",
        }

    url = conductor_url.rstrip("/") + f"/v1/mesh/peers/{peer_id}/consent"
    payload = {
        "tenant_id": tenant_id,
        "granted": True,
    }
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=_tls_verify()) as c:
            r = await c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return {
                "ok": True,
                "step": "consent",
                "peer_id": peer_id,
                "tenant_id": tenant_id,
            }
    except httpx.HTTPStatusError as e:
        logger.error("consent failed: %s %s", e.response.status_code, e.response.text)
        return {
            "ok": False,
            "error": f"consent failed: {e.response.status_code}",
            "step": "consent",
        }
    except Exception as e:
        logger.error("consent error: %s", e)
        return {
            "ok": False,
            "error": f"consent error: {e}",
            "step": "consent",
        }


async def poll_trust_tier(
    peer_id: str,
    strata_url: str = DEFAULT_STRATA_URL,
    auth_token: str | None = None,
    max_wait_seconds: int = 30,
) -> dict[str, Any]:
    """Poll for operator trust grant on the peer record.

    GET {strata}/strata/mesh/peers/{peer} and read the trust_level field
    (the authoritative shared peer record; a tenant-scoped mesh token may
    read only its OWN peer — Conductor has no peer GET route).

    Trust level:
      - 0 = guest (never routed, default)
      - 1 = trusted (eligible for routing)
      - 2 = verified

    Fail-closed: if trust is 0 or missing, the node is not yet eligible.
    This polls with a bounded timeout; the operator must manually promote
    the node (the node cannot self-promote).
    """
    import httpx

    if not peer_id:
        return {
            "ok": False,
            "error": "peer_id required",
            "step": "poll_trust",
        }

    url = strata_url.rstrip("/") + f"/strata/mesh/peers/{peer_id}"
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    start_time = time.time()
    poll_interval = 2.0  # seconds between polls

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            # Timeout: still guest (trust tier 0)
            return {
                "ok": False,
                "error": "operator trust grant timed out",
                "step": "poll_trust",
                "peer_id": peer_id,
                "waited_seconds": int(elapsed),
                "message": (
                    "Node is still in guest tier (0). An operator must promote it. "
                    "See the next step below for the operator command."
                ),
            }

        try:
            async with httpx.AsyncClient(timeout=5.0, verify=_tls_verify()) as c:
                r = await c.get(url, headers=headers)
                r.raise_for_status()
                peer_data = r.json()
                # Real wire shape: Strata returns trust_level (NOT trust_tier).
                try:
                    trust_level = int(peer_data.get("trust_level", 0))
                except (TypeError, ValueError):
                    trust_level = 0

                if trust_level >= 1:
                    # Promoted! Success.
                    return {
                        "ok": True,
                        "step": "poll_trust",
                        "peer_id": peer_id,
                        "trust_level": trust_level,
                    }
        except httpx.HTTPStatusError as e:
            logger.warning("poll_trust GET failed: %s", e.response.status_code)
        except Exception as e:
            logger.warning("poll_trust error: %s", e)

        # Poll interval: wait before retrying
        await asyncio.sleep(poll_interval)


async def provide(
    inference_url: str,
    inference_model: str,
    peer_id: str | None = None,
    tenant_id: str | None = None,
    wait_seconds: int = 30,
    strata_url: str | None = None,
    conductor_url: str | None = None,
    auth_token: str | None = None,
) -> dict[str, Any]:
    """One-command provider setup: advertise → consent → poll trust.

    Returns a comprehensive report with the status of each step and next
    actions (operator command) if applicable.

    Args:
        inference_url: OpenAI-compatible server URL (e.g., http://10.77.x.x:8000/v1)
        inference_model: Model name (e.g., "gemma4-12b")
        peer_id: Node's mesh peer ID (auto-resolved if not given)
        tenant_id: Owner's authenticated tenant (required for fail-closed gate)
        wait_seconds: Timeout for polling operator trust (default 30s)
        strata_url: Strata endpoint override
        conductor_url: Conductor endpoint override
        auth_token: Bearer token for API calls (auto-resolved if not given)

    Returns:
        dict with keys:
          - ok: True if all steps succeeded
          - steps: list of step results
          - next_action: operator command (if awaiting trust)
    """
    strata_url = strata_url or os.getenv("AITHER_STRATA_URL", DEFAULT_STRATA_URL)
    conductor_url = conductor_url or os.getenv(
        "AITHER_CONDUCTOR_URL", DEFAULT_CONDUCTOR_URL
    )
    auth_token = auth_token or _get_auth_token()

    # Fail-closed: auth_token is required for consent binding.
    # Without authentication, the provider cannot grant or revoke consent.
    if not auth_token:
        return {
            "ok": False,
            "error": (
                "auth_token required for consent binding. "
                "Set AITHER_AUTH_TOKEN environment variable or configure ~/.aither/config.json"
            ),
            "step": "init",
        }

    if not peer_id:
        try:
            peer_id = await _resolve_peer_id()
        except RuntimeError as e:
            return {
                "ok": False,
                "error": str(e),
                "step": "init",
            }

    # Fail-closed: tenant_id is required
    if not tenant_id:
        tenant_id = os.getenv("AITHER_TENANT_ID", "").strip()
    if not tenant_id:
        return {
            "ok": False,
            "error": (
                "tenant_id required (your authenticated owner identity). "
                "Set AITHER_TENANT_ID or pass --tenant-id."
            ),
            "step": "init",
        }

    report = {
        "ok": True,
        "steps": [],
        "peer_id": peer_id,
        "inference_url": inference_url,
        "inference_model": inference_model,
    }

    # Step 1: Advertise
    logger.info("advertise: peer=%s url=%s model=%s", peer_id, inference_url, inference_model)
    ad_result = await advertise(
        peer_id=peer_id,
        inference_url=inference_url,
        inference_model=inference_model,
        strata_url=strata_url,
        auth_token=auth_token,
    )
    report["steps"].append(ad_result)
    if not ad_result.get("ok"):
        report["ok"] = False
        return report

    # Step 2: Grant Consent
    logger.info("consent: peer=%s tenant=%s", peer_id, tenant_id)
    consent_result = await grant_consent(
        peer_id=peer_id,
        tenant_id=tenant_id,
        conductor_url=conductor_url,
        auth_token=auth_token,
    )
    report["steps"].append(consent_result)
    if not consent_result.get("ok"):
        report["ok"] = False
        return report

    # Step 3: Poll Trust (bounded timeout) — reads the authoritative Strata peer
    # record (Conductor has no peer GET route; the field is trust_level).
    logger.info("poll_trust: peer=%s wait=%ds", peer_id, wait_seconds)
    trust_result = await poll_trust_tier(
        peer_id=peer_id,
        strata_url=strata_url,
        auth_token=auth_token,
        max_wait_seconds=wait_seconds,
    )
    report["steps"].append(trust_result)

    if trust_result.get("ok"):
        # Trust granted! Ready for registration (operator-gated genesis call —
        # NOT something this node can do; see skill step 6).
        report["trust_granted"] = True
        report["next_action"] = (
            f"Operator: register the backend (can_execute-gated) with:\n"
            f"  POST http://localhost:8001/deploy/cloud-model/register-backend\n"
            f'  {{"name": "community_{peer_id.replace("-", "_")}", '
            f'"reach": "aithernet", "peer_id": "{peer_id}"}}'
        )
    else:
        # Still guest (trust 0) — awaiting operator promotion
        report["ok"] = False
        report["trust_granted"] = False
        report["awaiting_operator"] = True
        report["message"] = trust_result.get("message", "")
        report["next_action"] = (
            f"Operator (platform only): promote the node with:\n"
            f"  curl -X POST {conductor_url}/v1/mesh/peers/{peer_id}/trust-tier \\\n"
            f"    -H 'Content-Type: application/json' \\\n"
            f"    -d '{{\"trust_level\": 1}}'"
        )

    return report
