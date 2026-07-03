"""Rich endpoint enrollment — register this workstation with AitherIdentity.

This is the convergence point for the self-hosted managed-agent experience: the
customer's box probes its own hardware + inference readiness and registers with
AitherIdentity's node spine (``POST /v1/nodes/register``), so the portal can show
the node (GPU, models, last-seen) and steer per-agent routing. A 60s heartbeat keeps
it live.

Nodes are stored in AitherDirectory as aitherDevice entries, tenant-scoped.

Distinct from the legacy ``adk/fleet_enroll.py`` path (``FederationLiteClient`` →
``/federation/register``), which only carries agents, not hardware. ``fleet_enroll``
now calls :func:`rich_enroll` first and falls back to the federation path.

Everything here is best-effort: a failure logs and returns a non-enrolled result.
It MUST never raise into the caller — enrollment is optional and must not block
``adk start``.
"""

from __future__ import annotations

__all__ = ["rich_enroll", "build_registration", "heartbeat_loop"]

import asyncio
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger("adk.enrollment")

_AITHER_DIR = Path.home() / ".aither"
_WORKSPACE_FILE = _AITHER_DIR / "workspace.json"

# Default local inference ports probed to populate available_models.
_OLLAMA_URL = "http://localhost:11434"
_VLLM_URL = "http://localhost:8120"


def _probe_local_models() -> tuple[list[str], bool, bool]:
    """Best-effort probe of local inference backends.

    Returns ``(available_models, ollama_available, vllm_available)``. Uses short
    timeouts and stdlib urllib so a cold box returns fast without extra deps.
    """
    import urllib.error
    import urllib.request

    models: list[str] = []
    ollama = False
    vllm = False

    try:
        with urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
            ollama = True
            for m in data.get("models", []) or []:
                name = m.get("name") or m.get("model")
                if name:
                    models.append(str(name))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        pass

    try:
        with urllib.request.urlopen(f"{_VLLM_URL}/v1/models", timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8"))
            vllm = True
            for m in data.get("data", []) or []:
                mid = m.get("id")
                if mid:
                    models.append(str(mid))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        pass

    # de-dup, preserve order
    seen: set[str] = set()
    deduped = [m for m in models if not (m in seen or seen.add(m))]
    return deduped, ollama, vllm


def build_registration(node_id: str) -> Dict[str, Any]:
    """Probe hardware + local inference and build the EndpointRegistration payload.

    Reuses :func:`adk.hardware_probe.detect_system` for the hardware fields so the
    enrollment view matches what the first-run wizard detected.
    """
    try:
        from adk.hardware_probe import detect_system

        sysinfo = detect_system()
        ram_mb = int(round(sysinfo.ram_gb * 1024))
        cpu_count = int(sysinfo.cpu_cores or 0)
        gpu_name = sysinfo.gpu_name or ""
        gpu_vram_mb = int(sysinfo.gpu_vram_mb or 0)
        py_version = sysinfo.python_version or platform.python_version()
    except Exception as e:  # hardware probe is best-effort
        log.debug("Hardware probe failed, using minimal info: %s", e)
        ram_mb = cpu_count = gpu_vram_mb = 0
        gpu_name = ""
        py_version = platform.python_version()

    models, ollama_available, vllm_available = _probe_local_models()
    inference_ready = bool(models) or ollama_available or vllm_available

    return {
        "node_id": node_id,
        "hostname": platform.node(),
        "platform": platform.system(),
        "platform_version": platform.release(),
        "python_version": py_version,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "cpu_count": cpu_count,
        "ram_mb": ram_mb,
        "available_models": models,
        "capabilities": ["code_search", "memory", "file_tools"],
        "ollama_available": ollama_available,
        "vllm_available": vllm_available,
        "inference_ready": inference_ready,
    }


def _save_workspace(workspace: Dict[str, Any]) -> None:
    """Persist the returned workspace for local routing.

    Stores ONLY the non-sensitive routing fields — never ``settings``, which the
    server may populate with provider API-key material. Local routing needs only
    the roster + routing map; persisting secrets to a plaintext file would be a
    credential-at-rest leak. The file is also locked to 0600.
    """
    safe = {
        "workspace_id": workspace.get("workspace_id", ""),
        "name": workspace.get("name", ""),
        "tier": workspace.get("tier", ""),
        "agent_roster": workspace.get("agent_roster", []),
        "agent_routing": workspace.get("agent_routing", {}),
    }
    try:
        _AITHER_DIR.mkdir(parents=True, exist_ok=True)
        _WORKSPACE_FILE.write_text(json.dumps(safe, indent=2), encoding="utf-8")
        try:
            _WORKSPACE_FILE.chmod(0o600)
        except OSError:
            pass  # best-effort on platforms without POSIX perms
    except OSError as e:
        log.debug("Failed to persist workspace.json: %s", e)


async def heartbeat_loop(
    base_url: str,
    token: str,
    node_id: str,
    *,
    interval: int = 60,
) -> None:
    """Background heartbeat — POST /v1/nodes/heartbeat every ``interval`` seconds.

    Re-registers (full payload) if the server reports the node as unknown (e.g.
    after the registry was reset). Failures are logged but never break the loop.
    """
    import httpx

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = base_url.rstrip("/")
    log.info("Starting endpoint heartbeat loop (interval=%ds)", interval)
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            log.info("Heartbeat loop cancelled")
            break
        try:
            reg = build_registration(node_id)
            hb = {
                "node_id": node_id,
                "inference_ready": reg["inference_ready"],
                "available_models": reg["available_models"],
                "gpu_vram_mb": reg["gpu_vram_mb"],
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{base}/v1/nodes/heartbeat", json=hb, headers=headers
                )
                if resp.status_code == 200 and resp.json().get("status") == "unknown_node":
                    # Registry lost us — re-register with the full payload.
                    await client.post(
                        f"{base}/v1/nodes/register", json=reg, headers=headers
                    )
        except asyncio.CancelledError:
            log.info("Heartbeat loop cancelled")
            break
        except Exception as e:
            log.debug("Heartbeat error: %s", e)


async def _request_device_cert(
    base_url: str,
    token: str,
    node_id: str,
    tenant_id: str,
) -> Dict[str, Any]:
    """Request a device mTLS client cert from the identity service.

    After successful node registration, the client can request a client cert
    for mTLS authentication to AitherOS services. The cert is bound to the
    tenant+node (CN = devcert--<tenant>--<node>) so the cloud derives the
    tenant cryptographically instead of from a spoofable header.

    Args:
        base_url: Control-plane base (Identity service)
        token: Bearer token from the enrollment response
        node_id: Node identifier
        tenant_id: Tenant ID from the enrollment response

    Returns:
        ``{"success": bool, "mtls": {...}, "error": str}`` — never raises.
        On success, mtls is ``{certificate, private_key, chain}``.
    """
    if not token or not tenant_id:
        return {"success": False, "error": "missing token or tenant_id"}

    try:
        import httpx

        from adk.sync import device_identity

        base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"node_id": node_id, "tenant_id": tenant_id}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{base}/v1/nodes/mtls-cert", json=payload, headers=headers
            )

        if resp.status_code != 200:
            detail = resp.text[:200]
            log.warning("mtls-cert request HTTP %s: %s", resp.status_code, detail)
            return {"success": False, "error": f"HTTP {resp.status_code}: {detail}"}

        data = resp.json()
        mtls_bundle = data.get("mtls", {}) or {}
        if not mtls_bundle or not mtls_bundle.get("certificate"):
            log.debug("No device cert in response (best-effort; device will use bearer-token auth)")
            return {"success": False, "error": "no cert in response"}

        # Persist the cert locally
        try:
            device_identity.save_enrolled_identity(mtls_bundle)
            log.info("Device mTLS cert enrolled and persisted")
            return {"success": True, "mtls": mtls_bundle}
        except Exception as e:
            log.warning("Failed to persist device cert: %s", e)
            return {"success": False, "error": f"cert persistence failed: {e}"}

    except Exception as e:
        log.debug("Device cert request failed (best-effort): %s", e)
        return {"success": False, "error": str(e)}


async def rich_enroll(
    base_url: str,
    token: str,
    node_id: str,
    *,
    enable_heartbeat: bool = True,
) -> Dict[str, Any]:
    """Register this workstation with the rich endpoint spine.

    Args:
        base_url: Control-plane base (Identity service) exposing ``/v1/nodes/*``.
        token: Bearer token from ``adk login`` (caller→tenant on the server).
        node_id: Stable node identifier.
        enable_heartbeat: Start the background heartbeat loop on success.

    Returns:
        ``{"enrolled": bool, "node_id": str, "workspace": dict, ...}``. On any
        failure, ``{"enrolled": False, "error": str}`` — never raises.
    """
    if not token:
        return {"enrolled": False, "error": "no auth token (run `adk login`)"}

    try:
        import httpx

        reg = build_registration(node_id)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        base = base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base}/v1/nodes/register", json=reg, headers=headers
            )
        if resp.status_code != 200:
            detail = resp.text[:200]
            log.warning("Endpoint register HTTP %s: %s", resp.status_code, detail)
            return {"enrolled": False, "error": f"HTTP {resp.status_code}: {detail}"}

        data = resp.json()
        workspace = data.get("workspace", {}) or {}
        _save_workspace(workspace)

        # After successful registration, request a device client cert (best-effort).
        tenant_id = data.get("tenant_id", "")
        cert_result = await _request_device_cert(base, token, node_id, tenant_id)
        cert_enrolled = cert_result.get("success", False)

        if enable_heartbeat:
            try:
                asyncio.create_task(heartbeat_loop(base, token, node_id))
            except RuntimeError:
                # No running loop (sync context) — caller can start it later.
                log.debug("No event loop for heartbeat; skipping background task")

        log.info("Enrolled endpoint %s (tenant=%s, cert_enrolled=%s)",
                 node_id, tenant_id, cert_enrolled)
        return {
            "enrolled": True,
            "node_id": node_id,
            "tenant_id": tenant_id,
            "workspace_id": data.get("workspace_id", ""),
            "workspace": workspace,
            "registration": reg,
            "cert_enrolled": cert_enrolled,
            "_enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Tenant-scoped capability token (identity_nodes.py's /v1/nodes/register
            # now mints one) — lets this node self-service its OWN gateway API key
            # afterward instead of reusing the enrolling user's own access token.
            "bearer_token": data.get("bearer_token", ""),
        }
    except Exception as e:
        log.warning("Rich enrollment failed: %s", e)
        return {"enrolled": False, "error": str(e)}
