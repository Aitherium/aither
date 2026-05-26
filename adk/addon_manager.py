"""Client-side addon lifecycle manager.

Manages self-hosted service addons (Qdrant, Knowledge-RAG, CodeGraph, etc.)
alongside customer AitherOS apps.  Supports three addon types:

- ``docker``   — managed container via docker-py
- ``process``  — local binary/script
- ``external`` — already-running service at a URL (customer BYO)

State is persisted to ``~/.aitheros/addons/state.json``.

Usage::

    mgr = AddonManager()
    await mgr.enable("qdrant")
    await mgr.enable("knowledge-rag")
    inv = mgr.get_inventory()          # for federation heartbeat
    await mgr.health_check_all()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

log = logging.getLogger("adk.addon_manager")

# ---------------------------------------------------------------------------
# Addon manifest loading
# ---------------------------------------------------------------------------

_MANIFEST_DIRS: List[str] = [
    # Inside AitherOS monorepo (Docker or dev)
    "/app/AitherOS/config/addon_manifests",
    # Relative to repo root on local dev
    os.path.join(os.path.dirname(__file__), "..", "..", "AitherOS", "config", "addon_manifests"),
    # Bundled with ADK package
    os.path.join(os.path.dirname(__file__), "addon_manifests"),
]


def _find_manifest_dir() -> Optional[Path]:
    for d in _MANIFEST_DIRS:
        p = Path(d)
        if p.is_dir() and (p / "_schema.yaml").exists():
            return p
    return None


def load_addon_manifest(addon_id: str) -> Optional[Dict[str, Any]]:
    """Load a single addon manifest by ID."""
    mdir = _find_manifest_dir()
    if not mdir:
        return None
    candidate = mdir / f"{addon_id}.yaml"
    if not candidate.is_file():
        return None
    with open(candidate, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("id") != addon_id:
        data["id"] = addon_id
    return data


def load_all_manifests() -> List[Dict[str, Any]]:
    """Load all addon manifests from the manifest directory."""
    mdir = _find_manifest_dir()
    if not mdir:
        return []
    manifests = []
    for p in sorted(mdir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if data.get("id"):
                manifests.append(data)
        except Exception as e:
            log.warning("Failed to load manifest %s: %s", p.name, e)
    return manifests


# ---------------------------------------------------------------------------
# AddonInstance dataclass
# ---------------------------------------------------------------------------

@dataclass
class AddonInstance:
    addon_id: str
    pack_id: str = ""
    status: str = "stopped"         # running | stopped | error | starting
    container_id: str = ""
    endpoint: str = ""
    health_ok: bool = False
    started_at: float = 0.0
    error_message: str = ""
    addon_type: str = "docker"      # docker | process | external

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AddonInstance":
        known = {
            "addon_id", "pack_id", "status", "container_id", "endpoint",
            "health_ok", "started_at", "error_message", "addon_type",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _state_path() -> Path:
    return Path.home() / ".aitheros" / "addons" / "state.json"


def _load_state() -> Dict[str, Dict[str, Any]]:
    sp = _state_path()
    if sp.is_file():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: Dict[str, Dict[str, Any]]) -> None:
    sp = _state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# AddonManager
# ---------------------------------------------------------------------------

class AddonManager:
    """Client-side addon lifecycle engine."""

    def __init__(self) -> None:
        self._state = _load_state()

    def _persist(self) -> None:
        _save_state(self._state)

    # -- enable / disable ----------------------------------------------------

    async def enable(
        self,
        addon_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> AddonInstance:
        """Enable an addon: pull image, start container, health-check."""
        manifest = load_addon_manifest(addon_id)
        if not manifest:
            raise ValueError(f"Unknown addon: {addon_id}. Run 'adk addon list' to see available addons.")

        addon_type = manifest.get("type", "docker")
        instance = AddonInstance(
            addon_id=addon_id,
            pack_id=manifest.get("pack_id", ""),
            addon_type=addon_type,
            status="starting",
            started_at=time.time(),
        )

        # Check dependencies first
        deps = manifest.get("dependencies", [])
        for dep_id in deps:
            dep_state = self._state.get(dep_id, {})
            if dep_state.get("status") != "running":
                log.info("Dependency %s not running, enabling first...", dep_id)
                await self.enable(dep_id)

        if addon_type == "docker":
            instance = await self._start_docker(manifest, instance, config)
        elif addon_type == "external":
            # External: just record the endpoint
            endpoint = (config or {}).get("endpoint", "")
            if not endpoint:
                port = manifest.get("default_port", 8000)
                endpoint = f"http://localhost:{port}"
            instance.endpoint = endpoint
            instance.status = "running"
        elif addon_type == "process":
            instance.endpoint = f"http://localhost:{manifest.get('default_port', 8000)}"
            instance.status = "running"

        # Health check
        try:
            instance.health_ok = await self._check_health(manifest, instance)
            if instance.health_ok:
                instance.status = "running"
        except Exception as e:
            log.warning("Health check failed for %s: %s", addon_id, e)
            instance.health_ok = False

        self._state[addon_id] = instance.to_dict()
        self._persist()
        log.info("Addon %s enabled: status=%s endpoint=%s", addon_id, instance.status, instance.endpoint)
        return instance

    async def disable(self, addon_id: str) -> None:
        """Stop and remove an addon."""
        state = self._state.get(addon_id)
        if not state:
            log.info("Addon %s not found in state", addon_id)
            return

        if state.get("addon_type") == "docker" and state.get("container_id"):
            try:
                from adk.addon_docker import stop_container
                stop_container(state["container_id"])
            except Exception as e:
                log.warning("Failed to stop container for %s: %s", addon_id, e)

        del self._state[addon_id]
        self._persist()
        log.info("Addon %s disabled", addon_id)

    # -- status / inventory --------------------------------------------------

    async def status(self, addon_id: Optional[str] = None) -> List[AddonInstance]:
        """Get status of one or all addons."""
        if addon_id:
            s = self._state.get(addon_id)
            if not s:
                return []
            return [AddonInstance.from_dict(s)]

        return [AddonInstance.from_dict(s) for s in self._state.values()]

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all enabled addons."""
        results: Dict[str, bool] = {}
        for addon_id, state in self._state.items():
            if state.get("status") not in ("running", "starting"):
                results[addon_id] = False
                continue
            manifest = load_addon_manifest(addon_id)
            if not manifest:
                results[addon_id] = False
                continue
            inst = AddonInstance.from_dict(state)
            try:
                ok = await self._check_health(manifest, inst)
                results[addon_id] = ok
                state["health_ok"] = ok
                if ok and state.get("status") == "starting":
                    state["status"] = "running"
            except Exception:
                results[addon_id] = False
                state["health_ok"] = False
        self._persist()
        return results

    def get_inventory(self) -> List[Dict[str, Any]]:
        """Return addon inventory for federation heartbeat."""
        inventory = []
        for addon_id, state in self._state.items():
            inventory.append({
                "addon_id": addon_id,
                "status": state.get("status", "unknown"),
                "endpoint": state.get("endpoint", ""),
                "health_ok": state.get("health_ok", False),
                "pack_id": state.get("pack_id", ""),
            })
        return inventory

    # -- secrets sync --------------------------------------------------------

    async def sync_secrets(self, addon_id: str, federation_client=None) -> Dict[str, str]:
        """Pull secrets from hub for an addon."""
        if not federation_client:
            return {}
        manifest = load_addon_manifest(addon_id)
        if not manifest:
            return {}
        required = manifest.get("federation", {}).get("secrets_required", [])
        if not required:
            return {}
        result = await federation_client.pull_addon_secrets(addon_id)
        return result.get("secrets", {})

    # -- Docker helpers ------------------------------------------------------

    async def _start_docker(
        self,
        manifest: Dict[str, Any],
        instance: AddonInstance,
        config: Optional[Dict[str, Any]] = None,
    ) -> AddonInstance:
        """Start a Docker container via docker compose.

        Generates (or reuses) a compose file with proper networking,
        volumes, and health checks, then runs ``docker compose up -d``
        for the target service.
        """
        import shutil
        import subprocess

        image = manifest.get("image", "")
        if not image:
            instance.status = "error"
            instance.error_message = f"No image defined for addon {manifest['id']}"
            return instance

        if not shutil.which("docker"):
            instance.status = "error"
            instance.error_message = "docker not found on PATH"
            return instance

        port = manifest.get("default_port", 8000)
        addon_id = manifest["id"]
        svc_name = f"aitheros-addon-{addon_id}"
        compose_file = Path.home() / ".aither" / "docker-compose.addons.yml"

        try:
            from adk.addon_compose import generate_addon_compose

            # Include this addon + its deps in the compose file
            deps = manifest.get("dependencies", [])
            all_ids = deps + [addon_id]
            overrides = {}
            if config and config.get("env"):
                overrides[addon_id] = {"env": config["env"]}
            generate_addon_compose(all_ids, compose_file, overrides=overrides)

            # Ensure the shared network exists
            subprocess.run(
                ["docker", "network", "create", "aitheros_default"],
                capture_output=True,
            )

            # Pull + start just the target service (deps start via depends_on)
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "pull", svc_name],
                capture_output=True, timeout=300,
            )
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d", svc_name],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or f"compose up failed (rc={result.returncode})")

            # Grab the container ID
            id_result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file),
                 "ps", "-q", svc_name],
                capture_output=True, text=True, timeout=10,
            )
            instance.container_id = id_result.stdout.strip()[:12] if id_result.stdout else ""
            instance.endpoint = f"http://localhost:{port}"
            instance.status = "starting"

        except Exception as e:
            instance.status = "error"
            instance.error_message = str(e)
            log.error("Failed to start %s via compose: %s", addon_id, e)

        return instance

    async def _check_health(
        self, manifest: Dict[str, Any], instance: AddonInstance
    ) -> bool:
        """Check health endpoint of an addon."""
        hc = manifest.get("health_check", {})
        path = hc.get("path", "/health")
        endpoint = instance.endpoint
        if not endpoint:
            return False
        url = f"{endpoint}{path}"
        try:
            async with httpx.AsyncClient(timeout=5, verify=os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false") as client:
                resp = await client.get(url)
                return resp.status_code < 400
        except Exception:
            return False
