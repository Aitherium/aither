"""Saga Addon System — Plugin discovery and lifecycle management.

Addons extend Saga with new capabilities:
  - Simulation backends (Prometheus)
  - Asset generation (Iris for art/3D)
  - Voice/audio (TTS narration)
  - Export targets (ePub, Foundry VTT, etc.)
  - Custom tool modules

Discovery order:
  1. Python entry_points group "saga.addons" (pip-installed addons)
  2. ~/.saga/addons/*.yaml manifest files (local/dev addons)
  3. Built-in addons (shipped with Saga core)

Each addon implements the SagaAddon protocol and declares its capabilities
via an AddonManifest.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("saga.addons")

SAGA_HOME = Path.home() / ".saga"
ADDONS_DIR = SAGA_HOME / "addons"


class AddonCategory(str, Enum):
    SIMULATION = "simulation"     # World simulation backends (Prometheus)
    ART = "art"                   # Image/3D generation (Iris)
    VOICE = "voice"               # TTS / STT
    EXPORT = "export"             # Export format plugins
    TOOLS = "tools"               # Additional tool modules
    INTEGRATION = "integration"   # External service connectors


@dataclass
class AddonManifest:
    """Declares what an addon provides."""
    name: str
    version: str
    description: str
    category: AddonCategory
    author: str = "Aitherium"
    # What the addon provides
    simulation_backend: Optional[str] = None   # "module:ClassName"
    tool_modules: List[str] = field(default_factory=list)
    server_routers: List[str] = field(default_factory=list)  # FastAPI routers
    ui_panels: List[str] = field(default_factory=list)
    # Requirements
    requires: List[str] = field(default_factory=list)  # pip packages
    requires_gpu: bool = False
    min_vram_gb: float = 0
    # Pricing
    free: bool = False
    price: float = 0.0


class SagaAddon(Protocol):
    """Protocol that all addons implement."""
    manifest: AddonManifest

    def activate(self, saga_app: Any) -> None:
        """Called when addon is loaded. Register routes, backends, etc."""
        ...

    def deactivate(self) -> None:
        """Called on shutdown."""
        ...

    def health_check(self) -> dict:
        """Return addon status."""
        ...


@dataclass
class LoadedAddon:
    """A discovered and loaded addon."""
    manifest: AddonManifest
    instance: Optional[SagaAddon] = None
    active: bool = False
    error: Optional[str] = None


class AddonRegistry:
    """Discovers, loads, and manages Saga addons."""

    def __init__(self):
        self._addons: Dict[str, LoadedAddon] = {}
        self._simulation_backends: Dict[str, Any] = {}

    def discover(self) -> List[AddonManifest]:
        """Discover all available addons from entry_points and local manifests."""
        discovered = []

        # 1. Python entry_points (pip-installed addons)
        try:
            if hasattr(importlib.metadata, 'entry_points'):
                eps = importlib.metadata.entry_points()
                # Python 3.12+ returns a SelectableGroups
                if hasattr(eps, 'select'):
                    saga_eps = eps.select(group="saga.addons")
                elif isinstance(eps, dict):
                    saga_eps = eps.get("saga.addons", [])
                else:
                    saga_eps = [ep for ep in eps if ep.group == "saga.addons"]

                for ep in saga_eps:
                    try:
                        addon_cls = ep.load()
                        addon = addon_cls()
                        self._addons[addon.manifest.name] = LoadedAddon(
                            manifest=addon.manifest,
                            instance=addon,
                        )
                        discovered.append(addon.manifest)
                        logger.info("Discovered addon: %s (entry_point)", addon.manifest.name)
                    except Exception as e:
                        logger.warning("Failed to load addon %s: %s", ep.name, e)
        except Exception as e:
            logger.debug("entry_points discovery failed: %s", e)

        # 2. Local manifest files (~/.saga/addons/*.yaml)
        if ADDONS_DIR.exists():
            import yaml
            for manifest_path in ADDONS_DIR.glob("*.yaml"):
                try:
                    data = yaml.safe_load(manifest_path.read_text())
                    manifest = AddonManifest(
                        name=data["name"],
                        version=data.get("version", "0.0.0"),
                        description=data.get("description", ""),
                        category=AddonCategory(data.get("category", "tools")),
                        author=data.get("author", ""),
                        simulation_backend=data.get("simulation_backend"),
                        tool_modules=data.get("tool_modules", []),
                        server_routers=data.get("server_routers", []),
                        requires=data.get("requires", []),
                        requires_gpu=data.get("requires_gpu", False),
                        min_vram_gb=data.get("min_vram_gb", 0),
                        free=data.get("free", True),
                        price=data.get("price", 0),
                    )
                    if manifest.name not in self._addons:
                        self._addons[manifest.name] = LoadedAddon(manifest=manifest)
                        discovered.append(manifest)
                        logger.info("Discovered addon: %s (local manifest)", manifest.name)
                except Exception as e:
                    logger.warning("Failed to parse addon manifest %s: %s", manifest_path, e)

        # 3. Register built-in pseudo-addons
        self._register_builtins()

        return discovered

    def _register_builtins(self):
        """Register built-in capabilities as pseudo-addons for the UI."""
        builtins = [
            AddonManifest(
                name="saga-core",
                version="1.0.0",
                description="Core storytelling engine with StoryGraph, memory, and 101 creative tools",
                category=AddonCategory.TOOLS,
                free=True,
            ),
            AddonManifest(
                name="saga-simulation-local",
                version="1.0.0",
                description="Local simulation backend (Markov weather, NPC movement, faction AI)",
                category=AddonCategory.SIMULATION,
                free=True,
            ),
            AddonManifest(
                name="saga-nanogpt",
                version="1.0.0",
                description="Procedural asset generation (names, items, quests) via NanoGPT",
                category=AddonCategory.ART,
                free=True,
            ),
        ]
        for m in builtins:
            if m.name not in self._addons:
                self._addons[m.name] = LoadedAddon(manifest=m, active=True)

    def activate_all(self, saga_app: Any):
        """Activate all discovered addons."""
        for name, loaded in self._addons.items():
            if loaded.instance and not loaded.active:
                try:
                    loaded.instance.activate(saga_app)
                    loaded.active = True
                    logger.info("Activated addon: %s", name)

                    # Register simulation backend if provided
                    if loaded.manifest.simulation_backend:
                        self._register_simulation_backend(loaded)
                except Exception as e:
                    loaded.error = str(e)
                    logger.warning("Failed to activate addon %s: %s", name, e)

    def _register_simulation_backend(self, loaded: LoadedAddon):
        """Register a simulation backend from an addon."""
        spec = loaded.manifest.simulation_backend
        if not spec:
            return
        try:
            module_path, class_name = spec.rsplit(":", 1)
            module = importlib.import_module(module_path)
            backend_cls = getattr(module, class_name)
            self._simulation_backends[loaded.manifest.name] = backend_cls
            logger.info("Registered simulation backend: %s from %s",
                        class_name, loaded.manifest.name)
        except Exception as e:
            logger.warning("Failed to register simulation backend: %s", e)

    def get_simulation_backend(self, name: str = ""):
        """Get a simulation backend by addon name, or the first available."""
        if name:
            cls = self._simulation_backends.get(name)
            return cls() if cls else None
        # Return first available
        for cls in self._simulation_backends.values():
            return cls()
        return None

    def get_addon(self, name: str) -> Optional[LoadedAddon]:
        return self._addons.get(name)

    def list_addons(self) -> List[dict]:
        return [
            {
                "name": la.manifest.name,
                "version": la.manifest.version,
                "description": la.manifest.description,
                "category": la.manifest.category.value,
                "active": la.active,
                "error": la.error,
                "free": la.manifest.free,
                "price": la.manifest.price,
            }
            for la in self._addons.values()
        ]

    def list_available(self) -> List[dict]:
        """List addons available for purchase/install (not yet installed)."""
        # This would eventually hit an API. For now, static catalog.
        catalog = [
            {
                "name": "saga-prometheus",
                "version": "1.0.0",
                "description": "Full Prometheus world simulation — 39 game systems, MCTS faction AI, NPC lifecycle, weather Markov chains, economy, governance",
                "category": "simulation",
                "requires_gpu": False,
                "price": 0.0,
                "free": True,
                "installed": "saga-prometheus" in self._addons,
            },
            {
                "name": "saga-iris",
                "version": "1.0.0",
                "description": "AI art generation — character portraits, location art, item illustrations, 3D model generation via ComfyUI/Stable Diffusion",
                "category": "art",
                "requires_gpu": True,
                "min_vram_gb": 6,
                "price": 30.0,
                "free": False,
                "installed": "saga-iris" in self._addons,
            },
            {
                "name": "saga-voice",
                "version": "1.0.0",
                "description": "Character voice synthesis — unique TTS voices per character, narration, audiobook export",
                "category": "voice",
                "requires_gpu": True,
                "min_vram_gb": 4,
                "price": 20.0,
                "free": False,
                "installed": "saga-voice" in self._addons,
            },
            {
                "name": "saga-elysium",
                "version": "1.0.0",
                "description": "Cloud sync — share worlds, multiplayer co-storytelling, Aitherium account integration",
                "category": "integration",
                "requires_gpu": False,
                "price": 0.0,
                "free": True,
                "installed": "saga-elysium" in self._addons,
            },
        ]
        return catalog


# Module-level singleton
_registry: Optional[AddonRegistry] = None


def get_addon_registry() -> AddonRegistry:
    global _registry
    if _registry is None:
        _registry = AddonRegistry()
    return _registry
