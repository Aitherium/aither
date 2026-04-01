"""Environment-based configuration for ADK.

Security boundary:
    LOCAL mode (localhost connections): No auth required. All services on the same
    machine trust each other via localhost binding (127.0.0.1). Docker services are
    only exposed on 127.0.0.1, not 0.0.0.0, so LAN peers cannot reach them.

    REMOTE/CLOUD mode: All requests carry Authorization: Bearer <AITHER_API_KEY>.
    The API key is stored in ~/.aither/config.json or AITHER_API_KEY env var.
    Cloud gateway (mcp.aitherium.com) enforces HMAC tenant isolation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("adk.config")


# ---------------------------------------------------------------------------
# ~/.aither/config.json helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH_JSON = Path.home() / ".aither" / "config.json"
_CONFIG_PATH_YAML = Path.home() / ".aither" / "config.yaml"
# Prefer YAML (shared with AitherShell), fall back to JSON (legacy)
_CONFIG_PATH = _CONFIG_PATH_YAML if _CONFIG_PATH_YAML.exists() else _CONFIG_PATH_JSON


def load_saved_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load persisted config from ``~/.aither/config.yaml`` or ``config.json``.

    Tries YAML first (shared with AitherShell), falls back to JSON (legacy).
    Returns an empty dict when the file does not exist or cannot be parsed.
    """
    # Try YAML first
    yaml_path = config_path or _CONFIG_PATH_YAML
    if yaml_path.exists() and yaml_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.debug("Failed to read YAML config from %s", yaml_path)

    # Fall back to JSON
    json_path = config_path or _CONFIG_PATH_JSON
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Failed to read JSON config from %s", json_path)
    return {}


def save_saved_config(data: dict[str, Any], config_path: Path | None = None) -> Path:
    """Merge *data* into the persisted ADK config and write it back.

    Creates ``~/.aither/`` if it does not exist.  Returns the path that was
    written for caller convenience.
    """
    path = config_path or _CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_saved_config(path)
    existing.update(data)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


@dataclass
class Config:
    """ADK configuration, populated from environment variables with sensible defaults.

    If AITHER_PROFILE is set (or auto-detected via AgentSetup), the hardware profile
    YAML is loaded and its model/limits settings are applied as defaults — env vars
    always override profile values.
    """

    # LLM backend: "ollama", "openai", "anthropic", "auto"
    llm_backend: str = field(default_factory=lambda: os.getenv("AITHER_LLM_BACKEND", "auto"))

    # Model selection (env vars override profile)
    model: str = field(default_factory=lambda: os.getenv("AITHER_MODEL", ""))
    small_model: str = field(default_factory=lambda: os.getenv("AITHER_SMALL_MODEL", ""))
    large_model: str = field(default_factory=lambda: os.getenv("AITHER_LARGE_MODEL", ""))

    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    # OpenAI-compatible
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # General API key (for gateway or fallback)
    aither_api_key: str = field(default_factory=lambda: os.getenv("AITHER_API_KEY", ""))

    # Server
    server_port: int = field(default_factory=lambda: int(os.getenv("AITHER_PORT", "8080")))
    server_host: str = field(default_factory=lambda: os.getenv("AITHER_HOST", "0.0.0.0"))

    # Phonehome (opt-in)
    phonehome_enabled: bool = field(
        default_factory=lambda: os.getenv("AITHER_PHONEHOME", "").lower() in ("true", "1", "yes")
    )
    gateway_url: str = field(
        default_factory=lambda: os.getenv("AITHER_GATEWAY_URL", "https://gateway.aitherium.com")
    )

    # Prefer local inference over gateway even when AITHER_API_KEY is set
    prefer_local: bool = field(
        default_factory=lambda: os.getenv("AITHER_PREFER_LOCAL", "").lower() in ("true", "1", "yes")
    )

    # Register agent with gateway on startup (opt-in)
    register_agent: bool = field(
        default_factory=lambda: os.getenv("AITHER_REGISTER_AGENT", "").lower() in ("true", "1", "yes")
    )

    # Tenant context (set by ``aither connect``, stored in ~/.aither/config.json)
    tenant_id: str = field(
        default_factory=lambda: os.getenv("AITHER_TENANT_ID", "")
    )

    # Data directory
    data_dir: str = field(
        default_factory=lambda: os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))
    )

    # Observability — AitherOS service URLs (auto-detected from localhost)
    chronicle_url: str = field(
        default_factory=lambda: os.getenv("AITHER_CHRONICLE_URL", "")
    )
    watch_url: str = field(
        default_factory=lambda: os.getenv("AITHER_WATCH_URL", "")
    )
    pulse_url: str = field(
        default_factory=lambda: os.getenv("AITHER_PULSE_URL", "")
    )

    # LLMFit sidecar — hardware-aware model scoring
    # If empty, the llmfit client auto-resolves via convention (port 8793)
    llmfit_url: str = field(
        default_factory=lambda: os.getenv("AITHER_LLMFIT_URL", "")
    )

    # JSON structured logging (default on)
    json_logging: bool = field(
        default_factory=lambda: os.getenv("AITHER_JSON_LOGGING", "true").lower() not in ("false", "0", "no")
    )

    # Hardware profile (auto-detected or set via AITHER_PROFILE)
    profile: str = field(default_factory=lambda: os.getenv("AITHER_PROFILE", ""))

    # Profile-derived settings (populated by from_profile/apply_profile)
    max_context: int = 0          # 0 = unlimited (let model decide)
    max_concurrent: int = 0       # 0 = unlimited
    profile_models: dict = field(default_factory=dict)  # {default, small, large, embedding, ...}

    @classmethod
    def from_env(cls) -> Config:
        """Create config from current environment variables.

        If AITHER_PROFILE is set, loads and applies the hardware profile.
        If not set, checks ~/.aither/detected_profile from a previous auto_setup().

        Also loads ``tenant_id`` and ``api_key`` from ``~/.aither/config.json``
        when those values are not already set via environment variables.
        """
        config = cls()
        if config.profile:
            config.apply_profile(config.profile)
        else:
            # Try auto-detected profile from previous setup run
            marker = Path(config.data_dir) / "detected_profile"
            if marker.exists():
                try:
                    detected = marker.read_text(encoding="utf-8").strip()
                    if detected:
                        config.apply_profile(detected)
                except Exception:
                    pass

        # Backfill from saved config.json (env vars always win)
        saved = load_saved_config()
        if not config.tenant_id and saved.get("tenant_id"):
            config.tenant_id = saved["tenant_id"]
        if not config.aither_api_key and saved.get("api_key"):
            config.aither_api_key = saved["api_key"]

        return config

    @classmethod
    def from_profile(cls, profile_name: str) -> Config:
        """Create config from a hardware profile name."""
        config = cls(profile=profile_name)
        config.apply_profile(profile_name)
        return config

    def apply_profile(self, profile_name: str) -> None:
        """Load a hardware profile YAML and apply its settings.

        Profile settings are defaults — env vars always win.
        Looks for profiles in: ./profiles/, package profiles/, ~/.aither/profiles/
        """
        try:
            import yaml
        except ImportError:
            logger.debug("PyYAML not installed, skipping profile load")
            return

        # Search paths for profile YAML
        search_dirs = [
            Path("profiles"),                                    # CWD
            Path(__file__).parent.parent / "profiles",           # package root
            Path(self.data_dir) / "profiles",                    # ~/.aither/profiles/
        ]

        profile_path = None
        for d in search_dirs:
            candidate = d / f"{profile_name}.yaml"
            if candidate.exists():
                profile_path = candidate
                break

        if not profile_path:
            logger.debug("Profile '%s' not found in %s", profile_name, [str(d) for d in search_dirs])
            return

        try:
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Failed to load profile %s: %s", profile_name, e)
            return

        self.profile = profile_name

        # Apply models (env vars override)
        models = data.get("models", {})
        self.profile_models = models
        if not self.model:
            # Use 'default' or 'chat' key from profile
            self.model = models.get("default", models.get("chat", ""))
        if not self.small_model:
            self.small_model = models.get("small", "")
        if not self.large_model:
            self.large_model = models.get("large", models.get("reasoning", ""))

        # Apply limits
        limits = data.get("limits", {})
        if not self.max_context:
            self.max_context = limits.get("max_context", 0)
        if not self.max_concurrent:
            self.max_concurrent = limits.get("max_concurrent", 0)

        logger.info("Applied profile '%s': model=%s, small=%s, large=%s, max_context=%d",
                     profile_name, self.model, self.small_model, self.large_model, self.max_context)

    def get_api_key(self) -> str:
        """Return the best available API key for the configured backend."""
        if self.llm_backend == "anthropic":
            return self.anthropic_api_key or self.aither_api_key
        if self.llm_backend == "openai":
            return self.openai_api_key or self.aither_api_key
        return self.aither_api_key or self.openai_api_key or self.anthropic_api_key

    def get_llmfit_client(self):
        """Create a LLMFitClient initialized with this config's llmfit_url.

        Returns None if the llmfit module isn't installed.
        """
        try:
            from adk.llmfit import get_llmfit
            return get_llmfit(base_url=self.llmfit_url or None)
        except ImportError:
            logger.debug("adk.llmfit module not available")
            return None
