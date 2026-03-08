"""Environment-based configuration for ADK."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("adk.config")


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
