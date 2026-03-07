"""Environment-based configuration for ADK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """ADK configuration, populated from environment variables with sensible defaults."""

    # LLM backend: "ollama", "openai", "anthropic", "auto"
    llm_backend: str = field(default_factory=lambda: os.getenv("AITHER_LLM_BACKEND", "auto"))

    # Model selection
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

    @classmethod
    def from_env(cls) -> Config:
        """Create config from current environment variables."""
        return cls()

    def get_api_key(self) -> str:
        """Return the best available API key for the configured backend."""
        if self.llm_backend == "anthropic":
            return self.anthropic_api_key or self.aither_api_key
        if self.llm_backend == "openai":
            return self.openai_api_key or self.aither_api_key
        return self.aither_api_key or self.openai_api_key or self.anthropic_api_key
