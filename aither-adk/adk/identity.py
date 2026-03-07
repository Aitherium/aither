"""Identity loader — load agent personas from YAML files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("adk.identity")

# Bundled identities ship with the package
_IDENTITIES_DIR = Path(__file__).parent / "identities"


@dataclass
class Identity:
    """An agent identity loaded from YAML."""
    name: str
    role: str = "assistant"
    description: str = ""
    skills: list[str] = field(default_factory=list)
    effort_cap: int = 10
    system_prompt: str = ""

    # Spirit/personality
    core_trait: str = ""
    drive: str = ""
    temperament: str = ""

    # Will/autonomy
    priority: str = ""
    autonomy: str = "moderate"

    # Raw YAML data for extensibility
    raw: dict = field(default_factory=dict)

    def build_system_prompt(self) -> str:
        """Build a system prompt from this identity."""
        if self.system_prompt:
            return self.system_prompt

        parts = [f"You are {self.name}, an AI agent."]
        if self.description:
            parts.append(f"Role: {self.description}")
        if self.core_trait:
            parts.append(f"Core trait: {self.core_trait}")
        if self.drive:
            parts.append(f"Drive: {self.drive}")
        if self.temperament:
            parts.append(f"Temperament: {self.temperament}")
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}")
        return "\n".join(parts)


def load_identity(name: str, search_paths: list[Path] | None = None) -> Identity:
    """Load an identity by name from YAML files.

    Searches in order:
    1. Provided search_paths
    2. Current directory ./identities/
    3. Bundled package identities
    """
    paths_to_try = []
    if search_paths:
        for p in search_paths:
            paths_to_try.append(p / f"{name}.yaml")
            paths_to_try.append(p / f"{name}.yml")
    paths_to_try.append(Path("identities") / f"{name}.yaml")
    paths_to_try.append(_IDENTITIES_DIR / f"{name}.yaml")

    for path in paths_to_try:
        if path.exists():
            return _parse_identity(path)

    logger.warning(f"Identity '{name}' not found, using defaults")
    return Identity(name=name)


def list_identities(search_paths: list[Path] | None = None) -> list[str]:
    """List all available identity names."""
    names = set()
    dirs = [_IDENTITIES_DIR, Path("identities")]
    if search_paths:
        dirs.extend(search_paths)

    for d in dirs:
        if d.exists():
            for f in d.glob("*.yaml"):
                names.add(f.stem)
            for f in d.glob("*.yml"):
                names.add(f.stem)
    return sorted(names)


def _parse_identity(path: Path) -> Identity:
    """Parse a YAML identity file into an Identity object."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    spirit = data.get("spirit_snapshot", {})
    will = data.get("will_config", {})

    return Identity(
        name=data.get("name", path.stem),
        role=data.get("role", "assistant"),
        description=data.get("description", ""),
        skills=data.get("skills", []),
        effort_cap=data.get("effort_cap", 10),
        system_prompt=data.get("system_prompt", ""),
        core_trait=spirit.get("core_trait", ""),
        drive=spirit.get("drive", ""),
        temperament=spirit.get("temperament", ""),
        priority=will.get("priority", ""),
        autonomy=will.get("autonomy", "moderate"),
        raw=data,
    )
