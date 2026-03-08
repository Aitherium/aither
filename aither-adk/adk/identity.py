"""Identity loader — load agent personas from YAML files.

Enhanced with:
  - Skill manifests: structured capability declarations per identity
  - A2A protocol support: /.well-known/agent.json generation
  - Capability requirements for sandbox enforcement
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("adk.identity")

# Bundled identities ship with the package
_IDENTITIES_DIR = Path(__file__).parent / "identities"


# ─────────────────────────────────────────────────────────────────────────────
# Skill Manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillManifest:
    """Structured skill declaration for an agent identity.

    Structured skill declaration for sandbox and A2A integration.
    Declares what the agent CAN do, what it REQUIRES, and constraints.
    """
    name: str = ""
    description: str = ""
    capabilities_required: list[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: list[Dict[str, str]] = field(default_factory=list)
    max_tokens: int = 0
    timeout_seconds: float = 60.0
    tags: list[str] = field(default_factory=list)


@dataclass
class Identity:
    """An agent identity loaded from YAML.

    Enhanced with:
      - skills_manifest: structured skill declarations
      - a2a_card: A2A protocol agent card data
      - capabilities: sandbox capabilities this identity needs
      - version: identity schema version for forward compat
    """
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

    # Skill manifests
    skills_manifest: list[SkillManifest] = field(default_factory=list)

    # A2A protocol support
    a2a_card: Dict[str, Any] = field(default_factory=dict)

    # Sandbox capabilities this identity requires
    capabilities: list[str] = field(default_factory=list)

    # Schema version
    version: str = "1.0"

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

    def to_a2a_card(self, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Generate an A2A protocol agent card (/.well-known/agent.json).

        Follows Google A2A protocol spec with AitherOS extensions.
        """
        if self.a2a_card:
            return self.a2a_card

        skills_section = []
        for sm in self.skills_manifest:
            skill_entry = {
                "id": sm.name.lower().replace(" ", "_"),
                "name": sm.name or self.name,
                "description": sm.description or self.description,
                "tags": sm.tags or self.skills,
            }
            if sm.input_schema:
                skill_entry["inputModes"] = ["text/plain", "application/json"]
            if sm.output_schema:
                skill_entry["outputModes"] = ["text/plain", "application/json"]
            if sm.examples:
                skill_entry["examples"] = sm.examples
            skills_section.append(skill_entry)

        # Fallback: create skills from basic identity info
        if not skills_section and self.skills:
            for skill_name in self.skills[:5]:
                skills_section.append({
                    "id": skill_name.lower().replace(" ", "_"),
                    "name": skill_name,
                    "description": f"{self.name} skill: {skill_name}",
                    "tags": [skill_name],
                })

        return {
            "name": self.name,
            "description": self.description or f"AI agent: {self.name}",
            "url": base_url,
            "version": self.version,
            "provider": {
                "organization": "Aitherium",
                "url": "https://aitherium.com",
            },
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "authentication": {
                "schemes": ["bearer"],
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": skills_section,
        }

    def to_skill_manifest_yaml(self) -> str:
        """Export skills as YAML manifest."""
        manifests = []
        for sm in self.skills_manifest:
            entry = {
                "name": sm.name,
                "description": sm.description,
                "capabilities_required": sm.capabilities_required,
                "tags": sm.tags,
            }
            if sm.input_schema:
                entry["input_schema"] = sm.input_schema
            if sm.output_schema:
                entry["output_schema"] = sm.output_schema
            if sm.max_tokens:
                entry["max_tokens"] = sm.max_tokens
            if sm.timeout_seconds != 60.0:
                entry["timeout_seconds"] = sm.timeout_seconds
            manifests.append(entry)
        return yaml.dump({"skills": manifests}, default_flow_style=False)


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
    """Parse a YAML identity file into an Identity object.

    Supports new fields:
      - skills_manifest: list of structured skill declarations
      - a2a_card: pre-built A2A agent card
      - capabilities: sandbox capabilities
      - version: identity schema version
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    spirit = data.get("spirit_snapshot", {})
    will = data.get("will_config", {})

    # Parse skill manifests
    skills_manifest = []
    for sm_data in data.get("skills_manifest", []):
        skills_manifest.append(SkillManifest(
            name=sm_data.get("name", ""),
            description=sm_data.get("description", ""),
            capabilities_required=sm_data.get("capabilities_required", []),
            input_schema=sm_data.get("input_schema", {}),
            output_schema=sm_data.get("output_schema", {}),
            examples=sm_data.get("examples", []),
            max_tokens=sm_data.get("max_tokens", 0),
            timeout_seconds=sm_data.get("timeout_seconds", 60.0),
            tags=sm_data.get("tags", []),
        ))

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
        skills_manifest=skills_manifest,
        a2a_card=data.get("a2a_card", {}),
        capabilities=data.get("capabilities", []),
        version=data.get("version", "1.0"),
        raw=data,
    )
