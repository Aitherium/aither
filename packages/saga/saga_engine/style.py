"""
Style Configuration System — Multi-layer narrative style control.

7 layers, each overriding the one above:
    1. Genre Preset     — tone, magic_level, tech_level, starter_prompt
    2. Narrative Voice   — sentence patterns, vocabulary, metaphor density
    3. Content Rating    — violence, romance, language levels
    4. Prose Style       — description density, pacing, POV, tense
    5. Visual Style      — image prompt modifiers (for future art generation)
    6. Lore Bible        — user-defined world rules as inviolable memories
    7. Mode Override     — narrator/character/GM/collaborative
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenrePreset:
    """Layer 1: Genre foundation."""
    name: str = "Fantasy Epic"
    tone: str = "balanced"           # light, balanced, dark, grimdark
    magic_level: str = "high"        # none, low, moderate, high, epic
    tech_level: str = "medieval"     # primitive, medieval, renaissance, industrial, modern, futuristic
    default_voice: str = "epic"
    starter_prompt: str = ""


@dataclass
class NarrativeVoice:
    """Layer 2: How the prose sounds."""
    style: str = "epic"              # epic, intimate, mysterious, humorous, dark, sardonic, literary
    sentence_length: str = "varied"  # short, medium, long, varied
    vocabulary_level: str = "rich"   # simple, moderate, rich, archaic, technical
    metaphor_density: str = "moderate"  # sparse, moderate, dense
    dialogue_ratio: str = "balanced" # minimal, balanced, heavy


@dataclass
class ContentRating:
    """Layer 3: Content boundaries."""
    rating: str = "mature"           # general, mature, explicit
    violence_level: str = "moderate" # none, mild, moderate, graphic
    romance_level: str = "mild"      # none, mild, moderate, explicit
    language_level: str = "moderate"  # clean, mild, moderate, strong


@dataclass
class ProseStyle:
    """Layer 4: How the prose reads."""
    style: str = "descriptive"       # descriptive, action-focused, dialogue-heavy, literary, pulp
    description_density: str = "rich" # sparse, moderate, rich, lavish
    pacing: str = "moderate"          # breakneck, fast, moderate, slow, contemplative
    pov: str = "third_limited"        # first, second, third_limited, third_omniscient
    tense: str = "past"               # past, present


@dataclass
class VisualStyle:
    """Layer 5: Art generation preferences."""
    style: str = "painterly"         # anime, realistic, painterly, pixel, comic, sketch
    color_palette: str = "rich"      # muted, rich, neon, monochrome, pastel
    art_model: str = ""              # preferred image model


@dataclass
class LoreRule:
    """A single user-defined world rule (Layer 6)."""
    rule: str                        # The rule text
    category: str = "general"        # magic, physics, culture, history, character
    inviolable: bool = True          # If True, never break this rule


@dataclass
class ModeOverride:
    """Layer 7: Interaction mode."""
    mode: str = "narrator"           # narrator, character, gm, collaborative
    character_id: str = ""           # If mode=character, which character


@dataclass
class StyleConfig:
    """Complete style configuration with all layers merged."""
    genre: GenrePreset = field(default_factory=GenrePreset)
    voice: NarrativeVoice = field(default_factory=NarrativeVoice)
    rating: ContentRating = field(default_factory=ContentRating)
    prose: ProseStyle = field(default_factory=ProseStyle)
    visual: VisualStyle = field(default_factory=VisualStyle)
    lore_rules: List[LoreRule] = field(default_factory=list)
    mode: ModeOverride = field(default_factory=ModeOverride)

    def build_style_prompt(self) -> str:
        """Build the style section of the system prompt from active config."""
        parts = []

        # Genre
        parts.append(f"[GENRE] {self.genre.name}")
        parts.append(f"Tone: {self.genre.tone} | Magic: {self.genre.magic_level} | "
                      f"Technology: {self.genre.tech_level}")

        # Voice
        voice_desc = _VOICE_DESCRIPTIONS.get(self.voice.style, "")
        if voice_desc:
            parts.append(f"[NARRATIVE VOICE] {voice_desc}")
        parts.append(f"Sentences: {self.voice.sentence_length} | "
                      f"Vocabulary: {self.voice.vocabulary_level} | "
                      f"Metaphors: {self.voice.metaphor_density} | "
                      f"Dialogue: {self.voice.dialogue_ratio}")

        # Content rating
        parts.append(f"[CONTENT] Rating: {self.rating.rating} | "
                      f"Violence: {self.rating.violence_level} | "
                      f"Romance: {self.rating.romance_level}")

        # Prose style
        prose_desc = _PROSE_DESCRIPTIONS.get(self.prose.style, "")
        if prose_desc:
            parts.append(f"[PROSE STYLE] {prose_desc}")
        parts.append(f"Pacing: {self.prose.pacing} | POV: {self.prose.pov} | "
                      f"Tense: {self.prose.tense}")

        # Mode
        from prompts.modes import ROLEPLAY_MODES
        mode_info = ROLEPLAY_MODES.get(self.mode.mode, {})
        if mode_info:
            parts.append(f"[MODE] {mode_info.get('name', self.mode.mode)}: "
                          f"{mode_info.get('instruction', '')}")

        # Lore rules (inviolable)
        inviolable = [r for r in self.lore_rules if r.inviolable]
        if inviolable:
            parts.append("[WORLD RULES — INVIOLABLE]")
            for r in inviolable:
                parts.append(f"  - [{r.category.upper()}] {r.rule}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "genre": {
                "name": self.genre.name, "tone": self.genre.tone,
                "magic_level": self.genre.magic_level,
                "tech_level": self.genre.tech_level,
                "default_voice": self.genre.default_voice,
                "starter_prompt": self.genre.starter_prompt,
            },
            "voice": {
                "style": self.voice.style,
                "sentence_length": self.voice.sentence_length,
                "vocabulary_level": self.voice.vocabulary_level,
                "metaphor_density": self.voice.metaphor_density,
                "dialogue_ratio": self.voice.dialogue_ratio,
            },
            "rating": {
                "rating": self.rating.rating,
                "violence_level": self.rating.violence_level,
                "romance_level": self.rating.romance_level,
                "language_level": self.rating.language_level,
            },
            "prose": {
                "style": self.prose.style,
                "description_density": self.prose.description_density,
                "pacing": self.prose.pacing,
                "pov": self.prose.pov,
                "tense": self.prose.tense,
            },
            "visual": {
                "style": self.visual.style,
                "color_palette": self.visual.color_palette,
            },
            "lore_rules": [
                {"rule": r.rule, "category": r.category, "inviolable": r.inviolable}
                for r in self.lore_rules
            ],
            "mode": {"mode": self.mode.mode, "character_id": self.mode.character_id},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleConfig":
        """Deserialize from dict."""
        cfg = cls()
        if "genre" in data:
            cfg.genre = GenrePreset(**data["genre"])
        if "voice" in data:
            cfg.voice = NarrativeVoice(**data["voice"])
        if "rating" in data:
            cfg.rating = ContentRating(**data["rating"])
        if "prose" in data:
            cfg.prose = ProseStyle(**data["prose"])
        if "visual" in data:
            cfg.visual = VisualStyle(**{k: v for k, v in data["visual"].items()
                                        if k in ("style", "color_palette", "art_model")})
        if "lore_rules" in data:
            cfg.lore_rules = [LoreRule(**r) for r in data["lore_rules"]]
        if "mode" in data:
            cfg.mode = ModeOverride(**data["mode"])
        return cfg


# ── Voice descriptions for prompt injection ──

_VOICE_DESCRIPTIONS = {
    "epic": (
        "Write in an epic, sweeping voice. Grand scope, dramatic phrasing, "
        "a sense of history unfolding. Think Tolkien's narrator."
    ),
    "intimate": (
        "Write in a close, intimate voice. Focus on internal experience, "
        "sensory details, emotional undercurrents. Think literary fiction."
    ),
    "mysterious": (
        "Write with mystery and atmosphere. Hints and shadows, unanswered questions, "
        "a sense of something lurking beneath the surface."
    ),
    "humorous": (
        "Write with wit, irony, and comedic timing. Playful observations, "
        "unexpected turns of phrase, self-aware genre commentary."
    ),
    "dark": (
        "Write in a dark, intense voice. Raw emotion, moral ambiguity, "
        "unflinching portrayal of consequences. Think grimdark."
    ),
    "sardonic": (
        "Write with a sardonic, world-weary edge. Dry wit, cynical observations, "
        "characters who've seen too much. Think noir."
    ),
    "literary": (
        "Write with literary ambition. Careful prose, layered meaning, "
        "thematic depth. Every sentence earns its place."
    ),
}

_PROSE_DESCRIPTIONS = {
    "descriptive": (
        "Rich environmental detail. Paint the scene with sensory language — "
        "what's seen, heard, smelled, felt. The world is a character."
    ),
    "action-focused": (
        "Tight, kinetic prose. Short punchy sentences in action scenes. "
        "Focus on motion, impact, and consequence. Keep it moving."
    ),
    "dialogue-heavy": (
        "Drive the story through conversation. Characters reveal themselves "
        "through what they say and how they say it. Subtext matters."
    ),
    "literary": (
        "Careful, crafted prose. Varied rhythm, precise word choice, "
        "metaphor that illuminates rather than decorates."
    ),
    "pulp": (
        "Fast, fun, and vivid. Bold descriptions, exciting action, "
        "larger-than-life characters. Entertain first."
    ),
}


# ── Genre preset factory ──

def get_genre_preset(genre_key: str) -> Optional[GenrePreset]:
    """Get a genre preset by key. Returns None if not found."""
    from prompts.genre_presets import GENRE_PRESETS
    data = GENRE_PRESETS.get(genre_key)
    if not data:
        return None
    return GenrePreset(
        name=data.get("name", genre_key),
        tone=data.get("tone", "balanced"),
        magic_level=data.get("magic", "moderate"),
        tech_level=data.get("technology", "medieval"),
        default_voice=data.get("default_voice", "epic"),
        starter_prompt=data.get("starter", ""),
    )


# ── Module-level active config (one per process) ──

_active_style: Optional[StyleConfig] = None


def get_active_style() -> StyleConfig:
    """Get or create the active style config."""
    global _active_style
    if _active_style is None:
        _active_style = StyleConfig()
    return _active_style


def set_active_style(config: StyleConfig):
    """Set the active style config."""
    global _active_style
    _active_style = config
