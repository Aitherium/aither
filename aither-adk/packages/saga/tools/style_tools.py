"""Style Configuration Tools — Set genre, voice, prose style, and lore rules.

Exposes the multi-layer style system as tools the LLM can call to
dynamically adjust narrative presentation.
"""

from __future__ import annotations

from adk.tools import tool


def _get_style():
    from saga_engine.style import get_active_style
    return get_active_style()


def _get_memory():
    from .story_turn import _get_engine
    _, memory, _ = _get_engine()
    return memory


def _get_graph():
    from .story_turn import _get_engine
    graph, _, _ = _get_engine()
    return graph


@tool(
    name="set_genre",
    description=(
        "Set the story genre preset. Available: fantasy_epic, dark_fantasy, romance, "
        "litrpg, scifi, cyberpunk, mystery, horror, historical, isekai. "
        "This changes tone, magic level, tech level, and narrative voice."
    ),
)
def set_genre(genre: str) -> dict:
    """Set the genre preset.

    Args:
        genre: Genre key (fantasy_epic, dark_fantasy, scifi, cyberpunk, etc.)
    """
    from saga_engine.style import get_genre_preset, get_active_style

    preset = get_genre_preset(genre)
    if not preset:
        available = [
            "fantasy_epic", "dark_fantasy", "romance", "litrpg", "scifi",
            "cyberpunk", "mystery", "horror", "historical", "isekai",
        ]
        return {"error": f"Unknown genre '{genre}'", "available": available}

    style = get_active_style()
    style.genre = preset
    # Auto-set voice to match genre default
    if preset.default_voice:
        style.voice.style = preset.default_voice

    return {
        "genre": preset.name,
        "tone": preset.tone,
        "magic": preset.magic_level,
        "tech": preset.tech_level,
        "voice": style.voice.style,
        "starter": preset.starter_prompt[:100] if preset.starter_prompt else "",
    }


@tool(
    name="set_voice",
    description=(
        "Set the narrative voice style. Options: epic, intimate, mysterious, "
        "humorous, dark, sardonic, literary."
    ),
)
def set_voice(
    voice: str,
    sentence_length: str = "",
    vocabulary: str = "",
    metaphor_density: str = "",
    dialogue_ratio: str = "",
) -> dict:
    """Set narrative voice parameters.

    Args:
        voice: Voice style (epic, intimate, mysterious, humorous, dark, sardonic, literary)
        sentence_length: Sentence pattern (short, medium, long, varied)
        vocabulary: Vocabulary level (simple, moderate, rich, archaic, technical)
        metaphor_density: Metaphor usage (sparse, moderate, dense)
        dialogue_ratio: Dialogue balance (minimal, balanced, heavy)
    """
    style = _get_style()

    valid_voices = {"epic", "intimate", "mysterious", "humorous", "dark", "sardonic", "literary"}
    if voice not in valid_voices:
        return {"error": f"Unknown voice '{voice}'", "available": sorted(valid_voices)}

    style.voice.style = voice
    if sentence_length:
        style.voice.sentence_length = sentence_length
    if vocabulary:
        style.voice.vocabulary_level = vocabulary
    if metaphor_density:
        style.voice.metaphor_density = metaphor_density
    if dialogue_ratio:
        style.voice.dialogue_ratio = dialogue_ratio

    return {
        "voice": style.voice.style,
        "sentence_length": style.voice.sentence_length,
        "vocabulary": style.voice.vocabulary_level,
        "metaphors": style.voice.metaphor_density,
        "dialogue": style.voice.dialogue_ratio,
    }


@tool(
    name="set_prose_style",
    description=(
        "Set prose writing style. Options: descriptive, action-focused, "
        "dialogue-heavy, literary, pulp."
    ),
)
def set_prose_style(
    style_name: str,
    pacing: str = "",
    pov: str = "",
    tense: str = "",
) -> dict:
    """Set prose style parameters.

    Args:
        style_name: Prose style (descriptive, action-focused, dialogue-heavy, literary, pulp)
        pacing: Story pacing (breakneck, fast, moderate, slow, contemplative)
        pov: Point of view (first, second, third_limited, third_omniscient)
        tense: Narrative tense (past, present)
    """
    style = _get_style()

    valid_styles = {"descriptive", "action-focused", "dialogue-heavy", "literary", "pulp"}
    if style_name not in valid_styles:
        return {"error": f"Unknown style '{style_name}'", "available": sorted(valid_styles)}

    style.prose.style = style_name
    if pacing:
        style.prose.pacing = pacing
    if pov:
        style.prose.pov = pov
    if tense:
        style.prose.tense = tense

    return {
        "style": style.prose.style,
        "pacing": style.prose.pacing,
        "pov": style.prose.pov,
        "tense": style.prose.tense,
    }


@tool(
    name="set_content_rating",
    description="Set content rating and boundaries (general, mature, explicit).",
)
def set_content_rating(
    rating: str = "mature",
    violence: str = "",
    romance: str = "",
    language: str = "",
) -> dict:
    """Set content rating.

    Args:
        rating: Overall rating (general, mature, explicit)
        violence: Violence level (none, mild, moderate, graphic)
        romance: Romance level (none, mild, moderate, explicit)
        language: Language level (clean, mild, moderate, strong)
    """
    style = _get_style()

    valid_ratings = {"general", "mature", "explicit"}
    if rating not in valid_ratings:
        return {"error": f"Unknown rating '{rating}'", "available": sorted(valid_ratings)}

    style.rating.rating = rating
    if violence:
        style.rating.violence_level = violence
    if romance:
        style.rating.romance_level = romance
    if language:
        style.rating.language_level = language

    return {
        "rating": style.rating.rating,
        "violence": style.rating.violence_level,
        "romance": style.rating.romance_level,
        "language": style.rating.language_level,
    }


@tool(
    name="set_mode",
    description=(
        "Set the interaction mode: narrator (omniscient storyteller), "
        "character (speak AS a character), gm (game master with dice), "
        "collaborative (co-authoring with the player)."
    ),
)
def set_mode(mode: str, character_name: str = "") -> dict:
    """Set interaction mode.

    Args:
        mode: Mode (narrator, character, gm, collaborative)
        character_name: If mode=character, which character to play as
    """
    from saga_engine.style import ModeOverride

    style = _get_style()

    valid_modes = {"narrator", "character", "gm", "collaborative"}
    if mode not in valid_modes:
        return {"error": f"Unknown mode '{mode}'", "available": sorted(valid_modes)}

    character_id = ""
    if mode == "character" and character_name:
        graph = _get_graph()
        node = graph.find_node_by_name(character_name)
        if node:
            character_id = node.id
        else:
            return {"error": f"Character '{character_name}' not found in world graph"}

    style.mode = ModeOverride(mode=mode, character_id=character_id)

    return {"mode": mode, "character": character_name or "N/A"}


@tool(
    name="add_lore_rule",
    description=(
        "Add an inviolable world rule to the lore bible. These rules are ALWAYS "
        "included in context and the narrative MUST respect them. "
        "Examples: 'Dragons are extinct', 'Magic costs life force', "
        "'The king is secretly a lich'."
    ),
)
def add_lore_rule(
    rule: str,
    category: str = "general",
    inviolable: bool = True,
) -> dict:
    """Add a world rule to the lore bible and store as pinned procedural memory.

    Args:
        rule: The world rule text
        category: Category (magic, physics, culture, history, character, general)
        inviolable: If True, this rule can never be broken
    """
    from saga_engine.style import LoreRule
    from saga_engine.models import MemoryType

    style = _get_style()
    memory = _get_memory()
    graph = _get_graph()

    # Add to style config
    lore = LoreRule(rule=rule, category=category, inviolable=inviolable)
    style.lore_rules.append(lore)

    # Also store as pinned procedural memory so it surfaces in context pipeline
    mem = memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=f"[WORLD RULE - {category.upper()}] {rule}",
        summary=f"World rule ({category}): {rule[:80]}",
        importance=1.0 if inviolable else 0.8,
        pinned=inviolable,
        turn_number=graph.world.turn_number,
        created_by="user:lore_bible",
    )
    memory.save()

    return {
        "rule": rule,
        "category": category,
        "inviolable": inviolable,
        "memory_id": mem.id,
        "total_rules": len(style.lore_rules),
    }


@tool(
    name="remove_lore_rule",
    description="Remove a world rule from the lore bible by its text (partial match).",
)
def remove_lore_rule(rule_text: str) -> dict:
    """Remove a lore rule.

    Args:
        rule_text: Text of the rule to remove (partial match)
    """
    style = _get_style()
    rule_lower = rule_text.lower()

    removed = []
    remaining = []
    for r in style.lore_rules:
        if rule_lower in r.rule.lower():
            removed.append(r.rule)
        else:
            remaining.append(r)

    style.lore_rules = remaining

    return {
        "removed": removed,
        "removed_count": len(removed),
        "remaining_rules": len(remaining),
    }


@tool(
    name="get_style_config",
    description="Get the current complete style configuration.",
)
def get_style_config() -> dict:
    """Get current style config."""
    style = _get_style()
    return style.to_dict()
