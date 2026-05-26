"""Character Tools — Deep character interaction beyond CRUD.

Tools for character interviews, arc planning, relationship evolution,
dialogue style guides, motivation analysis, NPC generation, and more.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.character_tools")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _run_async(coro):
    from .story_turn import _run_async
    return _run_async(coro)


@tool(
    name="character_interview",
    description=(
        "Interview a character — Saga answers AS the character using their "
        "personality, knowledge, memories, and relationships. The character "
        "only knows what has been revealed to them in the story."
    ),
)
def character_interview(
    character_name: str,
    question: str,
) -> dict:
    """Set up a character interview with full context.

    Args:
        character_name: Name of the character to interview
        question: The question to ask them
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    # Gather character's complete knowledge
    props = node.properties
    relationships = []
    for neighbor, edge in graph.get_neighbors(node.id):
        relationships.append({
            "with": neighbor.name,
            "type": edge.type.value,
            "label": edge.label,
        })

    # Character's memories (things they've experienced)
    char_memories = memory.get_by_node(node.id)
    char_memories.sort(key=lambda m: m.turn_number, reverse=True)

    result = {
        "character": {
            "name": node.name,
            "description": node.description,
            "personality": props.get("personality", ""),
            "backstory": props.get("backstory", ""),
            "appearance": props.get("appearance", ""),
            "goals": props.get("goals", ""),
            "fears": props.get("fears", ""),
            "speech_style": props.get("speech_style", ""),
        },
        "relationships": relationships[:15],
        "recent_memories": [
            {"summary": m.summary, "type": m.type.value, "turn": m.turn_number}
            for m in char_memories[:10]
        ],
        "question": question,
    }

    # Pull emotional state + diary from simulation
    try:
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        if sim and graph.world.prometheus_world_id:
            dlg = _run_async(sim.get_dialogue_context(
                graph.world.prometheus_world_id, node.id
            ))
            if dlg:
                result["sim_emotional_state"] = dlg.get("emotional_state", "")
                result["sim_recent_events"] = dlg.get("recent_events", [])[:5]
    except Exception:
        pass

    emotion_note = ""
    if result.get("sim_emotional_state"):
        emotion_note = f" Current emotional state: {result['sim_emotional_state']}."

    result["instruction"] = (
        f"Answer the question AS {node.name}, in first person.{emotion_note} "
        f"Stay fully in character. Use their speech patterns, knowledge level, "
        f"and emotional state. They do NOT know things that haven't been "
        f"revealed to them in the story."
    )
    return result


@tool(
    name="character_arc_plan",
    description=(
        "Design a character's growth trajectory: starting state, catalyst, "
        "struggle, transformation, and new normal."
    ),
)
def character_arc_plan(
    character_name: str,
    arc_type: str = "growth",
) -> dict:
    """Plan a character arc.

    Args:
        character_name: Character to plan arc for
        arc_type: Arc type (growth, fall, redemption, corruption, revelation, steadfast)
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    props = node.properties
    char_memories = memory.get_by_node(node.id)

    return {
        "character": node.name,
        "current_state": {
            "personality": props.get("personality", ""),
            "goals": props.get("goals", ""),
            "fears": props.get("fears", ""),
            "backstory": props.get("backstory", ""),
        },
        "arc_type": arc_type,
        "history": [m.summary for m in char_memories[:8]],
        "instruction": (
            f"Design a {arc_type} arc for {node.name}. Structure: "
            f"1) Starting State (who they are now), "
            f"2) Catalyst (what disrupts their status quo), "
            f"3) Struggle (internal and external conflicts), "
            f"4) Transformation (the pivotal change), "
            f"5) New Normal (who they become). "
            f"Ground the arc in their established personality and history."
        ),
    }


@tool(
    name="relationship_evolve",
    description=(
        "Evolve the relationship between two characters based on recent events. "
        "Updates the graph edge weight and label."
    ),
)
def relationship_evolve(
    character_a: str,
    character_b: str,
    event_description: str,
    new_status: str = "",
) -> dict:
    """Evolve a relationship based on events.

    Args:
        character_a: First character name
        character_b: Second character name
        event_description: What happened between them
        new_status: New relationship status if changed (allies, enemies, lovers, strangers, etc.)
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType

    node_a = graph.find_node_by_name(character_a)
    node_b = graph.find_node_by_name(character_b)

    if not node_a:
        return {"error": f"Character '{character_a}' not found"}
    if not node_b:
        return {"error": f"Character '{character_b}' not found"}

    # Find existing edge
    existing_edge = None
    for neighbor, edge in graph.get_neighbors(node_a.id):
        if neighbor.id == node_b.id:
            existing_edge = edge
            break

    # Update or create the edge
    if existing_edge:
        updates = {}
        if new_status:
            updates["label"] = f"{node_a.name} and {node_b.name}: {new_status}"
        graph.update_node(existing_edge.id)  # Touch the edge
        result_status = "updated"
    else:
        from saga_engine.models import StoryEdge, EdgeType
        edge = StoryEdge(
            type=EdgeType.KNOWS,
            source_id=node_a.id,
            target_id=node_b.id,
            label=new_status or f"{node_a.name} knows {node_b.name}",
            bidirectional=True,
        )
        graph.add_edge(edge)
        result_status = "created"

    # Store the evolution as a memory
    mem = memory.create(
        memory_type=MemoryType.EPISODIC,
        content=f"Relationship change: {node_a.name} and {node_b.name} — {event_description}",
        summary=f"{node_a.name} & {node_b.name}: {event_description[:60]}",
        importance=0.7,
        related_nodes=[node_a.id, node_b.id],
        turn_number=graph.world.turn_number,
    )
    memory.save()
    graph.save()

    return {
        "character_a": node_a.name,
        "character_b": node_b.name,
        "event": event_description,
        "new_status": new_status,
        "edge_status": result_status,
        "memory_id": mem.id,
    }


@tool(
    name="dialogue_style_guide",
    description=(
        "Define or update a character's speech patterns: vocabulary, "
        "sentence length, verbal tics, accent markers, catchphrases."
    ),
)
def dialogue_style_guide(
    character_name: str,
    vocabulary: str = "",
    sentence_style: str = "",
    verbal_tics: str = "",
    accent_notes: str = "",
    catchphrase: str = "",
) -> dict:
    """Set a character's dialogue style.

    Args:
        character_name: Character to define style for
        vocabulary: Vocabulary level/type (simple, formal, archaic, technical, slang)
        sentence_style: How they construct sentences (short, complex, fragmented, poetic)
        verbal_tics: Repeated habits (says 'indeed' a lot, clears throat, trails off...)
        accent_notes: Accent or dialect notes
        catchphrase: A signature phrase
    """
    graph, _, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    speech_style = {}
    if vocabulary:
        speech_style["vocabulary"] = vocabulary
    if sentence_style:
        speech_style["sentence_style"] = sentence_style
    if verbal_tics:
        speech_style["verbal_tics"] = verbal_tics
    if accent_notes:
        speech_style["accent"] = accent_notes
    if catchphrase:
        speech_style["catchphrase"] = catchphrase

    # Merge with existing speech style
    existing = node.properties.get("speech_style", {})
    if isinstance(existing, str):
        existing = {"notes": existing}
    existing.update(speech_style)

    props = dict(node.properties)
    props["speech_style"] = existing
    graph.update_node(node.id, properties=props)
    graph.save()

    return {
        "character": node.name,
        "speech_style": existing,
    }


@tool(
    name="character_motivation",
    description=(
        "Analyze a character's motivation structure: wants (external goal), "
        "needs (internal growth), fears (obstacle), lie (false belief)."
    ),
)
def character_motivation(character_name: str) -> dict:
    """Analyze character motivation.

    Args:
        character_name: Character to analyze
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    props = node.properties
    char_memories = memory.get_by_node(node.id)

    # Get relationships for context
    relationships = []
    for neighbor, edge in graph.get_neighbors(node.id):
        relationships.append(f"{edge.type.value} {neighbor.name}: {edge.label}")

    return {
        "character": node.name,
        "personality": props.get("personality", ""),
        "backstory": props.get("backstory", ""),
        "goals": props.get("goals", ""),
        "fears": props.get("fears", ""),
        "relationships": relationships[:10],
        "history": [m.summary for m in char_memories[:8]],
        "instruction": (
            f"Analyze {node.name}'s motivation structure: "
            f"WANT (external, conscious goal — what they pursue), "
            f"NEED (internal, often unconscious — what they must learn), "
            f"FEAR (the obstacle or threat they avoid), "
            f"LIE (the false belief holding them back). "
            f"Ground each in their established personality and history."
        ),
    }


@tool(
    name="npc_generator",
    description=(
        "Generate a random NPC with name, appearance, personality, quirk, "
        "secret, and motivation. Automatically registers in the world graph."
    ),
)
def npc_generator(
    role: str = "commoner",
    location: str = "",
    faction: str = "",
) -> dict:
    """Generate and register a random NPC.

    Args:
        role: NPC role (commoner, merchant, guard, noble, scholar, rogue, healer, artisan)
        location: Location name to place them at
        faction: Faction name to affiliate them with
    """
    from saga_engine.models import StoryNode, StoryEdge, NodeType, EdgeType

    graph, _, _ = _get_engine()

    # Context for generation
    existing_chars = [n.name for n in graph.find_nodes_by_type(NodeType.CHARACTER)]
    existing_factions = [n.name for n in graph.find_nodes_by_type(NodeType.FACTION)]

    location_node = graph.find_node_by_name(location) if location else None
    faction_node = graph.find_node_by_name(faction) if faction else None

    result = {
        "role": role,
        "location": location_node.name if location_node else location,
        "faction": faction_node.name if faction_node else faction,
        "existing_characters": existing_chars[:20],
        "existing_factions": existing_factions,
        "world_tone": graph.world.mood,
    }

    # NanoGPT-generated name + visual profile
    try:
        from saga_engine.nanogpt_gen import get_asset_generator
        from .story_turn import _get_data_dir
        gen = get_asset_generator(_get_data_dir())
        names = _run_async(gen.generate_name(count=3, graph=graph))
        if names:
            # Filter out existing names
            existing_lower = {n.lower() for n in existing_chars}
            available = [n for n in names if n.lower() not in existing_lower]
            result["suggested_names"] = available[:3] if available else names[:3]
        visual = _run_async(gen.generate_visual_profile(graph=graph))
        if visual:
            result["visual_profile"] = visual
    except Exception:
        pass

    name_hint = ""
    if result.get("suggested_names"):
        name_hint = f" Suggested names: {', '.join(result['suggested_names'])}."

    result["instruction"] = (
        f"Generate a {role} NPC.{name_hint} Provide: "
        f"1) Name (fitting the world), "
        f"2) Appearance (2-3 sentences"
        + (", use visual profile as basis" if result.get("visual_profile") else "") + "), "
        f"3) Personality (3-5 traits), "
        f"4) Quirk (memorable behavior or habit), "
        f"5) Secret (something hidden), "
        f"6) Motivation (what drives them). "
        f"Then call create_character to register them in the world graph."
    )
    return result


@tool(
    name="cast_page",
    description="Generate a formatted cast of characters with roles and relationships.",
)
def cast_page() -> dict:
    """Generate the story's cast page."""
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    characters = graph.find_nodes_by_type(NodeType.CHARACTER)

    cast = []
    for char in characters:
        relationships = []
        for neighbor, edge in graph.get_neighbors(char.id):
            if neighbor.type == NodeType.CHARACTER:
                relationships.append({
                    "with": neighbor.name,
                    "type": edge.type.value,
                    "label": edge.label,
                })

        factions = []
        for neighbor, edge in graph.get_neighbors(char.id):
            if neighbor.type == NodeType.FACTION:
                factions.append(neighbor.name)

        cast.append({
            "name": char.name,
            "description": char.short_description or char.description[:100],
            "status": char.status.value,
            "personality": char.properties.get("personality", ""),
            "relationships": relationships[:5],
            "factions": factions,
            "icon": char.icon,
        })

    return {
        "story_name": graph.world.story_name,
        "cast_count": len(cast),
        "cast": cast,
    }


@tool(
    name="inner_monologue",
    description=(
        "Generate a character's private thoughts about recent events. "
        "These thoughts are NOT shared with the player — they inform "
        "the character's future behavior."
    ),
)
def inner_monologue(
    character_name: str,
    about: str = "",
) -> dict:
    """Generate inner monologue context.

    Args:
        character_name: Character whose thoughts to generate
        about: What they're thinking about (uses recent events if empty)
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    # Recent memories involving this character
    char_memories = memory.get_by_node(node.id)
    char_memories.sort(key=lambda m: m.turn_number, reverse=True)

    # Emotional state
    from saga_engine.models import MemoryType
    emotional = [m for m in char_memories if m.type == MemoryType.EMOTIONAL]

    return {
        "character": {
            "name": node.name,
            "personality": node.properties.get("personality", ""),
            "goals": node.properties.get("goals", ""),
            "fears": node.properties.get("fears", ""),
        },
        "topic": about or "recent events",
        "recent_events": [m.summary for m in char_memories[:5]],
        "emotional_state": [m.summary for m in emotional[:3]],
        "instruction": (
            f"Write {node.name}'s inner monologue — their private thoughts "
            f"about {about or 'recent events'}. Use first person. Reflect their "
            f"personality, fears, and goals. These are secret — the player "
            f"doesn't hear them, but they influence the character's actions."
        ),
    }
