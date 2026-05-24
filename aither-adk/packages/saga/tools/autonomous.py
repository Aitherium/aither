"""Autonomous Character Tools — Characters that live between sessions.

Tools for simulating off-screen character behavior: diary entries,
off-screen events, character reactions, faction turns, NPC conversations,
dream sequences, letters, and world ticks.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.autonomous")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _get_simulation():
    """Lazy-load the SimulationClient singleton."""
    try:
        from saga_engine.simulation import get_simulation_client
        return get_simulation_client()
    except Exception:
        return None


def _run_async(coro):
    """Run an async coroutine from sync context."""
    from .story_turn import _run_async
    return _run_async(coro)


def _get_event_mapper():
    from saga_engine.sim_event_mapper import SimEventMapper
    return SimEventMapper()


@tool(
    name="character_diary",
    description=(
        "Generate a diary entry from a character's perspective about recent events. "
        "Written in first person using the character's voice and knowledge."
    ),
)
def character_diary(
    character_name: str,
    topic: str = "",
) -> dict:
    """Generate a character diary entry.

    Args:
        character_name: Character writing the diary
        topic: What to write about (uses recent events if empty)
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    char_memories = memory.get_by_node(node.id)
    char_memories.sort(key=lambda m: m.turn_number, reverse=True)

    from saga_engine.models import MemoryType
    emotional = [m for m in char_memories if m.type == MemoryType.EMOTIONAL]

    return {
        "character": {
            "name": node.name,
            "personality": node.properties.get("personality", ""),
            "speech_style": node.properties.get("speech_style", ""),
        },
        "topic": topic or "recent events",
        "recent_events": [m.summary for m in char_memories[:8]],
        "emotional_state": [m.summary for m in emotional[:3]],
        "instruction": (
            f"Write a diary entry as {node.name}. First person, in their voice. "
            f"Reflect on {topic or 'recent events'}. Include: "
            f"personal observations, emotional reactions, hopes/fears, "
            f"things they'd never say aloud. 150-300 words."
        ),
    }


@tool(
    name="offscreen_events",
    description=(
        "Simulate what happens to NPCs while the player is away. "
        "Faction wars, romances, betrayals, discoveries — the world moves on."
    ),
)
def offscreen_events(
    time_passed: str = "a few days",
    focus_characters: str = "",
    focus_factions: str = "",
) -> dict:
    """Simulate off-screen events.

    Args:
        time_passed: How much time has passed
        focus_characters: Comma-separated character names to focus on
        focus_factions: Comma-separated faction names to focus on
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType, NodeStatus

    # All active characters not currently in scene
    all_chars = graph.find_nodes_by_type(NodeType.CHARACTER)
    present_ids = set(graph.world.present_characters)
    offscreen_chars = [
        {"name": c.name, "goals": c.properties.get("goals", ""),
         "personality": c.properties.get("personality", "")}
        for c in all_chars
        if c.id not in present_ids and c.status == NodeStatus.ACTIVE
    ]

    factions = graph.find_nodes_by_type(NodeType.FACTION)

    result = {
        "time_passed": time_passed,
        "offscreen_characters": offscreen_chars[:15],
        "factions": [{"name": f.name, "description": f.description[:100]} for f in factions],
        "focus_characters": focus_characters.split(",") if focus_characters else [],
        "focus_factions": focus_factions.split(",") if focus_factions else [],
    }

    # Pull real simulation events if connected
    try:
        sim = _get_simulation()
        if sim and graph.world.prometheus_world_id:
            events = _run_async(sim.get_recent_events(
                graph.world.prometheus_world_id, count=20
            ))
            if events:
                result["simulation_events"] = [
                    {"type": e.event_type, "description": e.description, "severity": e.severity}
                    for e in events
                ]
                result["simulation_backed"] = True
    except Exception:
        pass

    if result.get("simulation_backed"):
        result["instruction"] = (
            f"Simulate {time_passed} of off-screen activity. "
            f"Use the simulation events above as ground truth. "
            f"Expand on them with narrative detail. "
            f"Store each as a memory so they're discoverable later."
        )
    else:
        result["instruction"] = (
            f"Simulate {time_passed} of off-screen activity. "
            f"For each notable character/faction: what did they do? "
            f"Generate 3-5 events based on their goals and personality. "
            f"Some should create future plot hooks. "
            f"Store each as a memory so they're discoverable later."
        )

    return result


@tool(
    name="character_react",
    description=(
        "Given an event, generate how each present character would react "
        "based on their personality and relationships."
    ),
)
def character_react(event: str) -> dict:
    """Generate character reactions to an event.

    Args:
        event: The event to react to
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    reactions = []
    for char_id in graph.world.present_characters:
        char = graph.get_node(char_id)
        if not char:
            continue

        relationships = []
        for neighbor, edge in graph.get_neighbors(char.id):
            if neighbor.type == NodeType.CHARACTER:
                relationships.append(f"{edge.type.value} {neighbor.name}")

        reaction = {
            "name": char.name,
            "personality": char.properties.get("personality", ""),
            "relationships": relationships[:5],
        }

        # Get simulation-backed NPC decision if available
        try:
            sim = _get_simulation()
            if sim and graph.world.prometheus_world_id:
                decision = _run_async(sim.get_npc_decision(
                    graph.world.prometheus_world_id,
                    char.id,
                    situation=event,
                ))
                if decision:
                    reaction["chosen_action"] = decision.get("chosen_action", "")
                    reaction["reasoning"] = decision.get("reasoning", "")
                    reaction["emotional_state"] = decision.get("emotional_state", "")
        except Exception:
            pass

        reactions.append(reaction)

    has_sim = any("chosen_action" in r for r in reactions)

    return {
        "event": event,
        "characters": reactions,
        "simulation_backed": has_sim,
        "instruction": (
            f"For each character present, describe their immediate reaction to: "
            f"'{event}'. "
            + ("Use their simulation-decided actions as a basis. " if has_sim else "") +
            "Consider their personality, relationships, and what "
            f"they stand to gain or lose. Include: facial expression, body language, "
            f"spoken words (if any), and hidden thoughts."
        ),
    }


@tool(
    name="faction_turn",
    description=(
        "Advance faction politics one step. Each faction takes an action "
        "based on their goals, resources, and relationships."
    ),
)
def faction_turn() -> dict:
    """Advance faction politics by one step."""
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType, EdgeType

    factions = graph.find_nodes_by_type(NodeType.FACTION)

    faction_data = []
    for faction in factions:
        relations = []
        for neighbor, edge in graph.get_neighbors(faction.id):
            if neighbor.type == NodeType.FACTION:
                relations.append({
                    "with": neighbor.name,
                    "type": edge.type.value,
                    "label": edge.label,
                })

        members = []
        for neighbor, edge in graph.get_neighbors(faction.id):
            if neighbor.type == NodeType.CHARACTER:
                members.append(neighbor.name)

        fd = {
            "name": faction.name,
            "description": faction.description[:150],
            "goals": faction.properties.get("goals", ""),
            "resources": faction.properties.get("resources", ""),
            "relations": relations,
            "key_members": members[:5],
        }

        # MCTS-backed faction strategy if Prometheus connected
        try:
            sim = _get_simulation()
            if sim and graph.world.prometheus_world_id:
                mcts = _run_async(sim.mcts_search(
                    graph.world.prometheus_world_id, faction.id
                ))
                if mcts:
                    fd["mcts_best_action"] = mcts.get("best_action", "")
                    fd["mcts_score"] = mcts.get("best_score", 0)
                    fd["mcts_alternatives"] = mcts.get("evaluated_actions", [])[:3]
        except Exception:
            pass

        faction_data.append(fd)

    has_mcts = any("mcts_best_action" in f for f in faction_data)

    return {
        "faction_count": len(faction_data),
        "factions": faction_data,
        "simulation_backed": has_mcts,
        "instruction": (
            "Each faction takes ONE action this turn based on their goals"
            + (" (MCTS-evaluated actions provided)" if has_mcts else "") +
            ": expand territory, forge alliance, break treaty, recruit, "
            "gather resources, spy, or attack. "
            "Actions should cause ripple effects on other factions. "
            "Update relationships and store events as memories."
        ),
    }


@tool(
    name="npc_conversation",
    description=(
        "Simulate a conversation between two NPCs. The player can read "
        "it as discovered intel, overheard gossip, or intercepted messages."
    ),
)
def npc_conversation(
    character_a: str,
    character_b: str,
    topic: str = "",
    context: str = "private meeting",
) -> dict:
    """Simulate NPC conversation.

    Args:
        character_a: First character name
        character_b: Second character name
        topic: Conversation topic
        context: Setting (private meeting, tavern gossip, letter exchange, battlefield)
    """
    graph, _, _ = _get_engine()

    node_a = graph.find_node_by_name(character_a)
    node_b = graph.find_node_by_name(character_b)

    if not node_a:
        return {"error": f"Character '{character_a}' not found"}
    if not node_b:
        return {"error": f"Character '{character_b}' not found"}

    # Find their relationship
    relationship = "strangers"
    for neighbor, edge in graph.get_neighbors(node_a.id):
        if neighbor.id == node_b.id:
            relationship = edge.label or edge.type.value
            break

    result = {
        "character_a": {
            "name": node_a.name,
            "personality": node_a.properties.get("personality", ""),
            "speech_style": node_a.properties.get("speech_style", ""),
        },
        "character_b": {
            "name": node_b.name,
            "personality": node_b.properties.get("personality", ""),
            "speech_style": node_b.properties.get("speech_style", ""),
        },
        "relationship": relationship,
        "topic": topic,
        "context": context,
    }

    # Enrich with simulation dialogue context
    try:
        sim = _get_simulation()
        if sim and graph.world.prometheus_world_id:
            for key, node in [("character_a_sim", node_a), ("character_b_sim", node_b)]:
                dlg = _run_async(sim.get_dialogue_context(
                    graph.world.prometheus_world_id, node.id
                ))
                if dlg:
                    result[key] = {
                        "emotional_state": dlg.get("emotional_state", ""),
                        "recent_events": dlg.get("recent_events", [])[:3],
                    }
    except Exception:
        pass

    result["instruction"] = (
        f"Write a conversation between {node_a.name} and {node_b.name} "
        f"({context}). Topic: {topic or 'their situation'}. "
        f"Relationship: {relationship}. "
        f"Each character speaks in their own voice. Include subtext — "
        f"what they're really thinking beneath what they say. "
        f"The conversation should reveal useful information or advance a plot."
    )

    return result


@tool(
    name="dream_sequence",
    description=(
        "Generate a dream for a character based on their emotional memories, "
        "fears, and recent experiences."
    ),
)
def dream_sequence(character_name: str) -> dict:
    """Generate a dream sequence.

    Args:
        character_name: Character who is dreaming
    """
    graph, memory, _ = _get_engine()

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    from saga_engine.models import MemoryType

    char_memories = memory.get_by_node(node.id)
    emotional = [m for m in char_memories if m.type == MemoryType.EMOTIONAL]
    episodic = [m for m in char_memories if m.type == MemoryType.EPISODIC]

    return {
        "character": node.name,
        "personality": node.properties.get("personality", ""),
        "fears": node.properties.get("fears", ""),
        "goals": node.properties.get("goals", ""),
        "emotional_memories": [m.summary for m in emotional[:5]],
        "recent_events": [m.summary for m in sorted(episodic, key=lambda m: m.turn_number, reverse=True)[:5]],
        "instruction": (
            f"Write a dream sequence for {node.name}. Dreams blend: "
            f"recent events (distorted), deep fears, hidden desires, "
            f"symbolic imagery. The dream should feel surreal but meaningful — "
            f"it might foreshadow something or reveal something the character "
            f"hasn't consciously acknowledged."
        ),
    }


@tool(
    name="letter_from_npc",
    description=(
        "An NPC writes a letter to the player character. "
        "Found at the next location as a plot device."
    ),
)
def letter_from_npc(
    sender: str,
    recipient: str = "the player",
    tone: str = "urgent",
) -> dict:
    """Generate a letter from an NPC.

    Args:
        sender: Who writes the letter
        recipient: Who it's addressed to
        tone: Letter tone (urgent, friendly, threatening, cryptic, romantic, formal)
    """
    graph, _, _ = _get_engine()

    sender_node = graph.find_node_by_name(sender)

    return {
        "sender": {
            "name": sender_node.name if sender_node else sender,
            "personality": sender_node.properties.get("personality", "") if sender_node else "",
            "speech_style": sender_node.properties.get("speech_style", "") if sender_node else "",
        },
        "recipient": recipient,
        "tone": tone,
        "instruction": (
            f"Write a {tone} letter from {sender} to {recipient}. "
            f"Use the sender's voice and personality. The letter should: "
            f"1) Reveal new information or advance a plot, "
            f"2) Feel authentic to the character, "
            f"3) End with something that demands a response or action. "
            f"Include period-appropriate formatting (salutation, closing)."
        ),
    }


@tool(
    name="world_tick",
    description=(
        "Advance the world by N time units. Weather changes, NPC movements, "
        "event triggers, faction actions. The world lives and breathes."
    ),
)
def world_tick(
    time_units: int = 1,
    time_scale: str = "day",
) -> dict:
    """Advance the world clock.

    Args:
        time_units: Number of time units to advance
        time_scale: Scale (hour, day, week, month, season)
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeType, NodeStatus, MemoryType

    # Current world state
    all_chars = graph.find_nodes_by_type(NodeType.CHARACTER)
    active_chars = [c for c in all_chars if c.status == NodeStatus.ACTIVE]
    factions = graph.find_nodes_by_type(NodeType.FACTION)

    # Check for foreshadowing that might trigger
    foreshadowing = [
        m for m in memory.get_by_type(MemoryType.PROCEDURAL)
        if "foreshadowing" in m.tags
    ]

    # Try simulation backend first
    sim_events = []
    sim_npc_movements = []
    sim_faction_actions = []
    sim_used = False

    try:
        sim = _get_simulation()
        if sim and graph.world.prometheus_world_id:
            minutes_map = {"hour": 60, "day": 24 * 60, "week": 7 * 24 * 60,
                           "month": 30 * 24 * 60, "season": 90 * 24 * 60}
            minutes = time_units * minutes_map.get(time_scale, 24 * 60)
            result = _run_async(sim.tick(graph.world.prometheus_world_id, minutes))
            if result:
                sim_used = True
                new_weather = result.weather
                new_time = graph.world.time_of_day  # Sim doesn't track time_of_day directly
                sim_events = result.events
                sim_npc_movements = result.npc_movements
                sim_faction_actions = result.faction_actions
                if result.game_time:
                    graph.world.simulation_time = result.game_time

                # Store simulation events as memories
                mapper = _get_event_mapper()
                mapper.events_to_memories(sim_events, graph, memory)
                mapper.tick_result_to_state_update(result, graph)
    except Exception:
        pass

    # Fallback: local weather + time progression
    if not sim_used:
        weather_map = {
            "clear": ["clear", "cloudy", "clear"],
            "cloudy": ["cloudy", "rain", "clear", "storm"],
            "rain": ["rain", "storm", "cloudy", "clear"],
            "storm": ["storm", "rain", "cloudy"],
            "snow": ["snow", "blizzard", "cloudy", "snow"],
        }
        current_weather = graph.world.weather or "clear"
        new_weather = random.choice(weather_map.get(current_weather, ["clear"]))

    # Time of day progression (always local)
    time_progression = ["dawn", "day", "dusk", "night"]
    current_time = graph.world.time_of_day
    if current_time in time_progression:
        idx = time_progression.index(current_time)
        new_time = time_progression[(idx + time_units) % len(time_progression)]
    else:
        new_time = "day"

    # Update world state
    graph.world.weather = new_weather
    graph.world.time_of_day = new_time
    graph.save()

    result_dict = {
        "time_advanced": f"{time_units} {time_scale}(s)",
        "new_weather": new_weather,
        "new_time_of_day": new_time,
        "active_characters": len(active_chars),
        "factions": len(factions),
        "pending_foreshadowing": len(foreshadowing),
        "simulation_backed": sim_used,
    }

    if sim_used:
        result_dict["simulation_events"] = [
            {"type": e.event_type, "description": e.description, "severity": e.severity}
            for e in sim_events
        ]
        result_dict["npc_movements"] = sim_npc_movements
        result_dict["faction_actions"] = sim_faction_actions
        result_dict["instruction"] = (
            f"The world advances {time_units} {time_scale}(s) (simulation-backed). "
            f"Weather: {new_weather}. Narrate the simulation events above. "
            f"Check foreshadowing ({len(foreshadowing)} pending) for trigger conditions."
        )
    else:
        result_dict["instruction"] = (
            f"The world advances {time_units} {time_scale}(s). Generate: "
            f"1) Weather change: → {new_weather}, "
            f"2) NPC movements (who goes where), "
            f"3) Faction actions (based on goals), "
            f"4) Random events (one notable thing happens), "
            f"5) Check foreshadowing ({len(foreshadowing)} pending) for trigger conditions. "
            f"Store results as memories."
        )

    return result_dict
