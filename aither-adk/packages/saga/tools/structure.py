"""Structural Authoring Tools — Story architecture, planning, and pacing.

Tools for planning and shaping story structure beyond prose generation:
outlines, plot twist exploration, pacing analysis, foreshadowing, transitions.
"""

from __future__ import annotations

import logging
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.structure")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _get_memory():
    from .story_turn import _get_engine
    _, memory, _ = _get_engine()
    return memory


@tool(
    name="outline_chapter",
    description=(
        "Generate a structured chapter outline with scene beats, tension curve, "
        "and key events. Returns structured data for the outline panel."
    ),
)
def outline_chapter(
    chapter_title: str,
    theme: str = "",
    characters_involved: str = "",
    target_beats: int = 5,
) -> dict:
    """Generate a chapter outline.

    Args:
        chapter_title: Title or concept for the chapter
        theme: Central theme or conflict
        characters_involved: Comma-separated character names
        target_beats: Number of scene beats to plan
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeType

    # Gather character data
    char_data = []
    if characters_involved:
        for name in characters_involved.split(","):
            node = graph.find_node_by_name(name.strip())
            if node:
                char_data.append({
                    "name": node.name,
                    "personality": node.properties.get("personality", ""),
                    "goals": node.properties.get("goals", ""),
                })

    # Gather active plot threads
    active_plots = []
    for thread_id in graph.world.active_plot_threads:
        thread = graph.get_node(thread_id)
        if thread:
            active_plots.append(thread.name)

    return {
        "chapter_title": chapter_title,
        "theme": theme,
        "characters": char_data,
        "active_plots": active_plots,
        "target_beats": target_beats,
        "current_world_state": {
            "location": graph.world.current_location,
            "mood": graph.world.mood,
            "time": graph.world.time_of_day,
            "turn": graph.world.turn_number,
        },
        "instruction": (
            f"Create a {target_beats}-beat chapter outline for '{chapter_title}'. "
            f"Each beat should have: title, description, tension_level (1-10), "
            f"characters involved, and whether it advances a plot thread. "
            f"The tension curve should rise and fall naturally with a climax near beat {target_beats - 1}."
        ),
    }


@tool(
    name="suggest_plot_twists",
    description=(
        "Explore 3-5 possible plot twists at the current story point with "
        "consequence analysis. Uses world knowledge and character motivations "
        "to generate grounded twists."
    ),
)
def suggest_plot_twists(
    situation: str = "",
    twist_count: int = 3,
) -> dict:
    """Suggest plot twists with consequence analysis.

    Args:
        situation: Current situation to twist (uses world state if empty)
        twist_count: Number of twists to suggest (3-5)
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeType, MemoryType

    twist_count = max(2, min(5, twist_count))

    # Gather context for twist generation
    characters = graph.find_nodes_by_type(NodeType.CHARACTER)
    char_summaries = [
        {"name": c.name, "personality": c.properties.get("personality", ""),
         "secrets": c.properties.get("secrets", "")}
        for c in characters[:10]
    ]

    factions = graph.find_nodes_by_type(NodeType.FACTION)
    faction_names = [f.name for f in factions]

    # Hidden entities make great twist material
    from saga_engine.models import NodeStatus
    hidden = [n for n in graph.get_all_nodes() if n.status == NodeStatus.HIDDEN]
    hidden_hints = [{"name": n.name, "type": n.type.value} for n in hidden[:5]]

    # Emotional memories can fuel twists
    emotional = memory.get_by_type(MemoryType.EMOTIONAL)
    emotions = [m.summary for m in emotional[:5]]

    return {
        "situation": situation or f"Turn {graph.world.turn_number} at {graph.world.mood} mood",
        "twist_count": twist_count,
        "characters": char_summaries,
        "factions": faction_names,
        "hidden_entities": hidden_hints,
        "emotional_threads": emotions,
        "active_plots": [
            graph.get_node(tid).name
            for tid in graph.world.active_plot_threads
            if graph.get_node(tid)
        ],
        "instruction": (
            f"Generate {twist_count} plot twists. For each twist: "
            f"1) The twist itself, 2) Which characters are affected, "
            f"3) Immediate consequences, 4) Long-term implications, "
            f"5) Foreshadowing opportunities. Twists should be grounded in "
            f"established characters and world — not random events."
        ),
    }


@tool(
    name="analyze_pacing",
    description=(
        "Analyze the story's tension curve across recent turns. "
        "Identifies rising/falling action, plateaus, and pacing issues."
    ),
)
def analyze_pacing(lookback_turns: int = 10) -> dict:
    """Analyze pacing and tension across recent turns.

    Args:
        lookback_turns: How many turns back to analyze
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType

    current_turn = graph.world.turn_number
    start_turn = max(0, current_turn - lookback_turns)

    # Gather episodic memories in turn order
    all_memories = memory.get_all()
    turn_memories = sorted(
        [m for m in all_memories if start_turn <= m.turn_number <= current_turn],
        key=lambda m: m.turn_number,
    )

    turns = []
    for mem in turn_memories:
        turns.append({
            "turn": mem.turn_number,
            "summary": mem.summary,
            "type": mem.type.value,
            "importance": mem.importance,
        })

    return {
        "analyzed_range": f"Turns {start_turn}-{current_turn}",
        "turn_count": len(turns),
        "turns": turns,
        "current_mood": graph.world.mood,
        "active_plot_count": len(graph.world.active_plot_threads),
        "instruction": (
            "Analyze the tension curve of this sequence. For each turn, assign "
            "a tension level (1-10). Identify: rising action, climax, falling action, "
            "plateaus. Flag pacing issues: too many calm turns in a row, "
            "action fatigue, unresolved tension."
        ),
    }


@tool(
    name="foreshadow",
    description=(
        "Plant a foreshadowing seed that pays off N turns later. "
        "Stores the seed as a hidden procedural memory that surfaces "
        "when the payoff turn arrives."
    ),
)
def foreshadow(
    seed_detail: str,
    payoff_description: str,
    turns_until_payoff: int = 10,
    subtlety: str = "subtle",
) -> dict:
    """Plant a foreshadowing element.

    Args:
        seed_detail: The detail to plant now (what the reader notices)
        payoff_description: What it means when revealed
        turns_until_payoff: Estimated turns before the payoff
        subtlety: How obvious (subtle, moderate, obvious)
    """
    from saga_engine.models import MemoryType

    graph, memory, _ = _get_engine()
    current_turn = graph.world.turn_number
    payoff_turn = current_turn + turns_until_payoff

    # Store as procedural memory with high importance so it surfaces at payoff time
    mem = memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=(
            f"[FORESHADOWING] Seed planted at turn {current_turn}: {seed_detail}\n"
            f"Payoff (around turn {payoff_turn}): {payoff_description}\n"
            f"Subtlety: {subtlety}"
        ),
        summary=f"Foreshadowing: {seed_detail[:60]} → payoff ~turn {payoff_turn}",
        importance=0.85,
        pinned=True,
        turn_number=current_turn,
        tags=["foreshadowing", f"payoff_turn_{payoff_turn}"],
        created_by="saga:foreshadow",
    )
    memory.save()

    return {
        "memory_id": mem.id,
        "seed": seed_detail,
        "payoff": payoff_description,
        "planted_at_turn": current_turn,
        "expected_payoff_turn": payoff_turn,
        "subtlety": subtlety,
    }


@tool(
    name="scene_transition",
    description=(
        "Generate a smooth transition between two scenes. Handles "
        "time skips, location changes, and POV shifts."
    ),
)
def scene_transition(
    from_scene: str = "",
    to_scene: str = "",
    transition_type: str = "location_change",
    time_skip: str = "",
) -> dict:
    """Generate scene transition context.

    Args:
        from_scene: Description of the scene we're leaving
        to_scene: Description of the scene we're entering
        transition_type: Type (location_change, time_skip, pov_shift, flashback, dream)
        time_skip: Duration of time skip if applicable
    """
    graph, _, _ = _get_engine()

    return {
        "from_scene": from_scene or f"Current: {graph.world.mood} at {graph.world.time_of_day}",
        "to_scene": to_scene,
        "transition_type": transition_type,
        "time_skip": time_skip,
        "current_mood": graph.world.mood,
        "instruction": (
            f"Write a smooth {transition_type} transition. "
            f"Bridge the emotional tone from one scene to the next. "
            f"If time passes, hint at what happened in between. "
            f"Maintain narrative momentum — don't let the reader disengage."
        ),
    }


@tool(
    name="summarize_arc",
    description=(
        "Generate a 'Previously on...' recap for the current story arc. "
        "Useful at session start or after breaks."
    ),
)
def summarize_arc(lookback_turns: int = 20) -> dict:
    """Generate a story arc summary.

    Args:
        lookback_turns: How many turns back to summarize
    """
    graph, memory, _ = _get_engine()

    current_turn = graph.world.turn_number
    start_turn = max(0, current_turn - lookback_turns)

    # Gather important memories
    all_memories = memory.get_all()
    significant = sorted(
        [m for m in all_memories
         if m.turn_number >= start_turn and m.importance >= 0.4],
        key=lambda m: m.turn_number,
    )

    events = [{"turn": m.turn_number, "summary": m.summary, "type": m.type.value}
              for m in significant[:20]]

    return {
        "story_name": graph.world.story_name,
        "arc_range": f"Turns {start_turn}-{current_turn}",
        "key_events": events,
        "current_state": {
            "location": graph.world.current_location,
            "mood": graph.world.mood,
            "active_plots": len(graph.world.active_plot_threads),
        },
        "instruction": (
            "Write a compelling 'Previously on...' recap. Hit the major beats, "
            "remind the reader of key characters and their arcs, and end with "
            "the current dramatic question."
        ),
    }


@tool(
    name="parallel_threads",
    description="List all active and stalled plot threads with status.",
)
def parallel_threads() -> dict:
    """List all plot threads and their status."""
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeType

    threads = graph.find_nodes_by_type(NodeType.PLOT_THREAD)
    active_ids = set(graph.world.active_plot_threads)

    thread_list = []
    for t in threads:
        # Check for recent activity
        related_mems = memory.get_by_node(t.id)
        last_activity = max((m.turn_number for m in related_mems), default=0)

        thread_list.append({
            "id": t.id,
            "name": t.name,
            "description": t.short_description or t.description[:100],
            "status": "active" if t.id in active_ids else t.status.value,
            "last_activity_turn": last_activity,
            "turns_since_activity": graph.world.turn_number - last_activity,
            "stalled": (graph.world.turn_number - last_activity) > 10,
        })

    return {
        "total_threads": len(thread_list),
        "active": len([t for t in thread_list if t["status"] == "active"]),
        "stalled": len([t for t in thread_list if t.get("stalled")]),
        "threads": thread_list,
    }


@tool(
    name="tension_map",
    description=(
        "Generate tension data across the story for visualization. "
        "Returns structured data for a line chart."
    ),
)
def tension_map() -> dict:
    """Generate tension map data for the full story."""
    _, memory, _ = _get_engine()

    all_memories = sorted(memory.get_all(), key=lambda m: m.turn_number)

    turns = []
    for mem in all_memories:
        if mem.type.value == "episodic":
            turns.append({
                "turn": mem.turn_number,
                "importance": round(mem.importance, 2),
                "summary": mem.summary[:60],
                "type": mem.type.value,
            })

    return {
        "turn_count": len(turns),
        "turns": turns,
        "instruction": (
            "The importance values approximate tension. "
            "High importance = high-stakes moments. "
            "Analyze the pattern for pacing recommendations."
        ),
    }
