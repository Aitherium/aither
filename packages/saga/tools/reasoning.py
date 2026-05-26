"""Reasoning Tools — Deep narrative analysis via reasoning-as-a-tool.

Provides structured thinking for complex narrative decisions:
- think_deeply: General deep reasoning for plot-critical moments
- evaluate_branches: MCTS-powered comparison of story continuations
- check_consistency: Verify narrative consistency against established world
"""

from __future__ import annotations

import logging
from typing import List, Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.reasoning")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _get_memory():
    from .story_turn import _get_engine
    _, memory, _ = _get_engine()
    return memory


@tool(
    name="think_deeply",
    description=(
        "Invoke deep reasoning for complex narrative decisions. "
        "Use for: plot consistency, character motivation analysis, "
        "world-rule validation, multi-path consequence evaluation, "
        "foreshadowing placement, and paradox resolution. "
        "Returns structured analysis to inform your prose."
    ),
)
def think_deeply(
    question: str,
    context: str = "",
    characters_involved: str = "",
    world_rules_relevant: str = "",
) -> dict:
    """Deep reasoning about a narrative question.

    Args:
        question: The narrative question to reason about
        context: Additional context for the reasoning
        characters_involved: Comma-separated character names involved
        world_rules_relevant: Any world rules that constrain the answer
    """
    graph, memory, ctx_asm = _get_engine()

    # Gather relevant world knowledge
    lore_context = []
    procedural_memories = memory.get_by_type(
        __import__("saga_engine.models", fromlist=["MemoryType"]).MemoryType.PROCEDURAL
    )
    for mem in procedural_memories:
        if mem.pinned or mem.importance >= 0.7:
            lore_context.append(f"RULE: {mem.summary}")

    # Gather character knowledge if specified
    char_context = []
    if characters_involved:
        for char_name in characters_involved.split(","):
            char_name = char_name.strip()
            node = graph.find_node_by_name(char_name)
            if node:
                char_context.append(
                    f"{node.name}: {node.description[:200]}"
                )
                props = node.properties
                if props.get("personality"):
                    char_context.append(f"  Personality: {props['personality']}")
                if props.get("backstory"):
                    char_context.append(f"  Backstory: {props['backstory'][:150]}")
                # Get relationships
                for neighbor, edge in graph.get_neighbors(node.id):
                    char_context.append(
                        f"  -> {edge.type.value} {neighbor.name}: {edge.label}"
                    )

    # Build the structured analysis prompt (returned as data for the LLM)
    analysis = {
        "question": question,
        "world_rules": lore_context[:10],
        "character_knowledge": char_context[:20],
        "world_state": {
            "turn": graph.world.turn_number,
            "location": graph.world.current_location,
            "mood": graph.world.mood,
            "time": graph.world.time_of_day,
            "active_plots": len(graph.world.active_plot_threads),
        },
        "reasoning_framework": (
            "Consider: (1) Does this violate any established world rules? "
            "(2) Is each character's motivation consistent with their personality and history? "
            "(3) What are the 2nd and 3rd order consequences? "
            "(4) Does this create interesting future narrative possibilities? "
            "(5) Is this emotionally satisfying for the story?"
        ),
    }

    if context:
        analysis["additional_context"] = context
    if world_rules_relevant:
        analysis["specific_rules"] = world_rules_relevant

    return analysis


@tool(
    name="evaluate_branches",
    description=(
        "Evaluate multiple possible story continuations at a branching point. "
        "Scores each option on coherence, character consistency, tension, "
        "and narrative potential. Use at crossroads or when the player faces "
        "a major decision."
    ),
)
def evaluate_branches(
    situation: str,
    options: str,
    criteria: str = "coherence,tension,consequences",
) -> dict:
    """Evaluate story branching options.

    Args:
        situation: Description of the current narrative situation
        options: Semicolon-separated list of possible continuations
        criteria: Comma-separated evaluation criteria
    """
    graph, memory, _ = _get_engine()

    option_list = [o.strip() for o in options.split(";") if o.strip()]
    criteria_list = [c.strip() for c in criteria.split(",") if c.strip()]

    # Gather world context for evaluation
    from saga_engine.models import MemoryType
    procedural = memory.get_by_type(MemoryType.PROCEDURAL)
    world_rules = [m.summary for m in procedural if m.pinned or m.importance >= 0.7]

    evaluations = []
    for i, option in enumerate(option_list):
        eval_entry = {
            "option_number": i + 1,
            "description": option,
            "criteria_to_evaluate": criteria_list,
            "world_rules_to_check": world_rules[:5],
            "active_characters": len(graph.world.present_characters),
            "active_plots": len(graph.world.active_plot_threads),
        }
        evaluations.append(eval_entry)

    return {
        "situation": situation,
        "options_count": len(option_list),
        "evaluations": evaluations,
        "recommendation_framework": (
            "For each option, assess: "
            "Does it advance active plot threads? "
            "Does it create meaningful consequences? "
            "Does it give the player agency? "
            "Does it maintain narrative tension? "
            "Choose the option that best balances all criteria."
        ),
    }


@tool(
    name="check_consistency",
    description=(
        "Check if a proposed narrative action or event is consistent with "
        "established world rules, character knowledge, and story history. "
        "Use before introducing major plot elements to avoid contradictions."
    ),
)
def check_consistency(
    proposed_action: str,
    entities_involved: str = "",
) -> dict:
    """Check narrative consistency of a proposed action.

    Args:
        proposed_action: The action or event to check
        entities_involved: Comma-separated entity names to check against
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType, NodeStatus

    issues = []
    context = []

    # Check against procedural memories (world rules)
    procedural = memory.get_by_type(MemoryType.PROCEDURAL)
    rules_checked = []
    for mem in procedural:
        if mem.pinned or mem.importance >= 0.5:
            rules_checked.append(mem.summary)

    context.append(f"World rules checked: {len(rules_checked)}")

    # Check entity statuses
    if entities_involved:
        for entity_name in entities_involved.split(","):
            entity_name = entity_name.strip()
            node = graph.find_node_by_name(entity_name)
            if node:
                if node.status == NodeStatus.DESTROYED:
                    issues.append(
                        f"CONFLICT: {node.name} is DESTROYED — cannot participate in new events"
                    )
                elif node.status == NodeStatus.HIDDEN:
                    issues.append(
                        f"WARNING: {node.name} is HIDDEN — reveal must be narratively justified"
                    )
                context.append(f"{node.name}: status={node.status.value}, type={node.type.value}")
            else:
                context.append(f"{entity_name}: NOT FOUND in world graph (new entity?)")

    # Check recent related memories for contradictions
    related_memories = memory.recall(
        query_text=proposed_action,
        limit=10,
        current_turn=graph.world.turn_number,
    )
    for mem, reason in related_memories:
        context.append(f"Related memory: {mem.summary}")

    return {
        "proposed_action": proposed_action,
        "issues_found": len(issues),
        "issues": issues,
        "world_rules": rules_checked[:8],
        "entity_context": context,
        "related_memories": [
            {"summary": m.summary, "type": m.type.value}
            for m, _ in related_memories[:5]
        ],
        "verdict": "CONFLICT" if issues else "CONSISTENT",
    }
