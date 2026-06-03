"""Character Management Tool — Create, update, list characters."""

from __future__ import annotations

import logging
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.character")


def _get_graph():
    from .story_turn import _get_engine
    graph, _, _ = _get_engine()
    return graph


@tool(
    name="create_character",
    description="Create a new character in the story world with name, description, and properties.",
)
def create_character(
    name: str,
    description: str,
    short_description: str = "",
    personality: str = "",
    appearance: str = "",
    backstory: str = "",
    icon: str = "",
) -> dict:
    """Create a character node in the world graph.

    Args:
        name: Character's name
        description: Full description
        short_description: One-liner for context
        personality: Personality traits
        appearance: Physical appearance
        backstory: Character's history
        icon: Emoji icon for display
    """
    from saga_engine.models import StoryNode, NodeType

    graph = _get_graph()

    props = {}
    if personality:
        props["personality"] = personality
    if appearance:
        props["appearance"] = appearance
    if backstory:
        props["backstory"] = backstory

    node = StoryNode(
        type=NodeType.CHARACTER,
        name=name,
        description=description,
        short_description=short_description or f"{name}, a character in the story",
        properties=props,
        icon=icon or "🧑",
    )

    graph.add_node(node)
    graph.save()

    return {"id": node.id, "name": node.name, "type": "character"}


@tool(
    name="list_characters",
    description="List all characters in the current story world.",
)
def list_characters() -> dict:
    """List all character nodes."""
    from saga_engine.models import NodeType

    graph = _get_graph()
    characters = graph.find_nodes_by_type(NodeType.CHARACTER)

    return {
        "count": len(characters),
        "characters": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.short_description or c.description[:100],
                "status": c.status.value,
            }
            for c in characters
        ],
    }


@tool(
    name="update_character",
    description="Update an existing character's properties.",
)
def update_character(
    name: str,
    description: Optional[str] = None,
    personality: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """Update a character by name.

    Args:
        name: Character name to find
        description: New description (optional)
        personality: New personality traits (optional)
        status: New status: 'active', 'dormant', 'destroyed', 'hidden' (optional)
    """
    graph = _get_graph()
    node = graph.find_node_by_name(name)

    if not node:
        return {"error": f"Character '{name}' not found"}

    updates = {}
    if description is not None:
        updates["description"] = description
    if personality is not None:
        props = dict(node.properties)
        props["personality"] = personality
        updates["properties"] = props
    if status is not None:
        from saga_engine.models import NodeStatus
        status_map = {
            "active": NodeStatus.ACTIVE,
            "dormant": NodeStatus.DORMANT,
            "destroyed": NodeStatus.DESTROYED,
            "hidden": NodeStatus.HIDDEN,
        }
        if status in status_map:
            updates["status"] = status_map[status]

    if updates:
        graph.update_node(node.id, **updates)
        graph.save()

    return {"id": node.id, "name": node.name, "updated": list(updates.keys())}
