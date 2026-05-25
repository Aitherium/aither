"""World Management Tool — Locations, factions, lore, relationships."""

from __future__ import annotations

import logging
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.world")


def _get_graph():
    from .story_turn import _get_engine
    graph, _, _ = _get_engine()
    return graph


@tool(
    name="create_location",
    description="Create a new location in the story world.",
)
def create_location(
    name: str,
    description: str,
    short_description: str = "",
    icon: str = "",
) -> dict:
    """Create a location node.

    Args:
        name: Location name
        description: Full description of the place
        short_description: One-liner for context
        icon: Emoji icon
    """
    from saga_engine.models import StoryNode, NodeType

    graph = _get_graph()
    node = StoryNode(
        type=NodeType.LOCATION,
        name=name,
        description=description,
        short_description=short_description or name,
        icon=icon or "🏰",
    )
    graph.add_node(node)
    graph.save()
    return {"id": node.id, "name": node.name, "type": "location"}


@tool(
    name="create_faction",
    description="Create a new faction or organization in the world.",
)
def create_faction(
    name: str,
    description: str,
    short_description: str = "",
    icon: str = "",
) -> dict:
    """Create a faction node.

    Args:
        name: Faction name
        description: Description of the organization
        short_description: One-liner for context
        icon: Emoji icon
    """
    from saga_engine.models import StoryNode, NodeType

    graph = _get_graph()
    node = StoryNode(
        type=NodeType.FACTION,
        name=name,
        description=description,
        short_description=short_description or name,
        icon=icon or "🏛️",
    )
    graph.add_node(node)
    graph.save()
    return {"id": node.id, "name": node.name, "type": "faction"}


@tool(
    name="add_lore",
    description="Add a lore entry (history, culture, magic rules, etc.) to the world.",
)
def add_lore(
    title: str,
    content: str,
    category: str = "history",
    icon: str = "",
) -> dict:
    """Add a lore node.

    Args:
        title: Lore entry title
        content: Full lore text
        category: One of 'history', 'culture', 'magic', 'character', 'location', 'item'
        icon: Emoji icon
    """
    from saga_engine.models import StoryNode, NodeType

    graph = _get_graph()
    node = StoryNode(
        type=NodeType.LORE,
        name=title,
        description=content,
        short_description=f"[{category}] {title}",
        tags=[category],
        icon=icon or "📜",
    )
    graph.add_node(node)
    graph.save()
    return {"id": node.id, "name": node.name, "type": "lore", "category": category}


@tool(
    name="create_relationship",
    description="Create a relationship between two entities in the world (character knows character, character at location, etc.)",
)
def create_relationship(
    source_name: str,
    target_name: str,
    relationship_type: str = "related_to",
    label: str = "",
    bidirectional: bool = False,
) -> dict:
    """Create an edge between two nodes.

    Args:
        source_name: Name of the source entity
        target_name: Name of the target entity
        relationship_type: Edge type (knows, located_at, member_of, owns, hostile_to, allied_with, etc.)
        label: Human-readable label for the relationship
        bidirectional: If True, relationship goes both ways
    """
    from saga_engine.models import StoryEdge, EdgeType

    graph = _get_graph()

    source = graph.find_node_by_name(source_name)
    target = graph.find_node_by_name(target_name)

    if not source:
        return {"error": f"Entity '{source_name}' not found"}
    if not target:
        return {"error": f"Entity '{target_name}' not found"}

    type_map = {t.value: t for t in EdgeType}
    edge_type = type_map.get(relationship_type, EdgeType.RELATED_TO)

    edge = StoryEdge(
        type=edge_type,
        source_id=source.id,
        target_id=target.id,
        label=label or f"{source.name} {relationship_type} {target.name}",
        bidirectional=bidirectional,
    )
    graph.add_edge(edge)
    graph.save()

    return {
        "id": edge.id,
        "source": source.name,
        "target": target.name,
        "type": relationship_type,
    }


@tool(
    name="get_world_state",
    description="Get the current state of the story world (stats, characters, locations, etc.)",
)
def get_world_state() -> dict:
    """Get world state and graph statistics."""
    graph = _get_graph()
    stats = graph.stats()

    from saga_engine.models import NodeType
    locations = graph.find_nodes_by_type(NodeType.LOCATION)
    factions = graph.find_nodes_by_type(NodeType.FACTION)

    return {
        **stats,
        "locations": [{"name": l.name, "id": l.id} for l in locations],
        "factions": [{"name": f.name, "id": f.id} for f in factions],
    }
