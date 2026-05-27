"""Inventory Tool — Item tracking for RPG mechanics."""

from __future__ import annotations

from typing import Optional

from adk.tools import tool


def _get_graph():
    from .story_turn import _get_engine
    graph, _, _ = _get_engine()
    return graph


@tool(
    name="create_item",
    description="Create an item in the story world (weapon, potion, key, quest item, etc.)",
)
def create_item(
    name: str,
    description: str,
    item_type: str = "material",
    rarity: str = "common",
    properties: Optional[dict] = None,
    icon: str = "",
) -> dict:
    """Create an item node.

    Args:
        name: Item name
        description: Item description
        item_type: 'weapon', 'armor', 'consumable', 'material', 'quest', 'key'
        rarity: 'common', 'uncommon', 'rare', 'epic', 'legendary'
        properties: Dict of item stats (e.g. {"damage": 10, "durability": 50})
        icon: Emoji icon
    """
    from saga_engine.models import StoryNode, NodeType

    graph = _get_graph()

    props = properties or {}
    props["item_type"] = item_type
    props["rarity"] = rarity

    node = StoryNode(
        type=NodeType.ITEM,
        name=name,
        description=description,
        short_description=f"[{rarity}] {name} ({item_type})",
        properties=props,
        tags=[item_type, rarity],
        icon=icon or "🎒",
    )
    graph.add_node(node)
    graph.save()

    return {"id": node.id, "name": name, "type": item_type, "rarity": rarity}


@tool(
    name="give_item",
    description="Give an item to a character (creates an 'owns' relationship).",
)
def give_item(character_name: str, item_name: str) -> dict:
    """Give an item to a character.

    Args:
        character_name: Character who receives the item
        item_name: Item to give
    """
    from saga_engine.models import StoryEdge, EdgeType

    graph = _get_graph()

    char = graph.find_node_by_name(character_name)
    item = graph.find_node_by_name(item_name)

    if not char:
        return {"error": f"Character '{character_name}' not found"}
    if not item:
        return {"error": f"Item '{item_name}' not found"}

    edge = StoryEdge(
        type=EdgeType.OWNS,
        source_id=char.id,
        target_id=item.id,
        label=f"{char.name} owns {item.name}",
    )
    graph.add_edge(edge)
    graph.save()

    return {"character": char.name, "item": item.name, "relationship": "owns"}


@tool(
    name="list_inventory",
    description="List all items a character owns.",
)
def list_inventory(character_name: str) -> dict:
    """List items owned by a character.

    Args:
        character_name: Character whose inventory to check
    """
    from saga_engine.models import EdgeType

    graph = _get_graph()
    char = graph.find_node_by_name(character_name)

    if not char:
        return {"error": f"Character '{character_name}' not found"}

    items = []
    for edge in graph.get_edges_from(char.id):
        if edge.type == EdgeType.OWNS:
            item_node = graph.get_node(edge.target_id)
            if item_node:
                items.append({
                    "name": item_node.name,
                    "type": item_node.properties.get("item_type", "unknown"),
                    "rarity": item_node.properties.get("rarity", "common"),
                    "description": item_node.short_description,
                })

    return {"character": char.name, "items": items, "count": len(items)}
