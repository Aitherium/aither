"""World-Building Tools — Procedural and structured world creation.

Faction politics, magic systems, economies, calendars, languages,
cultures, timelines, rumors, and weather systems.
"""

from __future__ import annotations

import logging
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.worldcraft")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _sync_to_sim(system_type: str, config: dict, graph):
    """Push worldcraft data to simulation if connected."""
    try:
        from saga_engine.simulation import get_simulation_client
        from .story_turn import _run_async
        sim = get_simulation_client()
        if sim and graph.world.prometheus_world_id:
            _run_async(sim.register_system(
                graph.world.prometheus_world_id, system_type, config
            ))
    except Exception:
        pass


@tool(
    name="faction_politics",
    description=(
        "Generate or analyze political dynamics between factions: alliances, "
        "rivalries, trade agreements, and territorial disputes."
    ),
)
def faction_politics(action: str = "analyze") -> dict:
    """Analyze or generate faction politics.

    Args:
        action: 'analyze' existing factions or 'generate' new dynamics
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType, EdgeType

    factions = graph.find_nodes_by_type(NodeType.FACTION)

    faction_data = []
    for faction in factions:
        relations = []
        for neighbor, edge in graph.get_neighbors(faction.id):
            if neighbor.type == NodeType.FACTION:
                relations.append({
                    "faction": neighbor.name,
                    "type": edge.type.value,
                    "label": edge.label,
                })
        members = []
        for neighbor, edge in graph.get_neighbors(faction.id):
            if neighbor.type == NodeType.CHARACTER and edge.type == EdgeType.MEMBER_OF:
                members.append(neighbor.name)

        faction_data.append({
            "name": faction.name,
            "description": faction.description[:200],
            "relations": relations,
            "members": members[:10],
            "status": faction.status.value,
        })

    # Augment with simulation faction states if available
    try:
        from saga_engine.simulation import get_simulation_client
        from .story_turn import _run_async
        sim = get_simulation_client()
        if sim and graph.world.prometheus_world_id:
            state = _run_async(sim.get_state(graph.world.prometheus_world_id))
            if state and state.faction_states:
                for fd in faction_data:
                    sim_f = state.faction_states.get(fd["name"], {})
                    if sim_f:
                        fd["tension"] = sim_f.get("tension", 0)
                        fd["power"] = sim_f.get("power", 0)
    except Exception:
        pass

    return {
        "action": action,
        "faction_count": len(faction_data),
        "factions": faction_data,
        "instruction": (
            f"{'Analyze' if action == 'analyze' else 'Generate'} faction politics. "
            f"For each pair: alliance strength, rivalry causes, trade status, "
            f"territorial disputes, recent tensions. Identify power vacuums "
            f"and potential conflicts."
        ),
    }


@tool(
    name="magic_system",
    description=(
        "Define or document the world's magic system: rules, costs, "
        "limitations, schools, and forbidden arts. Stores as inviolable "
        "procedural memories."
    ),
)
def magic_system(
    name: str,
    rules: str,
    cost: str = "",
    limitations: str = "",
    schools: str = "",
    forbidden: str = "",
) -> dict:
    """Define a magic system.

    Args:
        name: Name of the magic system
        rules: Core rules of how magic works
        cost: What magic costs the user
        limitations: What magic cannot do
        schools: Comma-separated schools or disciplines
        forbidden: Forbidden or dark magic practices
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import StoryNode, NodeType, MemoryType

    # Create a lore node for the magic system
    node = StoryNode(
        type=NodeType.LORE,
        name=name,
        description=rules,
        short_description=f"Magic system: {name}",
        tags=["magic", "system"],
        icon="✨",
        pinned=True,
    )
    graph.add_node(node)

    # Store each aspect as a procedural memory (inviolable)
    stored = []
    for label, content in [
        ("Rules", rules),
        ("Cost", cost),
        ("Limitations", limitations),
        ("Forbidden", forbidden),
    ]:
        if content:
            mem = memory.create(
                memory_type=MemoryType.PROCEDURAL,
                content=f"[MAGIC SYSTEM: {name}] {label}: {content}",
                summary=f"Magic ({name}) - {label}: {content[:60]}",
                importance=1.0,
                pinned=True,
                related_nodes=[node.id],
                turn_number=graph.world.turn_number,
                created_by="user:magic_system",
            )
            stored.append({"aspect": label, "memory_id": mem.id})

    graph.save()
    memory.save()

    # Push to simulation
    _sync_to_sim("magic", {
        "name": name, "rules": rules, "cost": cost,
        "limitations": limitations, "schools": schools, "forbidden": forbidden,
    }, graph)

    return {
        "name": name,
        "node_id": node.id,
        "rules": rules[:200],
        "cost": cost,
        "limitations": limitations,
        "schools": schools.split(",") if schools else [],
        "forbidden": forbidden,
        "memories_stored": stored,
    }


@tool(
    name="economic_system",
    description="Define the world's economic system: currency, trade goods, prices.",
)
def economic_system(
    currency_name: str,
    denominations: str = "",
    trade_goods: str = "",
    price_examples: str = "",
) -> dict:
    """Define an economic system.

    Args:
        currency_name: Name of the currency
        denominations: Comma-separated denominations (e.g., "copper, silver, gold")
        trade_goods: Major trade goods
        price_examples: Example prices (e.g., "bread: 2 copper, sword: 5 gold")
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import StoryNode, NodeType, MemoryType

    node = StoryNode(
        type=NodeType.LORE,
        name=f"Economy: {currency_name}",
        description=(
            f"Currency: {currency_name}. "
            f"Denominations: {denominations}. "
            f"Trade goods: {trade_goods}. "
            f"Prices: {price_examples}"
        ),
        short_description=f"Economic system using {currency_name}",
        tags=["economy", "currency"],
        icon="💰",
        pinned=True,
    )
    graph.add_node(node)

    mem = memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=f"[ECONOMY] Currency: {currency_name}, Denominations: {denominations}, Prices: {price_examples}",
        summary=f"Economy: {currency_name} ({denominations})",
        importance=0.9,
        pinned=True,
        related_nodes=[node.id],
        turn_number=graph.world.turn_number,
        created_by="user:economy",
    )

    graph.save()
    memory.save()

    # Push to simulation
    _sync_to_sim("economy", {
        "currency": currency_name, "denominations": denominations,
        "trade_goods": trade_goods, "prices": price_examples,
    }, graph)

    return {
        "currency": currency_name,
        "node_id": node.id,
        "memory_id": mem.id,
        "denominations": denominations.split(",") if denominations else [],
    }


@tool(
    name="calendar_create",
    description="Create a custom in-world calendar with months, seasons, and holidays.",
)
def calendar_create(
    calendar_name: str,
    months: str,
    seasons: str = "",
    holidays: str = "",
    celestial_events: str = "",
) -> dict:
    """Create a custom calendar.

    Args:
        calendar_name: Name of the calendar system
        months: Semicolon-separated month names
        seasons: Comma-separated season names
        holidays: Semicolon-separated holidays (name: description)
        celestial_events: Notable celestial events
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import StoryNode, NodeType, MemoryType

    node = StoryNode(
        type=NodeType.LORE,
        name=f"Calendar: {calendar_name}",
        description=f"Months: {months}. Seasons: {seasons}. Holidays: {holidays}.",
        short_description=f"The {calendar_name} calendar",
        tags=["calendar", "time"],
        icon="📅",
    )
    graph.add_node(node)

    mem = memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=f"[CALENDAR: {calendar_name}] Months: {months}. Seasons: {seasons}. Holidays: {holidays}. Celestial: {celestial_events}",
        summary=f"Calendar: {calendar_name}",
        importance=0.7,
        pinned=True,
        related_nodes=[node.id],
        turn_number=graph.world.turn_number,
    )

    graph.save()
    memory.save()

    # Push to simulation
    _sync_to_sim("calendar", {
        "name": calendar_name, "months": months,
        "seasons": seasons, "holidays": holidays,
        "celestial_events": celestial_events,
    }, graph)

    return {
        "calendar": calendar_name,
        "node_id": node.id,
        "months": months.split(";"),
        "seasons": seasons.split(",") if seasons else [],
    }


@tool(
    name="language_fragments",
    description="Generate in-world language fragments: greetings, curses, naming conventions.",
)
def language_fragments(
    language_name: str,
    culture: str = "",
    fragment_types: str = "greetings,curses,names",
) -> dict:
    """Generate language fragments.

    Args:
        language_name: Name of the language
        culture: Associated culture or faction
        fragment_types: Comma-separated types to generate
    """
    graph, _, _ = _get_engine()

    return {
        "language": language_name,
        "culture": culture,
        "types_requested": fragment_types.split(","),
        "world_tone": graph.world.mood,
        "instruction": (
            f"Generate fragments of the {language_name} language: "
            f"{fragment_types}. Include: original phrase, pronunciation hint, "
            f"meaning. Create consistent phonetic patterns. "
            f"Store important ones as lore entries."
        ),
    }


@tool(
    name="culture_generator",
    description="Generate cultural practices, beliefs, taboos, and rituals for a faction or region.",
)
def culture_generator(
    culture_name: str,
    faction: str = "",
    region: str = "",
) -> dict:
    """Generate cultural details.

    Args:
        culture_name: Name of the culture
        faction: Associated faction name
        region: Associated region/location name
    """
    graph, _, _ = _get_engine()

    faction_node = graph.find_node_by_name(faction) if faction else None
    region_node = graph.find_node_by_name(region) if region else None

    return {
        "culture": culture_name,
        "faction": faction_node.name if faction_node else faction,
        "region": region_node.name if region_node else region,
        "instruction": (
            f"Generate cultural details for {culture_name}: "
            f"1) Core beliefs and values, "
            f"2) Social hierarchy, "
            f"3) Art forms and music, "
            f"4) Taboos and forbidden practices, "
            f"5) Coming-of-age rituals, "
            f"6) Funeral rites, "
            f"7) Greetings and customs. "
            f"Store key rules as lore entries."
        ),
    }


@tool(
    name="history_timeline",
    description="Generate a chronological timeline of world events. Populates EVENT nodes in the graph.",
)
def history_timeline(
    era_name: str = "",
    event_count: int = 5,
    focus: str = "",
) -> dict:
    """Generate historical timeline.

    Args:
        era_name: Name of the historical era
        event_count: Number of events to generate
        focus: Focus area (wars, discoveries, cultural, political)
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    existing_events = graph.find_nodes_by_type(NodeType.EVENT)

    return {
        "era": era_name or "General History",
        "event_count": min(event_count, 10),
        "focus": focus,
        "existing_events": [e.name for e in existing_events[:10]],
        "world_name": graph.world.story_name,
        "instruction": (
            f"Generate {event_count} historical events for {era_name or 'the world'}. "
            f"Each event needs: date/year, name, description, key figures, consequences. "
            f"Events should have causal connections. "
            f"Register each as an EVENT node in the graph."
        ),
    }


@tool(
    name="rumor_mill",
    description=(
        "Generate rumors that NPCs might share. Some true, some false — "
        "tracked in memory metadata so the narrator knows the truth."
    ),
)
def rumor_mill(
    location: str = "",
    count: int = 3,
    topic: str = "",
) -> dict:
    """Generate rumors.

    Args:
        location: Location where rumors circulate
        count: Number of rumors to generate
        topic: Topic focus (optional)
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    characters = [c.name for c in graph.find_nodes_by_type(NodeType.CHARACTER)[:10]]
    locations = [l.name for l in graph.find_nodes_by_type(NodeType.LOCATION)[:10]]

    return {
        "location": location or "the region",
        "count": min(count, 8),
        "topic": topic,
        "known_characters": characters,
        "known_locations": locations,
        "instruction": (
            f"Generate {count} rumors. For each: "
            f"1) The rumor as NPCs would tell it, "
            f"2) Truth value (true/false/partially_true), "
            f"3) Source (who started it and why), "
            f"4) Narrative potential (how it could become a plot hook). "
            f"Store each as a memory with metadata marking truth value."
        ),
    }


@tool(
    name="weather_system",
    description="Define climate patterns, seasonal weather, and magical weather events.",
)
def weather_system(
    region: str,
    climate: str = "temperate",
    magical_weather: str = "",
) -> dict:
    """Define weather patterns.

    Args:
        region: Region name
        climate: Base climate (arctic, temperate, tropical, desert, oceanic)
        magical_weather: Any magical weather phenomena
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType

    mem = memory.create(
        memory_type=MemoryType.PROCEDURAL,
        content=f"[WEATHER: {region}] Climate: {climate}. Magical: {magical_weather}",
        summary=f"Weather ({region}): {climate}",
        importance=0.6,
        turn_number=graph.world.turn_number,
    )
    memory.save()

    # Push to simulation
    _sync_to_sim("weather", {
        "region": region, "climate": climate,
        "magical_weather": magical_weather,
    }, graph)

    return {
        "region": region,
        "climate": climate,
        "magical_weather": magical_weather,
        "memory_id": mem.id,
        "instruction": (
            f"Define weather patterns for {region} ({climate}): "
            f"seasonal variations, storm patterns, average conditions. "
            f"{'Include magical weather: ' + magical_weather if magical_weather else ''}"
        ),
    }
