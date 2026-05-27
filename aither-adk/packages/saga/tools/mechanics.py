"""RPG/Game Mechanics Tools — Structured game interactions.

Combat encounters, skill challenges, level progression, quest generation,
random encounters, loot tables, dungeon generation, shops, and status effects.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.mechanics")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


def _run_async(coro):
    from .story_turn import _run_async
    return _run_async(coro)


@tool(
    name="combat_encounter",
    description=(
        "Run a structured combat encounter with initiative order, actions per turn, "
        "HP tracking, and outcome narration."
    ),
)
def combat_encounter(
    enemies: str,
    terrain: str = "open field",
    surprise: str = "none",
    difficulty: str = "medium",
) -> dict:
    """Set up a combat encounter.

    Args:
        enemies: Semicolon-separated enemy descriptions (e.g., "Goblin Scout;Goblin Archer;Goblin Chief")
        terrain: Battle terrain (open field, forest, dungeon, rooftop, ship deck, cave)
        surprise: Who has surprise (none, players, enemies)
        difficulty: Encounter difficulty (easy, medium, hard, deadly)
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    # Get player characters in scene
    player_chars = []
    for char_id in graph.world.present_characters:
        char = graph.get_node(char_id)
        if char:
            player_chars.append({
                "name": char.name,
                "stats": char.properties.get("stats", {}),
            })

    enemy_list = [e.strip() for e in enemies.split(";") if e.strip()]

    # Roll initiative for all combatants
    combatants = []
    for pc in player_chars:
        init = random.randint(1, 20) + pc.get("stats", {}).get("dexterity", 0)
        combatants.append({"name": pc["name"], "side": "player", "initiative": init, "hp": 100})
    for enemy in enemy_list:
        init = random.randint(1, 20)
        hp_map = {"easy": 30, "medium": 50, "hard": 80, "deadly": 120}
        combatants.append({
            "name": enemy, "side": "enemy", "initiative": init,
            "hp": hp_map.get(difficulty, 50),
        })

    combatants.sort(key=lambda c: c["initiative"], reverse=True)

    return {
        "terrain": terrain,
        "surprise": surprise,
        "difficulty": difficulty,
        "combatants": combatants,
        "initiative_order": [c["name"] for c in combatants],
        "instruction": (
            f"Run this combat encounter on {terrain}. "
            f"{'Surprise round for ' + surprise + '! ' if surprise != 'none' else ''}"
            f"For each round: describe actions, call roll_dice for attacks and damage, "
            f"track HP, narrate the action cinematically. "
            f"End when one side is defeated, flees, or surrenders."
        ),
    }


@tool(
    name="skill_check_complex",
    description=(
        "Run a multi-step skill challenge: 3 successes before 3 failures, "
        "with narrative consequences at each stage."
    ),
)
def skill_check_complex(
    challenge: str,
    skills_allowed: str = "any",
    successes_needed: int = 3,
    failures_allowed: int = 3,
    base_dc: int = 12,
) -> dict:
    """Set up a complex skill challenge.

    Args:
        challenge: Description of the challenge
        skills_allowed: Comma-separated allowed skills (or 'any')
        successes_needed: Successes required to win
        failures_allowed: Failures before losing
        base_dc: Base difficulty class
    """
    return {
        "challenge": challenge,
        "skills_allowed": skills_allowed.split(",") if skills_allowed != "any" else "any",
        "successes_needed": successes_needed,
        "failures_allowed": failures_allowed,
        "base_dc": base_dc,
        "current_successes": 0,
        "current_failures": 0,
        "instruction": (
            f"Run a skill challenge: '{challenge}'. "
            f"Need {successes_needed} successes before {failures_allowed} failures. "
            f"Each attempt: player chooses a skill, you set DC (base {base_dc}, "
            f"adjust for creativity), call roll_dice, narrate result. "
            f"Each success/failure changes the narrative situation. "
            f"Partial success on close rolls is fine."
        ),
    }


@tool(
    name="level_up",
    description=(
        "Level up a character: stat increases, new abilities, and a "
        "narrative transformation scene."
    ),
)
def level_up(
    character_name: str,
    new_level: int = 0,
    stat_increase: str = "",
    new_ability: str = "",
) -> dict:
    """Level up a character.

    Args:
        character_name: Character to level up
        new_level: New level (auto-increments if 0)
        stat_increase: Which stat to increase (strength, dexterity, wisdom, etc.)
        new_ability: Name of new ability gained
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    stats = node.properties.get("stats", {})
    current_level = stats.get("level", 1)
    next_level = new_level if new_level > 0 else current_level + 1

    # Update stats
    stats["level"] = next_level
    if stat_increase and stat_increase in stats:
        stats[stat_increase] = stats.get(stat_increase, 10) + 1

    if new_ability:
        abilities = stats.get("abilities", [])
        abilities.append(new_ability)
        stats["abilities"] = abilities

    props = dict(node.properties)
    props["stats"] = stats
    graph.update_node(node.id, properties=props)

    # Store as memory
    mem = memory.create(
        memory_type=MemoryType.EPISODIC,
        content=f"{node.name} reached level {next_level}! {f'New ability: {new_ability}' if new_ability else ''}",
        summary=f"{node.name} leveled up to {next_level}",
        importance=0.8,
        related_nodes=[node.id],
        turn_number=graph.world.turn_number,
    )

    graph.save()
    memory.save()

    return {
        "character": node.name,
        "old_level": current_level,
        "new_level": next_level,
        "stats": stats,
        "new_ability": new_ability,
        "memory_id": mem.id,
        "instruction": (
            f"Narrate {node.name}'s level up to {next_level}. "
            f"Describe the transformation: what they've learned, how they've grown. "
            f"{'New ability: ' + new_ability + '.' if new_ability else ''} "
            f"Make it feel earned and meaningful."
        ),
    }


@tool(
    name="quest_generator",
    description=(
        "Generate a procedural quest with hook, objective, complications, "
        "and reward. Registers as a PLOT_THREAD node."
    ),
)
def quest_generator(
    quest_type: str = "retrieval",
    difficulty: str = "medium",
    patron: str = "",
) -> dict:
    """Generate a quest.

    Args:
        quest_type: Type (retrieval, escort, investigation, assassination, rescue, diplomacy, exploration)
        difficulty: Difficulty (easy, medium, hard, legendary)
        patron: Quest giver name (optional)
    """
    graph, _, _ = _get_engine()
    from saga_engine.models import NodeType

    characters = [c.name for c in graph.find_nodes_by_type(NodeType.CHARACTER)[:10]]
    locations = [l.name for l in graph.find_nodes_by_type(NodeType.LOCATION)[:10]]
    factions = [f.name for f in graph.find_nodes_by_type(NodeType.FACTION)]

    result = {
        "quest_type": quest_type,
        "difficulty": difficulty,
        "patron": patron,
        "known_characters": characters,
        "known_locations": locations,
        "known_factions": factions,
    }

    # NanoGPT quest hook
    try:
        from saga_engine.nanogpt_gen import get_asset_generator
        from .story_turn import _get_data_dir
        gen = get_asset_generator(_get_data_dir())
        hook = _run_async(gen.generate_quest_hook(graph))
        if hook:
            result["generated_hook"] = hook
    except Exception:
        pass

    result["instruction"] = (
        f"Generate a {difficulty} {quest_type} quest. Structure: "
        f"1) HOOK — how the player learns about it"
        + (f" (suggested: '{result.get('generated_hook', '')}') " if result.get("generated_hook") else " ") +
        f"2) OBJECTIVE — what must be accomplished, "
        f"3) COMPLICATIONS — 2-3 obstacles or twists, "
        f"4) REWARD — tangible reward + story consequence. "
        f"{'Quest giver: ' + patron + '. ' if patron else ''}"
        f"Register as a PLOT_THREAD node in the graph."
    )
    return result


@tool(
    name="random_encounter",
    description="Roll on an encounter table weighted by location danger and time of day.",
)
def random_encounter(
    location: str = "",
    danger_level: int = 3,
    time_of_day: str = "",
) -> dict:
    """Generate a random encounter.

    Args:
        location: Current location name
        danger_level: Area danger level (1=safe, 5=deadly)
        time_of_day: Time of day (dawn, day, dusk, night)
    """
    graph, _, _ = _get_engine()

    danger_level = max(1, min(5, danger_level))
    time = time_of_day or graph.world.time_of_day

    # Encounter type probabilities by danger level
    encounter_types = {
        1: ["friendly_traveler", "merchant", "lost_animal", "scenic_vista", "nothing"],
        2: ["friendly_traveler", "merchant", "minor_creature", "puzzle", "nothing"],
        3: ["hostile_creature", "bandits", "puzzle", "merchant", "mysterious_stranger"],
        4: ["hostile_creature", "ambush", "elite_monster", "trap", "mysterious_stranger"],
        5: ["boss_creature", "ambush", "cursed_ground", "elite_monster", "dragon"],
    }

    encounter_pool = encounter_types.get(danger_level, encounter_types[3])
    # Night encounters skew more dangerous
    if time == "night":
        encounter_pool = [e for e in encounter_pool if e != "nothing" and e != "merchant"]
        encounter_pool.append("undead")

    selected = random.choice(encounter_pool)

    result = {
        "location": location or "unknown",
        "danger_level": danger_level,
        "time_of_day": time,
        "encounter_type": selected,
    }

    # Pull narrative context from simulation for richer encounters
    try:
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        if sim and graph.world.prometheus_world_id:
            ctx = _run_async(sim.get_narrative_context(
                graph.world.prometheus_world_id,
                player_location=location or "",
            ))
            if ctx:
                result["nearby_npcs"] = ctx.get("nearby_npcs", [])[:3]
                result["sim_weather"] = ctx.get("weather", "")
    except Exception:
        pass

    result["instruction"] = (
        f"Generate a {selected} encounter at danger level {danger_level}, "
        f"time: {time}. Include: description, threat assessment, "
        f"possible approaches (fight, flee, negotiate, sneak). "
        f"If hostile: set up combat_encounter. If social: roleplay the NPC."
    )
    return result


@tool(
    name="loot_table",
    description="Generate loot drops with rarity distribution. Registers items in the graph.",
)
def loot_table(
    source: str,
    count: int = 3,
    min_rarity: str = "common",
    max_rarity: str = "rare",
) -> dict:
    """Generate loot.

    Args:
        source: What was looted (enemy name, chest, etc.)
        count: Number of items to generate
        min_rarity: Minimum rarity (common, uncommon, rare, epic, legendary)
        max_rarity: Maximum rarity
    """
    rarity_weights = {
        "common": 50, "uncommon": 30, "rare": 15, "epic": 4, "legendary": 1,
    }

    # Filter to min-max range
    rarities = list(rarity_weights.keys())
    min_idx = rarities.index(min_rarity) if min_rarity in rarities else 0
    max_idx = rarities.index(max_rarity) if max_rarity in rarities else 2

    available = rarities[min_idx:max_idx + 1]
    weights = [rarity_weights[r] for r in available]

    # Roll rarities
    rolled_rarities = random.choices(available, weights=weights, k=min(count, 10))

    result = {
        "source": source,
        "items_to_generate": count,
        "rolled_rarities": rolled_rarities,
    }

    # NanoGPT item descriptions for each rarity
    try:
        from saga_engine.nanogpt_gen import get_asset_generator
        from .story_turn import _get_data_dir, _get_engine
        graph, _, _ = _get_engine()
        gen = get_asset_generator(_get_data_dir())
        descs = []
        for rarity in rolled_rarities:
            desc = _run_async(gen.generate_item_description(rarity, graph))
            descs.append({"rarity": rarity, "suggested_description": desc})
        if descs:
            result["generated_items"] = descs
    except Exception:
        pass

    result["instruction"] = (
        f"Generate {count} loot items from '{source}'. "
        f"Rarities rolled: {rolled_rarities}. "
        + ("Use the generated item descriptions as starting points. " if result.get("generated_items") else "") +
        f"For each: name, description, rarity, type (weapon/armor/potion/trinket/material), "
        f"value in currency. Register as ITEM nodes with create_item."
    )
    return result


@tool(
    name="dungeon_generator",
    description="Generate a procedural dungeon with rooms, traps, encounters, and treasure.",
)
def dungeon_generator(
    dungeon_name: str,
    room_count: int = 5,
    theme: str = "generic",
    boss: str = "",
) -> dict:
    """Generate a dungeon layout.

    Args:
        dungeon_name: Name of the dungeon
        room_count: Number of rooms (3-10)
        theme: Theme (crypt, mine, temple, sewer, castle, cave, library, elemental)
        boss: Boss enemy name (optional)
    """
    room_count = max(3, min(10, room_count))

    # Generate room types
    room_types = ["entrance", "corridor", "chamber", "trap_room", "puzzle_room",
                  "treasure_room", "guard_room", "boss_room"]

    rooms = [{"number": 1, "type": "entrance"}]
    for i in range(2, room_count):
        rooms.append({"number": i, "type": random.choice(room_types[1:-1])})
    rooms.append({"number": room_count, "type": "boss_room"})

    return {
        "dungeon_name": dungeon_name,
        "theme": theme,
        "room_count": room_count,
        "rooms": rooms,
        "boss": boss,
        "instruction": (
            f"Generate dungeon '{dungeon_name}' ({theme} theme, {room_count} rooms). "
            f"For each room: description, contents (enemies/traps/treasure/puzzle), "
            f"exits (which rooms connect). Room types: {[r['type'] for r in rooms]}. "
            f"{'Boss: ' + boss + '.' if boss else 'Generate a boss.'} "
            f"Register the dungeon as a LOCATION node."
        ),
    }


@tool(
    name="npc_shop",
    description="Interactive shop with inventory, prices, and haggling.",
)
def npc_shop(
    shop_type: str = "general",
    shopkeeper: str = "",
    quality: str = "standard",
) -> dict:
    """Set up a shop interaction.

    Args:
        shop_type: Shop type (general, weapons, armor, potions, magic, food, luxury, black_market)
        shopkeeper: Shopkeeper name
        quality: Stock quality (poor, standard, fine, masterwork)
    """
    graph, _, _ = _get_engine()

    result = {
        "shop_type": shop_type,
        "shopkeeper": shopkeeper,
        "quality": quality,
        "player_gold": 100,
    }

    # Pull economy data from simulation
    try:
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        if sim and graph.world.prometheus_world_id:
            state = _run_async(sim.get_state(graph.world.prometheus_world_id))
            if state and state.economy:
                result["economy"] = {
                    "inflation": state.economy.get("inflation", 1.0),
                    "market_mood": state.economy.get("market_mood", "normal"),
                    "currency": state.economy.get("currency", "gold"),
                }
    except Exception:
        pass

    econ_note = ""
    if result.get("economy"):
        econ = result["economy"]
        econ_note = f" Economy: {econ.get('market_mood', 'normal')} market, inflation {econ.get('inflation', 1.0):.1f}x."

    result["instruction"] = (
        f"Set up a {quality} {shop_type} shop"
        f"{' run by ' + shopkeeper if shopkeeper else ''}.{econ_note} "
        f"Generate 5-8 items with prices. Allow: browse, buy, sell, haggle. "
        f"Haggling: roll_dice d20 vs DC 12+. Success = 10-20% discount. "
        f"Shopkeeper has personality and opinions about their goods."
    )
    return result


@tool(
    name="status_effects",
    description="Apply or track status effects (buffs/debuffs) with duration.",
)
def status_effects(
    character_name: str,
    effect: str,
    duration: str = "3 turns",
    effect_type: str = "debuff",
) -> dict:
    """Apply a status effect to a character.

    Args:
        character_name: Character to affect
        effect: Effect name (poisoned, blessed, frightened, invisible, haste, etc.)
        duration: How long it lasts (e.g., "3 turns", "until dawn", "permanent")
        effect_type: Type (buff, debuff, neutral)
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import MemoryType

    node = graph.find_node_by_name(character_name)
    if not node:
        return {"error": f"Character '{character_name}' not found"}

    # Track in character properties
    props = dict(node.properties)
    effects = props.get("status_effects", [])
    effects.append({
        "effect": effect,
        "type": effect_type,
        "duration": duration,
        "applied_turn": graph.world.turn_number,
    })
    props["status_effects"] = effects
    graph.update_node(node.id, properties=props)

    # Store as memory for context
    mem = memory.create(
        memory_type=MemoryType.EPISODIC,
        content=f"{node.name} is affected by {effect} ({effect_type}, duration: {duration})",
        summary=f"{node.name}: {effect} ({duration})",
        importance=0.6,
        related_nodes=[node.id],
        turn_number=graph.world.turn_number,
        tags=["status_effect", effect_type],
    )

    graph.save()
    memory.save()

    return {
        "character": node.name,
        "effect": effect,
        "type": effect_type,
        "duration": duration,
        "active_effects": effects,
        "memory_id": mem.id,
    }


@tool(
    name="rest_and_recovery",
    description="Long or short rest with healing, dream sequences, and camp events.",
)
def rest_and_recovery(
    rest_type: str = "short",
    location: str = "",
) -> dict:
    """Rest and recover.

    Args:
        rest_type: Type of rest (short, long, full)
        location: Where they're resting
    """
    graph, _, _ = _get_engine()

    # Gather characters with status effects
    from saga_engine.models import NodeType
    characters = []
    for char_id in graph.world.present_characters:
        char = graph.get_node(char_id)
        if char:
            effects = char.properties.get("status_effects", [])
            characters.append({
                "name": char.name,
                "effects": effects,
            })

    healing = {"short": "partial", "long": "full", "full": "full+buffs"}

    return {
        "rest_type": rest_type,
        "location": location or "camp",
        "healing": healing.get(rest_type, "partial"),
        "characters": characters,
        "instruction": (
            f"Narrate a {rest_type} rest at {location or 'camp'}. "
            f"Healing: {healing.get(rest_type, 'partial')}. "
            f"Clear expired status effects. "
            f"{'Generate a dream sequence for one character. ' if rest_type == 'long' else ''}"
            f"{'Roll for a camp event (d20: 1-5 = encounter, 6-15 = peaceful, 16-20 = boon). ' if rest_type == 'long' else ''}"
            f"Advance time_of_day appropriately."
        ),
    }
