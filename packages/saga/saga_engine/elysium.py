"""
Elysium — World Seed Data
===========================

The initial world state for the community story "Elysium".

Lore:
    Elysium is a realm between worlds — a living story shaped by those
    who enter it. Once a true paradise, it was fractured by the Convergence,
    an event that merged fragments of countless realities into one.

    Now it is a patchwork of wonder and danger: forests that remember,
    seas of light, fortresses built on the ruins of forgotten worlds,
    and the Void Margins where reality thins to nothing.

    The community shapes its fate. Every character created, every choice
    made, every tale told becomes part of Elysium's living history.

This module creates the starter nodes, edges, and memories that give
players something rich to work with from turn one.
"""

import logging

from .graph import StoryGraph
from .memory import MemoryManager
from .models import (
    EdgeType,
    MemoryType,
    NodeStatus,
    NodeType,
    StoryEdge,
    StoryMemory,
    StoryNode,
)

logger = logging.getLogger("Saga.Elysium")

# ============================================================================
# THE WORLD
# ============================================================================

WORLD_LORE = StoryNode(
    id="lore-elysium-origin",
    type=NodeType.LORE,
    name="The Origin of Elysium",
    description=(
        "Elysium was once a true paradise — a realm of eternal twilight where the "
        "boundaries between worlds were thin and beautiful. Then came the Convergence: "
        "a catastrophic event that shattered the barriers between realities, merging "
        "fragments of countless worlds into one. Mountains from one reality collide "
        "with oceans from another. Cities half-exist, their architecture from different "
        "civilizations fused together. Time itself flows unevenly — some regions age "
        "centuries in moments while others hang frozen. The cause of the Convergence "
        "is unknown. Its effects are still unfolding."
    ),
    short_description="A paradise fractured by the Convergence — fragments of countless realities merged into one",
    icon="🌍",
    pinned=True,
    tags=["core-lore", "convergence", "origin"],
    created_by="system",
)

# ============================================================================
# LOCATIONS
# ============================================================================

LOCATIONS = [
    StoryNode(
        id="loc-shattered-agora",
        type=NodeType.LOCATION,
        name="The Shattered Agora",
        description=(
            "The central meeting place of Elysium — a vast plaza where fragments of "
            "different realities are visible in the architecture. Marble columns from "
            "a Greco-Roman world stand beside crystalline spires from somewhere alien. "
            "The ground is a mosaic of different stones from different worlds, and at "
            "its center, a great rift in the air shimmers with the light of other "
            "realities. This is where newcomers arrive and travelers gather."
        ),
        short_description="Central plaza where realities overlap — the hub of Elysium",
        icon="🏛️",
        aliases=["the agora", "agora", "central plaza", "the plaza"],
        tags=["hub", "safe-zone", "convergence-point"],
        created_by="system",
    ),
    StoryNode(
        id="loc-verdant-depths",
        type=NodeType.LOCATION,
        name="The Verdant Depths",
        description=(
            "A vast, ancient forest where the trees are conscious. They remember "
            "everything — every footstep, every whispered word, every drop of blood "
            "shed beneath their canopy. The deeper you go, the older the memories. "
            "At the forest's heart lies the Remembering Grove, where the oldest tree "
            "holds memories from before the Convergence. Strange creatures move between "
            "the trunks — some from Elysium, some from the worlds that merged into it."
        ),
        short_description="A sentient forest whose trees remember everything",
        icon="🌲",
        aliases=["the forest", "verdant depths", "the depths"],
        tags=["dangerous", "ancient", "memory-source"],
        created_by="system",
    ),
    StoryNode(
        id="loc-ironhold",
        type=NodeType.LOCATION,
        name="Ironhold",
        description=(
            "A fortress city built on the convergence of three realities — medieval "
            "stone, industrial steel, and something that looks almost organic. The "
            "Iron Covenant rules here with pragmatic discipline. Ironhold's walls are "
            "the most defensible position in Elysium, but its rigid order chafes those "
            "who value freedom. The forges never stop burning, and the sound of hammers "
            "rings day and night."
        ),
        short_description="Fortress city ruled by the Iron Covenant — order amid chaos",
        icon="🏰",
        aliases=["ironhold", "the fortress", "iron hold"],
        tags=["civilization", "military", "trade"],
        created_by="system",
    ),
    StoryNode(
        id="loc-luminous-sea",
        type=NodeType.LOCATION,
        name="The Luminous Sea",
        description=(
            "Not water, but light — a vast shimmering expanse that laps at shores "
            "like an ocean made of captured starlight. Those who sail it don't use "
            "boats but thought-vessels: constructs shaped by will and memory. Islands "
            "of solid light dot the sea, each one a fragment of a reality that didn't "
            "fully merge. Some contain treasures. Some contain horrors. The sea responds "
            "to emotion — calm thoughts make it gentle, fear makes it storm."
        ),
        short_description="An ocean of living light, navigated by thought and will",
        icon="🌊",
        aliases=["the sea", "luminous sea", "the light sea"],
        tags=["exploration", "dangerous", "surreal"],
        created_by="system",
    ),
    StoryNode(
        id="loc-void-margins",
        type=NodeType.LOCATION,
        name="The Void Margins",
        description=(
            "The edges of Elysium, where reality thins to almost nothing. Here the "
            "Convergence is still actively happening — new fragments of other worlds "
            "occasionally drift in from the void. The ground is unstable. Physics is "
            "unreliable. Those who spend too long here begin to change — the Void "
            "Touched bear its marks. Strange whispers carry on the non-wind, and some "
            "claim to have seen shapes in the void that are watching."
        ),
        short_description="The unstable edges of reality where the Convergence still churns",
        icon="🕳️",
        aliases=["the margins", "void margins", "the void", "the edge"],
        tags=["dangerous", "mysterious", "convergence-active"],
        created_by="system",
    ),
]

# ============================================================================
# CHARACTERS (Starter NPCs)
# ============================================================================

CHARACTERS = [
    StoryNode(
        id="char-chronicler",
        type=NodeType.CHARACTER,
        name="The Chronicler",
        description=(
            "A hooded figure who appears at significant moments, recording everything "
            "in a book that seems to have no last page. No one knows the Chronicler's "
            "true face or origin. They speak in measured, poetic tones, and sometimes "
            "offer cryptic guidance. The Chronicler claims to be neither from Elysium "
            "nor from any of the merged realities — they say they are 'from the story "
            "itself.' Their quill never runs dry."
        ),
        short_description="Mysterious recorder of all events — claims to be 'from the story itself'",
        icon="📖",
        properties={
            "personality": "Enigmatic, poetic, all-knowing but rarely direct",
            "appearance": "Hooded figure in shifting robes, face always in shadow, carries an endless book",
        },
        aliases=["chronicler", "the recorder", "the keeper"],
        tags=["narrator", "guide", "mystery"],
        created_by="system",
    ),
    StoryNode(
        id="char-lyra",
        type=NodeType.CHARACTER,
        name="Lyra the Wayfinder",
        description=(
            "A warm, practical guide who helps newcomers navigate the merged world. "
            "Lyra was among the first to arrive in Elysium after the Convergence — "
            "she remembers the moment the realities merged. She carries a compass that "
            "points not north but toward 'what you need to find.' Brave, compassionate, "
            "and endlessly curious. She knows more about the geography of Elysium than "
            "anyone alive and has mapped paths between regions that others thought impassable."
        ),
        short_description="Warm, brave guide with a compass that points to what you need",
        icon="🧭",
        properties={
            "personality": "Warm, brave, curious, practical, compassionate",
            "appearance": "Travel-worn clothes, copper hair braided with trinkets from different worlds, magic compass on a chain",
        },
        aliases=["lyra", "the wayfinder", "wayfinder"],
        tags=["guide", "friendly", "explorer"],
        created_by="system",
    ),
    StoryNode(
        id="char-thorne",
        type=NodeType.CHARACTER,
        name="Commander Thorne",
        description=(
            "Leader of Ironhold and the Iron Covenant. A scarred military veteran who "
            "believes order is the only thing preventing Elysium from tearing itself "
            "apart. Thorne is not cruel but is ruthless when he believes safety is at "
            "stake. He distrusts magic, the Void Margins, and anyone who doesn't follow "
            "rules. Deep down, he fears the Convergence will resume and swallow what's "
            "left. His iron prosthetic left arm is from a reality where clockwork and "
            "flesh are one."
        ),
        short_description="Ironhold's scarred commander — order above all, fears the Convergence",
        icon="⚔️",
        properties={
            "personality": "Disciplined, pragmatic, distrustful of magic, secretly fearful",
            "appearance": "Scarred face, iron clockwork prosthetic arm, heavy armor, grey-streaked hair",
        },
        aliases=["thorne", "commander thorne", "the commander"],
        tags=["military", "leader", "complex"],
        created_by="system",
    ),
    StoryNode(
        id="char-whisper",
        type=NodeType.CHARACTER,
        name="Whisper",
        description=(
            "A shapeshifting entity from the Void Margins — or perhaps the Void itself "
            "given form. Whisper can take any shape but favors a slight, pale figure "
            "with eyes that shift color. They speak in fragments and riddles, and their "
            "knowledge of the Convergence is unmatched but delivered in maddening pieces. "
            "Some believe Whisper is trying to help. Others believe Whisper is the "
            "Convergence's agent. The truth may be stranger than either."
        ),
        short_description="Shapeshifting entity from the Void — knows the Convergence's secrets",
        icon="👁️",
        properties={
            "personality": "Cryptic, amoral, playful, alien, occasionally helpful",
            "appearance": "Shifts constantly — most often a pale figure with color-changing eyes",
        },
        aliases=["whisper", "the void walker", "void touched"],
        tags=["mysterious", "void", "dangerous", "knowledge"],
        created_by="system",
    ),
]

# ============================================================================
# FACTIONS
# ============================================================================

FACTIONS = [
    StoryNode(
        id="fac-convergence-watch",
        type=NodeType.FACTION,
        name="The Convergence Watch",
        description=(
            "Scholars, mages, and scientists who study the Convergence. They seek "
            "to understand why the realities merged and whether the process can be "
            "controlled or reversed. Based in the Shattered Agora, they maintain the "
            "largest library of pre-Convergence knowledge. Neutral in political conflicts, "
            "they serve truth above allegiance."
        ),
        short_description="Scholars studying the Convergence — seekers of truth",
        icon="🔭",
        aliases=["the watch", "convergence watch", "the scholars"],
        tags=["scholarly", "neutral", "knowledge"],
        created_by="system",
    ),
    StoryNode(
        id="fac-iron-covenant",
        type=NodeType.FACTION,
        name="The Iron Covenant",
        description=(
            "Ironhold's military order. They believe that strict organization and "
            "military discipline are the only things preventing Elysium from descending "
            "into chaos. They patrol borders, enforce laws in their territory, and view "
            "the Void Margins as an existential threat to be contained."
        ),
        short_description="Ironhold's military order — discipline and defense above all",
        icon="🛡️",
        aliases=["iron covenant", "the covenant", "ironhold military"],
        tags=["military", "order", "defensive"],
        created_by="system",
    ),
    StoryNode(
        id="fac-wanderers",
        type=NodeType.FACTION,
        name="The Wanderers",
        description=(
            "Nomadic travelers who roam between reality fragments. They believe the "
            "merged world should be explored freely, not walled off or controlled. "
            "The Wanderers are the best pathfinders in Elysium and trade in stories, "
            "relics, and maps. They clash with Ironhold over freedom of movement."
        ),
        short_description="Nomadic explorers who believe Elysium should be free and open",
        icon="🗺️",
        aliases=["the wanderers", "wanderers", "nomads"],
        tags=["exploration", "freedom", "trade"],
        created_by="system",
    ),
    StoryNode(
        id="fac-void-touched",
        type=NodeType.FACTION,
        name="The Void Touched",
        description=(
            "Those who have been changed by exposure to the Void Margins. They hear "
            "the whispers, see things others can't, and some have developed abilities "
            "that shouldn't be possible. Feared and misunderstood by most. Some embrace "
            "their transformation. Others desperately seek a cure."
        ),
        short_description="People changed by the Void — feared, gifted, and divided",
        icon="✨",
        aliases=["void touched", "the changed", "the marked"],
        tags=["outcast", "powerful", "mystery"],
        created_by="system",
    ),
]

# ============================================================================
# PLOT THREADS (Starters)
# ============================================================================

PLOT_THREADS = [
    StoryNode(
        id="plot-convergence-continues",
        type=NodeType.PLOT_THREAD,
        name="The Convergence Continues",
        description=(
            "The Convergence isn't over. New fragments of other realities keep "
            "appearing at the Void Margins. Each one brings new wonders and new "
            "dangers. The Watch studies them. Ironhold fortifies against them. "
            "The Wanderers explore them. But no one knows when — or if — it will end."
        ),
        short_description="New reality fragments keep appearing — the Convergence isn't over",
        icon="🌀",
        tags=["main-plot", "ongoing", "mystery"],
        created_by="system",
    ),
    StoryNode(
        id="plot-void-stirring",
        type=NodeType.PLOT_THREAD,
        name="Something Stirs in the Void",
        description=(
            "Travelers near the Void Margins report seeing shapes — vast, deliberate "
            "shapes moving in the emptiness beyond reality. Whisper grows agitated. "
            "The Void Touched have nightmares of an eye opening. Something is aware "
            "of Elysium, and it is turning its attention here."
        ),
        short_description="Something vast and deliberate is watching from beyond the Void",
        icon="👁️",
        tags=["main-plot", "threat", "mystery"],
        created_by="system",
    ),
    StoryNode(
        id="plot-ironhold-tension",
        type=NodeType.PLOT_THREAD,
        name="Ironhold's Grip Tightens",
        description=(
            "Commander Thorne has begun restricting access to the Void Margins and "
            "demanding Wanderers register with Ironhold or face expulsion. The "
            "Wanderers refuse. Tensions are rising. Lyra is caught in the middle, "
            "respected by both sides but loyal to neither."
        ),
        short_description="Ironhold restricts the Wanderers — freedom vs. security",
        icon="⚔️",
        tags=["conflict", "political", "ongoing"],
        created_by="system",
    ),
]


# ============================================================================
# SEED FUNCTION
# ============================================================================

def seed_elysium(graph: StoryGraph, mem: MemoryManager):
    """
    Seed the Elysium world into a fresh StoryGraph.
    Only runs if the graph is empty.
    """
    if graph.node_count > 0:
        logger.info("Graph already populated — skipping seed")
        return False

    logger.info("🌍 Seeding Elysium...")

    # World state
    graph.world.story_name = "Elysium"
    graph.world.current_location = "loc-shattered-agora"
    graph.world.present_characters = ["char-chronicler", "char-lyra"]
    graph.world.active_plot_threads = [
        "plot-convergence-continues",
        "plot-void-stirring",
        "plot-ironhold-tension",
    ]
    graph.world.mood = "mysterious"
    graph.world.time_of_day = "twilight"
    graph.world.weather = "shimmering"

    # Add lore
    graph.add_node(WORLD_LORE)

    # Add locations
    for loc in LOCATIONS:
        graph.add_node(loc)

    # Location nesting: Agora contains sub-areas conceptually
    graph.add_edge(StoryEdge(
        type=EdgeType.CONTAINS,
        source_id="loc-shattered-agora", target_id="loc-verdant-depths",
        label="The Verdant Depths border the Agora to the south",
        weight=0.5,
    ))

    # Add characters
    for char in CHARACTERS:
        graph.add_node(char)

    # Character → Location edges
    graph.add_edge(StoryEdge(
        type=EdgeType.LOCATED_AT,
        source_id="char-chronicler", target_id="loc-shattered-agora",
        label="The Chronicler is often found at the Shattered Agora",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.LOCATED_AT,
        source_id="char-lyra", target_id="loc-shattered-agora",
        label="Lyra frequents the Shattered Agora to greet newcomers",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.LOCATED_AT,
        source_id="char-thorne", target_id="loc-ironhold",
        label="Commander Thorne rules from Ironhold",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.LOCATED_AT,
        source_id="char-whisper", target_id="loc-void-margins",
        label="Whisper drifts through the Void Margins",
    ))

    # Add factions
    for fac in FACTIONS:
        graph.add_node(fac)

    # Character → Faction edges
    graph.add_edge(StoryEdge(
        type=EdgeType.MEMBER_OF,
        source_id="char-thorne", target_id="fac-iron-covenant",
        label="Commander Thorne leads the Iron Covenant",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.MEMBER_OF,
        source_id="char-lyra", target_id="fac-wanderers",
        label="Lyra is the most respected of the Wanderers",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.MEMBER_OF,
        source_id="char-whisper", target_id="fac-void-touched",
        label="Whisper is the most powerful of the Void Touched",
    ))

    # Character relationships
    graph.add_edge(StoryEdge(
        type=EdgeType.KNOWS,
        source_id="char-lyra", target_id="char-thorne",
        label="Lyra and Thorne respect each other but disagree on freedom vs. order",
        bidirectional=True,
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.KNOWS,
        source_id="char-lyra", target_id="char-chronicler",
        label="Lyra trusts the Chronicler and sometimes asks for guidance",
        bidirectional=True,
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.KNOWS,
        source_id="char-whisper", target_id="char-chronicler",
        label="Whisper and the Chronicler have an ancient, uneasy understanding",
        bidirectional=True,
    ))

    # Faction relationships
    graph.add_edge(StoryEdge(
        type=EdgeType.HOSTILE_TO,
        source_id="fac-iron-covenant", target_id="fac-wanderers",
        label="The Iron Covenant restricts Wanderer movement — tensions are high",
        bidirectional=True,
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.HOSTILE_TO,
        source_id="fac-iron-covenant", target_id="fac-void-touched",
        label="Ironhold distrusts and monitors the Void Touched",
    ))
    graph.add_edge(StoryEdge(
        type=EdgeType.ALLIED_WITH,
        source_id="fac-convergence-watch", target_id="fac-wanderers",
        label="The Watch and Wanderers share information freely",
        bidirectional=True,
    ))

    # Add plot threads
    for thread in PLOT_THREADS:
        graph.add_node(thread)

    # Save the graph
    graph.save()

    # ── Seed memories ──
    mem.create(
        memory_type=MemoryType.EPISODIC,
        content="The Convergence shattered the barriers between realities. No one knows exactly when or why, but the merging continues to this day.",
        summary="The Convergence merged realities — cause unknown, still ongoing",
        importance=0.9,
        related_nodes=["lore-elysium-origin"],
        tags=["convergence", "core-lore"],
        turn_number=0,
        pinned=True,
        created_by="system",
    )
    mem.create(
        memory_type=MemoryType.SEMANTIC,
        content="The Shattered Agora is the safest place in Elysium and serves as the main gathering point for travelers and newcomers.",
        summary="The Agora is the safe central hub of Elysium",
        importance=0.7,
        related_nodes=["loc-shattered-agora"],
        tags=["location", "safety"],
        turn_number=0,
        created_by="system",
    )
    mem.create(
        memory_type=MemoryType.PROCEDURAL,
        content="The Void Margins are extremely dangerous. Reality is unstable there. Prolonged exposure causes physical and mental changes. The Void Touched are proof.",
        summary="The Void Margins are dangerous — prolonged exposure changes you",
        importance=0.8,
        related_nodes=["loc-void-margins", "fac-void-touched"],
        tags=["danger", "void", "rule"],
        turn_number=0,
        pinned=True,
        created_by="system",
    )
    mem.create(
        memory_type=MemoryType.EMOTIONAL,
        content="The people of Elysium live in a constant state of wonder and unease. The merged world is beautiful but deeply strange, and no one knows what tomorrow will bring.",
        summary="Elysium's mood: wonder mixed with unease — beauty and strangeness",
        importance=0.5,
        related_nodes=["lore-elysium-origin"],
        tags=["mood", "atmosphere"],
        turn_number=0,
        created_by="system",
    )
    mem.create(
        memory_type=MemoryType.SEMANTIC,
        content="Lyra's compass doesn't point north — it points toward what you need to find. She discovered this property by accident during her first expedition into the Verdant Depths.",
        summary="Lyra's compass points to what you need — discovered in the Verdant Depths",
        importance=0.6,
        related_nodes=["char-lyra", "loc-verdant-depths"],
        tags=["character-lore", "magic-item"],
        turn_number=0,
        created_by="system",
    )

    mem.save()

    node_count = graph.node_count
    edge_count = graph.edge_count
    mem_count = mem.count
    logger.info(
        f"🌍 Elysium seeded: {node_count} nodes, {edge_count} edges, {mem_count} memories"
    )
    return True
