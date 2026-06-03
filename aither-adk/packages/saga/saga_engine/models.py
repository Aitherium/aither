"""
StoryGraph Data Models
=======================

Every object in the story world is a typed, auditable data structure.
No magic dicts — everything is explicit Pydantic so the frontend can
render it and users can inspect/edit it.
"""

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS — The vocabulary of the story world
# ============================================================================

class NodeType(str, Enum):
    """What kind of thing this node represents."""
    CHARACTER = "character"
    LOCATION = "location"
    ITEM = "item"
    FACTION = "faction"
    EVENT = "event"
    LORE = "lore"
    PLOT_THREAD = "plot_thread"
    SCENE = "scene"


class EdgeType(str, Enum):
    """How two nodes relate to each other."""
    LOCATED_AT = "located_at"         # character → location
    KNOWS = "knows"                   # character ↔ character
    MEMBER_OF = "member_of"           # character → faction
    OWNS = "owns"                     # character → item
    PARTICIPATED_IN = "participated_in"  # character → event
    CAUSED = "caused"                 # event → event (causality)
    HAPPENED_AT = "happened_at"       # event → location
    CONTAINS = "contains"             # location → location (nesting)
    HOSTILE_TO = "hostile_to"         # faction ↔ faction
    ALLIED_WITH = "allied_with"       # faction ↔ faction
    SEEKS = "seeks"                   # character → item/goal
    RELATED_TO = "related_to"         # generic
    PART_OF = "part_of"               # plot_thread → plot_thread
    REVEALED_BY = "revealed_by"       # lore → event


class MemoryType(str, Enum):
    """What kind of memory this is."""
    EPISODIC = "episodic"       # A specific event: "When Kael betrayed the Order"
    SEMANTIC = "semantic"       # A fact: "Kael is a former knight"
    PROCEDURAL = "procedural"   # A rule: "The Shadowgate only opens at midnight"
    EMOTIONAL = "emotional"     # A sentiment: "The village fears the dark forest"


class NodeStatus(str, Enum):
    """Whether this node is active in the current narrative."""
    ACTIVE = "active"           # Currently relevant / present
    DORMANT = "dormant"         # Exists but not currently active
    DESTROYED = "destroyed"     # Gone (but remembered)
    HIDDEN = "hidden"           # Exists but not yet discovered


# ============================================================================
# HELPER
# ============================================================================

def _make_id(prefix: str = "n") -> str:
    """Generate a short unique ID."""
    ts = datetime.now(timezone.utc).isoformat()
    return prefix + "-" + hashlib.sha256(
        f"{ts}-{id(object())}".encode()
    ).hexdigest()[:10]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# CORE MODELS — Nodes, Edges, Memories
# ============================================================================

class StoryNode(BaseModel):
    """A thing in the story world — character, location, item, etc."""
    id: str = Field(default_factory=lambda: _make_id("node"))
    type: NodeType
    name: str
    description: str = ""
    short_description: str = ""      # One-liner for context assembly
    status: NodeStatus = NodeStatus.ACTIVE
    properties: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)  # Alt names for matching
    pinned: bool = False             # Always include in context if True
    icon: str = ""                   # Emoji for frontend display
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    created_by: str = "system"       # Display name of creator

    @property
    def search_names(self) -> List[str]:
        """All names this node can be found by (lowercase)."""
        names = [self.name.lower()]
        names.extend(a.lower() for a in self.aliases)
        return names

    def context_line(self) -> str:
        """Compact representation for context window."""
        if self.short_description:
            return f"{self.name}: {self.short_description}"
        return f"{self.name}: {self.description[:120]}"


class StoryEdge(BaseModel):
    """A relationship between two story nodes."""
    id: str = Field(default_factory=lambda: _make_id("edge"))
    type: EdgeType
    source_id: str
    target_id: str
    label: str = ""                  # Human-readable: "Kael lives in Ironhold"
    weight: float = 1.0              # Relevance weight (0-1)
    bidirectional: bool = False      # If True, relationship goes both ways
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    created_by: str = "system"


class StoryMemory(BaseModel):
    """A stored memory — something the story remembers."""
    id: str = Field(default_factory=lambda: _make_id("mem"))
    type: MemoryType
    content: str                     # Full memory text
    summary: str                     # One-line for context assembly
    importance: float = 0.5          # 0-1, how important (user-adjustable)
    related_nodes: List[str] = Field(default_factory=list)  # Node IDs
    tags: List[str] = Field(default_factory=list)
    story_time: Optional[str] = None  # When in the story timeline
    turn_number: int = 0              # Which story turn created this
    access_count: int = 0             # Times recalled
    last_accessed: Optional[str] = None
    decay_rate: float = 0.01          # How fast importance fades per turn
    pinned: bool = False              # Never decay if pinned
    created_at: str = Field(default_factory=_now)
    created_by: str = "system"

    def effective_importance(self, current_turn: int) -> float:
        """Importance after time decay."""
        if self.pinned:
            return self.importance
        turns_since = max(0, current_turn - self.turn_number)
        decay = self.decay_rate * turns_since
        return max(0.05, self.importance - decay)


# ============================================================================
# WORLD STATE — The current snapshot of the story
# ============================================================================

class WorldState(BaseModel):
    """Current state of the story world — what's happening RIGHT NOW."""
    story_name: str = "Elysium"
    current_scene: Optional[str] = None       # Node ID of current scene
    current_location: Optional[str] = None    # Node ID of current location
    present_characters: List[str] = Field(default_factory=list)  # Node IDs
    active_plot_threads: List[str] = Field(default_factory=list)  # Node IDs
    turn_number: int = 0
    last_turn_summary: str = ""
    mood: str = "neutral"                     # Current narrative mood
    time_of_day: str = "day"                  # In-story time
    weather: str = "clear"

    # ── AitherOne Notebook Integration ──
    notebook_doc_id: Optional[str] = None           # AitherOne doc ID for agent co-writing

    # ── Prometheus Integration ──
    # When a Prometheus simulation is attached, the world comes alive
    prometheus_world_id: Optional[str] = None      # ID of linked simulation
    prometheus_active: bool = False                 # Is simulation running?
    simulation_time: Optional[str] = None           # Sim game time (e.g. "Year 1, Month 3, Day 15")
    simulation_weather: Optional[str] = None        # Sim weather (overrides weather when active)
    last_sim_events: List[str] = Field(default_factory=list)  # Recent sim events for context


# ============================================================================
# CONTEXT ASSEMBLY MODELS — The glass box
# ============================================================================

class ActivatedNode(BaseModel):
    """A node that was activated during context assembly, with the reason WHY."""
    node_id: str
    node_name: str
    node_type: str
    reason: str               # "directly mentioned", "current location", "1-hop: knows Kael"
    relevance: float          # 0-1 score
    source_stage: str         # Which pipeline stage activated it
    included: bool = True     # Whether it made it into final context


class RecalledMemory(BaseModel):
    """A memory that was recalled during context assembly."""
    memory_id: str
    summary: str
    memory_type: str
    reason: str               # "related to Kael", "matches theme: betrayal"
    relevance: float
    included: bool = True


class PrunedItem(BaseModel):
    """Something that was considered but pruned from context, and WHY."""
    item_id: str
    item_name: str
    item_type: str            # "node" or "memory"
    reason: str               # "below threshold (0.15)", "token budget exceeded"
    relevance: float


class ContextStage(BaseModel):
    """One stage of the context assembly pipeline — visible to the user."""
    name: str                 # EXTRACT, ACTIVATE, EXPAND, RECALL, RANK, ASSEMBLE
    icon: str                 # Emoji for the frontend
    duration_ms: int = 0
    items_in: int = 0
    items_out: int = 0
    details: List[str] = Field(default_factory=list)  # Human-readable log lines


class ContextAssembly(BaseModel):
    """
    The complete output of the context assembly pipeline.
    This is the GLASS BOX — the user sees every decision.
    """
    assembly_id: str = Field(default_factory=lambda: _make_id("ctx"))
    timestamp: str = Field(default_factory=_now)
    user_input: str = ""
    turn_number: int = 0

    # The 6-stage pipeline log
    stages: List[ContextStage] = Field(default_factory=list)

    # What made it in
    activated_nodes: List[ActivatedNode] = Field(default_factory=list)
    recalled_memories: List[RecalledMemory] = Field(default_factory=list)

    # What got cut
    pruned: List[PrunedItem] = Field(default_factory=list)

    # The final assembled context
    context_sections: Dict[str, str] = Field(default_factory=dict)
    context_text: str = ""
    token_estimate: int = 0
    token_budget: int = 4096

    @property
    def included_count(self) -> int:
        return (
            len([n for n in self.activated_nodes if n.included])
            + len([m for m in self.recalled_memories if m.included])
        )

    @property
    def pruned_count(self) -> int:
        return len(self.pruned)

    @property
    def final_context(self) -> str:
        """Backward-compat alias for context_text."""
        return self.context_text


# ============================================================================
# REQUEST / RESPONSE MODELS (for the API)
# ============================================================================

class CreateNodeRequest(BaseModel):
    type: NodeType
    name: str
    description: str = ""
    short_description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    icon: str = ""
    display_name: str = "Anonymous"  # Who's creating it


class CreateEdgeRequest(BaseModel):
    type: EdgeType
    source_id: str
    target_id: str
    label: str = ""
    weight: float = 1.0
    bidirectional: bool = False
    display_name: str = "Anonymous"


class CreateMemoryRequest(BaseModel):
    type: MemoryType
    content: str
    summary: str = ""
    importance: float = 0.5
    related_nodes: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    story_time: Optional[str] = None
    pinned: bool = False
    display_name: str = "Anonymous"


class CreateCharacterRequest(BaseModel):
    """Convenience model for creating a full character (node + edges)."""
    name: str
    description: str
    short_description: str = ""
    personality: str = ""
    appearance: str = ""
    backstory: str = ""
    faction: Optional[str] = None         # Faction node ID to join
    location: Optional[str] = None        # Location node ID
    aliases: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    icon: str = "🧑"
    display_name: str = "Anonymous"


class StoryTurnRequest(BaseModel):
    """A story turn — the main interaction point."""
    message: str
    session_id: str = ""
    display_name: str = "Anonymous"
    # Optional overrides for context assembly
    pin_nodes: List[str] = Field(default_factory=list)      # Force-include these nodes
    exclude_nodes: List[str] = Field(default_factory=list)   # Force-exclude these nodes
    context_budget: int = 4096         # Max tokens for context


class SceneTurnRequest(BaseModel):
    """Multi-agent scene turn — characters as independent agents."""
    message: str
    context_budget: int = 2048
    max_rounds: int = 2
    narrate: bool = False              # Enable voice narration
    characters: Optional[List[str]] = None  # Override present characters (node IDs)


class CharacterOutputModel(BaseModel):
    """Output from a single character agent."""
    character_id: str
    character_name: str
    dialogue: str = ""
    action: str = ""
    inner_thought: str = ""
    emotion: str = "neutral"


class NarrativeSegmentModel(BaseModel):
    """One segment of narrated prose with speaker attribution."""
    speaker: str                       # Character name or "narrator"
    text: str
    voice_id: str = ""
    is_narration: bool = True


class SceneResult(BaseModel):
    """Full result of a multi-agent scene orchestration."""
    scene_id: str = ""
    narrator_prose: str = ""
    character_outputs: List[CharacterOutputModel] = Field(default_factory=list)
    segments: List[NarrativeSegmentModel] = Field(default_factory=list)
    audio_segments: int = 0
    rounds_completed: int = 0
    characters_present: List[str] = Field(default_factory=list)
    context_metadata: Dict[str, Any] = Field(default_factory=dict)
