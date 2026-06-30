"""Unified typed-activation memory contract.

Defines the ONE canonical record format (:class:`MemoryRecord`) used by the
self-maintaining memory layer, plus the typed weighted edge
(:class:`MemoryEdgeRecord`) and the planning constraint
(:class:`DecisionConstraint`) derived from decision/correction memories.

The central design idea is two *orthogonal* axes:

- :class:`Role`  — what the memory *is*.  Drives **authority**: a correction
  outranks a teaching outranks a discovery.
- :class:`Tier`  — how long the memory *lasts* and how fast it decays.

Pure stdlib; no third-party or framework dependencies — this module is the
shared vocabulary every other memory module speaks.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ═════════════════════════════════════════════════════════════════════════
# AXES — orthogonal: Role (what it is) × Tier (how long it lasts)
# ═════════════════════════════════════════════════════════════════════════

class Role(str, Enum):
    """What a memory *is* — drives authority during recall.

    Authority ordering (the multipliers live in the scorer) roughly:
    correction > decision > teaching/preference > fact/procedure > insight >
    discovery > interaction/task > temporary.
    """
    FACT = "fact"
    DECISION = "decision"
    CORRECTION = "correction"
    TASK = "task"
    TEMPORARY = "temporary"
    PROCEDURE = "procedure"
    INSIGHT = "insight"
    DISCOVERY = "discovery"
    PREFERENCE = "preference"
    INTERACTION = "interaction"
    TEACHING = "teaching"
    FEEDBACK = "feedback"
    IDENTITY = "identity"


class Tier(str, Enum):
    """How long a memory lasts / how fast it decays.

    Each tier carries a per-second decay rate (see :data:`TIER_DECAY_RATES`)
    so freshness can be computed as ``e^(-λ·Δt)``.
    """
    EPHEMERAL = "ephemeral"      # ~15 min
    SESSION = "session"          # ~1 hour
    PERSISTENT = "persistent"    # ~7 days
    RELATIONAL = "relational"    # ~30 days
    TRACE = "trace"              # ~90 days
    PERMANENT = "permanent"      # never decays


# Per-second decay constant λ for freshness = e^(-λ·Δt).
# Defaults chosen so freshness ≈ 0.5 at roughly the tier's nominal lifetime.
TIER_DECAY_RATES: Dict[Tier, float] = {
    Tier.EPHEMERAL: math.log(2) / 900.0,        # 15 min half-life
    Tier.SESSION: math.log(2) / 3600.0,         # 1 hour
    Tier.PERSISTENT: math.log(2) / 604800.0,    # 7 days
    Tier.RELATIONAL: math.log(2) / 2592000.0,   # 30 days
    Tier.TRACE: math.log(2) / 7776000.0,        # 90 days
    Tier.PERMANENT: 0.0,                        # no decay
}

# Nominal TTL (seconds) per tier — used to compute expires_at when unset.
TIER_TTL_SECONDS: Dict[Tier, Optional[float]] = {
    Tier.EPHEMERAL: 900.0,
    Tier.SESSION: 3600.0,
    Tier.PERSISTENT: 604800.0,
    Tier.RELATIONAL: 2592000.0,
    Tier.TRACE: 7776000.0,
    Tier.PERMANENT: None,
}


# ═════════════════════════════════════════════════════════════════════════
# EDGES — typed, weighted, time-decaying
# ═════════════════════════════════════════════════════════════════════════

class EdgeType(str, Enum):
    """Typed relationships between memories."""
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    DERIVED_FROM = "derived_from"
    ELABORATES = "elaborates"
    RELATED = "related"
    TAG_SIBLING = "tag_sibling"
    REINFORCED_BY = "reinforced_by"
    SAME_AGENT = "same_agent"
    SAME_SESSION = "same_session"
    TEMPORAL = "temporal"
    PART_OF = "part_of"


@dataclass
class MemoryEdgeRecord:
    """A typed, weighted edge between two :class:`MemoryRecord` ids.

    Edge weight decays over time (``weight_decay_rate``) so that, per the
    activation model, *reinforced paths stay strong while stale links fade*.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    weight_decay_rate: float = 1e-7   # per second; 0 = never fades
    created_at: float = field(default_factory=time.time)
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)
        if not self.id:
            self.id = f"edge_{uuid.uuid4().hex[:12]}"

    def current_weight(self, now: Optional[float] = None) -> float:
        """Edge weight after time-decay (``weight·e^(-λ·Δt)``)."""
        if self.weight_decay_rate <= 0:
            return self.weight
        now = now if now is not None else time.time()
        age = max(0.0, now - self.created_at)
        return self.weight * math.exp(-self.weight_decay_rate * age)


# ═════════════════════════════════════════════════════════════════════════
# THE CANONICAL RECORD
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryRecord:
    """Canonical unified memory representation.

    Fields are grouped: identity, classification (the two axes), authority &
    confidence, temporal/freshness, supersession, graph connectivity, and
    scoping.
    """

    # ── Identity ────────────────────────────────────────────────────────
    id: str = ""
    content: str = ""
    content_hash: str = ""

    # ── Classification (the two orthogonal axes) ────────────────────────
    role: Role = Role.FACT
    tier: Tier = Tier.PERSISTENT
    domain: str = "general"
    tags: List[str] = field(default_factory=list)

    # ── Authority & confidence ──────────────────────────────────────────
    source: str = "system"
    source_credibility: float = 0.5
    confidence: float = 0.5
    temporal_consistency: float = 1.0       # 1.0 = uncontradicted; 0.3 = superseded
    reinforcement_count: int = 0
    judge_confidence: float = 0.0           # 0 = judge not consulted
    judge_approved: bool = True

    # ── Temporal / freshness ────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)
    access_count: int = 0
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    expires_at: Optional[float] = None

    # ── Supersession / staleness ────────────────────────────────────────
    stale: bool = False
    superseded_by: Optional[str] = None
    supersedes: List[str] = field(default_factory=list)

    # ── Graph connectivity ──────────────────────────────────────────────
    edges: List[MemoryEdgeRecord] = field(default_factory=list)

    # ── Entity identity (additive; opt-in) ──────────────────────────────
    entity_id: Optional[str] = None         # canonical entity this fact is about
    entity_aliases: List[str] = field(default_factory=list)

    # ── Vector / scope / extra ──────────────────────────────────────────
    embedding: Optional[List[float]] = None
    scope_namespace: str = "platform"
    relevance: float = 0.0                  # set by retrieval scoring (transient)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = Role(self.role)
        if isinstance(self.tier, str):
            self.tier = Tier(self.tier)
        if not self.id:
            self.id = f"mem_{uuid.uuid4().hex[:12]}"
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode(errors="replace")
            ).hexdigest()[:16]
        if self.expires_at is None:
            ttl = TIER_TTL_SECONDS.get(self.tier)
            if ttl is not None:
                self.expires_at = self.created_at + ttl

    # ── Derived activation properties ───────────────────────────────────

    def freshness(self, now: Optional[float] = None) -> float:
        """``e^(-λ·Δt)`` since last reinforcement, λ from the tier."""
        rate = TIER_DECAY_RATES.get(self.tier, TIER_DECAY_RATES[Tier.PERSISTENT])
        if rate <= 0:
            return 1.0
        now = now if now is not None else time.time()
        elapsed = max(0.0, now - self.last_reinforced)
        return math.exp(-rate * elapsed)

    def reinforcement_bonus(self) -> float:
        """Log-scaled bonus: 2 reinforcements ≈ +0.3, 10 ≈ +1.0 (capped)."""
        if self.reinforcement_count <= 1:
            return 0.0
        return min(1.0, math.log2(self.reinforcement_count) / 3.32)

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        if self.valid_until is not None and now > self.valid_until:
            return True
        if self.expires_at is not None and now > self.expires_at:
            return True
        return False

    def is_valid_now(self, now: Optional[float] = None) -> bool:
        """Within the [valid_from, valid_until] window, if any is set."""
        now = now if now is not None else time.time()
        if self.valid_from is not None and now < self.valid_from:
            return False
        if self.valid_until is not None and now > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "content_hash": self.content_hash,
            "role": self.role.value,
            "tier": self.tier.value,
            "domain": self.domain,
            "tags": list(self.tags),
            "source": self.source,
            "source_credibility": self.source_credibility,
            "confidence": self.confidence,
            "temporal_consistency": self.temporal_consistency,
            "reinforcement_count": self.reinforcement_count,
            "judge_confidence": self.judge_confidence,
            "judge_approved": self.judge_approved,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_reinforced": self.last_reinforced,
            "access_count": self.access_count,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "expires_at": self.expires_at,
            "stale": self.stale,
            "superseded_by": self.superseded_by,
            "supersedes": list(self.supersedes),
            "entity_id": self.entity_id,
            "entity_aliases": list(self.entity_aliases),
            "scope_namespace": self.scope_namespace,
            "relevance": self.relevance,
            "metadata": dict(self.metadata),
        }


# ═════════════════════════════════════════════════════════════════════════
# PLANNING CONSTRAINT (derived from DECISION / CORRECTION memories)
# ═════════════════════════════════════════════════════════════════════════

class ConstraintType(str, Enum):
    HARD = "hard"      # must be satisfied — violating plans are rejected
    SOFT = "soft"      # should be satisfied — surfaced to planner
    AVOID = "avoid"    # must not be done
    PREFER = "prefer"  # preferred but optional


@dataclass
class DecisionConstraint:
    """A constraint extracted from a DECISION/CORRECTION memory, consumed by
    a planner so past decisions actually constrain future plans."""
    id: str
    memory_id: str
    description: str
    constraint_type: ConstraintType
    scope: Optional[str] = None
    applies_to: Optional[str] = None    # e.g. a tool/agent name
    priority: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.constraint_type, str):
            self.constraint_type = ConstraintType(self.constraint_type)

    def to_planner_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.constraint_type.value,
            "scope": self.scope,
            "applies_to": self.applies_to,
            "priority": self.priority,
            "source_memory_id": self.memory_id,
        }


# ═════════════════════════════════════════════════════════════════════════
# ROLE / TIER INFERENCE
# ═════════════════════════════════════════════════════════════════════════

# Map common alias strings onto the unified Role.
_LEGACY_ROLE_ALIASES: Dict[str, Role] = {
    "user_interaction": Role.INTERACTION,
    "user_preference": Role.PREFERENCE,
    "user_correction": Role.CORRECTION,
    "correction": Role.CORRECTION,
    "fact": Role.FACT,
    "codebase": Role.FACT,
    "decision": Role.DECISION,
    "teaching": Role.TEACHING,
    "discovery": Role.DISCOVERY,
    "insight": Role.INSIGHT,
    "procedure": Role.PROCEDURE,
    "context": Role.INTERACTION,
    "identity": Role.IDENTITY,
    "emotional": Role.INTERACTION,
    "neuron_feedback": Role.FEEDBACK,
    "feedback": Role.FEEDBACK,
    "preference": Role.PREFERENCE,
    "interaction": Role.INTERACTION,
    "task": Role.TASK,
    "temporary": Role.TEMPORARY,
}

# Map a durability string onto the unified Tier (1:1).
_DURABILITY_TO_TIER: Dict[str, Tier] = {
    "ephemeral": Tier.EPHEMERAL,
    "session": Tier.SESSION,
    "persistent": Tier.PERSISTENT,
    "relational": Tier.RELATIONAL,
    "trace": Tier.TRACE,
    "permanent": Tier.PERMANENT,
}

_TIER_TO_DURABILITY: Dict[Tier, str] = {t: d for d, t in _DURABILITY_TO_TIER.items()}


def infer_role(value: Optional[str], default: Role = Role.FACT) -> Role:
    """Best-effort map a type/category string onto a unified :class:`Role`."""
    if not value:
        return default
    key = str(value).strip().lower()
    if key in _LEGACY_ROLE_ALIASES:
        return _LEGACY_ROLE_ALIASES[key]
    try:
        return Role(key)
    except ValueError:
        return default


# Content-based role inference — for records stored WITHOUT an explicit
# role/type.  Without this every untyped record defaults to FACT, so authority
# can't differentiate and recall labels collapse to ``[NEW]``.  Inferring at
# read time types untyped data on recall with zero re-store / duplication.
# Best-effort and order-sensitive (most authoritative first).
_CONTENT_ROLE_PATTERNS: list[tuple[Role, re.Pattern[str]]] = [
    (Role.CORRECTION, re.compile(
        r"\b(actually|correction|not quite|that'?s wrong|i was wrong|"
        r"instead of|should be|use .+ not |don'?t use)\b", re.I)),
    (Role.DECISION, re.compile(
        r"\b(we (?:decided|chose|will use|agreed)|let'?s use|going with|"
        r"the plan is|decision:|we'?re using)\b", re.I)),
    (Role.PREFERENCE, re.compile(
        r"\b(i (?:prefer|like|want|always)|please always|my preference|"
        r"user prefers)\b", re.I)),
    (Role.TASK, re.compile(
        r"\b(todo|to-do|task:|remember to|need to|follow up|next step)\b", re.I)),
    (Role.TEMPORARY, re.compile(
        r"\b(for now|temporar|note to self|scratch|draft)\b", re.I)),
    (Role.TEACHING, re.compile(
        r"\b(here'?s how|the way to|you should|to do this|lesson:)\b", re.I)),
]


def infer_role_from_content(content: Optional[str]) -> Optional[Role]:
    """Infer a role from message content, or ``None`` if nothing matches."""
    if not content:
        return None
    text = str(content)
    for role, pat in _CONTENT_ROLE_PATTERNS:
        if pat.search(text):
            return role
    return None


def tier_from_durability(durability: Optional[str], default: Tier = Tier.PERSISTENT) -> Tier:
    if not durability:
        return default
    return _DURABILITY_TO_TIER.get(str(durability).strip().lower(), default)


def durability_from_tier(tier: Tier) -> str:
    return _TIER_TO_DURABILITY.get(tier, "persistent")
