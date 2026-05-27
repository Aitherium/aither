"""
Effort Mapper — Classify narrative complexity to ADK effort levels.

Maps user input to an effort level (1-10) that controls:
- Which LLM model is used (small/medium/large)
- Context depth (axioms-only vs full 6-pillar)
- Whether reasoning-as-a-tool is available
- Token budgets

Effort levels:
    1-2: Simple actions (walk, look, examine) → small model, fast
    3-4: Standard dialogue/combat/description → medium model, full context
    5-6: Character confrontation, reveals, betrayal → medium + SASE reasoning
    7-8: Plot twists, MCTS branch evaluation → large reasoning model
    9-10: Multi-chapter planning, world consistency audits → deep think
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

# ── Signal word sets for complexity detection ──

_SIMPLE_VERBS = frozenset({
    "walk", "go", "move", "run", "enter", "leave", "exit", "look", "examine",
    "inspect", "check", "open", "close", "sit", "stand", "wait", "rest",
    "pick", "grab", "take", "drop", "put", "eat", "drink", "sleep",
    "climb", "jump", "swim", "hide", "sneak", "listen", "knock",
})

_DIALOGUE_SIGNALS = frozenset({
    "say", "tell", "ask", "speak", "talk", "greet", "respond", "reply",
    "whisper", "shout", "yell", "call", "answer", "chat", "discuss",
})

_COMBAT_SIGNALS = frozenset({
    "fight", "attack", "strike", "defend", "block", "parry", "dodge",
    "cast", "shoot", "throw", "charge", "slash", "stab", "punch", "kick",
    "battle", "duel", "spar", "ambush",
})

_HIGH_STAKES_SIGNALS = frozenset({
    "confront", "betray", "reveal", "confess", "accuse", "murder", "kill",
    "sacrifice", "deceive", "manipulate", "seduce", "threaten", "blackmail",
    "assassinate", "overthrow", "rebel", "coup", "usurp", "exile", "execute",
    "condemn", "judge", "sentence", "curse", "transform", "resurrect",
    "die", "death", "destroy", "annihilate", "corrupt", "purify",
    "propose", "marry", "divorce", "abandon", "disown",
})

_CRITICAL_SIGNALS = frozenset({
    "plan", "outline", "design", "architect", "audit", "review", "analyze",
    "evaluate", "assess", "strategy", "timeline", "chapter", "arc",
    "continuity", "consistency", "worldbuild", "magic system",
    "summarize the story", "what happened so far", "recap",
})

_CONSEQUENCE_WORDS = frozenset({
    "because", "therefore", "consequence", "result", "leads to", "causes",
    "triggers", "sparks", "ignites", "provokes", "ripple", "chain reaction",
})

_EMOTION_AMPLIFIERS = frozenset({
    "desperately", "furiously", "passionately", "reluctantly", "tearfully",
    "bitterly", "triumphantly", "defiantly", "fearfully", "lovingly",
    "angrily", "sorrowfully", "joyfully", "grimly", "solemnly",
})

# Patterns that indicate complex narrative needs
_COMPLEX_PATTERNS = [
    re.compile(r"\bwhile\b.*\b(secretly|quietly|simultaneously)\b", re.I),
    re.compile(r"\b(but|however|although|despite)\b.*\b(actually|really|truly)\b", re.I),
    re.compile(r"\babout (the|his|her|their) (secret|past|betrayal|plan)\b", re.I),
    re.compile(r"\b(what if|what would happen|consequences of)\b", re.I),
    re.compile(r"\b(remember when|back when|long ago|years ago)\b", re.I),
]

# Meta-narrative patterns (user stepping outside the story)
_META_PATTERNS = [
    re.compile(r"\b(plan|outline|design|create|build)\b.*\b(chapter|arc|quest|system)\b", re.I),
    re.compile(r"\b(audit|check|review)\b.*\b(continuity|consistency|lore|timeline)\b", re.I),
    re.compile(r"\bwhat (does|do|would|should)\b.*\b(character|npc|faction)\b.*\bthink\b", re.I),
    re.compile(r"\bgenerate\b.*\b(map|dungeon|encounter|quest|npc|loot)\b", re.I),
    re.compile(r"\bexport|save|branch|rewrite|retcon\b", re.I),
]


@dataclass
class EffortResult:
    """Result of effort classification."""
    effort: int                          # 1-10
    reasoning: str                       # Why this level was chosen
    signals: List[str] = field(default_factory=list)  # What triggered the classification
    use_reasoning_tool: bool = False     # Should reasoning-as-a-tool be available
    context_budget: int = 4096           # Token budget for context assembly
    mcts_mode: Optional[str] = None      # None, "lightweight", "standard", "deep"


# ── Effort level configurations ──

_EFFORT_CONFIGS = {
    1: {"context_budget": 1024, "mcts_mode": None, "use_reasoning": False},
    2: {"context_budget": 2048, "mcts_mode": None, "use_reasoning": False},
    3: {"context_budget": 4096, "mcts_mode": None, "use_reasoning": False},
    4: {"context_budget": 4096, "mcts_mode": "lightweight", "use_reasoning": False},
    5: {"context_budget": 6144, "mcts_mode": "lightweight", "use_reasoning": False},
    6: {"context_budget": 6144, "mcts_mode": "standard", "use_reasoning": True},
    7: {"context_budget": 8192, "mcts_mode": "standard", "use_reasoning": True},
    8: {"context_budget": 8192, "mcts_mode": "deep", "use_reasoning": True},
    9: {"context_budget": 12288, "mcts_mode": "deep", "use_reasoning": True},
    10: {"context_budget": 16384, "mcts_mode": "deep", "use_reasoning": True},
}


def classify_effort(
    user_input: str,
    turn_number: int = 0,
    active_plot_threads: int = 0,
    present_characters: int = 0,
) -> EffortResult:
    """Classify narrative complexity of user input to an effort level.

    Args:
        user_input: The player's message
        turn_number: Current story turn (early turns get slight boost)
        active_plot_threads: Number of active plot threads
        present_characters: Number of characters in scene

    Returns:
        EffortResult with effort level, reasoning, and config
    """
    text = user_input.strip()
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    signals = []

    # Start with base score
    score = 3.0  # Default: standard interaction

    # ── Meta-narrative detection (highest priority) ──
    for pattern in _META_PATTERNS:
        if pattern.search(text):
            score = max(score, 8.0)
            signals.append(f"meta-narrative: {pattern.pattern[:40]}")

    # ── Critical signals (planning, auditing, system design) ──
    critical_hits = words & _CRITICAL_SIGNALS
    if critical_hits:
        score = max(score, 9.0)
        signals.append(f"critical: {', '.join(list(critical_hits)[:3])}")

    # ── High-stakes signals ──
    high_stakes_hits = words & _HIGH_STAKES_SIGNALS
    if high_stakes_hits:
        boost = min(3.0, len(high_stakes_hits) * 1.0)
        score = max(score, 5.0 + boost)
        signals.append(f"high-stakes: {', '.join(list(high_stakes_hits)[:3])}")

    # ── Complex sentence patterns ──
    for pattern in _COMPLEX_PATTERNS:
        if pattern.search(text):
            score = max(score, 5.0)
            score += 0.5
            signals.append(f"complex-pattern: {pattern.pattern[:30]}")

    # ── Emotion amplifiers ──
    emotion_hits = words & _EMOTION_AMPLIFIERS
    if emotion_hits:
        score += len(emotion_hits) * 0.5
        signals.append(f"emotional: {', '.join(list(emotion_hits)[:2])}")

    # ── Consequence language ──
    consequence_hits = words & _CONSEQUENCE_WORDS
    if consequence_hits:
        score += 1.0
        signals.append("consequence-language")

    # ── Combat (medium complexity) ──
    combat_hits = words & _COMBAT_SIGNALS
    if combat_hits:
        score = max(score, 4.0)
        signals.append(f"combat: {', '.join(list(combat_hits)[:2])}")

    # ── Dialogue (medium complexity) ──
    dialogue_hits = words & _DIALOGUE_SIGNALS
    if dialogue_hits and not high_stakes_hits:
        score = max(score, 3.0)
        signals.append("dialogue")

    # ── Simple actions (low complexity) ──
    simple_hits = words & _SIMPLE_VERBS
    if simple_hits and not high_stakes_hits and not combat_hits and not dialogue_hits:
        if len(words) < 8:
            score = min(score, 2.0)
            signals.append(f"simple-action: {', '.join(list(simple_hits)[:2])}")

    # ── Length-based adjustments ──
    word_count = len(text.split())
    if word_count <= 3 and score < 4:
        score = min(score, 2.0)
        signals.append("very-short-input")
    elif word_count > 30:
        score += 1.0
        signals.append("long-input")
    elif word_count > 60:
        score += 2.0
        signals.append("very-long-input")

    # ── Context-based adjustments ──
    if present_characters > 3:
        score += 0.5
        signals.append(f"multi-character-scene ({present_characters})")
    if active_plot_threads > 2:
        score += 0.5
        signals.append(f"multiple-plot-threads ({active_plot_threads})")

    # ── Early story boost (first 5 turns get minimum effort 3) ──
    if turn_number <= 5 and score < 3:
        score = 3.0
        signals.append("early-story-boost")

    # Clamp to 1-10
    effort = max(1, min(10, round(score)))

    # Build config from effort level
    config = _EFFORT_CONFIGS[effort]

    # Build reasoning string
    if not signals:
        signals.append("default-standard")

    reasoning = f"Effort {effort}: {'; '.join(signals)}"

    return EffortResult(
        effort=effort,
        reasoning=reasoning,
        signals=signals,
        use_reasoning_tool=config["use_reasoning"],
        context_budget=config["context_budget"],
        mcts_mode=config["mcts_mode"],
    )
