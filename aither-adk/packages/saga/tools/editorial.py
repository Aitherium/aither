"""Editorial/Craft Tools — Prose analysis, rewriting, and quality refinement.

Tools for analyzing and improving narrative prose quality:
continuity audits, style metrics, rewriting, expansion, compression,
show-don't-tell conversion, sentence variety, and emotional beat checks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from adk.tools import tool

logger = logging.getLogger("saga.tools.editorial")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


@tool(
    name="continuity_audit",
    description=(
        "Audit story continuity across recent turns. Checks for dead entities "
        "appearing, location conflicts, timeline gaps, and contradicted lore."
    ),
)
def continuity_audit(lookback_turns: int = 20) -> dict:
    """Run a continuity check.

    Args:
        lookback_turns: How many turns back to audit
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeStatus, MemoryType

    current_turn = graph.world.turn_number
    start_turn = max(0, current_turn - lookback_turns)

    issues = []

    # Check for destroyed entities that are still referenced as present
    for char_id in graph.world.present_characters:
        char = graph.get_node(char_id)
        if char and char.status == NodeStatus.DESTROYED:
            issues.append({
                "type": "dead_entity_present",
                "severity": "high",
                "detail": f"{char.name} is marked DESTROYED but still listed as present in scene",
            })

    # Check for hidden entities in active scenes
    for char_id in graph.world.present_characters:
        char = graph.get_node(char_id)
        if char and char.status == NodeStatus.HIDDEN:
            issues.append({
                "type": "hidden_entity_present",
                "severity": "medium",
                "detail": f"{char.name} is HIDDEN but listed as present — was there a reveal?",
            })

    # Check procedural rules for potential violations
    rules = memory.get_by_type(MemoryType.PROCEDURAL)
    pinned_rules = [r for r in rules if r.pinned]

    # Gather recent episode summaries for pattern checking
    recent = sorted(
        [m for m in memory.get_all() if m.turn_number >= start_turn],
        key=lambda m: m.turn_number,
    )
    recent_summaries = [m.summary for m in recent]

    return {
        "audit_range": f"Turns {start_turn}-{current_turn}",
        "issues_found": len(issues),
        "issues": issues,
        "world_rules_count": len(pinned_rules),
        "world_rules": [r.summary for r in pinned_rules[:10]],
        "recent_events": recent_summaries[:15],
        "instruction": (
            "Review the recent events for continuity issues: "
            "1) Characters in wrong locations, "
            "2) Dead/hidden characters acting normally, "
            "3) Timeline inconsistencies, "
            "4) World rule violations, "
            "5) Contradicted facts. "
            "Report each issue with severity and suggested fix."
        ),
    }


@tool(
    name="style_metrics",
    description=(
        "Analyze prose quality metrics: average sentence length, vocabulary diversity, "
        "dialogue-to-narration ratio, adverb density, passive voice usage."
    ),
)
def style_metrics(text: str) -> dict:
    """Analyze prose style metrics.

    Args:
        text: The prose text to analyze
    """
    if not text or len(text) < 20:
        return {"error": "Text too short to analyze"}

    words = text.split()
    word_count = len(words)

    # Sentence analysis
    sentences = [s.strip() for s in re.split(r'[.!?]+\s+', text) if s.strip()]
    sentence_count = len(sentences) or 1
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / sentence_count

    # Vocabulary diversity
    unique_words = set(w.lower().strip(".,!?;:\"'") for w in words)
    vocab_diversity = len(unique_words) / word_count if word_count else 0

    # Dialogue ratio
    dialogue_matches = re.findall(r'"[^"]*"', text)
    dialogue_words = sum(len(d.split()) for d in dialogue_matches)
    dialogue_ratio = dialogue_words / word_count if word_count else 0

    # Adverb density (words ending in -ly)
    adverbs = [w for w in words if w.lower().endswith("ly") and len(w) > 3]
    adverb_density = len(adverbs) / word_count if word_count else 0

    # Passive voice indicators
    passive_patterns = re.findall(
        r'\b(was|were|been|being|is|are)\s+\w+ed\b', text, re.I
    )
    passive_ratio = len(passive_patterns) / sentence_count

    # Sentence opening variety
    openings = [s.split()[0].lower() if s.split() else "" for s in sentences]
    opening_counter = Counter(openings)
    repetitive_openings = [
        word for word, count in opening_counter.items()
        if count >= 3 and word
    ]

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "sentence_length_range": {
            "min": min(sentence_lengths) if sentence_lengths else 0,
            "max": max(sentence_lengths) if sentence_lengths else 0,
        },
        "vocabulary_diversity": round(vocab_diversity, 3),
        "dialogue_ratio": round(dialogue_ratio, 3),
        "adverb_density": round(adverb_density, 4),
        "adverbs_found": adverbs[:10],
        "passive_voice_ratio": round(passive_ratio, 3),
        "repetitive_openings": repetitive_openings,
        "assessment": _assess_metrics(
            avg_sentence_length, vocab_diversity, dialogue_ratio,
            adverb_density, passive_ratio, repetitive_openings,
        ),
    }


def _assess_metrics(avg_len, vocab, dialogue, adverbs, passive, rep_openings):
    """Generate a quick assessment of prose quality."""
    notes = []
    if avg_len > 25:
        notes.append("Sentences are long — consider breaking some up")
    elif avg_len < 8:
        notes.append("Sentences are very short — vary the rhythm")
    if vocab < 0.4:
        notes.append("Low vocabulary diversity — use more varied word choices")
    if dialogue > 0.7:
        notes.append("Very dialogue-heavy — add more narration/description")
    elif dialogue < 0.05:
        notes.append("Very little dialogue — conversation adds life")
    if adverbs > 0.03:
        notes.append("High adverb density — show through action instead")
    if passive > 0.3:
        notes.append("Frequent passive voice — use active constructions")
    if rep_openings:
        notes.append(f"Repetitive sentence openings: {', '.join(rep_openings)}")
    if not notes:
        notes.append("Prose quality looks solid")
    return notes


@tool(
    name="rewrite_passage",
    description="Rewrite a passage in a different style (formal/casual, telling/showing, passive/active).",
)
def rewrite_passage(
    passage: str,
    target_style: str = "showing",
) -> dict:
    """Set up a passage rewrite.

    Args:
        passage: The passage to rewrite
        target_style: Target style (showing, formal, casual, active, concise, vivid, literary)
    """
    return {
        "original": passage,
        "target_style": target_style,
        "instruction": (
            f"Rewrite this passage in a {target_style} style. "
            f"Preserve the meaning and key details. "
            f"{'Convert telling to showing — use action and dialogue instead of exposition. ' if target_style == 'showing' else ''}"
            f"{'Convert passive to active voice. ' if target_style == 'active' else ''}"
            f"{'Trim excess words while keeping impact. ' if target_style == 'concise' else ''}"
        ),
    }


@tool(
    name="expand_scene",
    description="Take a brief summary and expand into full prose with sensory details.",
)
def expand_scene(
    summary: str,
    target_length: str = "medium",
    focus: str = "sensory",
) -> dict:
    """Expand a summary into full prose.

    Args:
        summary: Brief scene summary to expand
        target_length: Target length (short=1 para, medium=2-3 para, long=4-6 para)
        focus: What to emphasize (sensory, emotional, action, dialogue)
    """
    return {
        "summary": summary,
        "target_length": target_length,
        "focus": focus,
        "instruction": (
            f"Expand this summary into {target_length} prose with a {focus} focus. "
            f"Add sensory details (sight, sound, smell, touch), "
            f"emotional undercurrents, and character reactions. "
            f"Don't add new plot elements — flesh out what's described."
        ),
    }


@tool(
    name="compress_scene",
    description="Compress verbose prose to essential beats for recap or summary.",
)
def compress_scene(passage: str) -> dict:
    """Compress a passage to its essential beats.

    Args:
        passage: The verbose passage to compress
    """
    return {
        "original": passage,
        "original_word_count": len(passage.split()),
        "instruction": (
            "Compress this passage to its essential beats. "
            "Keep: key actions, important dialogue, emotional turning points. "
            "Cut: atmospheric description, internal monologue, transitional prose. "
            "Target: 30-40% of original length."
        ),
    }


@tool(
    name="show_dont_tell",
    description="Identify 'telling' passages and rewrite as 'showing' (action/dialogue/sensory).",
)
def show_dont_tell(passage: str) -> dict:
    """Convert telling to showing.

    Args:
        passage: The passage to analyze and rewrite
    """
    # Simple heuristic detection of telling patterns
    telling_patterns = [
        (r'\b\w+ (was|were|felt|seemed) (angry|happy|sad|afraid|excited|nervous|worried)\b',
         "emotion-telling"),
        (r'\b(obviously|clearly|apparently|evidently)\b', "adverb-telling"),
        (r'\b(it was|there was) (clear|obvious|apparent)\b', "exposition-telling"),
    ]

    detected = []
    for pattern, pattern_type in telling_patterns:
        matches = re.findall(pattern, passage, re.I)
        if matches:
            detected.append({"type": pattern_type, "count": len(matches)})

    return {
        "original": passage,
        "telling_patterns_detected": detected,
        "instruction": (
            "Identify 'telling' passages (e.g., 'She was angry') and rewrite "
            "as 'showing' through: actions (she slammed the door), "
            "dialogue ('Get out,' she hissed), physical reactions (her jaw tightened), "
            "or environmental reflection (the air crackled with tension)."
        ),
    }


@tool(
    name="vary_sentences",
    description="Fix repetitive sentence openings and monotonous structure.",
)
def vary_sentences(passage: str) -> dict:
    """Vary sentence structure.

    Args:
        passage: The passage with repetitive sentences
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+\s+', passage) if s.strip()]
    openings = [s.split()[0] if s.split() else "" for s in sentences]
    opening_counts = Counter(openings)
    repetitive = {word: count for word, count in opening_counts.items() if count >= 2}

    return {
        "original": passage,
        "sentence_count": len(sentences),
        "repetitive_openings": repetitive,
        "instruction": (
            "Rewrite this passage to vary sentence openings and structure. "
            f"Repetitive openings found: {repetitive}. "
            "Mix: short punchy sentences, longer flowing ones, "
            "questions, fragments (for effect), dialogue. "
            "Start with action, description, dialogue, or dependent clauses."
        ),
    }


@tool(
    name="emotional_beat_check",
    description="Verify the emotional arc of a scene hits the intended beats.",
)
def emotional_beat_check(
    passage: str,
    intended_arc: str = "",
) -> dict:
    """Check emotional beats.

    Args:
        passage: The scene to analyze
        intended_arc: Intended emotional arc (e.g., "tension → relief → dread")
    """
    return {
        "passage": passage[:500],
        "intended_arc": intended_arc,
        "instruction": (
            "Analyze the emotional arc of this scene. "
            f"{'Compare against intended arc: ' + intended_arc + '. ' if intended_arc else ''}"
            "Identify: opening emotion, emotional shifts, climax, resolution. "
            "Flag: abrupt mood changes, missing emotional beats, "
            "unearned emotional payoffs, monotone emotional register."
        ),
    }
