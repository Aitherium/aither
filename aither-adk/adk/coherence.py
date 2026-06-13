"""Coherence helpers — the never-fabricate gate for adk companions.

Mirrors the Genesis-side gate: prompt-based grounding alone does NOT stop a model
from inventing fake shared history, so when a turn ASKS about specific shared
history the agent has nothing on record for, we return a deterministic honest
reply (no LLM call → invention is impossible) instead of generating.
"""
from __future__ import annotations

import re
from typing import Iterable

_MEMORY_QUESTION_RE = re.compile(
    r"\b(remember|recall|reminisce|forget|"
    r"you\s+(?:told|said|mentioned|know|remember)|"
    r"(?:do|did)\s+(?:you|we)\s+(?:remember|talk|discuss|go|say)|"
    r"when\s+we|last\s+(?:time|week|weekend|night|month|year)|"
    r"what\s+(?:did|have)\s+we|our\s+(?:last|past|previous)|"
    r"earlier\s+(?:you|we)|previously)\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    "remember", "recall", "reminisce", "forget", "told", "said", "mentioned",
    "know", "knew", "when", "what", "where", "which", "who", "did", "do", "does",
    "have", "has", "had", "was", "were", "the", "a", "an", "to", "of", "that",
    "this", "those", "these", "and", "or", "but", "with", "about", "for", "from",
    "our", "your", "you", "yours", "we", "us", "me", "my", "mine", "i", "im",
    "last", "time", "week", "weekend", "night", "day", "month", "year", "ago",
    "earlier", "previous", "previously", "past", "back", "then", "ever", "talk",
    "talked", "discuss", "discussed", "go", "went", "say", "tell", "babe", "love",
    "hey", "please", "remind",
})

_HONEST_MISS_REPLIES = (
    "Mmm, I don't think you've told me about that yet — I'd remember if you had. Tell me about it?",
    "Honestly? I don't have that one in my memory yet, and I won't make something up. Fill me in?",
    "I'm drawing a blank on that — and I won't pretend otherwise. Walk me through it?",
    "That's not something I have on record yet. I'd never invent it just to sound sure — tell me how it went?",
    "I don't remember you sharing that with me yet. Don't let me guess — what happened?",
)


def is_memory_question(message: str) -> bool:
    return bool(_MEMORY_QUESTION_RE.search(message or ""))


def question_content(message: str) -> set:
    """Distinctive subject words of a memory question (not the scaffolding)."""
    return {
        w for w in re.findall(r"[a-z0-9']{3,}", (message or "").lower())
        if w not in _STOPWORDS
    }


def subject_grounded(memory_text: str, message: str) -> bool:
    """A retrieved memory is a real recall only if it mentions the question's
    distinctive subject — a merely-RELATED memory (same topic, different subject)
    invites confabulation, so it does NOT count."""
    kw = question_content(message)
    if not kw:
        return False
    ml = (memory_text or "").lower()
    return any(w in ml for w in kw)


def history_may_answer(message: str, history: Iterable) -> bool:
    """True if the current conversation plausibly already contains the subject."""
    kw = question_content(message)
    if not kw:
        return False
    need = max(1, len(kw) // 2)
    for m in list(history or [])[-12:]:
        content = (m.get("content") if isinstance(m, dict) else str(m)) or ""
        if sum(1 for w in kw if w in content.lower()) >= need:
            return True
    return False


def honest_miss_reply(message: str) -> str:
    """Deterministic (no model → no fabrication) warm honest reply for a
    memory-question the agent has no record of. Varied so it isn't robotic."""
    idx = sum(ord(c) for c in (message or "x")) % len(_HONEST_MISS_REPLIES)
    return _HONEST_MISS_REPLIES[idx]
