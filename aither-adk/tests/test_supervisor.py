"""Tests for Layer 3 supervised reasoning (``adk.supervisor``)."""

from __future__ import annotations

import asyncio

import pytest

from adk.reasoning_session import ReasoningSession
from adk.supervisor import (
    Goal,
    Supervisor,
    derive_goal,
    register_reasoning_tool,
)


# ── pure assessment ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_assess_escalate_on_repeated_failures():
    sup = Supervisor(ReasoningSession("a"), Goal("find it"), fail_threshold=2)
    v = await sup.assess({"failures": 2, "steps": 1})
    assert v.action == "escalate"


@pytest.mark.asyncio
async def test_assess_escalate_on_drift_without_answer():
    sup = Supervisor(ReasoningSession("a"), Goal("g"), drift_steps=4)
    v = await sup.assess({"failures": 0, "steps": 5, "answered": False})
    assert v.action == "escalate"


@pytest.mark.asyncio
async def test_assess_nudge_on_low_alignment():
    sup = Supervisor(ReasoningSession("a"), Goal("g"), align_fn=lambda st, goal: 0.1)
    v = await sup.assess({"failures": 0, "steps": 1})
    assert v.action == "nudge"
    assert v.score < 0.5


@pytest.mark.asyncio
async def test_assess_proceed_when_aligned():
    sup = Supervisor(ReasoningSession("a"), Goal("g"))  # default alignment 1.0
    v = await sup.assess({"failures": 0, "steps": 1})
    assert v.action == "proceed"


@pytest.mark.asyncio
async def test_alignment_score_async_fn():
    async def af(state, goal):
        return 0.3
    sup = Supervisor(ReasoningSession("a"), Goal("g"), align_fn=af)
    assert await sup.alignment_score({}) == pytest.approx(0.3)


# ── watcher loop reuses Layer-2 steering to intervene ────────────────────────
@pytest.mark.asyncio
async def test_supervisor_escalates_via_steering():
    """A loop that keeps failing tools → supervisor injects a deep_reasoning steer
    through the session's steering queue (the Layer-2 mechanism) + emits events."""
    sess = ReasoningSession("sup1")

    async def loop(s):
        s.emit({"type": "turn_start"})
        s.emit({"type": "tool_result", "ok": False, "error": "boom"})
        s.emit({"type": "turn_start"})
        s.emit({"type": "tool_result", "ok": False, "error": "boom2"})
        await asyncio.sleep(0.05)
        return "done"

    sup = Supervisor(sess, Goal("answer the question"), fail_threshold=2)
    sess.start(loop)
    verdict = await sup.run(tick_s=0.03)

    assert sup._escalated is True
    assert verdict.action == "escalate"
    pending = sess.pop_steering()
    assert any("[Supervisor]" in p and "deep_reasoning" in p for p in pending)


@pytest.mark.asyncio
async def test_supervisor_detects_run_react_vocab_failures():
    """portal-kit run_react emits step{ok:False} (not tool_result) on tool failure —
    the supervisor must still escalate. Locks in the integration contract."""
    sess = ReasoningSession("sup-rr")

    async def loop(s):
        s.emit({"type": "turn_start", "turn": 1})
        s.emit({"type": "step", "step": 1, "name": "list_events", "ok": False, "result": {"error": "x"}})
        s.emit({"type": "turn_start", "turn": 2})
        s.emit({"type": "step", "step": 2, "name": "list_events", "ok": False, "result": {"error": "y"}})
        await asyncio.sleep(0.05)
        return "done"

    sup = Supervisor(sess, Goal("look it up"), reasoning_tool_name="deep_reason", fail_threshold=2)
    sess.start(loop)
    verdict = await sup.run(tick_s=0.03)
    assert verdict.action == "escalate"
    assert any("deep_reason" in p for p in sess.pop_steering())


@pytest.mark.asyncio
async def test_supervisor_proceeds_quietly_when_aligned():
    sess = ReasoningSession("sup2")

    async def loop(s):
        s.emit({"type": "turn_start"})
        s.emit({"type": "tool_result", "ok": True})
        s.emit({"type": "done"})
        return "ok"

    sup = Supervisor(sess, Goal("g"))
    sess.start(loop)
    await sup.run(tick_s=0.03)
    assert sup._escalated is False
    assert sess.pop_steering() == []   # no intervention when on track


# ── reasoning-as-a-tool ──────────────────────────────────────────────────────
class _FakeTools:
    def __init__(self):
        self.registered = {}

    def register(self, fn, *, name=None, description=None):
        self.registered[name or fn.__name__] = fn


class _FakeAgent:
    def __init__(self):
        self._tools = _FakeTools()


@pytest.mark.asyncio
async def test_register_reasoning_tool_custom_reasoner():
    a = _FakeAgent()

    async def reasoner(p):
        return "CONCLUSION: " + p

    name = register_reasoning_tool(a, reasoner=reasoner, name="deep_reasoning")
    assert name == "deep_reasoning"
    fn = a._tools.registered["deep_reasoning"]
    assert await fn("hard problem") == "CONCLUSION: hard problem"


def test_register_reasoning_tool_disabled_is_noop():
    a = _FakeAgent()
    assert register_reasoning_tool(a, enabled=False) is None
    assert a._tools.registered == {}


def test_derive_goal_trims():
    g = derive_goal("  build the thing  ", "c1")
    assert g.text == "build the thing"
    assert g.conversation_id == "c1"
