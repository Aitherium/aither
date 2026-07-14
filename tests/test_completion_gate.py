"""Tests for the composable completion gate (adk.gate).

Proves the gate turns liveness-success into verified-or-retried, and — the whole
point — that a mechanically-checkable criterion CANNOT be soft-passed."""
from __future__ import annotations

import pytest

from adk.agent import AgentResponse
from adk.gate import CompletionGate, GateVerdict, gated_run, hard_checks


class FakeAgent:
    """Minimal stand-in for AitherAgent: returns queued outputs, one per run()."""

    def __init__(self, outputs, llm=None):
        self._outputs = list(outputs)
        self.calls = []
        self.llm = llm

    async def run(self, task, **kwargs):
        self.calls.append(task)
        idx = min(len(self.calls) - 1, len(self._outputs) - 1)
        return AgentResponse(content=self._outputs[idx])


# ── hard_checks (pure) ────────────────────────────────────────────────────
def test_hard_required_token_present():
    v = hard_checks("the product is 391", ['output must contain "391"'])
    assert v is not None and v.approved and v.method == "hard"


def test_hard_required_token_absent_is_the_silent_success_case():
    v = hard_checks("a totally unrelated answer", ['output must contain "BANANA"'])
    assert v is not None and not v.approved


def test_hard_artifact_exists(tmp_path):
    p = tmp_path / "out.json"
    p.write_text("{}", encoding="utf-8")
    v = hard_checks("done", [f"the file {p} must exist"])
    assert v is not None and v.approved


def test_hard_artifact_missing():
    v = hard_checks("i created it, trust me", ["/definitely/not/real/xyz.txt must exist"])
    assert v is not None and not v.approved


def test_hard_subjective_only_falls_through():
    assert hard_checks("prose", ["the explanation should be clear"]) is None


# ── CompletionGate.run ────────────────────────────────────────────────────
async def test_pass_first_try():
    agent = FakeAgent(["the answer is OK"])
    gate = CompletionGate(criteria=['output must contain "OK"'], max_retries=2)
    resp = await gate.run(agent, "do it")
    assert resp.finish_reason == "verified"
    assert len(agent.calls) == 1


async def test_retry_then_recover_feeds_feedback_back():
    agent = FakeAgent(["nope", "now it is OK"])
    gate = CompletionGate(criteria=['output must contain "OK"'], max_retries=2)
    resp = await gate.run(agent, "do it")
    assert resp.finish_reason == "verified"
    assert len(agent.calls) == 2
    # the 2nd attempt must carry the failure feedback
    assert "FAILED THE COMPLETION CHECK" in agent.calls[1]
    assert gate.last_verdict.attempts == 2


async def test_never_soft_passes_exhausts_and_fails_honestly():
    agent = FakeAgent(["nope"])  # always wrong
    gate = CompletionGate(criteria=['output must contain "BANANA"'], max_retries=2)
    resp = await gate.run(agent, "do it")
    assert resp.finish_reason == "unverified"
    assert len(agent.calls) == 3  # 1 + 2 retries
    assert not gate.last_verdict.approved


async def test_artifact_gate_end_to_end(tmp_path):
    p = tmp_path / "report.json"
    # agent "does the work" only on its 2nd attempt
    written = {"n": 0}

    class WritingAgent(FakeAgent):
        async def run(self, task, **kwargs):
            self.calls.append(task)
            written["n"] += 1
            if written["n"] >= 2:
                p.write_text("{}", encoding="utf-8")
            return AgentResponse(content="ok")

    agent = WritingAgent([])
    gate = CompletionGate(criteria=[f"writes {p}"], max_retries=2)
    resp = await gate.run(agent, "write the report")
    assert resp.finish_reason == "verified"
    assert len(agent.calls) == 2
    assert p.exists()


async def test_no_criteria_is_liveness_pass():
    agent = FakeAgent(["whatever"])
    gate = CompletionGate(max_retries=1)
    resp = await gate.run(agent, "chat")
    assert resp.finish_reason == "verified"
    assert gate.last_verdict.method == "none"


async def test_subjective_without_judge_fails_closed():
    agent = FakeAgent(["some prose"], llm=None)  # no llm → no derived judge
    gate = CompletionGate(criteria=["the explanation should be clear and correct"], max_retries=0)
    resp = await gate.run(agent, "explain")
    assert resp.finish_reason == "unverified"
    assert gate.last_verdict.method == "none"


async def test_subjective_with_judge_approves():
    async def judge(_prompt):
        return '{"approved": true, "reason": "looks complete"}'

    agent = FakeAgent(["some prose"])
    gate = CompletionGate(
        criteria=["the explanation should be clear"], judge=judge, max_retries=0
    )
    resp = await gate.run(agent, "explain")
    assert resp.finish_reason == "verified"
    assert gate.last_verdict.method == "judge"


async def test_judge_exception_fails_closed():
    async def judge(_prompt):
        raise RuntimeError("model down")

    agent = FakeAgent(["prose"])
    gate = CompletionGate(criteria=["should be clear"], judge=judge, max_retries=0)
    resp = await gate.run(agent, "explain")
    assert resp.finish_reason == "unverified"


async def test_requires_action_short_circuits():
    class PausingAgent(FakeAgent):
        async def run(self, task, **kwargs):
            self.calls.append(task)
            return AgentResponse(content="", requires_action=True, pending=[{"tool": "x"}])

    agent = PausingAgent([])
    gate = CompletionGate(criteria=['output must contain "OK"'], max_retries=2)
    resp = await gate.run(agent, "do it")
    assert resp.requires_action is True
    assert resp.finish_reason == "stop"  # untouched — not marked verified/unverified
    assert len(agent.calls) == 1


async def test_gated_run_convenience():
    agent = FakeAgent(["contains OK here"])
    resp = await gated_run(agent, "do it", criteria=['output must contain "OK"'])
    assert resp.finish_reason == "verified"
