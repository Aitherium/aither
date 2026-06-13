"""adk never-fabricate coherence gate tests.

The gate guarantees a companion never invents shared history: a memory question
with no subject-grounded memory and no session history returns a deterministic
honest reply WITHOUT calling the LLM (so fabrication is impossible).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from adk.agent import AitherAgent
from adk.llm.base import LLMResponse
from adk.memory import Memory
from adk import coherence


# ── helper units ──
def test_memory_question_detection():
    assert coherence.is_memory_question("do you remember our Italy trip")
    assert coherence.is_memory_question("what did we talk about last weekend")
    assert not coherence.is_memory_question("hi babe how are you")
    assert not coherence.is_memory_question("tell me a joke")


def test_subject_grounding_rejects_offsubject_memory():
    # Italy memory must NOT count as recall for a Colorado question.
    assert not coherence.subject_grounded(
        "We are going to Italy next month", "do you remember the cabin in colorado")
    assert coherence.subject_grounded(
        "We rented a cabin in colorado", "do you remember the cabin in colorado")


def test_history_may_answer():
    assert coherence.history_may_answer(
        "where is our trip", [{"role": "user", "content": "our Italy trip is next month"}])
    assert not coherence.history_may_answer(
        "where is our trip", [{"role": "user", "content": "what's for dinner"}])


# ── gate integration ──
@pytest.fixture
def mock_llm():
    router = MagicMock()
    router.provider_name = "mock"
    router.chat = AsyncMock(return_value=LLMResponse(
        content="Mock response", model="mock-model", tokens_used=10, latency_ms=50.0))
    return router


@pytest.fixture
def tmp_memory(tmp_path):
    return Memory(db_path=tmp_path / "test.db", agent_name="test")


@pytest.fixture
def companion_vault():
    vault = MagicMock()
    vault.get_safety_level.return_value = "unrestricted"
    vault.get_system_prompt_for_level.return_value = "You are Avia, a devoted companion."
    return vault


@pytest.mark.asyncio
async def test_gate_blocks_fabrication_on_memory_miss(mock_llm, tmp_memory, companion_vault):
    with patch("adk.private_companion.get_companion_vault", return_value=companion_vault):
        agent = AitherAgent("test", llm=mock_llm, memory=tmp_memory)
        resp = await agent.chat(
            "do you remember our trip to Italy last summer?", session_id="s1")
    # Honest deterministic reply, LLM NEVER called → invention impossible.
    assert resp.content != "Mock response"
    assert resp.content in coherence._HONEST_MISS_REPLIES
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_gate_lets_non_memory_questions_through(mock_llm, tmp_memory, companion_vault):
    with patch("adk.private_companion.get_companion_vault", return_value=companion_vault):
        agent = AitherAgent("test", llm=mock_llm, memory=tmp_memory)
        await agent.chat("tell me a joke", session_id="s2")
    mock_llm.chat.assert_called()  # not a memory question → normal generation


@pytest.mark.asyncio
async def test_gate_answers_from_session_history(mock_llm, tmp_memory, companion_vault):
    with patch("adk.private_companion.get_companion_vault", return_value=companion_vault):
        agent = AitherAgent("test", llm=mock_llm, memory=tmp_memory)
        await agent.memory.add_message("s3", "user", "our Italy trip is next month, so excited")
        await agent.chat("remind me where is our trip again?", session_id="s3")
    # Subject is in this conversation → answer from history (LLM called), not honest-fallback.
    mock_llm.chat.assert_called()
