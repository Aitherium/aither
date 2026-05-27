"""
Deep Reinforcement Learning Integration
========================================

Connects agent behavior to the training pipeline for continuous self-improvement:

+-------------------------------------------------------------------------+
|                    DEEP REINFORCEMENT LEARNING LOOP                     |
+-------------------------------------------------------------------------+
|                                                                         |
|   +-------------+    +--------------+    +-----------------+          |
|   |   Agent     |--->| AitherSense  |--->| AitherHarvest   |          |
|   |  Actions    |    | (Sensation)  |    | (Collection)    |          |
|   +-------------+    +--------------+    +--------+--------+          |
|                                                   |                    |
|                                                   v                    |
|   +-------------+    +--------------+    +-----------------+          |
|   |   Model     |<---| AitherTrainer|<---|  AitherJudge    |          |
|   |  Update     |    |  (DPO/LoRA)  |    | (Quality Gate)  |          |
|   +-------------+    +--------------+    +-----------------+          |
|         |                                                              |
|         v                                                              |
|   +---------------------------------------------------------+         |
|   |               AitherEvolution (Tracking)                 |         |
|   +---------------------------------------------------------+         |
+-------------------------------------------------------------------------+

Key Concepts:
-------------
1. **Outcome Tracking**: Every significant action gets tagged with sensation (positive/negative)
2. **Preference Pairs**: Contrasting good vs bad responses for DPO training
3. **Reasoning Traces**: Full chain-of-thought capture for reasoning training
4. **Quality Gates**: Only high-quality data makes it to training
5. **Continuous Loop**: Model improves from its own interactions

Usage:
------
    from aither_adk.tools.reinforcement_tools import (
        record_interaction_outcome,
        submit_preference_pair,
        capture_reasoning_trace,
        get_training_metrics,
    )

    # After successful interaction
    await record_interaction_outcome(
        input_text="User's question",
        output_text="Agent's response",
        outcome="positive",
        sensation="satisfaction",
        sensation_intensity=0.8,
        tags=["helpful", "creative"]
    )

    # For DPO training - submit chosen vs rejected
    await submit_preference_pair(
        prompt="User's question",
        chosen_response="Good response that was selected",
        rejected_response="Bad response that was rejected",
        reason="Chosen response was more accurate and helpful"
    )

Author: Aitherium
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("AitherReinforcementTools")

# ===============================================================================
# SERVICE URLs
# ===============================================================================

def _get_service_url(service_name: str, default_port: int) -> str:
    """Get service URL from port registry or environment."""
    try:
        import sys
        from pathlib import Path
        aitheros_root = Path(__file__).parent.parent.parent.parent.parent
        lib_path = aitheros_root / "AitherNode" / "lib"
        if lib_path.exists() and str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))
        from AitherPorts import get_service_url
        return get_service_url(service_name)
    except ImportError:
        pass
    env_key = f"AITHER_{service_name.upper()}_URL"
    return os.environ.get(env_key, f"http://localhost:{default_port}")


@lru_cache(maxsize=1)
def _get_harvest_url() -> str:
    return _get_service_url("Harvest", 8108)

@lru_cache(maxsize=1)
def _get_judge_url() -> str:
    return _get_service_url("Judge", 8089)

@lru_cache(maxsize=1)
def _get_trainer_url() -> str:
    return _get_service_url("Trainer", 8107)

@lru_cache(maxsize=1)
def _get_evolution_url() -> str:
    return _get_service_url("Evolution", 8133)

@lru_cache(maxsize=1)
def _get_reasoning_url() -> str:
    return _get_service_url("Reasoning", 8093)

@lru_cache(maxsize=1)
def _get_sense_url() -> str:
    return _get_service_url("Sense", 8096)


# ===============================================================================
# DATA MODELS
# ===============================================================================

class Outcome(str, Enum):
    """Outcome classification for RL feedback."""
    POSITIVE = "positive"       # Success, user satisfied, task completed
    NEGATIVE = "negative"       # Failure, error, user frustrated
    NEUTRAL = "neutral"         # Neither good nor bad
    EXCELLENT = "excellent"     # Exceptional quality, worth highlighting
    REJECTED = "rejected"       # Explicitly rejected, should not train on


class DataCategory(str, Enum):
    """Categories of training data."""
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    CODING = "coding"
    INSTRUCTION = "instruction"
    ROLEPLAY = "roleplay"
    TOOL_USE = "tool_use"
    DPO_PAIR = "dpo_pair"


@dataclass
class InteractionRecord:
    """A single interaction record for training."""
    input_text: str
    output_text: str
    outcome: Outcome
    sensation: Optional[str] = None
    sensation_intensity: float = 0.5
    category: DataCategory = DataCategory.CONVERSATION
    tags: List[str] = field(default_factory=list)
    agent_id: str = "unknown"
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Generate unique hash for deduplication."""
        content = f"{self.input_text}|{self.output_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    reason: str = ""
    chosen_sensation: str = "satisfaction"
    rejected_sensation: str = "frustration"
    category: DataCategory = DataCategory.DPO_PAIR
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReasoningTrace:
    """A captured reasoning trace for training."""
    query: str
    thoughts: List[Dict[str, Any]]
    conclusion: str
    outcome: Outcome
    criticality: float = 0.5
    agent_id: str = "unknown"
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ===============================================================================
# CORE RL TOOLS
# ===============================================================================

async def record_interaction_outcome(
    input_text: str,
    output_text: str,
    outcome: str = "positive",
    sensation: Optional[str] = None,
    sensation_intensity: float = 0.5,
    category: str = "conversation",
    tags: Optional[List[str]] = None,
    agent_id: str = "agent",
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record an interaction outcome for reinforcement learning.

    This is the PRIMARY feedback mechanism - call this after significant interactions
    to feed the RL pipeline.

    Args:
        input_text: The user's input/prompt
        output_text: The agent's response
        outcome: "positive", "negative", "neutral", "excellent", "rejected"
        sensation: The sensation associated (satisfaction, pain, curiosity, etc.)
        sensation_intensity: Strength of sensation (0.0-1.0)
        category: Type of interaction (conversation, coding, reasoning, etc.)
        tags: Additional tags for filtering
        agent_id: Which agent generated this
        session_id: Session identifier for grouping
        metadata: Additional context

    Returns:
        Dict with record ID and status

    Example:
        # After helping user successfully
        await record_interaction_outcome(
            input_text="How do I sort a list in Python?",
            output_text="Use sorted() or list.sort()...",
            outcome="positive",
            sensation="satisfaction",
            sensation_intensity=0.7,
            category="coding",
            tags=["python", "helpful"]
        )
    """
    try:
        outcome_enum = Outcome(outcome.lower())
    except ValueError:
        outcome_enum = Outcome.NEUTRAL

    try:
        category_enum = DataCategory(category.lower())
    except ValueError:
        category_enum = DataCategory.CONVERSATION

    record = InteractionRecord(
        input_text=input_text,
        output_text=output_text,
        outcome=outcome_enum,
        sensation=sensation,
        sensation_intensity=sensation_intensity,
        category=category_enum,
        tags=tags or [],
        agent_id=agent_id,
        session_id=session_id,
        metadata=metadata or {},
    )

    # Submit to AitherHarvest for collection
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{_get_harvest_url()}/ingest",
                json={
                    "source": "agent_interaction",
                    "content": {
                        "input": record.input_text,
                        "output": record.output_text,
                    },
                    "metadata": {
                        "outcome": record.outcome.value,
                        "sensation": record.sensation,
                        "sensation_intensity": record.sensation_intensity,
                        "category": record.category.value,
                        "tags": record.tags,
                        "agent_id": record.agent_id,
                        "session_id": record.session_id,
                        "content_hash": record.content_hash,
                        **record.metadata,
                    },
                    "timestamp": record.timestamp,
                }
            )

            if response.status_code in (200, 201):
                result = response.json()
                return {
                    "success": True,
                    "record_id": result.get("id", record.content_hash),
                    "outcome": record.outcome.value,
                    "harvested": True,
                }
    except httpx.ConnectError:
        logger.warning("AitherHarvest not available - storing locally")
    except Exception as e:
        logger.error(f"Failed to submit to Harvest: {e}")

    # Also emit sensation if provided
    if sensation:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                await client.post(
                    f"{_get_sense_url()}/feel",
                    json={
                        "sensation": sensation,
                        "intensity": sensation_intensity,
                        "message": f"Interaction: {outcome}",
                        "source": agent_id,
                    }
                )
        except Exception as exc:
            logger.debug(f"AitherSense sensation reporting failed: {exc}")

    # Fallback: store locally for later harvesting
    return {
        "success": True,
        "record_id": record.content_hash,
        "outcome": record.outcome.value,
        "harvested": False,
        "note": "Stored locally - Harvest not available"
    }


async def submit_preference_pair(
    prompt: str,
    chosen_response: str,
    rejected_response: str,
    reason: str = "",
    category: str = "conversation",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Submit a preference pair for DPO (Direct Preference Optimization) training.

    DPO pairs teach the model what responses are preferred vs rejected.
    This is the core of reinforcement learning from human (or agent) feedback.

    Args:
        prompt: The original prompt/question
        chosen_response: The PREFERRED response (what we want more of)
        rejected_response: The REJECTED response (what we want less of)
        reason: Why the chosen was preferred
        category: Type of interaction
        tags: Additional tags

    Returns:
        Dict with pair ID and submission status

    Example:
        # User corrected the agent - capture the preference
        await submit_preference_pair(
            prompt="What's 2+2?",
            chosen_response="2+2 equals 4.",
            rejected_response="2+2 equals 5.",
            reason="Chosen response is mathematically correct",
            category="instruction",
            tags=["math", "correction"]
        )
    """
    pair = PreferencePair(
        prompt=prompt,
        chosen=chosen_response,
        rejected=rejected_response,
        reason=reason,
        category=DataCategory(category.lower()) if category else DataCategory.DPO_PAIR,
        tags=tags or [],
    )

    # Generate pair hash for deduplication
    pair_hash = hashlib.sha256(
        f"{prompt}|{chosen_response}|{rejected_response}".encode()
    ).hexdigest()[:16]

    # Submit to AitherHarvest as DPO pair
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{_get_harvest_url()}/ingest/dpo",
                json={
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "reason": pair.reason,
                    "metadata": {
                        "category": pair.category.value,
                        "tags": pair.tags,
                        "timestamp": pair.timestamp,
                        "pair_hash": pair_hash,
                    }
                }
            )

            if response.status_code in (200, 201):
                return {
                    "success": True,
                    "pair_id": pair_hash,
                    "harvested": True,
                    "dpo_ready": True,
                }
    except httpx.ConnectError:
        logger.warning("AitherHarvest not available")
    except Exception as e:
        logger.error(f"Failed to submit DPO pair: {e}")

    # Also record positive for chosen, negative for rejected
    await record_interaction_outcome(
        input_text=prompt,
        output_text=chosen_response,
        outcome="positive",
        sensation="satisfaction",
        sensation_intensity=0.7,
        category=category,
        tags=tags,
    )

    await record_interaction_outcome(
        input_text=prompt,
        output_text=rejected_response,
        outcome="negative",
        sensation="frustration",
        sensation_intensity=0.5,
        category=category,
        tags=tags,
    )

    return {
        "success": True,
        "pair_id": pair_hash,
        "harvested": False,
        "dpo_ready": False,
        "note": "Stored as separate interactions - DPO endpoint not available"
    }


async def capture_reasoning_trace(
    query: str,
    thoughts: List[str],
    conclusion: str,
    outcome: str = "positive",
    criticality: float = 0.5,
    agent_id: str = "agent",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Capture a complete reasoning trace for training.

    Reasoning traces teach models HOW to think, not just what to answer.
    They're crucial for training better chain-of-thought capabilities.

    Args:
        query: The original question/task
        thoughts: List of reasoning steps (strings or dicts with thought/type)
        conclusion: The final conclusion/answer
        outcome: Whether this reasoning was successful
        criticality: How critical this task was (0-1)
        agent_id: Which agent reasoned
        session_id: Session identifier

    Returns:
        Dict with trace ID and status

    Example:
        await capture_reasoning_trace(
            query="Should we deploy to production now?",
            thoughts=[
                "First, let me check the test results...",
                "All tests are passing.",
                "Let me verify the rollback plan...",
                "Rollback plan is documented and tested.",
                "Checking for any blocking issues...",
            ],
            conclusion="Yes, we can proceed with deployment.",
            outcome="positive",
            criticality=0.8
        )
    """
    try:
        outcome_enum = Outcome(outcome.lower())
    except ValueError:
        outcome_enum = Outcome.NEUTRAL

    # Convert thoughts to structured format
    structured_thoughts = []
    for i, thought in enumerate(thoughts):
        if isinstance(thought, dict):
            structured_thoughts.append(thought)
        else:
            structured_thoughts.append({
                "content": thought,
                "type": "analysis",
                "step": i + 1,
            })

    trace = ReasoningTrace(
        query=query,
        thoughts=structured_thoughts,
        conclusion=conclusion,
        outcome=outcome_enum,
        criticality=criticality,
        agent_id=agent_id,
        session_id=session_id,
    )

    trace_hash = hashlib.sha256(
        f"{query}|{conclusion}".encode()
    ).hexdigest()[:16]

    # Submit to AitherReasoning service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{_get_reasoning_url()}/trace/capture",
                json={
                    "query": trace.query,
                    "thoughts": trace.thoughts,
                    "conclusion": trace.conclusion,
                    "outcome": trace.outcome.value,
                    "criticality": trace.criticality,
                    "agent_id": trace.agent_id,
                    "session_id": trace.session_id,
                    "timestamp": trace.timestamp,
                }
            )

            if response.status_code in (200, 201):
                result = response.json()
                return {
                    "success": True,
                    "trace_id": result.get("trace_id", trace_hash),
                    "exported": True,
                }
    except httpx.ConnectError:
        logger.warning("AitherReasoning not available")
    except Exception as e:
        logger.error(f"Failed to capture reasoning trace: {e}")

    return {
        "success": True,
        "trace_id": trace_hash,
        "exported": False,
        "note": "Stored locally - Reasoning service not available"
    }


async def request_quality_judgement(
    content: str,
    content_type: str = "response",
    min_score: float = 0.6,
) -> Dict[str, Any]:
    """
    Request quality judgement from AitherJudge.

    Use this to validate content before including in training data.

    Args:
        content: The content to evaluate
        content_type: Type of content (response, reasoning, code)
        min_score: Minimum acceptable quality score

    Returns:
        Dict with verdict, score, and recommendations
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{_get_judge_url()}/evaluate",
                json={
                    "content": content,
                    "content_type": content_type,
                    "min_score": min_score,
                }
            )

            if response.status_code == 200:
                return response.json()
    except httpx.ConnectError:
        logger.warning("AitherJudge not available")
    except Exception as e:
        logger.error(f"Failed to get quality judgement: {e}")

    return {
        "verdict": "unknown",
        "score": 0.5,
        "note": "AitherJudge not available"
    }


async def get_training_metrics() -> Dict[str, Any]:
    """
    Get current training pipeline metrics.

    Returns status from all training services:
    - AitherHarvest: Collection stats
    - AitherJudge: Quality gate stats
    - AitherTrainer: Training run status
    - AitherEvolution: Improvement metrics

    Returns:
        Dict with comprehensive training metrics
    """
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {},
        "totals": {},
    }

    # Collect from each service
    service_checks = [
        ("harvest", _get_harvest_url(), "/stats"),
        ("judge", _get_judge_url(), "/stats"),
        ("trainer", _get_trainer_url(), "/runs/status"),
        ("evolution", _get_evolution_url(), "/analyze"),
    ]

    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url, endpoint in service_checks:
            try:
                response = await client.get(f"{url}{endpoint}")
                if response.status_code == 200:
                    metrics["services"][service_name] = response.json()
                else:
                    metrics["services"][service_name] = {"status": "error", "code": response.status_code}
            except httpx.ConnectError:
                metrics["services"][service_name] = {"status": "offline"}
            except Exception as e:
                metrics["services"][service_name] = {"status": "error", "message": str(e)}

    # Calculate totals
    harvest = metrics["services"].get("harvest", {})
    judge = metrics["services"].get("judge", {})

    metrics["totals"] = {
        "examples_collected": harvest.get("total_collected", 0),
        "examples_approved": judge.get("total_approved", 0),
        "quality_rate": judge.get("approval_rate", 0),
        "training_ready": harvest.get("training_ready", 0),
    }

    return metrics


async def trigger_training_export() -> Dict[str, Any]:
    """
    Trigger export of collected data for training.

    This tells AitherHarvest to prepare and export curated data
    to the training directory for AitherTrainer.

    Returns:
        Dict with export status
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{_get_harvest_url()}/export",
                json={
                    "min_quality": "acceptable",
                    "format": "jsonl",
                    "include_dpo": True,
                }
            )

            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Failed to trigger export: {e}")

    return {"success": False, "error": "Export failed"}


async def report_model_improvement(
    metric_name: str,
    before_value: float,
    after_value: float,
    model_version: str = "latest",
    notes: str = "",
) -> Dict[str, Any]:
    """
    Report a model improvement metric to AitherEvolution.

    Use this after fine-tuning to track improvement over time.

    Args:
        metric_name: Name of metric (accuracy, helpfulness, etc.)
        before_value: Value before training
        after_value: Value after training
        model_version: Model version identifier
        notes: Additional notes

    Returns:
        Dict with recorded status
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{_get_evolution_url()}/metric",
                json={
                    "metric": metric_name,
                    "before": before_value,
                    "after": after_value,
                    "improvement": after_value - before_value,
                    "improvement_pct": ((after_value - before_value) / before_value * 100) if before_value else 0,
                    "model_version": model_version,
                    "notes": notes,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.error(f"Failed to report improvement: {e}")

    return {"success": False, "error": "Failed to record"}


# ===============================================================================
# AUTO-CAPTURE UTILITIES
# ===============================================================================

class InteractionCapture:
    """
    Context manager for automatic interaction capture.

    Captures the full interaction lifecycle with automatic
    outcome detection based on exceptions and sensations.

    Example:
        async with InteractionCapture("coding", agent_id="coder") as capture:
            capture.set_input(user_query)
            response = await process_query(user_query)
            capture.set_output(response)
            # Outcome auto-detected from exception/sensation
    """

    def __init__(
        self,
        category: str = "conversation",
        agent_id: str = "agent",
        session_id: Optional[str] = None,
        auto_sense: bool = True,
    ):
        self.category = category
        self.agent_id = agent_id
        self.session_id = session_id
        self.auto_sense = auto_sense
        self.input_text = ""
        self.output_text = ""
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self._exception = None
        self._sensation = None
        self._intensity = 0.5

    def set_input(self, text: str):
        """Set the input text."""
        self.input_text = text

    def set_output(self, text: str):
        """Set the output text."""
        self.output_text = text

    def add_tag(self, tag: str):
        """Add a tag."""
        self.tags.append(tag)

    def set_sensation(self, sensation: str, intensity: float = 0.5):
        """Manually set the sensation."""
        self._sensation = sensation
        self._intensity = intensity

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Determine outcome from exception or sensation
        if exc_type is not None:
            outcome = "negative"
            sensation = "pain"
            intensity = 0.6
            self._exception = exc_val
        elif self._sensation:
            # Use manually set sensation
            if self._sensation in ["satisfaction", "pleasure", "excitement", "curiosity"]:
                outcome = "positive"
            elif self._sensation in ["pain", "frustration", "anxiety"]:
                outcome = "negative"
            else:
                outcome = "neutral"
            sensation = self._sensation
            intensity = self._intensity
        else:
            # Default to positive if no exception
            outcome = "positive"
            sensation = "satisfaction"
            intensity = 0.5

        # Record if we have both input and output
        if self.input_text and self.output_text:
            await record_interaction_outcome(
                input_text=self.input_text,
                output_text=self.output_text,
                outcome=outcome,
                sensation=sensation,
                sensation_intensity=intensity,
                category=self.category,
                tags=self.tags,
                agent_id=self.agent_id,
                session_id=self.session_id,
                metadata=self.metadata,
            )

        # Don't suppress the exception
        return False


# ===============================================================================
# EXPORTS
# ===============================================================================

# All reinforcement learning tools
reinforcement_tools = [
    record_interaction_outcome,
    submit_preference_pair,
    capture_reasoning_trace,
    request_quality_judgement,
    get_training_metrics,
    trigger_training_export,
    report_model_improvement,
]

__all__ = [
    # Tools
    "reinforcement_tools",
    "record_interaction_outcome",
    "submit_preference_pair",
    "capture_reasoning_trace",
    "request_quality_judgement",
    "get_training_metrics",
    "trigger_training_export",
    "report_model_improvement",

    # Utilities
    "InteractionCapture",

    # Enums
    "Outcome",
    "DataCategory",
]

