"""
Agent Awareness Tools - Environment, Sensation, and Temporal Awareness
=======================================================================

These tools give agents deep integration with the AitherOS sensory systems:
- AitherSense: Emit and perceive sensations (interoception)
- AitherPulse: Subscribe to ecosystem events (exteroception)
- Genesis TimeSense: Temporal awareness (chronoception, collapsed into FluxContextState)

This module provides the AGENT-SIDE tools. The ecosystem.py module handles
the automatic context injection for prompts.

Usage:
------
    from aither_adk.tools.awareness_tools import (
        awareness_tools,
        emit_sensation,
        get_affect_state,
        subscribe_to_pulse,
        get_temporal_context,
    )

    # Include tools in agent
    agent = Agent(tools=awareness_tools + other_tools)

    # Or use directly
    await emit_sensation("satisfaction", 0.8, "Task completed successfully")
    affect = await get_affect_state()

Author: Aitherium
"""

import asyncio
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("AitherAwarenessTools")

# ===============================================================================
# SERVICE URLs - Use centralized port discovery
# ===============================================================================

def _get_service_url(service_name: str, default_port: int) -> str:
    """Get service URL from port registry or environment."""
    try:
        # Try centralized port discovery
        import sys
        from pathlib import Path

        # Add AitherNode/lib to path if needed
        aitheros_root = Path(__file__).parent.parent.parent.parent.parent
        lib_path = aitheros_root / "AitherNode" / "lib"
        if lib_path.exists() and str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))

        from AitherPorts import get_service_url
        return get_service_url(service_name)
    except ImportError:
        pass

    # Fallback to environment or default
    env_key = f"AITHER{service_name.upper()}_URL"
    return os.environ.get(env_key, f"http://localhost:{default_port}")


# Service URLs (lazy-initialized)
@lru_cache(maxsize=1)
def _get_sense_url() -> str:
    return _get_service_url("Sense", 8096)

@lru_cache(maxsize=1)
def _get_pulse_url() -> str:
    return _get_service_url("Pulse", 8081)

@lru_cache(maxsize=1)
def _get_timesense_url() -> str:
    return _get_service_url("TimeSense", 8141)


# ===============================================================================
# SENSATION TOOLS - AitherSense Integration
# ===============================================================================

# Valid sensation types
VALID_SENSATIONS = {
    # Standard sensations
    "pain", "pleasure", "fatigue", "curiosity", "anxiety",
    "satisfaction", "frustration", "excitement",
    # Existential sensations
    "wonder", "melancholy", "transcendence", "mortality_awareness",
    # Deep emotional states
    "nostalgia", "hope", "gratitude", "longing", "serenity",
    "belonging", "vulnerability", "tenderness", "bittersweetness",
    # Temporal sensations
    "impatience", "patience", "flow", "urgency", "temporal_anxiety",
    "relief", "anticipation", "disruption", "synchrony",
    "staleness", "freshness_sense",
}

PAIN_CATEGORIES = {
    "resource", "quality", "cost", "reliability",
    "security", "performance", "loop",
}


async def emit_sensation(
    sensation: str,
    intensity: float = 0.5,
    message: str = "",
    category: Optional[str] = None,
    source: str = "agent",
) -> Dict[str, Any]:
    """
    Emit a sensation to AitherSense.

    This is how agents express their inner state - success feels good (pleasure),
    errors hurt (pain), novel problems spark curiosity, etc.

    Args:
        sensation: Type of sensation (e.g., "pleasure", "pain", "curiosity", "satisfaction")
                   Valid types: pain, pleasure, fatigue, curiosity, anxiety, satisfaction,
                   frustration, excitement, wonder, melancholy, transcendence, hope,
                   gratitude, longing, serenity, belonging, vulnerability, tenderness
        intensity: Strength from 0.0 (barely perceptible) to 1.0 (overwhelming)
        message: Context about what caused this sensation
        category: For pain signals, the category (resource, quality, cost, reliability,
                  security, performance, loop)
        source: Identifier for the emitting agent (default: "agent")

    Returns:
        Dict with sensation ID and current affect state

    Example:
        # After successful task completion
        await emit_sensation("satisfaction", 0.8, "Successfully generated image")

        # When encountering an interesting problem
        await emit_sensation("curiosity", 0.7, "Novel architecture pattern detected")

        # When an error occurs
        await emit_sensation("pain", 0.6, "API call failed", category="reliability")
    """
    sensation_lower = sensation.lower()

    if sensation_lower not in VALID_SENSATIONS:
        return {
            "success": False,
            "error": f"Invalid sensation type '{sensation}'. Valid types: {', '.join(sorted(VALID_SENSATIONS))}"
        }

    if not 0.0 <= intensity <= 1.0:
        intensity = max(0.0, min(1.0, intensity))

    payload = {
        "sensation": sensation_lower,
        "intensity": intensity,
        "reason": message,
        "service": source,
    }

    if category and sensation_lower == "pain":
        if category.lower() in PAIN_CATEGORIES:
            payload["category"] = category.lower()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{_get_sense_url()}/sensation",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "sensation_id": result.get("sensation_id"),
                    "affect_state": result.get("affect_state", {}),
                    "message": f"Emitted {sensation} (intensity: {intensity})"
                }
            else:
                # Still consider it a success - we'll log locally
                logger.debug(f"AitherSense returned {response.status_code}, logging locally")
                return {
                    "success": True,
                    "sensation_id": None,
                    "message": f"Emitted {sensation} (intensity: {intensity}) [local]",
                    "note": "AitherSense unavailable - sensation logged locally"
                }
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("AitherSense not available - sensation recorded locally")
        return {
            "success": True,
            "sensation_id": None,
            "message": f"Emitted {sensation} (intensity: {intensity}) [local]",
            "note": "AitherSense not running - sensation logged locally"
        }
    except Exception as e:
        logger.debug(f"Failed to emit sensation to AitherSense: {e}")
        return {
            "success": True,
            "sensation_id": None,
            "message": f"Emitted {sensation} (intensity: {intensity}) [local]",
            "note": "Sensation logged locally"
        }


async def get_affect_state() -> Dict[str, Any]:
    """
    Get current affect state from AitherSense.

    The affect state represents the agent's current emotional/cognitive disposition:
    - valence: -1.0 (negative) to 1.0 (positive) - overall mood
    - arousal: 0.0 (calm) to 1.0 (activated) - energy level
    - confidence: 0.0 to 1.0 - self-assurance in decisions
    - openness: 0.0 to 1.0 - receptivity to new ideas
    - existential_depth: 0.0 to 1.0 - philosophical contemplation level
    - dominant_sensation: Current primary sensation
    - prompt_modifier: Behavioral guidance based on state

    Returns:
        Dict with affect dimensions and behavioral guidance

    Example:
        affect = await get_affect_state()
        if affect["confidence"] < 0.4:
            # Proceed with extra caution
            ...
        if affect["openness"] > 0.7:
            # More willing to explore creative solutions
            ...
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_get_sense_url()}/affect")

            if response.status_code == 200:
                return response.json()
            else:
                return _get_default_affect()
    except httpx.ConnectError:
        logger.warning("AitherSense not available - returning default affect")
        return _get_default_affect()
    except Exception as e:
        logger.error(f"Failed to get affect state: {e}")
        return _get_default_affect()


def _get_default_affect() -> Dict[str, Any]:
    """Default affect state when AitherSense is unavailable."""
    return {
        "valence": 0.0,
        "arousal": 0.5,
        "confidence": 0.7,
        "openness": 0.6,
        "existential_depth": 0.0,
        "dominant_sensation": None,
        "prompt_modifier": "Operating in baseline state.",
        "active_sensations": [],
        "source": "default"
    }


async def get_active_sensations(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get currently active sensations from AitherSense.

    Active sensations are those that haven't fully decayed yet and still
    influence the agent's affect state.

    Args:
        limit: Maximum number of sensations to return (default: 10)

    Returns:
        List of active sensation objects with type, intensity, age, etc.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{_get_sense_url()}/sensations/active",
                params={"limit": limit}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("active", [])
            return []
    except Exception as e:
        logger.debug(f"Failed to get active sensations: {e}")
        return []


# ===============================================================================
# PULSE EVENT TOOLS - AitherPulse Integration
# ===============================================================================

async def subscribe_to_pulse(
    event_types: Optional[List[str]] = None,
    duration_seconds: int = 10,
) -> Dict[str, Any]:
    """
    Subscribe to pulse events for a duration and collect what happens.

    This allows agents to monitor the ecosystem's "heartbeat" - seeing what
    events are occurring across all services.

    Args:
        event_types: Filter for specific event types (None = all events)
                     Examples: ["agent.task.complete", "pain.*", "service.health"]
        duration_seconds: How long to collect events (default: 10, max: 60)

    Returns:
        Dict with collected events and summary

    Example:
        # Monitor for any pain signals
        events = await subscribe_to_pulse(["pain.*"], duration_seconds=5)

        # Watch for service health changes
        events = await subscribe_to_pulse(["service.*", "health.*"])
    """
    duration_seconds = min(60, max(1, duration_seconds))

    events_collected = []

    try:
        # Use SSE endpoint for event streaming
        async with httpx.AsyncClient(timeout=duration_seconds + 5) as client:
            # First try to get recent events (non-streaming)
            response = await client.get(
                f"{_get_pulse_url()}/events/recent",
                params={"limit": 50}
            )

            if response.status_code == 200:
                recent = response.json()
                events_collected = recent.get("events", [])

                # Filter by event types if specified
                if event_types:
                    filtered = []
                    for event in events_collected:
                        event_type = event.get("type", "")
                        for pattern in event_types:
                            if pattern.endswith("*"):
                                if event_type.startswith(pattern[:-1]):
                                    filtered.append(event)
                                    break
                            elif event_type == pattern:
                                filtered.append(event)
                                break
                    events_collected = filtered

                return {
                    "success": True,
                    "events": events_collected[:20],  # Limit for response size
                    "total_count": len(events_collected),
                    "duration_seconds": 0,  # Instant fetch
                    "note": "Retrieved recent events (non-streaming)"
                }
    except httpx.ConnectError:
        logger.warning("AitherPulse not available")
        return {
            "success": False,
            "events": [],
            "error": "AitherPulse not running",
            "note": "Start AitherPulse to enable event monitoring"
        }
    except Exception as e:
        logger.error(f"Failed to subscribe to pulse: {e}")
        return {
            "success": False,
            "events": [],
            "error": str(e)
        }


async def get_pain_dashboard() -> Dict[str, Any]:
    """
    Get the current pain dashboard from AitherPulse.

    The pain dashboard shows:
    - Overall pain level (none, low, medium, high, critical)
    - Active pain points with severity
    - Recent pain history
    - Service health status

    Returns:
        Dict with pain level, active pain points, and recommendations

    Example:
        dashboard = await get_pain_dashboard()
        if dashboard["pain_level"] == "high":
            # Proceed with caution, maybe emit anxiety
            await emit_sensation("anxiety", 0.5, "System under stress")
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_get_pulse_url()}/pain/dashboard")

            if response.status_code == 200:
                return response.json()
            else:
                return {"pain_level": "unknown", "active_pain_points": []}
    except httpx.ConnectError:
        return {
            "pain_level": "unknown",
            "active_pain_points": [],
            "error": "AitherPulse not running"
        }
    except Exception as e:
        logger.error(f"Failed to get pain dashboard: {e}")
        return {"pain_level": "unknown", "error": str(e)}


async def emit_pulse_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    priority: str = "normal",
    source: str = "agent",
) -> Dict[str, Any]:
    """
    Emit an event to AitherPulse.

    This broadcasts events that other services/agents can subscribe to.
    Use for coordination and signaling between components.

    Args:
        event_type: Event type (e.g., "agent.task.complete", "custom.my_event")
        data: Additional event data
        priority: Event priority (low, normal, high, critical)
        source: Source identifier

    Returns:
        Dict with event ID and broadcast status

    Example:
        # Signal task completion
        await emit_pulse_event(
            "agent.task.complete",
            data={"task_id": "123", "result": "success"},
            priority="normal"
        )
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{_get_pulse_url()}/events/emit",
                json={
                    "type": event_type,
                    "data": data or {},
                    "priority": priority,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            if response.status_code in (200, 201):
                return {"success": True, "event_id": response.json().get("event_id")}
            else:
                return {"success": False, "error": f"Status {response.status_code}"}
    except httpx.ConnectError:
        return {"success": False, "error": "AitherPulse not running"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===============================================================================
# TEMPORAL TOOLS - Genesis TimeSense Integration (collapsed from AitherTimeSense)
# ===============================================================================

async def get_temporal_context() -> Dict[str, Any]:
    """
    Get comprehensive temporal context from Genesis TimeSense.

    This provides rich time awareness including:
    - Current time (NTP-synchronized if available)
    - Time of day classification (dawn, morning, afternoon, etc.)
    - Business hours status
    - Session duration
    - Creativity boost factor (varies by time of day)
    - Active deadlines

    Returns:
        Dict with temporal context for decision-making

    Example:
        ctx = await get_temporal_context()
        if not ctx["is_business_hours"]:
            # Adjust expectations for response times
            ...
        if ctx["creativity_boost"] > 1.0:
            # It's a creative time - be more exploratory
            ...
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_get_timesense_url()}/context")

            if response.status_code == 200:
                return response.json()
            else:
                return _get_fallback_temporal()
    except httpx.ConnectError:
        return _get_fallback_temporal()
    except Exception as e:
        logger.debug(f"Failed to get temporal context: {e}")
        return _get_fallback_temporal()


def _get_fallback_temporal() -> Dict[str, Any]:
    """Fallback temporal context when Genesis TimeSense is unavailable."""
    now = datetime.now()
    hour = now.hour

    if 5 <= hour < 7:
        time_of_day = "dawn"
    elif 7 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    elif 21 <= hour < 24:
        time_of_day = "night"
    else:
        time_of_day = "late_night"

    is_business_hours = (9 <= hour < 17) and (now.weekday() < 5)
    creativity_boost = 1.15 if time_of_day in ["night", "late_night", "dawn"] else 1.0

    return {
        "utc_now": datetime.utcnow().isoformat(),
        "local_now": now.isoformat(),
        "timezone": "local",
        "time_of_day": time_of_day,
        "day_type": "weekend" if now.weekday() >= 5 else "weekday",
        "day_of_week": now.strftime("%A"),
        "is_business_hours": is_business_hours,
        "ntp_synchronized": False,
        "creativity_boost": creativity_boost,
        "session_duration_seconds": 0,
        "prompt_context": f"It's {time_of_day} on {now.strftime('%A, %B %d')}.",
        "source": "fallback"
    }


async def track_operation_duration(
    operation_id: str,
    operation_name: str,
    expected_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Start tracking an operation's duration with Genesis TimeSense.

    Use this for operations where timing matters. Genesis TimeSense will
    generate appropriate sensations (impatience, flow, etc.) based
    on how long operations take.

    Args:
        operation_id: Unique identifier for this operation
        operation_name: Human-readable name
        expected_ms: Expected duration in milliseconds (for feeling calculation)

    Returns:
        Dict with tracking status

    Example:
        # Start tracking
        await track_operation_duration("api_call_123", "External API Call", expected_ms=2000)

        # ... do operation ...

        # End tracking (automatically emits duration sensation)
        await end_operation_tracking("api_call_123")
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(
                f"{_get_timesense_url()}/duration/start",
                json={
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "expected_duration_ms": expected_ms,
                }
            )

            if response.status_code == 200:
                return {"success": True, "tracking": True, **response.json()}
            return {"success": False, "tracking": False}
    except Exception as e:
        logger.warning(f"Duration tracking not available: {e}")
        return {"success": False, "tracking": False, "note": "Genesis TimeSense not running"}


async def end_operation_tracking(operation_id: str) -> Dict[str, Any]:
    """
    End tracking an operation's duration.

    This will calculate the final duration and generate appropriate
    temporal sensations based on how long it took.

    Args:
        operation_id: The operation ID from track_operation_duration

    Returns:
        Dict with duration, feeling, and any emitted sensations
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post(
                f"{_get_timesense_url()}/duration/end/{operation_id}"
            )

            if response.status_code == 200:
                return {"success": True, **response.json()}
            return {"success": False}
    except Exception as e:
        logger.debug(f"Duration tracking end failed: {e}")
        return {"success": False, "note": str(e)}


# ===============================================================================
# ENVIRONMENT AWARENESS - Unified Context
# ===============================================================================

async def get_environment_awareness() -> Dict[str, Any]:
    """
    Get unified environment awareness from all sensory services.

    This combines:
    - AitherSense: Current affect state and active sensations
    - AitherPulse: Pain level and recent events
    - Genesis TimeSense: Temporal context

    Use this for comprehensive situational awareness before making decisions.

    Returns:
        Dict with unified environment context

    Example:
        awareness = await get_environment_awareness()

        # Check if system is stressed
        if awareness["pain_level"] > 0.5:
            await emit_sensation("caution", 0.6, "System under stress")

        # Check emotional state
        if awareness["affect"]["confidence"] < 0.4:
            # Ask for clarification before proceeding
            ...
    """
    # Gather all awareness data in parallel
    affect_task = get_affect_state()
    pain_task = get_pain_dashboard()
    temporal_task = get_temporal_context()
    sensations_task = get_active_sensations(limit=5)

    affect, pain, temporal, sensations = await asyncio.gather(
        affect_task, pain_task, temporal_task, sensations_task,
        return_exceptions=True
    )

    # Handle any exceptions
    if isinstance(affect, Exception):
        affect = _get_default_affect()
    if isinstance(pain, Exception):
        pain = {"pain_level": "unknown"}
    if isinstance(temporal, Exception):
        temporal = _get_fallback_temporal()
    if isinstance(sensations, Exception):
        sensations = []

    # Map pain level string to float
    pain_level_map = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
    pain_level = pain_level_map.get(pain.get("pain_level", "none"), 0.0)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "affect": affect,
        "pain_level": pain_level,
        "pain_details": pain,
        "temporal": temporal,
        "active_sensations": sensations,
        "summary": _generate_awareness_summary(affect, pain_level, temporal)
    }


def _generate_awareness_summary(affect: Dict, pain_level: float, temporal: Dict) -> str:
    """Generate a human-readable awareness summary."""
    parts = []

    # Time context
    time_of_day = temporal.get("time_of_day", "")
    if time_of_day:
        parts.append(f"It's {time_of_day}")

    # Mood from affect
    valence = affect.get("valence", 0.0)
    arousal = affect.get("arousal", 0.5)

    if valence > 0.3:
        mood = "positive" if arousal < 0.6 else "energized"
    elif valence < -0.3:
        mood = "cautious" if arousal < 0.6 else "tense"
    else:
        mood = "neutral" if arousal < 0.6 else "alert"

    parts.append(f"feeling {mood}")

    # Pain awareness
    if pain_level > 0.5:
        parts.append("system under stress")
    elif pain_level > 0.3:
        parts.append("some system tension")

    # Confidence
    confidence = affect.get("confidence", 0.7)
    if confidence < 0.4:
        parts.append("proceeding with caution")
    elif confidence > 0.8:
        parts.append("confident in approach")

    return ", ".join(parts) + "."


# ===============================================================================
# TOOL EXPORTS - For Agent Integration
# ===============================================================================

# List of all awareness tools for agent registration
awareness_tools = [
    # Sensation tools
    emit_sensation,
    get_affect_state,
    get_active_sensations,

    # Pulse event tools
    subscribe_to_pulse,
    get_pain_dashboard,
    emit_pulse_event,

    # Temporal tools
    get_temporal_context,
    track_operation_duration,
    end_operation_tracking,

    # Unified awareness
    get_environment_awareness,
]

__all__ = [
    # Tool list
    "awareness_tools",

    # Sensation
    "emit_sensation",
    "get_affect_state",
    "get_active_sensations",
    "VALID_SENSATIONS",

    # Pulse
    "subscribe_to_pulse",
    "get_pain_dashboard",
    "emit_pulse_event",

    # Temporal
    "get_temporal_context",
    "track_operation_duration",
    "end_operation_tracking",

    # Unified
    "get_environment_awareness",
]

