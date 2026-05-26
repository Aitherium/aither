"""
Aither ADK - Tools Module
=========================

Tool loading, MCP integration, and shared tool implementations.

Import directly from submodules:
    >>> from aither_adk.tools.tool_loader import aither_tools, remember, recall
    >>> from aither_adk.tools.mcp_client_tools import mcp_server_tools
    >>> from aither_adk.tools.awareness_tools import emit_sensation, get_affect_state
    >>> from aither_adk.tools.reinforcement_tools import record_interaction_outcome

Key Tool Categories:
-------------------
- Memory: remember, recall, add_to_working_memory, get_current_context
- Vision: analyze_image_content, generate_image, refine_image
- Personas: list_personas, get_persona_details, update_persona
- Infrastructure: run_script, get_service_status
- RBAC: rbac_list_users, rbac_check_permission
- Awareness: emit_sensation, get_affect_state, subscribe_to_pulse, get_temporal_context
- Reinforcement: record_interaction_outcome, submit_preference_pair, capture_reasoning_trace
"""

# Re-export awareness tools for convenience
try:
    from aither_adk.tools.awareness_tools import (
        awareness_tools,
        emit_sensation,
        get_affect_state,
        get_active_sensations,
        subscribe_to_pulse,
        get_pain_dashboard,
        emit_pulse_event,
        get_temporal_context,
        track_operation_duration,
        end_operation_tracking,
        get_environment_awareness,
        VALID_SENSATIONS,
    )
    _AWARENESS_AVAILABLE = True
except ImportError:
    _AWARENESS_AVAILABLE = False
    awareness_tools = []

# Re-export reinforcement learning tools
try:
    from aither_adk.tools.reinforcement_tools import (
        reinforcement_tools,
        record_interaction_outcome,
        submit_preference_pair,
        capture_reasoning_trace,
        request_quality_judgement,
        get_training_metrics,
        trigger_training_export,
        report_model_improvement,
        InteractionCapture,
        Outcome,
        DataCategory,
    )
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False
    reinforcement_tools = []

__all__ = [
    # Awareness tools
    "awareness_tools",
    "emit_sensation",
    "get_affect_state",
    "get_active_sensations",
    "subscribe_to_pulse",
    "get_pain_dashboard",
    "emit_pulse_event",
    "get_temporal_context",
    "track_operation_duration",
    "end_operation_tracking",
    "get_environment_awareness",
    "VALID_SENSATIONS",
    # Reinforcement learning tools
    "reinforcement_tools",
    "record_interaction_outcome",
    "submit_preference_pair",
    "capture_reasoning_trace",
    "request_quality_judgement",
    "get_training_metrics",
    "trigger_training_export",
    "report_model_improvement",
    "InteractionCapture",
    "Outcome",
    "DataCategory",
]
