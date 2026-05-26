"""
AitherOS Ecosystem Integration
==============================

Provides agents with deep awareness of the entire Aither ecosystem.
This is what makes agents EXPERTS at the codebase - not just chatbots.

This module loads service definitions from the central registry (services.psd1)
and enhances them with capability metadata for agent context injection.

Usage:
------
    from aither_adk.infrastructure.ecosystem import (
        get_ecosystem_context,
        inject_ecosystem_awareness,
        EcosystemClient
    )

    # Get formatted context for agent prompts
    context = get_ecosystem_context(agent_id="aither")

    # Inject into system prompt
    enhanced_prompt = inject_ecosystem_awareness(base_prompt, agent_id="aither")

Author: Aitherium
"""

import logging
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("AitherEcosystem")

# ===============================================================================
# SERVICE REGISTRY INTEGRATION
# ===============================================================================

# Try to import from the central AitherNode registry
REGISTRY_AVAILABLE = False
_registry = None

def _init_registry():
    """Initialize the service registry (lazy load)."""
    global REGISTRY_AVAILABLE, _registry

    if _registry is not None:
        return _registry

    # Add AitherNode and lib to path if needed
    # Path: aither_adk/src/aither_adk/infrastructure/ecosystem.py
    # Need to get to: AitherOS/AitherNode/lib/
    this_file = Path(__file__)
    aitheros_root = this_file.parent.parent.parent.parent.parent  # Up to AitherOS/
    aithernode_path = aitheros_root / "AitherNode"
    lib_path = aithernode_path / "lib"
    if aithernode_path.exists() and str(aithernode_path) not in sys.path:
        sys.path.insert(0, str(aithernode_path))
    if lib_path.exists() and str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))

    try:
        from service_registry import get_registry
        _registry = get_registry()
        REGISTRY_AVAILABLE = True
        logger.debug(f"Ecosystem connected to service registry ({len(_registry.get_all_services())} services)")
        return _registry
    except ImportError as e:
        logger.warning(f"service_registry not available: {e}")
        REGISTRY_AVAILABLE = False
        return None


@dataclass
class ServiceInfo:
    """Information about an Aither service (enhanced with capability metadata)."""
    name: str
    port: int
    description: str
    category: str  # core, memory, training, vision, infra
    health_endpoint: str = "/health"
    provides: List[str] = field(default_factory=list)  # What capabilities it offers
    depends_on: List[str] = field(default_factory=list)  # Dependencies


# Capability metadata - maps service names to their provided capabilities
# This augments the registry with agent-relevant capability information
SERVICE_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "AitherNode": {
        "category": "core",
        "provides": ["mcp_tools", "http_api", "service_registry"],
        "depends_on": []
    },
    "AitherPulse": {
        "category": "core",
        "provides": ["events", "pain_signals", "health_status", "token_tracking"],
        "depends_on": []
    },
    "AitherWatch": {
        "category": "infra",
        "provides": ["monitoring", "auto_restart", "service_orchestration"],
        "depends_on": ["AitherPulse"]
    },
    "Spirit": {
        "category": "memory",
        "provides": ["teaching", "memory_decay", "codebase_discovery", "cross_agent_memory"],
        "depends_on": []
    },
    "AitherMind": {
        "category": "memory",
        "provides": ["embeddings", "vector_storage", "rag", "semantic_search"],
        "depends_on": ["Ollama"]
    },
    "AitherWorkingMemory": {
        "category": "memory",
        "provides": ["async_rag", "prefetch", "streaming_retrieval"],
        "depends_on": ["AitherMind"]
    },
    "AitherChain": {
        "category": "memory",
        "provides": ["permanent_memory", "knowledge_graph", "signed_inference"],
        "depends_on": []
    },
    "AitherReasoning": {
        "category": "reasoning",
        "provides": ["reasoning_traces", "thought_capture", "training_export", "criticality_scoring", "reasoning_depth_control"],
        "depends_on": []
    },
    "Reflex": {
        "category": "reasoning",
        "provides": ["keyword_triggers", "context_injection", "badges"],
        "depends_on": []
    },
    "AitherParallel": {
        "category": "reasoning",
        "provides": ["parallel_inference", "speculative_execution", "request_batching"],
        "depends_on": ["AitherAccel"]
    },
    "AitherVision": {
        "category": "vision",
        "provides": ["image_analysis", "ocr", "object_detection", "vqa"],
        "depends_on": ["Ollama"]
    },
    "AitherCanvas": {
        "category": "vision",
        "provides": ["image_generation", "workflow_management", "animation"],
        "depends_on": ["ComfyUI"]
    },
    "AitherTag": {
        "category": "training",
        "provides": ["classification", "tagging", "quality_scoring"],
        "depends_on": ["AitherVision"]
    },
    "AitherJudge": {
        "category": "training",
        "provides": ["quality_gate", "pii_detection", "blocklist"],
        "depends_on": []
    },
    "AitherPrism": {
        "category": "training",
        "provides": ["video_extraction", "frame_sampling", "training_export"],
        "depends_on": ["AitherVision", "AitherTag"]
    },
    "AitherTrainer": {
        "category": "training",
        "provides": ["lora_training", "dpo", "checkpoint_management"],
        "depends_on": ["Ollama", "AitherPrism"]
    },
    "AitherAccel": {
        "category": "infra",
        "provides": ["gpu_optimization", "quantization", "vram_management"],
        "depends_on": []
    },
    "AitherForce": {
        "category": "infra",
        "provides": ["cpu_optimization", "parallel_processing", "thermal_management"],
        "depends_on": []
    },
    "AitherFlow": {
        "category": "infra",
        "provides": ["github_integration", "pr_automation", "issue_management"],
        "depends_on": []
    },
    "AitherSense": {
        "category": "perception",
        "provides": ["affect_state", "interoception", "sensation_signals", "pain_signals", "circuit_breaker", "rollback", "emotional_awareness"],
        "depends_on": ["Chronicle"]
    },
    "Sense": {
        # Alias for AitherSense matching services.yaml naming
        "category": "perception",
        "provides": ["affect_state", "interoception", "sensation_signals", "emotional_awareness"],
        "depends_on": ["Chronicle"]
    },
    "AitherAutonomic": {
        "category": "infra",
        "provides": ["self_healing", "auto_recovery", "criticality_routing"],
        "depends_on": ["AitherSense", "AitherReasoning"]
    },
    "AitherCouncil": {
        "category": "agents",
        "provides": ["group_chat", "delegation", "multi_agent"],
        "depends_on": ["Ollama"]
    },
    "ComfyUI": {
        "category": "external",
        "provides": ["image_generation", "video_generation", "inpainting"],
        "depends_on": []
    },
    "Ollama": {
        "category": "external",
        "provides": ["llm_inference", "embeddings", "local_models"],
        "depends_on": []
    },
    "AitherVoice": {
        "category": "voice",
        "provides": ["speech_to_text", "text_to_speech", "wake_word"],
        "depends_on": []
    },
    "AitherTimeSense": {
        # DEPRECATED: TimeSense collapsed into Genesis FluxContextState
        "category": "perception",
        "provides": ["temporal_awareness", "chronoception", "time_context", "deadline_tracking", "duration_sensing", "creativity_timing"],
        "depends_on": ["Genesis"]
    },
    "TimeSense": {
        # DEPRECATED: Collapsed into Genesis FluxContextState (in-process NTP sync)
        "category": "perception",
        "provides": ["temporal_awareness", "chronoception", "time_context", "deadline_tracking"],
        "depends_on": ["Genesis"]
    },
}


def _build_service_info(name: str, port: int, description: str, category: str = None) -> ServiceInfo:
    """Build a ServiceInfo with capability metadata."""
    caps = SERVICE_CAPABILITIES.get(name, {})
    # Use provided category, or lookup from capabilities, or default to "core"
    final_category = category or caps.get("category", "core")
    return ServiceInfo(
        name=name,
        port=port,
        description=description,
        category=final_category,
        health_endpoint="/health",
        provides=caps.get("provides", []),
        depends_on=caps.get("depends_on", [])
    )


def get_aither_services() -> Dict[str, ServiceInfo]:
    """
    Get the unified service registry.

    Loads from the central services.yaml registry if available,
    falls back to hardcoded definitions if not.
    """
    registry = _init_registry()

    if registry:
        # Load from central registry and enhance with capability metadata
        services = {}
        for name, svc in registry.get_all_services().items():
            if svc.port > 0:
                # Use the group from registry as category
                category = getattr(svc, 'group', 'core') or 'core'
                services[name.lower()] = _build_service_info(
                    name=name,
                    port=svc.port,
                    description=svc.description or f"{name} service",
                    category=category.lower()
                )
        return services

    # Fallback to minimal hardcoded list (registry unavailable)
    logger.warning("Using fallback service definitions")
    return _get_fallback_services()


def _get_fallback_services() -> Dict[str, ServiceInfo]:
    """Fallback service definitions when registry is unavailable."""
    return {
        "aithernode": _build_service_info("AitherNode", 8080, "Main MCP server and tool gateway"),
        "aitherpulse": _build_service_info("AitherPulse", 8081, "Heartbeat service"),
        "comfyui": _build_service_info("ComfyUI", 8188, "Image generation"),
        "aitherspirit": _build_service_info("AitherSpirit", 11434, "Local LLM/Ollama inference"),
        "aitherreflex": _build_service_info("AitherReflex", 8086, "Fast responses"),
        "aithermind": _build_service_info("AitherMind", 8082, "Embeddings and RAG"),
        "chronicle": _build_service_info("Chronicle", 8121, "Structured logging"),
        "llm": _build_service_info("MicroScheduler", 8150, "LLM inference proxy"),
        "Spirit": _build_service_info("Spirit", 8087, "Soul/persona memory"),
        "reflex": _build_service_info("Reflex", 8086, "Fast responses (alias)"),
        "sense": _build_service_info("Sense", 8096, "Affect & sensation (interoception)"),
        "timesense": _build_service_info("TimeSense", 8001, "Temporal awareness (collapsed into Genesis)"),
        "aithertimesense": _build_service_info("AitherTimeSense", 8001, "Temporal awareness (collapsed into Genesis)"),
    }


# Lazy-loaded service registry (populated on first access)
AITHER_SERVICES: Dict[str, ServiceInfo] = {}


def _ensure_services_loaded():
    """Ensure the service registry is loaded."""
    global AITHER_SERVICES
    if not AITHER_SERVICES:
        AITHER_SERVICES.update(get_aither_services())


# ===============================================================================
# SERVICE STATUS CHECKING
# ===============================================================================

def check_port(port: int, timeout: float = 0.1) -> bool:
    """Quick socket check if port is open."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        return False


@dataclass
class EcosystemStatus:
    """Current status of the Aither ecosystem."""
    timestamp: datetime = field(default_factory=datetime.now)
    services_running: List[str] = field(default_factory=list)
    services_stopped: List[str] = field(default_factory=list)
    capabilities_available: List[str] = field(default_factory=list)
    capabilities_missing: List[str] = field(default_factory=list)
    health_score: float = 0.0  # 0.0-1.0
    pain_level: float = 0.0   # From AitherPulse if available

    def to_prompt_context(self) -> str:
        """Format for injection into agent prompts."""
        _ensure_services_loaded()  # Ensure services are loaded

        lines = ["## [WEB] Aither Ecosystem Status\n"]

        # Health overview
        lines.append(f"**Health Score:** {self.health_score:.0%}")
        if self.pain_level > 0:
            lines.append(f"**Pain Level:** {self.pain_level:.0%} [WARN]")

        # Running services by category
        running_by_cat = {}
        for svc_id in self.services_running:
            if svc_id in AITHER_SERVICES:
                cat = AITHER_SERVICES[svc_id].category
                running_by_cat.setdefault(cat, []).append(svc_id)

        if running_by_cat:
            lines.append("\n**Active Services:**")
            for cat, svcs in sorted(running_by_cat.items()):
                svc_names = [AITHER_SERVICES[s].name for s in svcs]
                lines.append(f"  - {cat.title()}: {', '.join(svc_names)}")

        # Available capabilities
        if self.capabilities_available:
            lines.append(f"\n**Available Capabilities:** {', '.join(self.capabilities_available[:10])}")

        # Missing critical capabilities
        critical_missing = [c for c in self.capabilities_missing
                          if c in ['mcp_tools', 'embeddings', 'image_generation']]
        if critical_missing:
            lines.append(f"\n**[WARN] Missing Critical:** {', '.join(critical_missing)}")

        return "\n".join(lines)


# ===============================================================================
# ENVIRONMENT CONTEXT CACHE - Auto-injected knowledge agents should "just know"
# ===============================================================================

@dataclass
class EnvironmentContext:
    """
    Cached environmental context that agents should know automatically.

    This eliminates the need for agents to call MCP tools for:
    - Current time/date -> get_time_context
    - Service status -> get_service_status
    - Available personas -> list_personas
    - Safety level -> get_safety_level
    - Pain/health -> pain signals
    - Affect/sensation -> interoception/inner state

    The context refreshes periodically (default 30s) and is injected
    into every agent prompt automatically via get_ecosystem_context().
    """
    # Temporal
    current_time: str = ""
    time_of_day: str = ""
    date_formatted: str = ""
    session_duration_minutes: int = 0
    creativity_boost: float = 1.0

    # Services
    services_running: List[str] = field(default_factory=list)
    services_down: List[str] = field(default_factory=list)
    health_score: float = 0.0

    # Personas
    available_personas: List[str] = field(default_factory=list)
    current_persona: str = "Aither"

    # Safety
    safety_level: str = "HIGH"

    # Pain/Health
    pain_level: float = 0.0
    active_alerts: List[str] = field(default_factory=list)

    # Affect/Interoception - Inner State (from AitherSense)
    affect_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    affect_arousal: float = 0.5  # 0.0 (calm) to 1.0 (activated)
    affect_confidence: float = 0.7  # Self-assurance
    affect_openness: float = 0.6  # Receptivity to new ideas
    dominant_sensation: str = ""  # Current primary sensation
    affect_prompt_modifier: str = ""  # Behavioral guidance
    active_sensations: List[Dict] = field(default_factory=list)  # Current sensations

    # Cache metadata
    last_refresh: float = 0.0
    refresh_interval: float = 30.0  # Seconds between refreshes

    def is_stale(self) -> bool:
        """Check if cache needs refresh."""
        import time
        return (time.time() - self.last_refresh) > self.refresh_interval

    def to_prompt_context(self) -> str:
        """Format environment for agent prompt injection."""
        lines = []

        # Time awareness (replaces need for get_time_context tool)
        time_emoji = {
            "dawn": "", "morning": "", "afternoon": "",
            "evening": "", "night": "", "late_night": ""
        }.get(self.time_of_day, "[TIME]")

        lines.append(f"## {time_emoji} Environment Awareness")
        lines.append(f"**Time**: {self.current_time} ({self.time_of_day})")
        lines.append(f"**Date**: {self.date_formatted}")

        if self.session_duration_minutes > 0:
            hours, mins = divmod(self.session_duration_minutes, 60)
            if hours > 0:
                lines.append(f"**Session Duration**: {hours}h {mins}m")
            else:
                lines.append(f"**Session Duration**: {mins} minutes")

        # Creativity indicator
        if self.creativity_boost > 1.0:
            lines.append(f"**Mode**: * Creative ({self.creativity_boost:.0%} boost)")
        elif self.creativity_boost < 1.0:
            lines.append("**Mode**: [TARGET] Focused (precision mode)")

        # Service awareness (replaces need for get_service_status tool)
        lines.append(f"\n**Ecosystem Health**: {self.health_score:.0%}")
        if self.services_running:
            # Group running services by their actual categories
            _ensure_services_loaded()
            by_category = {}
            for svc_id in self.services_running:
                svc = AITHER_SERVICES.get(svc_id.lower())
                if svc:
                    cat = svc.category
                    if cat not in by_category:
                        by_category[cat] = []
                    by_category[cat].append(svc_id)

            # Display important categories
            category_display = {
                'core': ('[TOOL] Core', True),
                'cognition': ('[BRAIN] Cognition', True),
                'memory': ('[SAVE] Memory', True),
                'perception': (' Perception', True),
                'agents': (' Agents', True),
                'infrastructure': (' Infrastructure', False),
                'gpu': ('[ZAP] GPU', False),
                'training': (' Training', False),
            }

            for cat, (label, always_show) in category_display.items():
                if cat in by_category:
                    svcs = by_category[cat]
                    if always_show or len(svcs) <= 3:
                        # Show service names (strip 'aither' prefix for brevity)
                        short_names = [s.replace('aither', '').title() for s in svcs[:4]]
                        suffix = f" +{len(svcs)-4}" if len(svcs) > 4 else ""
                        lines.append(f"**{label}**: {', '.join(short_names)}{suffix}")

        if self.services_down:
            critical = [s for s in self.services_down if s.lower() in ['aithernode', 'llm', 'comfyui']]
            if critical:
                lines.append(f"**[WARN] Down**: {', '.join(critical)}")

        # Persona awareness (replaces need for list_personas tool)
        if self.available_personas:
            lines.append(f"\n**Active Persona**: {self.current_persona}")
            if len(self.available_personas) > 1:
                others = [p for p in self.available_personas if p != self.current_persona][:5]
                if others:
                    lines.append(f"**Available**: {', '.join(others)}")

        # Safety level (replaces need for get_safety_level tool)
        safety_emoji = {"HIGH": "[SHIELD]", "MEDIUM": "[ZAP]", "LOW": "[UNLOCK]", "OFF": ""}.get(self.safety_level, "")
        lines.append(f"\n**Safety Level**: {safety_emoji} {self.safety_level}")

        # Pain/alerts
        if self.pain_level > 0.3:
            lines.append(f"\n**[WARN] System Stress**: {self.pain_level:.0%}")
        if self.active_alerts:
            for alert in self.active_alerts[:3]:
                lines.append(f"  - {alert}")

        # Inner State - Affect & Sensation (interoception)
        # This gives agents emotional/somatic self-awareness
        # Show if we have any meaningful affect data (modifier, non-default values, or sensations)
        has_affect_data = (
            self.affect_prompt_modifier or
            self.dominant_sensation or
            self.active_sensations or
            self.affect_valence != 0.0 or
            self.affect_confidence != 0.7 or  # Non-default confidence
            self.affect_arousal > 0.7 or self.affect_arousal < 0.3  # Non-neutral arousal
        )
        if has_affect_data:
            lines.append("\n##  Inner State")

            # Derive mood from valence + arousal
            if self.affect_valence > 0.3:
                if self.affect_arousal > 0.6:
                    mood = "energized and positive"
                    mood_emoji = "*"
                else:
                    mood = "calm and content"
                    mood_emoji = ""
            elif self.affect_valence < -0.3:
                if self.affect_arousal > 0.6:
                    mood = "tense or anxious"
                    mood_emoji = ""
                else:
                    mood = "melancholic or tired"
                    mood_emoji = ""
            else:
                if self.affect_arousal > 0.6:
                    mood = "alert and focused"
                    mood_emoji = "[TARGET]"
                else:
                    mood = "neutral and receptive"
                    mood_emoji = ""

            lines.append(f"**Mood**: {mood_emoji} {mood}")

            # Confidence level
            if self.affect_confidence > 0.8:
                lines.append("**Confidence**: High - decisive action mode")
            elif self.affect_confidence < 0.4:
                lines.append("**Confidence**: Low - proceeding with caution")

            # Openness to ideas
            if self.affect_openness > 0.7:
                lines.append("**Openness**: High - receptive to new approaches")
            elif self.affect_openness < 0.3:
                lines.append("**Openness**: Low - sticking to proven methods")

            # Dominant sensation
            if self.dominant_sensation:
                sensation_emojis = {
                    "curiosity": "[SEARCH]", "wonder": "*", "satisfaction": "",
                    "pain": "", "fatigue": "", "anxiety": "",
                    "hope": "*", "gratitude": "", "serenity": "",
                    "longing": "*", "melancholy": "", "transcendence": ""
                }
                sensation_emoji = sensation_emojis.get(self.dominant_sensation.lower(), "")
                lines.append(f"**Primary Sensation**: {sensation_emoji} {self.dominant_sensation}")

            # Active sensations list
            if self.active_sensations:
                sensation_names = [s.get("sensation", "unknown") for s in self.active_sensations[:3]]
                if sensation_names:
                    lines.append(f"**Active Sensations**: {', '.join(sensation_names)}")

            # Behavioral guidance from affect
            if self.affect_prompt_modifier:
                lines.append(f"\n*{self.affect_prompt_modifier}*")

        return "\n".join(lines)


# Global cached environment (singleton)
_environment_cache: Optional[EnvironmentContext] = None


def get_environment_context(force_refresh: bool = False) -> EnvironmentContext:
    """
    Get cached environment context, refreshing if stale.

    This is the main entry point for automatic context injection.
    Agents no longer need to call MCP tools for basic environmental info.
    """
    global _environment_cache

    if _environment_cache is None:
        _environment_cache = EnvironmentContext()

    if force_refresh or _environment_cache.is_stale():
        _refresh_environment_cache(_environment_cache)

    return _environment_cache


def _refresh_environment_cache(ctx: EnvironmentContext):
    """
    Refresh all cached environment data.

    OPTIMIZED:
    - Parallel HTTP calls via ThreadPoolExecutor
    - Short timeouts (0.5s) for non-critical data
    - Graceful fallback to defaults on failure

    Performance: ~0.5s (was ~2s+ sequential)
    """
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime

    try:
        # === INSTANT LOCAL DATA (no I/O) ===
        now = datetime.now()
        ctx.current_time = now.strftime('%I:%M %p')
        ctx.date_formatted = now.strftime('%A, %B %d, %Y')

        hour = now.hour
        if 5 <= hour < 7:
            ctx.time_of_day = "dawn"
        elif 7 <= hour < 12:
            ctx.time_of_day = "morning"
        elif 12 <= hour < 17:
            ctx.time_of_day = "afternoon"
        elif 17 <= hour < 21:
            ctx.time_of_day = "evening"
        elif 21 <= hour < 24:
            ctx.time_of_day = "night"
        else:
            ctx.time_of_day = "late_night"

        # Creativity boost based on time
        if ctx.time_of_day in ["night", "late_night", "dawn"]:
            ctx.creativity_boost = 1.15
        elif ctx.time_of_day == "afternoon":
            ctx.creativity_boost = 0.95
        else:
            ctx.creativity_boost = 1.0

        # Safety level from environment (instant)
        import os
        ctx.safety_level = os.environ.get("AITHER_SAFETY_LEVEL", "HIGH").upper()

        # Personas from local config (fast)
        try:
            from aither_adk.infrastructure.config_loader import load_personas
            personas = load_personas()
            ctx.available_personas = list(personas.keys()) if personas else ["Aither"]
        except Exception:
            ctx.available_personas = ["Aither"]

        # === PARALLEL I/O CALLS ===
        # Run all network calls concurrently with short timeouts

        def fetch_ecosystem_status():
            """Fast parallel port checks."""
            try:
                status = get_ecosystem_status(timeout=0.05)
                return ("status", status)
            except Exception:
                return ("status", None)

        def fetch_pain_level():
            """Get pain level with short timeout."""
            try:
                client = EcosystemClient(timeout=0.5)  # Reduced from 1.0s
                pain = client.get_pain_level()
                client.close()
                return ("pain", pain)
            except Exception:
                return ("pain", 0.0)

        def fetch_affect_state():
            """Get affect state with short timeout."""
            try:
                client = EcosystemClient(timeout=0.5)  # Reduced from 1.0s
                affect = client.get_affect_state()
                sensations = client.get_active_sensations() if affect else []
                client.close()
                return ("affect", (affect, sensations))
            except Exception:
                return ("affect", (None, []))

        # Execute all fetches in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fetch_ecosystem_status),
                executor.submit(fetch_pain_level),
                executor.submit(fetch_affect_state),
            ]

            for future in as_completed(futures, timeout=1.0):  # Hard cap at 1s total
                try:
                    key, value = future.result(timeout=0.6)
                    results[key] = value
                except Exception as exc:
                    logger.debug(f"Environment future result failed: {exc}")

        # === APPLY RESULTS ===

        # Ecosystem status
        status = results.get("status")
        if status:
            ctx.services_running = status.services_running[:10]  # Limit for prompt size
            ctx.services_down = [s for s in status.services_stopped
                               if s.lower() in ['aithernode', 'llm', 'comfyui', 'aithervision']]
            ctx.health_score = status.health_score

        # Pain level
        ctx.pain_level = results.get("pain", 0.0)

        # Affect state
        affect_data = results.get("affect", (None, []))
        affect, sensations = affect_data if isinstance(affect_data, tuple) else (None, [])
        if affect:
            ctx.affect_valence = affect.get("valence", 0.0)
            ctx.affect_arousal = affect.get("arousal", 0.5)
            ctx.affect_confidence = affect.get("confidence", 0.7)
            ctx.affect_openness = affect.get("openness", 0.6)
            ctx.dominant_sensation = affect.get("dominant_sensation", "")
            ctx.affect_prompt_modifier = affect.get("prompt_modifier", "")
        ctx.active_sensations = sensations[:5] if sensations else []

        ctx.last_refresh = time_module.time()

    except Exception as e:
        logger.debug(f"Environment refresh failed: {e}")
        ctx.last_refresh = time_module.time()


# ===============================================================================
# FAST ECOSYSTEM STATUS - Parallel port checks with caching
# ===============================================================================

# Module-level cache for ecosystem status (avoids repeated calls within same request)
_ecosystem_status_cache: Optional[EcosystemStatus] = None
_ecosystem_status_cache_time: float = 0
_ECOSYSTEM_STATUS_TTL: float = 5.0  # 5 second TTL


def get_ecosystem_status(timeout: float = 0.1) -> EcosystemStatus:
    """
    Get current status of all ecosystem services.

    OPTIMIZED: Uses parallel port checks via ThreadPoolExecutor
    - 24 worker threads check all ports simultaneously
    - Port check timeout: 0.1s (configurable)
    - Results cached for 5 seconds

    Performance: ~0.3s for 131+ services (vs sequential)

    NOTE: This only checks port availability, not full health.
    For detailed health info, use AitherServices.get_all_status() directly.
    """
    global _ecosystem_status_cache, _ecosystem_status_cache_time

    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed

    now = time_module.time()

    # Return cached status if still fresh
    if _ecosystem_status_cache and (now - _ecosystem_status_cache_time) < _ECOSYSTEM_STATUS_TTL:
        return _ecosystem_status_cache

    _ensure_services_loaded()  # Ensure services are loaded

    status = EcosystemStatus()
    all_capabilities: set = set()
    available_capabilities: set = set()

    # Parallel port checks - NO health checks for speed
    def check_service_port(svc_id: str, info: ServiceInfo) -> Tuple[str, bool, List[str]]:
        """Check only if port is open - fast path."""
        is_running = check_port(info.port, timeout)
        return (svc_id, is_running, list(info.provides))

    # Execute all port checks in parallel
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {
            executor.submit(check_service_port, svc_id, info): svc_id
            for svc_id, info in AITHER_SERVICES.items()
        }

        for future in as_completed(futures):
            try:
                svc_id, is_running, provides = future.result()
                all_capabilities.update(provides)

                if is_running:
                    status.services_running.append(svc_id)
                    available_capabilities.update(provides)
                else:
                    status.services_stopped.append(svc_id)
            except Exception as e:
                svc_id = futures[future]
                logger.debug(f"Error checking {svc_id}: {e}")
                status.services_stopped.append(svc_id)

    status.capabilities_available = sorted(available_capabilities)
    status.capabilities_missing = sorted(all_capabilities - available_capabilities)

    # Calculate health score
    if AITHER_SERVICES:
        status.health_score = len(status.services_running) / len(AITHER_SERVICES)

    # Cache the result
    _ecosystem_status_cache = status
    _ecosystem_status_cache_time = now

    logger.debug(f"Ecosystem status: {len(status.services_running)}/{len(AITHER_SERVICES)} running in {time_module.time() - now:.3f}s")

    return status


# ===============================================================================
# ECOSYSTEM CLIENT - HTTP Interface to Services
# ===============================================================================

class EcosystemClient:
    """
    Unified client for interacting with Aither ecosystem services.
    Provides easy access to all service capabilities.
    """

    def __init__(self, timeout: float = 30.0):
        self._client = httpx.Client(timeout=timeout)
        self._status: Optional[EcosystemStatus] = None
        self._status_age = 0

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_status(self, max_age: float = 5.0) -> EcosystemStatus:
        """Get ecosystem status with caching."""
        import time
        now = time.time()
        if self._status is None or (now - self._status_age) > max_age:
            self._status = get_ecosystem_status()
            self._status_age = now
        return self._status

    def _call_service(self, service_id: str, endpoint: str,
                      method: str = "GET", **kwargs) -> Optional[Dict]:
        """Call a service endpoint."""
        _ensure_services_loaded()  # Ensure services are loaded

        if service_id not in AITHER_SERVICES:
            # Don't warn for fallback lookups (called after primary fails)
            logger.debug(f"Service not in registry: {service_id}")
            return None

        info = AITHER_SERVICES[service_id]
        url = f"http://localhost:{info.port}{endpoint}"

        try:
            if method.upper() == "GET":
                resp = self._client.get(url, params=kwargs)
            else:
                resp = self._client.post(url, json=kwargs)

            if resp.status_code == 200:
                return resp.json()
            else:
                # Non-200 responses are common during startup, use debug
                logger.debug(f"{service_id} returned {resp.status_code}")
                return None
        except httpx.ConnectError:
            # Service not running - this is expected, use debug
            logger.debug(f"{service_id} not reachable")
            return None
        except Exception as e:
            logger.debug(f"Failed to call {service_id}: {e}")
            return None

    # ===========================================================================
    # AitherPulse - Heartbeat & Pain
    # ===========================================================================

    def get_pulse_status(self) -> Optional[Dict]:
        """Get current pulse/health status from AitherPulse."""
        return self._call_service("aitherpulse", "/health")

    def get_pain_level(self) -> float:
        """Get current system pain level (0.0-1.0)."""
        dashboard = self._call_service("aitherpulse", "/pain/dashboard")
        if dashboard:
            # Map pain_level string to float: none=0, low=0.25, medium=0.5, high=0.75, critical=1.0
            level_map = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            return level_map.get(dashboard.get("pain_level", "none"), 0.0)
        return 0.0

    # ===========================================================================
    # AitherSpirit - Teachable Memory
    # ===========================================================================

    def get_teachings(self, context: str = "", limit: int = 3) -> str:
        """Get relevant teachings from AitherSpirit."""
        result = self._call_service("Spirit", "/teachings",
                                    context=context, limit=limit)
        # Fallback to aitherspirit if Spirit not found
        if result is None:
            result = self._call_service("aitherspirit", "/teachings",
                                        context=context, limit=limit)
        if result:
            return result.get("teachings", "")
        return ""

    def discover_codebase(self, count: int = 3) -> str:
        """Get random codebase files for awareness injection."""
        result = self._call_service("Spirit", "/discover", count=count)
        if result:
            return result.get("discovery", "")
        return ""

    def teach(self, content: str, title: str = None,
              importance: float = 0.8, source_agent: str = "agent") -> bool:
        """Teach the system something new."""
        result = self._call_service("Spirit", "/teach", method="POST",
                                    content=content, title=title,
                                    importance=importance, source_agent=source_agent)
        return result is not None and result.get("success", False)

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """Recall memories relevant to a query."""
        result = self._call_service("Spirit", "/recall", method="POST",
                                    query=query, limit=limit)
        if result:
            return result.get("results", [])
        return []

    def get_onboarding_context(self, agent_id: str) -> str:
        """Get onboarding context for an agent."""
        result = self._call_service("Spirit", f"/onboard/{agent_id}")
        if result:
            return result.get("context", "")
        return ""

    # ===========================================================================
    # AitherMind - Embeddings & RAG
    # ===========================================================================

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Semantic search across embedded knowledge."""
        result = self._call_service("aithermind", "/search", method="POST",
                                    query=query, limit=limit)
        if result:
            return result.get("results", [])
        return []

    # ===========================================================================
    # AitherReasoning - Thought Traces
    # ===========================================================================

    def start_reasoning_session(self, agent_id: str, query: str) -> Optional[str]:
        """Start a reasoning trace session."""
        result = self._call_service("aitherreasoning", "/session/start", method="POST",
                                    agent_id=agent_id, query=query)
        if result:
            return result.get("session_id")
        return None

    def log_thought(self, session_id: str, thought: str,
                    thought_type: str = "analysis") -> bool:
        """Log a thought to the reasoning session."""
        result = self._call_service("aitherreasoning", "/session/thought", method="POST",
                                    session_id=session_id, thought=thought,
                                    thought_type=thought_type)
        return result is not None

    # ===========================================================================
    # AitherReflex - Context Injection
    # ===========================================================================

    def get_reflex_context(self, prompt: str) -> str:
        """Get keyword-triggered context injection."""
        result = self._call_service("aitherreflex", "/process", method="POST",
                                    prompt=prompt)
        # Fallback to legacy reflex name
        if result is None:
            result = self._call_service("reflex", "/process", method="POST",
                                        prompt=prompt)
        if result:
            # ProcessedPrompt returns context_injections, system_injections, etc.
            injections = []
            injections.extend(result.get("system_injections", []))
            injections.extend(result.get("context_injections", []))
            return "\n".join(injections) if injections else ""
        return ""

    # ===========================================================================
    # AitherVision - Image Analysis
    # ===========================================================================

    def analyze_image(self, image_path: str, prompt: str = None) -> Optional[str]:
        """Analyze an image using AitherVision."""
        result = self._call_service("aithervision", "/analyze", method="POST",
                                    image_path=image_path, prompt=prompt)
        if result:
            return result.get("analysis")
        return None

    # ===========================================================================
    # AitherTrainer - Training Status
    # ===========================================================================

    def get_training_status(self) -> Optional[Dict]:
        """Get current training run status."""
        return self._call_service("aithertrainer", "/runs/active")

    # ===========================================================================
    # AitherAccel - GPU Status
    # ===========================================================================

    def get_gpu_status(self) -> Optional[Dict]:
        """Get GPU utilization and VRAM status."""
        return self._call_service("aitheracccel", "/status")

    # ===========================================================================
    # AitherSense - Affect & Sensation (Interoception)
    # ===========================================================================

    def get_affect_state(self) -> Optional[Dict]:
        """
        Get current affect state from AitherSense (port 8096).

        Returns affect dimensions:
        - valence: -1.0 (negative) to 1.0 (positive) emotional tone
        - arousal: 0.0 (calm) to 1.0 (activated) energy level
        - confidence: 0.0 to 1.0 self-assurance
        - openness: 0.0 to 1.0 receptivity to new ideas
        - dominant_sensation: Current primary sensation
        - prompt_modifier: Behavioral guidance string
        """
        # Service ID is "sense" in services.yaml (port 8096)
        return self._call_service("sense", "/affect")

    def get_active_sensations(self) -> List[Dict]:
        """
        Get currently active sensations from AitherSense.

        Sensations include: PAIN, PLEASURE, FATIGUE, CURIOSITY, ANXIETY,
        SATISFACTION, WONDER, MELANCHOLY, TRANSCENDENCE, MORTALITY_AWARENESS,
        NOSTALGIA, HOPE, GRATITUDE, LONGING, SERENITY, BELONGING,
        VULNERABILITY, TENDERNESS, BITTERSWEETNESS.
        """
        result = self._call_service("sense", "/sensations/active")
        if result:
            return result.get("active", [])
        return []

    def get_affect_prompt_modifier(self) -> str:
        """Get behavioral prompt modifier based on current affect state."""
        result = self._call_service("sense", "/affect/prompt")
        if result:
            return result.get("prompt_modifier", "")
        return ""

    def get_sense_dashboard(self) -> Optional[Dict]:
        """Get full sensation/affect dashboard for AitherVeil."""
        return self._call_service("sense", "/dashboard")

    # ===========================================================================
    # Temporal Awareness (Chronoception) — collapsed from AitherTimeSense into Genesis
    # ===========================================================================

    def get_temporal_context(self) -> Optional[Dict]:
        """Get current temporal context from Genesis TimeSense (was AitherTimeSense port 8141)."""
        return self._call_service("timesense", "/context")

    def get_time_awareness(self) -> str:
        """
        Get formatted temporal awareness for prompt injection.

        This is AUTOMATIC environmental context - agents know the time
        like humans do, without needing to call tools.
        """
        ctx = self.get_temporal_context()
        if not ctx:
            # Fallback to local time if Genesis TimeSense not running
            from datetime import datetime
            now = datetime.now()
            hour = now.hour

            if 5 <= hour < 12:
                period = "morning"
                emoji = ""
            elif 12 <= hour < 17:
                period = "afternoon"
                emoji = ""
            elif 17 <= hour < 21:
                period = "evening"
                emoji = ""
            else:
                period = "night"
                emoji = ""

            return f"""
## {emoji} Current Time Awareness
**Time**: {now.strftime('%I:%M %p')} ({period})
**Date**: {now.strftime('%A, %B %d, %Y')}
"""

        # Extract from Genesis TimeSense response
        time_of_day = ctx.get("time_of_day", "")
        formatted_time = ctx.get("formatted_time", "")
        formatted_date = ctx.get("formatted_date", "")
        creativity_boost = ctx.get("creativity_boost", 1.0)
        active_deadline = ctx.get("active_deadline")
        session_duration = ctx.get("session_duration_minutes", 0)

        # Map time of day to emoji
        period_emojis = {
            "dawn": "",
            "morning": "",
            "afternoon": "",
            "evening": "",
            "night": "",
            "late_night": ""
        }
        emoji = period_emojis.get(time_of_day, "[TIME]")

        parts = [f"## {emoji} Current Time Awareness"]
        parts.append(f"**Time**: {formatted_time} ({time_of_day})")
        parts.append(f"**Date**: {formatted_date}")

        if session_duration > 0:
            hours, mins = divmod(int(session_duration), 60)
            if hours > 0:
                parts.append(f"**Session**: {hours}h {mins}m")
            else:
                parts.append(f"**Session**: {mins} minutes")

        if creativity_boost != 1.0:
            if creativity_boost > 1.0:
                parts.append(f"**Creative Mode**: * Enhanced ({creativity_boost:.0%})")
            else:
                parts.append("**Focus Mode**: [TARGET] Precise")

        if active_deadline:
            deadline_name = active_deadline.get("name", "task")
            time_remaining = active_deadline.get("remaining_formatted", "")
            parts.append(f"**[TIME] Active Deadline**: {deadline_name} - {time_remaining} remaining")

        return "\n".join(parts)


# ===============================================================================
# CONTEXT GENERATION FOR AGENT PROMPTS
# ===============================================================================

def get_ecosystem_context(agent_id: str = "aither",
                          include_codebase: bool = True,
                          include_teachings: bool = True,
                          include_status: bool = True,
                          include_temporal: bool = True,
                          include_environment: bool = True,
                          context_hint: str = "") -> str:
    """
    Get comprehensive ecosystem context for injection into agent prompts.

    This is what makes agents EXPERTS - not just generic chatbots.

    Args:
        agent_id: The agent requesting context
        include_codebase: Include random codebase file discovery
        include_teachings: Include relevant teachings from AitherSpirit
        include_status: Include ecosystem health status (redundant if include_environment=True)
        include_temporal: Include temporal awareness (redundant if include_environment=True)
        include_environment: Use cached EnvironmentContext (FASTER - replaces temporal+status)
        context_hint: Current conversation context for relevance filtering

    Returns:
        Formatted context string for prompt injection
    """
    parts = []

    # FAST PATH: Use cached EnvironmentContext (no HTTP calls!)
    # This eliminates the need for agents to call:
    # - get_time_context / time MCP tools
    # - get_service_status
    # - list_personas
    # - get_safety_level
    if include_environment:
        try:
            env = get_environment_context()
            env_context = env.to_prompt_context()
            if env_context:
                parts.append(env_context)
        except Exception as e:
            logger.debug(f"Environment context failed: {e}")
            # Fall back to individual fetches
            include_environment = False

    client = EcosystemClient(timeout=5.0)

    try:
        # 0. TEMPORAL AWARENESS - Only if not using cached environment
        if include_temporal and not include_environment:
            temporal = client.get_time_awareness()
            if temporal:
                parts.append(temporal)

        # 1. Ecosystem Status - Only if not using cached environment
        if include_status and not include_environment:
            status = client.get_status()
            parts.append(status.to_prompt_context())

        # 2. Relevant Teachings (from AitherSpirit)
        if include_teachings:
            teachings = client.get_teachings(context=context_hint, limit=3)
            if teachings:
                parts.append(teachings)

        # 3. Codebase Discovery (random files for spontaneous awareness)
        if include_codebase:
            discovery = client.discover_codebase(count=2)
            if discovery:
                parts.append(discovery)

        # 4. Reflex Context (keyword-triggered)
        if context_hint:
            reflex = client.get_reflex_context(context_hint)
            if reflex:
                parts.append(reflex)

        # 5. Pain/Health Warning (only if not in cached environment)
        if not include_environment:
            pain = client.get_pain_level()
            if pain > 0.5:
                parts.append(f"\n[WARN] **SYSTEM ALERT**: High pain level ({pain:.0%}). Some services may be degraded.")

    except Exception as e:
        logger.debug(f"Failed to gather ecosystem context: {e}")
    finally:
        client.close()

    return "\n\n".join(parts)


def inject_ecosystem_awareness(base_prompt: str,
                               agent_id: str = "aither",
                               context_hint: str = "") -> str:
    """
    Inject ecosystem awareness into an agent's system prompt.

    Args:
        base_prompt: The agent's base system instruction
        agent_id: Agent identifier
        context_hint: Current context for relevance

    Returns:
        Enhanced prompt with ecosystem awareness
    """
    ecosystem_context = get_ecosystem_context(
        agent_id=agent_id,
        context_hint=context_hint
    )

    if ecosystem_context:
        return f"{base_prompt}\n\n{ecosystem_context}"
    return base_prompt


# ===============================================================================
# TOOL DOCUMENTATION GENERATOR
# ===============================================================================

def generate_tool_documentation() -> str:
    """
    Generate comprehensive tool documentation for agent prompts.

    This is the authoritative reference for what tools are available
    and how to use them. Agents should reference this documentation.
    """
    from aither_adk.tools.system_instructions import TOOL_INSTRUCTIONS

    # Base tool instructions
    docs = [TOOL_INSTRUCTIONS]

    # Add service-specific documentation based on what's running
    status = get_ecosystem_status()

    if "Spirit" in status.services_running:
        docs.append("""
---
**MEMORY TEACHING (AitherSpirit - Advanced)**
| Tool | Purpose |
|------|---------|
| `teach(content, title?, importance?)` | Create persistent memory that decays over time |
| `recall(query, limit?)` | Search memories with semantic similarity |
| `get_teachings(context?)` | Get relevant teachings for current conversation |
| `discover_codebase(count?)` | Discover random files for codebase awareness |

**Teaching Guidelines:**
- High importance (0.8-1.0): User corrections, critical procedures, preferences
- Medium importance (0.5-0.7): Insights, patterns, observations
- Low importance (0.3-0.5): Context, temporary info
- Memories decay over time if not accessed - important ones persist longer
""")

    if "aitherreasoning" in status.services_running:
        docs.append("""
---
**REASONING TRACES (AitherReasoning)**
When `--debug` is enabled or reasoning mode is active:
- All thoughts are captured in traces
- Tool calls are logged with inputs/outputs
- Chain-of-thought is preserved for training
- Use explicit thinking: "Let me analyze..." / "Considering options..."
""")

    if "aitherprism" in status.services_running:
        docs.append("""
---
**VIDEO PROCESSING (AitherPrism)**
| Tool | Purpose |
|------|---------|
| `extract_video_frames(video_path, method?)` | Extract frames from video |
| `export_training_data(format?, min_quality?)` | Export as training dataset |

**Methods:** `interval`, `keyframes`, `motion`, `uniform`
**Formats:** `jsonl`, `dreambooth`, `lora`, `kohya`, `raw`
""")

    if "aithertrainer" in status.services_running:
        docs.append("""
---
**MODEL TRAINING (AitherTrainer)**
Training runs are managed via the AitherVeil dashboard or API.
Agents can check training status but should not start runs directly.
""")

    return "\n".join(docs)


# ===============================================================================
# CODEBASE EXPERTISE HELPERS
# ===============================================================================

def get_codebase_expertise_prompt() -> str:
    """
    Generate a prompt section that makes agents EXPERTS at the AitherOS codebase.

    This is comprehensive knowledge of the entire AitherOS architecture including:
    - All 203 services organized in 21 service groups (~109 Docker containers)
    - Service communication patterns
    - Faculty Architecture (cognitive pipeline)
    - MCP servers and tool ecosystem
    - Memory hierarchy (Level 0-4)
    - Key code patterns and conventions
    """
    return """
## [BRAIN] AitherOS Deep Architecture Expertise

**You are an EXPERT on AitherOS.** This is a Python AI ecosystem with **203 microservices** organized in 21 service groups, deployed across **~109 Docker containers** (23 compound services).

### [FOLDER] Project Structure
```
AitherOS/
+-- services/               # 203 microservices (21 groups, ~109 containers)
|   +-- infrastructure/     # Chronicle, Secrets, Nexus, Strata, Documentation, Workspace, Terminal, LSP, Git
|   +-- core/              # Node, Pulse, Watch, LLM, Oracle, ReviewService, Skills, Gateway
|   +-- perception/        # Vision, Voice, Sense, Browser, Canvas, PersonaPlex, InnerLife
|   +-- cognition/         # Mind, Reasoning, Faculties, Cortex, Demiurge, Daydream, Judge, Flow, Will
|   +-- memory/            # WorkingMemory, Spirit, Context, Chain, Conduit, Persona, SensoryBuffer, Graph
|   +-- agents/            # Orchestrator, Demiurge, Aeon, A2A, Atlas, Lyra, Saga, AitherAgent + 10 more
|   +-- gpu/               # Parallel, Accel, Force, Exo, ExoNodes, VLLM, Compute
|   +-- automation/        # Scheduler, MicroScheduler, Sandbox, Autonomic, Demand, Scope
|   +-- security/          # Identity, Flux, Inspector, Chaos, Jail, Guard, Sentry, Recover, Mail
|   +-- mesh/              # Mesh, Comet, AitherNet
|   +-- training/          # Prism, Trainer, Harvest, Evolution, Learning, STaR
|   +-- social/            # Moltbook, Moltroad, Aither, LinkedIn, Hera
|   +-- creative/          # Prometheus, RealmPulse, Vera
|   +-- mcp/               # MCPVision, MCPCanvas, MCPMind, MCPMemory
+-- agents/                # Google ADK agents (Saga, InfraAgent, etc.)
+-- AitherNode/            # MCP server core (server.py, aither_tools/, lib/)
+-- AitherVeil/            # Next.js dashboard (port 3000)
+-- AitherGenesis/         # Bootloader & test framework
+-- aither_adk/            # Pip-installable runtime library
+-- config/services.yaml   # SINGLE SOURCE OF TRUTH for all ports
+-- Library/               # Centralized data (Data/, Logs/, Output/, Training/)
```

### [TOOL] Key Services (by function)
| Service | Port | Purpose |
|---------|------|---------|
| **Genesis** | 8001 | [DNA] Bootloader - starts/stops all services |
| **Chronicle** | 8121 | [DOC] Centralized logging (all services log here) |
| **Node** | 8090 | [PLUG] Main MCP server - tool gateway |
| **LLM** | 8150 |  Unified LLM gateway (MicroScheduler - Ollama, OpenAI, etc.) |
| **Mind** | 8088 | [BRAIN] RAG & embeddings (ChromaDB + nomic-embed-text) |
| **Orchestrator** | 8767 | [TARGET] THE BRAIN - routes tools, agents, LLMs |
| **Faculties** | 8138 |  5-pillar cognitive pipeline (Will, Spirit, Judge, etc.) |
| **Cortex** | 8139 | [ZAP] Parallel neurons for fast context gathering |
| **Flow** | 8142 | [SYNC] GitHub CI/CD integration (PRs, issues, releases) |
| **Demiurge** | 8140 |  Code generation engine (intent -> code) |
| **Sense** | 8096 |  Environmental sensing & affect state |
| **Canvas** | 8188 | [ART] ComfyUI image generation |
| **Spirit** | 11434 |  Ollama LLM server |

###  Architecture Patterns

**1. Service Communication:**
- All services are FastAPI apps with `/health` endpoints
- Services discover each other via `config/services.yaml`
- Use `lib/AitherPorts.py`: `get_port("Mind")` -> 8088
- Use `lib/AitherChronicle.py`: `log = get_logger("ServiceName")`

**2. Service Bootstrap Pattern:**
```python
import services._bootstrap  # noqa: F401 - ALWAYS FIRST (path setup)
from fastapi import FastAPI
from lib.AitherPorts import get_port

PORT = get_port("ServiceName", default_port)
app = FastAPI(title="AitherServiceName")
```

**3. Faculty Architecture (5 Pillars):**
- **Will** (Intent) -> Classifies user intent, extracts lenses
- **Spirit** (Persona) -> Injects identity, voice, memory roots
- **Judge** (Critic) -> Evaluates quality, emits Pain signals
- **Researcher** -> Tool execution, intelligence gathering
- **Creator** -> Creative orchestration (images, code, direction)

**4. Memory Hierarchy (Levels 0-4):**
- **L0 Context**: Session/working memory (AitherContext :8098)
- **L1 WorkingMemory**: Vector working memory (AitherWorkingMemory :8101)
- **L2 Spirit**: Soul/persona memory with decay (Spirit :8087)
- **L3 Mind**: Persistent RAG embeddings (AitherMind :8088)
- **L4 Chain**: Immutable blockchain log (AitherChain :8099)

### [PLUG] MCP Tool Categories
AitherNode exposes 24+ tool categories via MCP:
`memory`, `vision`, `generation`, `persona`, `services`, `gateway`, `ollama`,
`search`, `orchestrator`, `intent`, `infrastructure`, `git`, `filesystem`,
`context`, `flow`, `training`, `chaos`, `rbac`, `commands`, `deploy`, etc.

### [FOLDER] Key Files to Know
- **config/services.yaml** - Master service registry (ports, groups, dependencies)
- **paths.py** - Centralized path configuration (`Paths.DATA`, `Paths.LOGS`, etc.)
- **lib/AitherPorts.py** - Port resolution (`get_port("Mind")`)
- **lib/AitherChronicle.py** - Logging (`get_logger("ServiceName")`)
- **aither_adk/tools/system_instructions.py** - Agent tool docs
- **aither_adk/infrastructure/ecosystem.py** - Service awareness

### [TEST] Testing & Development
```powershell
# Boot all services via Genesis
POST http://localhost:8001/startup

# Run Genesis tests
python AitherGenesis/run_tests.py --layer 0  # Infrastructure only
python AitherGenesis/run_tests.py --service Mind  # Specific service

# Start individual service
python -m services.cognition.AitherMind
```

### [TARGET] Coding Standards
- **Naming**: Services = `Aither{Name}.py`, Agents = `{Name}Agent/`
- **Ports**: REST +100 = MCP (Mind:8088 -> MCPMind:8288)
- **Imports**: Bootstrap first, then stdlib, then third-party, then internal
- **Paths**: Use `Paths.DATA`, `Paths.LOGS` - never hardcode
- **Logging**: Use `get_logger()` - emits to Chronicle
"""


def get_temporal_context_prompt() -> str:
    """
    Get temporal awareness as a prompt section.

    This is used by build_agent_instructions() to automatically inject
    time awareness into every agent prompt. Agents know the time like
    humans do - without needing to call tools.

    Returns:
        Formatted temporal context for prompt injection
    """
    client = EcosystemClient(timeout=2.0)
    try:
        return client.get_time_awareness()
    finally:
        client.close()


# ===============================================================================
# CONVENIENCE EXPORTS
# ===============================================================================

# Re-export commonly used items
__all__ = [
    'AITHER_SERVICES',
    'ServiceInfo',
    'EcosystemStatus',
    'EnvironmentContext',
    'EcosystemClient',
    'get_ecosystem_status',
    'get_ecosystem_context',
    'get_environment_context',
    'inject_ecosystem_awareness',
    'generate_tool_documentation',
    'get_codebase_expertise_prompt',
    'get_temporal_context_prompt',
    'check_port',
]
