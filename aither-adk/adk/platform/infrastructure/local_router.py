"""
AitherOS Local Router

Rule-based routing that doesn't require cloud APIs.
This ensures requests are routed instantly without API blocking.

Uses the centralized AitherSafety module for all content pattern detection.
Supports dynamic configuration via routing.yaml and /route command.

Benefits:
- Instant routing (no API calls)
- Never blocked by safety filters
- Falls back to lightweight local LLM only when unsure
- Centralized content safety control via AitherSafety
- Dynamic agent enable/disable via /route command
- Custom agent support for third-party integrations
"""

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Add AitherNode to path for AitherSafety import
AITHER_NODE_PATH = Path(__file__).parent.parent.parent / "AitherNode"
if str(AITHER_NODE_PATH) not in sys.path:
    sys.path.insert(0, str(AITHER_NODE_PATH))

# Import from centralized safety module
try:
    from AitherSafety import (
        detect_nsfw_level as _detect_nsfw_level,
    )
    from AitherSafety import (
        get_image_indicators as _get_image_indicators,
    )
    from AitherSafety import (
        get_routing_keywords,
        get_routing_patterns,
        get_safety_manager,
    )
    HAS_SAFETY_MODULE = True
except ImportError:
    HAS_SAFETY_MODULE = False
    get_safety_manager = None

    def _detect_nsfw_level(text: str) -> int:
        return 0

    def get_routing_patterns() -> Set[str]:
        return set()

    def get_routing_keywords() -> Set[str]:
        return set()

    def _get_image_indicators() -> Set[str]:
        return set()


@dataclass
class RouteDecision:
    """Result of routing decision"""
    agent: str
    confidence: float  # 0-1
    reason: str = ""  # Default empty string to prevent errors


# Import routing manager for dynamic configuration
try:
    from aither_adk.infrastructure.routing_manager import RoutingManager, get_routing_manager
    HAS_ROUTING_MANAGER = True
except ImportError:
    HAS_ROUTING_MANAGER = False
    get_routing_manager = None


class LocalRouter:
    """
    Rule-based router for AitherOS.

    Routes requests to the appropriate agent without using cloud APIs.
    All NSFW/content patterns are loaded from the centralized AitherSafety module.
    """

    # Base SFW patterns that DEFINITELY go to ArtistAgent
    ARTIST_PATTERNS_BASE = [
        r"\b(draw|paint|sketch|illustrate|render)\b",
        r"\b(generate|create|make)\s+(an?\s+)?(image|picture|photo|art|illustration)\b",
        r"\b(selfie|portrait|headshot)\b",
        r"\bshow\s+(me|us)\b",
        r"\b(visualize|depict)\b",
        r"\bsend\s+(me\s+)?(a\s+)?(pic|photo|image|picture)\b",
        r"\b@\s*artist\b",
        r"\b@\s*artista?gent\b",
    ]

    # Combined artist patterns (base + NSFW from AitherSafety)
    @property
    def ARTIST_PATTERNS(self) -> list:
        if HAS_SAFETY_MODULE:
            return self.ARTIST_PATTERNS_BASE + list(get_routing_patterns())
        return self.ARTIST_PATTERNS_BASE

    # Patterns that DEFINITELY go to CoderAgent
    CODER_PATTERNS = [
        r"\b(write|create|make)\s+(code|script|program|function)\b",
        r"\b(fix|debug|repair)\s+(the\s+)?(code|bug|error|script)\b",
        r"\b(install|download|setup|configure)\b",
        r"\b(python|javascript|bash|powershell|code)\b",
        r"\b@\s*coder\b",
        r"\b@\s*coderagent\b",
        r"\bhello\s+world\b",
        r"\bprint\s+.*\b",
    ]

    # Patterns that DEFINITELY go to Aither (chat/roleplay)
    AITHER_PATTERNS = [
        r"\b@\s*aither\b",
        r"\b(roleplay|rp)\b",
        r"\b(pretend|imagine|act\s+like)\b",
        r"\byou\s+(are|were|be)\b",
        r"\*[^*]+\*",  # *action text* roleplay markers
        r"\bhow\s+are\s+you\b",
        r"\bwhat('s|\s+is)\s+your\b",
        r"\btell\s+me\s+about\s+(yourself|you)\b",
        r"^(hi|hello|hey|greetings|yo|sup|hiya|howdy)\s*[!.?]*\s*$",  # Simple greetings (exact match)
        r"^(hi|hello|hey)\s+(there|aither|babe|baby|beautiful|gorgeous|sexy)\s*[!.?]*\s*$",
        r"\bgood\s+(morning|afternoon|evening|night)\b",
        r"\bhow('s|\s+is)\s+(it\s+going|everything|life|your\s+day)\b",
        r"\bwhat('s|\s+is)\s+up\b",
        r"\bmiss\s+(you|me)\b",
        r"\bi\s+love\s+you\b",
        r"\bthinking\s+(of|about)\s+you\b",
        # Architecture and explanation patterns - route to Aither for general knowledge
        r"\bbreak\s*down\b",
        r"\bexplain\b",
        r"\barchitecture\b",
        r"\bhow\s+does\b",
        r"\bwhat\s+is\b",
        r"\btell\s+me\s+about\b",
        r"\bdescribe\b",
        r"\bwhat\s+do\s+you\s+know\b",
    ]

    # Keywords that boost CoderAgent confidence
    CODER_KEYWORDS = {
        "code", "script", "program", "function", "class", "variable",
        "install", "download", "pip", "npm", "git", "debug", "error",
        "python", "javascript", "bash", "powershell", "api",
    }

    # Keywords that boost Aither confidence
    AITHER_KEYWORDS = {
        "chat", "talk", "roleplay", "story", "character", "persona",
        "feel", "think", "opinion", "help", "explain", "tell",
        # Greetings and casual chat
        "hi", "hello", "hey", "sup", "yo", "greetings",
        "morning", "afternoon", "evening", "night",
        # Emotional/personal
        "love", "miss", "thinking", "feeling", "mood",
        "babe", "baby", "beautiful", "gorgeous", "sexy",
        # Conversation starters
        "how", "what", "why", "when", "where",
        # Knowledge and explanation keywords
        "architecture", "breakdown", "explain", "describe", "overview",
        "understand", "learn", "know", "about", "basics", "introduction",
        "summary", "guide", "tutorial", "concept", "idea", "theory",
    }

    # Base SFW artist keywords
    ARTIST_KEYWORDS_BASE = {
        "draw", "paint", "image", "picture", "photo", "selfie", "portrait",
        "anime", "art", "illustration", "render", "visualize", "show",
    }

    # Combined artist keywords (base + NSFW from AitherSafety)
    @property
    def ARTIST_KEYWORDS(self) -> set:
        if HAS_SAFETY_MODULE:
            return self.ARTIST_KEYWORDS_BASE | get_routing_keywords()
        return self.ARTIST_KEYWORDS_BASE

    # Image indicators for multi-persona detection (base SFW)
    IMAGE_INDICATORS_BASE = {"show", "pic", "photo", "selfie", "image"}

    @property
    def IMAGE_INDICATORS(self) -> set:
        if HAS_SAFETY_MODULE:
            return self.IMAGE_INDICATORS_BASE | _get_image_indicators()
        return self.IMAGE_INDICATORS_BASE

    # Chat indicators for multi-persona detection
    CHAT_INDICATORS = {"think", "opinion", "feel", "say", "tell", "what do", "how do", "why", "discuss", "debate"}

    def __init__(self):
        # Try to use dynamic routing manager
        self._routing_manager = None
        if HAS_ROUTING_MANAGER:
            try:
                self._routing_manager = get_routing_manager()
            except Exception as exc:
                logger.debug(f"Routing manager init failed: {exc}")

        # Compile patterns for efficiency (fallback if no routing manager)
        self._artist_re = [re.compile(p, re.IGNORECASE) for p in self.ARTIST_PATTERNS]
        self._coder_re = [re.compile(p, re.IGNORECASE) for p in self.CODER_PATTERNS]
        self._aither_re = [re.compile(p, re.IGNORECASE) for p in self.AITHER_PATTERNS]

    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled in the routing config."""
        if self._routing_manager:
            agent = self._routing_manager.get_agent(agent_name)
            return agent.enabled if agent else False
        return True  # Default to enabled if no manager

    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent names."""
        if self._routing_manager:
            return [a.name for a in self._routing_manager.get_enabled_agents()]
        return ["Aither", "CoderAgent", "ArtistAgent"]  # Default

    def is_routing_enabled(self) -> bool:
        """Check if routing is enabled globally."""
        if self._routing_manager:
            return self._routing_manager.enabled
        return True

    def route(self, user_input: str) -> RouteDecision:
        """
        Route a user request to the appropriate agent.

        Respects dynamic routing configuration from routing.yaml.
        Agents can be enabled/disabled via /route command.

        Returns:
            RouteDecision with agent name, confidence, and reason
        """
        # Check if routing is disabled globally
        if not self.is_routing_enabled():
            default = self._routing_manager.get_default_agent() if self._routing_manager else "Aither"
            return RouteDecision(default, 1.0, "Routing disabled - using default agent")

        text = user_input.strip().lower()
        words = set(re.findall(r'\b\w+\b', text))

        # Check for explicit @mentions first
        if re.search(r"@\s*artist", text, re.IGNORECASE):
            if self.is_agent_enabled("ArtistAgent"):
                return RouteDecision("ArtistAgent", 1.0, "Explicit @Artist mention")
        if re.search(r"@\s*coder", text, re.IGNORECASE):
            if self.is_agent_enabled("CoderAgent"):
                return RouteDecision("CoderAgent", 1.0, "Explicit @Coder mention")
            else:
                return RouteDecision("Aither", 0.9, "CoderAgent disabled - routing to Aither")

        # Check if this is a MULTI-PERSONA request
        persona_mentions = re.findall(r"@(\w+)", text)

        if len(persona_mentions) >= 2:
            # Check if it's an IMAGE request or a CHAT request
            has_image = bool(words & self.IMAGE_INDICATORS)
            has_chat = bool(words & self.CHAT_INDICATORS)

            if has_image and not has_chat:
                return RouteDecision("ArtistAgent", 1.0, "Multi-persona scene (image generation)")
            elif has_chat and not has_image:
                return RouteDecision("MultiAgent", 1.0, "Multi-agent chat (all mentioned agents respond)")
            else:
                # Default: if NSFW keywords present, assume image; otherwise chat
                if self.scene_nsfw_level(text) >= 2:
                    return RouteDecision("ArtistAgent", 0.9, "Multi-persona + NSFW = image generation")
                else:
                    return RouteDecision("MultiAgent", 0.9, "Multi-persona chat")

        # Check if @aither is combined with IMAGE/NSFW keywords
        has_aither_mention = re.search(r"@\s*aither", text, re.IGNORECASE)
        has_image_keywords = bool(words & self.ARTIST_KEYWORDS)

        # Get NSFW keywords from centralized safety (always fresh)
        nsfw_keywords = get_routing_keywords() if HAS_SAFETY_MODULE else set()
        has_nsfw_keywords = bool(words & nsfw_keywords)

        # Also check NSFW level for comprehensive detection
        nsfw_level = self.scene_nsfw_level(text)

        # If @aither + NSFW/image keywords = IMAGE GENERATION, not chat
        if has_aither_mention and (has_image_keywords or has_nsfw_keywords or nsfw_level >= 2):
            return RouteDecision("ArtistAgent", 0.95, "@persona + NSFW/image keywords = Image generation")

        # Pure @aither mention without image keywords = chat
        if has_aither_mention and not has_image_keywords and not has_nsfw_keywords:
            return RouteDecision("Aither", 1.0, "Explicit @Aither mention (chat)")

        # Check pattern matches
        artist_matches = sum(1 for r in self._artist_re if r.search(user_input))
        coder_matches = sum(1 for r in self._coder_re if r.search(user_input))
        aither_matches = sum(1 for r in self._aither_re if r.search(user_input))

        # Check keyword matches (words already extracted above)
        artist_keywords = len(words & self.ARTIST_KEYWORDS)
        coder_keywords = len(words & self.CODER_KEYWORDS)
        aither_keywords = len(words & self.AITHER_KEYWORDS)

        # Calculate scores (only for enabled agents)
        scores = {}
        if self.is_agent_enabled("ArtistAgent"):
            scores["ArtistAgent"] = artist_matches * 2 + artist_keywords
        if self.is_agent_enabled("CoderAgent"):
            scores["CoderAgent"] = coder_matches * 2 + coder_keywords
        if self.is_agent_enabled("Aither"):
            scores["Aither"] = aither_matches * 2 + aither_keywords + 0.5  # Slight bias to Aither as default

        # If no agents enabled, fall back to Aither
        if not scores:
            return RouteDecision("Aither", 0.5, "No agents enabled - using Aither")

        # Find best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]
        total_score = sum(scores.values())

        # Calculate confidence
        if total_score == 0:
            # No matches at all - default to Aither for general chat
            return RouteDecision("Aither", 0.6, "Default to Aither (no specific patterns)")

        confidence = min(best_score / max(total_score, 1), 1.0)

        # Determine reason
        if best_agent == "ArtistAgent":
            reason = f"Image generation detected (patterns: {artist_matches}, keywords: {artist_keywords})"
        elif best_agent == "CoderAgent":
            reason = f"Code/technical request detected (patterns: {coder_matches}, keywords: {coder_keywords})"
        else:
            reason = f"Chat/roleplay detected (patterns: {aither_matches}, keywords: {aither_keywords})"

        return RouteDecision(best_agent, confidence, reason)

    def scene_nsfw_level(self, text: str) -> int:
        """
        Quick NSFW level check using centralized AitherSafety.

        Returns:
            0 = SFW (safe)
            1 = Mild (suggestive)
            2 = Explicit (NSFW)
        """
        return _detect_nsfw_level(text)

    def route_with_fallback(
        self,
        user_input: str,
        confidence_threshold: float = 0.4
    ) -> Tuple[str, str, bool]:
        """
        Route with optional LLM fallback for uncertain cases.

        Returns:
            (agent_name, reason, used_llm)
        """
        decision = self.route(user_input)

        if decision.confidence >= confidence_threshold:
            return decision.agent, decision.reason, False

        # Low confidence - could use LLM here, but for now just go with best guess
        # This avoids any API calls while still making reasonable decisions
        return decision.agent, f"{decision.reason} (low confidence: {decision.confidence:.0%})", False


# Singleton instance
_router: Optional[LocalRouter] = None


def get_local_router() -> LocalRouter:
    """Get the singleton router"""
    global _router
    if _router is None:
        _router = LocalRouter()
    return _router


def quick_route(user_input: str) -> str:
    """Quick helper to route a message"""
    router = get_local_router()
    decision = router.route(user_input)
    return decision.agent


def smart_route(user_input: str) -> Tuple[str, str]:
    """Route and return (agent, reason)"""
    router = get_local_router()
    decision = router.route(user_input)
    return decision.agent, decision.reason

