"""
AitherOS Service Connector
===========================
Fast connection to running AitherOS services.

Checks if services are already running and uses them directly,
avoiding expensive re-initialization.

This module integrates with the central service registry (services.psd1)
for consistent port definitions across the entire ecosystem.

Usage:
    from aither_adk.infrastructure.services import get_services, is_aithernode_running

    services = get_services()
    if services.aithernode:
        # Use HTTP API instead of importing heavy modules
        result = await services.call_tool("generate_image", {"prompt": "..."})
"""

import logging
import socket
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Suppress unclosed resource warnings at exit (these are harmless cleanup messages)
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")


def _load_ports_from_registry() -> Dict[str, int]:
    """
    Load service ports from the central registry (services.psd1).
    Falls back to hardcoded defaults if registry is unavailable.
    """
    # Try to import from the AitherNode registry
    aithernode_path = Path(__file__).parent.parent.parent / "AitherNode"
    if aithernode_path.exists() and str(aithernode_path) not in sys.path:
        sys.path.insert(0, str(aithernode_path))

    try:
        from service_registry import get_registry
        registry = get_registry()

        # Map service names to lowercase keys for this module
        ports = {}
        name_mapping = {
            "AitherNode": "aithernode",
            "AitherPulse": "pulse",
            "AitherReasoning": "reasoning",
            "AitherPrism": "prism",
            "AitherTrainer": "trainer",
            "AitherLLM": "aitherllm",
            "ComfyUI": "comfyui",
            "Ollama": "ollama",
        }

        for svc_name, svc_info in registry.get_all_services().items():
            if svc_name in name_mapping:
                ports[name_mapping[svc_name]] = svc_info.port

        return ports
    except ImportError:
        pass

    # Fallback to hardcoded defaults
    return {
        "aithernode": 8080,
        "pulse": 8081,
        "reasoning": 8093,
        "prism": 8106,
        "trainer": 8107,
        "aitherllm": 8150,
        "comfyui": 8188,
        "ollama": 11434,
    }


# Service ports - loaded from registry or fallback
PORTS: Dict[str, int] = _load_ports_from_registry()

# Cache timeout for service checks
_service_cache: Dict[str, bool] = {}
_cache_time: Dict[str, float] = {}  # Per-service cache times
CACHE_TTL = 30  # seconds


def _check_port(port: int, host: str = "127.0.0.1", timeout: float = 0.1) -> bool:
    """Fast port check using socket. 100ms timeout for speed."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _refresh_cache_if_needed(service: str = None):
    """Refresh service cache if TTL expired. Only checks requested service."""
    import time
    global _service_cache, _cache_time

    now = time.time()

    if service:
        # Only check the specific service
        if now - _cache_time.get(service, 0) > CACHE_TTL:
            port = PORTS.get(service)
            if port:
                _service_cache[service] = _check_port(port)
                _cache_time[service] = now
    else:
        # Refresh all (called from get_running_services)
        for name, port in PORTS.items():
            if now - _cache_time.get(name, 0) > CACHE_TTL:
                _service_cache[name] = _check_port(port)
                _cache_time[name] = now


def is_service_running(service: str) -> bool:
    """Check if a service is running (cached, lazy)."""
    _refresh_cache_if_needed(service)
    return _service_cache.get(service, False)


def is_aithernode_running() -> bool:
    """Check if AitherNode MCP server is running."""
    return is_service_running("aithernode")


def is_reasoning_running() -> bool:
    """Check if AitherReasoning is running."""
    return is_service_running("reasoning")


def is_comfyui_running() -> bool:
    """Check if ComfyUI is running."""
    return is_service_running("comfyui")


def is_ollama_running() -> bool:
    """Check if Ollama is running."""
    return is_service_running("ollama")


def get_running_services() -> Dict[str, bool]:
    """Get status of all services."""
    _refresh_cache_if_needed()  # Refreshes all services
    return _service_cache.copy()


@dataclass
class ServiceStatus:
    """Service status summary."""
    aithernode: bool
    reasoning: bool
    comfyui: bool
    ollama: bool
    pulse: bool
    prism: bool
    trainer: bool
    aitherllm: bool = False  # Unified LLM gateway

    @property
    def any_running(self) -> bool:
        return any([self.aithernode, self.reasoning, self.comfyui, self.ollama])

    def __str__(self) -> str:
        services = []
        if self.aithernode: services.append("AitherNode:8080")
        if self.pulse: services.append("Pulse:8081")
        if self.reasoning: services.append("Reasoning:8093")
        if self.aitherllm: services.append("MicroScheduler:8150")
        if self.comfyui: services.append("ComfyUI:8188")
        if self.ollama: services.append("Ollama:11434")
        return f"Running: {', '.join(services) if services else 'None'}"


def get_service_status() -> ServiceStatus:
    """Get status of all AitherOS services."""
    _refresh_cache_if_needed()  # Refreshes all services
    return ServiceStatus(
        aithernode=_service_cache.get("aithernode", False),
        reasoning=_service_cache.get("reasoning", False),
        comfyui=_service_cache.get("comfyui", False),
        ollama=_service_cache.get("ollama", False),
        pulse=_service_cache.get("pulse", False),
        prism=_service_cache.get("prism", False),
        trainer=_service_cache.get("trainer", False),
        aitherllm=_service_cache.get("aitherllm", False),
    )


class AitherNodeClient:
    """HTTP client for AitherNode MCP server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def health(self) -> bool:
        """Check if AitherNode is healthy."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a tool on AitherNode via HTTP."""
        try:
            session = await self._get_session()
            # MCP tools are exposed at /tools/{tool_name} or via SSE
            # For now, use direct endpoint if available
            async with session.post(
                f"{self.base_url}/api/tools/{tool_name}",
                json=arguments or {}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def get_service_status(self, refresh: bool = False) -> Dict[str, Any]:
        """Get service status from AitherNode."""
        try:
            session = await self._get_session()
            params = {"refresh": "true"} if refresh else {}
            async with session.get(
                f"{self.base_url}/api/services/status",
                params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as exc:
            logger.debug(f"Failed to get service status: {exc}")
        return {}

    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image via AitherNode."""
        return await self.call_tool("generate_image", {"prompt": prompt, **kwargs})

    async def remember(self, content: str, category: str = "general", **kwargs) -> Dict[str, Any]:
        """Store in long-term memory via AitherNode."""
        return await self.call_tool("remember", {"content": content, "category": category, **kwargs})

    async def recall(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search memory via AitherNode."""
        return await self.call_tool("recall", {"query": query, "limit": limit})


class ReasoningClient:
    """HTTP client for AitherReasoning service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8093"):
        self.base_url = base_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_available(self) -> bool:
        """Check if reasoning service is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def create_session(self, agent: str, user_query: str) -> Optional[str]:
        """Create a reasoning session, returns session ID."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sessions",
                json={"agent": agent, "user_query": user_query}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("id")
        except Exception as exc:
            logger.debug(f"Failed to create reasoning session: {exc}")
        return None

    async def add_thought(self, session_id: str, thought_type: str, content: str, **kwargs) -> bool:
        """Add a thought to a session."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/thoughts",
                json={
                    "session_id": session_id,
                    "type": thought_type,
                    "content": content,
                    **kwargs
                }
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def end_session(self, session_id: str, final_response: str = None) -> bool:
        """End a reasoning session."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/sessions/{session_id}/end",
                json={"final_response": final_response}
            ) as resp:
                return resp.status == 200
        except Exception:
            return False


class PulseClient:
    """HTTP client for AitherPulse - ecosystem health and pain signals."""

    def __init__(self, base_url: str = "http://127.0.0.1:8081"):
        self.base_url = base_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_pain_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive pain status - the 'feeling' of the ecosystem."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/pain/dashboard") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as exc:
            logger.debug(f"Failed to get pain dashboard: {exc}")
        return {}

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts that need attention."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/alerts") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("alerts", [])
        except Exception as exc:
            logger.debug(f"Failed to get active alerts: {exc}")
        return []

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics (CPU, memory, etc)."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/metrics") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as exc:
            logger.debug(f"Failed to get system metrics: {exc}")
        return {}

    async def track_tokens(self, agent_id: str, session_id: str,
                          input_tokens: int, output_tokens: int,
                          action: str = None) -> bool:
        """Report token usage to pulse for burn rate tracking."""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/tokens/track",
                json={
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "action": action
                }
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_ecosystem_summary(self) -> str:
        """Get a human-readable summary of ecosystem state for agent context."""
        try:
            dashboard = await self.get_pain_dashboard()
            metrics = await self.get_metrics()
            alerts = await self.get_active_alerts()

            lines = ["[Ecosystem Status]"]

            # Pain level
            pain_level = dashboard.get("pain_level", "unknown")
            pain_score = dashboard.get("total_pain_score", 0)
            if pain_level != "none":
                lines.append(f"[WARN] Pain Level: {pain_level.upper()} (score: {pain_score:.1f})")
                if dashboard.get("interrupt_message"):
                    lines.append(f"   -> {dashboard['interrupt_message']}")

            # Active alerts
            critical_alerts = [a for a in alerts if a.get("severity", 0) > 0.7]
            if critical_alerts:
                lines.append(f" {len(critical_alerts)} critical alert(s):")
                for alert in critical_alerts[:3]:
                    lines.append(f"   * {alert.get('title', 'Unknown')}: {alert.get('message', '')[:50]}")

            # Resources
            cpu = metrics.get("cpu_percent", 0)
            mem = metrics.get("memory_percent", 0)
            if cpu > 80 or mem > 80:
                lines.append(f"[CHART] Resources: CPU {cpu:.0f}%, Memory {mem:.0f}%")

            # Token burn
            burn_rate = dashboard.get("token_burn_rate", 0)
            if burn_rate > 5000:
                lines.append(f" Token burn rate: {burn_rate:.0f}/min (high)")

            # Return empty if nothing notable
            if len(lines) == 1:
                return ""

            return "\n".join(lines)
        except Exception:
            return ""


# Global clients (lazy init)
_aithernode_client: Optional[AitherNodeClient] = None
_reasoning_client: Optional[ReasoningClient] = None
_pulse_client: Optional[PulseClient] = None


def is_pulse_running() -> bool:
    """Check if AitherPulse is running."""
    return is_service_running("pulse")


def get_aithernode_client() -> Optional[AitherNodeClient]:
    """Get AitherNode client if service is running."""
    global _aithernode_client
    if not is_aithernode_running():
        return None
    if _aithernode_client is None:
        _aithernode_client = AitherNodeClient()
    return _aithernode_client


def get_reasoning_client() -> Optional[ReasoningClient]:
    """Get Reasoning client if service is running."""
    global _reasoning_client
    if not is_reasoning_running():
        return None
    if _reasoning_client is None:
        _reasoning_client = ReasoningClient()
    return _reasoning_client


def get_pulse_client() -> Optional[PulseClient]:
    """Get Pulse client if service is running."""
    global _pulse_client
    if not is_pulse_running():
        return None
    if _pulse_client is None:
        _pulse_client = PulseClient()
    return _pulse_client


async def cleanup_clients():
    """Close all client sessions."""
    global _aithernode_client, _reasoning_client, _pulse_client
    if _aithernode_client:
        await _aithernode_client.close()
        _aithernode_client = None
    if _reasoning_client:
        await _reasoning_client.close()
        _reasoning_client = None
    if _pulse_client:
        await _pulse_client.close()
        _pulse_client = None


# Quick status check for startup
def print_service_status():
    """Print service status for debugging."""
    status = get_service_status()
    print(f"[PLUG] {status}")
