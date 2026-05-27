"""
AitherAeon Client - Unified interface for group chat and all Aither services.

This module provides CLI agents with full access to the AitherAeon API,
ensuring parity between the web dashboard and CLI experiences.

Features:
- Group chat with multiple agents (Aeon discussions)
- Access to all integrated services (Vision, Canvas, Spirit, Will, etc.)
- Response depth control (concise, thoughtful, deep)
- Safety level management
- Service health monitoring

Usage:
    from aither_adk.communication.aeon_client import AeonClient, aeon

    # Quick group chat
    response = await aeon.chat("What should we prioritize?", ["aither", "hydra", "terra"])

    # Access tools
    analysis = await aeon.analyze_image("/path/to/image.png")
    status = await aeon.get_services_health()

Note: CouncilClient is still available as a backward compatibility alias.
"""

import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Default Aeon API URL
AEON_API_URL = os.getenv("AITHERAEON_URL", os.getenv("AITHERCOUNCIL_URL", "http://localhost:8765"))


class ResponseDepth(str, Enum):
    """Response depth levels."""
    CONCISE = "concise"      # 2-3 sentences, quick answers
    THOUGHTFUL = "thoughtful"  # Default, balanced depth
    DEEP = "deep"            # In-depth analysis, multi-paragraph


@dataclass
class AgentMessage:
    """A message from an agent in a group chat."""
    agent_id: str
    agent_name: str
    persona: Optional[str]
    content: str
    type: str  # message, system, memory, thought
    timestamp: str


@dataclass
class ChatResult:
    """Result of a group chat session."""
    messages: List[AgentMessage]
    memory_ids: List[str]
    reasoning_session_id: Optional[str]
    success: bool
    error: Optional[str] = None


class AeonClient:
    """
    Client for interacting with the AitherAeon API.

    Provides unified access to:
    - Group chat with multiple agents
    - Vision analysis
    - Image generation
    - Spirit memories
    - Will management
    - Service monitoring
    - Safety filtering
    - Voice capabilities
    - Job scheduling
    """

    def __init__(self, base_url: str = None):
        self.base_url = base_url or AEON_API_URL
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # Group Chat
    # =========================================================================

    async def chat(
        self,
        prompt: str,
        participants: List[str] = None,
        topic: str = "General Discussion",
        rounds: int = 1,
        depth: ResponseDepth = ResponseDepth.THOUGHTFUL,
        model: str = "aither-orchestrator-8b-v4",
        save_to_memory: bool = True
    ) -> ChatResult:
        """
        Start a group chat with multiple agents.

        Args:
            prompt: The discussion prompt/question
            participants: List of agent IDs (default: ["aither", "hydra", "terra"])
            topic: Discussion topic for context
            rounds: Number of discussion rounds
            depth: Response depth (concise, thoughtful, deep)
            model: LLM model to use
            save_to_memory: Whether to save to episodic memory

        Returns:
            ChatResult with all agent messages
        """
        if participants is None:
            participants = ["aither", "hydra", "terra"]

        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.base_url}/chat",
                json={
                    "prompt": prompt,
                    "participants": participants,
                    "topic": topic,
                    "rounds": rounds,
                    "depth": depth.value if isinstance(depth, ResponseDepth) else depth,
                    "model": model,
                    "save_to_memory": save_to_memory
                }
            )

            if response.status_code == 200:
                data = response.json()
                messages = [
                    AgentMessage(
                        agent_id=m.get("agent_id", ""),
                        agent_name=m.get("agent_name", ""),
                        persona=m.get("persona"),
                        content=m.get("content", ""),
                        type=m.get("type", "message"),
                        timestamp=m.get("timestamp", "")
                    )
                    for m in data.get("messages", [])
                ]
                return ChatResult(
                    messages=messages,
                    memory_ids=data.get("memory_ids", []),
                    reasoning_session_id=data.get("reasoning_session_id"),
                    success=True
                )
            else:
                return ChatResult(
                    messages=[],
                    memory_ids=[],
                    reasoning_session_id=None,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            return ChatResult(
                messages=[],
                memory_ids=[],
                reasoning_session_id=None,
                success=False,
                error=str(e)
            )

    async def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of available aeon agents."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/agents")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    # =========================================================================
    # Vision & Canvas (delegated to full client)
    # =========================================================================

    async def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail",
        analysis_type: str = "describe"
    ) -> Dict[str, Any]:
        """Analyze an image using AitherVision."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/tools/vision/analyze",
                json={
                    "image_path": image_path,
                    "prompt": prompt,
                    "analysis_type": analysis_type
                }
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    async def generate_image(
        self,
        prompt: str,
        style: str = "realistic",
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = ""
    ) -> Dict[str, Any]:
        """Generate an image using AitherCanvas."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/tools/canvas/generate",
                json={
                    "prompt": prompt,
                    "style": style,
                    "width": width,
                    "height": height,
                    "negative_prompt": negative_prompt
                }
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Spirit Memory
    # =========================================================================

    async def teach_memory(
        self,
        content: str,
        memory_type: str = "teaching",
        importance: float = 0.7
    ) -> Dict[str, Any]:
        """Teach a new memory to AitherSpirit."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/tools/spirit/teach",
                json={
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance
                }
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    async def recall_memories(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Recall relevant memories from AitherSpirit."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.base_url}/tools/spirit/recall",
                params={"query": query, "limit": limit}
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Service Monitoring
    # =========================================================================

    async def get_services_health(self) -> Dict[str, Any]:
        """Get health status of all Aither services."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/tools/watch/services")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    async def get_tool_status(self) -> Dict[str, Any]:
        """Get status of all available tools/integrations."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/tools/status")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Settings & Customization
    # =========================================================================

    async def get_settings(self) -> Dict[str, Any]:
        """Get current Aeon settings."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/settings")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    async def update_settings(self, **kwargs) -> Dict[str, Any]:
        """Update Aeon settings."""
        client = await self._get_client()
        try:
            response = await client.patch(
                f"{self.base_url}/settings",
                json=kwargs
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}


# Global singleton instance
aeon = AeonClient()

# Backward compatibility aliases
CouncilClient = AeonClient
council = aeon
COUNCIL_API_URL = AEON_API_URL


# =========================================================================
# CLI Helper Functions
# =========================================================================

async def quick_aeon_chat(
    prompt: str,
    agents: List[str] = None,
    depth: str = "thoughtful"
) -> str:
    """
    Quick helper for CLI to run an aeon chat and format output.

    Returns formatted string of agent responses.
    """
    result = await aeon.chat(
        prompt=prompt,
        participants=agents or ["aither", "hydra", "terra"],
        depth=ResponseDepth(depth)
    )

    if not result.success:
        return f"[FAIL] Aeon chat failed: {result.error}"

    output = []
    for msg in result.messages:
        if msg.type == "message":
            output.append(f"**{msg.agent_name}** ({msg.persona}):\n{msg.content}\n")
        elif msg.type == "system":
            output.append(f"_[{msg.content}]_\n")

    return "\n".join(output)


# Backward compatibility alias
quick_council_chat = quick_aeon_chat


def format_tool_status(status: Dict[str, Any]) -> str:
    """Format tool status for CLI display."""
    lines = ["[CHART] **Aither Ecosystem Status**\n"]

    for tool, info in status.items():
        enabled = info.get("enabled", False)
        desc = info.get("description", "")
        icon = "[DONE]" if enabled else "[FAIL]"
        lines.append(f"{icon} **{tool.title()}**: {desc}")

    return "\n".join(lines)
