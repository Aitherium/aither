"""
AitherDemiurge Client - CLI Interface for Development Interaction

This module provides CLI agents with access to the AitherDemiurge service,
enabling Claude Code-like development interaction capabilities.

Features:
- Intent-based code generation and analysis
- Project scaffolding and architecture
- Code review and suggestions
- Integration with AitherNeurons for codebase search
- Artifact storage and retrieval

Usage:
    from aither_adk.communication.demiurge_client import DemiurgeClient, demiurge
    
    # Quick intent request
    response = await demiurge.intent("analyze the authentication flow")
    
    # Code generation
    result = await demiurge.create("a FastAPI endpoint for user registration")
    
    # Project analysis
    analysis = await demiurge.analyze("/services/auth")
"""

import os
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


# Default Demiurge API URL
DEMIURGE_API_URL = os.getenv("AITHERDEMIURGE_URL", "http://localhost:8140")


class DemiurgeMood(str, Enum):
    """Demiurge mood states."""
    NEUTRAL = "neutral"
    FOCUSED = "focused"
    CREATIVE = "creative"
    CONCERNED = "concerned"
    EXCITED = "excited"
    SLEEPING = "sleeping"


@dataclass
class DemiurgeResponse:
    """Response from the Demiurge."""
    response: str
    mood: str
    creativity_boost: float
    status: str
    request_id: str
    neurons_streaming: bool = False
    roadmap_context: Optional[str] = None
    error: Optional[str] = None


@dataclass
class NeuronResult:
    """Result from a neuron search."""
    neuron_type: str
    found: bool
    sources: List[str]
    summary: str
    confidence: float
    execution_ms: float


class DemiurgeClient:
    """
    Client for interacting with the AitherDemiurge service.
    
    Provides CLI agents with development capabilities similar to Claude Code:
    - Intent-based code generation
    - Project analysis and review
    - Codebase search via neurons
    - Infrastructure requests
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or DEMIURGE_API_URL
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
    
    async def is_available(self) -> bool:
        """Check if Demiurge service is available."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False
    
    # =========================================================================
    # Core Intent Interface (like Claude Code)
    # =========================================================================
    
    async def intent(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        stream_neurons: bool = True
    ) -> DemiurgeResponse:
        """
        Send an intent to the Demiurge for processing.
        
        This is the main entry point for development interaction,
        similar to how Claude Code processes user requests.
        
        Args:
            message: The intent/request (e.g., "analyze the auth flow")
            context: Additional context (file paths, neuron results, etc.)
            stream_neurons: Whether to fire neurons in background
            
        Returns:
            DemiurgeResponse with the result
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/intent",
                json={
                    "intent": message,
                    "context": context or {},
                    "stream_neurons": stream_neurons
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return DemiurgeResponse(
                    response=data.get("response", ""),
                    mood=data.get("mood", "neutral"),
                    creativity_boost=data.get("creativity_boost", 1.0),
                    status=data.get("status", "unknown"),
                    request_id=data.get("request_id", ""),
                    neurons_streaming=data.get("neurons_streaming", False),
                    roadmap_context=data.get("roadmap_context")
                )
            else:
                return DemiurgeResponse(
                    response="",
                    mood="concerned",
                    creativity_boost=1.0,
                    status="error",
                    request_id="",
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            return DemiurgeResponse(
                response="",
                mood="sleeping",
                creativity_boost=1.0,
                status="error",
                request_id="",
                error=str(e)
            )
    
    async def create(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DemiurgeResponse:
        """
        Request code creation from the Demiurge.
        
        Args:
            description: What to create
            context: Additional context (project type, language, etc.)
            
        Returns:
            DemiurgeResponse with generated code
        """
        intent = f"create {description}"
        return await self.intent(intent, context)
    
    async def analyze(
        self,
        path: str,
        focus: str = "structure"
    ) -> DemiurgeResponse:
        """
        Request code analysis from the Demiurge.
        
        Args:
            path: File or directory to analyze
            focus: Analysis focus (structure, security, performance, etc.)
            
        Returns:
            DemiurgeResponse with analysis
        """
        intent = f"analyze {path} focusing on {focus}"
        return await self.intent(intent, {"file": path, "focus": focus})
    
    async def review(
        self,
        path: str,
        changes: str = None
    ) -> DemiurgeResponse:
        """
        Request code review from the Demiurge.
        
        Args:
            path: File to review
            changes: Description of changes to review
            
        Returns:
            DemiurgeResponse with review feedback
        """
        intent = f"review code at {path}"
        if changes:
            intent += f" with changes: {changes}"
        return await self.intent(intent, {"file": path})
    
    async def explain(self, topic: str) -> DemiurgeResponse:
        """
        Request explanation of code or concept.
        
        Args:
            topic: What to explain
            
        Returns:
            DemiurgeResponse with explanation
        """
        return await self.intent(f"explain {topic}")
    
    async def suggest_next(self) -> DemiurgeResponse:
        """
        Get suggestions for what to work on next.
        
        Uses roadmap context to provide prioritized suggestions.
        
        Returns:
            DemiurgeResponse with suggestions
        """
        return await self.intent("what should I work on next?")
    
    # =========================================================================
    # Mood & Status
    # =========================================================================
    
    async def get_mood(self) -> Dict[str, Any]:
        """Get the Demiurge's current mood and temporal context."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/pulse/mood")
            if response.status_code == 200:
                return response.json()
            return {"mood": "unknown", "error": response.text}
        except Exception as e:
            return {"mood": "unknown", "error": str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get Demiurge service status."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/")
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "error": response.text}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    async def get_temporal(self) -> Dict[str, Any]:
        """Get Demiurge's temporal awareness status."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/temporal")
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    async def awaken(self) -> Dict[str, Any]:
        """Awaken the Demiurge if sleeping."""
        client = await self._get_client()
        try:
            response = await client.post(f"{self.base_url}/pulse/awaken")
            if response.status_code == 200:
                return response.json()
            return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # Artifact Management
    # =========================================================================
    
    async def store_artifact(
        self,
        artifact_id: str,
        content: str,
        artifact_type: str = "code",
        language: str = "python",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a code artifact via Demiurge.
        
        Artifacts are persisted to AitherStrata for:
        - Version history
        - Training data extraction
        - Cross-session retrieval
        """
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/artifacts/store",
                json={
                    "artifact_id": artifact_id,
                    "content": content,
                    "artifact_type": artifact_type,
                    "language": language,
                    "metadata": metadata or {}
                }
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_artifacts(
        self,
        artifact_type: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """List stored artifacts."""
        client = await self._get_client()
        try:
            params = {"limit": limit}
            if artifact_type:
                params["artifact_type"] = artifact_type
            response = await client.get(f"{self.base_url}/artifacts", params=params)
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # Infrastructure Delegation
    # =========================================================================
    
    async def request_infrastructure(
        self,
        request_type: str,
        project_name: str = None,
        template: str = "standard",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Request infrastructure deployment via Demiurge.
        
        Demiurge delegates to Aither Orchestrator -> ServicesManagerAgent.
        
        Args:
            request_type: Type (deploy_vm, deploy_cluster, deploy_exo, status)
            project_name: Project name for deployment
            template: Template (minimal, standard, gpu, exo-node, llm)
            parameters: Additional parameters
            
        Returns:
            Response with request status
        """
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/request-infrastructure",
                json={
                    "request_type": request_type,
                    "project_name": project_name,
                    "template": template,
                    "parameters": parameters or {}
                }
            )
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get infrastructure status."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/infrastructure/status")
            return response.json() if response.status_code == 200 else {"error": response.text}
        except Exception as e:
            return {"error": str(e)}


# Global singleton instance
demiurge = DemiurgeClient()


# =========================================================================
# CLI Helper Functions
# =========================================================================

async def quick_demiurge_intent(message: str) -> str:
    """
    Quick helper for CLI to send intent and get formatted response.
    
    Returns formatted string of Demiurge response.
    """
    result = await demiurge.intent(message)
    
    if result.error:
        return f"[FAIL] Demiurge error: {result.error}"
    
    mood_emoji = {
        "neutral": "",
        "focused": "[TARGET]",
        "creative": "*",
        "concerned": "",
        "excited": "[!]",
        "sleeping": ""
    }.get(result.mood, "")
    
    output = f"{mood_emoji} **Demiurge** ({result.mood}):\n\n{result.response}"
    
    if result.neurons_streaming:
        output += "\n\n_[Neurons gathering context in background...]_"
    
    return output


def format_demiurge_status(status: Dict[str, Any]) -> str:
    """Format Demiurge status for CLI display."""
    if "error" in status:
        return f"[FAIL] Demiurge offline: {status['error']}"
    
    mood = status.get("mood", "unknown")
    mood_emoji = {
        "neutral": "",
        "focused": "[TARGET]",
        "creative": "*",
        "concerned": "",
        "excited": "[!]",
        "sleeping": ""
    }.get(mood, "")
    
    creativity = status.get("creativity_boost", 1.0)
    temporal = status.get("temporal_context", {})
    time_of_day = temporal.get("time_of_day", "unknown") if temporal else "unknown"
    
    lines = [
        f"[HOT] **The Forge** - AitherDemiurge",
        f"",
        f"Mood: {mood_emoji} {mood.title()}",
        f"Creativity Boost: {creativity:.1f}x",
        f"Time of Day: {time_of_day}",
    ]
    
    return "\n".join(lines)

