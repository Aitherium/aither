#!/usr/bin/env python3
"""
AitherA2A Client - Python client for Google's A2A Protocol.

This client allows Aither agents to communicate with other A2A-compatible agents,
both within AitherOS and external agents implementing the A2A protocol.

Usage:
    from a2a_client import A2AClient, A2ATask
    
    async with A2AClient() as client:
        # Discover available agents
        agents = await client.list_agents()
        
        # Get an agent's capabilities
        card = await client.get_agent_card("terra")
        
        # Send a task and wait for response
        task = await client.send_task(
            agent="terra",
            message="What is the current system status?"
        )
        
        # Stream task updates
        async for update in client.send_task_streaming("aither", "Analyze this codebase"):
            print(update)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

# Default A2A Gateway
A2A_BASE_URL = os.getenv("AITHER_A2A_URL", "http://localhost:8766")


class TaskState(str, Enum):
    """Task lifecycle states per A2A v0.3.0 spec."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


@dataclass
class AgentSkill:
    """An agent skill/capability."""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class AgentCard:
    """A2A Agent Card - describes an agent's capabilities."""
    name: str
    description: str
    url: str
    version: str
    skills: List[AgentSkill] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        skills = [
            AgentSkill(**s) if isinstance(s, dict) else s
            for s in data.get("skills", [])
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            url=data["url"],
            version=data.get("version", "1.0.0"),
            skills=skills
        )


@dataclass
class A2ATask:
    """An A2A task with its current state."""
    id: str
    context_id: str
    state: TaskState
    message: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2ATask":
        status = data.get("status", {})
        state = TaskState(status.get("state", "submitted"))
        message = None
        if status.get("message"):
            parts = status["message"].get("parts", [])
            message = "\n".join(
                p.get("text", "") for p in parts
                if p.get("kind") == "text" or p.get("type") == "text"
            )
        
        return cls(
            id=data["id"],
            context_id=data.get("contextId") or data.get("sessionId", ""),
            state=state,
            message=message,
            artifacts=data.get("artifacts", []),
            metadata=data.get("metadata", {})
        )


class A2AClient:
    """
    Async client for Google's A2A Protocol.
    
    Supports:
    - Agent discovery via Agent Cards
    - Task submission (send and wait)
    - Streaming task updates (SSE)
    - Task cancellation
    """
    
    def __init__(self, base_url: str = A2A_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "A2AClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("A2AClient must be used as async context manager")
        return self._client
    
    # =========================================================================
    # DISCOVERY
    # =========================================================================
    
    async def get_master_card(self) -> AgentCard:
        """Get the master Agent Card for AitherOS (v0.3.0 path first, fallback to legacy)."""
        try:
            response = await self.client.get(f"{self.base_url}/.well-known/agent-card.json")
            response.raise_for_status()
            return AgentCard.from_dict(response.json())
        except httpx.HTTPStatusError:
            # Fallback to legacy path
            response = await self.client.get(f"{self.base_url}/.well-known/agent.json")
            response.raise_for_status()
            return AgentCard.from_dict(response.json())
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents."""
        response = await self.client.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json().get("agents", [])
    
    async def get_agent_card(self, agent_id: str) -> AgentCard:
        """Get the Agent Card for a specific agent."""
        response = await self.client.get(
            f"{self.base_url}/agents/{agent_id}/.well-known/agent.json"
        )
        response.raise_for_status()
        return AgentCard.from_dict(response.json())
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed info about a specific agent."""
        response = await self.client.get(f"{self.base_url}/agents/{agent_id}")
        response.raise_for_status()
        return response.json()
    
    # =========================================================================
    # TASK MANAGEMENT (JSON-RPC)
    # =========================================================================
    
    async def _rpc(self, method: str, params: Dict[str, Any] = None) -> Any:
        """Execute a JSON-RPC 2.0 call."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": str(id(self))
        }
        response = await self.client.post(f"{self.base_url}/rpc", json=request)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result and result["error"]:
            raise Exception(f"A2A RPC Error: {result['error']}")
        return result.get("result")
    
    async def send_task(
        self,
        agent: str,
        message: str,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None,  # legacy param, maps to context_id
        wait: bool = True
    ) -> A2ATask:
        """
        Send a message to an agent (v0.3.0: message/send).
        
        Args:
            agent: Target agent ID (e.g., "aither", "terra", "aeon")
            message: The message/prompt to send
            context_id: Context ID for conversation grouping
            session_id: Deprecated alias for context_id
            wait: If True, poll until task completes
            
        Returns:
            A2ATask with the result
        """
        ctx_id = context_id or session_id
        msg_obj = {
            "role": "user",
            "parts": [{"kind": "text", "text": message}]
        }
        if ctx_id:
            msg_obj["contextId"] = ctx_id
        
        result = await self._rpc("message/send", {
            "agent": agent,
            "message": msg_obj,
        })
        
        task = A2ATask.from_dict(result)
        
        if wait:
            while task.state in [TaskState.SUBMITTED, TaskState.WORKING]:
                await asyncio.sleep(1)
                task = await self.get_task(task.id)
        
        return task
    
    async def get_task(self, task_id: str) -> A2ATask:
        """Get the current status of a task."""
        result = await self._rpc("tasks/get", {"id": task_id})
        return A2ATask.from_dict(result)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        result = await self._rpc("tasks/cancel", {"id": task_id})
        return result.get("success", False)
    
    # =========================================================================
    # STREAMING (SSE)
    # =========================================================================
    
    async def send_task_streaming(
        self,
        agent: str,
        message: str,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None  # legacy alias
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message and stream updates via SSE.
        
        Yields:
            Dict updates with 'type' or v0.3.0 event kinds
        """
        ctx_id = context_id or session_id
        msg_obj = {
            "role": "user",
            "parts": [{"kind": "text", "text": message}]
        }
        if ctx_id:
            msg_obj["contextId"] = ctx_id
        
        request = {
            "jsonrpc": "2.0",
            "method": "message/stream",
            "params": {
                "agent": agent,
                "message": msg_obj,
            },
            "id": str(id(self))
        }
        
        async with self.client.stream("POST", f"{self.base_url}/tasks/sendSubscribe", json=request) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                        
                        # Check for terminal state (v0.3.0 final flag or legacy type)
                        if data.get("final") is True:
                            return
                        if data.get("type") == "status":
                            state = data.get("task", {}).get("status", {}).get("state")
                            if state in ["completed", "canceled", "failed", "rejected"]:
                                return
                    except json.JSONDecodeError:
                        continue
    
    async def subscribe_to_task(self, task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to updates for an existing task."""
        async with self.client.stream("GET", f"{self.base_url}/tasks/{task_id}/subscribe") as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
    
    # =========================================================================
    # REST CONVENIENCE METHODS
    # =========================================================================
    
    async def quick_task(self, agent: str, prompt: str) -> str:
        """Send a task and return just the text response."""
        task = await self.send_task(agent, prompt, wait=True)
        return task.message or ""
    
    async def ask_aeon(self, prompt: str) -> str:
        """Ask the full Aither Aeon (multi-agent discussion)."""
        return await self.quick_task("aeon", prompt)

    # Backward compatibility alias
    ask_council = ask_aeon
    
    async def ask_aither(self, prompt: str) -> str:
        """Ask Aither directly."""
        return await self.quick_task("aither", prompt)
    
    async def check_infrastructure(self, query: str = "What is the system status?") -> str:
        """Ask Terra about infrastructure."""
        return await self.quick_task("terra", query)
    
    async def code_review(self, code_or_question: str) -> str:
        """Ask Hydra for code review or development guidance."""
        return await self.quick_task("hydra", code_or_question)
    
    async def security_check(self, query: str) -> str:
        """Ask Ignis for security analysis."""
        return await self.quick_task("ignis", query)
    
    async def network_analysis(self, query: str) -> str:
        """Ask Aeros for network/API analysis."""
        return await self.quick_task("aeros", query)
    
    async def data_analysis(self, query: str) -> str:
        """Ask Gluttony for data analysis."""
        return await self.quick_task("gluttony", query)


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

async def discover_agents(base_url: str = A2A_BASE_URL) -> List[Dict[str, Any]]:
    """Discover all available A2A agents."""
    async with A2AClient(base_url) as client:
        return await client.list_agents()


async def send_to_agent(
    agent: str,
    message: str,
    base_url: str = A2A_BASE_URL
) -> str:
    """Send a message to an agent and get the response."""
    async with A2AClient(base_url) as client:
        return await client.quick_task(agent, message)


async def ask_aeon(prompt: str, base_url: str = A2A_BASE_URL) -> str:
    """Quick way to ask the Aither Aeon."""
    async with A2AClient(base_url) as client:
        return await client.ask_aeon(prompt)


# Backward compatibility alias
ask_council = ask_aeon


# =========================================================================
# CLI TESTING
# =========================================================================

if __name__ == "__main__":
    import sys
    
    async def main():
        print("[LINK] AitherA2A Client Test")
        print("=" * 50)
        
        try:
            async with A2AClient() as client:
                # Discover
                print("\n Discovering agents...")
                agents = await client.list_agents()
                for agent in agents:
                    print(f"  * {agent['name']} ({agent['id']})")
                    print(f"    {agent['description'][:60]}...")
                
                # Test task
                if len(sys.argv) > 1:
                    prompt = " ".join(sys.argv[1:])
                    print(f"\n[MSG] Asking Aither: {prompt}")
                    
                    response = await client.ask_aither(prompt)
                    print(f"\n Response:\n{response}")
                else:
                    print("\n[TIP] Usage: python a2a_client.py <your question>")
                
        except httpx.ConnectError:
            print("[FAIL] Cannot connect to A2A Gateway at http://localhost:8766")
            print("   Start it with: python AitherA2A.py")
    
    asyncio.run(main())

