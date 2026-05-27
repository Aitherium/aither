"""
A2A Federation Client - Connect AitherOS to External A2A Agents
================================================================

This module enables AitherOS to federate with external A2A-compatible agents
across the internet. Any agent implementing Google's A2A protocol can be
discovered and communicated with.

Use Cases:
- Connect to external AI services (GPT-4 agents, Claude agents, etc.)
- Federate with other AitherOS instances
- Access specialized A2A services (translation, search, etc.)
- Build multi-organization agent networks

Security:
- All external connections should use HTTPS in production
- API key authentication supported per Google A2A spec
- Rate limiting recommended for external calls

Example Usage:
    from aither_adk.communication.a2a_federation import (
        FederatedA2AClient, discover_external_agent, federate_task
    )
    
    # Discover an external agent
    agent_card = await discover_external_agent("https://external-ai.com")
    
    # Send a task to external agent
    result = await federate_task(
        agent_url="https://external-ai.com",
        message="Translate this to French: Hello world"
    )
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime
from enum import Enum

import httpx

logger = logging.getLogger("A2AFederation")


class FederationStatus(str, Enum):
    """Status of a federated connection."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    AUTHENTICATION_FAILED = "auth_failed"


@dataclass
class ExternalAgentCard:
    """Agent Card from an external A2A service."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocol_version: str = "0.3.0"
    skills: List[Dict[str, Any]] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    authentication: Optional[Dict[str, Any]] = None
    discovered_at: Optional[datetime] = None
    status: FederationStatus = FederationStatus.UNKNOWN
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], url: str) -> "ExternalAgentCard":
        return cls(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            url=url,
            version=data.get("version", "1.0.0"),
            protocol_version=data.get("protocolVersion", "0.3.0"),
            skills=data.get("skills", []),
            capabilities=data.get("capabilities", {}),
            authentication=data.get("authentication"),
            discovered_at=datetime.now(),
            status=FederationStatus.HEALTHY
        )


@dataclass 
class FederatedTask:
    """A task sent to a federated agent."""
    id: str
    agent_url: str
    message: str
    status: str = "submitted"
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class FederatedA2AClient:
    """
    Client for communicating with external A2A-compatible agents.
    
    This enables AitherOS to federate with:
    - Other AitherOS instances
    - Third-party A2A agents
    - Cloud AI services implementing A2A
    
    Usage:
        async with FederatedA2AClient() as client:
            # Discover external agent
            card = await client.discover("https://external-ai.com")
            
            # Send task
            result = await client.send_task(
                "https://external-ai.com",
                "What's the weather in Tokyo?"
            )
    """
    
    def __init__(
        self,
        timeout: float = 120.0,
        api_key: str = None,
        verify_ssl: bool = True
    ):
        self.timeout = timeout
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self._client: Optional[httpx.AsyncClient] = None
        self._discovered_agents: Dict[str, ExternalAgentCard] = {}
    
    async def __aenter__(self) -> "FederatedA2AClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("FederatedA2AClient must be used as async context manager")
        return self._client
    
    def _get_headers(self, agent_url: str = None) -> Dict[str, str]:
        """Get headers including authentication if configured."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AitherOS/1.0 A2A-Federation"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Check if we have agent-specific auth
        if agent_url and agent_url in self._discovered_agents:
            agent = self._discovered_agents[agent_url]
            if agent.authentication:
                auth_type = agent.authentication.get("type")
                if auth_type == "apiKey":
                    # Would need to get key from config
                    pass
        
        return headers
    
    # =========================================================================
    # DISCOVERY
    # =========================================================================
    
    async def discover(self, agent_url: str) -> Optional[ExternalAgentCard]:
        """
        Discover an external A2A agent by fetching its Agent Card.
        
        Args:
            agent_url: Base URL of the external agent
            
        Returns:
            ExternalAgentCard if discovery successful, None otherwise
        """
        agent_url = agent_url.rstrip("/")
        
        try:
            # Fetch agent-card.json from well-known location (v0.3.0 canonical path)
            response = await self.client.get(
                f"{agent_url}/.well-known/agent-card.json",
                headers=self._get_headers()
            )
            
            # Fallback to legacy path
            if response.status_code == 404:
                response = await self.client.get(
                    f"{agent_url}/.well-known/agent.json",
                    headers=self._get_headers()
                )
            
            if response.status_code == 200:
                data = response.json()
                card = ExternalAgentCard.from_dict(data, agent_url)
                self._discovered_agents[agent_url] = card
                logger.info(f"Discovered external agent: {card.name} at {agent_url}")
                return card
            else:
                logger.warning(f"Discovery failed for {agent_url}: {response.status_code}")
                return None
                
        except httpx.ConnectError as e:
            logger.warning(f"Cannot connect to {agent_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Discovery error for {agent_url}: {e}")
            return None
    
    async def discover_multiple(self, agent_urls: List[str]) -> Dict[str, ExternalAgentCard]:
        """
        Discover multiple external agents in parallel.
        
        Args:
            agent_urls: List of agent base URLs
            
        Returns:
            Dict of url -> ExternalAgentCard for successful discoveries
        """
        tasks = [self.discover(url) for url in agent_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        discovered = {}
        for url, result in zip(agent_urls, results):
            if isinstance(result, ExternalAgentCard):
                discovered[url] = result
        
        return discovered
    
    async def check_health(self, agent_url: str) -> FederationStatus:
        """
        Check the health of an external agent.
        
        Args:
            agent_url: Base URL of the external agent
            
        Returns:
            FederationStatus
        """
        agent_url = agent_url.rstrip("/")
        
        try:
            response = await self.client.get(
                f"{agent_url}/health",
                headers=self._get_headers(),
                timeout=5.0
            )
            
            if response.status_code == 200:
                return FederationStatus.HEALTHY
            elif response.status_code == 401:
                return FederationStatus.AUTHENTICATION_FAILED
            elif response.status_code >= 500:
                return FederationStatus.DEGRADED
            else:
                return FederationStatus.UNKNOWN
                
        except httpx.ConnectError:
            return FederationStatus.OFFLINE
        except Exception:
            return FederationStatus.UNKNOWN
    
    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================
    
    async def send_task(
        self,
        agent_url: str,
        message: str,
        skill: str = None,
        session_id: str = None,
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Send a task to an external A2A agent.
        
        Args:
            agent_url: Base URL of the external agent
            message: The message/prompt to send
            skill: Optional specific skill to invoke
            session_id: Optional session ID for continuity
            wait: If True, poll until task completes
            
        Returns:
            Task result dict
        """
        agent_url = agent_url.rstrip("/")
        
        # Build JSON-RPC request per A2A v0.3.0 spec
        msg_obj = {
            "role": "user",
            "parts": [{"kind": "text", "text": message}]
        }
        if session_id:
            msg_obj["contextId"] = session_id
        
        request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": msg_obj,
            },
            "id": f"aitheros-{datetime.now().timestamp()}"
        }
        
        if skill:
            request["params"]["skill"] = skill
        
        try:
            # Send to RPC endpoint
            response = await self.client.post(
                f"{agent_url}/rpc",
                json=request,
                headers=self._get_headers(agent_url)
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"External agent returned {response.status_code}",
                    "agent_url": agent_url
                }
            
            result = response.json()
            
            if "error" in result and result["error"]:
                return {
                    "success": False,
                    "error": result["error"].get("message", "Unknown error"),
                    "agent_url": agent_url
                }
            
            task_data = result.get("result", {})
            task_id = task_data.get("id")
            
            # If waiting, poll for completion
            if wait and task_id:
                task_data = await self._poll_task(agent_url, task_id)
            
            # Extract response text
            response_text = ""
            status = task_data.get("status", {})
            if isinstance(status, dict):
                message_obj = status.get("message", {})
                parts = message_obj.get("parts", [])
                for part in parts:
                    if part.get("kind") == "text" or part.get("type") == "text":
                        response_text += part.get("text", "")
            
            return {
                "success": True,
                "response": response_text,
                "task_id": task_id,
                "status": status.get("state", "completed") if isinstance(status, dict) else "completed",
                "agent_url": agent_url,
                "raw": task_data
            }
            
        except httpx.ConnectError as e:
            return {
                "success": False,
                "error": f"Cannot connect to {agent_url}: {e}",
                "agent_url": agent_url
            }
        except Exception as e:
            logger.error(f"Task send error: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_url": agent_url
            }
    
    async def _poll_task(
        self,
        agent_url: str,
        task_id: str,
        max_wait: float = 120.0,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Poll for task completion."""
        start = datetime.now()
        
        while True:
            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > max_wait:
                return {"status": {"state": "timeout"}}
            
            try:
                response = await self.client.post(
                    f"{agent_url}/rpc",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tasks/get",
                        "params": {"id": task_id},
                        "id": f"poll-{task_id}"
                    },
                    headers=self._get_headers(agent_url)
                )
                
                if response.status_code == 200:
                    result = response.json().get("result", {})
                    state = result.get("status", {}).get("state", "")
                    
                    if state in ["completed", "failed", "canceled", "rejected"]:
                        return result
                
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            
            await asyncio.sleep(poll_interval)
    
    # =========================================================================
    # STREAMING
    # =========================================================================
    
    async def send_task_streaming(
        self,
        agent_url: str,
        message: str,
        skill: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a task and stream responses via SSE.
        
        Yields:
            Update dicts with type ('initial', 'status', 'artifact')
        """
        agent_url = agent_url.rstrip("/")
        
        request = {
            "jsonrpc": "2.0",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}]
                }
            },
            "id": f"stream-{datetime.now().timestamp()}"
        }
        
        if skill:
            request["params"]["skill"] = skill
        
        try:
            async with self.client.stream(
                "POST",
                f"{agent_url}/tasks/sendSubscribe",
                json=request,
                headers=self._get_headers(agent_url)
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            yield data
                            
                            # Check for terminal state (v0.3.0 final flag or legacy)
                            if data.get("final") is True:
                                return
                            if data.get("type") == "status":
                                state = data.get("task", {}).get("status", {}).get("state")
                                if state in ["completed", "canceled", "failed", "rejected"]:
                                    return
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield {"type": "error", "error": str(e)}
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    async def quick_ask(self, agent_url: str, question: str) -> str:
        """
        Quick way to ask an external agent a question.
        
        Args:
            agent_url: Base URL of the external agent
            question: The question to ask
            
        Returns:
            Response text (empty string on failure)
        """
        result = await self.send_task(agent_url, question, wait=True)
        return result.get("response", "") if result.get("success") else ""
    
    async def register_with_peer(self, peer_url: str) -> Dict[str, Any]:
        """
        Register AitherOS with an external A2A peer (mutual discovery).
        
        Calls POST /federation/register on the peer, providing our agent
        card URL. The peer stores us and returns their own agent card.
        
        Args:
            peer_url: Base URL of the external peer
            
        Returns:
            Registration response from the peer
        """
        from lib.core.AitherPorts import get_service_url
        our_url = get_service_url("A2A")
        
        try:
            response = await self.client.post(
                f"{peer_url.rstrip('/')}/federation/register",
                json={
                    "agent_card_url": f"{our_url}/.well-known/agent-card.json",
                    "name": "AitherOS",
                },
                headers=self._get_headers(peer_url),
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[FED] Registered with peer: {peer_url}")
                
                # Also discover them if not already
                if peer_url not in self._discovered_agents:
                    await self.discover(peer_url)
                
                return {"success": True, **result}
            else:
                logger.warning(f"[FED] Registration failed with {peer_url}: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"[FED] Cannot register with {peer_url}: {e}")
            return {"success": False, "error": str(e)}
    
    async def negotiate_workflow(
        self,
        peer_url: str,
        steps: List[Dict[str, Any]],
        requirements: Dict[str, Any] = None,
        name: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Propose a workflow negotiation with an external peer.
        
        Args:
            peer_url: Base URL of the peer
            steps: Workflow steps (skill, params, etc.)
            requirements: Constraints (max_latency_ms, gpu_required, etc.)
            name: Workflow name
            description: What this workflow does
            
        Returns:
            Negotiation response (accepted/rejected/countered)
        """
        try:
            response = await self.client.post(
                f"{peer_url.rstrip('/')}/federation/negotiate",
                json={
                    "name": name,
                    "description": description,
                    "steps": steps,
                    "requirements": requirements or {},
                    "proposer": "AitherOS",
                },
                headers=self._get_headers(peer_url),
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[FED] Workflow negotiation with {peer_url}: {result.get('response', {}).get('status', 'unknown')}")
                return {"success": True, **result}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"[FED] Workflow negotiation failed with {peer_url}: {e}")
            return {"success": False, "error": str(e)}

    def list_discovered(self) -> List[ExternalAgentCard]:
        """List all discovered external agents."""
        return list(self._discovered_agents.values())


# ============================================================================
# GLOBAL REGISTRY - Track known external A2A endpoints
# ============================================================================

# Default registry of known A2A-compatible services
# Add your external agents here
KNOWN_A2A_ENDPOINTS = {
    # Example entries (commented out - add your own):
    # "example_gpt": "https://api.example.com/a2a",
    # "translation_service": "https://translate.example.com",
}

# Global client instance
_federation_client: Optional[FederatedA2AClient] = None


async def get_federation_client() -> FederatedA2AClient:
    """Get the global federation client (creates if needed)."""
    global _federation_client
    if _federation_client is None:
        _federation_client = FederatedA2AClient()
        await _federation_client.__aenter__()
    return _federation_client


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def discover_external_agent(agent_url: str) -> Optional[ExternalAgentCard]:
    """
    Discover an external A2A agent.
    
    Example:
        card = await discover_external_agent("https://ai-service.com")
        if card:
            print(f"Found: {card.name} with {len(card.skills)} skills")
    """
    client = await get_federation_client()
    return await client.discover(agent_url)


async def federate_task(
    agent_url: str,
    message: str,
    skill: str = None,
    wait: bool = True
) -> Dict[str, Any]:
    """
    Send a task to a federated external agent.
    
    Example:
        result = await federate_task(
            "https://translator.example.com",
            "Translate 'Hello' to Japanese"
        )
        print(result["response"])  # "こんにちは"
    """
    client = await get_federation_client()
    return await client.send_task(agent_url, message, skill=skill, wait=wait)


async def quick_external_ask(agent_url: str, question: str) -> str:
    """
    Quick way to ask an external agent a question.
    
    Example:
        answer = await quick_external_ask("https://ai.example.com", "What is 2+2?")
    """
    client = await get_federation_client()
    return await client.quick_ask(agent_url, question)


async def discover_and_list_skills(agent_url: str) -> Dict[str, List[str]]:
    """
    Discover an agent and list its skills.
    
    Returns:
        Dict with agent info and skill list
    """
    card = await discover_external_agent(agent_url)
    if not card:
        return {"error": "Discovery failed", "agent_url": agent_url}
    
    skill_ids = []
    for skill in card.skills:
        if isinstance(skill, dict):
            skill_ids.append(skill.get("id", skill.get("name", "unknown")))
        else:
            skill_ids.append(str(skill))
    
    return {
        "name": card.name,
        "description": card.description,
        "skills": skill_ids,
        "status": card.status.value,
        "agent_url": agent_url
    }


async def register_with_peer(peer_url: str) -> Dict[str, Any]:
    """
    Register AitherOS with an external A2A peer (mutual discovery).
    
    Example:
        result = await register_with_peer("https://other-agent.example.com")
        if result["success"]:
            print(f"Registered! Peer name: {result.get('registered', {}).get('name')}")
    """
    client = await get_federation_client()
    return await client.register_with_peer(peer_url)


async def negotiate_workflow_with_peer(
    peer_url: str,
    steps: List[Dict[str, Any]],
    requirements: Dict[str, Any] = None,
    name: str = "",
) -> Dict[str, Any]:
    """
    Propose a workflow negotiation with an external peer.
    
    Example:
        result = await negotiate_workflow_with_peer(
            "https://translator.example.com",
            steps=[{"skill": "translate", "params": {"to": "ja"}}],
            name="translate_pipeline",
        )
    """
    client = await get_federation_client()
    return await client.negotiate_workflow(peer_url, steps, requirements, name)


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    async def test_federation():
        print("[WEB] A2A Federation Client Test")
        print("=" * 60)
        
        # Test with local Aither Gateway (A2A)
        local_url = "http://localhost:8766"
        
        async with FederatedA2AClient() as client:
            # Discover local Aither Gateway
            print(f"\n Discovering {local_url}...")
            card = await client.discover(local_url)
            
            if card:
                print(f"  [DONE] Found: {card.name}")
                print(f"  [NOTE] Description: {card.description[:60]}...")
                print(f"  [TOOL] Skills: {len(card.skills)}")
                for skill in card.skills[:3]:
                    if isinstance(skill, dict):
                        print(f"     - {skill.get('id', skill.get('name'))}")
                
                # Test sending a task
                print(f"\n[MSG] Sending test task...")
                result = await client.send_task(
                    local_url,
                    "What agents are available?",
                    wait=True
                )
                
                if result.get("success"):
                    print(f"  [DONE] Response: {result['response'][:100]}...")
                else:
                    print(f"  [FAIL] Failed: {result.get('error')}")
            else:
                print(f"  [FAIL] Cannot discover {local_url}")
                print("     Make sure AitherA2A is running (port 8766)")
        
        print("\n[DONE] Federation test complete")
    
    asyncio.run(test_federation())

