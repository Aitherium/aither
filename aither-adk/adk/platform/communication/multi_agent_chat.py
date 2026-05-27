"""
AitherOS Multi-Agent Chat

Natural multi-agent conversations with contextual response decisions.
Supports @mentions, #artifact references, task delegation, and sub-agents.

Features:
  - Agents evaluate if they should respond based on context
  - Direct @mentions force a response
  - Random delays for natural conversation flow
  - Staggered response delivery (not all at once)
  - Aither can delegate tasks to council members as sub-agents

Example:
  @aither @leo what do you think about this?
  -> Both evaluate, may respond with staggered timing

  @aither! urgent task
  -> Aither MUST respond (forced by @name pattern)

  Aither delegating: "Leo, analyze this image"
  -> Leo runs as sub-agent to complete the task
"""

import json
import logging
import os
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Response timing (natural conversation delays)
AGENT_EVAL_DELAY_MIN = float(os.getenv("AGENT_EVAL_DELAY_MIN", "0.5"))  # Min seconds before evaluating
AGENT_EVAL_DELAY_MAX = float(os.getenv("AGENT_EVAL_DELAY_MAX", "2.0"))  # Max seconds before evaluating
AGENT_RESPONSE_DELAY_MIN = float(os.getenv("AGENT_RESPONSE_DELAY_MIN", "0.3"))  # Min delay between responses
AGENT_RESPONSE_DELAY_MAX = float(os.getenv("AGENT_RESPONSE_DELAY_MAX", "1.5"))  # Max delay between responses

# Agent behavior
AGENT_BASE_SKIP_PROBABILITY = float(os.getenv("AGENT_SKIP_PROBABILITY", "0.20"))  # Base 20% skip chance
AGENT_MIN_RESPONDERS = int(os.getenv("AGENT_MIN_RESPONDERS", "1"))  # At least 1 agent must respond
AGENT_CHIME_IN_PROBABILITY = float(os.getenv("AGENT_CHIME_IN_PROBABILITY", "0.15"))  # 15% chance non-mentioned agents join

# Delegation
AITHER_CAN_DELEGATE = True  # Aither can spawn sub-agents for tasks


# Path to personas
AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NARRATIVE_AGENT_DIR = os.path.join(AGENT_DIR, "Saga")
PERSONAS_DIR = os.path.join(NARRATIVE_AGENT_DIR, "config", "personas")


# ============================================================================
# RESPONSE QUEUE FOR NATURAL DELIVERY
# ============================================================================

@dataclass
class QueuedResponse:
    """A response waiting to be delivered."""
    agent: str
    response: str
    delay: float  # Seconds to wait before delivery
    timestamp: datetime = field(default_factory=datetime.now)
    mentions: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    is_delegation: bool = False
    delegated_to: Optional[str] = None


class ResponseQueue:
    """
    Queue for staggered response delivery.
    Responses don't all appear at once - they come in naturally.
    """

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._responses: List[QueuedResponse] = []
        self._lock = threading.Lock()

    def add(self, response: QueuedResponse):
        """Add a response to the queue with its delivery delay."""
        delivery_time = time.time() + response.delay
        with self._lock:
            self._queue.put((delivery_time, response))
            self._responses.append(response)

    def get_ready_responses(self) -> List[QueuedResponse]:
        """Get all responses that are ready to be delivered."""
        ready = []
        current_time = time.time()

        with self._lock:
            temp_queue = queue.PriorityQueue()

            while not self._queue.empty():
                delivery_time, response = self._queue.get()
                if delivery_time <= current_time:
                    ready.append(response)
                else:
                    temp_queue.put((delivery_time, response))

            self._queue = temp_queue

        return ready

    def wait_and_get_all(self, timeout: float = 60.0) -> List[QueuedResponse]:
        """Wait for all queued responses to be ready, then return them in order."""
        start_time = time.time()
        all_responses = []

        while time.time() - start_time < timeout:
            ready = self.get_ready_responses()
            all_responses.extend(ready)

            with self._lock:
                if self._queue.empty():
                    break

            time.sleep(0.1)  # Check every 100ms

        return sorted(all_responses, key=lambda r: r.timestamp)


# Global response queue
_response_queue = ResponseQueue()


def load_persona(name: str) -> Optional[Dict[str, Any]]:
    """Load a persona from YAML."""
    try:
        import yaml
        path = os.path.join(PERSONAS_DIR, f"{name.lower()}.yaml")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as exc:
        logger.debug(f"Persona load failed: {exc}")
    return None


def get_random_eval_delay() -> float:
    """Get a random delay before an agent evaluates whether to respond."""
    return random.uniform(AGENT_EVAL_DELAY_MIN, AGENT_EVAL_DELAY_MAX)


def get_random_response_delay() -> float:
    """Get a random delay between response deliveries for natural flow."""
    return random.uniform(AGENT_RESPONSE_DELAY_MIN, AGENT_RESPONSE_DELAY_MAX)


def is_direct_mention(message: str, agent_name: str) -> bool:
    """
    Check if an agent is directly/forcefully mentioned.

    Direct mentions that FORCE a response:
    - @agent! (exclamation = urgent)
    - @agent: (colon = addressing directly)
    - Message starts with @agent
    - Only one agent mentioned (sole addressee)
    """
    agent_lower = agent_name.lower()
    msg_lower = message.lower()

    # Check for urgent/direct patterns
    if f"@{agent_lower}!" in msg_lower:
        return True
    if f"@{agent_lower}:" in msg_lower:
        return True
    if f"@{agent_lower}," in msg_lower:
        return True

    # Check if message starts with this agent's mention
    if msg_lower.strip().startswith(f"@{agent_lower}"):
        return True

    # Check if this is the only agent mentioned
    mentions = re.findall(r"@(\w+)", msg_lower)
    agent_mentions = [m for m in mentions if m not in ["admin", "user", "human"]]
    if len(agent_mentions) == 1 and agent_mentions[0] == agent_lower:
        return True

    return False


def evaluate_should_respond(
    agent_name: str,
    message: str,
    conversation_context: str,
    persona_data: Dict[str, Any],
    total_agents: int,
    responding_count: int,
    model: str = "aither-orchestrator-8b-v4"
) -> Tuple[bool, str]:
    """
    Have the agent evaluate whether they should respond to this message.

    The agent considers:
    - Is this relevant to my expertise/personality?
    - Am I directly addressed?
    - Would my input add value?
    - Should I let others speak first?

    Args:
        agent_name: The agent's name
        message: The user's message
        conversation_context: Recent conversation history
        persona_data: The agent's persona configuration
        total_agents: How many agents are in this conversation
        responding_count: How many have already committed to respond
        model: LLM model for evaluation

    Returns:
        Tuple of (should_respond: bool, reason: str)
    """
    # Direct mentions ALWAYS respond
    if is_direct_mention(message, agent_name):
        return True, "directly addressed"

    # If we need responders to meet minimum, respond
    if total_agents - responding_count <= AGENT_MIN_RESPONDERS:
        return True, "ensuring minimum responders"

    # Add random thinking delay before evaluation
    time.sleep(get_random_eval_delay())

    # Build evaluation prompt
    persona_instruction = extract_persona_instruction(persona_data)

    eval_prompt = f"""You are {agent_name.title()}. Based on your personality and expertise, evaluate if you should respond to this message.

YOUR PERSONALITY:
{persona_instruction[:500]}

CONVERSATION CONTEXT:
{conversation_context[-500:] if conversation_context else "(start of conversation)"}

NEW MESSAGE: {message}

OTHER AGENTS IN CHAT: {total_agents - 1} others
AGENTS ALREADY RESPONDING: {responding_count}

Consider:
1. Is this relevant to your expertise or interests?
2. Would your perspective add unique value?
3. Are others better suited to answer?
4. Should you observe first and respond later?

Respond with ONLY one of:
- RESPOND: [brief reason why you should speak]
- SKIP: [brief reason to stay quiet]

Your decision:"""

    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": eval_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50,
                }
            },
            timeout=15
        )

        if response.status_code == 200:
            decision = response.json().get("response", "").strip().upper()

            if decision.startswith("RESPOND"):
                reason = decision.replace("RESPOND:", "").replace("RESPOND", "").strip()
                return True, reason or "decided to contribute"
            elif decision.startswith("SKIP"):
                reason = decision.replace("SKIP:", "").replace("SKIP", "").strip()
                return False, reason or "letting others speak"
            else:
                # Unclear response - use probability fallback
                if random.random() < AGENT_BASE_SKIP_PROBABILITY:
                    return False, "uncertain, stepping back"
                return True, "uncertain, contributing anyway"
    except Exception as e:
        # On error, fall back to random skip with base probability
        if random.random() < AGENT_BASE_SKIP_PROBABILITY:
            return False, f"evaluation error, skipping: {str(e)[:30]}"
        return True, "evaluation error, responding anyway"

    return True, "default response"


def infer_best_agent(message: str, available_agents: List[str] = None) -> str:
    """
    Infer the best agent to respond based on message content.
    Used when no @mentions are present.

    Returns the agent name most likely to be relevant.
    """
    msg_lower = message.lower()

    # Keyword -> Agent mapping (order matters - first match wins)
    agent_keywords = {
        "terra": ["infrastructure", "terraform", "opentofu", "vm", "server", "deploy", "provision", "state", "foundation"],
        "hydra": ["code", "coding", "git", "pipeline", "ci/cd", "commit", "branch", "merge", "devops", "flow"],
        "ignis": ["security", "vulnerability", "debug", "error", "log", "trace", "root cause", "hardening", "firewall"],
        "aeros": ["network", "dns", "http", "api", "route", "connect", "ssh", "protocol", "message"],
        "leo": ["strategy", "plan", "roadmap", "priority", "lead", "decision", "direction"],
        "maya": ["research", "analyze", "study", "investigate", "learn", "data", "insight"],
        "rex": ["technical", "implement", "build", "engineer", "architecture", "system"],
        "nova": ["creative", "design", "idea", "innovate", "new", "concept", "imagine"],
    }

    # Check for keyword matches
    for agent, keywords in agent_keywords.items():
        if available_agents and agent not in available_agents:
            continue
        for keyword in keywords:
            if keyword in msg_lower:
                return agent

    # Default to Aither (the orchestrator) for general questions
    if available_agents and "aither" in available_agents:
        return "aither"
    elif available_agents:
        return available_agents[0]
    else:
        return "aither"


def should_chime_in(agent_name: str, message: str, mentioned_agents: List[str]) -> Tuple[bool, str]:
    """
    Determine if a non-mentioned agent should spontaneously join the conversation.

    Uses a combination of:
    1. Base probability (AGENT_CHIME_IN_PROBABILITY)
    2. Relevance to the agent's expertise

    Returns:
        Tuple of (should_join, reason)
    """
    # Never chime in if already mentioned
    if agent_name.lower() in [a.lower() for a in mentioned_agents]:
        return False, "already mentioned"

    # Base random chance
    if random.random() > AGENT_CHIME_IN_PROBABILITY:
        return False, "random skip"

    # Check if message is relevant to this agent's domain
    msg_lower = message.lower()

    agent_domains = {
        "terra": ["infrastructure", "terraform", "vm", "server", "deploy"],
        "hydra": ["code", "git", "pipeline", "ci/cd", "devops"],
        "ignis": ["security", "vulnerability", "debug", "error"],
        "aeros": ["network", "dns", "http", "api", "ssh"],
        "aither": ["orchestrate", "coordinate", "balance", "overview"],
        "leo": ["strategy", "plan", "priority", "lead"],
        "maya": ["research", "analyze", "data", "insight"],
        "rex": ["technical", "build", "implement", "system"],
        "nova": ["creative", "design", "idea", "new"],
    }

    # Check relevance
    domain_keywords = agent_domains.get(agent_name.lower(), [])
    relevance_score = sum(1 for kw in domain_keywords if kw in msg_lower)

    if relevance_score > 0:
        return True, f"relevant to my expertise ({relevance_score} keyword matches)"

    # Small chance to join anyway (curious agents)
    if random.random() < 0.05:  # 5% curiosity factor
        return True, "curious about this topic"

    return False, "not relevant to my domain"


def should_skip_turn(agent_name: str, total_agents: int, responding_count: int) -> bool:
    """
    Simple random skip (legacy function for backward compatibility).
    Prefer evaluate_should_respond() for smarter decisions.
    """
    if total_agents - responding_count <= AGENT_MIN_RESPONDERS:
        return False

    if random.random() < AGENT_BASE_SKIP_PROBABILITY:
        return True

    return False


def get_skip_reason() -> str:
    """Get a random in-character reason for skipping."""
    reasons = [
        "busy with other tasks",
        "letting others speak first",
        "thinking before responding",
        "observing the conversation",
        "processing information",
        "waiting for the right moment",
        "this isn't my area of expertise",
        "others have it covered",
    ]
    return random.choice(reasons)


def extract_persona_instruction(persona_data: Dict[str, Any]) -> str:
    """Extract the core instruction from persona data."""
    instruction = persona_data.get("instruction", "")

    # Get key sections
    if "[PERSONA & BEHAVIOR]" in instruction:
        # Extract behavior section
        parts = instruction.split("[PERSONA & BEHAVIOR]")
        if len(parts) > 1:
            behavior = parts[1]
            if "[" in behavior:
                behavior = behavior.split("[")[0]
            return behavior.strip()[:1500]

    return instruction[:1500]


def parse_artifact_references(message: str) -> List[str]:
    """Extract #artifact references from a message."""
    return re.findall(r'#([a-f0-9]{8})', message.lower())


def resolve_artifact_context(artifact_ids: List[str]) -> str:
    """
    Resolve artifact IDs to context that can be included in the prompt.
    """
    if not artifact_ids:
        return ""

    try:
        from aither_adk.communication.mailbox import get_artifact_store
        store = get_artifact_store()

        context_parts = []
        for aid in artifact_ids:
            artifact = store.get(aid)
            if artifact:
                if artifact.type.value == "image":
                    context_parts.append(f"[Referenced Image: {artifact.name} at {artifact.path}]")
                elif artifact.type.value == "code":
                    context_parts.append(f"[Referenced Code '{artifact.name}':\n```\n{artifact.content[:500]}...\n```]")
                elif artifact.type.value == "link":
                    context_parts.append(f"[Referenced Link: {artifact.name} - {artifact.url}]")
                elif artifact.type.value == "message":
                    context_parts.append(f"[Referenced Message: {artifact.name}]")
                else:
                    context_parts.append(f"[Referenced Artifact: {artifact.name}]")

        if context_parts:
            return "\n[REFERENCED ARTIFACTS]\n" + "\n".join(context_parts) + "\n"
    except Exception as e:
        print(f"Error resolving artifacts: {e}")

    return ""


# ============================================================================
# DELEGATION SUPPORT (Aither -> Council Sub-Agents)
# ============================================================================

@dataclass
class DelegatedTask:
    """A task delegated from one agent to another."""
    id: str
    from_agent: str
    to_agent: str
    task: str
    context: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class DelegationManager:
    """
    Manages task delegation between agents.
    Primarily used for Aither delegating to council members.
    """

    def __init__(self):
        self._tasks: Dict[str, DelegatedTask] = {}
        self._lock = threading.Lock()

    def delegate(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        context: str = ""
    ) -> DelegatedTask:
        """Create a delegation task."""
        import uuid
        task_id = str(uuid.uuid4())[:8]

        delegated = DelegatedTask(
            id=task_id,
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            context=context
        )

        with self._lock:
            self._tasks[task_id] = delegated

        return delegated

    def execute_delegation(self, task: DelegatedTask) -> str:
        """
        Execute a delegated task as a sub-agent.
        The target agent runs the task and returns results.
        """
        task.status = "running"

        try:
            # Load target agent's persona
            persona_data = load_persona(task.to_agent)
            if not persona_data:
                task.status = "failed"
                task.result = f"Unknown agent: {task.to_agent}"
                return task.result

            persona_instruction = extract_persona_instruction(persona_data)

            # Build sub-agent prompt
            prompt = f"""[SUB-AGENT TASK EXECUTION]
You are {task.to_agent.title()}, executing a task delegated by {task.from_agent.title()}.

YOUR ROLE:
{persona_instruction[:800]}

DELEGATED TASK:
{task.task}

CONTEXT:
{task.context}

Execute this task thoroughly and report your findings/results.
Be specific and actionable in your response.

[{task.to_agent.upper()}'S TASK RESULT]:"""

            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "aither-orchestrator-8b-v4",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 600,
                    }
                },
                timeout=90
            )

            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                task.status = "completed"
                task.result = result
                task.completed_at = datetime.now()
                return result
            else:
                task.status = "failed"
                task.result = f"API error: {response.status_code}"
                return task.result

        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            return task.result

    def get_task(self, task_id: str) -> Optional[DelegatedTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_pending_tasks(self, agent: str) -> List[DelegatedTask]:
        """Get pending tasks for an agent."""
        with self._lock:
            return [t for t in self._tasks.values()
                    if t.to_agent.lower() == agent.lower() and t.status == "pending"]


# Global delegation manager
_delegation_manager = DelegationManager()


def parse_delegation_request(response: str, from_agent: str) -> Optional[Tuple[str, str]]:
    """
    Parse a response to see if it contains a delegation request.

    Delegation patterns:
    - "@leo, please analyze this image"
    - "Leo, I'm delegating the analysis to you"
    - "Delegating to @maya: research this topic"

    Returns:
        Tuple of (target_agent, task) or None
    """
    # Pattern: "Delegating to @agent: task"
    match = re.search(r"[Dd]elegat(?:e|ing) to @?(\w+)[:\s]+(.+?)(?:\.|$)", response)
    if match:
        return match.group(1).lower(), match.group(2).strip()

    # Pattern: "@agent, please/could you [task]"
    match = re.search(r"@(\w+),?\s+(?:please|could you|can you|would you)\s+(.+?)(?:\?|\.|\n|$)", response, re.IGNORECASE)
    if match:
        agent = match.group(1).lower()
        if agent not in ["admin", "user", "human"]:
            return agent, match.group(2).strip()

    return None


def generate_agent_response(
    agent_name: str,
    user_message: str,
    conversation_context: str = "",
    model: str = "aither-orchestrator-8b-v4",
    reply_to: str = "Admin",
    artifact_refs: List[str] = None,
    allow_delegation: bool = True
) -> Dict[str, Any]:
    """
    Generate a response from a specific agent.

    Args:
        agent_name: Name of the agent to respond
        user_message: The user's message
        conversation_context: Previous conversation context
        model: LLM model to use
        reply_to: Who the agent should @mention (default: Admin)
        artifact_refs: List of artifact IDs being referenced
        allow_delegation: Whether this agent can delegate tasks (Aither only)

    Returns:
        Dict with response data including delegation info if applicable
    """
    try:
        # Load persona
        persona_data = load_persona(agent_name)
        if not persona_data:
            return {
                "agent": agent_name,
                "response": None,
                "success": False,
                "error": f"Unknown persona: {agent_name}",
                "mentions": [],
                "artifacts": [],
                "delegation": None
            }

        persona_instruction = extract_persona_instruction(persona_data)

        # Resolve artifact context
        artifact_context = resolve_artifact_context(artifact_refs or [])

        # Build delegation awareness for Aither
        delegation_instruction = ""
        if allow_delegation and agent_name.lower() == "aither" and AITHER_CAN_DELEGATE:
            delegation_instruction = """

[DELEGATION CAPABILITY]
As Aither, you can delegate specific tasks to council members when appropriate:
- Use "@agent, please [task]" to delegate
- Good for: research, analysis, creative tasks, specialized work
- Your council: Leo (strategy), Maya (research), Rex (technical), Nova (creative)
Example: "@maya, please research the history of this topic"
"""

        # Build prompt with @mention instruction
        prompt = f"""[CREATIVE ROLEPLAY - CHARACTER: {agent_name.upper()}]
You are {agent_name.title()}, a unique AI character with your own personality.
Stay in character. Be expressive and engaging. Use *asterisks* for actions.

IMPORTANT: You MUST start your response with "@{reply_to}" to properly address who you're responding to.
If you reference any artifacts (images, files, etc), use the #id format like #abc12345.
{delegation_instruction}

{persona_instruction}

{conversation_context}
{artifact_context}

[USER MESSAGE from {reply_to}]: {user_message}

[{agent_name.upper()}'S RESPONSE (must start with @{reply_to})]:"""

        # Call Ollama
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "2m",
                "options": {
                    "temperature": 0.85,
                    "num_predict": 400,
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            text = response.json().get("response", "").strip()

            # Clean up if it starts with agent name
            if text.upper().startswith(agent_name.upper()):
                text = text.split(":", 1)[-1].strip()

            # Ensure the response starts with the @mention
            if not text.lower().startswith(f"@{reply_to.lower()}"):
                text = f"@{reply_to} {text}"

            # Extract any artifact references in the response
            response_artifacts = parse_artifact_references(text)

            # Check for delegation requests
            delegation = None
            if allow_delegation and agent_name.lower() == "aither":
                delegation_request = parse_delegation_request(text, agent_name)
                if delegation_request:
                    target_agent, task = delegation_request
                    # Create delegation task
                    delegated = _delegation_manager.delegate(
                        from_agent=agent_name,
                        to_agent=target_agent,
                        task=task,
                        context=f"Original message: {user_message}\n\nConversation: {conversation_context[-500:]}"
                    )
                    delegation = {
                        "task_id": delegated.id,
                        "to_agent": target_agent,
                        "task": task,
                        "status": delegated.status
                    }

            return {
                "agent": agent_name,
                "response": text,
                "success": True,
                "error": None,
                "mentions": [reply_to],
                "artifacts": response_artifacts,
                "input_artifacts": artifact_refs or [],
                "delegation": delegation,
                "response_delay": get_random_response_delay()  # For staggered delivery
            }
        else:
            return {
                "agent": agent_name,
                "response": None,
                "success": False,
                "error": f"API error: {response.status_code}",
                "mentions": [],
                "artifacts": [],
                "delegation": None
            }

    except Exception as e:
        return {
            "agent": agent_name,
            "response": None,
            "success": False,
            "error": str(e),
            "mentions": [],
            "artifacts": [],
            "delegation": None
        }


class MultiAgentDispatcher:
    """
    Dispatches messages to multiple agents and collects responses.

    Features:
    - Agents evaluate contextually whether to respond
    - Direct @mentions force responses
    - Staggered response delivery for natural flow
    - Support for task delegation (Aither -> council)
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.response_queue = ResponseQueue()

    def parse_mentions(self, message: str) -> List[str]:
        """Extract @mentioned agents from message."""
        mentions = re.findall(r"@(\w+)", message.lower())

        # Filter to valid personas (exclude admin/user)
        valid = []
        for m in mentions:
            if m.lower() in ["admin", "user", "human"]:
                continue  # Skip user mentions, we track those separately
            if os.path.exists(os.path.join(PERSONAS_DIR, f"{m}.yaml")):
                valid.append(m)

        return list(set(valid))  # Dedupe

    def parse_direct_mentions(self, message: str) -> List[str]:
        """Extract agents that are DIRECTLY mentioned (must respond)."""
        all_mentions = self.parse_mentions(message)
        return [agent for agent in all_mentions if is_direct_mention(message, agent)]

    def parse_artifacts(self, message: str) -> List[str]:
        """Extract #artifact references from message."""
        return parse_artifact_references(message)

    def clean_message(self, message: str) -> str:
        """Remove @mentions from message to get the actual question."""
        return re.sub(r"@\w+\s*", "", message).strip()

    def detect_sender(self, message: str) -> str:
        """
        Detect who sent the message (for proper @mention in response).
        If message contains @admin or is from CLI, return "Admin".
        """
        mentions = re.findall(r"@(\w+)", message.lower())
        for m in mentions:
            if m.lower() in ["admin", "user", "human"]:
                return "Admin"
        return "Admin"

    def dispatch_parallel(
        self,
        message: str,
        agents: List[str] = None,
        conversation_context: str = "",
        callback: Callable[[str, str], None] = None,
        sender: str = "Admin",
        allow_skipping: bool = True,
        smart_evaluation: bool = True,
        staggered_delivery: bool = True,
        require_mention: bool = True,  # Only @mentioned agents respond by default
        allow_chime_in: bool = True     # Allow non-mentioned agents to join
    ) -> List[Dict[str, Any]]:
        """
        Send message to multiple agents with natural response patterns.

        Response Logic:
        1. @mentioned agents MUST respond
        2. If NO @mentions, infer best agent from message content
        3. Non-mentioned agents have small chance to chime in if relevant

        Args:
            message: The user's message
            agents: List of agent names (or None to auto-determine)
            conversation_context: Optional context from conversation
            callback: Optional callback(agent_name, response) called as each completes
            sender: Who sent the message (for @mention in response)
            allow_skipping: Whether mentioned agents can skip turns
            smart_evaluation: Use LLM to decide if agent should respond
            staggered_delivery: Add delays between responses for natural flow
            require_mention: If True, prefer @mentioned agents (default: True)
            allow_chime_in: If True, non-mentioned agents may join (default: True)

        Returns:
            List of response dicts with timing info
        """
        # Get @mentioned agents from message
        mentioned_agents = self.parse_mentions(message)

        # Get all available agents for potential chime-in
        all_available_agents = []
        if allow_chime_in:
            for persona_file in os.listdir(PERSONAS_DIR):
                if persona_file.endswith(".yaml"):
                    all_available_agents.append(persona_file.replace(".yaml", ""))

        # Determine primary responders
        if mentioned_agents:
            # Use mentioned agents as primary responders
            primary_agents = mentioned_agents
        else:
            # No @mentions - infer best agent from message content
            best_agent = infer_best_agent(message, all_available_agents or ["aither"])
            primary_agents = [best_agent]

        # If explicit agents list provided, respect it but filter by mentions if required
        if agents is not None:
            if require_mention and mentioned_agents:
                primary_agents = [a for a in agents if a in mentioned_agents]
            else:
                primary_agents = agents

        if not primary_agents:
            # Fallback to Aither
            primary_agents = ["aither"] if os.path.exists(os.path.join(PERSONAS_DIR, "aither.yaml")) else []

        if not primary_agents:
            return []

        clean_msg = self.clean_message(message)
        artifact_refs = self.parse_artifacts(message)
        reply_to = sender or self.detect_sender(message)
        direct_mentions = self.parse_direct_mentions(message)

        # Phase 1: Determine which agents should respond
        responding_agents = []
        chiming_in_agents = []  # Agents joining spontaneously
        skipped_agents = []
        responding_count = 0

        # Process primary responders (mentioned or inferred)
        for agent in primary_agents:
            # Direct mentions ALWAYS respond
            if agent in direct_mentions:
                responding_agents.append(agent)
                responding_count += 1
                continue

            # Smart evaluation for mentioned (but not directly mentioned) agents
            if allow_skipping and len(primary_agents) > 1:
                if smart_evaluation:
                    persona_data = load_persona(agent)
                    should_respond, reason = evaluate_should_respond(
                        agent_name=agent,
                        message=clean_msg,
                        conversation_context=conversation_context,
                        persona_data=persona_data or {},
                        total_agents=len(primary_agents),
                        responding_count=responding_count,
                        model="aither-orchestrator-8b-v4"
                    )

                    if should_respond:
                        responding_agents.append(agent)
                        responding_count += 1
                    else:
                        skipped_agents.append({
                            "agent": agent,
                            "response": None,
                            "success": True,
                            "skipped": True,
                            "skip_reason": reason,
                            "mentions": [],
                            "artifacts": [],
                            "delegation": None
                        })
                else:
                    # Simple random skip
                    if should_skip_turn(agent, len(primary_agents), responding_count):
                        skipped_agents.append({
                            "agent": agent,
                            "response": None,
                            "success": True,
                            "skipped": True,
                            "skip_reason": get_skip_reason(),
                            "mentions": [],
                            "artifacts": [],
                            "delegation": None
                        })
                    else:
                        responding_agents.append(agent)
                        responding_count += 1
            else:
                # Single agent or skipping disabled - always respond
                responding_agents.append(agent)
                responding_count += 1

        # Phase 1.5: Check if other agents want to chime in
        if allow_chime_in and all_available_agents:
            for agent in all_available_agents:
                # Skip if already in responding list
                if agent in responding_agents or agent in [s["agent"] for s in skipped_agents]:
                    continue

                # Check if agent wants to chime in
                should_join, reason = should_chime_in(agent, clean_msg, mentioned_agents)
                if should_join:
                    chiming_in_agents.append(agent)

        # Combine responders (primary + chiming in)
        all_responding = responding_agents + chiming_in_agents

        # Phase 2: Generate responses in parallel
        futures = {
            self.executor.submit(
                generate_agent_response,
                agent,
                clean_msg,
                conversation_context,
                "aither-orchestrator-8b-v4",
                reply_to,
                artifact_refs,
                agent.lower() == "aither"  # Only Aither can delegate
            ): agent
            for agent in all_responding
        }

        results = list(skipped_agents)

        # Phase 3: Collect and queue responses with staggered timing
        cumulative_delay = 0.0

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result()
                result["skipped"] = False

                # Add staggered delivery delay
                if staggered_delivery:
                    base_delay = result.get("response_delay", get_random_response_delay())
                    result["delivery_delay"] = cumulative_delay + base_delay
                    cumulative_delay += base_delay
                else:
                    result["delivery_delay"] = 0

                results.append(result)

                # Queue for staggered delivery if enabled
                if staggered_delivery and result["success"] and result["response"]:
                    queued = QueuedResponse(
                        agent=result["agent"],
                        response=result["response"],
                        delay=result["delivery_delay"],
                        mentions=result.get("mentions", []),
                        artifacts=result.get("artifacts", []),
                        is_delegation=result.get("delegation") is not None,
                        delegated_to=result.get("delegation", {}).get("to_agent") if result.get("delegation") else None
                    )
                    self.response_queue.add(queued)

                # Call callback if provided (immediate, not staggered)
                if callback and result["success"]:
                    if staggered_delivery:
                        # Schedule callback after delay
                        delay = result["delivery_delay"]
                        threading.Timer(delay, callback, args=(result["agent"], result["response"])).start()
                    else:
                        callback(result["agent"], result["response"])

            except Exception as e:
                results.append({
                    "agent": agent_name,
                    "response": None,
                    "success": False,
                    "skipped": False,
                    "error": str(e),
                    "mentions": [],
                    "artifacts": [],
                    "delegation": None,
                    "delivery_delay": 0
                })

        return results

    def dispatch_to_mailbox(
        self,
        message: str,
        mailbox_path: str,
        agents: List[str] = None,
        conversation_context: str = "",
        sender: str = "Admin",
        allow_skipping: bool = True,
        smart_evaluation: bool = True,
        require_mention: bool = True  # NEW: Only mentioned agents respond
    ) -> int:
        """
        Dispatch to agents and save responses to mailbox.
        Uses smart evaluation and staggered timing.

        Args:
            require_mention: If True, only @mentioned agents respond (default: True)

        Returns:
            Number of successful responses
        """
        reply_to = sender or self.detect_sender(message)

        results = self.dispatch_parallel(
            message, agents, conversation_context,
            sender=reply_to,
            allow_skipping=allow_skipping,
            smart_evaluation=smart_evaluation,
            staggered_delivery=False,  # Mailbox doesn't need staggering
            require_mention=require_mention
        )

        success_count = 0

        # Load existing mailbox
        messages = []
        if os.path.exists(mailbox_path):
            try:
                with open(mailbox_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
            except Exception as exc:
                logger.debug(f"Mailbox load failed: {exc}")

        # Add responses
        import uuid
        for result in results:
            if result.get("skipped", False):
                continue

            if result["success"] and result["response"]:
                messages.append({
                    "id": str(uuid.uuid4()),
                    "sender": result["agent"].title(),
                    "recipient": "user",
                    "subject": f"[MSG] Re: {message[:30]}...",
                    "content": result["response"],
                    "timestamp": datetime.now().isoformat(),
                    "read": False,
                    "reply_to": reply_to,
                    "mentions": result.get("mentions", [reply_to]),
                    "artifacts": result.get("artifacts", []) + result.get("input_artifacts", []),
                    "delegation": result.get("delegation")
                })
                success_count += 1

        # Save mailbox
        with open(mailbox_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)

        return success_count

    def execute_pending_delegations(self) -> List[Dict[str, Any]]:
        """
        Execute any pending delegation tasks.
        Returns results from sub-agent executions.
        """
        results = []

        # Get all agents that might have pending tasks
        for persona_file in os.listdir(PERSONAS_DIR):
            if persona_file.endswith(".yaml"):
                agent_name = persona_file.replace(".yaml", "")
                pending = _delegation_manager.get_pending_tasks(agent_name)

                for task in pending:
                    result = _delegation_manager.execute_delegation(task)
                    results.append({
                        "task_id": task.id,
                        "from_agent": task.from_agent,
                        "to_agent": task.to_agent,
                        "task": task.task,
                        "result": result,
                        "status": task.status
                    })

        return results


# Singleton
_dispatcher: Optional[MultiAgentDispatcher] = None


def get_dispatcher() -> MultiAgentDispatcher:
    """Get the singleton dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = MultiAgentDispatcher()
    return _dispatcher


def dispatch_multi_agent(
    message: str,
    mailbox_path: str = None,
    agents: List[str] = None,
    callback: Callable = None,
    sender: str = "Admin",
    allow_skipping: bool = True,
    smart_evaluation: bool = True,
    staggered_delivery: bool = True,
    require_mention: bool = True,  # Prefer @mentioned agents
    allow_chime_in: bool = True    # Allow others to join spontaneously
) -> List[Dict[str, Any]]:
    """
    Convenience function to dispatch to multiple agents.

    Response Logic:
    1. **@mentioned agents respond** (always)
    2. **If no @mentions, infer best agent** from message content
    3. **Other agents may chime in** with small probability if topic is relevant

    Args:
        message: The message to send
        mailbox_path: Path to mailbox JSON file (optional)
        agents: List of agent names (or None to auto-determine)
        callback: Callback function(agent, response)
        sender: Who sent the message (for @mention in response)
        allow_skipping: Whether agents can skip turns
        smart_evaluation: Use LLM to decide if agent should respond
        staggered_delivery: Add natural delays between responses
        require_mention: If True, prefer @mentioned agents (default: True)
        allow_chime_in: If True, non-mentioned agents may join (default: True)

    Returns:
        List of response dicts
    """
    dispatcher = get_dispatcher()

    if mailbox_path:
        count = dispatcher.dispatch_to_mailbox(
            message, mailbox_path, agents=agents, sender=sender,
            allow_skipping=allow_skipping, smart_evaluation=smart_evaluation,
            require_mention=require_mention
        )
        return [{"success_count": count}]
    else:
        return dispatcher.dispatch_parallel(
            message, agents=agents, callback=callback, sender=sender,
            allow_skipping=allow_skipping, smart_evaluation=smart_evaluation,
            staggered_delivery=staggered_delivery,
            require_mention=require_mention,
            allow_chime_in=allow_chime_in
        )


def dispatch_with_staggered_callback(
    message: str,
    agents: List[str] = None,
    on_response: Callable[[str, str, float], None] = None,
    sender: str = "Admin"
) -> List[Dict[str, Any]]:
    """
    Dispatch with staggered callbacks - responses delivered with natural timing.

    The on_response callback receives (agent_name, response, delay_seconds)
    and is called after the delay has elapsed.

    Example:
        def handle_response(agent, response, delay):
            print(f"[After {delay:.1f}s] {agent}: {response}")

        dispatch_with_staggered_callback("@aither @leo thoughts?", on_response=handle_response)
    """
    def delayed_callback(agent: str, response: str):
        # This is already called after delay by dispatch_parallel
        if on_response:
            on_response(agent, response, 0)  # Delay already elapsed

    return dispatch_multi_agent(
        message,
        agents=agents,
        callback=delayed_callback,
        sender=sender,
        staggered_delivery=True
    )


def delegate_task(
    from_agent: str,
    to_agent: str,
    task: str,
    context: str = "",
    execute_immediately: bool = True
) -> Dict[str, Any]:
    """
    Delegate a task from one agent to another.

    Primarily used for Aither delegating to council members.

    Args:
        from_agent: The delegating agent (usually "aither")
        to_agent: The agent to execute the task
        task: Description of the task
        context: Additional context for the task
        execute_immediately: Whether to run the task now

    Returns:
        Dict with task_id, status, and result (if executed)
    """
    delegated = _delegation_manager.delegate(from_agent, to_agent, task, context)

    result = {
        "task_id": delegated.id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "task": task,
        "status": delegated.status,
        "result": None
    }

    if execute_immediately:
        execution_result = _delegation_manager.execute_delegation(delegated)
        result["status"] = delegated.status
        result["result"] = execution_result

    return result


def is_multi_agent_query(message: str) -> bool:
    """Check if message has multiple @mentions (multi-agent query)."""
    mentions = re.findall(r"@(\w+)", message.lower())
    valid = [m for m in mentions
             if m.lower() not in ["admin", "user", "human"]
             and os.path.exists(os.path.join(PERSONAS_DIR, f"{m}.yaml"))]
    return len(set(valid)) >= 2


def get_mentioned_agents(message: str) -> List[str]:
    """Get list of @mentioned agents."""
    return get_dispatcher().parse_mentions(message)


def get_direct_mentions(message: str) -> List[str]:
    """Get agents that are directly addressed (must respond)."""
    return get_dispatcher().parse_direct_mentions(message)


def get_referenced_artifacts(message: str) -> List[str]:
    """Get list of #artifact references in message."""
    return get_dispatcher().parse_artifacts(message)


def get_pending_delegations() -> List[DelegatedTask]:
    """Get all pending delegation tasks."""
    all_pending = []
    for persona_file in os.listdir(PERSONAS_DIR):
        if persona_file.endswith(".yaml"):
            agent_name = persona_file.replace(".yaml", "")
            all_pending.extend(_delegation_manager.get_pending_tasks(agent_name))
    return all_pending


def execute_all_delegations() -> List[Dict[str, Any]]:
    """Execute all pending delegation tasks."""
    return get_dispatcher().execute_pending_delegations()

