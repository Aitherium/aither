"""
AitherForge - Sub-Agent Factory and Manager
============================================

The forge spawns, tracks, and collects results from sub-agents.
Sub-agents are lightweight, focused workers that explore specific
aspects of a codebase or problem space.

Key principles:
1. OBJECTIVE - Sub-agents report what they find, not what they think
2. GROUNDED - All findings reference actual code/files
3. TRACEABLE - Every result has a source path
4. PARALLEL - Multiple scouts can explore simultaneously

Events:
- All scout lifecycle events are emitted to AitherPulse for visualization
- AitherWatch can display active scouts in the node view
- Events: scout.spawned, scout.progress, scout.completed, scout.failed

Author: Aitherium
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("AitherForge")

# ===============================================================================
# EVENT EMISSION
# ===============================================================================

# Try to import AitherEvents for Pulse integration
_EVENTS_AVAILABLE = False
try:
    import os
    import sys
    # Add lib path for imports
    _lib_path = Path(__file__).parent.parent.parent.parent / "AitherNode" / "lib"
    if str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

    from AitherEvents import (
        emit_scout_cancelled,
        emit_scout_completed,
        emit_scout_failed,
        emit_scout_finding,
        emit_scout_progress,
        emit_scout_spawned,
    )
    _EVENTS_AVAILABLE = True
    logger.debug("AitherEvents available - scout events will be emitted to AitherPulse")
except ImportError as e:
    logger.warning(f"AitherEvents not available - events will only use local callbacks: {e}")
    # Provide no-op fallbacks
    async def emit_scout_spawned(*args, **kwargs): return False
    async def emit_scout_progress(*args, **kwargs): return False
    async def emit_scout_completed(*args, **kwargs): return False
    async def emit_scout_failed(*args, **kwargs): return False
    async def emit_scout_cancelled(*args, **kwargs): return False
    async def emit_scout_finding(*args, **kwargs): return False


class SubAgentState(str, Enum):
    """Lifecycle states for sub-agents."""
    PENDING = "pending"      # Queued but not started
    SPAWNING = "spawning"    # Being initialized
    EXPLORING = "exploring"  # Actively researching
    ANALYZING = "analyzing"  # Processing findings
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Encountered error
    CANCELLED = "cancelled"  # Stopped by parent


@dataclass
class SubAgentResult:
    """Result from a sub-agent's exploration."""
    agent_id: str
    agent_type: str
    objective: str
    state: SubAgentState
    findings: List[Dict[str, Any]] = field(default_factory=list)
    files_explored: List[str] = field(default_factory=list)
    patterns_matched: Dict[str, int] = field(default_factory=dict)
    summary: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Time taken for exploration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "objective": self.objective,
            "state": self.state.value,
            "findings": self.findings,
            "files_explored": self.files_explored,
            "patterns_matched": self.patterns_matched,
            "summary": self.summary,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class SubAgent:
    """
    A spawned sub-agent instance.

    Sub-agents are ephemeral workers that:
    - Receive a specific objective
    - Explore the codebase
    - Report objective findings
    - Terminate when done
    """
    id: str
    agent_type: str
    objective: str
    state: SubAgentState = SubAgentState.PENDING
    parent_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[SubAgentResult] = None
    task: Optional[asyncio.Task] = None
    created_at: datetime = field(default_factory=datetime.now)

    def cancel(self):
        """Cancel this sub-agent's task."""
        if self.task and not self.task.done():
            self.task.cancel()
            self.state = SubAgentState.CANCELLED


class AitherForge:
    """
    Factory for spawning and managing sub-agents.

    The Forge maintains a registry of active sub-agents and provides
    methods to spawn new ones, track their progress, and collect results.

    Usage:
        forge = AitherForge(workspace_root="/path/to/project")

        # Spawn a scout to find files
        scout = await forge.spawn_scout(
            objective="Find all database models",
            search_paths=["src/models", "lib/db"],
            patterns=["class.*Model", "Table", "Column"]
        )

        # Wait for results
        result = await forge.wait_for(scout.id)

        # Or spawn multiple and collect all
        results = await forge.spawn_and_collect([
            {"objective": "Find auth code", "patterns": ["auth", "login"]},
            {"objective": "Find API routes", "patterns": ["@router", "@app.route"]},
        ])
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        max_concurrent: int = 5,
        timeout_seconds: float = 60.0,
    ):
        """
        Initialize the forge.

        Args:
            workspace_root: Root directory for file searches
            max_concurrent: Maximum sub-agents running simultaneously
            timeout_seconds: Default timeout for sub-agent tasks
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds

        self._agents: Dict[str, SubAgent] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._callbacks: Dict[str, List[Callable]] = {
            "on_spawn": [],
            "on_complete": [],
            "on_error": [],
        }

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for sub-agent events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _emit(self, event: str, agent: SubAgent):
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent)
                else:
                    callback(agent)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def _generate_id(self, agent_type: str) -> str:
        """Generate a unique ID for a sub-agent."""
        short_uuid = str(uuid.uuid4())[:8]
        return f"{agent_type}-{short_uuid}"

    async def spawn_scout(
        self,
        objective: str,
        search_paths: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_depth: int = 10,
        context: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> SubAgent:
        """
        Spawn a scout sub-agent for codebase exploration.

        Args:
            objective: What the scout should find/understand
            search_paths: Directories to search (relative to workspace_root)
            patterns: Text/regex patterns to look for in file contents
            file_patterns: Glob patterns for file names (e.g., "*.py")
            exclude_patterns: Patterns to skip
            max_depth: Maximum directory depth to traverse
            context: Additional context to share with the scout
            parent_id: ID of the parent agent (for tracking)

        Returns:
            SubAgent instance (task starts immediately)
        """
        from .scout import AitherScout, ScoutTask

        agent_id = self._generate_id("scout")

        agent = SubAgent(
            id=agent_id,
            agent_type="scout",
            objective=objective,
            parent_id=parent_id,
            context=context or {},
        )

        self._agents[agent_id] = agent

        # Create scout task
        task = ScoutTask(
            objective=objective,
            search_paths=search_paths or ["."],
            patterns=patterns or [],
            file_patterns=file_patterns or ["*.py", "*.ps1", "*.ts", "*.js", "*.md"],
            exclude_patterns=exclude_patterns or ["node_modules", ".venv", "__pycache__", ".git"],
            max_depth=max_depth,
            workspace_root=str(self.workspace_root),
        )

        # Spawn the scout
        async def run_scout():
            async with self._semaphore:
                agent.state = SubAgentState.SPAWNING
                await self._emit("on_spawn", agent)

                # Emit spawn event to AitherPulse
                await emit_scout_spawned(
                    scout_id=agent_id,
                    objective=objective,
                    parent_id=parent_id,
                    search_paths=search_paths or ["."],
                    patterns=patterns or [],
                )

                try:
                    scout = AitherScout(task, agent_id)
                    agent.state = SubAgentState.EXPLORING

                    # Emit exploring status
                    await emit_scout_progress(
                        scout_id=agent_id,
                        phase="exploring",
                        message=f"Searching for: {objective[:50]}...",
                        progress=0.1,
                    )

                    result = await asyncio.wait_for(
                        scout.explore(),
                        timeout=self.timeout_seconds
                    )

                    agent.state = SubAgentState.COMPLETED
                    agent.result = result

                    # Emit completion event
                    await emit_scout_completed(
                        scout_id=agent_id,
                        objective=objective,
                        files_explored=len(result.files_explored),
                        findings_count=len(result.findings),
                        duration_seconds=result.duration_seconds,
                        summary=result.summary,
                    )

                    await self._emit("on_complete", agent)

                except asyncio.TimeoutError:
                    agent.state = SubAgentState.FAILED
                    agent.result = SubAgentResult(
                        agent_id=agent_id,
                        agent_type="scout",
                        objective=objective,
                        state=SubAgentState.FAILED,
                        error=f"Timeout after {self.timeout_seconds}s",
                    )

                    # Emit failure event
                    await emit_scout_failed(
                        scout_id=agent_id,
                        objective=objective,
                        error=f"Timeout after {self.timeout_seconds}s",
                        phase="exploring",
                    )

                    await self._emit("on_error", agent)

                except asyncio.CancelledError:
                    agent.state = SubAgentState.CANCELLED

                    # Emit cancellation event
                    await emit_scout_cancelled(
                        scout_id=agent_id,
                        objective=objective,
                        reason="task_cancelled",
                    )

                    raise

                except Exception as e:
                    agent.state = SubAgentState.FAILED
                    agent.result = SubAgentResult(
                        agent_id=agent_id,
                        agent_type="scout",
                        objective=objective,
                        state=SubAgentState.FAILED,
                        error=str(e),
                    )

                    # Emit failure event
                    await emit_scout_failed(
                        scout_id=agent_id,
                        objective=objective,
                        error=str(e),
                        phase="exploring",
                    )

                    logger.exception(f"Scout {agent_id} failed: {e}")
                    await self._emit("on_error", agent)

        agent.task = asyncio.create_task(run_scout())
        return agent

    async def spawn_multiple(
        self,
        tasks: List[Dict[str, Any]],
        agent_type: str = "scout",
    ) -> List[SubAgent]:
        """
        Spawn multiple sub-agents for different objectives.

        Args:
            tasks: List of task configurations (objective, patterns, etc.)
            agent_type: Type of agent to spawn

        Returns:
            List of spawned SubAgent instances
        """
        agents = []
        for task_config in tasks:
            if agent_type == "scout":
                agent = await self.spawn_scout(**task_config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            agents.append(agent)
        return agents

    async def wait_for(self, agent_id: str) -> Optional[SubAgentResult]:
        """Wait for a specific sub-agent to complete."""
        agent = self._agents.get(agent_id)
        if not agent or not agent.task:
            return None

        try:
            await agent.task
        except asyncio.CancelledError:
            pass

        return agent.result

    async def wait_all(
        self,
        agent_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> List[SubAgentResult]:
        """
        Wait for multiple sub-agents to complete.

        Args:
            agent_ids: Specific agents to wait for (None = all active)
            timeout: Optional timeout for all agents

        Returns:
            List of results from completed agents
        """
        if agent_ids is None:
            agent_ids = list(self._agents.keys())

        tasks = []
        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if agent and agent.task:
                tasks.append(agent.task)

        if not tasks:
            return []

        if timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            # Cancel pending tasks
            for task in pending:
                task.cancel()
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for agent_id in agent_ids:
            agent = self._agents.get(agent_id)
            if agent and agent.result:
                results.append(agent.result)

        return results

    async def spawn_and_collect(
        self,
        tasks: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> List[SubAgentResult]:
        """
        Spawn multiple scouts and wait for all results.

        Convenience method that combines spawn_multiple and wait_all.
        """
        agents = await self.spawn_multiple(tasks)
        return await self.wait_all(
            [a.id for a in agents],
            timeout=timeout or self.timeout_seconds * len(agents)
        )

    def cancel_agent(self, agent_id: str):
        """Cancel a specific sub-agent."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.cancel()

    def cancel_all(self):
        """Cancel all active sub-agents."""
        for agent in self._agents.values():
            agent.cancel()

    def get_status(self) -> Dict[str, Any]:
        """Get status of all sub-agents."""
        states = {}
        for state in SubAgentState:
            states[state.value] = sum(
                1 for a in self._agents.values() if a.state == state
            )

        return {
            "total": len(self._agents),
            "states": states,
            "agents": [
                {
                    "id": a.id,
                    "type": a.agent_type,
                    "state": a.state.value,
                    "objective": a.objective[:50] + "..." if len(a.objective) > 50 else a.objective,
                }
                for a in self._agents.values()
            ]
        }

    def cleanup(self, completed_only: bool = True):
        """Remove completed/failed agents from registry."""
        to_remove = []
        for agent_id, agent in self._agents.items():
            if completed_only:
                if agent.state in (SubAgentState.COMPLETED, SubAgentState.FAILED, SubAgentState.CANCELLED):
                    to_remove.append(agent_id)
            else:
                to_remove.append(agent_id)

        for agent_id in to_remove:
            del self._agents[agent_id]


# Singleton instance for easy access
_forge_instance: Optional[AitherForge] = None


def get_forge(workspace_root: Optional[str] = None) -> AitherForge:
    """Get or create the global AitherForge instance."""
    global _forge_instance
    if _forge_instance is None:
        _forge_instance = AitherForge(workspace_root=workspace_root)
    return _forge_instance
