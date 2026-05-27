"""
AitherForge - Sub-Agent Spawning and Management System
=======================================================

"Give your ideas a body." - The sub-agent system allows parent agents
to fork specialized research agents that explore codebases, gather context,
and return objective findings.

This module provides:
- AitherForge: The factory for spawning sub-agents
- AitherScout: Research-focused sub-agent for codebase exploration
- ContextFork: Mechanism for sharing context between parent/child agents

Components:
    - forge.py: Core AitherForge manager for spawning and tracking sub-agents
    - scout.py: AitherScout - specialized exploration sub-agent
    - context.py: Context forking and sharing utilities
    - tools.py: Tools available to sub-agents for research
    - results.py: Result aggregation and reporting

Architecture:
                        +---------------------+
                        |   Parent Agent      |
                        |  (Orchestrator)     |
                        +----------+----------+
                                   | spawn()
                        +----------v----------+
                        |    AitherForge      |
                        |  (Sub-Agent Factory)|
                        +----------+----------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
    +-----v-----+           +------v------+          +------v------+
    |  Scout A  |           |  Scout B    |          |  Scout C    |
    | (Files)   |           | (Patterns)  |          | (Deps)      |
    +-----+-----+           +------+------+          +------+------+
          |                        |                        |
          +------------------------+------------------------+
                                   | collect()
                        +----------v----------+
                        |   Aggregated        |
                        |   Research Results  |
                        +---------------------+

Usage:
    from AitherOS.agents.common.subagent import AitherForge, ScoutTask
    
    forge = AitherForge()
    
    # Spawn scouts to research a topic
    task = ScoutTask(
        objective="Find all authentication-related code",
        search_paths=["src/", "lib/"],
        patterns=["auth", "login", "token", "session"]
    )
    
    results = await forge.spawn_scouts(task, num_scouts=3)
    
    # Results are objective, grounded in actual code
    for finding in results.findings:
        print(f"{finding.file}: {finding.summary}")
"""

from .forge import AitherForge, SubAgent, SubAgentState, SubAgentResult
from .scout import AitherScout, ScoutTask, ScoutFinding
from .context import ContextFork, SharedContext

__all__ = [
    "AitherForge",
    "SubAgent", 
    "SubAgentState",
    "SubAgentResult",
    "AitherScout",
    "ScoutTask",
    "ScoutFinding",
    "ContextFork",
    "SharedContext",
]
