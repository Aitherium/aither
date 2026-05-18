"""
Sub-Agent Tools for Google ADK Agents
======================================

Tools that allow ADK agents to spawn research sub-agents (Scouts)
for codebase exploration and context gathering.

These tools integrate with the AitherForge service to spawn
and manage sub-agents.

Usage in an ADK agent:
    from google.adk import Agent
    from AitherOS.agents.common.tools.subagent_tools import subagent_tools
    
    agent = Agent(
        name="MyAgent",
        model="gemini-2.5-flash",
        tools=subagent_tools,
    )

Author: Aitherium
"""

import os
import json
from typing import Optional, List, Dict, Any
import httpx

# Get forge URL from environment or use default
FORGE_URL = os.getenv("AITHER_FORGE_URL", "http://localhost:8768")

# Try to import FunctionTool from google.adk
try:
    from google.adk.tools import FunctionTool
except ImportError:
    # Fallback for standalone usage
    FunctionTool = lambda x: x


async def spawn_scout(
    objective: str,
    patterns: Optional[List[str]] = None,
    search_paths: Optional[List[str]] = None,
    file_patterns: Optional[List[str]] = None,
    max_depth: int = 10,
    wait_for_result: bool = True,
) -> str:
    """
    Spawn a scout sub-agent to explore the codebase.
    
    Scouts are specialized research agents that:
    - Search files for patterns
    - Report objective findings with file paths and line numbers
    - Do NOT make assumptions - only report what they find
    
    Args:
        objective: What the scout should find or understand (be specific!)
        patterns: Text or regex patterns to search for in file contents
        search_paths: Directories to search (relative to project root, default: ["."])
        file_patterns: File types to include (default: ["*.py", "*.ps1", "*.ts", "*.js", "*.md"])
        max_depth: Maximum directory depth to traverse
        wait_for_result: If True, wait for scout to finish and return results
        
    Returns:
        JSON string with scout findings including:
        - files_explored: List of files searched
        - findings: List of matches with file path, line number, and content
        - summary: Overview of what was found
        
    Example:
        # Find all authentication-related code
        result = await spawn_scout(
            objective="Find authentication and login code",
            patterns=["auth", "login", "authenticate", "session"],
            search_paths=["AitherOS/services", "AitherOS/agents"],
            file_patterns=["*.py"]
        )
    """
    request = {
        "objective": objective,
        "patterns": patterns or [],
        "search_paths": search_paths or ["."],
        "file_patterns": file_patterns or ["*.py", "*.ps1", "*.ts", "*.js", "*.md"],
        "max_depth": max_depth,
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Spawn the scout
            response = await client.post(f"{FORGE_URL}/spawn", json=request)
            response.raise_for_status()
            agent_info = response.json()
            
            if not wait_for_result:
                return json.dumps({
                    "status": "spawned",
                    "agent_id": agent_info["id"],
                    "objective": objective,
                    "message": f"Scout {agent_info['id']} spawned. Use get_scout_result to retrieve findings."
                }, indent=2)
            
            # Wait for result
            result_response = await client.get(
                f"{FORGE_URL}/result/{agent_info['id']}",
                params={"wait": True}
            )
            
            if result_response.status_code == 202:
                return json.dumps({
                    "status": "running",
                    "agent_id": agent_info["id"],
                    "message": "Scout is still exploring. Try again later."
                }, indent=2)
            
            result_response.raise_for_status()
            result = result_response.json()
            
            # Format for readability
            return json.dumps({
                "status": "completed",
                "objective": result.get("objective"),
                "files_explored": result.get("files_explored_count", 0),
                "findings_count": result.get("findings_count", 0),
                "summary": result.get("summary"),
                "pattern_matches": result.get("patterns_matched", {}),
                "findings": result.get("findings", [])[:30],  # Limit for context
                "error": result.get("error"),
            }, indent=2)
            
        except httpx.HTTPStatusError as e:
            return json.dumps({
                "error": f"HTTP error: {e.response.status_code}",
                "detail": str(e),
            }, indent=2)
        except httpx.RequestError as e:
            return json.dumps({
                "error": "Connection error - is AitherForge running?",
                "detail": str(e),
                "hint": "Start AitherForge with: python AitherOS/services/agents/AitherForge.py"
            }, indent=2)


async def spawn_research_team(
    topic: str,
    aspects: List[str],
    search_root: str = ".",
) -> str:
    """
    Spawn multiple scouts to research different aspects of a topic.
    
    This is useful when you need to understand a complex system
    from multiple angles simultaneously.
    
    Args:
        topic: The main topic to research (e.g., "authentication system")
        aspects: Different aspects to investigate (e.g., ["models", "routes", "tests"])
        search_root: Base directory to search from
        
    Returns:
        JSON with aggregated findings from all scouts
        
    Example:
        # Research the memory system from multiple angles
        result = await spawn_research_team(
            topic="memory system",
            aspects=[
                "memory storage and persistence",
                "memory retrieval and search",
                "memory caching and performance",
            ],
            search_root="AitherOS/services/memory"
        )
    """
    # Build tasks for each aspect
    tasks = []
    for aspect in aspects:
        # Generate patterns from the aspect description
        words = aspect.lower().split()
        patterns = [w for w in words if len(w) > 3]  # Skip short words
        
        tasks.append({
            "objective": f"Research {topic}: {aspect}",
            "patterns": patterns,
            "search_paths": [search_root],
            "file_patterns": ["*.py", "*.ps1", "*.ts", "*.js", "*.md"],
        })
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{FORGE_URL}/spawn-multiple",
                json={
                    "tasks": tasks,
                    "collect_results": True,
                }
            )
            response.raise_for_status()
            result = response.json()
            
            return json.dumps({
                "status": "completed",
                "topic": topic,
                "aspects_researched": aspects,
                "summary": result.get("summary", {}),
                "unique_findings": result.get("unique_findings", [])[:50],
                "files_explored": result.get("files_explored", [])[:100],
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "topic": topic,
            }, indent=2)


async def search_codebase(
    pattern: str,
    paths: Optional[List[str]] = None,
    file_types: Optional[List[str]] = None,
    max_results: int = 30,
    context_lines: int = 2,
) -> str:
    """
    Quick synchronous search of the codebase (no sub-agent).
    
    Use this for simple searches where you know exactly what pattern
    you're looking for. For complex exploration, use spawn_scout instead.
    
    Args:
        pattern: Text or regex pattern to search for
        paths: Directories to search (default: entire workspace)
        file_types: File extensions to include (default: common code files)
        max_results: Maximum matches to return
        context_lines: Lines of context before/after each match
        
    Returns:
        JSON with list of matches including file, line, and context
    """
    request = {
        "pattern": pattern,
        "paths": paths or ["."],
        "file_patterns": file_types or ["*.py", "*.ps1", "*.ts", "*.js", "*.md"],
        "max_results": max_results,
        "context_lines": context_lines,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(f"{FORGE_URL}/tools/search", json=request)
            response.raise_for_status()
            result = response.json()
            
            # Format matches for readability
            formatted_matches = []
            for match in result.get("matches", []):
                formatted_matches.append({
                    "file": match.get("relative_path", match.get("file_path")),
                    "line": match.get("line_number"),
                    "content": match.get("line_content"),
                    "context": {
                        "before": match.get("context_before", []),
                        "after": match.get("context_after", []),
                    }
                })
            
            return json.dumps({
                "pattern": pattern,
                "total_matches": result.get("count", 0),
                "matches": formatted_matches,
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "pattern": pattern,
            }, indent=2)


async def get_file_content(
    file_path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    max_lines: int = 100,
) -> str:
    """
    Read content from a specific file.
    
    Use this after spawn_scout finds relevant files to read their
    full content for deeper understanding.
    
    Args:
        file_path: Path to the file (relative to workspace root)
        start_line: First line to read (1-indexed)
        end_line: Last line to read (None = start_line + max_lines)
        max_lines: Maximum lines to read
        
    Returns:
        JSON with file content and metadata
    """
    request = {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "max_lines": max_lines,
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(f"{FORGE_URL}/tools/read-file", json=request)
            response.raise_for_status()
            result = response.json()
            
            return json.dumps({
                "file": result.get("relative_path", file_path),
                "type": result.get("file_type"),
                "lines": f"{result.get('start_line')}-{result.get('end_line')} of {result.get('total_lines')}",
                "content": result.get("content"),
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "file_path": file_path,
            }, indent=2)


async def get_project_structure(max_depth: int = 2) -> str:
    """
    Get an overview of the project structure.
    
    Useful for understanding the codebase layout before
    spawning scouts for specific exploration.
    
    Args:
        max_depth: How deep to show the directory structure
        
    Returns:
        JSON with project structure and statistics
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{FORGE_URL}/tools/structure",
                params={"max_depth": max_depth}
            )
            response.raise_for_status()
            result = response.json()
            
            return json.dumps({
                "root": result.get("root"),
                "statistics": result.get("statistics"),
                "key_files": result.get("key_files"),
                # Simplified structure for context efficiency
                "top_directories": [
                    c.get("name") for c in result.get("structure", {}).get("children", [])
                    if c.get("type") == "directory"
                ][:20],
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
            }, indent=2)


async def get_forge_status() -> str:
    """
    Get status of all active sub-agents.
    
    Use this to check on running scouts or see what
    research has been done.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{FORGE_URL}/status")
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)


# Export as FunctionTools for ADK
subagent_tools = [
    FunctionTool(spawn_scout),
    FunctionTool(spawn_research_team),
    FunctionTool(search_codebase),
    FunctionTool(get_file_content),
    FunctionTool(get_project_structure),
    FunctionTool(get_forge_status),
]

# Also export individual functions
__all__ = [
    "spawn_scout",
    "spawn_research_team",
    "search_codebase",
    "get_file_content",
    "get_project_structure",
    "get_forge_status",
    "subagent_tools",
]
