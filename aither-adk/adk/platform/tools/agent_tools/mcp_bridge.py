"""
MCP Bridge - Call MCP Tools from ADK Agents

This module provides FunctionTool wrappers that allow Google ADK agents
to call AitherNode MCP tools. It bridges the two architectures:

    ADK Agent -> mcp_bridge -> HTTP/MCP -> AitherNode -> Tool Execution

Key Features:
- Async HTTP calls to MCP server
- Error handling with graceful fallbacks  
- Common tool shortcuts (remember, recall, run_script)
- Generic call_mcp_tool for any registered MCP tool

Usage:
    from agents.common.agent_tools.mcp_bridge import get_tools
    
    tools = get_tools()  # Returns list of FunctionTool instances
    
    # Or import specific functions
    from agents.common.agent_tools.mcp_bridge import remember, recall

Author: Aitherium
Version: 2.0.0
"""

import os
import json
import httpx
from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool

# MCP Server configuration
MCP_BASE_URL = os.environ.get("AITHERNODE_URL", "http://localhost:8080")
MCP_TIMEOUT = float(os.environ.get("MCP_TIMEOUT", "30"))


async def call_mcp_tool(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call any registered MCP tool by name.
    
    This is the generic bridge that allows ADK agents to invoke any tool
    registered on the AitherNode MCP server.
    
    Args:
        tool_name: Name of the MCP tool (e.g., "remember", "generate_image")
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        Dict with result or error information
        
    Example:
        result = await call_mcp_tool(
            "generate_image",
            {"prompt": "A beautiful sunset", "size": "1024x1024"}
        )
    """
    arguments = arguments or {}
    
    try:
        async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
            response = await client.post(
                f"{MCP_BASE_URL}/mcp/tools/call",
                json={
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"MCP call failed: {response.status_code}",
                    "detail": response.text,
                }
                
    except httpx.TimeoutException:
        return {"error": f"MCP call timed out after {MCP_TIMEOUT}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to MCP server at {MCP_BASE_URL}"}
    except Exception as e:
        return {"error": f"MCP call error: {str(e)}"}


async def remember(
    content: str,
    category: str = "conversation",
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
) -> Dict[str, Any]:
    """
    Store information in long-term memory via MCP.
    
    Use this to save important facts, preferences, or context that should
    persist across conversation sessions.
    
    Args:
        content: The information to remember
        category: Category for organization (user, project, system, conversation)
        tags: Optional tags for better retrieval
        importance: Importance score 0.0-1.0 (higher = more important)
        
    Returns:
        Dict with memory_id or error
        
    Example:
        await remember(
            "User prefers dark fantasy themes in images",
            category="user",
            tags=["preference", "style"],
            importance=0.8
        )
    """
    return await call_mcp_tool(
        "remember",
        {
            "content": content,
            "category": category,
            "tags": tags or [],
            "importance": importance,
        }
    )


async def recall(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Search long-term memory for relevant information via MCP.
    
    Use semantic search to find previously stored memories that are
    relevant to the current context.
    
    Args:
        query: Search query (semantic matching)
        category: Optional category filter
        limit: Maximum results to return
        
    Returns:
        Dict with memories list or error
        
    Example:
        memories = await recall(
            "user's image style preferences",
            category="user"
        )
    """
    return await call_mcp_tool(
        "recall",
        {
            "query": query,
            "category": category,
            "limit": limit,
        }
    )


async def run_script(
    script_number: str,
    parameters: Optional[Dict[str, Any]] = None,
    show_output: bool = True,
) -> Dict[str, Any]:
    """
    Run an AitherZero automation script via MCP.
    
    Execute PowerShell automation scripts from the AitherZero library.
    Scripts are identified by their 4-digit number prefix.
    
    Args:
        script_number: Script number (e.g., "0011", "0402")
        parameters: Optional parameters to pass to the script
        show_output: Whether to capture and return output
        
    Returns:
        Dict with execution result or error
        
    Example:
        # Run system info script
        result = await run_script("0011")
        
        # Run tests with parameters
        result = await run_script("0402", {"Tag": "Unit"})
    """
    return await call_mcp_tool(
        "run_automation_script",
        {
            "script_number": script_number,
            "parameters": parameters or {},
            "show_output": show_output,
        }
    )


async def analyze_image(
    image_path: str,
    prompt: Optional[str] = None,
    analysis_type: str = "describe",
) -> Dict[str, Any]:
    """
    Analyze an image using AitherVision via MCP.
    
    Args:
        image_path: Path to the image file
        prompt: Optional specific question about the image
        analysis_type: Type of analysis (describe, detailed, ocr, objects, style)
        
    Returns:
        Dict with analysis result or error
    """
    return await call_mcp_tool(
        "analyze_image_content",
        {
            "image_path": image_path,
            "prompt": prompt or "",
            "analysis_type": analysis_type,
        }
    )


async def list_mcp_tools() -> Dict[str, Any]:
    """
    List all available MCP tools.
    
    Returns:
        Dict with list of tool names and descriptions
    """
    try:
        async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
            response = await client.get(f"{MCP_BASE_URL}/mcp/tools")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to list tools: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# FUNCTION TOOL WRAPPERS
# ============================================================================

def get_tools(persona: Optional[str] = None) -> List[FunctionTool]:
    """
    Get MCP bridge tools as FunctionTool instances for ADK agents.
    
    Args:
        persona: Not used for bridge tools, but kept for API consistency
        
    Returns:
        List of FunctionTool instances
    """
    return [
        FunctionTool(call_mcp_tool),
        FunctionTool(remember),
        FunctionTool(recall),
        FunctionTool(run_script),
        FunctionTool(analyze_image),
        FunctionTool(list_mcp_tools),
    ]


# Export key functions
__all__ = [
    "call_mcp_tool",
    "remember",
    "recall", 
    "run_script",
    "analyze_image",
    "list_mcp_tools",
    "get_tools",
]
