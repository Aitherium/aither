import os
import json
import asyncio
import re
from typing import Optional, Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _get_workspace_root() -> str:
    """Get the workspace root from AITHERZERO_ROOT or default."""
    return os.environ.get("AITHERZERO_ROOT", "/workspaces/AitherZero")


def _resolve_variables(value: str) -> str:
    """
    Resolve VS Code-style variables in config values.
    
    Supports:
    - ${workspaceFolder} -> AITHERZERO_ROOT
    - ${env:VAR_NAME} -> environment variable
    """
    if not isinstance(value, str):
        return value
    
    workspace_root = _get_workspace_root()
    
    # Replace ${workspaceFolder}
    value = value.replace("${workspaceFolder}", workspace_root)
    
    # Replace ${env:VAR_NAME} patterns
    env_pattern = re.compile(r'\$\{env:([^}]+)\}')
    def replace_env(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    
    value = env_pattern.sub(replace_env, value)
    
    return value


def _resolve_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve variables in config."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = _resolve_variables(value)
        elif isinstance(value, list):
            resolved[key] = [_resolve_variables(v) if isinstance(v, str) else v for v in value]
        elif isinstance(value, dict):
            resolved[key] = _resolve_config_values(value)
        else:
            resolved[key] = value
    return resolved


# Helper to get config
def _get_mcp_config() -> Dict[str, Any]:
    """Load and resolve MCP configuration."""
    root = _get_workspace_root()
    config_path = os.path.join(root, ".vscode/mcp.json")

    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return _resolve_config_values(config)
    except Exception:
        return {}


def _get_server_params(server_name: str) -> Optional[StdioServerParameters]:
    """Get resolved server parameters for a named MCP server."""
    config = _get_mcp_config()
    servers = config.get("mcpServers", config.get("servers", {}))
    server_config = servers.get(server_name)

    if not server_config:
        return None

    command = server_config.get("command")
    args = server_config.get("args", [])
    env = server_config.get("env", {})

    # Merge resolved env with current env
    run_env = os.environ.copy()
    if env:
        for k, v in env.items():
            # Values should already be resolved by _resolve_config_values
            run_env[k] = str(v) if v else ""

    return StdioServerParameters(
        command=command,
        args=args,
        env=run_env
    )

async def list_mcp_servers() -> str:
    """
    Lists all configured MCP servers found in .vscode/mcp.json.
    """
    config = _get_mcp_config()
    servers = config.get("mcpServers", config.get("servers", {}))
    if not servers:
        return "No MCP servers configured."

    return json.dumps(list(servers.keys()), indent=2)

async def list_mcp_tools(server_name: str) -> str:
    """
    Lists tools available on a specific MCP server.
    Args:
        server_name: The name of the registered MCP server.
    """
    params = _get_server_params(server_name)
    if not params:
        return f"Server '{server_name}' not found in configuration."

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.list_tools()

                # Format tools for display
                tools_info = []
                for tool in result.tools:
                    tools_info.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    })

                return json.dumps(tools_info, indent=2)
    except Exception as e:
        return f"Error listing tools on '{server_name}': {str(e)}"

async def call_mcp_tool(server_name: str, tool_name: str, arguments: Optional[dict] = None) -> str:
    """
    Calls a tool on a specific MCP server.
    Args:
        server_name: The registered server name.
        tool_name: The name of the tool to call.
        arguments: Dictionary of arguments for the tool.
    """
    params = _get_server_params(server_name)
    if not params:
        return f"Server '{server_name}' not found."

    if arguments is None:
        arguments = {}

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Call tool
                result = await session.call_tool(tool_name, arguments)

                # Process content
                output = []
                if result.content:
                    for content in result.content:
                        if content.type == 'text':
                            output.append(content.text)
                        # Handle other types if needed (image, etc)

                final_output = "\n".join(output)
                if result.isError:
                     return f"Tool Error: {final_output}"

                return final_output

    except Exception as e:
        return f"Error calling tool '{tool_name}' on '{server_name}': {str(e)}"
