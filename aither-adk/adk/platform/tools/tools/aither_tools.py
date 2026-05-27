import json
import os
from typing import Optional, List, Dict
from google.adk.tools import FunctionTool

# Import MCP Client tools
try:
    from .mcp_client import list_mcp_servers, list_mcp_tools, call_mcp_tool
except ImportError:
    # Fallback for different import contexts
    try:
        from AitherOS.agents.common.tools.mcp_client import list_mcp_servers, list_mcp_tools, call_mcp_tool
    except ImportError:
        pass

# Import Workflow Engine
try:
    from AitherOS.agents.AitherZeroAutomationAgent.workflows.engine import WorkflowEngine
except ImportError:
    try:
        from workflows.engine import WorkflowEngine
    except ImportError:
        WorkflowEngine = None

# Determine project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# AitherOS/agents/common/tools -> 4 levels up to Repo Root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))

# MCP Server Name
MCP_SERVER_NAME = "aitherzero"

async def execute_aither_script(
    script_name: str,
    parameters: Optional[dict] = None,
    verbose: bool = False,
    show_output: bool = False,
    show_transcript: bool = False,
    dry_run: bool = False,
    transcript: bool = True
) -> str:
    """
    Executes a PowerShell automation script from the AitherZero library via MCP.

    Args:
        script_name (str): The name or ID of the script to execute (e.g., '0206', '0206_Install-Python.ps1').
        parameters (dict, optional): A dictionary of parameters to pass to the script.
        verbose (bool, optional): Whether to run with verbose output.
        show_output (bool, optional): Whether to show script output in console.
        show_transcript (bool, optional): Whether to display the transcript content after execution.
        dry_run (bool, optional): Whether to show what would be executed without actually running.
        transcript (bool, optional): Whether to enable transcript logging.

    Returns:
        str: The standard output and error from the script execution.
    """
    # Extract script number if possible (e.g. "0206_Install..." -> "0206")
    script_number = script_name.split('_')[0] if '_' in script_name else script_name
    if not script_number.isdigit() or len(script_number) != 4:
        # If not a number, try to find it via list_scripts (not implemented here, assuming number or full name passed)
        # For now, pass as is, but MCP expects scriptNumber.
        # If user passes full name, we might need to search.
        # But let's assume the agent passes the number or we rely on MCP to handle it (MCP expects scriptNumber).
        pass

    args = {
        "scriptNumber": script_number,
        "params": parameters if parameters else {},
        "verbose": verbose,
        "dryRun": dry_run,
        "showOutput": show_output
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "run_script", args)

async def list_automation_scripts(filter: str = "") -> str:
    """
    Lists available automation scripts in the AitherZero library via MCP.

    Args:
        filter (str, optional): A keyword to filter scripts by name or description.
    """
    if filter:
        return await call_mcp_tool(MCP_SERVER_NAME, "search_scripts", {"query": filter})
    else:
        return await call_mcp_tool(MCP_SERVER_NAME, "list_scripts", {})

async def list_playbooks(filter: str = "") -> str:
    """
    Lists available automation playbooks in the AitherZero library via MCP.

    Args:
        filter (str, optional): A keyword to filter playbooks by name or description.
    """
    # MCP list_playbooks doesn't support filter yet, so we filter client side or just return all
    result = await call_mcp_tool(MCP_SERVER_NAME, "list_playbooks", {})
    # If filter is provided, we could parse and filter, but for now return all
    return result

async def execute_aither_playbook(playbook_name: str, variables: Optional[dict] = None) -> str:
    """
    Executes an automation playbook via MCP.

    Args:
        playbook_name (str): The name of the playbook to execute.
        variables (dict, optional): Variables to pass to the playbook.
    """
    args = {
        "playbookName": playbook_name,
        "variables": variables if variables else {}
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "execute_playbook", args)

async def get_script_info(target: str) -> str:
    """
    Retrieves detailed information about a script via MCP.
    """
    return await get_automation_help(target, type='script')

async def get_automation_help(target: str, type: str = 'script') -> str:
    """
    Retrieves help/documentation for a specific script or playbook via MCP.
    """
    args = {
        "target": target,
        "type": type
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "get_automation_help", args)

async def get_aither_config(section: Optional[str] = None, key: Optional[str] = None) -> str:
    """
    Retrieves the current AitherZero configuration via MCP.
    """
    args = {}
    if section:
        args["section"] = section
    if key:
        args["key"] = key
    return await call_mcp_tool(MCP_SERVER_NAME, "get_configuration", args)

async def set_aither_config(section: str, key: str, value: str, scope: str = "local") -> str:
    """
    Updates AitherZero configuration via MCP.
    """
    args = {
        "section": section,
        "key": key,
        "value": value,
        "scope": scope
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "set_configuration", args)

async def manage_mcp_server(action: str, name: Optional[str] = None, command: Optional[str] = None, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> str:
    """
    Manages MCP Servers via MCP (meta!).
    """
    arguments = {
        "action": action,
        "name": name,
        "command": command,
        "args": args,
        "env": env
    }
    # Remove None values
    arguments = {k: v for k, v in arguments.items() if v is not None}
    return await call_mcp_tool(MCP_SERVER_NAME, "manage_mcp_server", arguments)


# ============================================================================
# Development Workflow Tools - For autonomous code fix & test workflows
# ============================================================================

async def create_dev_branch(name: str, use_worktree: bool = False, source_branch: str = "main") -> str:
    """
    Creates a new development branch for autonomous code fixes.
    
    This tool enables agents to create isolated branches for testing fixes
    before merging to main. Optionally creates a git worktree for parallel work.
    
    Args:
        name (str): Branch name (e.g., 'fix/issue-123', 'feature/new-api').
        use_worktree (bool): If True, creates a git worktree for parallel development.
        source_branch (str): Branch to create from (default: 'main').
    
    Returns:
        str: JSON with branch_name, worktree_path (if applicable), and status.
    """
    args = {
        "name": name,
        "useWorktree": use_worktree,
        "sourceBranch": source_branch
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "create_dev_branch", args)


async def checkout_branch(branch: str, create: bool = False) -> str:
    """
    Checks out a git branch in the repository.
    
    Args:
        branch (str): Name of the branch to checkout.
        create (bool): If True, creates the branch if it doesn't exist.
    
    Returns:
        str: JSON with success status and current branch name.
    """
    args = {
        "branch": branch,
        "create": create
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "checkout_branch", args)


async def get_git_status() -> str:
    """
    Gets the current git repository status.
    
    Returns:
        str: JSON with current branch, changed files, staged files, and status.
    """
    return await call_mcp_tool(MCP_SERVER_NAME, "get_git_status", {})


async def git_commit(message: str, stage_all: bool = True, files: Optional[List[str]] = None) -> str:
    """
    Commits changes to the current branch.
    
    Args:
        message (str): Commit message (follows conventional commits if possible).
        stage_all (bool): If True, stages all changes before committing.
        files (list, optional): Specific files to stage and commit.
    
    Returns:
        str: JSON with commit hash and status.
    """
    args = {
        "message": message,
        "stageAll": stage_all,
        "files": files
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "git_commit", args)


async def execute_sandbox_code(
    code: str,
    language: str = "python",
    timeout: int = 30,
    allowed_imports: Optional[List[str]] = None
) -> str:
    """
    Executes code in an isolated sandbox environment for testing.
    
    This is the key tool for testing code fixes before committing.
    The sandbox provides isolation and security validation.
    
    Args:
        code (str): The code to execute.
        language (str): Programming language (default: 'python').
        timeout (int): Maximum execution time in seconds (default: 30).
        allowed_imports (list, optional): List of allowed module imports.
    
    Returns:
        str: JSON with stdout, stderr, return_code, and execution_time.
    """
    args = {
        "code": code,
        "language": language,
        "timeout": timeout,
        "allowedImports": allowed_imports
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "execute_sandbox_code", args)


async def run_genesis_test(quick: bool = True, report: bool = False) -> str:
    """
    Runs the Genesis test suite to validate system health.
    
    Genesis tests are comprehensive validation tests for the AitherZero/AitherOS
    ecosystem. Use this after making changes to verify nothing is broken.
    
    Args:
        quick (bool): If True, runs quick tests only (default).
        report (bool): If True, generates a detailed report.
    
    Returns:
        str: JSON with test results, pass/fail counts, and any errors.
    """
    args = {
        "quick": quick,
        "report": report
    }
    return await call_mcp_tool(MCP_SERVER_NAME, "run_genesis_test", args)

async def execute_agent_workflow(workflow_name: str, variables: Optional[dict] = None) -> str:
    """
    Executes an Agent Workflow (multi-step orchestration) defined in YAML.
    Workflows are stored in library/agent-workflows/.

    Note: This runs locally using the Python WorkflowEngine, not via MCP server yet.
    """
    try:
        root = os.environ.get("AITHERZERO_ROOT", PROJECT_ROOT)
        path = os.path.join(root, "library/agent-workflows", f"{workflow_name}.yaml")

        if WorkflowEngine is None:
            return "Error: WorkflowEngine not available."

        engine = WorkflowEngine(mcp_client=True)
        result = await engine.execute_workflow(path, variables)
        return f"Workflow completed.\nResult Context: {json.dumps(result, indent=2)}"
    except Exception as e:
        return f"Error executing workflow: {str(e)}"

# Export as FunctionTools
aither_tools = [
    FunctionTool(execute_aither_script),
    FunctionTool(get_script_info),
    FunctionTool(list_automation_scripts),
    FunctionTool(list_playbooks),
    FunctionTool(execute_aither_playbook),
    FunctionTool(execute_agent_workflow),
    FunctionTool(get_automation_help),
    FunctionTool(get_aither_config),
    FunctionTool(set_aither_config),
    FunctionTool(manage_mcp_server),
    # Development Workflow Tools
    FunctionTool(create_dev_branch),
    FunctionTool(checkout_branch),
    FunctionTool(get_git_status),
    FunctionTool(git_commit),
    FunctionTool(execute_sandbox_code),
    FunctionTool(run_genesis_test),
    # MCP Client Tools
    FunctionTool(list_mcp_servers),
    FunctionTool(list_mcp_tools),
    FunctionTool(call_mcp_tool)
]
