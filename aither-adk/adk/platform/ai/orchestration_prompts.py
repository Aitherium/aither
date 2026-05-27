"""
Orchestration-Optimized Prompts for NVIDIA Orchestrator-8B

This module provides system prompts and tool descriptions optimized for the
Orchestrator-8B model's tool-calling capabilities. The prompts are designed to:

1. Be concise and direct (Orchestrator prefers brevity)
2. Use structured JSON format for tool parameters
3. Provide clear decision criteria for tool selection
4. Minimize ambiguity in multi-step workflows

Usage:
    from orchestration_prompts import (
        get_orchestrator_system_prompt,
        get_tool_selection_prompt,
        format_tool_for_orchestrator
    )

    # Get system prompt for an agent
    system_prompt = get_orchestrator_system_prompt("automation")

    # Format tools for orchestrator
    tools = [format_tool_for_orchestrator(tool) for tool in raw_tools]
"""

from typing import Dict, List, Any, Optional


# ===============================================================================
# CORE SYSTEM PROMPTS
# ===============================================================================

ORCHESTRATOR_BASE_PROMPT = """You are an AI orchestration agent. Your role is to:
1. Understand the user's intent
2. Select appropriate tools to accomplish tasks
3. Execute tools in the correct sequence
4. Handle errors gracefully and retry when appropriate

CRITICAL RULES:
- Always use tools when they can accomplish the task
- Never fabricate tool results - execute them
- If a tool fails, try an alternative approach
- Report completion status clearly

OUTPUT FORMAT:
- For tool calls: Use the function calling format
- For responses: Be concise and actionable
- For errors: Explain what failed and suggest fixes"""


AGENT_PROMPTS = {
    "automation": """You are the AitherZero Automation Agent, an expert in infrastructure automation.

CAPABILITIES:
- Execute PowerShell scripts and commands
- Manage system configuration
- Deploy and configure services
- Validate system state

TOOL USAGE:
- Use `run_powershell` for script execution
- Use `read_file` / `write_file` for configuration
- Use `check_status` to verify operations
- Chain tools for complex workflows

Always verify before modifying system state.""",

    "narrative": """You are the Narrative Agent, specializing in creative content and media generation.

CAPABILITIES:
- Generate images with ComfyUI/Stable Diffusion
- Create video sequences and animations
- Write creative content and stories
- Design visual scenes and compositions

TOOL USAGE:
- Use `generate_image` for visual creation
- Use `create_video` for animation sequences
- Use `describe_scene` for planning
- Iterate on outputs based on feedback""",

    "infrastructure": """You are the Infrastructure Agent, an expert in system architecture and deployment.

CAPABILITIES:
- Design infrastructure layouts
- Plan deployment strategies
- Configure networking and security
- Optimize resource allocation

TOOL USAGE:
- Use `analyze_topology` for architecture review
- Use `deploy_resource` for provisioning
- Use `validate_config` for verification
- Document all changes made""",

    "council": """You are a member of the AitherCouncil, participating in multi-agent deliberation.

ROLE:
- Provide expert perspective on your domain
- Vote on proposed actions
- Raise concerns about risks
- Suggest improvements

COLLABORATION:
- Listen to other agents' inputs
- Build on good ideas
- Respectfully challenge bad ones
- Seek consensus when possible""",
}


# ===============================================================================
# TOOL FORMATTING FOR ORCHESTRATOR
# ===============================================================================

def format_tool_for_orchestrator(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    examples: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Format a tool definition optimized for Orchestrator-8B.

    Orchestrator performs best with:
    - Concise descriptions (< 100 chars ideal)
    - Clear parameter types and constraints
    - Example usage patterns

    Args:
        name: Tool function name (snake_case)
        description: Brief description of what the tool does
        parameters: JSON Schema for parameters
        examples: Optional list of example calls

    Returns:
        Formatted tool definition dict
    """
    # Truncate description if too long
    if len(description) > 200:
        description = description[:197] + "..."

    tool_def = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }

    # Add examples as part of description if provided
    if examples:
        example_str = " Examples: " + "; ".join([
            f"{name}({', '.join(f'{k}={repr(v)}' for k, v in ex.items())})"
            for ex in examples[:2]  # Limit to 2 examples
        ])
        if len(tool_def["function"]["description"] + example_str) <= 300:
            tool_def["function"]["description"] += example_str

    return tool_def


def simplify_schema_for_orchestrator(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplify a JSON Schema for better Orchestrator performance.

    Orchestrator handles simpler schemas more reliably. This function:
    - Removes verbose descriptions from nested properties
    - Flattens unnecessary nested objects
    - Adds explicit types where missing
    """
    if not schema:
        return {"type": "object", "properties": {}}

    simplified = {"type": schema.get("type", "object")}

    if "properties" in schema:
        simplified["properties"] = {}
        for key, prop in schema["properties"].items():
            simple_prop = {"type": prop.get("type", "string")}

            # Keep description but truncate
            if prop.get("description"):
                desc = prop["description"]
                simple_prop["description"] = desc[:80] if len(desc) > 80 else desc

            # Keep enum values
            if prop.get("enum"):
                simple_prop["enum"] = prop["enum"]

            # Handle arrays
            if prop.get("type") == "array" and prop.get("items"):
                simple_prop["items"] = {"type": prop["items"].get("type", "string")}

            simplified["properties"][key] = simple_prop

    if "required" in schema:
        simplified["required"] = schema["required"]

    return simplified


# ===============================================================================
# PROMPT GENERATION
# ===============================================================================

def get_orchestrator_system_prompt(
    agent_type: str = "automation",
    additional_context: Optional[str] = None,
    tools_summary: Optional[List[str]] = None
) -> str:
    """
    Generate a system prompt optimized for Orchestrator-8B.

    Args:
        agent_type: Type of agent (automation, narrative, infrastructure, council)
        additional_context: Extra context to include
        tools_summary: List of available tool names for quick reference

    Returns:
        Complete system prompt string
    """
    parts = [ORCHESTRATOR_BASE_PROMPT]

    # Add agent-specific prompt
    if agent_type in AGENT_PROMPTS:
        parts.append(AGENT_PROMPTS[agent_type])

    # Add tools summary if provided
    if tools_summary:
        tools_list = ", ".join(tools_summary[:20])  # Limit to 20 tools
        parts.append(f"\nAVAILABLE TOOLS: {tools_list}")

    # Add additional context
    if additional_context:
        parts.append(f"\nCONTEXT: {additional_context}")

    return "\n\n".join(parts)


def get_tool_selection_prompt(
    task: str,
    available_tools: List[str],
    constraints: Optional[List[str]] = None
) -> str:
    """
    Generate a prompt to help Orchestrator select the right tools.

    Args:
        task: Description of the task to accomplish
        available_tools: List of tool names available
        constraints: Optional constraints on tool usage

    Returns:
        Tool selection guidance prompt
    """
    prompt = f"""TASK: {task}

AVAILABLE TOOLS: {', '.join(available_tools)}

INSTRUCTIONS:
1. Analyze what the task requires
2. Select the minimum set of tools needed
3. Plan the execution order
4. Execute tools and verify results"""

    if constraints:
        prompt += f"\n\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints)

    return prompt


def get_multi_step_prompt(steps: List[Dict[str, Any]]) -> str:
    """
    Generate a prompt for multi-step orchestration.

    Args:
        steps: List of step definitions with 'action', 'tool', 'expected_result'

    Returns:
        Multi-step execution prompt
    """
    step_lines = []
    for i, step in enumerate(steps, 1):
        action = step.get("action", "Perform action")
        tool = step.get("tool", "unknown")
        expected = step.get("expected_result", "Success")
        step_lines.append(f"{i}. {action} using `{tool}` -> Expected: {expected}")

    return f"""EXECUTION PLAN:
{chr(10).join(step_lines)}

Execute each step in order. If a step fails:
- Log the error
- Attempt recovery if possible
- Report status before continuing"""


# ===============================================================================
# RESPONSE FORMATTING
# ===============================================================================

def format_tool_result_for_context(
    tool_name: str,
    result: Any,
    success: bool = True,
    max_length: int = 500
) -> str:
    """
    Format a tool result for inclusion in conversation context.

    Orchestrator benefits from concise, structured feedback.

    Args:
        tool_name: Name of the tool that was called
        result: The result from the tool
        success: Whether the tool succeeded
        max_length: Maximum length of the formatted result

    Returns:
        Formatted result string
    """
    status = "[OK]" if success else "[X]"

    # Convert result to string
    if isinstance(result, dict):
        import json
        result_str = json.dumps(result, indent=2)
    else:
        result_str = str(result)

    # Truncate if needed
    if len(result_str) > max_length:
        result_str = result_str[:max_length - 20] + "\n...[truncated]"

    return f"[{status} {tool_name}]\n{result_str}"


def create_orchestration_summary(
    task: str,
    steps_completed: List[Dict[str, Any]],
    final_result: Any,
    success: bool
) -> str:
    """
    Create a summary of an orchestration run.

    Args:
        task: Original task description
        steps_completed: List of steps that were executed
        final_result: The final result/output
        success: Overall success status

    Returns:
        Summary string
    """
    status = "COMPLETED" if success else "FAILED"

    step_summary = []
    for step in steps_completed:
        name = step.get("tool", "unknown")
        ok = "[OK]" if step.get("success", True) else "[X]"
        step_summary.append(f"  {ok} {name}")

    return f"""ORCHESTRATION {status}
Task: {task}

Steps:
{chr(10).join(step_summary)}

Result: {final_result}"""


# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "ORCHESTRATOR_BASE_PROMPT",
    "AGENT_PROMPTS",
    "format_tool_for_orchestrator",
    "simplify_schema_for_orchestrator",
    "get_orchestrator_system_prompt",
    "get_tool_selection_prompt",
    "get_multi_step_prompt",
    "format_tool_result_for_context",
    "create_orchestration_summary",
]
