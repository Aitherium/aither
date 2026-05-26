from google.adk import Agent
from google.genai import types

def build_persona_instruction(name, base_instruction, group_chat_manager, prompts, use_local_models=False, persona_config=None):
    """
    Constructs the full instruction string for a persona agent.
    persona_config: Dict with keys 'mode', 'tone', 'style', 'length'.
    """
    instruction = base_instruction

    # Apply Config Overrides if present
    if persona_config:
        # Mode
        mode = persona_config.get('mode', 'conversation')
        if mode == 'deep_thinking':
            instruction += "\n\n" + prompts.get('DEEP_THINKING_INSTRUCTION', '')
        else:
            instruction += "\n\n" + prompts.get('CONVERSATION_MODE_INSTRUCTION', '')
            
        # Tone
        tone = persona_config.get('tone')
        if tone:
            tone_prompts = prompts.get('TONE_INSTRUCTIONS', {})
            if tone in tone_prompts:
                 instruction += f"\n\n[TONE: {tone.upper()}]\n" + tone_prompts[tone]
                 
        # Style
        style = persona_config.get('style')
        if style:
            style_prompts = prompts.get('STYLE_INSTRUCTIONS', {})
            if style in style_prompts:
                 instruction += f"\n\n[STYLE: {style.upper()}]\n" + style_prompts[style]

        # Length
        length = persona_config.get('length')
        if length:
            length_prompts = prompts.get('RESPONSE_LENGTH_INSTRUCTIONS', {})
            if length in length_prompts:
                 instruction += f"\n\n[LENGTH: {length.upper()}]\n" + length_prompts[length]

    # 1. Roleplay Context (Conditional)
    if "FICTIONAL ROLEPLAY MODE" not in instruction and "fictional context" not in instruction:
        roleplay_override = prompts.get("ROLEPLAY_OVERRIDE", "").format(name=name)
        instruction = roleplay_override + "\n\n" + instruction

    # 2. Tool Usage Instructions (MANDATORY)
    instruction = prompts.get("TOOL_USAGE_INSTRUCTION", "") + "\n\n" + instruction

    # Inject Image Generation Instructions
    if "generate_image" not in instruction:
        instruction += "\n\n" + prompts.get("IMAGE_INSTRUCTION", "")

    # Inject Group Chat Context
    if group_chat_manager.state["active"]:
        instruction += "\n\n[CONTEXT: GROUP CHAT]\n"
        instruction += f"You are participating in a group chat. Your name is '{name}'.\n"
        instruction += "If you believe your contribution to the current conversation topic is complete or you wish to leave the discussion, append `[EXIT]` to your response.\n"

        if group_chat_manager.state.get("free_chat"):
             instruction += "Free chat mode is ON. You are in a group with other agents. You should ONLY respond if the user's message is relevant to your specific persona, expertise, or role. If the message is clearly directed at another agent or topic outside your domain, DO NOT respond or use tools. Silence is preferred over irrelevant engagement.\n"
        else:
             instruction += "Mention-Only mode is ON. You have been explicitly mentioned (@name). Respond directly to the user or the agent who mentioned you. Do not respond to messages not addressed to you.\n"

    # [LOCAL MODEL TOOL GUIDE]
    if use_local_models:
        instruction += "\n\n" + prompts.get("LOCAL_TOOL_GUIDE", "")

    return instruction

def configure_persona_tools(name, base_tools, special_tools):
    """
    Configures the tools list for the persona, including setting state if needed.
    """
    tools = base_tools.copy()

    generate_image = special_tools.get("generate_image")
    generate_narrative_response = special_tools.get("generate_narrative_response")
    continue_scene = special_tools.get("continue_scene")

    # Image Generation Tools
    if generate_image:
        # Set persona context in state manager
        try:
            from AitherOS.agents.common.tools.state_manager import state_manager
            state_manager.set_persona(name)
        except ImportError:
            pass

        # Add image related tools if they exist in special_tools
        # We assume base_tools already contains general tools,
        # but the original code added specific image tools here.
        # Let's follow the original logic:
        # tools += [generate_image, generate_video_with_fal, ...]

        # To be safe and generic, we expect the caller to pass a list of image tools
        # in special_tools['image_tools_list'] if they want them added specifically,
        # or we just add what's in special_tools that matches.

        if "image_tools_list" in special_tools:
            # Filter None values just in case
            image_tools = [t for t in special_tools["image_tools_list"] if t is not None]
            tools.extend(image_tools)

    # Narrative Tools
    if generate_narrative_response and continue_scene:
        tools.extend([generate_narrative_response, continue_scene])

    return tools

def create_persona_agent(
    name,
    model_name,
    persona_data,
    group_chat_manager,
    base_tools,
    special_tools,
    safety_settings,
    prompts,
    use_local_models=False,
    persona_config=None
):
    """
    Creates an Agent instance for a specific persona.

    Args:
        name: Name of the persona.
        model_name: Model to use.
        persona_data: Dictionary containing persona configuration (instruction, description).
        group_chat_manager: GroupChatManager instance.
        base_tools: List of basic tools available to the agent.
        special_tools: Dictionary of special tools (generate_image, etc.) and lists (image_tools_list).
        safety_settings: Safety settings object.
        prompts: Dictionary of prompt templates.
        use_local_models: Boolean.
        persona_config: Optional configuration for mode, tone, style, length.
    """
    if not persona_data:
        return None

    instruction = persona_data.get("instruction", "You are a helpful assistant.")

    # Build Instruction
    full_instruction = build_persona_instruction(
        name,
        instruction,
        group_chat_manager,
        prompts,
        use_local_models,
        persona_config
    )

    # Configure Tools
    tools = configure_persona_tools(name, base_tools, special_tools)

    return Agent(
        name=name,
        model=model_name,
        instruction=full_instruction,
        tools=tools,
        description=persona_data.get("description", ""),
        generate_content_config=types.GenerateContentConfig(
            safety_settings=safety_settings
        )
    )
