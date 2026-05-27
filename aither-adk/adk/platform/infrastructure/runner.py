import asyncio
import hashlib
import json
import time
import warnings

from google.genai import types
from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from aither_adk.infrastructure.cost import CostCalculator
from aither_adk.infrastructure.utils import (
    extract_thinking,
    get_show_thinking,
    strip_ansi,
    strip_thinking,
)
from aither_adk.ui.console import console, safe_print

# Suppress Google SDK warnings globally
warnings.filterwarnings("ignore", message=".*non-text parts.*")
warnings.filterwarnings("ignore", message=".*concatenated text.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google")

# Reasoning integration (uses AitherReasoning service if running on port 8093)
try:
    from aither_adk.ai.reasoning import ThoughtType, with_reasoning
    HAS_REASONING = True
except ImportError:
    HAS_REASONING = False
    with_reasoning = None
    ThoughtType = None

# Ecosystem integration (makes agents EXPERTS at the codebase)
try:
    from aither_adk.infrastructure.ecosystem import EcosystemClient, get_ecosystem_context
    HAS_ECOSYSTEM = True
except ImportError:
    HAS_ECOSYSTEM = False
    EcosystemClient = None
    get_ecosystem_context = None

# Flux Nervous System integration (real-time context)
try:
    import httpx
    HAS_FLUX = True
except ImportError:
    HAS_FLUX = False

# Try to import shared get_flux_context from FluxEmitter (with guaranteed time fallback)
try:
    import sys
    from pathlib import Path
    # Add AitherNode lib to path if not already there
    aither_node_lib = Path(__file__).parent.parent.parent.parent.parent.parent / "AitherNode" / "lib"
    if str(aither_node_lib) not in sys.path:
        sys.path.insert(0, str(aither_node_lib))
    from FluxEmitter import get_flux_context as _shared_get_flux_context
    _HAS_SHARED_FLUX = True
except ImportError:
    _HAS_SHARED_FLUX = False

import logging

logger = logging.getLogger(__name__)


async def get_flux_context() -> str:
    """
    Get real-time nervous system context from AitherFlux.
    Uses LOCAL DynamicPromptBuilder first for instant access.
    Falls back to shared FluxEmitter or HTTP as needed.
    """
    # Try LOCAL DynamicPromptBuilder first (instant, no HTTP)
    try:
        import sys
        from pathlib import Path
        aither_os = Path(__file__).parent.parent.parent.parent.parent.parent
        if str(aither_os) not in sys.path:
            sys.path.insert(0, str(aither_os))
        from lib.core.DynamicPromptBuilder import get_instant_context
        result = get_instant_context()
        if result:
            return result
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"DynamicPromptBuilder instant context failed: {exc}")

    # Try shared FluxEmitter if available
    if _HAS_SHARED_FLUX:
        return await _shared_get_flux_context()

    # Fallback to HTTP if FluxEmitter not available
    if not HAS_FLUX:
        return ""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:8098/context/flux/yaml")
            if response.status_code == 200:
                data = response.json()
                return data.get("yaml", "")
    except Exception as exc:
        logger.debug(f"Flux context HTTP fetch failed: {exc}")
    return ""

async def emit_flux_event(event_type: str, **kwargs):
    """Emit an event to the nervous system."""
    if not HAS_FLUX:
        return
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.post(
                "http://localhost:8098/context/flux/event",
                params={"event_type": event_type},
                json=kwargs
            )
    except Exception as exc:
        logger.debug(f"Flux event emission failed: {exc}")

# Global flag for showing reasoning traces in CLI
SHOW_REASONING_TRACES = False

# Global flag for showing detailed sub-thoughts
SHOW_SUBTHOUGHTS = True

# Global flag for ecosystem injection depth
ECOSYSTEM_INJECTION_LEVEL = "standard"  # "minimal", "standard", "full"

def set_show_reasoning(show: bool):
    """Enable/disable reasoning trace display in CLI."""
    global SHOW_REASONING_TRACES
    SHOW_REASONING_TRACES = show

def get_show_reasoning() -> bool:
    """Check if reasoning traces should be shown."""
    return SHOW_REASONING_TRACES

def set_show_subthoughts(show: bool):
    """Enable/disable detailed sub-thought display."""
    global SHOW_SUBTHOUGHTS
    SHOW_SUBTHOUGHTS = show

def get_show_subthoughts() -> bool:
    """Check if sub-thoughts should be shown."""
    return SHOW_SUBTHOUGHTS

def set_ecosystem_level(level: str):
    """Set ecosystem context injection level: minimal, standard, full."""
    global ECOSYSTEM_INJECTION_LEVEL
    if level in ("minimal", "standard", "full"):
        ECOSYSTEM_INJECTION_LEVEL = level

def get_ecosystem_level() -> str:
    """Get current ecosystem injection level."""
    return ECOSYSTEM_INJECTION_LEVEL

# Dynamic Agent Colors - On-brand cyan/pastel pink theme
def get_agent_color(name):
    """Generates a consistent color for an agent name."""
    if name == "User":
        return "bright_cyan"  # Cyan for user
    if name == "Aither":
        return "#ffb3d9"  # Pastel pink for Aither

    colors = [
        "cyan", "#ffb3d9", "bright_cyan", "#ffd6eb",
        "blue", "bright_blue", "yellow", "bright_yellow"
    ]
    # Deterministic hash
    hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return colors[hash_val % len(colors)]

# Global State for Attachments
PENDING_ATTACHMENTS = []

def add_attachment(part):
    global PENDING_ATTACHMENTS
    PENDING_ATTACHMENTS.append(part)

class LiveStatus:
    def __init__(self, spinner, toolbar_renderer):
        self.spinner = spinner
        self.toolbar_renderer = toolbar_renderer

    def __rich__(self):
        if self.toolbar_renderer:
            return Group(self.spinner, self.toolbar_renderer())
        return self.spinner


def _format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


async def process_turn(runner, user_id, session_id, user_input, model_name, session_stats, root_agent=None, debug_mode=False, memory_manager=None, show_spinner=True, toolbar_renderer=None, mailbox=None):
    """Processes a single turn of conversation."""
    global PENDING_ATTACHMENTS

    # Track timing for this turn
    turn_start_time = time.time()
    tool_times = {}  # Track time spent in each tool
    nodes_used = set()  # Track which nodes/services were used

    # Start reasoning session if AitherReasoning is available
    reasoning_session = None
    if HAS_REASONING and with_reasoning:
        agent_name = root_agent.name if root_agent else "Agent"
        reasoning_session = await with_reasoning(agent_name, user_input)
        if reasoning_session:
            if SHOW_REASONING_TRACES:
                safe_print(f"[dim cyan][BRAIN] Reasoning session started for {agent_name}[/]")
                safe_print(f"[dim cyan] Query: {user_input[:100]}{'...' if len(user_input) > 100 else ''}[/]")
            elif debug_mode:
                safe_print("[dim][BRAIN] AitherReasoning session started[/]")
        reasoning_session = await with_reasoning(agent_name, user_input)
        if reasoning_session:
            if SHOW_REASONING_TRACES:
                safe_print(f"[dim cyan][BRAIN] Reasoning session started for {agent_name}[/]")
                safe_print(f"[dim cyan] Query: {user_input[:100]}{'...' if len(user_input) > 100 else ''}[/]")
            elif debug_mode:
                safe_print("[dim][BRAIN] AitherReasoning session started[/]")

    # Memory Retrieval
    memory_context = ""
    if memory_manager:
        try:
            memory_context = memory_manager.get_context_string(user_input)
            if memory_context:
                safe_print("[dim][BRAIN] Retrieved relevant memories...[/]")
        except Exception as e:
            safe_print(f"[warning]Memory retrieval failed: {e}[/]")

    # Mailbox Context
    mailbox_context = ""
    if mailbox:
        unread = mailbox.get_unread_count("user")
        if unread > 0:
            # Inject a subtle hint so the model knows about pending messages
            # This helps context if the user refers to "the message"
            mailbox_context = f"\n[System: User has {unread} unread messages in /inbox]"

    #  AitherPulse - Ecosystem Awareness
    # Inject pain signals and ecosystem state so agents can "feel" the system
    ecosystem_context = ""
    try:
        if HAS_ECOSYSTEM and get_ecosystem_context:
            # Get comprehensive ecosystem awareness
            level = get_ecosystem_level()
            if level == "full":
                ecosystem_context = get_ecosystem_context(
                    agent_id=root_agent.name if root_agent else "agent",
                    include_codebase=True,
                    include_teachings=True,
                    include_status=True,
                    context_hint=user_input
                )
            elif level == "standard":
                ecosystem_context = get_ecosystem_context(
                    agent_id=root_agent.name if root_agent else "agent",
                    include_codebase=False,
                    include_teachings=True,
                    include_status=True,
                    context_hint=user_input
                )
            elif level == "minimal":
                # Just status, no teachings or codebase
                from aither_adk.infrastructure.ecosystem import get_ecosystem_status
                status = get_ecosystem_status()
                ecosystem_context = status.to_prompt_context()

            if ecosystem_context and debug_mode:
                safe_print(f"[dim][WEB] Ecosystem context injected ({level})[/]")
        else:
            # Fallback to old pulse-only injection
            from aither_adk.infrastructure.services import get_pulse_client
            pulse = get_pulse_client()
            if pulse:
                ecosystem_summary = await pulse.get_ecosystem_summary()
                if ecosystem_summary:
                    ecosystem_context = f"\n{ecosystem_summary}"
                    if debug_mode:
                        safe_print("[dim] Ecosystem pulse injected[/]")
    except Exception as e:
        if debug_mode:
            safe_print(f"[dim]Ecosystem unavailable: {e}[/]")

    # [BRAIN] Flux Nervous System - Real-time system state
    # Agents know service health, affect state, recent activity without asking
    flux_context = ""
    try:
        flux_ctx = await get_flux_context()
        if flux_ctx:
            flux_context = f"\n[SYSTEM STATE]\n{flux_ctx}\n[/SYSTEM STATE]"
            if debug_mode:
                safe_print("[dim][ZAP] Flux nervous system context injected[/]")
        # Emit user message event
        await emit_flux_event("usr.m", msg=user_input[:50])
    except Exception as e:
        if debug_mode:
            safe_print(f"[dim]Flux unavailable: {e}[/]")

    # Construct Content
    # Inject memory context if available
    context_parts = [p for p in [memory_context, mailbox_context, ecosystem_context, flux_context] if p]
    if context_parts:
        final_input = "".join(context_parts) + "\n" + user_input
    else:
        final_input = user_input
    parts = [types.Part(text=final_input)]

    # Add pending attachments
    if PENDING_ATTACHMENTS:
        safe_print(f"[dim]Attaching {len(PENDING_ATTACHMENTS)} file(s) to this turn...[/]")
        parts.extend(PENDING_ATTACHMENTS)
        PENDING_ATTACHMENTS = [] # Clear after using

    content = types.Content(role='user', parts=parts)

    # PUSH ACTIVITY: Agent is processing this turn
    agent_name = root_agent.name if root_agent else "agent"
    try:
        from lib.core.FluxEmitter import clear_agent_activity, inject_agent_activity
        inject_agent_activity(agent_name.lower(), {
            "state": "processing",
            "task": f"Processing turn: {user_input[:50]}...",
            "will": "default",
            "session_id": session_id,
        })
    except ImportError:
        pass
    except Exception as e:
        if debug_mode:
            safe_print(f"[dim]Failed to push agent activity: {e}[/]")

    full_response = ""
    active_tool = None
    turn_usage = {"input": 0, "output": 0}
    stop_reason = None
    current_author = None
    last_printed_len = 0

    # We use one Live display for the whole turn (optional)
    if show_spinner:
        spinner = Spinner("dots", text="[dim]Thinking...[/]", style="#ffb3d9")
        renderable = LiveStatus(spinner, toolbar_renderer)
        # Use global console to prevent fighting for stdout
        # Set transient=True to remove the spinner/toolbar after the turn is done (prevents ghosting)
        # [FIX] transient=True causes the toolbar to disappear, but transient=False causes ghosting if not cleared.
        # The issue of spamming is likely due to external prints or wrapping.
        # We will try transient=True but ensure we print the final result.
        live = Live(renderable, console=console, refresh_per_second=12, transient=True)
        live.start()
    else:
        live = None

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if debug_mode:
                # Convert event to string safely
                event_str = str(event)[:500] + "..." if len(str(event)) > 500 else str(event)
                if live:
                    live.console.print(f"[dim]Debug Event: {event_str}[/]")
                else:
                    safe_print(f"[dim]Debug Event: {event_str}[/]")

            # Check for errors
            if (hasattr(event, 'error_message') and event.error_message) or (hasattr(event, 'error_code') and event.error_code):
                msg = event.error_message if event.error_message else "Unknown Error"
                code = event.error_code if event.error_code else "Unknown Code"

                # Provide helpful error explanations
                explanation = ""
                if "UNEXPECTED_TOOL_CALL" in str(code):
                    explanation = "\n[dim yellow][TIP] The model tried to call a tool when it shouldn't have. This can happen with newer Gemini models on simple queries. Try rephrasing or using a different model.[/]"
                elif "SAFETY" in str(code).upper():
                    explanation = "\n[dim yellow][TIP] Content was blocked by safety filters. Try rephrasing your request.[/]"
                elif "RECITATION" in str(code).upper():
                    explanation = "\n[dim yellow][TIP] The model detected potential copyrighted content. Try being more original in your request.[/]"

                if live:
                    live.console.print(f"[bold red]API Error: {msg} (Code: {code})[/]{explanation}")
                else:
                    safe_print(f"[bold red]API Error: {msg} (Code: {code})[/]{explanation}")
                stop_reason = f"ERROR ({code})"

                # [HISTORY ROLLBACK]
                # If we hit a safety block or API error, the user's message is likely stuck in history
                # without a valid response, which corrupts the context for the next turn.
                # We should remove the last message (the user's input) to reset the state.
                try:
                    # Use keyword arguments for get_session as it seems to require them
                    session = await runner.session_service.get_session(session_id=session_id, user_id=user_id, app_name=runner.app.name)
                    if session and hasattr(session, 'events') and session.events:
                        # Check if the last event is from the user (meaning no response was added)
                        last_event = session.events[-1]
                        if hasattr(last_event, 'user_content') or (hasattr(last_event, 'role') and last_event.role == 'user'):
                            session.events.pop()
                            if live:
                                live.console.print("[dim][WARN] Rolled back session events to prevent context corruption.[/]")
                            else:
                                safe_print("[dim][WARN] Rolled back session events to prevent context corruption.[/]")
                except Exception as rollback_ex:
                    if live:
                        live.console.print(f"[dim]History rollback failed: {rollback_ex}[/]")
                    else:
                        safe_print(f"[dim]History rollback failed: {rollback_ex}[/]")

            # Capture usage metadata if present
            if hasattr(event, 'usage_metadata') and event.usage_metadata:
                turn_usage["input"] = event.usage_metadata.prompt_token_count or turn_usage["input"]
                turn_usage["output"] = event.usage_metadata.candidates_token_count or turn_usage["output"]

            # Check for non-standard finish reasons
            if event.finish_reason:
                stop_reason = event.finish_reason.name
                if event.finish_reason.name != "STOP":
                    if live:
                        live.console.print(f"[warning]Model stopped due to: {event.finish_reason.name}[/]")
                    else:
                        safe_print(f"[warning]Model stopped due to: {event.finish_reason.name}[/]")

                # Track Agent Switching
                if hasattr(event, 'author') and event.author and event.author != 'user':
                    if event.author != current_author:
                        current_author = event.author
                        # Try to find model name
                        if root_agent:
                            if root_agent.name == current_author:
                                pass
                            else:
                                # Search sub-agents
                                found = False
                                if root_agent.sub_agents:
                                    for sub in root_agent.sub_agents:
                                        if sub.name == current_author:
                                            found = True
                                            break
                                if not found:
                                    # Try recursive search if needed, but for now flat list
                                    pass

                        # Minimal switching notice - only if debug or significant change
                        # live.console.print(Rule(f"[bold magenta][SYNC] Switched to {current_author} ({agent_model})[/]", style="dim magenta")) if live else console.print(Rule(f"[bold magenta][SYNC] Switched to {current_author} ({agent_model})[/]", style="dim magenta"))

                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # Handle Tool Execution
                        if part.function_call:
                            active_tool = part.function_call.name
                            tool_args = dict(part.function_call.args) if part.function_call.args else {}
                            tool_start_time = time.time()

                            # Track which node/service is being used
                            if "image" in active_tool.lower() or "canvas" in active_tool.lower():
                                nodes_used.add("ComfyUI")
                            elif "ollama" in active_tool.lower() or "chat" in active_tool.lower():
                                nodes_used.add("Ollama")
                            elif "memory" in active_tool.lower() or "remember" in active_tool.lower() or "recall" in active_tool.lower():
                                nodes_used.add("AitherMind")
                            elif "vision" in active_tool.lower() or "analyze" in active_tool.lower():
                                nodes_used.add("AitherVision")
                            else:
                                nodes_used.add("AitherNode")

                            # Show sub-thoughts if enabled (detailed reasoning before tool call)
                            if SHOW_SUBTHOUGHTS and full_response and full_response.strip():
                                reasoning_text = full_response.strip()
                                if len(reasoning_text) > 20:
                                    # Extract the last few sentences as "sub-thought"
                                    sentences = reasoning_text.split('.')
                                    sub_thought = '. '.join(sentences[-3:]).strip()
                                    if sub_thought:
                                        if live:
                                            live.console.print(f"[dim italic cyan]    {sub_thought[:200]}{'...' if len(sub_thought) > 200 else ''}[/]")
                                        else:
                                            safe_print(f"[dim italic cyan]    {sub_thought[:200]}{'...' if len(sub_thought) > 200 else ''}[/]")

                            # Log REASONING for this tool call (text accumulated before the call)
                            # This is what gives transparency into WHY the agent called this tool
                            if reasoning_session and full_response and full_response.strip():
                                try:
                                    reasoning_text = full_response.strip()
                                    # Only log if there's meaningful reasoning (not just whitespace or very short)
                                    if len(reasoning_text) > 20:
                                        await reasoning_session.think(
                                            f"Deciding to use {active_tool}:\n{reasoning_text[-500:]}",
                                            ThoughtType.REASONING,
                                            confidence=0.7,
                                            metadata={"leads_to_tool": active_tool}
                                        )
                                except Exception as exc:
                                    logger.debug(f"Reasoning session tool decision logging failed: {exc}")

                            # Log tool call to reasoning session AND display if enabled
                            if reasoning_session:
                                try:
                                    await reasoning_session.think(
                                        f"Calling tool: {active_tool}\nArgs: {json.dumps(tool_args, default=str)[:500]}",
                                        ThoughtType.TOOL_CALL,
                                        tool_name=active_tool,
                                        tool_args=tool_args
                                    )
                                    # Show reasoning trace if enabled
                                    if SHOW_REASONING_TRACES:
                                        args_preview = json.dumps(tool_args, default=str)[:200]
                                        if live:
                                            live.console.print(f"[dim cyan] Reasoning: Calling {active_tool}({args_preview})[/]")
                                        else:
                                            safe_print(f"[dim cyan] Reasoning: Calling {active_tool}({args_preview})[/]")
                                except Exception as exc:
                                    logger.debug(f"Reasoning service tool call logging failed: {exc}")

                            if live:
                                live.console.print(f"[bold cyan] {current_author or 'Agent'} is calling tool: {active_tool}[/]")
                            else:
                                safe_print(f"[bold cyan] {current_author or 'Agent'} is calling tool: {active_tool}[/]")
                                # Print accumulated text if any
                                if full_response:
                                    new_text = full_response[last_printed_len:]
                                    if new_text.strip():
                                        # Strip <think> blocks and ANSI codes
                                        clean_text = strip_thinking(strip_ansi(new_text))
                                        if clean_text:  # Only show if there's content after stripping
                                            color = get_agent_color(current_author or 'Aither')
                                            # Use Text instead of Markdown for cleaner chat output
                                            safe_print(Panel(Text(clean_text), title=f"[{color}]{current_author or 'Aither'}[/]", border_style=color, title_align="left", box=box.HEAVY))
                                        last_printed_len = len(full_response)

                            # Render: Text (if any) + Tool Spinner
                            renderables = []
                            if full_response and live:
                                clean_text = strip_thinking(strip_ansi(full_response))
                                if clean_text:  # Only show panel if content remains after stripping
                                    color = get_agent_color(current_author or 'Aither')
                                    renderables.append(Panel(Text(clean_text), title=f"[{color}]{current_author or 'Aither'}[/]", border_style=color, title_align="left", box=box.HEAVY))

                            if live:
                                renderables.append(Spinner("clock", text=f"[tool]Executing {active_tool}...[/]", style="cyan"))
                                # [FIX] Do not update live with renderables here, as it might conflict with the main loop update
                                # Instead, we rely on the next loop iteration or just let the spinner spin.
                                # But we want to show "Executing tool..."
                                # We can update the spinner text in the LiveStatus object if we had access to it.
                                # But LiveStatus holds the spinner.
                                # Let's just print the status line above and let the main spinner continue.
                                pass
                            continue

                        # Handle Tool Response
                        if part.function_response:
                            tool_name = part.function_response.name
                            response_content = part.function_response.response

                            # Track tool execution time
                            if 'tool_start_time' in dir():
                                tool_duration = time.time() - tool_start_time
                                tool_times[tool_name] = tool_times.get(tool_name, 0) + tool_duration

                            # Log tool result to reasoning session AND display if enabled
                            if reasoning_session:
                                try:
                                    result_str = str(response_content)[:1000]
                                    await reasoning_session.think(
                                        f"Tool result from {tool_name}: {result_str}",
                                        ThoughtType.TOOL_RESULT,
                                        tool_name=tool_name,
                                        tool_result=result_str
                                    )
                                    # Show reasoning trace if enabled
                                    if SHOW_REASONING_TRACES:
                                        preview = result_str[:150] + "..." if len(result_str) > 150 else result_str
                                        if live:
                                            live.console.print(f"[dim green] Tool result: {preview}[/]")
                                        else:
                                            safe_print(f"[dim green] Tool result: {preview}[/]")
                                except Exception as exc:
                                    logger.debug(f"Reasoning service tool result logging failed: {exc}")

                            # Extract content
                            content_to_display = response_content
                            if isinstance(response_content, dict) and 'result' in response_content:
                                content_to_display = response_content['result']

                            # Create renderable
                            renderable = str(content_to_display)
                            if isinstance(content_to_display, str):
                                try:
                                    # Try to render ANSI codes if present
                                    renderable = Text.from_ansi(content_to_display)
                                except Exception as exc:
                                    logger.debug(f"ANSI rendering failed: {exc}")
                            elif isinstance(content_to_display, (dict, list)):
                                try:
                                    json_str = json.dumps(content_to_display, indent=2)
                                    from rich.syntax import Syntax
                                    renderable = Syntax(json_str, "json", theme="monokai", word_wrap=True)
                                except Exception:
                                    renderable = str(content_to_display)

                            # [FIX] Only print tool output if it's short or if debug mode is on.
                            # Large outputs (like search results) spam the chat.
                            # We can show a summary instead.
                            str_content = str(content_to_display)
                            if len(str_content) > 500 and not debug_mode:
                                summary = f"[dim italic]Tool output hidden ({len(str_content)} chars). Agent has received it.[/]"
                                if live:
                                    live.console.print(Panel(summary, title=f"[tool]Output: {tool_name}[/]", border_style="dim cyan"))
                                else:
                                    safe_print(Panel(summary, title=f"[tool]Output: {tool_name}[/]", border_style="dim cyan"))
                            else:
                                if live:
                                    live.console.print(Panel(renderable, title=f"[tool]Output: {tool_name}[/]", border_style="dim cyan"))
                                else:
                                    safe_print(Panel(renderable, title=f"[tool]Output: {tool_name}[/]", border_style="dim cyan"))
                            continue

                        # Handle Text Response
                        if part.text:
                            active_tool = None # Tool finished if we are getting text

                            # Typewriter effect
                            for char in part.text:
                                full_response += char
                                if len(full_response) % 2 == 0: # Optimization
                                    # Check if model is in thinking mode (unclosed <think> tag)
                                    is_thinking = '<think>' in full_response.lower() and '</think>' not in full_response.lower()

                                    # Extract thinking and main content
                                    thinking_content, clean_text = extract_thinking(strip_ansi(full_response))

                                    # Show thinking indicator during active thinking
                                    if live and is_thinking:
                                        # Model is actively thinking - show purple thinking indicator
                                        think_text = full_response.split('<think>')[-1] if '<think>' in full_response.lower() else ''
                                        think_preview = think_text[-200:] if len(think_text) > 200 else think_text
                                        renderables = [
                                            Panel(
                                                Text(f" {think_preview}▌", style="dim magenta italic"),
                                                title="[magenta][BRAIN] Thinking...[/]",
                                                border_style="magenta",
                                                title_align="left",
                                                box=box.ROUNDED
                                            )
                                        ]
                                        live.update(Group(*renderables))
                                    elif live and clean_text:
                                        color = get_agent_color(current_author or 'Aither')
                                        renderables = [Panel(Markdown(clean_text + "▌"), title=f"[{color}]{current_author or 'Aither'}[/]", border_style=color, title_align="left")]
                                        live.update(Group(*renderables))
                                    await asyncio.sleep(0.002)

                            # Ensure final update for the chunk - also handle thinking display
                            if live:
                                is_thinking = '<think>' in full_response.lower() and '</think>' not in full_response.lower()
                                thinking_content, clean_text = extract_thinking(strip_ansi(full_response))

                                renderables = []
                                if is_thinking:
                                    # Still in thinking mode
                                    think_text = full_response.split('<think>')[-1] if '<think>' in full_response.lower() else ''
                                    think_preview = think_text[-300:] if len(think_text) > 300 else think_text
                                    renderables.append(Panel(
                                        Text(f" {think_preview}", style="dim magenta italic"),
                                        title="[magenta][BRAIN] Thinking...[/]",
                                        border_style="magenta",
                                        title_align="left",
                                        box=box.ROUNDED
                                    ))
                                elif clean_text:
                                    color = get_agent_color(current_author or 'Aither')
                                    renderables.append(Panel(Markdown(clean_text), title=f"[{color}]{current_author or 'Aither'}[/]", border_style=color, title_align="left"))

                                if renderables:
                                    live.update(Group(*renderables))
    except Exception as e:
        error_str = str(e)

        # Log error to reasoning session
        if reasoning_session:
            try:
                await reasoning_session.error(error_str, exception=e)
            except Exception as exc:
                logger.debug(f"Reasoning session error logging failed: {exc}")

        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
             # Suppress stack trace for quota errors, just re-raise
             raise e

        if live:
            live.console.print(f"[bold red]Exception during generation: {e}[/]")
            import traceback
            live.console.print(f"[dim]{traceback.format_exc()}[/]")
        else:
            safe_print(f"[bold red]Exception during generation: {e}[/]")
            import traceback
            safe_print(f"[dim]{traceback.format_exc()}[/]")

        # [HISTORY ROLLBACK ON EXCEPTION]
        try:
            session = await runner.session_service.get_session(session_id=session_id, user_id=user_id, app_name=runner.app.name)
            if session and hasattr(session, 'events') and session.events:
                # Check if the last event is from the user (meaning no response was added)
                last_event = session.events[-1]
                if hasattr(last_event, 'user_content') or (hasattr(last_event, 'role') and last_event.role == 'user'):
                    session.events.pop()
                    if live:
                        live.console.print("[dim][WARN] Rolled back session events (exception) to prevent context corruption.[/]")
                    else:
                        safe_print("[dim][WARN] Rolled back session events (exception) to prevent context corruption.[/]")
        except Exception as exc:
            logger.debug(f"Session event rollback failed: {exc}")

        raise e # Re-raise to allow caller to handle (e.g. fallback)
    finally:
        if live:
            live.stop()

        # Ensure final text is printed
        if full_response:
            new_text = full_response[last_printed_len:]
            if new_text.strip():
                # Strip <think> blocks and ANSI codes for cleaner output
                thinking, main_text = extract_thinking(strip_ansi(new_text))

                # Always show thinking panel so users see the model was working
                # Uses magenta/purple theme to match the live thinking indicator
                if thinking and get_show_thinking():
                    safe_print(Panel(
                        Text(thinking, style="dim magenta italic"),
                        title="[magenta][BRAIN] Thoughts[/]",
                        border_style="magenta",
                        title_align="left",
                        box=box.ROUNDED
                    ))

                # Show main response
                if main_text:
                    color = get_agent_color(current_author or 'Aither')
                    # Use Text instead of Markdown, and HEAVY box for chat
                    safe_print(Panel(Text(main_text), title=f"[{color}]{current_author or 'Aither'}[/]", border_style=color, title_align="left", box=box.HEAVY))


    if not full_response and stop_reason != "STOP":
        safe_print(f"[bold red]No response generated. Stop reason: {stop_reason}[/]")

    # Conclude reasoning session with final response
    if reasoning_session and full_response:
        try:
            await reasoning_session.conclude(full_response[:2000], confidence=0.8)
            if SHOW_REASONING_TRACES:
                safe_print(f"[dim green][BRAIN] Reasoning complete - response generated ({len(full_response)} chars)[/]")
            elif debug_mode:
                safe_print("[dim][BRAIN] AitherReasoning session concluded[/]")
        except Exception as exc:
            logger.debug(f"Reasoning session conclusion failed: {exc}")

    # Fallback for missing usage metadata (e.g. Ollama or some Gemini responses)
    if turn_usage["input"] == 0 and final_input:
        turn_usage["input"] = int(len(final_input) / 4)

    if turn_usage["output"] == 0 and full_response:
        turn_usage["output"] = int(len(full_response) / 4)

    # Calculate and display cost
    cost = CostCalculator.calculate(model_name, turn_usage["input"], turn_usage["output"])
    session_stats["total_cost"] += cost
    session_stats["total_input"] += turn_usage["input"]
    session_stats["total_output"] += turn_usage["output"]
    session_stats["last_response"] = full_response

    # Calculate total turn duration
    turn_duration = time.time() - turn_start_time

    # Build timing info string
    timing_info = f"[TIMER] {_format_duration(turn_duration)}"
    if tool_times:
        tool_breakdown = ", ".join([f"{t}: {_format_duration(d)}" for t, d in sorted(tool_times.items(), key=lambda x: -x[1])[:3]])
        timing_info += f" (tools: {tool_breakdown})"

    # Build nodes info string
    nodes_info = ""
    if nodes_used:
        nodes_info = f" | [PLUG] {', '.join(sorted(nodes_used))}"

    safe_print(f"[dim]{timing_info}{nodes_info}[/]")
    safe_print(f"[dim]Usage: {turn_usage['input']} in / {turn_usage['output']} out | Cost: ${cost:.6f} (Total: ${session_stats['total_cost']:.6f})[/]")

    # Context Limit Warning (assuming 1M limit for safety, though Pro has 2M)
    if turn_usage["input"] > 900000:
            safe_print("[bold red]WARNING: Approaching context limit (1M tokens). Consider restarting the session.[/]")

    # CLEAR ACTIVITY: Turn complete
    try:
        from lib.core.FluxEmitter import clear_agent_activity
        clear_agent_activity(agent_name.lower())
    except ImportError:
        pass
    except Exception as e:
        if debug_mode:
            safe_print(f"[dim]Failed to clear agent activity: {e}[/]")

    return stop_reason
