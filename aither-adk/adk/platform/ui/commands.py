import logging
import os
import sys

import yaml
from google.genai import types
from prompt_toolkit.completion import Completer, Completion
from rich.panel import Panel

from aither_adk.infrastructure.profiling import Profiler, profile
from aither_adk.infrastructure.runner import add_attachment, process_turn
from aither_adk.ui.console import console, print_banner, safe_print

logger = logging.getLogger(__name__)

class SlashCommandCompleter(Completer):
    def __init__(self, models, personas):
        self.models = models
        self.personas = personas

    @profile(name="SlashCommandCompleter.get_completions")
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # List of available commands - organized logically
        commands = [
            # Help & Navigation
            "/help", "/keys", "/clear", "/exit", "/quit",
            # Model & Resources
            "/model", "/models", "/vram", "/resources", "/load", "/unload",
            # Agent & Persona
            "/role", "/persona-info", "/create-persona", "/init-personas", "/group", "/free-chat",
            # Routing
            "/route",
            # Council (Group Chat via AitherCouncil API)
            "/council",
            # Google A2A Protocol
            "/a2a",
            # Safety & Mode
            "/mode", "/safety", "/detail", "/reasoning", "/debug", "/tone", "/style", "/thinking",
            # Image Generation & Vision
            "/gen", "/scene", "/anchor", "/comic", "/vision",
            # Memory
            "/memory", "/remember",
            # Communication
            "/inbox", "/mail", "/search",
            # Quality of Life Tools
            "/notebook", "/calendar", "/timer", "/stopwatch", "/alarm",
            # System & Info
            "/settings", "/cost", "/history", "/profile", "/attach", "/turns", "/continue"
        ]

        # Check if we are starting a command
        if text.startswith("/"):
            parts = text.split()

            # If we are typing the command itself (e.g. "/" or "/mo")
            if len(parts) <= 1 and not text.endswith(" "):
                for cmd in commands:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text))

            # If we are typing arguments for /model
            elif parts[0] == "/model" and (len(parts) > 1 or text.endswith(" ")):
                # Get the current word being typed (if any)
                current_word = parts[-1] if not text.endswith(" ") else ""

                for model in self.models:
                    if model.startswith(current_word):
                        yield Completion(model, start_position=-len(current_word))

            # If we are typing arguments for /role
            elif parts[0] == "/role" and (len(parts) > 1 or text.endswith(" ")):
                current_word = parts[-1] if not text.endswith(" ") else ""
                for role in self.personas:
                    if role.startswith(current_word):
                        yield Completion(role, start_position=-len(current_word))

            # If we are typing arguments for /attach
            elif parts[0] == "/attach" and (len(parts) > 1 or text.endswith(" ")):
                current_word = parts[-1] if not text.endswith(" ") else ""
                # Suggest "last"
                if "last".startswith(current_word):
                    yield Completion("last", start_position=-len(current_word))

                # Suggest files in current directory
                import glob
                for file in glob.glob(current_word + "*"):
                    yield Completion(file, start_position=-len(current_word))

def parse_multi_commands(input_str: str) -> list:
    """
    Parse multiple slash commands from a single line.

    Examples:
        "/mode unrestricted /safety low" -> ["/mode unrestricted", "/safety low"]
        "/model gemini-pro" -> ["/model gemini-pro"]
        "/help" -> ["/help"]

    Returns a list of individual command strings.
    """
    input_str = input_str.strip()
    if not input_str.startswith("/"):
        return [input_str]

    # Find all command starts (positions where / appears)
    commands = []
    current_start = 0
    i = 1  # Start after the first /

    while i < len(input_str):
        # Look for space followed by /
        if input_str[i] == '/' and i > 0 and input_str[i-1] == ' ':
            # Found a new command
            cmd = input_str[current_start:i-1].strip()
            if cmd:
                commands.append(cmd)
            current_start = i
        i += 1

    # Add the last command
    last_cmd = input_str[current_start:].strip()
    if last_cmd:
        commands.append(last_cmd)

    return commands if commands else [input_str]


async def handle_command(command, session_stats, agent, runner, session_id, models, personas, debug_mode_callback, memory_manager=None, prompt_session=None, personas_file=None, group_chat_manager=None, groups_file=None, task_manager=None, mailbox=None):
    """Handles slash commands. Supports multiple commands on one line (e.g., '/mode unrestricted /safety low')."""

    # Parse for multiple commands
    commands = parse_multi_commands(command)

    # If multiple commands, process each one
    if len(commands) > 1:
        final_result = None
        for single_cmd in commands:
            result = await _handle_single_command(
                single_cmd, session_stats, agent, runner, session_id, models, personas,
                debug_mode_callback, memory_manager, prompt_session, personas_file,
                group_chat_manager, groups_file, task_manager, mailbox
            )
            if result == "EXIT_SIGNAL":
                return result
            # RECREATE_AGENT should be returned after all commands are processed
            if result == "RECREATE_AGENT":
                final_result = result
        return final_result

    # Single command - process normally
    return await _handle_single_command(
        command, session_stats, agent, runner, session_id, models, personas,
        debug_mode_callback, memory_manager, prompt_session, personas_file,
        group_chat_manager, groups_file, task_manager, mailbox
    )


async def _handle_single_command(command, session_stats, agent, runner, session_id, models, personas, debug_mode_callback, memory_manager=None, prompt_session=None, personas_file=None, group_chat_manager=None, groups_file=None, task_manager=None, mailbox=None):
    """Handles a single slash command."""
    parts = command.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    # Debug: uncomment to diagnose command issues
    # safe_print(f"[dim]Debug: received command='{command}' -> cmd='{cmd}' args={args}[/]")

    if cmd in ["/exit", "/quit"]:
        return "EXIT_SIGNAL"

    if cmd == "/search":
        if not args:
            safe_print("[warning]Usage: /search <query>[/]")
            return

        query = " ".join(args)
        safe_print(f"[dim][SEARCH] Searching for: {query}...[/]")

        # Import here to avoid circular imports
        try:
            from .tools.personal_assistant_tools import web_search
            results = web_search(query)

            # Agentic processing
            safe_print("[dim][?] Analyzing results...[/]")

            prompt = f"""
I have performed a web search for: "{query}"

Here are the results:
{results}

Please analyze these results and generate a comprehensive report answering the user's question: "{query}".
Format the report clearly with headings and bullet points if necessary.
"""
            model_name = getattr(agent, "model", "gemini-2.0-flash-exp")

            await process_turn(runner, "user", session_id, prompt, model_name, session_stats, root_agent=agent, debug_mode=False, memory_manager=memory_manager, mailbox=mailbox)

            # Capture response and send to inbox
            if mailbox and session_stats.get("last_response"):
                report_content = session_stats["last_response"]
                mailbox.send_message("SearchAgent", "user", f"Search Report: {query}", report_content)
                safe_print("[green][DONE] Report generated and sent to inbox.[/]")

        except Exception as e:
            safe_print(f"[red]Search failed: {e}[/]")
        return

    if cmd == "/inbox":
        if not mailbox:
            safe_print("[warning]Mailbox is not enabled.[/]")
            return

        if not args:
            # List messages with pagination (5 per page)
            messages = mailbox.get_messages("user", sort_by="timestamp", sort_order="asc")
            if not messages:
                safe_print("[dim]No messages.[/]")
            else:
                total_messages = len(messages)
                page_size = 5

                # Calculate which messages to show (last 5, since newest is at bottom)
                start_idx = max(0, total_messages - page_size)
                end_idx = total_messages
                page_messages = messages[start_idx:end_idx]

                safe_print(f"[bold cyan]Inbox ({total_messages} messages, showing newest {len(page_messages)}):[/bold cyan]")
                for i, msg in enumerate(page_messages):
                    # Calculate display number (from end of list)
                    display_num = start_idx + i + 1
                    status = "🆕" if not msg['read'] else "  "
                    safe_print(f"{display_num}. {status} [{msg['timestamp'][:16]}] From: {msg['sender']} - {msg['subject']}")

                if total_messages > page_size:
                    safe_print(f"\n[dim]Showing newest {len(page_messages)} of {total_messages} messages.[/]")
                    safe_print("[dim]Type '/inbox <number>' to read, '/inbox search <query>' to search, '/inbox sort <option>' to sort[/]")
                else:
                    safe_print("\n[dim]Type '/inbox <number>' to read, '/inbox search <query>' to search, '/inbox sort <option>' to sort[/]")
        elif args[0].lower() == "search":
            # Search messages
            if len(args) < 2:
                safe_print("[warning]Usage: /inbox search <query>[/]")
                safe_print("[dim]Example: /inbox search selfie[/]")
                return

            query = " ".join(args[1:])
            messages = mailbox.get_messages("user", search_query=query, sort_by="timestamp", sort_order="desc")

            if not messages:
                safe_print(f"[dim]No messages found matching '{query}'[/]")
            else:
                safe_print(f"[bold cyan]Search results for '{query}' ({len(messages)} messages):[/bold cyan]")
                for i, msg in enumerate(messages[:20]):  # Show up to 20 results
                    status = "🆕" if not msg['read'] else "  "
                    safe_print(f"{i+1}. {status} [{msg['timestamp'][:16]}] From: {msg['sender']} - {msg['subject']}")
                if len(messages) > 20:
                    safe_print(f"[dim]... and {len(messages) - 20} more. Use '/inbox <number>' to read.[/]")
                else:
                    safe_print("[dim]Type '/inbox <number>' to read a message.[/]")
        elif args[0].lower() == "sort":
            # Sort messages
            sort_options = {
                "newest": ("timestamp", "desc"),
                "oldest": ("timestamp", "asc"),
                "sender": ("sender", "asc"),
                "unread": ("read", "asc"),  # Unread first
                "read": ("read", "desc"),   # Read first
            }

            if len(args) < 2:
                safe_print("[warning]Usage: /inbox sort <option>[/]")
                safe_print(f"[dim]Options: {', '.join(sort_options.keys())}[/]")
                return

            sort_option = args[1].lower()
            if sort_option not in sort_options:
                safe_print(f"[red]Invalid sort option. Use: {', '.join(sort_options.keys())}[/]")
                return

            sort_by, sort_order = sort_options[sort_option]
            messages = mailbox.get_messages("user", sort_by=sort_by, sort_order=sort_order)

            if not messages:
                safe_print("[dim]No messages.[/]")
            else:
                safe_print(f"[bold cyan]Inbox sorted by {sort_option} ({len(messages)} messages, showing first 10):[/bold cyan]")
                for i, msg in enumerate(messages[:10]):
                    status = "🆕" if not msg['read'] else "  "
                    safe_print(f"{i+1}. {status} [{msg['timestamp'][:16]}] From: {msg['sender']} - {msg['subject']}")
                if len(messages) > 10:
                    safe_print(f"[dim]... and {len(messages) - 10} more. Use '/inbox <number>' to read.[/]")
                else:
                    safe_print("[dim]Type '/inbox <number>' to read a message.[/]")
        elif args[0].lower() == "unread":
            # Show only unread messages
            messages = mailbox.get_messages("user", unread_only=True, sort_by="timestamp", sort_order="desc")

            if not messages:
                safe_print("[dim]No unread messages.[/]")
            else:
                safe_print(f"[bold cyan]Unread messages ({len(messages)}):[/bold cyan]")
                # Show ALL unread messages (no limit)
                for i, msg in enumerate(messages):
                    safe_print(f"{i+1}. 🆕 [{msg['timestamp'][:16]}] From: {msg['sender']} - {msg['subject']}")
                safe_print("[dim]Type '/inbox <number>' to read a message.[/]")
        elif args[0].lower() == "from":
            # Filter by sender
            if len(args) < 2:
                safe_print("[warning]Usage: /inbox from <sender>[/]")
                safe_print("[dim]Example: /inbox from Aither[/]")
                return

            sender = " ".join(args[1:])
            messages = mailbox.get_messages("user", sender=sender, sort_by="timestamp", sort_order="desc")

            if not messages:
                safe_print(f"[dim]No messages from '{sender}'[/]")
            else:
                safe_print(f"[bold cyan]Messages from '{sender}' ({len(messages)}):[/bold cyan]")
                for i, msg in enumerate(messages[:10]):
                    status = "🆕" if not msg['read'] else "  "
                    safe_print(f"{i+1}. {status} [{msg['timestamp'][:16]}] - {msg['subject']}")
                if len(messages) > 10:
                    safe_print(f"[dim]... and {len(messages) - 10} more. Use '/inbox <number>' to read.[/]")
                else:
                    safe_print("[dim]Type '/inbox <number>' to read a message.[/]")
        else:
            # Read message by number
            try:
                idx = int(args[0]) - 1
                messages = mailbox.get_messages("user", sort_by="timestamp", sort_order="asc")
                if 0 <= idx < len(messages):
                    msg = messages[idx]
                    safe_print(Panel(msg['content'], title=f"From: {msg['sender']} | Subject: {msg['subject']}", subtitle=msg['timestamp']))
                    mailbox.mark_as_read(msg['id'])
                else:
                    safe_print(f"[red]Invalid message number. Valid range: 1-{len(messages)}[/]")
            except ValueError:
                safe_print("[red]Usage: /inbox <number> | /inbox search <query> | /inbox sort <option> | /inbox unread | /inbox from <sender>[/]")
        return

    if cmd == "/mail":
        """Send a message to multiple agents via /mail @agent1 @agent2 message"""
        if not args:
            safe_print("[warning]Usage: /mail @agent1 @agent2 ... <message>[/]")
            safe_print("[dim]Example: /mail @aither @leo I need you to do the dishes[/]")
            return

        # Extract @mentions and message
        mentions = []
        message_parts = []
        for arg in args:
            if arg.startswith("@"):
                mentions.append(arg[1:].lower())  # Remove @ and lowercase
            else:
                message_parts.append(arg)

        if not mentions:
            safe_print("[warning]No @mentions found. Usage: /mail @agent1 @agent2 ... <message>[/]")
            return

        if not message_parts:
            safe_print("[warning]No message provided. Usage: /mail @agent1 @agent2 ... <message>[/]")
            return

        message = " ".join(message_parts)

        try:
            from aither_adk.communication.multi_agent_chat import dispatch_multi_agent

            # Find mailbox path
            mailbox_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Saga", "mailbox.json"
            )

            # Build message with @mentions
            full_message = f"{' '.join(['@' + m for m in mentions])} {message}"

            safe_print(f"[dim][MSG] Sending to {len(mentions)} agents: {', '.join(mentions)}...[/]")
            count = dispatch_multi_agent(full_message, mailbox_path, agents=mentions)
            safe_print(f"[green][DONE] {count} agents responded! Check /inbox[/]")
        except Exception as e:
            safe_print(f"[red]Mail failed: {e}[/]")
        return

    if cmd == "/todo":
        if not task_manager:
            safe_print("[warning]Task manager is not enabled.[/]")
            return

        if not args:
            safe_print("[bold cyan]Usage:[/bold cyan]")
            safe_print("  /todo add <title>       - Add a new task")
            safe_print("  /todo list [status]     - List tasks (todo, in-progress, done)")
            safe_print("  /todo done <id>         - Mark task as done")
            safe_print("  /todo remove <id>       - Remove a task")
            safe_print("  /todo clear             - Clear completed tasks")
            return

        subcmd = args[0].lower()

        if subcmd == "add":
            if len(args) < 2:
                safe_print("[warning]Usage: /todo add <title>[/]")
                return
            title = " ".join(args[1:])
            task = task_manager.add_task(title)
            safe_print(f"[green]Task added: #{task.id} {task.title}[/]")

        elif subcmd == "list":
            status = args[1] if len(args) > 1 else None
            tasks = task_manager.list_tasks(status)
            if not tasks:
                safe_print("[dim]No tasks found.[/]")
            else:
                safe_print(Panel("[bold cyan]Task List[/]", border_style="cyan"))
                for t in tasks:
                    icon = "[DONE]" if t.status == "done" else "[WAIT]" if t.status == "in-progress" else "[ ]"
                    color = "dim" if t.status == "done" else "white"
                    safe_print(f"[{color}]{icon} [bold]#{t.id}[/bold] {t.title} ({t.status})[/]")

        elif subcmd == "done":
            if len(args) < 2:
                safe_print("[warning]Usage: /todo done <id>[/]")
                return
            try:
                tid = int(args[1])
                task = task_manager.update_status(tid, "done")
                if task:
                    safe_print(f"[green]Task #{tid} marked as done![/]")
                else:
                    safe_print(f"[red]Task #{tid} not found.[/]")
            except ValueError:
                safe_print("[red]Invalid ID.[/]")

        elif subcmd == "remove":
            if len(args) < 2:
                safe_print("[warning]Usage: /todo remove <id>[/]")
                return
            try:
                tid = int(args[1])
                if task_manager.remove_task(tid):
                    safe_print(f"[green]Task #{tid} removed.[/]")
                else:
                    safe_print(f"[red]Task #{tid} not found.[/]")
            except ValueError:
                safe_print("[red]Invalid ID.[/]")

        elif subcmd == "clear":
            task_manager.clear_completed()
            safe_print("[green]Completed tasks cleared.[/]")

        else:
            safe_print(f"[red]Unknown todo command: {subcmd}[/]")

    elif cmd == "/sysinfo":
        safe_print("[dim]Gathering system info...[/]")
        import subprocess
        try:
            # Try to find the script
            script_path = "/workspaces/AitherZero/AitherZero/library/automation-scripts/0011_Get-SystemInfo.ps1"
            if not os.path.exists(script_path):
                 safe_print(f"[red]Script not found: {script_path}[/]")
                 return

            result = subprocess.run(["pwsh", "-File", script_path], capture_output=True, text=True)
            # Clean up output (remove banner if present)
            output = result.stdout.strip()
            safe_print(Panel(output, title="System Info", border_style="blue"))
            if result.stderr:
                safe_print(f"[dim red]Stderr: {result.stderr}[/]")
        except Exception as e:
            safe_print(f"[red]Failed to run sysinfo: {e}[/]")

    elif cmd == "/create-persona":
        if not prompt_session or not personas_file:
            safe_print("[red]Interactive mode not available.[/]")
            return

        safe_print(Panel("[bold cyan]Interactive Persona Builder[/]", border_style="cyan"))

        try:
            # 1. Name
            name = await prompt_session.prompt_async("Enter Persona Name (e.g. 'detective'): ")
            name = name.strip().lower()
            if not name:
                safe_print("[red]Name is required.[/]")
                return
            if name in personas:
                safe_print(f"[warning]Persona '{name}' already exists. Overwriting...[/]")

            # 2. Description
            description = await prompt_session.prompt_async("Enter Description: ")

            # 3. Instruction
            safe_print("[dim]Enter System Instruction (Press Ctrl+Enter to submit multiline):[/]")
            instruction = await prompt_session.prompt_async("Instruction > ", multiline=True)

            # 4. Save
            new_persona = {
                "description": description,
                "instruction": instruction
            }

            # Load existing yaml to preserve comments/structure if possible, but yaml lib might reformat
            # For safety, we read, update, write.
            current_data = {}
            if os.path.exists(personas_file):
                with open(personas_file, 'r') as f:
                    current_data = yaml.safe_load(f) or {}

            current_data[name] = new_persona

            with open(personas_file, 'w') as f:
                yaml.dump(current_data, f, sort_keys=False, default_flow_style=False)

            # Update in-memory dict
            personas[name] = new_persona
            safe_print(f"[green][DONE] Persona '{name}' created and saved![/]")
            safe_print(f"[dim]You can now use it with: /role {name}[/]")

        except KeyboardInterrupt:
            safe_print("\n[warning]Persona creation cancelled.[/]")
            return

    elif cmd == "/free-chat":
        if not group_chat_manager:
            safe_print("[red]Group chat manager not available.[/]")
            return

        current = group_chat_manager.state.get("free_chat", False)
        group_chat_manager.set_free_chat(not current)
        status = "ON" if not current else "OFF"
        color = "green" if not current else "yellow"
        safe_print(f"[bold {color}]Free Chat Mode: {status}[/]")
        safe_print("[dim]When ON, agents will respond to relevant context without @mentions.[/]")

    elif cmd == "/group":
        if not group_chat_manager:
            safe_print("[red]Group chat manager not available.[/]")
            return

        if not args:
            safe_print("[bold cyan]Usage:[/bold cyan] /group <start|stop|switch|list|status> [args]")
            return

        subcmd = args[0].lower()

        if subcmd == "start":
            # /group start <name> <agent1> <agent2> ...
            if len(args) < 3:
                safe_print("[warning]Usage: /group start <name> <agent1> <agent2> ...[/]")
                return

            group_name = args[1]
            agents = args[2:]

            # Validate agents
            valid_agents = []
            for a in agents:
                # Simple fuzzy match against personas
                found = False
                for p in personas:
                    if p.lower() == a.lower():
                        valid_agents.append(p)
                        found = True
                        break
                if not found:
                    # Try partial match
                    for p in personas:
                        if a.lower() in p.lower():
                            valid_agents.append(p)
                            found = True
                            break

                if not found:
                    safe_print(f"[red]Unknown persona: {a}[/]")

            if not valid_agents:
                return

            # Save group
            group_chat_manager.state["groups"][group_name] = valid_agents

            # Persist to file
            if groups_file:
                try:
                    with open(groups_file, 'w') as f:
                        yaml.dump(group_chat_manager.state["groups"], f, sort_keys=False, default_flow_style=False)
                    safe_print(f"[dim]Group saved to {groups_file}[/]")
                except Exception as e:
                    safe_print(f"[red]Failed to save group: {e}[/]")

            # Activate
            success, msg = group_chat_manager.start_group(group_name)
            if success:
                safe_print(f"[green] {msg}[/]")
                safe_print("[dim]Use @mention to talk to specific agents, or /free-chat to enable open conversation.[/]")
            else:
                safe_print(f"[red]{msg}[/]")

        elif subcmd == "switch" or subcmd == "load":
            if len(args) < 2:
                safe_print(f"[warning]Usage: /group {subcmd} <name>[/]")
                return

            group_name = args[1]
            success, msg = group_chat_manager.start_group(group_name)
            if success:
                safe_print(f"[green]Switched to group '{group_name}'[/]")
            else:
                safe_print(f"[red]{msg}[/]")

        elif subcmd == "list":
            if not group_chat_manager.state["groups"]:
                safe_print("[dim]No groups created.[/]")
            else:
                safe_print("[bold cyan]Available Groups:[/]")
                for name, members in group_chat_manager.state["groups"].items():
                    active_marker = "*" if name == group_chat_manager.state.get("active_group_name") and group_chat_manager.state["active"] else " "
                    safe_print(f" {active_marker} [bold #ffb3d9]{name}[/]: {', '.join(members)}")

        elif subcmd == "stop":
            group_chat_manager.stop_group()
            safe_print("[yellow]Group Chat Stopped.[/]")

        elif subcmd == "status":
            status = "Active" if group_chat_manager.state["active"] else "Inactive"
            group_name = group_chat_manager.state.get("active_group_name", "None")
            members = ", ".join(group_chat_manager.state["members"]) if group_chat_manager.state["members"] else "None"
            mode = "Free Chat" if group_chat_manager.state.get("free_chat") else "Mention Only"

            safe_print(f"[bold]Status:[/bold] {status}")
            safe_print(f"[bold]Group:[/bold] {group_name}")
            safe_print(f"[bold]Members:[/bold] {members}")
            safe_print(f"[bold]Mode:[/bold] {mode}")

        else:
            safe_print(f"[red]Unknown group command: {subcmd}[/]")
            safe_print("[dim]Available: start, switch (or load), list, stop, status[/]")

    elif cmd == "/council":
        # Council group chat via AitherCouncil API
        try:
            from aither_adk.communication.council_client import (
                ResponseDepth,
                council,
                format_tool_status,
            )
        except ImportError:
            safe_print("[red]Council client not available. Install httpx: pip install httpx[/]")
            return

        if not args:
            safe_print("""[bold cyan] AitherCouncil - Group Chat & Services[/]

[bold]Chat:[/]
  /council chat <prompt>              Start a council discussion
  /council chat -deep <prompt>        Deep analysis mode
  /council chat -agents a,b,c <prompt> Specify agents

[bold] Settings & Customization:[/]
  /council settings                   Show all settings
  /council toggle <feature>           Toggle feature (web_search, memory, affect, safety, etc.)
  /council depth <level>              Set response depth (concise, thoughtful, deep)
  /council style <type> <value>       Set style (formality/verbosity/creativity)
  /council prompt <text>              Set custom system prompt (prepended)
  /council prompt -override <text>    Set full override prompt (replaces instructions)
  /council prompt clear               Clear custom prompt
  /council reset                      Reset all settings to defaults

[bold][TIMER] Response Timing:[/]
  /council timing preset <preset>     Apply timing preset (instant|fast|natural|thoughtful)
  /council timing eval <min> <max>    Set evaluation delay (seconds)
  /council timing stagger <min> <max> Set response stagger (seconds)
  /council timing typing <on|off>     Toggle typing simulation
  /council timing speed <10-200>      Set typing speed (chars/sec)

[bold] Agent Behavior:[/]
  /council behavior preset <preset>   Apply behavior preset (responsive|natural|selective|chaotic)
  /council behavior skip <0-100>      Set skip probability (percent)
  /council behavior chime <0-100>     Set chime-in probability (percent)
  /council behavior min <1-10>        Set minimum responders
  /council behavior smart <on|off>    Toggle smart evaluation
  /council behavior delegate <on|off> Toggle delegation

[bold][SEARCH] Search:[/]
  /council search <query>             Search the web
  /council news <query>               Search news articles

[bold][BRAIN] Three-Tier Memory:[/]
  /council world                      View world memory (shared facts)
  /council world add <fact>           Add world fact
  /council system                     View system memory (how things work)
  /council system add <knowledge>     Add system knowledge
  /council persona <name>             View persona's memories
  /council persona <name> add <mem>   Add persona memory

[bold]* Soul Memory:[/]
  /council teach <content>            Teach Spirit a new memory
  /council recall <query>             Recall relevant memories

[bold][GEAR] Services:[/]
  /council status                     Show all service statuses
  /council agents                     List council agents
  /council safety [level]             Show/set safety level
  /council will [name]                Show/switch behavioral Will
  /council services                   Show service health

[dim]Council provides full parity with the web dashboard.[/]""")
            return

        import asyncio
        subcmd = args[0].lower()

        if subcmd == "chat":
            # Parse arguments
            depth = ResponseDepth.THOUGHTFUL
            agents = ["aither", "hydra", "terra"]
            prompt_start = 1

            for i, arg in enumerate(args[1:], 1):
                if arg == "-deep":
                    depth = ResponseDepth.DEEP
                    prompt_start = i + 1
                elif arg == "-concise":
                    depth = ResponseDepth.CONCISE
                    prompt_start = i + 1
                elif arg == "-agents" and i + 1 < len(args):
                    agents = args[i + 1].split(",")
                    prompt_start = i + 2
                elif not arg.startswith("-"):
                    break

            prompt = " ".join(args[prompt_start:])
            if not prompt:
                safe_print("[warning]Usage: /council chat [-deep] [-agents a,b,c] <prompt>[/]")
                return

            safe_print(f"[dim] Starting council discussion with {', '.join(agents)}...[/]")

            async def run_chat():
                result = await council.chat(
                    prompt=prompt,
                    participants=agents,
                    depth=depth
                )

                if not result.success:
                    safe_print(f"[red]Council chat failed: {result.error}[/]")
                    return

                for msg in result.messages:
                    if msg.type == "message":
                        safe_print(f"\n[bold #ffb3d9]{msg.agent_name}[/] [dim]({msg.persona})[/]:")
                        safe_print(f"  {msg.content}")
                    elif msg.type == "system":
                        safe_print(f"[dim][{msg.content}][/]")

                if result.memory_ids:
                    safe_print(f"\n[dim][SAVE] Saved to memory: {', '.join(result.memory_ids[:3])}[/]")

            asyncio.get_event_loop().run_until_complete(run_chat())

        elif subcmd == "status":
            async def get_status():
                status = await council.get_tool_status()
                if "error" in status:
                    safe_print(f"[red]Failed to get status: {status['error']}[/]")
                    return
                safe_print(format_tool_status(status))

            asyncio.get_event_loop().run_until_complete(get_status())

        elif subcmd == "agents":
            async def list_agents():
                agents = await council.get_agents()
                if not agents:
                    safe_print("[dim]No agents available or Council API not running.[/]")
                    return
                safe_print("[bold cyan] Council Agents:[/]")
                for agent in agents:
                    safe_print(f"  * [bold]{agent.get('name', agent.get('id'))}[/] - {agent.get('persona', '')}")

            asyncio.get_event_loop().run_until_complete(list_agents())

        elif subcmd == "safety":
            async def handle_safety():
                if len(args) > 1:
                    # Set safety level
                    level = args[1].lower()
                    result = await council.set_safety_level(level)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Safety level set to: {level}[/]")
                else:
                    # Get safety level
                    result = await council.get_safety_level()
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[bold]Current Safety Level:[/] {result.get('level', 'unknown')}")
                        safe_print(f"[dim]Available: {', '.join(result.get('levels_available', []))}[/]")

            asyncio.get_event_loop().run_until_complete(handle_safety())

        elif subcmd == "memory":
            async def handle_memory():
                if len(args) > 1 and args[1].lower() == "clear":
                    result = await council.clear_working_memory()
                    safe_print("[green]Working memory cleared.[/]" if "error" not in result else f"[red]{result['error']}[/]")
                else:
                    result = await council.get_working_memory()
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        items = result.get("items", [])
                        safe_print(f"[bold]Working Memory ({len(items)} items):[/]")
                        for item in items[:10]:
                            safe_print(f"  * {item}")

            asyncio.get_event_loop().run_until_complete(handle_memory())

        elif subcmd == "teach":
            content = " ".join(args[1:])
            if not content:
                safe_print("[warning]Usage: /council teach <content to remember>[/]")
                return

            async def teach():
                result = await council.teach_memory(content)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    safe_print(f"[green]Memory taught: {result.get('memory_id', 'success')}[/]")

            asyncio.get_event_loop().run_until_complete(teach())

        elif subcmd == "recall":
            query = " ".join(args[1:])
            if not query:
                safe_print("[warning]Usage: /council recall <search query>[/]")
                return

            async def recall():
                result = await council.recall_memories(query)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    memories = result.get("memories", [])
                    safe_print(f"[bold]Recalled {len(memories)} memories:[/]")
                    for m in memories:
                        safe_print(f"  [{m.get('type', '?')}] {m.get('content', '')[:100]}...")

            asyncio.get_event_loop().run_until_complete(recall())

        elif subcmd == "will":
            async def handle_will():
                if len(args) > 1:
                    # Switch will
                    will_name = args[1]
                    result = await council.switch_will(will_name)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Switched to Will: {will_name}[/]")
                else:
                    # Get active will
                    result = await council.get_active_will()
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[bold]Active Will:[/] {result.get('active_will', 'none')}")

            asyncio.get_event_loop().run_until_complete(handle_will())

        elif subcmd == "services":
            async def get_services():
                result = await council.get_services_health()
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    services = result.get("services", {})
                    safe_print("[bold][SEARCH] Service Health:[/]")
                    for name, status in services.items():
                        icon = "[DONE]" if status.get("healthy", False) else "[FAIL]"
                        safe_print(f"  {icon} {name}: {status.get('status', 'unknown')}")

            asyncio.get_event_loop().run_until_complete(get_services())

        elif subcmd == "search":
            query = " ".join(args[1:])
            if not query:
                safe_print("[warning]Usage: /council search <query>[/]")
                return

            async def do_search():
                safe_print(f"[dim][SEARCH] Searching: {query}...[/]")
                result = await council.search_web(query)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    results = result.get("results", [])
                    safe_print(f"[bold]Found {len(results)} results:[/]")
                    for r in results:
                        safe_print(f"\n  [bold cyan]{r.get('title', 'No title')}[/]")
                        safe_print(f"  [dim]{r.get('url', '')}[/]")
                        safe_print(f"  {r.get('snippet', '')[:200]}...")

            asyncio.get_event_loop().run_until_complete(do_search())

        elif subcmd == "news":
            query = " ".join(args[1:])
            if not query:
                safe_print("[warning]Usage: /council news <query>[/]")
                return

            async def do_news():
                safe_print(f"[dim] Searching news: {query}...[/]")
                result = await council.search_news(query)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    results = result.get("results", [])
                    safe_print(f"[bold]Found {len(results)} news articles:[/]")
                    for r in results:
                        safe_print(f"\n  [bold cyan]{r.get('title', 'No title')}[/]")
                        safe_print(f"  [dim]{r.get('source', '')} - {r.get('date', '')}[/]")
                        safe_print(f"  {r.get('snippet', '')[:200]}...")

            asyncio.get_event_loop().run_until_complete(do_news())

        elif subcmd == "world":
            async def handle_world():
                if len(args) > 1 and args[1].lower() == "add":
                    content = " ".join(args[2:])
                    if not content:
                        safe_print("[warning]Usage: /council world add <fact>[/]")
                        return
                    result = await council.add_world_memory(content)
                    safe_print("[green]World fact added![/]" if "error" not in result else f"[red]{result['error']}[/]")
                else:
                    result = await council.get_world_memory()
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[bold] World Memory ({result.get('count', 0)} facts):[/]")
                        safe_print(result.get("context", "No world facts yet."))

            asyncio.get_event_loop().run_until_complete(handle_world())

        elif subcmd == "system":
            async def handle_system():
                if len(args) > 1 and args[1].lower() == "add":
                    content = " ".join(args[2:])
                    if not content:
                        safe_print("[warning]Usage: /council system add <knowledge>[/]")
                        return
                    result = await council.add_system_memory(content)
                    safe_print("[green]System knowledge added![/]" if "error" not in result else f"[red]{result['error']}[/]")
                else:
                    result = await council.get_system_memory()
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[bold][GEAR] System Memory ({result.get('count', 0)} items):[/]")
                        safe_print(result.get("context", "No system knowledge yet."))

            asyncio.get_event_loop().run_until_complete(handle_system())

        elif subcmd == "persona":
            if len(args) < 2:
                safe_print("[warning]Usage: /council persona <name> [add <memory>][/]")
                return

            persona_name = args[1]

            async def handle_persona():
                if len(args) > 2 and args[2].lower() == "add":
                    content = " ".join(args[3:])
                    if not content:
                        safe_print("[warning]Usage: /council persona <name> add <memory>[/]")
                        return
                    result = await council.add_persona_memory(persona_name, content)
                    safe_print(f"[green]Memory added to {persona_name}![/]" if "error" not in result else f"[red]{result['error']}[/]")
                else:
                    result = await council.get_persona_memory(persona_name)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[bold] {persona_name}'s Memories:[/]")
                        safe_print(result.get("context", "No memories yet."))

            asyncio.get_event_loop().run_until_complete(handle_persona())

        elif subcmd == "settings":
            async def show_settings():
                result = await council.get_settings()
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                    return

                safe_print("[bold cyan] Council Settings[/]\n")

                safe_print("[bold]Response:[/]")
                safe_print(f"  Depth: {result.get('response_depth', 'thoughtful')}")
                safe_print(f"  Temperature: {result.get('temperature', 0.85)}")
                safe_print(f"  Max Tokens: {result.get('max_tokens', 600)}")

                safe_print("\n[bold]Toggles:[/]")
                toggles = result.get("toggles", {})
                for name, enabled in toggles.items():
                    icon = "[DONE]" if enabled else "[FAIL]"
                    safe_print(f"  {icon} {name}")

                safe_print("\n[bold]Style:[/]")
                style = result.get("style", {})
                for name, value in style.items():
                    safe_print(f"  {name}: {value}")

                if result.get("custom_system_prompt"):
                    safe_print(f"\n[bold]Custom Prompt:[/] {result['custom_system_prompt']}")
                    if result.get("full_override_mode"):
                        safe_print("[yellow]  [WARN] Full override mode enabled[/]")

            asyncio.get_event_loop().run_until_complete(show_settings())

        elif subcmd == "toggle":
            if len(args) < 2:
                safe_print("[warning]Usage: /council toggle <feature>[/]")
                safe_print("[dim]Features: web_search, memory, affect, collaborative, safety, reasoning, will, context[/]")
                return

            feature = args[1]

            async def do_toggle():
                result = await council.toggle_feature(feature)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    icon = "[DONE]" if result.get("enabled") else "[FAIL]"
                    safe_print(f"{icon} {feature}: {'enabled' if result.get('enabled') else 'disabled'}")

            asyncio.get_event_loop().run_until_complete(do_toggle())

        elif subcmd == "depth":
            if len(args) < 2:
                safe_print("[warning]Usage: /council depth <concise|thoughtful|deep>[/]")
                return

            depth = args[1].lower()

            async def set_depth():
                result = await council.set_depth(depth)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    safe_print(f"[green]Response depth set to: {depth}[/]")

            asyncio.get_event_loop().run_until_complete(set_depth())

        elif subcmd == "style":
            if len(args) < 3:
                safe_print("[warning]Usage: /council style <type> <value>[/]")
                safe_print("[dim]Types: formality (casual/balanced/formal)[/]")
                safe_print("[dim]       verbosity (terse/normal/verbose)[/]")
                safe_print("[dim]       creativity (conservative/balanced/creative)[/]")
                return

            style_type = args[1].lower()
            value = args[2].lower()

            async def set_style():
                result = await council.set_style(style_type, value)
                if "error" in result:
                    safe_print(f"[red]{result['error']}[/]")
                else:
                    safe_print(f"[green]{style_type} set to: {value}[/]")

            asyncio.get_event_loop().run_until_complete(set_style())

        elif subcmd == "prompt":
            if len(args) < 2:
                safe_print("[warning]Usage: /council prompt <text> OR /council prompt -override <text> OR /council prompt clear[/]")
                return

            if args[1].lower() == "clear":
                async def clear_prompt():
                    result = await council.clear_custom_prompt()
                    safe_print("[green]Custom prompt cleared.[/]" if "error" not in result else f"[red]{result['error']}[/]")
                asyncio.get_event_loop().run_until_complete(clear_prompt())
            else:
                full_override = args[1] == "-override"
                prompt_start = 2 if full_override else 1
                prompt = " ".join(args[prompt_start:])

                if not prompt:
                    safe_print("[warning]Please provide a prompt.[/]")
                    return

                async def set_prompt():
                    result = await council.set_custom_prompt(prompt, full_override=full_override)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        mode = "[yellow]FULL OVERRIDE[/]" if full_override else "prepended"
                        safe_print(f"[green]Custom prompt set ({mode})[/]")
                        if full_override:
                            safe_print("[dim][WARN] Most agent instructions replaced. Safety filtering still applies to output.[/]")

                asyncio.get_event_loop().run_until_complete(set_prompt())

        elif subcmd == "reset":
            async def reset():
                result = await council.reset_settings()
                safe_print("[green]All settings reset to defaults.[/]" if "error" not in result else f"[red]{result['error']}[/]")
            asyncio.get_event_loop().run_until_complete(reset())

        # =======================================================================
        # Timing & Behavior Commands
        # =======================================================================

        elif subcmd == "timing":
            if len(args) < 2:
                safe_print("[bold cyan]Response Timing Settings[/]")
                safe_print("/council timing preset <instant|fast|natural|thoughtful>")
                safe_print("/council timing eval <min> <max>  - Evaluation delay (seconds)")
                safe_print("/council timing stagger <min> <max>  - Response stagger (seconds)")
                safe_print("/council timing typing <on|off>  - Typing simulation")
                safe_print("/council timing speed <10-200>  - Typing speed (chars/sec)")
                return

            timing_cmd = args[1].lower()

            async def update_timing():
                if timing_cmd == "preset":
                    if len(args) < 3:
                        safe_print("[warning]Presets: instant, fast, natural, thoughtful[/]")
                        return
                    result = await council.set_timing_preset(args[2].lower())
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Timing preset applied: {args[2]}[/]")
                        settings = result.get("settings", {})
                        safe_print(f"[dim]  Eval: {settings.get('eval_delay_min', 0)}-{settings.get('eval_delay_max', 0)}s[/]")
                        safe_print(f"[dim]  Stagger: {settings.get('response_delay_min', 0)}-{settings.get('response_delay_max', 0)}s[/]")
                        safe_print(f"[dim]  Typing: {'on' if settings.get('typing_simulation') else 'off'}[/]")

                elif timing_cmd == "eval":
                    if len(args) < 4:
                        safe_print("[warning]Usage: /council timing eval <min_sec> <max_sec>[/]")
                        return
                    result = await council.update_timing(
                        eval_delay_min=float(args[2]),
                        eval_delay_max=float(args[3])
                    )
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Eval delay set to {args[2]}-{args[3]} seconds[/]")

                elif timing_cmd == "stagger":
                    if len(args) < 4:
                        safe_print("[warning]Usage: /council timing stagger <min_sec> <max_sec>[/]")
                        return
                    result = await council.update_timing(
                        response_delay_min=float(args[2]),
                        response_delay_max=float(args[3])
                    )
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Response stagger set to {args[2]}-{args[3]} seconds[/]")

                elif timing_cmd == "typing":
                    enabled = args[2].lower() in ("on", "true", "yes", "1") if len(args) > 2 else True
                    result = await council.update_timing(typing_simulation=enabled)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Typing simulation: {'on' if enabled else 'off'}[/]")

                elif timing_cmd == "speed":
                    if len(args) < 3:
                        safe_print("[warning]Usage: /council timing speed <10-200>[/]")
                        return
                    result = await council.update_timing(typing_speed=int(args[2]))
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Typing speed set to {args[2]} chars/sec[/]")

                else:
                    safe_print(f"[red]Unknown timing command: {timing_cmd}[/]")

            asyncio.get_event_loop().run_until_complete(update_timing())

        elif subcmd == "behavior":
            if len(args) < 2:
                safe_print("[bold cyan]Agent Behavior Settings[/]")
                safe_print("/council behavior preset <responsive|natural|selective|chaotic>")
                safe_print("/council behavior skip <0-100>  - Skip probability (percent)")
                safe_print("/council behavior chime <0-100>  - Chime-in probability (percent)")
                safe_print("/council behavior min <1-10>  - Minimum responders")
                safe_print("/council behavior smart <on|off>  - Smart evaluation")
                safe_print("/council behavior delegate <on|off>  - Allow delegation")
                return

            behavior_cmd = args[1].lower()

            async def update_behavior():
                if behavior_cmd == "preset":
                    if len(args) < 3:
                        safe_print("[warning]Presets: responsive, natural, selective, chaotic[/]")
                        return
                    result = await council.set_behavior_preset(args[2].lower())
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Behavior preset applied: {args[2]}[/]")
                        settings = result.get("settings", {})
                        safe_print(f"[dim]  Skip: {int(settings.get('base_skip_probability', 0)*100)}%[/]")
                        safe_print(f"[dim]  Chime-in: {int(settings.get('chime_in_probability', 0)*100)}%[/]")
                        safe_print(f"[dim]  Smart eval: {'on' if settings.get('smart_evaluation') else 'off'}[/]")

                elif behavior_cmd == "skip":
                    if len(args) < 3:
                        safe_print("[warning]Usage: /council behavior skip <0-100>[/]")
                        return
                    result = await council.update_behavior(
                        base_skip_probability=float(args[2]) / 100
                    )
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Skip probability set to {args[2]}%[/]")

                elif behavior_cmd == "chime":
                    if len(args) < 3:
                        safe_print("[warning]Usage: /council behavior chime <0-100>[/]")
                        return
                    result = await council.update_behavior(
                        chime_in_probability=float(args[2]) / 100
                    )
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Chime-in probability set to {args[2]}%[/]")

                elif behavior_cmd == "min":
                    if len(args) < 3:
                        safe_print("[warning]Usage: /council behavior min <1-10>[/]")
                        return
                    result = await council.update_behavior(min_responders=int(args[2]))
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Minimum responders set to {args[2]}[/]")

                elif behavior_cmd == "smart":
                    enabled = args[2].lower() in ("on", "true", "yes", "1") if len(args) > 2 else True
                    result = await council.update_behavior(smart_evaluation=enabled)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Smart evaluation: {'on' if enabled else 'off'}[/]")

                elif behavior_cmd == "delegate":
                    enabled = args[2].lower() in ("on", "true", "yes", "1") if len(args) > 2 else True
                    result = await council.update_behavior(allow_delegation=enabled)
                    if "error" in result:
                        safe_print(f"[red]{result['error']}[/]")
                    else:
                        safe_print(f"[green]Allow delegation: {'on' if enabled else 'off'}[/]")

                else:
                    safe_print(f"[red]Unknown behavior command: {behavior_cmd}[/]")

            asyncio.get_event_loop().run_until_complete(update_behavior())

        else:
            safe_print(f"[red]Unknown council command: {subcmd}[/]")
            safe_print("[dim]Use /council for help.[/]")

    elif cmd == "/a2a":
        # Google A2A Protocol - Agent-to-Agent Communication
        try:
            from aither_adk.communication.a2a_client import (
                A2AClient,
                discover_agents,
                send_to_agent,
            )
        except ImportError:
            safe_print("[red]A2A client not available. Install httpx: pip install httpx[/]")
            return

        if not args:
            safe_print("""[bold cyan][LINK] Google A2A Protocol - Agent-to-Agent Communication[/]

[bold]Discovery:[/]
  /a2a agents                         List all A2A-compatible agents
  /a2a card <agent>                   Get agent's capability card
  /a2a discover <url>                 Discover agents at external URL

[bold]Tasks:[/]
  /a2a ask <agent> <prompt>           Send a task to an agent
  /a2a council <prompt>               Ask the full Aither Council
  /a2a aither <prompt>                Ask Aither directly
  /a2a terra <prompt>                 Ask Terra (Infrastructure)
  /a2a hydra <prompt>                 Ask Hydra (Code)
  /a2a ignis <prompt>                 Ask Ignis (Security)
  /a2a aeros <prompt>                 Ask Aeros (Network)
  /a2a gluttony <prompt>              Ask Gluttony (Data)

[bold]External A2A:[/]
  /a2a connect <url>                  Connect to external A2A endpoint
  /a2a external <url> <agent> <prompt> Call external A2A agent

[bold]Info:[/]
  /a2a status                         Gateway health
  /a2a spec                           Protocol specification

[dim]A2A is Google's open standard for agent interoperability.[/]
[dim]See: https://github.com/google-a2a/A2A[/]""")
            return

        import asyncio
        subcmd = args[0].lower()

        if subcmd == "agents":
            async def list_a2a_agents():
                try:
                    agents = await discover_agents()
                    safe_print("[bold cyan][LINK] A2A-Compatible Agents:[/]")
                    for agent in agents:
                        safe_print(f"  * [bold]{agent.get('name', agent.get('id'))}[/]")
                        safe_print(f"    {agent.get('description', '')[:60]}...")
                        safe_print(f"    Skills: {', '.join(agent.get('skills', []))}")
                except Exception as e:
                    safe_print(f"[red]Failed to discover agents: {e}[/]")

            asyncio.get_event_loop().run_until_complete(list_a2a_agents())

        elif subcmd == "card":
            if len(args) < 2:
                safe_print("[warning]Usage: /a2a card <agent_id>[/]")
                return

            agent_id = args[1]

            async def get_card():
                try:
                    async with A2AClient() as client:
                        card = await client.get_agent_card(agent_id)
                        safe_print(f"[bold cyan] Agent Card: {card.name}[/]")
                        safe_print(f"[dim]{card.description}[/]")
                        safe_print(f"URL: {card.url}")
                        safe_print(f"Version: {card.version}")
                        safe_print("\n[bold]Skills:[/]")
                        for skill in card.skills:
                            safe_print(f"  * [bold]{skill.name}[/] ({skill.id})")
                            safe_print(f"    {skill.description}")
                            if skill.examples:
                                safe_print(f"    [dim]Examples: {', '.join(skill.examples[:2])}[/]")
                except Exception as e:
                    safe_print(f"[red]Failed to get card: {e}[/]")

            asyncio.get_event_loop().run_until_complete(get_card())

        elif subcmd == "status":
            async def check_status():
                try:
                    async with A2AClient() as client:
                        card = await client.get_master_card()
                        safe_print("[green][OK] A2A Gateway Online[/]")
                        safe_print(f"  System: {card.name}")
                        safe_print(f"  Description: {card.description[:60]}...")
                        safe_print(f"  URL: {card.url}")
                except Exception as e:
                    safe_print("[red][X] A2A Gateway Offline[/]")
                    safe_print(f"[dim]Error: {e}[/]")
                    safe_print("[dim]Start with: python AitherA2A.py[/]")

            asyncio.get_event_loop().run_until_complete(check_status())

        elif subcmd == "ask":
            if len(args) < 3:
                safe_print("[warning]Usage: /a2a ask <agent> <prompt>[/]")
                return

            agent_id = args[1]
            prompt = " ".join(args[2:])

            safe_print(f"[dim][LINK] Sending task to {agent_id}...[/]")

            async def do_ask():
                try:
                    response = await send_to_agent(agent_id, prompt)
                    safe_print(f"\n[bold cyan]{agent_id.capitalize()}:[/]")
                    safe_print(response)
                except Exception as e:
                    safe_print(f"[red]Task failed: {e}[/]")

            asyncio.get_event_loop().run_until_complete(do_ask())

        # Quick agent shortcuts
        elif subcmd in ["aither", "terra", "hydra", "ignis", "aeros", "gluttony", "council"]:
            prompt = " ".join(args[1:])
            if not prompt:
                safe_print(f"[warning]Usage: /a2a {subcmd} <prompt>[/]")
                return

            safe_print(f"[dim][LINK] Sending to {subcmd}...[/]")

            async def quick_ask():
                try:
                    response = await send_to_agent(subcmd, prompt)
                    safe_print(f"\n[bold cyan]{subcmd.capitalize()}:[/]")
                    safe_print(response)
                except Exception as e:
                    safe_print(f"[red]Failed: {e}[/]")

            asyncio.get_event_loop().run_until_complete(quick_ask())

        elif subcmd == "spec":
            safe_print("""[bold cyan][LINK] Google A2A Protocol v0.3.0 Specification[/]

[bold]Core Concepts:[/]
  * Agent Cards - JSON discovery documents at /.well-known/agent-card.json
  * JSON-RPC 2.0 - Standardized messaging over HTTP(S)
  * SSE Streaming - Real-time message updates
  * contextId - Persistent context across messages
  * Task Lifecycle - submitted -> working -> completed/failed/canceled

[bold]Endpoints:[/]
  /.well-known/agent-card.json  Agent discovery card (v0.3.0)
  /.well-known/agent.json       Agent discovery card (deprecated)
  /rpc                          JSON-RPC 2.0 endpoint

[bold]Methods:[/]
  message/send                Send a message (v0.3.0)
  message/stream              Send and stream response (v0.3.0)
  tasks/send                  Legacy: Submit a task
  tasks/get                   Get task status
  tasks/cancel                Cancel a running task

[bold]Task States:[/]
  submitted     -> Task created, waiting to start
  working       -> Agent is processing
  input-required -> Agent needs more information
  completed     -> Task finished successfully
  canceled      -> Task was canceled
  failed        -> Task failed with error

[link]https://github.com/google-a2a/A2A[/link]""")

        elif subcmd == "connect":
            if len(args) < 2:
                safe_print("[warning]Usage: /a2a connect <url>[/]")
                return

            url = args[1]

            async def test_connection():
                try:
                    async with A2AClient(base_url=url) as client:
                        card = await client.get_master_card()
                        safe_print(f"[green][OK] Connected to {card.name}[/]")
                        safe_print(f"  {card.description}")

                        agents = await client.list_agents()
                        safe_print(f"\n[bold]Available Agents ({len(agents)}):[/]")
                        for agent in agents:
                            safe_print(f"  * {agent.get('name', agent.get('id'))}")
                except Exception as e:
                    safe_print(f"[red][X] Connection failed: {e}[/]")

            asyncio.get_event_loop().run_until_complete(test_connection())

        elif subcmd == "external":
            if len(args) < 4:
                safe_print("[warning]Usage: /a2a external <url> <agent> <prompt>[/]")
                return

            url = args[1]
            agent_id = args[2]
            prompt = " ".join(args[3:])

            safe_print(f"[dim][LINK] Sending to {agent_id} at {url}...[/]")

            async def call_external():
                try:
                    async with A2AClient(base_url=url) as client:
                        response = await client.quick_task(agent_id, prompt)
                        safe_print(f"\n[bold cyan]{agent_id}:[/]")
                        safe_print(response)
                except Exception as e:
                    safe_print(f"[red]External call failed: {e}[/]")

            asyncio.get_event_loop().run_until_complete(call_external())

        else:
            safe_print(f"[red]Unknown A2A command: {subcmd}[/]")
            safe_print("[dim]Use /a2a for help.[/]")

    elif cmd == "/remember":
        if not memory_manager:
            safe_print("[warning]Memory system is not enabled for this agent.[/]")
            return

        if not args:
            safe_print("[bold cyan]Usage:[/bold cyan] /remember <text to remember>")
            return

        text = " ".join(args)
        try:
            memory_manager.add_memory(text, source="user_command")
            safe_print(f"[green][BRAIN] Memory stored: '{text}'[/]")
        except Exception as e:
            safe_print(f"[red]Failed to store memory: {e}[/]")

    elif cmd == "/keys":
        # Show keyboard shortcuts reference
        try:
            from rich.table import Table

            from aither_adk.ui.ui import KEYBOARD_SHORTCUTS

            safe_print("\n[bold cyan][KB]  Keyboard Shortcuts[/]\n")

            for category, shortcuts in KEYBOARD_SHORTCUTS.items():
                table = Table(title=f"[bold]{category}[/]", show_header=True, header_style="bold cyan")
                table.add_column("Key", style="yellow", width=15)
                table.add_column("Action", style="white")

                for key, action in shortcuts.items():
                    table.add_row(key, action)

                console.print(table)
                safe_print("")

            safe_print("[dim]Tip: Use Esc+Enter to submit, Enter for newline[/]")
        except Exception as e:
            safe_print(f"[warning]Error loading shortcuts: {e}[/]")
        return

    elif cmd == "/help":
        safe_print(Panel("""
[bold cyan]Available Commands:[/bold cyan]

[bold yellow]Help & Navigation:[/bold yellow]
  [bold #ffb3d9]/help[/bold #ffb3d9]           - Show this help menu
  [bold #ffb3d9]/keys[/bold #ffb3d9]           - Show all keyboard shortcuts
  [bold #ffb3d9]/clear[/bold #ffb3d9]          - Clear terminal
  [bold #ffb3d9]/exit[/bold #ffb3d9]           - Exit the agent
  [bold #ffb3d9]/quit[/bold #ffb3d9]           - Exit the agent (alias)

[bold yellow]Model & Resource Management:[/bold yellow]
  [bold #ffb3d9]/model [name][/bold #ffb3d9]   - Switch LLM models
  [bold #ffb3d9]/models[/bold #ffb3d9]         - Show currently loaded models
  [bold #ffb3d9]/vram[/bold #ffb3d9]           - Show GPU VRAM usage and status
  [bold #ffb3d9]/resources[/bold #ffb3d9]     - Show detailed resource status
  [bold #ffb3d9]/load [model][/bold #ffb3d9]   - Preload a model into VRAM
  [bold #ffb3d9]/unload [model|all][/bold #ffb3d9] - Unload model(s) to free VRAM

[bold yellow]Agent & Persona:[/bold yellow]
  [bold #ffb3d9]/role [name][/bold #ffb3d9]    - Switch agent persona
  [bold #ffb3d9]/persona-info [name][/bold #ffb3d9] - Show persona visual identity
  [bold #ffb3d9]/create-persona[/bold #ffb3d9] - Create a new persona
  [bold #ffb3d9]/init-personas[-g][/bold #ffb3d9] - Initialize all personas (use -g to auto-generate anchors)
  [bold #ffb3d9]/group [cmd][/bold #ffb3d9]    - Manage group chats (start, stop, switch, list)
  [bold #ffb3d9]/free-chat[/bold #ffb3d9]      - Toggle free chat mode

[bold yellow]Safety & Mode:[/bold yellow]
  [bold #ffb3d9]/mode[/bold #ffb3d9]            - Show current mode and options
  [bold #ffb3d9]/mode <level>[/bold #ffb3d9]    - Set mode (professional|casual|unrestricted)
  [bold #ffb3d9]/mode cycle[/bold #ffb3d9]     - Cycle through modes (F2 shortcut)
  [bold #ffb3d9]/safety[/bold #ffb3d9]         - Show safety settings (alias for /mode)
  [bold #ffb3d9]/detail[/bold #ffb3d9]         - Control roleplay detail level (minimal|moderate|detailed|extensive)
  [bold #ffb3d9]/tone <tone>[/bold #ffb3d9]    - Set tone (professional|casual|empathetic|instructional)
  [bold #ffb3d9]/style <style>[/bold #ffb3d9]  - Set style (concise|detailed|creative|technical)
  [bold #ffb3d9]/thinking[/bold #ffb3d9]       - Toggle <think> block visibility (show model's internal reasoning)
  [bold #ffb3d9]/reasoning[/bold #ffb3d9]       - Toggle reasoning trace visibility (AitherReasoning service)
  [bold #ffb3d9]/debug[/bold #ffb3d9]           - Toggle debug mode (verbose event logging)
  [dim]Override prefix for one turn: :: or ~ or >>>[/dim]
  [dim]Example: ::generate something spicy[/dim]

[bold yellow]Image Generation:[/bold yellow]
  [bold #ffb3d9]/gen <persona> <prompt>[/bold #ffb3d9] - Preview generated prompt
  [bold #ffb3d9]/scene [location] [lighting][/bold #ffb3d9] - Set/show scene context
  [bold #ffb3d9]/anchor <persona> <type> <path>[/bold #ffb3d9] - Set reference image (face|body|style)

[bold yellow]Memory System:[/bold yellow]
  [bold #ffb3d9]/memory[/bold #ffb3d9]         - Show memory summary
  [bold #ffb3d9]/memory show [persona][/bold #ffb3d9] - Show full context
  [bold #ffb3d9]/memory add <type> <text>[/bold #ffb3d9] - Add memory (world|system|persona:name)
  [bold #ffb3d9]/memory clear <type>[/bold #ffb3d9] - Clear memory
  [bold #ffb3d9]/remember [text][/bold #ffb3d9] - Add memory to long-term storage

[bold yellow]Communication:[/bold yellow]
  [bold #ffb3d9]/inbox[/bold #ffb3d9]          - Check message inbox
  [bold #ffb3d9]/mail @agent1 @agent2 ...[/bold #ffb3d9] - Send message to multiple agents
  [bold #ffb3d9]/search <query>[/bold #ffb3d9] - Web search and analysis

[bold yellow]Quality of Life Tools:[/bold yellow]
  [bold #ffb3d9]/notebook[/bold #ffb3d9]       - Notebook: store and search notes
  [bold #ffb3d9]/calendar[/bold #ffb3d9]       - Calendar: manage events and reminders
  [bold #ffb3d9]/timer[/bold #ffb3d9]          - Timer: countdown timers
  [bold #ffb3d9]/stopwatch[/bold #ffb3d9]      - Stopwatch: track elapsed time
  [bold #ffb3d9]/alarm[/bold #ffb3d9]          - Alarm: set alarms

[bold yellow]System & Info:[/bold yellow]
  [bold #ffb3d9]/settings[/bold #ffb3d9]       - Show config. Use '/settings model <name>' to set default
  [bold #ffb3d9]/cost[/bold #ffb3d9]           - Show session cost and usage
  [bold #ffb3d9]/history[/bold #ffb3d9]        - Show recent conversation history
  [bold #ffb3d9]/profile[/bold #ffb3d9]        - Show agent profile
  [bold #ffb3d9]/attach [path][/bold #ffb3d9]  - Attach file. Use 'last' for last image.
  [bold #ffb3d9]/vision [path][/bold #ffb3d9]  - Analyze image with AitherVision (OCR, describe, compare)
  [bold #ffb3d9]/turns[/bold #ffb3d9]          - Show scene turn count
  [bold #ffb3d9]/continue[/bold #ffb3d9]       - Continue scene

[bold cyan][KB] Quick Keys (VSCode-safe):[/bold cyan]
  F6:help F7:inbox F8:mode F9:vram F12:exit
  Alt+I:inbox Alt+M:mode Alt+U::: Alt+K:shortcuts Alt+H:help

[dim]Note: Alt = Esc then key (e.g., Esc then I for inbox)[/dim]
""", title="[bold cyan]Help[/]", border_style="cyan"))

    elif cmd == "/debug":
        new_mode = debug_mode_callback()
        safe_print(f"[info]Debug mode: [bold]{'ON' if new_mode else 'OFF'}[/][/]")

    elif cmd == "/route":
        # Agent routing configuration
        try:
            from aither_adk.infrastructure.routing_manager import get_routing_manager

            manager = get_routing_manager()

            if not args:
                # Show current routing status
                status = manager.get_status()
                safe_print("\n[bold cyan] Agent Routing Configuration[/]")
                safe_print("[dim]-------------------------------------[/]")
                safe_print(f"Routing: [bold]{'[DONE] Enabled' if status['enabled'] else '[FAIL] Disabled'}[/]")
                safe_print(f"Default Agent: [bold]{status['default_agent']}[/]")
                safe_print(f"Confidence Threshold: [bold]{status['confidence_threshold']:.0%}[/]")
                safe_print(f"Gemini Fallback: [bold]{'Yes' if status['gemini_fallback'] else 'No'}[/]")
                safe_print("\n[bold]Agents:[/]")
                for name, info in status['agents'].items():
                    emoji = "[DONE]" if info['enabled'] else "[FAIL]"
                    type_badge = f"[dim]({info['type']})[/]"
                    safe_print(f"  {emoji} [cyan]{name}[/] {type_badge} - P{info['priority']}, {info['patterns']} patterns, {info['keywords']} keywords")
                if status['aliases']:
                    safe_print("\n[bold]Aliases:[/]")
                    for alias, target in status['aliases'].items():
                        safe_print(f"  [dim]{alias}[/] -> [cyan]{target}[/]")
                safe_print("\n[dim]Commands: /route enable|disable <agent>, /route add|remove <agent>, /route on|off[/]")
                return

            action = args[0].lower()

            if action in ["on", "enable-all"]:
                manager.enabled = True
                safe_print("[green][DONE] Routing enabled[/]")
            elif action in ["off", "disable-all"]:
                manager.enabled = False
                safe_print("[yellow][WARN] Routing disabled - all requests go to default agent[/]")
            elif action == "reload":
                manager.reload()
                safe_print("[green][DONE] Routing config reloaded from file[/]")
            elif action == "enable" and len(args) >= 2:
                agent_name = args[1]
                if manager.enable_agent(agent_name):
                    safe_print(f"[green][DONE] {agent_name} enabled[/]")
                else:
                    safe_print(f"[red]Agent not found: {agent_name}[/]")
            elif action == "disable" and len(args) >= 2:
                agent_name = args[1]
                if manager.disable_agent(agent_name):
                    safe_print(f"[yellow][WARN] {agent_name} disabled[/]")
                else:
                    safe_print(f"[red]Agent not found: {agent_name}[/]")
            elif action == "default" and len(args) >= 2:
                agent_name = args[1]
                if manager.set_default_agent(agent_name):
                    safe_print(f"[green][DONE] Default agent set to {agent_name}[/]")
                else:
                    safe_print(f"[red]Agent not found: {agent_name}[/]")
            elif action == "add" and len(args) >= 2:
                agent_name = args[1]
                description = " ".join(args[2:]) if len(args) > 2 else ""
                manager.add_agent(agent_name, description=description)
                safe_print(f"[green][DONE] Custom agent '{agent_name}' added[/]")
                safe_print("[dim]Edit config/routing.yaml to add patterns and keywords[/]")
            elif action == "remove" and len(args) >= 2:
                agent_name = args[1]
                if manager.remove_agent(agent_name):
                    safe_print(f"[green][DONE] Agent '{agent_name}' removed[/]")
                else:
                    safe_print(f"[red]Cannot remove builtin agent or agent not found: {agent_name}[/]")
            elif action == "alias" and len(args) >= 3:
                alias = args[1]
                target = args[2]
                if manager.add_alias(alias, target):
                    safe_print(f"[green][DONE] Alias '{alias}' -> '{target}' added[/]")
                else:
                    safe_print(f"[red]Target agent not found: {target}[/]")
            else:
                safe_print("[bold]Usage:[/]")
                safe_print("  /route                 - Show routing status")
                safe_print("  /route on|off          - Enable/disable routing globally")
                safe_print("  /route enable <agent>  - Enable specific agent")
                safe_print("  /route disable <agent> - Disable specific agent")
                safe_print("  /route default <agent> - Set default agent")
                safe_print("  /route add <name> [desc] - Add custom agent")
                safe_print("  /route remove <name>   - Remove custom agent")
                safe_print("  /route alias <alias> <agent> - Add alias")
                safe_print("  /route reload          - Reload from config file")

        except ImportError as e:
            safe_print(f"[warning]Routing manager not available: {e}[/]")
        except Exception as e:
            safe_print(f"[red]Error: {e}[/]")
        return

    elif cmd == "/reasoning":
        # Toggle reasoning trace visibility
        try:
            from aither_adk.infrastructure.runner import get_show_reasoning, set_show_reasoning
            current = get_show_reasoning()
            set_show_reasoning(not current)
            new_state = not current
            safe_print(f"[info]Reasoning traces: [bold]{'ON - thoughts will be shown' if new_state else 'OFF'}[/][/]")
            if new_state:
                safe_print("[dim]You'll now see the agent's thinking process during responses.[/]")
                safe_print("[dim]Note: Requires AitherReasoning service (port 8093) to be running.[/]")
        except ImportError:
            safe_print("[warning]Reasoning module not available.[/]")
        return

    elif cmd == "/thinking":
        # Toggle display of <think>...</think> blocks from model output
        # With no args: toggle visibility
        # With 'on'/'off': explicit set
        # With 'show': alias for on, 'hide': alias for off
        try:
            from aither_adk.infrastructure.utils import get_show_thinking, set_show_thinking

            if not args:
                # Toggle
                current = get_show_thinking()
                set_show_thinking(not current)
                new_state = not current
            else:
                arg = args[0].lower()
                if arg in ["on", "show", "true", "enable"]:
                    set_show_thinking(True)
                    new_state = True
                elif arg in ["off", "hide", "false", "disable"]:
                    set_show_thinking(False)
                    new_state = False
                else:
                    safe_print("[warning]Usage: /thinking [on|off|show|hide][/]")
                    return

            if new_state:
                safe_print("[info]Thinking display: [bold]ON[/] - <think> blocks will be shown[/]")
                safe_print("[dim]Model reasoning will appear in a separate panel above responses.[/]")
            else:
                safe_print("[info]Thinking display: [bold]OFF[/] - <think> blocks will be hidden[/]")
        except ImportError:
            safe_print("[warning]Thinking module not available.[/]")
        return

    elif cmd == "/reload":
        import importlib
        try:
            import prompts
            importlib.reload(prompts)
            agent.instruction = prompts.SYSTEM_INSTRUCTION
            safe_print("[green]System instruction reloaded from prompts.py[/]")
            preview = agent.instruction[:50].replace("\n", " ") + "..."
            safe_print(f"[dim]Preview: {preview}[/]")
        except Exception as e:
            safe_print(f"[red]Failed to reload prompts: {e}[/]")

    elif cmd == "/cost":
        safe_print(f"[info]Session Total: {session_stats['total_input']} in / {session_stats['total_output']} out | Cost: ${session_stats['total_cost']:.6f}[/]")

    elif cmd == "/clear":
        os.system('cls' if os.name == 'nt' else 'clear')
        safe_print("[dim]Console cleared.[/]")
        return

    elif cmd == "/history":
        # Show recent history
        try:
            # Note: app_name might be 'agents' based on agent.py
            session_obj = await runner.session_service.get_session(session_id=session_id, app_name="agents", user_id="user")
            if session_obj and hasattr(session_obj, 'events') and session_obj.events:
                safe_print(Panel("[bold cyan]Session History (Last 10)[/]", border_style="cyan"))
                # Show last 10 events
                for event in session_obj.events[-10:]:
                    # Extract role and content from event
                    role = "unknown"
                    content = ""

                    # Check different event structures
                    if hasattr(event, 'role'):
                        role = event.role
                    elif hasattr(event, 'user_content'):
                        role = "user"
                        content = str(event.user_content)
                    elif hasattr(event, 'parts'):
                        # Try to determine role from parts
                        for part in event.parts:
                            if hasattr(part, 'text') and part.text:
                                content += part.text + " "
                            if hasattr(part, 'role'):
                                role = part.role

                    # If we still don't have content, try to stringify the event
                    if not content:
                        content = str(event)[:200]

                    color = "bright_cyan" if role == "user" else "#ffb3d9"
                    safe_print(f"[{color}]{role.upper()}:[/] {content[:200]}...")
            else:
                safe_print("[dim]No history found.[/]")
        except Exception as e:
            safe_print(f"[red]Error fetching history: {e}[/]")
            import traceback
            safe_print(f"[dim]{traceback.format_exc()}[/]")
        return

    elif cmd == "/safety" or cmd == "/mode":
        # Safety mode control
        try:
            from aither_adk.ai.safety_mode import (
                OVERRIDE_PREFIXES,
                SafetyLevel,
                get_level_emoji,
                get_level_name,
                get_safety_manager,
            )

            manager = get_safety_manager()

            if not args:
                # Show current mode
                level = manager.current_level
                config = manager.get_config()

                safe_print(f"\n[bold]Current Mode: {get_level_emoji(level)} {get_level_name(level)}[/]")
                safe_print(f"[dim]LLM: {'Gemini (Cloud)' if config.use_cloud_llm else 'Local'}[/]")
                safe_print(f"[dim]Content: {'Creative mode' if config.allow_explicit else 'Filtered'}[/]")
                safe_print("\n[bold]Available Modes:[/]")
                safe_print("   [cyan]professional[/] - Business mode, Gemini (cloud)")
                safe_print("   [yellow]casual[/] - Friendly, Gemini (cloud)")
                safe_print("   [#ffb3d9]unrestricted[/] - Creative mode, local LLM")
                safe_print("\n[bold]Creative Mode Prefixes:[/]")
                safe_print(f"  [cyan]{' | '.join(OVERRIDE_PREFIXES[:4])}[/]")
                safe_print("  [dim]Example: ::creative content request[/]")
                return

            # Set mode
            mode_arg = args[0].lower()

            # Track if level actually changed
            old_level = manager.current_level
            new_level = old_level

            if mode_arg == "cycle":
                # Cycle through modes: PROFESSIONAL -> CASUAL -> UNRESTRICTED -> PROFESSIONAL
                if old_level == SafetyLevel.PROFESSIONAL:
                    new_level = SafetyLevel.CASUAL
                    manager.set_level(new_level)
                    safe_print("[green][DONE] Mode:  Casual[/] (Gemini, light content)")
                elif old_level == SafetyLevel.CASUAL:
                    new_level = SafetyLevel.UNRESTRICTED
                    manager.set_level(new_level)
                    safe_print("[green][DONE] Mode: Unrestricted[/] (Local LLM, creative mode)")
                else:
                    new_level = SafetyLevel.PROFESSIONAL
                    manager.set_level(new_level)
                    safe_print("[green][DONE] Mode:  Professional[/] (Gemini, filtered)")
            elif mode_arg in ["professional", "pro", "work", "business"]:
                new_level = SafetyLevel.PROFESSIONAL
                manager.set_level(new_level)
                safe_print("[green][DONE] Mode set to:  Professional[/]")
                safe_print("[dim]Using Gemini (cloud), content filtered[/]")
            elif mode_arg in ["casual", "chill", "relaxed"]:
                new_level = SafetyLevel.CASUAL
                manager.set_level(new_level)
                safe_print("[green][DONE] Mode set to:  Casual[/]")
                safe_print("[dim]Using Gemini (cloud), light content allowed[/]")
            elif mode_arg in ["unrestricted", "creative", "raw", "local", "off", "none"]:
                new_level = SafetyLevel.UNRESTRICTED
                manager.set_level(new_level)
                safe_print("[green][DONE] Mode set to: Unrestricted[/]")
                safe_print("[dim]Using local LLM, creative mode enabled[/]")
            else:
                safe_print(f"[warning]Unknown mode: {mode_arg}[/]")
                safe_print("[dim]Use: professional, casual, unrestricted, or cycle[/]")
                return

            # Signal that agent needs to be recreated with new safety instruction
            if new_level != old_level:
                safe_print("[dim]Reinitializing agent with new safety settings...[/]")
                return "RECREATE_AGENT"

        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/detail":
        """Control roleplay detail level for Aither responses."""
        try:
            # Try to get current level from environment or default
            # Note: os is already imported at top of file
            current_level = os.getenv("AITHER_ROLEPLAY_DETAIL", "minimal").lower()

            if not args:
                # Show current level
                level_emoji = {
                    "minimal": "[NOTE]",
                    "moderate": "[FILE]",
                    "detailed": "",
                    "extensive": ""
                }
                emoji = level_emoji.get(current_level, "[NOTE]")
                safe_print(f"[bold #ffb3d9]Current Roleplay Detail Level:[/bold #ffb3d9] {emoji} {current_level.title()}")
                safe_print("[dim]Use '/detail <level>' to change: minimal, moderate, detailed, extensive[/]")
                safe_print("[dim]minimal: Brief (2-4 sentences) | moderate: Standard (3-6) | detailed: Immersive (6-10) | extensive: Vivid (10+)[/]")
            else:
                # Set detail level
                new_level = args[0].lower()
                valid_levels = ["minimal", "moderate", "detailed", "extensive"]

                if new_level not in valid_levels:
                    safe_print(f"[red]Invalid detail level. Use: {', '.join(valid_levels)}[/]")
                    return

                # Update environment variable (will be picked up on next import)
                os.environ["AITHER_ROLEPLAY_DETAIL"] = new_level

                # Try to update the module if it's already loaded
                agent_module = sys.modules.get('AitherOS.agents.Saga.agent')
                if agent_module:
                    agent_module.ROLEPLAY_DETAIL_LEVEL = new_level

                safe_print(f"[green][DONE] Roleplay detail level set to: {new_level.title()}[/]")
                safe_print("[dim]Level will apply to next Aither response[/]")
        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/tone":
        """Set the tone for Aither's responses."""
        valid_tones = ["professional", "casual", "empathetic", "instructional"]
        if not args:
            current = os.getenv("AITHER_TONE", "default").lower()
            safe_print(f"[bold #ffb3d9]Current Tone:[/bold #ffb3d9] {current}")
            safe_print(f"[dim]Available: {', '.join(valid_tones)}[/]")
        else:
            new_tone = args[0].lower()
            if new_tone in valid_tones or new_tone == "default":
                if new_tone == "default": new_tone = ""
                os.environ["AITHER_TONE"] = new_tone
                safe_print(f"[green][DONE] Tone set to: {new_tone if new_tone else 'Default'}[/]")
            else:
                safe_print(f"[red]Invalid tone. Use: {', '.join(valid_tones)}[/]")
        return

    elif cmd == "/style":
        """Set the style for Aither's responses."""
        valid_styles = ["concise", "detailed", "creative", "technical"]
        if not args:
            current = os.getenv("AITHER_STYLE", "default").lower()
            safe_print(f"[bold #ffb3d9]Current Style:[/bold #ffb3d9] {current}")
            safe_print(f"[dim]Available: {', '.join(valid_styles)}[/]")
        else:
            new_style = args[0].lower()
            if new_style in valid_styles or new_style == "default":
                if new_style == "default": new_style = ""
                os.environ["AITHER_STYLE"] = new_style
                safe_print(f"[green][DONE] Style set to: {new_style if new_style else 'Default'}[/]")
            else:
                safe_print(f"[red]Invalid style. Use: {', '.join(valid_styles)}[/]")
        return

    elif cmd == "/think":
        # Alias for /thinking - redirect
        safe_print("[dim]Hint: /think is now /thinking[/]")
        # Fall through to show current status
        try:
            from aither_adk.infrastructure.utils import get_show_thinking
            current = get_show_thinking()
            safe_print(f"[info]Thinking display: [bold]{'ON' if current else 'OFF'}[/][/]")
            safe_print("[dim]Use /thinking [on|off] to change.[/]")
        except ImportError:
            pass
        return

    elif cmd == "/notebook":
        """Notebook - store and search notes."""
        try:
            from aither_adk.tools.qol_tools import get_notebook
            notebook = get_notebook()

            if not args:
                # Show recent notes
                notes = notebook.get_notes(limit=10)
                if not notes:
                    safe_print("[dim]No notes yet. Use '/notebook add <text>' to create one.[/]")
                else:
                    safe_print(f"[bold #ffb3d9] Notebook ({len(notebook.notes)} total, showing 10 most recent):[/bold #ffb3d9]")
                    for i, note in enumerate(notes, 1):
                        tags_str = f" [{', '.join(note.get('tags', []))}]" if note.get('tags') else ""
                        safe_print(f"{i}. [{note.get('created', '')[:16]}] {note.get('content', '')[:60]}...{tags_str}")
                    safe_print("\n[dim]Use '/notebook add <text>', '/notebook search <query>', '/notebook tag <tag>'[/]")
            elif args[0].lower() == "add":
                if len(args) < 2:
                    safe_print("[warning]Usage: /notebook add <text> [--tags tag1,tag2][/]")
                    return

                # Parse tags if provided
                content_parts = []
                tags = []
                i = 1
                while i < len(args):
                    if args[i] == "--tags" and i + 1 < len(args):
                        tags = [t.strip() for t in args[i+1].split(",")]
                        i += 2
                    else:
                        content_parts.append(args[i])
                        i += 1

                content = " ".join(content_parts)
                note = notebook.add_note(content, tags)
                safe_print("[green][DONE] Note added[/]")
                safe_print(f"[dim]ID: {note['id'][:8]}... | Tags: {', '.join(tags) if tags else 'none'}[/]")
            elif args[0].lower() == "search":
                if len(args) < 2:
                    safe_print("[warning]Usage: /notebook search <query>[/]")
                    return

                query = " ".join(args[1:])
                results = notebook.search_notes(query)
                if not results:
                    safe_print(f"[dim]No notes found matching '{query}'[/]")
                else:
                    safe_print(f"[bold #ffb3d9] Search results for '{query}' ({len(results)}):[/bold #ffb3d9]")
                    for i, note in enumerate(results[:10], 1):
                        tags_str = f" [{', '.join(note.get('tags', []))}]" if note.get('tags') else ""
                        safe_print(f"{i}. [{note.get('created', '')[:16]}] {note.get('content', '')[:60]}...{tags_str}")
            elif args[0].lower() == "tag":
                if len(args) < 2:
                    safe_print("[warning]Usage: /notebook tag <tag>[/]")
                    return

                tag = args[1]
                notes = notebook.get_notes(tag=tag)
                if not notes:
                    safe_print(f"[dim]No notes with tag '{tag}'[/]")
                else:
                    safe_print(f"[bold #ffb3d9] Notes tagged '{tag}' ({len(notes)}):[/bold #ffb3d9]")
                    for i, note in enumerate(notes[:10], 1):
                        safe_print(f"{i}. [{note.get('created', '')[:16]}] {note.get('content', '')[:60]}...")
            elif args[0].lower() == "delete":
                if len(args) < 2:
                    safe_print("[warning]Usage: /notebook delete <id>[/]")
                    return

                note_id = args[1]
                if notebook.delete_note(note_id):
                    safe_print("[green][DONE] Note deleted[/]")
                else:
                    safe_print("[red]Note not found[/]")
            else:
                safe_print("[warning]Usage: /notebook [add|search|tag|delete][/]")
        except Exception as e:
            safe_print(f"[red]Notebook error: {e}[/]")
        return

    elif cmd == "/calendar":
        """Calendar - manage events and reminders."""
        try:
            from aither_adk.tools.qol_tools import get_calendar
            calendar = get_calendar()

            if not args:
                # Show upcoming events
                events = calendar.get_events(upcoming=True)
                if not events:
                    safe_print("[dim]No upcoming events. Use '/calendar add <title> <date> [time]' to create one.[/]")
                else:
                    safe_print(f"[bold cyan] Upcoming Events ({len(events)}):[/bold cyan]")
                    for i, event in enumerate(events[:10], 1):
                        date_str = event.get('date', '')
                        time_str = event.get('time', '00:00')
                        safe_print(f"{i}. [{date_str} {time_str}] {event.get('title', '')}")
                        if event.get('description'):
                            safe_print(f"   {event.get('description')[:50]}...")
            elif args[0].lower() == "add":
                if len(args) < 3:
                    safe_print("[warning]Usage: /calendar add <title> <date> [time] [--desc description][/]")
                    safe_print("[dim]Date format: YYYY-MM-DD, Time format: HH:MM[/]")
                    return

                title = args[1]
                date = args[2]
                time_str = None
                description = None

                # Parse optional time and description
                i = 3
                while i < len(args):
                    if args[i] == "--desc" and i + 1 < len(args):
                        description = " ".join(args[i+1:])
                        break
                    elif ":" in args[i] and not time_str:
                        time_str = args[i]
                    i += 1

                event = calendar.add_event(title, date, time_str, description)
                safe_print(f"[green][DONE] Event added: {title} on {date}[/]")
            elif args[0].lower() == "today":
                today = datetime.datetime.now().date().isoformat()
                events = calendar.get_events(date=today)
                if not events:
                    safe_print(f"[dim]No events scheduled for today ({today})[/]")
                else:
                    safe_print(f"[bold cyan] Today's Events ({len(events)}):[/bold cyan]")
                    for i, event in enumerate(events, 1):
                        time_str = event.get('time', '00:00')
                        safe_print(f"{i}. [{time_str}] {event.get('title', '')}")
            elif args[0].lower() == "delete":
                if len(args) < 2:
                    safe_print("[warning]Usage: /calendar delete <id>[/]")
                    return

                event_id = args[1]
                if calendar.delete_event(event_id):
                    safe_print("[green][DONE] Event deleted[/]")
                else:
                    safe_print("[red]Event not found[/]")
            else:
                safe_print("[warning]Usage: /calendar [add|today|delete][/]")
        except Exception as e:
            safe_print(f"[red]Calendar error: {e}[/]")
        return

    elif cmd == "/timer":
        """Timer - countdown timers."""
        try:
            from aither_adk.tools.qol_tools import get_timer_manager
            timer_mgr = get_timer_manager()

            if not args:
                # Show active timers
                timers = timer_mgr.get_timers(active_only=True)
                if not timers:
                    safe_print("[dim]No active timers. Use '/timer set <name> <seconds>' to create one.[/]")
                else:
                    safe_print(f"[bold cyan][TIMER] Active Timers ({len(timers)}):[/bold cyan]")
                    for i, timer in enumerate(timers, 1):
                        remaining = timer.get('remaining_seconds', 0)
                        mins, secs = divmod(remaining, 60)
                        safe_print(f"{i}. {timer.get('name', 'Unnamed')}: {int(mins)}m {int(secs)}s remaining")
            elif args[0].lower() == "set":
                if len(args) < 3:
                    safe_print("[warning]Usage: /timer set <name> <seconds> [message][/]")
                    return

                name = args[1]
                try:
                    seconds = int(args[2])
                    message = " ".join(args[3:]) if len(args) > 3 else None

                    timer = timer_mgr.create_timer(name, seconds, message)
                    mins, secs = divmod(seconds, 60)
                    safe_print(f"[green][DONE] Timer '{name}' set for {int(mins)}m {int(secs)}s[/]")
                except ValueError:
                    safe_print("[red]Invalid seconds value[/]")
            elif args[0].lower() == "cancel":
                if len(args) < 2:
                    safe_print("[warning]Usage: /timer cancel <id>[/]")
                    return

                timer_id = args[1]
                if timer_mgr.cancel_timer(timer_id):
                    safe_print("[green][DONE] Timer cancelled[/]")
                else:
                    safe_print("[red]Timer not found or already finished[/]")
            else:
                safe_print("[warning]Usage: /timer [set|cancel][/]")
        except Exception as e:
            safe_print(f"[red]Timer error: {e}[/]")
        return

    elif cmd == "/stopwatch":
        """Stopwatch - track elapsed time."""
        try:
            from aither_adk.tools.qol_tools import get_stopwatch_manager
            sw_mgr = get_stopwatch_manager()

            if not args:
                # Show status
                sw = sw_mgr.stopwatches.get("default", {})
                if sw.get('status') == 'running':
                    elapsed = sw_mgr.get_elapsed("default")
                    mins, secs = divmod(int(elapsed), 60)
                    hours, mins = divmod(mins, 60)
                    safe_print(f"[bold cyan][TIMER] Stopwatch Running: {int(hours)}h {int(mins)}m {int(secs)}s[/bold cyan]")
                elif sw.get('status') == 'paused':
                    elapsed = sw_mgr.get_elapsed("default")
                    mins, secs = divmod(int(elapsed), 60)
                    hours, mins = divmod(mins, 60)
                    safe_print(f"[bold cyan][TIMER] Stopwatch Paused: {int(hours)}h {int(mins)}m {int(secs)}s[/bold cyan]")
                else:
                    safe_print("[dim]Stopwatch not running. Use '/stopwatch start' to begin.[/]")
            elif args[0].lower() == "start":
                sw_mgr.start_stopwatch("default")
                safe_print("[green][DONE] Stopwatch started[/]")
            elif args[0].lower() == "pause":
                sw_mgr.pause_stopwatch("default")
                elapsed = sw_mgr.get_elapsed("default")
                mins, secs = divmod(int(elapsed), 60)
                hours, mins = divmod(mins, 60)
                safe_print(f"[yellow]⏸ Stopwatch paused: {int(hours)}h {int(mins)}m {int(secs)}s[/yellow]")
            elif args[0].lower() == "stop":
                sw = sw_mgr.stop_stopwatch("default")
                if sw:
                    elapsed = sw_mgr.get_elapsed("default")
                    mins, secs = divmod(int(elapsed), 60)
                    hours, mins = divmod(mins, 60)
                    safe_print(f"[green][DONE] Stopwatch stopped: {int(hours)}h {int(mins)}m {int(secs)}s[/green]")
            elif args[0].lower() == "reset":
                sw_mgr.reset_stopwatch("default")
                safe_print("[green][DONE] Stopwatch reset[/]")
            else:
                safe_print("[warning]Usage: /stopwatch [start|pause|stop|reset][/]")
        except Exception as e:
            safe_print(f"[red]Stopwatch error: {e}[/]")
        return

    elif cmd == "/alarm":
        """Alarm - set alarms."""
        try:
            from aither_adk.tools.qol_tools import get_alarm_manager
            alarm_mgr = get_alarm_manager()

            if not args:
                # Show all alarms
                alarms = alarm_mgr.get_alarms(enabled_only=False)
                if not alarms:
                    safe_print("[dim]No alarms set. Use '/alarm set <time> [message]' to create one.[/]")
                else:
                    safe_print(f"[bold #ffb3d9][BELL] Alarms ({len(alarms)}):[/bold #ffb3d9]")
                    for i, alarm in enumerate(alarms, 1):
                        status = "[DONE]" if alarm.get('enabled') else "[FAIL]"
                        repeat = alarm.get('repeat', 'once')
                        safe_print(f"{i}. {status} [{alarm.get('time', '')}] {alarm.get('message', 'Alarm!')} ({repeat})")
            elif args[0].lower() == "set":
                if len(args) < 2:
                    safe_print("[warning]Usage: /alarm set <time> [message] [--repeat daily|weekly|weekdays][/]")
                    safe_print("[dim]Time format: HH:MM (24-hour)[/]")
                    return

                time_str = args[1]
                message = None
                repeat = None

                # Parse message and repeat
                i = 2
                msg_parts = []
                while i < len(args):
                    if args[i] == "--repeat" and i + 1 < len(args):
                        repeat = args[i+1]
                        i += 2
                    else:
                        msg_parts.append(args[i])
                        i += 1

                if msg_parts:
                    message = " ".join(msg_parts)

                alarm = alarm_mgr.add_alarm(time_str, message, repeat)
                safe_print(f"[green][DONE] Alarm set for {time_str}[/]")
            elif args[0].lower() == "toggle":
                if len(args) < 2:
                    safe_print("[warning]Usage: /alarm toggle <id>[/]")
                    return

                alarm_id = args[1]
                if alarm_mgr.toggle_alarm(alarm_id):
                    safe_print("[green][DONE] Alarm toggled[/]")
                else:
                    safe_print("[red]Alarm not found[/]")
            elif args[0].lower() == "delete":
                if len(args) < 2:
                    safe_print("[warning]Usage: /alarm delete <id>[/]")
                    return

                alarm_id = args[1]
                if alarm_mgr.delete_alarm(alarm_id):
                    safe_print("[green][DONE] Alarm deleted[/]")
                else:
                    safe_print("[red]Alarm not found[/]")
            else:
                safe_print("[warning]Usage: /alarm [set|toggle|delete][/]")
        except Exception as e:
            safe_print(f"[red]Alarm error: {e}[/]")
        return

    elif cmd == "/resources":
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager

            # Get resource manager instance (singleton)
            rm = resource_manager

            # Get status (synchronous method)
            status = rm.get_status()

            # Extract GPU info from status
            total_mb = status.get('vram_total_mb', 0)
            used_mb = status.get('vram_used_mb', 0)
            free_mb = status.get('vram_free_mb', 0)
            gpu_util = status.get('gpu_utilization', 0)

            if total_mb > 0:
                percent = (used_mb / total_mb) * 100

                # Color based on usage
                if percent < 50:
                    gpu_color = "green"
                elif percent < 80:
                    gpu_color = "yellow"
                else:
                    gpu_color = "red"

                gpu_status = f"[{gpu_color}]{used_mb}MB / {total_mb}MB ({percent:.1f}%)[/]"
                gpu_free = f"[{gpu_color}]{free_mb}MB free[/]"
                gpu_util_str = f"[{gpu_color}]{gpu_util:.1f}%[/]"
            else:
                gpu_status = "[dim]Unavailable[/]"
                gpu_free = "[dim]N/A[/]"
                gpu_util_str = "[dim]N/A[/]"

            active = status.get('active_tasks', 0)
            queue_size = status.get('queue_size', 0)
            diffusion_busy = status.get('diffusion_busy', False)
            llm_busy = status.get('llm_busy', False)
            stats = status.get('stats', {})

            safe_print(Panel(f"""
[bold]GPU Memory:[/bold] {gpu_status}
[bold]GPU Free:[/bold] {gpu_free}
[bold]GPU Utilization:[/bold] {gpu_util_str}
[bold]Active Tasks:[/bold] {active}
[bold]Queue Size:[/bold] {queue_size}
[bold]Resource Status:[/bold]
  * Diffusion: {"[red]Busy[/]" if diffusion_busy else "[green]Available[/]"}
  * LLM: {"[red]Busy[/]" if llm_busy else "[green]Available[/]"}
[bold]Statistics:[/bold]
  * Total Tasks: {stats.get('total_tasks', 0)}
  * Local Tasks: {stats.get('local_tasks', 0)}
  * Cloud Tasks: {stats.get('cloud_tasks', 0)}
  * VRAM Conflicts: {stats.get('vram_conflicts', 0)}
""", title="[bold cyan]Resource Status[/bold cyan]", border_style="cyan"))
        except ImportError:
            safe_print("[yellow]Resource manager not available.[/]")
        except Exception as e:
            safe_print(f"[red]Failed to get resource status: {e}[/]")

    elif cmd == "/settings":
        # Check for subcommand (e.g., /settings model mistral-nemo)
        if args:
            subcmd = args[0].lower()

            if subcmd == "model" and len(args) >= 2:
                # Set default model
                new_default = args[1]
                try:
                    import AitherOS.agents.Saga.agent as saga_agent

                    # Check if it's in the available models list
                    if new_default in models or new_default.startswith('ollama/'):
                        # Update the defaults
                        if new_default.startswith('ollama/'):
                            saga_agent.LOCAL_MODEL_NAME = new_default.replace('ollama/', '')
                        elif not new_default.startswith(('gemini', 'gpt', 'claude')):
                            saga_agent.LOCAL_MODEL_NAME = new_default
                            saga_agent.USE_LOCAL_MODELS = True
                        else:
                            saga_agent.USE_LOCAL_MODELS = False

                        # Also update current agent model and toolbar
                        agent.model = new_default
                        try:
                            from aither_adk.ui.ui import update_current_model
                            update_current_model(new_default)
                        except ImportError:
                            pass

                        safe_print(f"[green][OK] Default model set to: [bold]{new_default}[/bold][/green]")
                        return
                    else:
                        safe_print(f"[yellow]Model '{new_default}' not in available models list.[/yellow]")
                        safe_print(f"[dim]Available: {', '.join(models[:5])}...[/dim]")
                        return
                except (ImportError, AttributeError) as e:
                    safe_print(f"[red]Failed to set default model: {e}[/red]")
                    return

            elif subcmd == "help":
                safe_print(Panel(
                    "[bold]Settings Commands:[/bold]\n\n"
                    "  [bold #ffb3d9]/settings[/bold #ffb3d9]              - Show all settings\n"
                    "  [bold #ffb3d9]/settings model <name>[/bold #ffb3d9] - Set default LLM model\n"
                    "  [bold #ffb3d9]/settings help[/bold #ffb3d9]         - Show this help\n\n"
                    "[dim]Examples:[/dim]\n"
                    "  /settings model mistral-nemo\n"
                    "  /settings model gemini-2.5-flash",
                    title="[bold cyan]Settings Help[/bold cyan]",
                    border_style="cyan"
                ))
                return
            else:
                safe_print(f"[yellow]Unknown setting: {subcmd}. Try /settings help[/yellow]")
                return

        # Build comprehensive settings display
        settings_lines = []

        # Model & Session
        model_name = getattr(agent, 'model', 'unknown')
        settings_lines.append(f"[bold]Current Model:[/bold] {model_name}")

        # Default model info
        try:
            import AitherOS.agents.Saga.agent as saga_agent
            default_model = saga_agent.LOCAL_MODEL_NAME if saga_agent.USE_LOCAL_MODELS else "gemini-2.5-flash"
            use_local = saga_agent.USE_LOCAL_MODELS
            settings_lines.append(f"[bold]Default Model:[/bold] {default_model} ({'local' if use_local else 'cloud'})")
        except Exception as exc:
            logger.debug(f"Default model info retrieval failed: {exc}")

        settings_lines.append(f"[bold]Session ID:[/bold] {session_id[:8]}...")

        # Safety Mode (with emoji and details)
        try:
            from aither_adk.ai.safety_mode import (
                get_level_emoji,
                get_level_name,
                get_safety_manager,
            )
            safety = get_safety_manager()
            level = safety.current_level
            config = safety.get_config()
            settings_lines.append(f"[bold]Safety Mode:[/bold] {get_level_emoji(level)} {get_level_name(level)}")
            settings_lines.append(f"[bold]LLM:[/bold] {'Gemini (Cloud)' if config.use_cloud_llm else 'Local (Ollama)'}")
            settings_lines.append(f"[bold]Explicit Content:[/bold] {'Allowed' if config.allow_explicit else 'Blocked'}")
        except Exception:
            # Fallback to old method
            safety_status = "Unknown"
            if agent.generate_content_config and agent.generate_content_config.safety_settings:
                first_setting = agent.generate_content_config.safety_settings[0]
                safety_status = str(first_setting.threshold).split('.')[-1] if first_setting.threshold else "Default"
            settings_lines.append(f"[bold]Safety Level:[/bold] {safety_status}")

        # Roleplay Detail Level
        try:
            roleplay_detail = os.getenv("AITHER_ROLEPLAY_DETAIL", "minimal").lower()
            level_emoji = {
                "minimal": "[NOTE]",
                "moderate": "[FILE]",
                "detailed": "",
                "extensive": ""
            }
            emoji = level_emoji.get(roleplay_detail, "[NOTE]")
            settings_lines.append(f"[bold]Roleplay Detail:[/bold] {emoji} {roleplay_detail.title()}")
        except Exception:
            # Fallback if there's an issue
            settings_lines.append("[bold]Roleplay Detail:[/bold] Minimal (default)")

        # Cost & Usage
        settings_lines.append(f"[bold]Total Cost:[/bold] ${session_stats.get('total_cost', 0):.6f}")
        settings_lines.append(f"[bold]Tokens:[/bold] {session_stats.get('total_input', 0)} in / {session_stats.get('total_output', 0)} out")

        # Active Persona
        try:
            if hasattr(agent, 'name'):
                settings_lines.append(f"[bold]Active Persona:[/bold] {agent.name}")
        except Exception as exc:
            logger.debug(f"Active persona retrieval failed: {exc}")

        # QOL Status
        try:
            from aither_adk.tools.qol_tools import (
                get_alarm_manager,
                get_calendar,
                get_notebook,
                get_stopwatch_manager,
                get_timer_manager,
            )

            notebook = get_notebook()
            calendar = get_calendar()
            timer_mgr = get_timer_manager()
            sw_mgr = get_stopwatch_manager()
            alarm_mgr = get_alarm_manager()

            settings_lines.append("")
            settings_lines.append("[bold cyan]Quality of Life:[/bold cyan]")
            settings_lines.append(f"   Notes: {len(notebook.notes)}")
            settings_lines.append(f"   Events: {len(calendar.events)}")
            settings_lines.append(f"  [TIMER] Active Timers: {len([t for t in timer_mgr.get_timers(active_only=True)])}")

            sw = sw_mgr.stopwatches.get("default", {})
            if sw.get('status') == 'running':
                elapsed = sw_mgr.get_elapsed("default")
                mins, secs = divmod(int(elapsed), 60)
                settings_lines.append(f"  [TIMER] Stopwatch: {int(mins)}m {int(secs)}s")

            settings_lines.append(f"  [BELL] Alarms: {len([a for a in alarm_mgr.get_alarms(enabled_only=True)])}")
        except Exception:
            pass

        # Memory Status
        try:
            if memory_manager:
                mem = memory_manager
                world_count = len(mem.world.entries) if hasattr(mem.world, 'entries') else 0
                system_count = len(mem.system.config) if hasattr(mem.system, 'config') else 0
                settings_lines.append("")
                settings_lines.append("[bold cyan]Memory:[/bold cyan]")
                settings_lines.append(f"   World: {world_count} entries")
                settings_lines.append(f"  [GEAR] System: {system_count} entries")
        except Exception as exc:
            logger.debug(f"Memory manager info retrieval failed: {exc}")

        safe_print(Panel("\n".join(settings_lines), title="[bold cyan]Settings[/bold cyan]", border_style="cyan"))

    elif cmd == "/role":
        if not args:
            safe_print("[bold cyan]Available Personas:[/]")
            for role, data in personas.items():
                safe_print(f" - [bold #ffb3d9]{role}[/]: {data['description']}")
            return

        new_role = args[0].lower()
        if new_role in personas:
            safe_print(f"[info]Switching persona to [bold #ffb3d9]{new_role}[/]...[/]")

            # Update instruction
            new_instruction = personas[new_role]["instruction"]
            if new_instruction is None:
                # Reload default
                try:
                    from prompts import SYSTEM_INSTRUCTION
                except ImportError:
                    # Fallback if prompts module is not available in current scope
                    # This might need adjustment depending on how prompts are loaded
                    new_instruction = "You are a helpful assistant."
                new_instruction = SYSTEM_INSTRUCTION

            # Update agent instruction
            agent.instruction = new_instruction
            # Update persona name for toolbar (stored in session_stats)
            session_stats['persona_name'] = new_role
            safe_print(f"[green]Persona updated to {new_role}.[/]")
        else:
            safe_print(f"[danger]Persona '{new_role}' not found.[/]")

    elif cmd == "/model":
        if not args:
            safe_print("[bold cyan]Available Models:[/]")
            for m in models:
                safe_print(f" - {m}")
            safe_print(f"\nCurrent: [bold green]{agent.model}[/]")
            return

        new_model = args[0]
        # Simple fuzzy match or exact match
        matched_model = None
        if new_model in models:
            matched_model = new_model
        else:
            # Try to find by index or partial name
            try:
                idx = int(new_model) - 1
                if 0 <= idx < len(models):
                    matched_model = models[idx]
            except ValueError:
                pass

        # If not in models list, check if it's a local Ollama model
        if not matched_model:
            try:
                from aither_adk.ai.models import is_local_model
                # If it looks like a local model (mistral, llama, etc.) or has :tag suffix
                if is_local_model(new_model) or ':' in new_model:
                    # Add ollama/ prefix if not already present
                    if not new_model.startswith('ollama/'):
                        matched_model = f"ollama/{new_model}"
                    else:
                        matched_model = new_model
                    safe_print(f"[dim]Detected Ollama model: {matched_model}[/]")
            except ImportError:
                pass

        # Ensure Ollama models have the proper prefix for ADK LLMRegistry
        # Even if matched from the models list, local models need ollama/ prefix
        if matched_model and not matched_model.startswith(('gemini', 'gpt-', 'claude', 'o1-', 'o3-', 'ollama/', 'aither/')):
            try:
                from aither_adk.ai.models import is_local_model
                if is_local_model(matched_model):
                    matched_model = f"ollama/{matched_model}"
                    safe_print(f"[dim]Added provider prefix: {matched_model}[/]")
            except ImportError:
                pass

        if matched_model:
            safe_print(f"[info]Switching model to {matched_model}...[/]")
            agent.model = matched_model

            # Update the toolbar to show the new model
            try:
                from aither_adk.ui.ui import update_current_model
                update_current_model(matched_model)
            except ImportError:
                pass

            # Also update the module-level LOCAL_MODEL_NAME for the aither() tool
            # so it uses the correct model for chat/roleplay
            try:
                import AitherOS.agents.Saga.agent as saga_agent
                # Check if this is a local Ollama model (not gemini/gpt)
                if not matched_model.startswith(('gemini', 'gpt', 'claude', 'ollama/')):
                    saga_agent.LOCAL_MODEL_NAME = matched_model
                    saga_agent.USE_LOCAL_MODELS = True
                elif matched_model.startswith('ollama/'):
                    saga_agent.LOCAL_MODEL_NAME = matched_model.replace('ollama/', '')
                    saga_agent.USE_LOCAL_MODELS = True
                else:
                    saga_agent.USE_LOCAL_MODELS = False
            except (ImportError, AttributeError):
                pass  # Module not available

            print_banner(matched_model)
        else:
            safe_print(f"[danger]Model '{new_model}' not found.[/]")

    elif cmd == "/attach":
        if not args:
            safe_print("[red]Usage: /attach <path> or /attach last[/]")
            return

        path = args[0]

        if path.lower() == "last":
            # Find latest file in generated_images
            import glob
            try:
                list_of_files = glob.glob('generated_images/*')
                if not list_of_files:
                    safe_print("[red]No generated images found.[/]")
                    return
                latest_file = max(list_of_files, key=os.path.getctime)
                path = latest_file
            except Exception as e:
                safe_print(f"[red]Error finding last image: {e}[/]")
                return

        if not os.path.exists(path):
            safe_print(f"[red]File not found: {path}[/]")
            return

        try:
            with open(path, "rb") as f:
                data = f.read()

            # Determine mime type
            mime_type = "application/octet-stream"
            if path.endswith(".png"): mime_type = "image/png"
            elif path.endswith(".jpg") or path.endswith(".jpeg"): mime_type = "image/jpeg"
            elif path.endswith(".txt"): mime_type = "text/plain"
            elif path.endswith(".pdf"): mime_type = "application/pdf"

            add_attachment(types.Part.from_bytes(data=data, mime_type=mime_type))
            safe_print(f"[green]Attached '{path}' ({len(data)} bytes). It will be sent with your next message.[/]")
        except Exception as e:
            safe_print(f"[red]Failed to attach file: {e}[/]")

    elif cmd == "/turns":
        if not group_chat_manager:
            safe_print("[red]Group chat manager not available.[/]")
            return

        if not args:
            current = group_chat_manager.state.get("scene_max_turns", "Auto")
            count = group_chat_manager.state.get("scene_turn_count", 0)
            safe_print(f"[bold cyan]Scene Turns:[/][bold white] {count} / {current}[/]")
            safe_print("[dim]Usage: /turns <number> (Set max turns for current/next scene)[/]")
            return

        try:
            turns = int(args[0])
            group_chat_manager.state["scene_max_turns"] = turns
            group_chat_manager.state["manual_turn_limit"] = True # Flag to prevent auto-override
            safe_print(f"[green]Max turns set to {turns}.[/]")

            # If scene is active, update status
            if group_chat_manager.state.get("scene_active"):
                current = group_chat_manager.state.get("scene_turn_count", 0)
                remaining = turns - current
                if remaining <= 0:
                    safe_print("[yellow]Turn limit reached! Next turn will trigger conclusion.[/]")
                else:
                    safe_print(f"[dim]{remaining} turns remaining in current scene.[/]")
        except ValueError:
            safe_print("[red]Invalid number.[/]")

    elif cmd == "/continue":
        # This is handled in the main loop, but we can acknowledge it here
        safe_print("[dim]Continuing conversation...[/]")
        return "CONTINUE_SIGNAL" # Special return value to signal continuation

    elif cmd == "/profile":
        if not args:
            Profiler.print_stats()
        elif args[0] == "reset":
            Profiler.reset()
            safe_print("[green]Profiling stats reset.[/]")
        elif args[0] == "enable":
            Profiler.enable()
            safe_print("[green]Profiling enabled.[/]")
        elif args[0] == "disable":
            Profiler.disable()
            safe_print("[yellow]Profiling disabled.[/]")
        else:
            safe_print("[warning]Usage: /profile [reset|enable|disable][/]")
        return

    elif cmd == "/vram":
        # Show GPU VRAM status
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager
            resource_manager.print_status()
        except ImportError:
            safe_print("[warning]Resource manager not available. Install pynvml.[/]")
        return

    elif cmd == "/models":
        # Show loaded models using SYNCHRONOUS requests (avoids event loop issues)
        try:
            import requests

            from lib.core.AitherPorts import ollama_url as get_ollama_url
            ollama_url = get_ollama_url()

            safe_print("\n" + "="*50)
            safe_print("[BRAIN] Loaded Models")
            safe_print("="*50)

            try:
                resp = requests.get(f"{ollama_url}/api/ps", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", [])
                    if models:
                        safe_print("[cyan]Ollama (loaded in VRAM):[/]")
                        total_vram = 0
                        for m in models:
                            name = m.get("name", "unknown")
                            m.get("size", 0) // (1024*1024)  # Convert to MB
                            vram = m.get("size_vram", 0) // (1024*1024)
                            safe_print(f"  * {name} ({vram:,}MB VRAM)")
                            total_vram += vram
                        safe_print(f"  [dim]Total VRAM: ~{total_vram:,}MB[/]")
                    else:
                        safe_print("[dim]No Ollama models currently loaded in VRAM[/]")
            except Exception as e:
                safe_print(f"[dim]Could not query Ollama: {e}[/]")

            safe_print("\n[dim]Commands:[/]")
            safe_print("  /unload <model>  - Unload specific model")
            safe_print("  /unload all      - Unload all models")
            safe_print("  /load <model>    - Preload a model")
            safe_print("="*50 + "\n")
        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/unload":
        # Unload models using ResourceManager
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager
            resource_manager.unload_all_models()
            safe_print("[green][DONE] Unloaded all models (Ollama & ComfyUI)[/]")
        except Exception as e:
            safe_print(f"[red]Error unloading models: {e}[/]")



    elif cmd == "/load":
        # Preload a model using SYNCHRONOUS requests
        if not args:
            safe_print("[warning]Usage: /load <model_name>[/]")
            safe_print("[dim]Available: mistral-nemo, llama3.2-vision, llama3.2, phi3[/]")
            return

        try:
            import requests

            from lib.core.AitherPorts import ollama_url as get_ollama_url
            ollama_url = get_ollama_url()
            model = args[0]

            safe_print(f"[dim]Loading {model} (this may take a moment)...[/]")
            resp = requests.post(f"{ollama_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": "5m"},
                timeout=120)

            if resp.status_code == 200:
                safe_print(f"[green][DONE] {model} loaded (will stay for 5 minutes)[/]")
            else:
                safe_print(f"[warning]Failed to load {model}: {resp.text}[/]")
        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/anchor":
        # Dual mode: Manual Set or Auto-Generate
        # Usage 1 (Auto): /anchor [persona_name|all] [--force]
        # Usage 2 (Manual): /anchor <persona> <type> <image_path>

        if not args:
             safe_print("[warning]Usage 1 (Auto): /anchor [persona_name|all] [--force][/]")
             safe_print("[warning]Usage 2 (Manual): /anchor <persona> <type> <image_path>[/]")
             return

        # Check for force flag
        force = False
        if "--force" in args:
            force = True
            args.remove("--force")

        # Mode 1: Auto-Generate (1 argument)
        if len(args) == 1:
            target = args[0]
            try:
                from aither_adk.memory.anchor_generator import AnchorGenerator
                generator = AnchorGenerator()

                if target.lower() == "all":
                    await generator.generate_all_anchors(force=force)
                else:
                    await generator.generate_anchors_for_persona(target, force=force)
            except Exception as e:
                safe_print(f"[red]Error generating anchors: {e}[/]")
                import traceback
                traceback.print_exc()
            return

        # Mode 2: Manual Set (3+ arguments)
        if len(args) < 3:
            safe_print("[warning]Usage: /anchor <persona> <type> <image_path>[/]")
            safe_print("[dim]Types: face, body, style[/]")
            return

        try:
            from aither_adk.ai.persona_image_system import set_persona_anchor
            persona_name = args[0]
            anchor_type = args[1]
            image_path = " ".join(args[2:])

            if anchor_type not in ["face", "body", "style"]:
                safe_print("[warning]Type must be: face, body, or style[/]")
                return

            if not os.path.exists(image_path):
                safe_print(f"[warning]Image not found: {image_path}[/]")
                return

            safe_print(f"[dim]Setting {anchor_type} anchor for {persona_name}...[/]")
            safe_print("[dim]Analyzing with vision model...[/]")

            if set_persona_anchor(persona_name, image_path, anchor_type):
                safe_print(f"[green][DONE] {anchor_type.title()} anchor set for {persona_name}[/]")
            else:
                safe_print("[warning]Failed to set anchor[/]")
        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/persona-info":
        # Show info about a persona's visual identity
        # Usage: /persona-info <persona>
        if not args:
            safe_print("[warning]Usage: /persona-info <persona>[/]")
            return

        try:
            from aither_adk.ai.persona_image_system import get_persona_image_system
            system = get_persona_image_system()
            persona_name = args[0].lower()

            anchor = system.get_anchor(persona_name)
            if not anchor:
                anchor = system.create_persona_from_yaml(persona_name)

            if anchor:
                from rich.table import Table
                table = Table(title=f"[bold]{anchor.display_name}[/] Visual Identity")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")

                # Identity
                table.add_row("Hair", f"{anchor.identity.hair_color} {anchor.identity.hair_style} {anchor.identity.hair_length}".strip())
                table.add_row("Eyes", f"{anchor.identity.eye_color} {anchor.identity.eye_style}".strip())
                table.add_row("Skin", anchor.identity.skin_tone)
                table.add_row("Body", anchor.identity.body_type)
                table.add_row("Bust", anchor.identity.bust_size)
                table.add_row("Hips", anchor.identity.hip_width)
                table.add_row("Marks", anchor.identity.distinguishing_marks)
                table.add_row("Face", anchor.identity.facial_features)

                # Anchors
                table.add_row("-" * 20, "-" * 30)
                table.add_row("Face Anchor", anchor.face_reference or "[dim]Not set[/]")
                table.add_row("Body Anchor", anchor.body_reference or "[dim]Not set[/]")
                table.add_row("Style Anchor", anchor.style_reference or "[dim]Not set[/]")

                console.print(table)

                # Show prompt preview
                if anchor.identity.to_prompt_tags():
                    safe_print("\n[dim]Prompt tags:[/]")
                    safe_print(f"[cyan]{anchor.identity.to_prompt_tags()}[/]")
            else:
                safe_print(f"[warning]Persona '{persona_name}' not found[/]")
        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/scene":
        # Set or show current scene context
        # Usage: /scene [location] [lighting]
        try:
            from aither_adk.ai.persona_image_system import (
                get_persona_image_system,
                set_scene_context,
            )
            system = get_persona_image_system()

            if not args:
                # Show current scene
                safe_print("[bold]Current Scene:[/]")
                safe_print(f"  Location: {system.scene.location or '[dim]Not set[/]'}")
                safe_print(f"  Lighting: {system.scene.lighting or '[dim]Not set[/]'}")
                safe_print(f"  Time: {system.scene.time_of_day or '[dim]Not set[/]'}")
                safe_print(f"  Atmosphere: {system.scene.atmosphere or '[dim]Not set[/]'}")
                safe_print(f"  Active: {', '.join(system.scene.active_personas) or '[dim]None[/]'}")
                return

            if args[0] == "clear":
                system.clear_scene()
                safe_print("[green][DONE] Scene cleared[/]")
                return

            # Parse: /scene location="office" lighting="neon"
            location = None
            lighting = None

            for arg in args:
                if "=" in arg:
                    key, val = arg.split("=", 1)
                    val = val.strip('"').strip("'")
                    if key == "location":
                        location = val
                    elif key == "lighting":
                        lighting = val
                else:
                    # Assume first arg is location
                    if not location:
                        location = arg

            set_scene_context(location=location, lighting=lighting)
            safe_print("[green][DONE] Scene updated[/]")

        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/gen":
        # Quick image generation preview
        # Usage: /gen <prompt>  (assumes aither)
        # Usage: /gen @<persona> <prompt>
        if not args:
            safe_print("[warning]Usage: /gen <prompt> or /gen @persona <prompt>[/]")
            safe_print("[dim]Example: /gen send a selfie[/]")
            safe_print("[dim]Example: /gen @aither show your ass[/]")
            return

        try:
            # Parse persona (optional @persona prefix)
            persona_name = "aither"
            request = " ".join(args)

            if args[0].startswith("@"):
                persona_name = args[0][1:]  # Remove @
                request = " ".join(args[1:])

            from aither_adk.ai.persona_image_system import generate_persona_prompt

            safe_print(f"[dim]Building prompt for {persona_name}...[/]")
            result = generate_persona_prompt(persona_name, request)

            # Show FULL prompts, no truncation!
            safe_print("\n[bold cyan]=== GENERATED PROMPT ===[/]")
            safe_print(f"[cyan]{result['prompt']}[/]")

            safe_print("\n[bold red]=== NEGATIVE PROMPT ===[/]")
            safe_print(f"[red]{result['negative_prompt']}[/]")

            if result.get('controlnet_image'):
                safe_print("\n[bold yellow]=== CONTROLNET ===[/]")
                safe_print(f"[yellow]Reference: {result['controlnet_image']}[/]")
                safe_print(f"[yellow]Model: {result.get('controlnet_model', 'auto')}[/]")

            safe_print(f"\n[bold]Model: {result.get('model_preference', 'pony')} | Seed: {result.get('seed', 'random')}[/]")

            # Copy to clipboard if available
            try:
                import pyperclip
                pyperclip.copy(result['prompt'])
                safe_print("[green] Prompt copied to clipboard[/]")
            except Exception as exc:
                logger.debug(f"Clipboard copy failed: {exc}")

            # Auto-refinement if requested (hidden flag for now or just auto-run)
            # For now, we just show the prompt.
            # But if the user wants to run it, they can use /refine

        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
            import traceback
            traceback.print_exc()
        return

    elif cmd == "/refine":
        # Usage: /refine <image_path> <instruction>
        # Usage: /refine "last" <instruction>
        if len(args) < 2:
            safe_print("[warning]Usage: /refine <image_path|last> <instruction>[/]")
            return

        path = args[0]
        instruction = " ".join(args[1:])

        # Resolve "last"
        if path.lower() == "last":
            # Find most recent image in output dir
            # This is a bit hacky, need a better way to track last generated image
            # For now, let's look in Saga/output/refinements
            try:
                from aither_adk.paths import get_saga_subdir
                base_dir = get_saga_subdir("output", "refinements")
            except ImportError:
                base_dir = os.path.join(os.path.dirname(__file__), "..", "Saga", "output", "refinements")
            if os.path.exists(base_dir):
                files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".png")]
                if files:
                    path = max(files, key=os.path.getctime)
                else:
                    safe_print("[red]No recent images found.[/]")
                    return
            else:
                safe_print("[red]No output directory found.[/]")
                return

        if not os.path.exists(path):
            safe_print(f"[red]Image not found: {path}[/]")
            return

        try:
            from aither_adk.ai.refinement_engine import RefinementEngine
            engine = RefinementEngine()

            result_path = await engine.refine_existing_image(path, instruction)

            if result_path:
                safe_print(f"[green][DONE] Refined image: {result_path}[/]")
                # Try to open
                import subprocess
                if os.name == 'nt':
                    os.startfile(result_path)
                elif os.name == 'posix':
                    subprocess.call(('xdg-open', result_path))
            else:
                safe_print("[red]Refinement failed.[/]")

        except Exception as e:
            safe_print(f"[red]Error: {e}[/]")
            import traceback
            traceback.print_exc()
        return

    elif cmd == "/comic":
        if not args:
            safe_print("[warning]Usage: /comic <topic>[/]")
            return

        topic = " ".join(args)
        safe_print(f"[cyan][ART] Starting comic generation for: {topic}[/]")

        try:
            from aither_adk.memory.storyboard import StoryboardEngine
            engine = StoryboardEngine()
            # Use current persona if available
            persona_name = "aither"
            if hasattr(agent, "persona_name"):
                persona_name = agent.persona_name
            elif hasattr(agent, "name"):
                persona_name = agent.name

            output_path = await engine.create_page(topic, persona_name)

            if output_path:
                safe_print(f"[green][DONE] Comic page generated: {output_path}[/]")
                # Try to open it?
                import subprocess
                if os.name == 'nt':
                    os.startfile(output_path)
                elif os.name == 'posix':
                    subprocess.call(('xdg-open', output_path))
            else:
                safe_print("[red][FAIL] Failed to generate comic page.[/]")

        except Exception as e:
            safe_print(f"[red]Error generating comic: {e}[/]")
            import traceback
            traceback.print_exc()
        return

    elif cmd == "/memory":
        # Memory system commands
        # Usage: /memory [show|add|clear] [world|system|persona] [content]
        try:
            from aither_adk.memory.memory_system import get_memory_system
            mem = get_memory_system()

            if not args:
                # Show memory summary
                safe_print("[bold] Memory System Status[/]")
                safe_print(f"  World facts: {len(mem.world.facts)}")
                safe_print(f"  System entries: {len(mem.system.entries)}")
                safe_print(f"  Personas loaded: {len(mem._persona_cache)}")

                if mem.world.facts:
                    safe_print("\n[bold]World Facts:[/]")
                    for fact in mem.world.facts[-5:]:
                        safe_print(f"  * {fact.content[:60]}...")
                return

            action = args[0].lower()

            if action == "show":
                # /memory show [persona_name]
                if len(args) > 1:
                    persona_name = args[1]
                    ctx = mem.get_full_context(persona_name)
                    safe_print(f"\n[bold]Context for {persona_name}:[/]")
                    safe_print(ctx[:2000])
                else:
                    # Show all with LIVE runtime state
                    safe_print("\n[bold]World Memory:[/]")
                    safe_print(mem.world.get_context())

                    safe_print("\n[bold]System Memory:[/]")
                    safe_print(mem.system.get_context())

                    # Add LIVE runtime state (not from static memory)
                    safe_print("\n[bold]Runtime State (Live):[/]")
                    try:
                        from aither_adk.ai.safety_mode import (
                            get_level_emoji,
                            get_level_name,
                            get_safety_manager,
                        )
                        safety = get_safety_manager()
                        level = safety.current_level
                        config = safety.get_config()
                        safe_print(f"  Safety Mode: {get_level_emoji(level)} {get_level_name(level)}")
                        safe_print(f"  LLM: {'Gemini (Cloud)' if config.use_cloud_llm else 'Local (Ollama)'}")
                        safe_print(f"  Explicit Content: {'Allowed' if config.allow_explicit else 'Blocked'}")
                    except Exception as e:
                        safe_print(f"  [dim]Could not get runtime state: {e}[/]")

            elif action == "add":
                # /memory add world "fact content"
                if len(args) < 3:
                    safe_print("[warning]Usage: /memory add <world|system|persona:name> <content>[/]")
                    return

                mem_type = args[1].lower()
                content = " ".join(args[2:])

                if mem_type == "world":
                    mem.world.add_fact(content)
                    safe_print("[green][DONE] Added world fact[/]")
                elif mem_type == "system":
                    mem.system.add_entry(content)
                    safe_print("[green][DONE] Added system entry[/]")
                elif mem_type.startswith("persona:"):
                    persona_name = mem_type.split(":")[1]
                    mem.get_persona(persona_name).add_trait(content)
                    safe_print(f"[green][DONE] Added trait to {persona_name}[/]")

            elif action == "clear":
                # /memory clear [world|system|persona:name]
                if len(args) < 2:
                    safe_print("[warning]Usage: /memory clear <world|system|persona:name>[/]")
                    return

                mem_type = args[1].lower()
                if mem_type == "world":
                    mem.world.facts = []
                    mem.world.save()
                    safe_print("[yellow]Cleared world memory[/]")
                elif mem_type == "system":
                    mem.system.entries = []
                    mem.system.save()
                    safe_print("[yellow]Cleared system memory[/]")
                elif mem_type.startswith("persona:"):
                    persona_name = mem_type.split(":")[1]
                    persona = mem.get_persona(persona_name)
                    persona.conversation_history.clear()
                    persona.events = []
                    persona.save()
                    safe_print(f"[yellow]Cleared {persona_name}'s memory[/]")

        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
        return

    elif cmd == "/init-personas":
        # Initialize all personas from YAML files
        # Usage: /init-personas [--generate-anchors]
        try:
            from aither_adk.ai.persona_image_system import get_persona_image_system
            system = get_persona_image_system()

            auto_gen = "--generate-anchors" in args or "-g" in args

            if auto_gen:
                safe_print("[yellow][WARN] This will generate anchor images for all personas.[/]")
                safe_print("[yellow]This may take a while and use GPU resources.[/]")

            safe_print("[dim]Initializing personas from YAML files...[/]")

            results = system.initialize_all_personas(auto_generate=auto_gen)

            # Display results
            from rich.table import Table
            table = Table(title="Persona Initialization Results")
            table.add_column("Persona", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Anchor", style="white")

            for name, success in results.items():
                status = "[DONE]" if success else "[FAIL]"
                anchor = system.anchors.get(name)
                has_anchor = "[CAM]" if anchor and anchor.face_reference else "--"
                table.add_row(name, status, has_anchor)

            console.print(table)
            safe_print(f"\n[green]Initialized {sum(results.values())}/{len(results)} personas[/]")

        except Exception as e:
            safe_print(f"[warning]Error: {e}[/]")
            import traceback
            traceback.print_exc()
        return

    elif cmd == "/vision":
        # Vision analysis using AitherVision service
        # Usage: /vision <image_path> [analysis_type] [question]
        # /vision image.png              - Describe the image
        # /vision image.png ocr          - Extract text (OCR)
        # /vision image.png "what color is the car?"
        # /vision compare img1.png img2.png

        if not args:
            safe_print(Panel("""
[bold cyan]Vision Analysis[/bold cyan]

[bold yellow]Usage:[/bold yellow]
  [bold #ffb3d9]/vision <path>[/bold #ffb3d9]                  - Describe an image
  [bold #ffb3d9]/vision <path> detailed[/bold #ffb3d9]         - Detailed analysis
  [bold #ffb3d9]/vision <path> ocr[/bold #ffb3d9]              - Extract text (OCR)
  [bold #ffb3d9]/vision <path> objects[/bold #ffb3d9]          - Identify objects
  [bold #ffb3d9]/vision <path> style[/bold #ffb3d9]            - Analyze artistic style
  [bold #ffb3d9]/vision <path> emotions[/bold #ffb3d9]         - Analyze emotions/mood
  [bold #ffb3d9]/vision <path> "question"[/bold #ffb3d9]       - Ask a question about image
  [bold #ffb3d9]/vision compare <path1> <path2>[/bold #ffb3d9] - Compare two images
  [bold #ffb3d9]/vision status[/bold #ffb3d9]                  - Check vision service status
  [bold #ffb3d9]/vision last[/bold #ffb3d9]                    - Analyze last generated image

[dim]Powered by AitherVision (Ollama llama3.2-vision or Gemini Vision)[/dim]
""", title="[bold cyan]Vision Help[/]", border_style="cyan"))
            return

        import glob

        import requests

        # Check for subcommands
        if args[0].lower() == "status":
            try:
                resp = requests.get("http://localhost:8084/status", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    safe_print("[green][DONE] AitherVision is running[/]")
                    safe_print(f"  Provider: {data.get('provider', 'unknown')}")
                    ollama = data.get('ollama', {})
                    safe_print(f"  Ollama: {'[OK]' if ollama.get('available') else '[X]'}")
                    safe_print(f"  Model: {ollama.get('model', 'unknown')}")
                    safe_print(f"  Model Loaded: {'[OK]' if ollama.get('model_loaded') else '[X]'}")
                else:
                    safe_print(f"[red]AitherVision returned {resp.status_code}[/]")
            except requests.exceptions.ConnectionError:
                safe_print("[red][FAIL] AitherVision is not running[/]")
                safe_print("[dim]Start it with: Start-AitherOS -Service Vision[/]")
            except Exception as e:
                safe_print(f"[red]Error: {e}[/]")
            return

        if args[0].lower() == "compare":
            if len(args) < 3:
                safe_print("[warning]Usage: /vision compare <image1> <image2> [aspect][/]")
                return

            path1 = args[1]
            path2 = args[2]
            aspect = args[3] if len(args) > 3 else "overall"

            if not os.path.exists(path1):
                safe_print(f"[red]Image not found: {path1}[/]")
                return
            if not os.path.exists(path2):
                safe_print(f"[red]Image not found: {path2}[/]")
                return

            safe_print(f"[dim]Comparing images ({aspect})...[/]")
            try:
                resp = requests.post("http://localhost:8084/compare", json={
                    "image_path1": os.path.abspath(path1),
                    "image_path2": os.path.abspath(path2),
                    "aspect": aspect
                }, timeout=180)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        safe_print("[bold green]Comparison Result:[/]")
                        safe_print(data.get("analysis", "No analysis returned"))
                        safe_print(f"\n[dim]Provider: {data.get('provider', 'unknown')} | Model: {data.get('model', 'unknown')}[/]")
                    else:
                        safe_print(f"[red]Error: {data.get('error', 'Unknown error')}[/]")
                else:
                    safe_print(f"[red]Vision service returned {resp.status_code}[/]")
            except Exception as e:
                safe_print(f"[red]Error: {e}[/]")
            return

        # Regular analysis
        path = args[0]

        # Handle "last" keyword
        if path.lower() == "last":
            try:
                list_of_files = glob.glob('generated_images/*')
                if not list_of_files:
                    safe_print("[red]No generated images found.[/]")
                    return
                path = max(list_of_files, key=os.path.getctime)
                safe_print(f"[dim]Using: {path}[/]")
            except Exception as e:
                safe_print(f"[red]Error finding last image: {e}[/]")
                return

        if not os.path.exists(path):
            safe_print(f"[red]File not found: {path}[/]")
            return

        # Determine analysis type and custom prompt
        analysis_type = "describe"
        custom_prompt = ""

        if len(args) > 1:
            second_arg = args[1].lower()
            if second_arg in ["describe", "detailed", "ocr", "objects", "style", "emotions"]:
                analysis_type = second_arg
            else:
                # Treat as custom question
                custom_prompt = " ".join(args[1:])
                analysis_type = "describe"  # Will use question endpoint if custom_prompt

        safe_print(f"[dim]Analyzing image ({analysis_type})...[/]")

        try:
            if custom_prompt:
                # Use question endpoint
                resp = requests.post("http://localhost:8084/question", json={
                    "image_path": os.path.abspath(path),
                    "question": custom_prompt
                }, timeout=120)
            else:
                resp = requests.post("http://localhost:8084/analyze", json={
                    "image_path": os.path.abspath(path),
                    "analysis_type": analysis_type,
                    "custom_prompt": ""
                }, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    safe_print("\n[bold green]Analysis Result:[/]")
                    safe_print(data.get("analysis", "No analysis returned"))
                    safe_print(f"\n[dim]Provider: {data.get('provider', 'unknown')} | Model: {data.get('model', 'unknown')} | {data.get('duration_ms', 0)}ms[/]")
                else:
                    safe_print(f"[red]Error: {data.get('error', 'Unknown error')}[/]")
            else:
                safe_print(f"[red]Vision service returned {resp.status_code}[/]")
        except requests.exceptions.ConnectionError:
            safe_print("[red][FAIL] AitherVision is not running[/]")
            safe_print("[dim]Start it with: Start-AitherOS -Service Vision[/]")
        except Exception as e:
            safe_print(f"[red]Error: {e}[/]")
        return

    else:
        safe_print(f"[warning]Unknown command: {cmd}. Type /help for available commands.[/]")
