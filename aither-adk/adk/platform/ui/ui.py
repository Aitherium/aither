"""
AitherOS UI Components - Interactive toolbar with keyboard shortcuts
"""
import datetime
import logging

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.text import Text

from aither_adk.infrastructure.status_monitor import monitor

logger = logging.getLogger(__name__)

# Toolbar state for expansion toggle
TOOLBAR_STATE = {"expanded": False}

# Store current model name globally so toolbar can access it
CURRENT_MODEL = {"name": "aither-orchestrator-8b-v4", "is_local": True}

# Toolbar style - dark theme, improved readability
TOOLBAR_STYLE = Style.from_dict({
    'bottom-toolbar': 'bg:#1a1a1a #aaaaaa',  # Lighter text for better readability
    'bottom-toolbar.text': '#aaaaaa',  # Lighter gray instead of dark gray
})

def get_toolbar_style():
    """Returns the toolbar style for prompt_toolkit."""
    return TOOLBAR_STYLE

def toggle_toolbar_expansion():
    """Toggle between compact and expanded toolbar views."""
    TOOLBAR_STATE["expanded"] = not TOOLBAR_STATE["expanded"]
    return TOOLBAR_STATE["expanded"]

def update_current_model(model_name: str):
    """Update the current model displayed in toolbar."""
    CURRENT_MODEL["name"] = model_name
    # Check if local
    CURRENT_MODEL["is_local"] = not model_name.startswith(('gemini', 'gpt', 'claude'))

def get_vram_display():
    """Get compact VRAM display string."""
    try:
        vram = monitor.get_vram_usage()
        if vram and vram.get('total', 0) > 0:
            used_gb = vram['used'] / 1024
            total_gb = vram['total'] / 1024
            pct = (vram['used'] / vram['total']) * 100
            # Color code: green < 50%, yellow 50-80%, red > 80%
            if pct < 50:
                return f"<style fg='#00ff00'>{used_gb:.1f}/{total_gb:.0f}GB</style>"
            elif pct < 80:
                return f"<style fg='#ffff00'>{used_gb:.1f}/{total_gb:.0f}GB</style>"
            else:
                return f"<style fg='#ff6666'>{used_gb:.1f}/{total_gb:.0f}GB</style>"
        return "<style fg='#666666'>No GPU</style>"
    except (KeyError, TypeError, ZeroDivisionError):
        return ""

def create_toolbar_callback(session_stats, task_manager_provider, group_chat_manager,
                           safety_level_provider, internet_enabled=False, clipboard_enabled=False,
                           vision_status_provider=None, mailbox_provider=None,
                           comfy_status_provider=None, ollama_status_provider=None, model_name_provider=None):
    """
    Creates an interactive toolbar with useful real-time info.
    Shows: Model | VRAM | Mode | Mail | Time | Shortcuts
    """
    def get_toolbar():
        parts = []

        # 1. Current Model (most important!)
        model = CURRENT_MODEL["name"]
        is_local = CURRENT_MODEL["is_local"]
        # Clean up model name for display
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "")
        display_model = model[:20]  # Truncate long names
        if is_local:
            parts.append(f"<style fg='#ff99cc'> {display_model}</style>")
        else:
            parts.append(f"<style fg='#00ffff'>[CLOUD] {display_model}</style>")

        # 2. VRAM Usage
        vram = get_vram_display()
        if vram:
            parts.append(f"<style fg='#888888'>GPU:</style>{vram}")

        # 3. Safety/Mode (clear visual)
        level = safety_level_provider()
        if level == "HIGH":
            parts.append("<style fg='#00ff00' bg='#003300'> PRO </style>")
        elif level == "MEDIUM":
            parts.append("<style fg='#ffff00' bg='#333300'> CAS </style>")
        elif level == "LOW" or level == "UNRESTRICTED":
            parts.append("<style fg='#ff6666' bg='#330000'> RAW </style>")
        else:
            parts.append(f"<style fg='#888888'>{level[:3]}</style>")

        # 4. Unread Mail (only if > 0)
        if mailbox_provider:
            try:
                mailbox = mailbox_provider()
                if mailbox:
                    unread = mailbox.get_unread_count("user")
                    if unread > 0:
                        parts.append(f"<style fg='#ffff00'> {unread}</style>")
            except Exception as exc:
                logger.debug(f"Mail unread count check failed: {exc}")

        # 5. Services (compact icons)
        services = []
        if ollama_status_provider:
            try:
                o = ollama_status_provider()
                if o and "Running" in str(o):
                    services.append("<style fg='#00ff00'>*</style>")
                else:
                    services.append("<style fg='#ff6666'>o</style>")
            except Exception as exc:
                logger.debug(f"Ollama status check failed: {exc}")
                services.append("<style fg='#666666'>o</style>")
        if comfy_status_provider:
            try:
                c = comfy_status_provider()
                if c and "Connected" in str(c):
                    services.append("<style fg='#00ff00'>[ART]</style>")
            except Exception as exc:
                logger.debug(f"ComfyUI status check failed: {exc}")
        if services:
            parts.append("".join(services))

        # 6. Time
        now = datetime.datetime.now().strftime("%H:%M")
        parts.append(f"<style fg='#666666'>{now}</style>")

        # 7. Quick shortcuts hint (compact)
        parts.append("<style fg='#555555'>F8:Mode F9:VRAM /help</style>")

        return HTML(" | ".join(parts))

    return get_toolbar


def create_rich_toolbar_callback(session_stats, task_manager_provider, group_chat_manager,
                                 safety_level_provider, internet_enabled=False, clipboard_enabled=False,
                                 vision_status_provider=None, mailbox_provider=None,
                                 comfy_status_provider=None, ollama_status_provider=None):
    """
    Creates a Rich Text toolbar for the spinner/processing display.
    """
    def get_toolbar():
        text = Text()

        # Model
        model = CURRENT_MODEL["name"]
        is_local = CURRENT_MODEL["is_local"]
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "")
        display_model = model[:20]

        if is_local:
            text.append("  ", style="bold #ffb3d9")
            text.append(display_model, style="#ffb3d9")
        else:
            text.append(" [CLOUD] ", style="bold cyan")
            text.append(display_model, style="cyan")
        text.append(" |", style="dim")

        # VRAM
        try:
            vram = monitor.get_vram_usage()
            if vram and vram.get('total', 0) > 0:
                used_gb = vram['used'] / 1024
                total_gb = vram['total'] / 1024
                pct = (vram['used'] / vram['total']) * 100
                if pct < 50:
                    text.append(f" {used_gb:.1f}/{total_gb:.0f}GB ", style="green")
                elif pct < 80:
                    text.append(f" {used_gb:.1f}/{total_gb:.0f}GB ", style="yellow")
                else:
                    text.append(f" {used_gb:.1f}/{total_gb:.0f}GB ", style="red")
                text.append("|", style="dim")
        except Exception as exc:
            logger.debug(f"VRAM usage check failed: {exc}")

        # Safety/Mode with visual distinction
        level = safety_level_provider()
        if level == "HIGH":
            text.append(" PRO ", style="bold green reverse")
        elif level == "MEDIUM":
            text.append(" CAS ", style="bold yellow reverse")
        elif level == "LOW" or level == "UNRESTRICTED":
            text.append(" RAW ", style="bold red reverse")
        else:
            text.append(f" {level[:3]} ", style="bold white reverse")
        text.append("|", style="dim")

        # Mail with emphasis on unread
        if mailbox_provider:
            try:
                mailbox = mailbox_provider()
                if mailbox:
                    unread = mailbox.get_unread_count("user")
                    if unread > 0:
                        text.append(f"  {unread} ", style="bold yellow")
                        text.append("|", style="dim")
            except Exception as exc:
                logger.debug(f"Rich toolbar mail check failed: {exc}")

        # Time
        now = datetime.datetime.now().strftime("%H:%M")
        text.append(f" {now} ", style="dim")

        return text

    return get_toolbar


def create_keybindings_with_shortcuts(handle_f1=None, handle_f2=None, handle_f3=None,
                                      handle_f4=None, handle_f5=None):
    """
    Create key bindings with F-key shortcuts.
    Deprecated: Use console.create_keybindings() instead for full shortcut support.
    """
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add('f1')
    def _(event):
        if handle_f1:
            handle_f1()
        else:
            event.app.current_buffer.text = '/help'
            event.app.current_buffer.validate_and_handle()

    @kb.add('f2')
    def _(event):
        if handle_f2:
            handle_f2()
        else:
            event.app.current_buffer.text = '/mode cycle'
            event.app.current_buffer.validate_and_handle()

    @kb.add('f3')
    def _(event):
        if handle_f3:
            handle_f3()
        else:
            event.app.current_buffer.text = '/inbox'
            event.app.current_buffer.validate_and_handle()

    @kb.add('f4')
    def _(event):
        if handle_f4:
            handle_f4()
        else:
            event.app.current_buffer.text = '/vram'
            event.app.current_buffer.validate_and_handle()

    @kb.add('f5')
    def _(event):
        if handle_f5:
            handle_f5()
        else:
            event.app.current_buffer.text = '/clear'
            event.app.current_buffer.validate_and_handle()

    @kb.add('c-m') # Ctrl+Enter (primary submit)
    @kb.add('c-j') # Ctrl+J (alternative for Ctrl+Enter)
    def _(event):
        event.app.current_buffer.validate_and_handle()

    return kb


def toggle_toolbar():
    """Toggle toolbar expansion state."""
    return toggle_toolbar_expansion()


# Keyboard shortcuts reference (for help command)
# NOTE: Avoid F1-F5 and most Ctrl+key as they conflict with VSCode
KEYBOARD_SHORTCUTS = {
    "F-Keys (Safe)": {
        "F6": "Show models",
        "F7": "Show inbox",
        "F8": "Cycle mode (PRO->CAS->RAW)",
        "F9": "VRAM status",
        "F10": "Settings",
        "F12": "Exit",
    },
    "Alt+Key (Esc then Key)": {
        "Alt+H": "Help",
        "Alt+I": "Inbox",
        "Alt+1": "Read latest message",
        "Alt+M": "Cycle mode (PRO->CAS->RAW)",
        "Alt+S": "Safety status",
        "Alt+V": "VRAM status",
        "Alt+K": "Show shortcuts",
        "Alt+C": "Clear screen",
        "Alt+U": "Add :: override",
        "Alt+G": "Start image prompt",
        "Alt+L": "Show loaded models",
        "Alt+X": "Unload all models",
        "Alt+A": "Aither persona info",
        "Alt+E": "Show cost/expense",
    },
    "Editing": {
        "Enter": "New line",
        "Ctrl+Enter": "Submit",
        "Ctrl+J": "Submit (alternative)",
        "Esc+Backspace": "Exit",
    }
}
