import os
import sys
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich import box
from io import StringIO
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console as RichConsole
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle

# Detect PowerShell 5 (conhost) vs modern terminal
# PS5/conhost has issues with certain ANSI sequences and bright colors
def _is_legacy_console() -> bool:
    """Check if running in legacy Windows console (PowerShell 5 / conhost)."""
    if sys.platform != "win32":
        return False
    # Check for Windows Terminal or modern console
    if os.environ.get("WT_SESSION"):  # Windows Terminal
        return False
    if os.environ.get("TERM_PROGRAM"):  # VSCode, etc.
        return False
    # Check PSVersion - PS5 uses conhost, PS7 uses modern console
    ps_version = os.environ.get("PSVersionTable", "")
    if "5." in ps_version:
        return True
    # Fallback: assume legacy if no indicators of modern terminal
    return os.environ.get("TERM") is None and os.environ.get("COLORTERM") is None

IS_LEGACY_CONSOLE = _is_legacy_console()

# Custom Theme - On-brand cyan/pastel pink
# Use yellow instead of bright_red for errors on legacy consoles (PS5 shows red as garbled)
custom_theme = Theme({
    "info": "cyan",
    "warning": "#ffb3d9" if not IS_LEGACY_CONSOLE else "#ffb3d9",
    "danger": "yellow" if IS_LEGACY_CONSOLE else "bright_red",
    "error": "yellow" if IS_LEGACY_CONSOLE else "red",
    "user": "bright_cyan" if not IS_LEGACY_CONSOLE else "cyan",
    "agent": "#ffb3d9" if not IS_LEGACY_CONSOLE else "#ff99cc",
    "tool": "cyan"
})

# Force terminal to ensure colors work even with patch_stdout
console = Console(theme=custom_theme, force_terminal=True)

def safe_print(renderable=""):
    """Prints a Rich renderable using prompt_toolkit's ANSI handling to avoid garbage output."""
    sio = StringIO()
    # force_terminal=True ensures ANSI codes are generated
    c = RichConsole(file=sio, force_terminal=True, theme=custom_theme)
    c.print(renderable)
    output = sio.getvalue()
    # On legacy console, strip some problematic sequences
    if IS_LEGACY_CONSOLE:
        # Remove hyperlinks which PS5 can't handle
        import re
        output = re.sub(r'\x1b\]8;;[^\x1b]*\x1b\\', '', output)
    print_formatted_text(ANSI(output))

def print_banner(model_name="gemini-2.0-flash", title_text="Genesis", subtitle_text="AitherOS CLI", version="v1.2", system_info=None):
    """Print startup banner with system info."""
    
    # Title
    title = Text()
    title.append(title_text, style="bold cyan")
    title.append("\n", style="")
    title.append(subtitle_text, style="dim")
    
    # Model
    title.append("\n\n", style="")
    title.append("Powered by ", style="dim")
    title.append(model_name, style="bold #4285F4")
    
    # OS Info
    import platform
    title.append(f"\nOS: {platform.system()} {platform.release()} | Python: {platform.python_version()}", style="dim grey50")
    
    # System info
    if system_info:
        title.append("\n", style="")
        for key, value in system_info.items():
            # Strip Rich markup for cleaner display
            val_str = str(value).replace("[green]", "").replace("[red]", "").replace("[/]", "").replace("[/green]", "").replace("[/red]", "")
            title.append(f"\n{key}: ", style="bold cyan")
            title.append(val_str, style="dim cyan")
    
    panel = Panel(
        title,
        subtitle=f"[dim]{subtitle_text} {version} | Model: {model_name}[/]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2)
    )
    safe_print(panel)
    safe_print()

def get_toolbar_style():
    """Dark, minimal toolbar style - no ugly white."""
    return PromptStyle.from_dict({
        'bottom-toolbar': '#888888 bg:#1a1a1a',
        'bottom-toolbar.text': '#666666',
    })

def create_keybindings():
    """
    Create keybindings for AitherOS CLI.
    
    IMPORTANT: Avoid conflicts with VSCode terminal shortcuts!
    - F1-F5 are used by VSCode (command palette, rename, find, debug)
    - Ctrl+S/P/G/L/I/M/U are used by VSCode
    - F6-F9, F12 are generally safe
    - Alt+key (Escape+key) combinations are safe
    """
    kb = KeyBindings()

    @kb.add('c-j') # Ctrl+Enter / Ctrl+J
    @kb.add('escape', 'enter') # Alt+Enter / Esc+Enter
    def _(event):
        """Accept input (Submit)."""
        event.current_buffer.validate_and_handle()

    @kb.add('enter')
    def _(event):
        """Insert newline."""
        event.current_buffer.insert_text('\n')

    @kb.add('escape', 'backspace')
    def _(event):
        """Exit the application."""
        event.app.exit(result="/exit")
    
    # ============================================
    # SAFE F-KEY SHORTCUTS
    # ============================================
    
    @kb.add('f6')
    def _(event):
        """F6: Show help."""
        event.current_buffer.text = '/help'
        event.current_buffer.validate_and_handle()
    
    @kb.add('f7')
    def _(event):
        """F7: Show inbox."""
        event.current_buffer.text = '/inbox'
        event.current_buffer.validate_and_handle()
    
    @kb.add('f8')
    def _(event):
        """F8: Cycle safety mode."""
        event.current_buffer.text = '/mode cycle'
        event.current_buffer.validate_and_handle()
    
    @kb.add('f9')
    def _(event):
        """F9: VRAM/GPU status."""
        event.current_buffer.text = '/vram'
        event.current_buffer.validate_and_handle()
    
    @kb.add('f10')
    def _(event):
        """F10: Settings."""
        event.current_buffer.text = '/settings'
        event.current_buffer.validate_and_handle()
    
    @kb.add('f12')
    def _(event):
        """F12: Exit gracefully."""
        event.current_buffer.text = '/exit'
        event.current_buffer.validate_and_handle()
    
    # ============================================
    # ALT+KEY SHORTCUTS (Escape+key - VSCode safe)
    # These are the primary shortcuts to use
    # ============================================
    
    @kb.add('escape', 'h')
    def _(event):
        """Alt+H: Help."""
        event.current_buffer.text = '/help'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'i')
    def _(event):
        """Alt+I: Inbox."""
        event.current_buffer.text = '/inbox'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', '1')
    def _(event):
        """Alt+1: Read latest inbox message."""
        event.current_buffer.text = '/inbox 1'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'm')
    def _(event):
        """Alt+M: Mode/Safety cycle."""
        event.current_buffer.text = '/mode cycle'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 's')
    def _(event):
        """Alt+S: Safety mode status."""
        event.current_buffer.text = '/safety'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'v')
    def _(event):
        """Alt+V: VRAM status."""
        event.current_buffer.text = '/vram'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'k')
    def _(event):
        """Alt+K: Show keyboard shortcuts."""
        event.current_buffer.text = '/keys'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'c')
    def _(event):
        """Alt+C: Clear screen."""
        event.current_buffer.text = '/clear'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'u')
    def _(event):
        """Alt+U: Add :: override prefix."""
        current = event.current_buffer.text
        event.current_buffer.text = ':: ' + current
        event.current_buffer.cursor_position = len(event.current_buffer.text)
    
    @kb.add('escape', 'g')
    def _(event):
        """Alt+G: Start image generation prompt."""
        event.current_buffer.text = 'generate a pic of '
    
    @kb.add('escape', 'l')
    def _(event):
        """Alt+L: Show loaded models."""
        event.current_buffer.text = '/models'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'x')
    def _(event):
        """Alt+X: Unload all models."""
        event.current_buffer.text = '/unload all'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'a')
    def _(event):
        """Alt+A: Persona info for Aither."""
        event.current_buffer.text = '/persona-info aither'
        event.current_buffer.validate_and_handle()
    
    @kb.add('escape', 'e')
    def _(event):
        """Alt+E: Show cost/expense."""
        event.current_buffer.text = '/cost'
        event.current_buffer.validate_and_handle()

    return kb
