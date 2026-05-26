import os
import sys
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from aither_adk.ui.console import console

def configure_auth():
    """Interactively configures API keys."""
    console.print(Panel("No API Key found in environment variables.", title="[warning]Authentication Missing[/]", border_style="yellow"))

    key = Prompt.ask("Enter your Google GenAI API Key", password=True)
    if not key:
        console.print("[danger]API Key is required to proceed.[/]")
        sys.exit(1)

    os.environ["GOOGLE_API_KEY"] = key

    if Confirm.ask("Save to .env file for future use?"):
        with open(".env", "a") as f:
            f.write(f"\nGOOGLE_API_KEY={key}\n")
        console.print("[green]Saved to .env[/]")
