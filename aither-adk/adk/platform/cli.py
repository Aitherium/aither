"""
Aither ADK CLI
==============

Command-line interface for the Aither Development Kit.

Usage:
    adk info              # Show ADK version and info
    adk check             # Verify ADK installation
    adk services          # Check AitherOS services
    adk models            # List available models

    # Run Agents
    adk run list                      # List available agents
    adk run agent Aither              # Run Aither agent
    adk run agent Saga -l   # Run with local model
    adk run info Aither               # Show agent details

    # Scaffolding
    adk new agent MyAgent -d "Description"
    adk new service MyService -l cognition -p 8142 -d "Description"

    # Genesis (Testing & Boot)
    adk genesis run                   # Run system tests
    adk genesis boot                  # Start AitherOS
    adk genesis shutdown              # Stop AitherOS

    # GCP Deployment
    adk deploy init       # Initialize GCP project
    adk deploy build      # Build containers
    adk deploy run        # Deploy to Cloud Run
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="adk",
    help="Aither Development Kit - CLI tools for agent development and deployment",
    no_args_is_help=True,
)

# Sub-commands
new_app = typer.Typer(help="Create new agents, services, or tools")
deploy_app = typer.Typer(help="Deploy to Google Cloud Platform")
genesis_app = typer.Typer(help="AitherGenesis - System testing and bootloader")
docs_app = typer.Typer(help="Documentation commands")
run_app = typer.Typer(help="Run agents")
dev_app = typer.Typer(help="Development mode - Demiurge interaction (like Claude Code)")

app.add_typer(new_app, name="new")
app.add_typer(deploy_app, name="deploy")
app.add_typer(genesis_app, name="genesis")
app.add_typer(docs_app, name="docs")
app.add_typer(run_app, name="run")
app.add_typer(dev_app, name="dev")

# Register AitherTrainer if available
try:
    _workspace = Path.cwd()
    if (_workspace / "AitherOS" / "AitherTrainer" / "src").exists():
        sys.path.append(str(_workspace / "AitherOS" / "AitherTrainer" / "src"))
    from aither_trainer.cli import app as trainer_app
    app.add_typer(trainer_app, name="train")
except ImportError:
    pass # Should warn, but for brevity just skip

# Register AitherCanvas if available
try:
    if (_workspace / "AitherOS" / "AitherCanvas" / "src").exists():
        sys.path.append(str(_workspace / "AitherOS" / "AitherCanvas" / "src"))
    from aither_canvas.cli import app as canvas_app
    app.add_typer(canvas_app, name="canvas")
except ImportError:
    pass

console = Console()


# =============================================================================
# INFO COMMANDS
# =============================================================================

@app.command()
def info():
    """Show ADK version and package information."""
    from . import __version__, get_info

    info_data = get_info()

    console.print(Panel.fit(
        f"[bold cyan]Aither Development Kit[/bold cyan]\n"
        f"Version: [green]{__version__}[/green]\n"
        f"Author: {info_data['author']}\n"
        f"Email: {info_data['email']}",
        title=" ADK Info",
        border_style="cyan",
    ))

    # Modules table
    table = Table(title="Available Modules", box=box.ROUNDED)
    table.add_column("Module", style="cyan")
    table.add_column("Import Path", style="dim")
    table.add_column("Description")

    modules = [
        ("ai", "aither_adk.ai", "LLM providers, models, safety"),
        ("communication", "aither_adk.communication", "Inter-agent messaging"),
        ("infrastructure", "aither_adk.infrastructure", "Auth, config, services"),
        ("memory", "aither_adk.memory", "Memory management"),
        ("tools", "aither_adk.tools", "Tool loader, MCP integration"),
        ("ui", "aither_adk.ui", "Console, CLI, commands"),
    ]

    for name, path, desc in modules:
        table.add_row(name, path, desc)

    console.print(table)


@app.command()
def check():
    """Verify ADK installation and imports."""
    console.print("[bold][SEARCH] Checking ADK Installation...[/bold]\n")

    checks = []

    # Check main package
    try:
        from . import __version__
        checks.append(("aither_adk", True, __version__))
    except ImportError as e:
        checks.append(("aither_adk", False, str(e)))

    # Check submodules - use direct submodule imports
    submodules = [
        ("aither_adk.ai", "from aither_adk.ai.ollama import OllamaLlm"),
        ("aither_adk.communication", "from aither_adk.communication.mailbox import Mailbox"),
        ("aither_adk.infrastructure", "from aither_adk.infrastructure.utils import configure_logging"),
        ("aither_adk.memory", "from aither_adk.memory.memory import MemoryManager"),
        ("aither_adk.tools", "from aither_adk.tools.tool_loader import aither_tools"),
        ("aither_adk.ui", "from aither_adk.ui.console import console"),
    ]

    for module, test_import in submodules:
        try:
            exec(test_import)
            checks.append((module, True, "[OK]"))
        except ImportError as e:
            checks.append((module, False, str(e)[:50]))
        except Exception as e:
            checks.append((module, False, f"Error: {str(e)[:40]}"))

    # Display results
    table = Table(box=box.ROUNDED)
    table.add_column("Module", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    all_ok = True
    for module, ok, detail in checks:
        status = "[green][OK] OK[/green]" if ok else "[red][X] FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(module, status, detail)

    console.print(table)

    if all_ok:
        console.print("\n[bold green][DONE] All checks passed![/bold green]")
    else:
        console.print(
            "\n[bold red][FAIL] Some checks failed.[/bold red] "
            "Run `pip install -e AitherOS/packages/aither_adk` to install. "
            "For new standalone agent projects, see the `AitherZero/library/templates/adk-agent` template."
        )
        raise typer.Exit(1)


@app.command()
def services():
    """Check AitherOS service status."""
    console.print("[bold][SIGNAL] Checking AitherOS Services...[/bold]\n")

    try:
        from .infrastructure.services import get_service_status

        status = get_service_status()

        table = Table(title="Service Status", box=box.ROUNDED)
        table.add_column("Service", style="cyan")
        table.add_column("Port", style="dim")
        table.add_column("Status")

        services_list = [
            ("AitherNode", 8080, status.aithernode),
            ("Ollama", 11434, status.ollama),
            ("ComfyUI", 8188, status.comfyui),
        ]

        for name, port, running in services_list:
            status_str = "[green]* Running[/green]" if running else "[red]o Stopped[/red]"
            table.add_row(name, str(port), status_str)

        console.print(table)

        if status.aithernode:
            console.print("\n[green][OK][/green] AitherNode is running - agents will use fast HTTP path")
        else:
            console.print("\n[yellow][WARN][/yellow] AitherNode not running - agents will use direct imports (slower)")

    except Exception as e:
        console.print(f"[red]Error checking services: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models():
    """List available AI models."""
    console.print("[bold] Available Models...[/bold]\n")

    try:
        from .ai.models import get_available_models

        available = get_available_models()

        if not available:
            console.print("[yellow]No models found. Is Ollama running?[/yellow]")
            return

        table = Table(title="Available Models", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="dim")
        table.add_column("Type")

        for model in available:
            if model.startswith("ollama/"):
                table.add_row(model, "Ollama", "Local")
            elif model.startswith("gemini"):
                table.add_row(model, "Google", "Cloud")
            elif model.startswith("gpt"):
                table.add_row(model, "OpenAI", "Cloud")
            elif model.startswith("claude"):
                table.add_row(model, "Anthropic", "Cloud")
            else:
                table.add_row(model, "Unknown", "?")

        console.print(table)
        console.print(f"\n[dim]Total: {len(available)} models[/dim]")

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


@app.command()
def version():
    """Show ADK version."""
    from . import __version__
    console.print(f"aither-adk {__version__}")


# =============================================================================
# NEW COMMANDS - Scaffolding
# =============================================================================

@new_app.command("agent")
def new_agent(
    name: str = typer.Argument(..., help="Agent name in PascalCase (e.g., DataAnalyst)"),
    description: str = typer.Option(
        ..., "--description", "-d",
        help="Brief description of what the agent does"
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p",
        help="Service port (auto-assigned if not specified)"
    ),
    specialties: Optional[str] = typer.Option(
        None, "--specialties", "-s",
        help="Comma-separated list of specialties"
    ),
    author: Optional[str] = typer.Option(
        None, "--author", "-a",
        help="Author name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing files"
    ),
):
    """
    Create a new AitherOS agent.

    Example:
        adk new agent DataAnalyst -d "Analyzes data and generates insights"
    """
    try:
        from aither_toolkit.core.agent import AgentScaffolder
        from aither_toolkit.core.config import get_config

        config = get_config()

        kwargs = {
            "name": name,
            "description": description,
            "config": config,
        }

        if port:
            kwargs["port"] = port
        if specialties:
            kwargs["specialties"] = specialties
        if author:
            kwargs["author"] = author

        scaffolder = AgentScaffolder(**kwargs)
        agent_dir = scaffolder.create(force=force)

        console.print(f"\n[green][OK][/] Agent created at: {agent_dir}")
        console.print("\n[dim]Next steps:[/]")
        console.print(f"  cd {agent_dir}")
        console.print("  pip install -r requirements.txt")
        console.print("  python agent.py")

    except ImportError:
        console.print("[red]Error:[/] aither-toolkit not installed. Run: pip install -e AitherOS/packages/aither_toolkit")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@new_app.command("service")
def new_service(
    name: str = typer.Argument(..., help="Service name in PascalCase (e.g., MyProcessor)"),
    layer: str = typer.Option(
        ..., "--layer", "-l",
        help="Service layer (cognition, memory, perception, core, gpu, agents, security, training, automation, infrastructure, mesh)"
    ),
    port: int = typer.Option(
        ..., "--port", "-p",
        help="Service port number"
    ),
    description: str = typer.Option(
        ..., "--description", "-d",
        help="Brief description of what the service does"
    ),
    depends_on: Optional[str] = typer.Option(
        None, "--depends-on",
        help="Comma-separated list of dependencies"
    ),
    author: Optional[str] = typer.Option(
        None, "--author", "-a",
        help="Author name"
    ),
    no_client: bool = typer.Option(
        False, "--no-client",
        help="Skip creating the client library"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing files"
    ),
):
    """
    Create a new AitherOS service.

    Example:
        adk new service DataProcessor -l cognition -p 8142 -d "Processes data"
    """
    try:
        from aither_toolkit.core.config import get_config
        from aither_toolkit.core.service import ServiceScaffolder

        config = get_config()

        deps = None
        if depends_on:
            deps = [d.strip() for d in depends_on.split(",")]

        scaffolder = ServiceScaffolder(
            name=name,
            layer=layer,
            port=port,
            description=description,
            depends_on=deps,
            author=author,
            create_client=not no_client,
            config=config,
        )

        service_file = scaffolder.create(force=force)
        console.print(f"\n[green][OK][/] Service created at: {service_file}")

    except ImportError:
        console.print("[red]Error:[/] aither-toolkit not installed. Run: pip install -e AitherOS/packages/aither_toolkit")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@new_app.command("tool")
def new_tool(
    name: str = typer.Argument(..., help="Tool name in snake_case (e.g., image_processor)"),
    agent: str = typer.Option(
        ..., "--agent", "-a",
        help="Agent to add the tool to"
    ),
    description: str = typer.Option(
        "Custom tool functions", "--description", "-d",
        help="Tool description"
    ),
    functions: Optional[str] = typer.Option(
        None, "--functions", "-f",
        help="Comma-separated function names"
    ),
):
    """
    Create a new tool module for an agent.

    Example:
        adk new tool image_processor -a Saga -f "process_image,resize"
    """
    try:
        from aither_toolkit.core.tool import ToolScaffolder

        func_list = None
        if functions:
            func_list = [f.strip() for f in functions.split(",")]

        scaffolder = ToolScaffolder(
            name=name,
            agent=agent,
            description=description,
            functions=func_list,
        )

        result = scaffolder.create()

        if result["success"]:
            console.print(f"[green][OK][/] {result['message']}")
            for f in result["files"]:
                console.print(f"  Created: {f}")
        else:
            console.print(f"[red]Error:[/] {result['message']}")
            raise typer.Exit(1)

    except ImportError:
        console.print("[red]Error:[/] aither-toolkit not installed. Run: pip install -e AitherOS/packages/aither_toolkit")
        raise typer.Exit(1)


# =============================================================================
# DEPLOY COMMANDS - GCP Deployment
# =============================================================================

def _find_workspace() -> Path:
    """Find the AitherZero workspace root."""
    cwd = Path.cwd()

    # Check if we're already in the workspace
    if (cwd / "AitherOS").exists():
        return cwd

    # Check parent directories
    for parent in cwd.parents:
        if (parent / "AitherOS").exists():
            return parent

    return cwd


def _run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    """Run a command with progress display."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description, total=None)

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                progress.stop()
                console.print(f"[red]Error:[/] {result.stderr}")
                return False

            return True

        except FileNotFoundError:
            progress.stop()
            console.print(f"[red]Error:[/] Command not found: {cmd[0]}")
            return False


@deploy_app.command("init")
def deploy_init(
    project_id: str = typer.Option(
        None, "--project", "-p",
        help="GCP Project ID (will prompt if not provided)"
    ),
    region: str = typer.Option(
        "us-central1", "--region", "-r",
        help="GCP Region"
    ),
    service_account: Optional[str] = typer.Option(
        None, "--service-account", "-s",
        help="Service account email"
    ),
):
    """
    Initialize GCP project for AitherOS deployment.

    This command will:
    - Verify gcloud CLI is installed
    - Set up the GCP project
    - Enable required APIs
    - Create Artifact Registry repository
    - Generate cloudbuild.yaml

    Example:
        adk deploy init --project my-aither-project --region us-central1
    """
    console.print(Panel.fit(
        "[bold cyan]GCP Deployment Initialization[/bold cyan]",
        title="[LAUNCH] AitherOS Deploy",
        border_style="cyan",
    ))

    workspace = _find_workspace()

    # Check gcloud is installed
    console.print("\n[bold]Step 1:[/] Checking gcloud CLI...")
    try:
        result = subprocess.run(["gcloud", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError()
        console.print("[green][OK][/] gcloud CLI found")
    except FileNotFoundError:
        console.print("[red][X][/] gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install")
        raise typer.Exit(1)

    # Get project ID
    if not project_id:
        project_id = typer.prompt("Enter GCP Project ID")

    console.print(f"\n[bold]Step 2:[/] Setting project to [cyan]{project_id}[/]...")
    if not _run_command(["gcloud", "config", "set", "project", project_id], "Setting project..."):
        raise typer.Exit(1)
    console.print(f"[green][OK][/] Project set to {project_id}")

    # Enable APIs
    console.print("\n[bold]Step 3:[/] Enabling required GCP APIs...")
    apis = [
        "cloudbuild.googleapis.com",
        "run.googleapis.com",
        "artifactregistry.googleapis.com",
        "secretmanager.googleapis.com",
    ]

    for api in apis:
        if _run_command(["gcloud", "services", "enable", api], f"Enabling {api}..."):
            console.print(f"[green][OK][/] {api} enabled")
        else:
            console.print(f"[yellow]![/] Failed to enable {api} (may already be enabled)")

    # Create Artifact Registry repository
    console.print("\n[bold]Step 4:[/] Creating Artifact Registry repository...")
    repo_result = subprocess.run(
        ["gcloud", "artifacts", "repositories", "create", "aitheros",
         "--repository-format=docker", f"--location={region}",
         "--description=AitherOS container images"],
        capture_output=True,
        text=True,
    )
    if repo_result.returncode == 0:
        console.print("[green][OK][/] Artifact Registry repository 'aitheros' created")
    else:
        if "already exists" in repo_result.stderr.lower():
            console.print("[yellow]![/] Repository 'aitheros' already exists")
        else:
            console.print(f"[red]Error:[/] {repo_result.stderr}")

    # Generate cloudbuild.yaml
    console.print("\n[bold]Step 5:[/] Generating cloudbuild.yaml...")
    cloudbuild_config = _generate_cloudbuild_yaml(project_id, region)

    cloudbuild_path = workspace / "cloudbuild.yaml"
    with open(cloudbuild_path, "w") as f:
        f.write(cloudbuild_config)
    console.print(f"[green][OK][/] Generated {cloudbuild_path}")

    # Generate deployment config
    deploy_config = {
        "project_id": project_id,
        "region": region,
        "service_account": service_account,
        "created_at": datetime.now().isoformat(),
        "artifact_registry": f"{region}-docker.pkg.dev/{project_id}/aitheros",
    }

    config_path = workspace / ".aither-deploy.json"
    with open(config_path, "w") as f:
        json.dump(deploy_config, f, indent=2)
    console.print(f"[green][OK][/] Generated {config_path}")

    # Summary
    console.print(Panel.fit(
        f"[bold green][DONE] GCP Project Initialized![/]\n\n"
        f"Project: [cyan]{project_id}[/]\n"
        f"Region: [cyan]{region}[/]\n"
        f"Registry: [cyan]{region}-docker.pkg.dev/{project_id}/aitheros[/]\n\n"
        f"[dim]Next steps:[/]\n"
        f"  adk deploy build    # Build container images\n"
        f"  adk deploy run      # Deploy to Cloud Run",
        title="Summary",
        border_style="green",
    ))


@deploy_app.command("build")
def deploy_build(
    service: Optional[str] = typer.Option(
        None, "--service", "-s",
        help="Specific service to build (builds all if not specified)"
    ),
    push: bool = typer.Option(
        True, "--push/--no-push",
        help="Push to Artifact Registry after building"
    ),
    tag: str = typer.Option(
        "latest", "--tag", "-t",
        help="Image tag"
    ),
):
    """
    Build container images for AitherOS services.

    Example:
        adk deploy build                    # Build all services
        adk deploy build -s AitherNode      # Build specific service
        adk deploy build --tag v1.0.0       # Build with version tag
    """
    workspace = _find_workspace()

    # Load deploy config
    config_path = workspace / ".aither-deploy.json"
    if not config_path.exists():
        console.print("[red]Error:[/] Deployment not initialized. Run: adk deploy init")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = json.load(f)

    config["project_id"]
    region = config["region"]
    registry = config["artifact_registry"]

    console.print(Panel.fit(
        f"[bold cyan]Building Container Images[/]\n"
        f"Registry: {registry}",
        title=" Build",
        border_style="cyan",
    ))

    # Services to build
    services_to_build = []

    if service:
        services_to_build = [service]
    else:
        # Auto-discover services with Dockerfiles
        dockerfile_paths = [
            ("AitherNode", workspace / "AitherOS" / "Dockerfile"),
            ("AitherVeil", workspace / "AitherOS" / "AitherNode" / "AitherVeil" / "Dockerfile"),
        ]

        for svc_name, dockerfile in dockerfile_paths:
            if dockerfile.exists():
                services_to_build.append(svc_name)

    if not services_to_build:
        console.print("[yellow]No services found to build.[/]")
        return

    console.print(f"\n[bold]Services to build:[/] {', '.join(services_to_build)}\n")

    for svc in services_to_build:
        image_name = f"{registry}/{svc.lower()}:{tag}"

        console.print(f"\n[bold]Building {svc}...[/]")

        # Determine Dockerfile path
        if svc == "AitherNode":
            dockerfile_dir = workspace / "AitherOS"
        elif svc == "AitherVeil":
            dockerfile_dir = workspace / "AitherOS" / "AitherNode" / "AitherVeil"
        else:
            dockerfile_dir = workspace

        # Build
        build_cmd = ["docker", "build", "-t", image_name, "."]

        if _run_command(build_cmd, f"Building {svc}...", cwd=dockerfile_dir):
            console.print(f"[green][OK][/] Built: {image_name}")

            if push:
                # Configure docker for artifact registry
                _run_command(
                    ["gcloud", "auth", "configure-docker", f"{region}-docker.pkg.dev", "--quiet"],
                    "Configuring Docker..."
                )

                # Push
                if _run_command(["docker", "push", image_name], f"Pushing {svc}..."):
                    console.print(f"[green][OK][/] Pushed: {image_name}")
                else:
                    console.print(f"[red][X][/] Failed to push {svc}")
        else:
            console.print(f"[red][X][/] Failed to build {svc}")


@deploy_app.command("run")
def deploy_run(
    service: str = typer.Option(
        ..., "--service", "-s",
        help="Service to deploy"
    ),
    memory: str = typer.Option(
        "2Gi", "--memory", "-m",
        help="Memory allocation (e.g., 512Mi, 2Gi)"
    ),
    cpu: str = typer.Option(
        "1", "--cpu", "-c",
        help="CPU allocation (e.g., 1, 2)"
    ),
    min_instances: int = typer.Option(
        0, "--min-instances",
        help="Minimum instances (0 = scale to zero)"
    ),
    max_instances: int = typer.Option(
        10, "--max-instances",
        help="Maximum instances"
    ),
    port: int = typer.Option(
        8080, "--port", "-p",
        help="Container port"
    ),
    env_vars: Optional[str] = typer.Option(
        None, "--env", "-e",
        help="Environment variables (KEY=VALUE,KEY2=VALUE2)"
    ),
    allow_unauthenticated: bool = typer.Option(
        False, "--public",
        help="Allow unauthenticated access"
    ),
    tag: str = typer.Option(
        "latest", "--tag", "-t",
        help="Image tag to deploy"
    ),
):
    """
    Deploy a service to Google Cloud Run.

    Example:
        adk deploy run -s AitherNode --memory 4Gi --cpu 2 --public
        adk deploy run -s AitherVeil -e "NODE_ENV=production"
    """
    workspace = _find_workspace()

    # Load deploy config
    config_path = workspace / ".aither-deploy.json"
    if not config_path.exists():
        console.print("[red]Error:[/] Deployment not initialized. Run: adk deploy init")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = json.load(f)

    config["project_id"]
    region = config["region"]
    registry = config["artifact_registry"]

    image_name = f"{registry}/{service.lower()}:{tag}"
    service_name = f"aither-{service.lower()}"

    console.print(Panel.fit(
        f"[bold cyan]Deploying to Cloud Run[/]\n"
        f"Service: {service_name}\n"
        f"Image: {image_name}\n"
        f"Region: {region}",
        title="[LAUNCH] Deploy",
        border_style="cyan",
    ))

    # Build deploy command
    cmd = [
        "gcloud", "run", "deploy", service_name,
        f"--image={image_name}",
        f"--region={region}",
        f"--memory={memory}",
        f"--cpu={cpu}",
        f"--min-instances={min_instances}",
        f"--max-instances={max_instances}",
        f"--port={port}",
        "--platform=managed",
        "--quiet",
    ]

    if allow_unauthenticated:
        cmd.append("--allow-unauthenticated")

    if env_vars:
        cmd.append(f"--set-env-vars={env_vars}")

    console.print("\n[bold]Deploying...[/]")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Extract service URL from output
        url_match = None
        for line in result.stdout.split("\n"):
            if "Service URL:" in line or "https://" in line:
                url_match = line.strip()
                break

        console.print(Panel.fit(
            f"[bold green][DONE] Deployment Successful![/]\n\n"
            f"Service: [cyan]{service_name}[/]\n"
            f"Region: [cyan]{region}[/]\n"
            f"{f'URL: [cyan]{url_match}[/]' if url_match else ''}",
            title="Success",
            border_style="green",
        ))
    else:
        console.print(f"[red]Error:[/] {result.stderr}")
        raise typer.Exit(1)


@deploy_app.command("status")
def deploy_status():
    """Show deployment status for all services."""
    workspace = _find_workspace()

    # Load deploy config
    config_path = workspace / ".aither-deploy.json"
    if not config_path.exists():
        console.print("[red]Error:[/] Deployment not initialized. Run: adk deploy init")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = json.load(f)

    region = config["region"]

    console.print("[bold][CHART] Cloud Run Services Status[/]\n")

    result = subprocess.run(
        ["gcloud", "run", "services", "list", f"--region={region}", "--format=json"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        services = json.loads(result.stdout) if result.stdout else []

        if not services:
            console.print("[yellow]No services deployed yet.[/]")
            return

        table = Table(box=box.ROUNDED)
        table.add_column("Service", style="cyan")
        table.add_column("Region")
        table.add_column("URL")
        table.add_column("Status")

        for svc in services:
            name = svc.get("metadata", {}).get("name", "unknown")
            url = svc.get("status", {}).get("url", "-")
            conditions = svc.get("status", {}).get("conditions", [])
            status = "Unknown"
            for cond in conditions:
                if cond.get("type") == "Ready":
                    status = "[green]Ready[/]" if cond.get("status") == "True" else "[red]Not Ready[/]"
                    break

            table.add_row(name, region, url[:50] + "..." if len(url) > 50 else url, status)

        console.print(table)
    else:
        console.print(f"[red]Error:[/] {result.stderr}")


@deploy_app.command("logs")
def deploy_logs(
    service: str = typer.Option(
        ..., "--service", "-s",
        help="Service to view logs for"
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f",
        help="Stream logs"
    ),
    limit: int = typer.Option(
        50, "--limit", "-n",
        help="Number of log entries"
    ),
):
    """
    View logs for a deployed service.

    Example:
        adk deploy logs -s AitherNode -f
    """
    workspace = _find_workspace()

    config_path = workspace / ".aither-deploy.json"
    if not config_path.exists():
        console.print("[red]Error:[/] Deployment not initialized. Run: adk deploy init")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = json.load(f)

    region = config["region"]
    service_name = f"aither-{service.lower()}"

    cmd = ["gcloud", "run", "services", "logs", "read", service_name, f"--region={region}", f"--limit={limit}"]

    if follow:
        # For streaming, we need to use a different approach
        console.print(f"[bold] Streaming logs for {service_name}...[/]\n")
        os.system(f"gcloud beta run services logs tail {service_name} --region={region}")
    else:
        console.print(f"[bold] Logs for {service_name} (last {limit})...[/]\n")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print(f"[red]Error:[/] {result.stderr}")


# =============================================================================
# GENESIS COMMANDS - System Testing & Bootloader
# =============================================================================

@genesis_app.command("run")
def genesis_run(
    mode: str = typer.Option(
        "quick", "--mode", "-m",
        help="Test mode: quick, standard, exhaustive, autonomous"
    ),
    suites: Optional[str] = typer.Option(
        None, "--suites", "-s",
        help="Comma-separated list of test suites to run"
    ),
    models: Optional[str] = typer.Option(
        None, "--models",
        help="Comma-separated list of models to test"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output file for results (JSON)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed output"
    ),
):
    """
    Run AitherGenesis tests.

    Genesis is the system testing framework for AitherOS. It validates
    that all services are functioning correctly.

    Example:
        adk genesis run                       # Quick smoke test
        adk genesis run --mode standard       # Standard test suite
        adk genesis run --suites services,llm # Specific suites
    """
    import asyncio

    workspace = _find_workspace()

    console.print(Panel.fit(
        f"[bold cyan]AitherGenesis Test Runner[/]\n"
        f"Mode: {mode}",
        title="[DNA] Genesis",
        border_style="cyan",
    ))

    try:
        # Import Genesis modules
        sys.path.insert(0, str(workspace / "AitherOS" / "AitherNode"))

        from AitherGenesis.genesis_config import TEST_SUITES, GenesisConfig
        from AitherGenesis.genesis_runner import GenesisRunner, quick_health_check

        # Build config
        config = GenesisConfig()

        if suites:
            config.suites = [s.strip() for s in suites.split(",")]

        if models:
            config.models = [m.strip() for m in models.split(",")]

        # Run based on mode
        if mode == "quick":
            console.print("\n[bold]Running quick health check...[/]\n")

            async def run_quick():
                results = await quick_health_check()
                return results

            results = asyncio.run(run_quick())

            # Display results
            table = Table(title="Health Check Results", box=box.ROUNDED)
            table.add_column("Service", style="cyan")
            table.add_column("Status")
            table.add_column("Details", style="dim")

            for service, status in results.items():
                if isinstance(status, dict):
                    ok = status.get("healthy", False)
                    msg = status.get("message", "")
                else:
                    ok = status
                    msg = ""

                status_str = "[green][OK] Healthy[/]" if ok else "[red][X] Unhealthy[/]"
                table.add_row(service, status_str, msg[:50])

            console.print(table)

        else:
            console.print(f"\n[bold]Running {mode} test suite...[/]\n")

            async def run_full():
                runner = GenesisRunner(config)
                return await runner.run()

            result = asyncio.run(run_full())

            # Summary
            console.print(Panel.fit(
                f"[bold]Test Results[/]\n\n"
                f"Total: {result.total_tests}\n"
                f"[green]Passed: {result.passed}[/]\n"
                f"[red]Failed: {result.failed}[/]\n"
                f"[yellow]Skipped: {result.skipped}[/]\n"
                f"Duration: {result.duration_ms / 1000:.1f}s",
                title="Summary",
                border_style="green" if result.failed == 0 else "red",
            ))

            # Save results if output specified
            if output:
                with open(output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                console.print(f"\n[dim]Results saved to: {output}[/]")

    except ImportError as e:
        console.print("[red]Error:[/] Genesis not found. Make sure AitherOS is properly installed.")
        console.print(f"[dim]Details: {e}[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@genesis_app.command("status")
def genesis_status():
    """Show current Genesis status and last run results."""
    workspace = _find_workspace()

    console.print("[bold][DNA] Genesis Status[/]\n")

    # Check if Genesis service is running
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            console.print("[green][OK][/] Genesis service is running on port 8001")

            # Get status
            status_response = requests.get("http://localhost:8001/status", timeout=5)
            if status_response.status_code == 200:
                status = status_response.json()

                table = Table(box=box.ROUNDED)
                table.add_column("Property", style="cyan")
                table.add_column("Value")

                table.add_row("Status", status.get("status", "unknown"))
                table.add_row("Last Run", status.get("last_run", "never"))
                table.add_row("Current Mode", status.get("mode", "-"))
                table.add_row("Running Tests", str(status.get("running_tests", 0)))

                console.print(table)
        else:
            console.print("[yellow]![/] Genesis service returned non-200 status")
    except requests.exceptions.ConnectionError:
        console.print("[yellow]![/] Genesis service is not running")
        console.print("[dim]Start it with: python -m AitherOS.AitherNode.AitherGenesis.genesis_service[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")

    # Check for last results
    results_dir = workspace / "AitherOS" / "AitherNode" / "AitherGenesis" / "results"
    if results_dir.exists():
        results = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if results:
            console.print(f"\n[bold]Recent Results:[/] ({len(results)} total)")
            for r in results[:5]:
                console.print(f"  {r.name}")


@genesis_app.command("boot")
def genesis_boot(
    timeout: int = typer.Option(
        300, "--timeout", "-t",
        help="Timeout in seconds for boot"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential",
        help="Boot services in parallel"
    ),
):
    """
    Boot AitherOS using Genesis bootloader.

    This starts all AitherOS services in dependency order.

    Example:
        adk genesis boot
        adk genesis boot --timeout 600
    """
    import requests

    console.print(Panel.fit(
        "[bold cyan]Booting AitherOS[/]\n"
        "Starting all services in dependency order...",
        title="[LAUNCH] Genesis Boot",
        border_style="cyan",
    ))

    # Check if Genesis is running
    genesis_running = False
    try:
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            genesis_running = True
    except Exception as exc:
        logger.debug(f"Genesis health check failed: {exc}")

    if not genesis_running:
        console.print("[yellow]Starting Genesis bootloader...[/]")
        workspace = _find_workspace()

        # Try to start via Windows service first (most reliable)
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Start-Service", "AitherGenesis", "-ErrorAction", "Stop"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                console.print("[green]  [OK] Started via Windows service[/]")
            else:
                console.print("[dim]  Windows service unavailable; starting as subprocess...[/]")
                subprocess.Popen(
                    [sys.executable, "-m", "AitherGenesis.genesis_service"],
                    cwd=workspace / "AitherOS" / "AitherNode",
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            console.print(f"[dim]  Service start failed ({exc}); starting as subprocess...[/]")
            subprocess.Popen(
                [sys.executable, "-m", "AitherGenesis.genesis_service"],
                cwd=workspace / "AitherOS" / "AitherNode",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

        # Wait for Genesis to be ready (with progress)
        import time
        for i in range(15):  # 15 second timeout
            time.sleep(1)
            try:
                response = requests.get("http://localhost:8001/health", timeout=2)
                if response.status_code == 200:
                    console.print("[green]  [OK] Genesis ready[/]")
                    genesis_running = True
                    break
            except requests.RequestException:
                console.print(f"[dim]  Waiting for Genesis... ({i+1}s)[/]")

        if not genesis_running:
            console.print("[red]Error: Genesis failed to start. Check logs.[/]")
            raise typer.Exit(1)

    # Call startup endpoint - use /startup/fast for parallel boot without Watch dependency
    try:
        console.print("\n[bold]Initiating startup sequence...[/]\n")

        # Use /startup/fast for best performance and no AitherWatch dependency
        response = requests.post(
            "http://localhost:8001/startup/fast",
            params={"open_browser": "true"},
            timeout=timeout + 10,
        )

        if response.status_code == 200:
            result = response.json()

            # Show results
            table = Table(title="Boot Status", box=box.ROUNDED)
            table.add_column("Service", style="cyan")
            table.add_column("Status")
            table.add_column("Port", style="dim")

            for svc in result.get("services", []):
                name = svc.get("name", "unknown")
                status = svc.get("status", "unknown")
                port = svc.get("port", "-")

                if status == "healthy":
                    status_str = "[green][OK] Running[/]"
                elif status == "starting":
                    status_str = "[yellow]⋯ Starting[/]"
                else:
                    status_str = f"[red][X] {status}[/]"

                table.add_row(name, status_str, str(port))

            console.print(table)

            if result.get("success"):
                console.print("\n[bold green][DONE] AitherOS boot complete![/]")
            else:
                console.print("\n[bold yellow][WARN] Boot completed with warnings[/]")
        else:
            console.print(f"[red]Error:[/] Boot failed - {response.text}")
            raise typer.Exit(1)

    except requests.exceptions.Timeout:
        console.print(f"[red]Error:[/] Boot timed out after {timeout}s")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@genesis_app.command("shutdown")
def genesis_shutdown(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force shutdown without graceful stop"
    ),
):
    """
    Shutdown all AitherOS services.

    Example:
        adk genesis shutdown
        adk genesis shutdown --force
    """
    import requests

    console.print(Panel.fit(
        "[bold red]Shutting down AitherOS[/]\n"
        "Stopping all services...",
        title=" Genesis Shutdown",
        border_style="red",
    ))

    try:
        response = requests.post(
            "http://localhost:8001/shutdown",
            json={"force_after_timeout": force},
            timeout=120,
        )

        if response.status_code == 200:
            console.print("\n[green][OK][/] All services stopped")
        else:
            console.print(f"[red]Error:[/] {response.text}")

    except requests.exceptions.ConnectionError:
        console.print("[yellow]Genesis service not running[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")


@genesis_app.command("benchmark")
def genesis_benchmark(
    models: Optional[str] = typer.Option(
        None, "--models", "-m",
        help="Comma-separated list of models to benchmark"
    ),
    categories: Optional[str] = typer.Option(
        None, "--categories", "-c",
        help="Benchmark categories (reasoning, coding, creative, speed)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output file for results"
    ),
):
    """
    Run model benchmarks.

    Example:
        adk genesis benchmark
        adk genesis benchmark --models llama3,mistral
        adk genesis benchmark --categories speed,reasoning
    """
    console.print(Panel.fit(
        "[bold cyan]Model Benchmarks[/]\n"
        "Testing model performance...",
        title="[CHART] Benchmark",
        border_style="cyan",
    ))

    import requests

    try:
        payload = {}
        if models:
            payload["models"] = [m.strip() for m in models.split(",")]
        if categories:
            payload["categories"] = [c.strip() for c in categories.split(",")]

        response = requests.post(
            "http://localhost:8001/benchmark",
            json=payload,
            timeout=600,
        )

        if response.status_code == 200:
            result = response.json()

            # Display results
            for model, scores in result.get("results", {}).items():
                console.print(f"\n[bold cyan]{model}[/]")

                table = Table(box=box.SIMPLE)
                table.add_column("Category")
                table.add_column("Score", justify="right")
                table.add_column("Time (ms)", justify="right")

                for cat, data in scores.items():
                    table.add_row(
                        cat,
                        f"{data.get('score', 0):.1f}",
                        f"{data.get('time_ms', 0):.0f}"
                    )

                console.print(table)

            if output:
                with open(output, "w") as f:
                    json.dump(result, f, indent=2)
                console.print(f"\n[dim]Results saved to: {output}[/]")
        else:
            console.print(f"[red]Error:[/] {response.text}")

    except requests.exceptions.ConnectionError:
        console.print("[red]Error:[/] Genesis service not running. Start it with:")
        console.print("[dim]  python -m AitherOS.AitherNode.AitherGenesis.genesis_service[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")


# =============================================================================
# DOCS COMMANDS - Documentation
# =============================================================================

@docs_app.command("serve")
def docs_serve(
    port: int = typer.Option(
        8000, "--port", "-p",
        help="Port to serve docs on"
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open",
        help="Open browser automatically"
    ),
):
    """
    Serve the documentation locally.

    Example:
        adk docs serve
        adk docs serve --port 9000
    """
    workspace = _find_workspace()

    console.print(f"[bold] Serving documentation on http://localhost:{port}[/]\n")

    # Check if mkdocs is installed
    try:
        subprocess.run(["mkdocs", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[yellow]Installing mkdocs...[/]")
        subprocess.run([sys.executable, "-m", "pip", "install", "mkdocs", "mkdocs-material"], check=True)

    # Start mkdocs serve
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(2, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    os.chdir(workspace)
    os.system(f"mkdocs serve -a localhost:{port}")


@docs_app.command("build")
def docs_build(
    output: str = typer.Option(
        "site", "--output", "-o",
        help="Output directory"
    ),
):
    """
    Build documentation as static HTML.

    Example:
        adk docs build
        adk docs build --output docs_build
    """
    workspace = _find_workspace()

    console.print("[bold] Building documentation...[/]\n")

    os.chdir(workspace)
    result = subprocess.run(["mkdocs", "build", "-d", output], capture_output=True, text=True)

    if result.returncode == 0:
        console.print(f"[green][OK][/] Documentation built to: {output}/")
    else:
        console.print(f"[red]Error:[/] {result.stderr}")


@docs_app.command("api")
def docs_api():
    """
    Show ADK API reference.
    """
    console.print(Panel.fit(
        "[bold cyan]Aither Development Kit - API Reference[/]",
        border_style="cyan",
    ))

    api_docs = """
[bold]aither_adk.ai[/]
  OllamaLlm          - Local LLM provider (Ollama)
  AitherLlm          - Unified LLM gateway
  get_available_models() - List available models
  get_safety_settings()  - Get Gemini safety settings
  SafetyLevel        - Safety level enum (HIGH, MEDIUM, LOW)

[bold]aither_adk.infrastructure[/]
  configure_logging()    - Setup logging
  configure_auth()       - Setup API keys
  load_personas()        - Load persona configs
  get_service_status()   - Check service health
  is_aithernode_running() - Check if AitherNode is up
  process_turn()         - Run conversation turn

[bold]aither_adk.communication[/]
  Mailbox            - Inter-agent messaging
  GroupChatManager   - Multi-agent conversations
  CouncilClient      - Council API client
  A2AClient          - A2A protocol client

[bold]aither_adk.memory[/]
  MemoryManager      - Conversation memory
  MemorySystem       - Multi-layer memory
  GameEngine         - RPG game state

[bold]aither_adk.tools[/]
  aither_tools       - PowerShell automation tools
  mcp_server_tools   - MCP protocol tools
  remember()         - Store to memory
  recall()           - Retrieve from memory
  generate_image()   - Image generation

[bold]aither_adk.ui[/]
  console            - Rich console instance
  safe_print()       - Safe console output
  print_banner()     - Print agent banner
  run_agent_app()    - Run interactive agent CLI
"""

    console.print(api_docs)


@docs_app.command("quickstart")
def docs_quickstart():
    """
    Show quickstart guide.
    """
    quickstart = """
[bold cyan][LAUNCH] AitherOS Quickstart Guide[/]

[bold]1. Choose the right path[/]
    New standalone agent project:
      - Start from AitherZero/library/templates/adk-agent for local Ollama/vLLM-backed agents

    Full AitherOS platform development:
      pip install -e AitherOS/packages/aither_adk
      pip install -e AitherOS/packages/aither_toolkit

[bold]2. Verify Installation[/]
   adk check
   adk services

[bold]3. Create an Internal AitherOS Agent[/]
   adk new agent MyAgent -d "A helpful assistant"
   cd AitherOS/agents/MyAgent
   python agent.py

[bold]4. Create a Local Backend Agent from the Template[/]
    cd AitherZero/library/templates/adk-agent
    pip install -r requirements.txt
    python agent.py --backend ollama --model llama3.1:8b

[bold]5. Run Genesis Tests[/]
   adk genesis run --mode quick

[bold]6. Deploy to GCP[/]
   adk deploy init --project my-project
   adk deploy build
   adk deploy run -s AitherNode --public

[bold]Key Commands:[/]
   adk info          - Show ADK version
   adk check         - Verify installation
   adk services      - Check service status
   adk models        - List available models
   adk new agent     - Create new agent
   adk new service   - Create new service
   adk genesis run   - Run system tests
   adk genesis boot  - Start AitherOS
   adk deploy init   - Initialize GCP project

[dim]For more help: adk --help[/]
"""
    console.print(quickstart)


# =============================================================================
# RUN COMMANDS - Run Agents
# =============================================================================

def _discover_agents(workspace: Path) -> List[dict]:
    """Discover all available agents in the workspace."""
    agents_dir = workspace / "AitherOS" / "agents"
    agents = []

    if not agents_dir.exists():
        return agents

    for item in agents_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            agent_py = item / "agent.py"
            if agent_py.exists():
                # Try to extract description from docstring
                description = ""
                try:
                    with open(agent_py, "r", encoding="utf-8") as f:
                        content = f.read(2000)  # Read first 2KB
                        # Look for docstring
                        if '"""' in content:
                            start = content.find('"""') + 3
                            end = content.find('"""', start)
                            if end > start:
                                docstring = content[start:end].strip()
                                # Get first line as description
                                description = docstring.split("\n")[0].strip()
                                # Remove dashes
                                if " - " in description:
                                    description = description.split(" - ")[1].strip()
                                elif description.startswith(item.name):
                                    lines = docstring.split("\n")
                                    if len(lines) > 2:
                                        description = lines[2].strip()
                except Exception as exc:
                    logger.debug(f"Agent docstring extraction failed: {exc}")

                agents.append({
                    "name": item.name,
                    "path": str(item),
                    "description": description[:60] if description else "No description",
                    "has_service": (item / "service.py").exists(),
                    "has_requirements": (item / "requirements.txt").exists(),
                })

    return sorted(agents, key=lambda x: x["name"])


@run_app.callback(invoke_without_command=True)
def run_callback(ctx: typer.Context):
    """
    Run AitherOS agents.

    Example:
        adk run agent Aither                   # Run Aither agent
        adk run agent Saga --local   # Run with local model
        adk run list                           # List all agents
        adk run info Aither                    # Show agent details
    """
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand
        console.print(ctx.get_help())


@run_app.command("agent")
def run_agent(
    name: str = typer.Argument(..., help="Name of the agent to run"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model to use (e.g., gemini-2.5-flash, ollama/llama3)"
    ),
    local: bool = typer.Option(
        False, "--local", "-l",
        help="Use local Ollama model"
    ),
    persona: Optional[str] = typer.Option(
        None, "--persona", "-p",
        help="Persona to use (e.g., aither, terra)"
    ),
    safety: Optional[str] = typer.Option(
        None, "--safety", "-s",
        help="Safety level: high, medium, low"
    ),
    persistent: bool = typer.Option(
        False, "--persistent",
        help="Run as persistent server"
    ),
    port: int = typer.Option(
        8000, "--port",
        help="Port for persistent server mode"
    ),
):
    """
    Run an AitherOS agent interactively.

    Example:
        adk run agent Aither                           # Run Aither agent
        adk run agent Saga --local           # Run with local model
        adk run agent Aither --model gemini-2.5-flash  # Specific model
        adk run agent Aither --persona terra           # As Terra persona
    """
    workspace = _find_workspace()

    # Find the agent
    agents = _discover_agents(workspace)
    agent_info = next((a for a in agents if a["name"].lower() == name.lower()), None)

    if not agent_info:
        console.print(f"[red]Error:[/] Agent '{name}' not found")
        console.print("\n[bold]Available agents:[/]")
        for a in agents:
            console.print(f"  * {a['name']}")
        console.print("\n[dim]List all agents: adk run list[/]")
        raise typer.Exit(1)

    agent_path = Path(agent_info["path"])
    agent_py = agent_path / "agent.py"

    console.print(Panel.fit(
        f"[bold cyan]Running {agent_info['name']}[/]\n"
        f"{agent_info['description']}",
        title=" Agent",
        border_style="cyan",
    ))

    # Build command
    cmd = [sys.executable, str(agent_py)]

    if model:
        cmd.extend(["--model", model])

    if local:
        cmd.append("--local")

    if persona:
        cmd.extend(["--persona", persona])

    if safety:
        cmd.extend(["--safety", safety])

    if persistent:
        cmd.extend(["--persistent", "--port", str(port)])

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/]\n")

    # Run the agent
    try:
        os.chdir(agent_path)
        result = subprocess.run(cmd)
        raise typer.Exit(result.returncode)
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent stopped.[/]")
    except Exception as e:
        console.print(f"[red]Error running agent:[/] {e}")
        raise typer.Exit(1)


@run_app.command("list")
def run_list():
    """List all available agents."""
    workspace = _find_workspace()
    agents = _discover_agents(workspace)

    if not agents:
        console.print("[yellow]No agents found in AitherOS/agents/[/]")
        return

    console.print(Panel.fit(
        "[bold cyan]Available Agents[/]",
        border_style="cyan",
    ))

    table = Table(box=box.ROUNDED)
    table.add_column("Agent", style="cyan")
    table.add_column("Description")
    table.add_column("Service", style="dim")
    table.add_column("Requirements", style="dim")

    for a in agents:
        service_status = "[green][OK][/]" if a["has_service"] else "-"
        req_status = "[green][OK][/]" if a["has_requirements"] else "-"
        table.add_row(a["name"], a["description"], service_status, req_status)

    console.print(table)
    console.print("\n[dim]Run an agent: adk run agent <name>[/]")
    console.print("[dim]Options: --model, --local, --persona, --safety, --persistent[/]")


@run_app.command("info")
def run_info(
    agent: str = typer.Argument(..., help="Agent name"),
):
    """Show detailed information about an agent."""
    workspace = _find_workspace()
    agents = _discover_agents(workspace)

    agent_info = next((a for a in agents if a["name"].lower() == agent.lower()), None)

    if not agent_info:
        console.print(f"[red]Error:[/] Agent '{agent}' not found")
        raise typer.Exit(1)

    agent_path = Path(agent_info["path"])

    console.print(Panel.fit(
        f"[bold cyan]{agent_info['name']}[/]",
        border_style="cyan",
    ))

    # Basic info
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Path", str(agent_path))
    table.add_row("Description", agent_info["description"])
    table.add_row("Has service.py", "[green]Yes[/]" if agent_info["has_service"] else "[yellow]No[/]")
    table.add_row("Has requirements.txt", "[green]Yes[/]" if agent_info["has_requirements"] else "[yellow]No[/]")

    # Check for other files
    has_prompts = (agent_path / "prompts.py").exists()
    has_tools = (agent_path / "tools").exists()
    has_config = (agent_path / "config").exists()
    has_memory = (agent_path / "memory").exists()

    table.add_row("Has prompts.py", "[green]Yes[/]" if has_prompts else "[yellow]No[/]")
    table.add_row("Has tools/", "[green]Yes[/]" if has_tools else "[yellow]No[/]")
    table.add_row("Has config/", "[green]Yes[/]" if has_config else "[yellow]No[/]")
    table.add_row("Has memory/", "[green]Yes[/]" if has_memory else "[yellow]No[/]")

    console.print(table)

    # Show requirements if present
    req_file = agent_path / "requirements.txt"
    if req_file.exists():
        console.print("\n[bold]Dependencies:[/]")
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    console.print(f"  * {line}")

    # Usage hint
    console.print("\n[bold]Run:[/]")
    console.print(f"  adk run agent {agent_info['name']}")
    console.print(f"  adk run agent {agent_info['name']} --local")
    console.print(f"  adk run agent {agent_info['name']} --model gemini-2.5-flash")


@run_app.command("server")
def run_server(
    agent: str = typer.Argument(..., help="Agent name"),
    port: int = typer.Option(
        8000, "--port", "-p",
        help="Port to run server on"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model to use"
    ),
):
    """
    Run an agent as a persistent server.

    Example:
        adk run server Aither --port 8000
    """
    workspace = _find_workspace()
    agents = _discover_agents(workspace)

    agent_info = next((a for a in agents if a["name"].lower() == agent.lower()), None)

    if not agent_info:
        console.print(f"[red]Error:[/] Agent '{agent}' not found")
        raise typer.Exit(1)

    if not agent_info["has_service"]:
        console.print(f"[yellow]Warning:[/] {agent} does not have a service.py")
        console.print("[dim]Running in persistent mode with agent.py instead...[/]")

    agent_path = Path(agent_info["path"])

    console.print(Panel.fit(
        f"[bold cyan]Starting {agent_info['name']} Server[/]\n"
        f"Port: {port}",
        title="[LAUNCH] Server",
        border_style="cyan",
    ))

    # Try service.py first, fall back to agent.py --persistent
    if agent_info["has_service"]:
        cmd = [sys.executable, str(agent_path / "service.py"), "--port", str(port)]
    else:
        cmd = [sys.executable, str(agent_path / "agent.py"), "--persistent", "--port", str(port)]

    if model:
        cmd.extend(["--model", model])

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/]\n")

    try:
        os.chdir(agent_path)
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


def _generate_cloudbuild_yaml(project_id: str, region: str) -> str:
    """Generate a cloudbuild.yaml file for AitherOS."""
    return f'''# AitherOS Cloud Build Configuration
# Generated by: adk deploy init
# Project: {project_id}
# Region: {region}

steps:
  # Build AitherNode
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:$COMMIT_SHA'
      - '-t'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:latest'
      - '-f'
      - 'AitherOS/Dockerfile'
      - '.'
    id: 'build-aithernode'

  # Build AitherVeil (Frontend)
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:$COMMIT_SHA'
      - '-t'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:latest'
      - '.'
    dir: 'AitherOS/AitherNode/AitherVeil'
    id: 'build-aitherveil'

  # Push AitherNode
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:$COMMIT_SHA'
    id: 'push-aithernode'
    waitFor: ['build-aithernode']

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:latest'
    waitFor: ['push-aithernode']

  # Push AitherVeil
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:$COMMIT_SHA'
    id: 'push-aitherveil'
    waitFor: ['build-aitherveil']

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:latest'
    waitFor: ['push-aitherveil']

  # Deploy AitherNode to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'aither-node'
      - '--image={region}-docker.pkg.dev/{project_id}/aitheros/aithernode:$COMMIT_SHA'
      - '--region={region}'
      - '--platform=managed'
      - '--memory=4Gi'
      - '--cpu=2'
      - '--min-instances=0'
      - '--max-instances=10'
      - '--port=8080'
      - '--allow-unauthenticated'
    id: 'deploy-aithernode'
    waitFor: ['push-aithernode']

  # Deploy AitherVeil to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'aither-veil'
      - '--image={region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:$COMMIT_SHA'
      - '--region={region}'
      - '--platform=managed'
      - '--memory=1Gi'
      - '--cpu=1'
      - '--min-instances=0'
      - '--max-instances=5'
      - '--port=3000'
      - '--allow-unauthenticated'
    id: 'deploy-aitherveil'
    waitFor: ['push-aitherveil']

images:
  - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:$COMMIT_SHA'
  - '{region}-docker.pkg.dev/{project_id}/aitheros/aithernode:latest'
  - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:$COMMIT_SHA'
  - '{region}-docker.pkg.dev/{project_id}/aitheros/aitherveil:latest'

options:
  logging: CLOUD_LOGGING_ONLY
  machineType: 'E2_HIGHCPU_8'

timeout: '1800s'
'''


# =============================================================================
# DEV COMMANDS - Demiurge Development Mode (like Claude Code)
# =============================================================================

@dev_app.callback(invoke_without_command=True)
def dev_callback(ctx: typer.Context):
    """
    Development mode - Interactive Demiurge interface.

    Like Claude Code, provides AI-assisted development:
    - Intent-based code generation
    - Project analysis and review
    - Codebase search via neurons
    - Roadmap-aware suggestions

    Example:
        adk dev                    # Interactive mode
        adk dev intent "analyze auth flow"
        adk dev status
        adk dev suggest
    """
    if ctx.invoked_subcommand is None:
        # Start interactive mode
        _run_dev_interactive()


def _run_dev_interactive():
    """Run interactive Demiurge session."""
    import asyncio

    console.print(Panel.fit(
        "[bold cyan][HOT] The Forge - Development Mode[/bold cyan]\n\n"
        "Interactive Demiurge interface for AI-assisted development.\n"
        "Type your intent or use commands: /help, /status, /suggest, /exit",
        title="AitherDemiurge",
        border_style="cyan",
    ))

    async def interactive_loop():
        try:
            from .communication.demiurge_client import demiurge, format_demiurge_status
        except ImportError:
            console.print("[red]Error:[/] Demiurge client not available")
            return

        # Check if Demiurge is available
        if not await demiurge.is_available():
            console.print("[yellow][WARN] Demiurge service not running.[/]")
            console.print("[dim]Start AitherNode to enable Demiurge.[/]")
            console.print("[dim]Or run: python AitherOS/services/agents/AitherDemiurge.py[/]")
            return

        # Get initial status
        status = await demiurge.get_status()
        console.print(format_demiurge_status(status))
        console.print()

        while True:
            try:
                user_input = console.input("[bold cyan][HOT] >[/] ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input[1:].lower().split()[0]

                    if cmd in ("exit", "quit", "q"):
                        console.print("[dim]Leaving The Forge...[/]")
                        break
                    elif cmd == "help":
                        console.print("""
[bold]Commands:[/]
  /status     - Show Demiurge status
  /suggest    - Get suggestions for next task
  /mood       - Show current mood and creativity
  /awaken     - Awaken Demiurge if sleeping
  /exit       - Exit interactive mode

[bold]Intent Examples:[/]
  analyze the authentication flow
  create a FastAPI endpoint for user registration
  review services/auth/AitherAuth.py
  explain how the safety system works
  what should I work on next?
""")
                    elif cmd == "status":
                        status = await demiurge.get_status()
                        console.print(format_demiurge_status(status))
                    elif cmd == "suggest":
                        result = await demiurge.suggest_next()
                        console.print(f"\n{result.response}\n")
                    elif cmd == "mood":
                        mood = await demiurge.get_mood()
                        emoji = {"neutral": "", "focused": "[TARGET]", "creative": "*",
                                "concerned": "", "excited": "[!]", "sleeping": ""
                        }.get(mood.get("mood", ""), "")
                        console.print(f"\n{emoji} Mood: {mood.get('mood', 'unknown')}")
                        console.print(f"[ART] Creativity: {mood.get('creativity_boost', 1.0):.1f}x\n")
                    elif cmd == "awaken":
                        result = await demiurge.awaken()
                        console.print(f"\n{result}\n")
                    else:
                        console.print(f"[yellow]Unknown command: {cmd}[/]")
                    continue

                # Send intent to Demiurge
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Consulting The Forge...", total=None)
                    result = await demiurge.intent(user_input)

                if result.error:
                    console.print(f"[red]Error:[/] {result.error}")
                else:
                    mood_emoji = {"neutral": "", "focused": "[TARGET]", "creative": "*",
                                 "concerned": "", "excited": "[!]", "sleeping": ""
                    }.get(result.mood, "[HOT]")

                    console.print(f"\n{mood_emoji} [bold]Demiurge[/] ({result.mood}):\n")
                    console.print(result.response)

                    if result.neurons_streaming:
                        console.print("\n[dim][SIGNAL] Neurons gathering additional context...[/]")
                    console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Leaving The Forge...[/]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/] {e}")

        await demiurge.close()

    asyncio.run(interactive_loop())


@dev_app.command("status")
def dev_status():
    """Show Demiurge service status."""
    import asyncio

    async def check_status():
        try:
            from .communication.demiurge_client import demiurge, format_demiurge_status

            if not await demiurge.is_available():
                console.print("[red][X][/] Demiurge service is not running")
                console.print("\n[dim]Start with: python AitherOS/services/agents/AitherDemiurge.py[/]")
                return

            status = await demiurge.get_status()
            console.print(format_demiurge_status(status))

            # Also show temporal awareness
            temporal = await demiurge.get_temporal()
            if "error" not in temporal:
                console.print(f"\n[TIME] Time of Day: {temporal.get('temporal_context', {}).get('time_of_day', 'unknown')}")
                if temporal.get("active_deadline"):
                    deadline = temporal["active_deadline"]
                    console.print(f"[WARN]  Active Deadline: {deadline.get('name')} (urgency: {deadline.get('urgency', 0):.0%})")

            await demiurge.close()
        except ImportError:
            console.print("[red]Error:[/] Demiurge client not available")

    asyncio.run(check_status())


@dev_app.command("intent")
def dev_intent(
    message: str = typer.Argument(..., help="Intent/request for Demiurge"),
    context_file: Optional[str] = typer.Option(None, "--file", "-f", help="Context file path"),
):
    """
    Send a single intent to Demiurge.

    Example:
        adk dev intent "analyze the authentication flow"
        adk dev intent "create a user registration endpoint" -f services/auth/
    """
    import asyncio

    async def send_intent():
        try:
            from .communication.demiurge_client import demiurge

            if not await demiurge.is_available():
                console.print("[red][X][/] Demiurge service is not running")
                return

            context = {}
            if context_file:
                context["file"] = context_file

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Consulting The Forge...", total=None)
                result = await demiurge.intent(message, context)

            if result.error:
                console.print(f"[red]Error:[/] {result.error}")
            else:
                mood_emoji = {"neutral": "", "focused": "[TARGET]", "creative": "*",
                             "concerned": "", "excited": "[!]", "sleeping": ""
                }.get(result.mood, "[HOT]")

                console.print(Panel(
                    result.response,
                    title=f"{mood_emoji} Demiurge ({result.mood})",
                    border_style="cyan"
                ))

                if result.neurons_streaming:
                    console.print("[dim][SIGNAL] Neurons gathering additional context in background...[/]")

            await demiurge.close()
        except ImportError:
            console.print("[red]Error:[/] Demiurge client not available")

    asyncio.run(send_intent())


@dev_app.command("suggest")
def dev_suggest():
    """Get suggestions for what to work on next."""
    import asyncio

    async def get_suggestions():
        try:
            from .communication.demiurge_client import demiurge

            if not await demiurge.is_available():
                console.print("[red][X][/] Demiurge service is not running")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Checking roadmap...", total=None)
                result = await demiurge.suggest_next()

            if result.error:
                console.print(f"[red]Error:[/] {result.error}")
            else:
                console.print(Panel(
                    result.response,
                    title=" Roadmap Suggestions",
                    border_style="green"
                ))

                if result.roadmap_context:
                    console.print(f"\n[dim]Based on: {result.roadmap_context}[/]")

            await demiurge.close()
        except ImportError:
            console.print("[red]Error:[/] Demiurge client not available")

    asyncio.run(get_suggestions())


@dev_app.command("analyze")
def dev_analyze(
    path: str = typer.Argument(..., help="File or directory to analyze"),
    focus: str = typer.Option("structure", "--focus", "-f",
                              help="Analysis focus: structure, security, performance, deps"),
):
    """
    Analyze code with Demiurge.

    Example:
        adk dev analyze services/auth/
        adk dev analyze services/auth/AitherAuth.py --focus security
    """
    import asyncio

    async def run_analysis():
        try:
            from .communication.demiurge_client import demiurge

            if not await demiurge.is_available():
                console.print("[red][X][/] Demiurge service is not running")
                return

            console.print(f"[bold]Analyzing:[/] {path}")
            console.print(f"[bold]Focus:[/] {focus}\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Analyzing...", total=None)
                result = await demiurge.analyze(path, focus)

            if result.error:
                console.print(f"[red]Error:[/] {result.error}")
            else:
                console.print(Panel(
                    result.response,
                    title=f"[SEARCH] Analysis: {path}",
                    border_style="cyan"
                ))

            await demiurge.close()
        except ImportError:
            console.print("[red]Error:[/] Demiurge client not available")

    asyncio.run(run_analysis())


@dev_app.command("safety")
def dev_safety(
    level: Optional[str] = typer.Argument(None, help="Set safety level: professional, casual, unrestricted"),
):
    """
    Get or set safety level (syncs with UI and Council).

    Example:
        adk dev safety              # Show current level
        adk dev safety casual       # Set to casual
        adk dev safety unrestricted # Set to unrestricted (local LLM)
    """
    import asyncio

    async def handle_safety():
        try:
            from .communication.council_client import council
        except ImportError:
            council = None

        if level:
            # Set safety level
            try:
                from ..infrastructure.args import SAFETY_LEVEL_MAP, _sync_safety_with_service
                canonical = SAFETY_LEVEL_MAP.get(level.lower(), "professional")
                synced = _sync_safety_with_service(canonical)

                emoji = {"professional": "", "casual": "", "unrestricted": ""}.get(canonical, "")
                sync_status = "[green]synced with UI/Council[/]" if synced else "[yellow]local only[/]"

                console.print(f"\n{emoji} Safety level set to: [bold]{canonical.title()}[/]")
                console.print(f"[dim]Status: {sync_status}[/]\n")
            except ImportError:
                console.print("[red]Error:[/] Safety module not available")
        else:
            # Show current level
            try:
                from aither_adk.ai.safety_mode import (
                    get_current_level,
                    get_level_emoji,
                    get_level_name,
                )
                level_obj = get_current_level()
                emoji = get_level_emoji(level_obj)
                name = get_level_name(level_obj)
                console.print(f"\n{emoji} Current safety level: [bold]{name}[/]\n")

                # Show available levels
                console.print("[dim]Available levels:[/]")
                console.print("   professional - Business-safe, cloud LLM, strict filtering")
                console.print("   casual       - Relaxed but filtered, cloud LLM")
                console.print("  [UNLOCK] unrestricted - Local LLM, no content filters\n")
            except ImportError:
                # Fallback to council
                if council:
                    result = await council.get_safety_level()
                    if "error" not in result:
                        console.print(f"\nCurrent safety level: [bold]{result.get('level', 'unknown')}[/]\n")
                    else:
                        console.print(f"[red]Error:[/] {result['error']}")
                else:
                    console.print("[red]Error:[/] Safety module not available")

    asyncio.run(handle_safety())


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
