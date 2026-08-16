"""
AitherShell expedition status plugin
====================================

Integrated proactive visibility for expeditions.

Commands:
  aither expedition list               — show all active expeditions
  aither expedition status <id>        — show expedition + current phase + next gate
  aither expedition gate <id>          — list pending gates for expedition
  aither expedition open <id>          — open expedition in portal
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from adk.shell import ShellContext
from adk.client.services.AitherClient import get_aither_client

console = Console()


@dataclass
class ExpeditionSummary:
    """Cached expedition summary for display."""
    id: str
    title: str
    status: str
    current_phase: Optional[str]
    completed_tasks: int
    total_tasks: int
    pending_gates: int
    created_at: str


async def _fetch_expedition(expedition_id: str) -> Optional[Dict[str, Any]]:
    """Fetch expedition from Genesis."""
    try:
        client = await get_aither_client()
        resp = await client.get(f"/expedition/{expedition_id}/status")
        if resp.status_code == 200:
            return resp.json()
        console.print(f"[red]Error: {resp.status_code} fetching expedition[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error fetching expedition: {e}[/red]")
        return None


async def _fetch_expeditions() -> Optional[List[Dict[str, Any]]]:
    """Fetch all expeditions from Genesis."""
    try:
        client = await get_aither_client()
        resp = await client.get("/expedition/list")
        if resp.status_code == 200:
            return resp.json().get("expeditions", [])
        return []
    except Exception as e:
        console.print(f"[red]Error fetching expeditions: {e}[/red]")
        return []


async def _fetch_gates(expedition_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch pending gates for expedition."""
    try:
        client = await get_aither_client()
        resp = await client.get(f"/expedition/{expedition_id}/decisions")
        if resp.status_code == 200:
            return resp.json().get("decisions", [])
        return []
    except Exception as e:
        console.print(f"[red]Error fetching gates: {e}[/red]")
        return []


@click.group(name="expedition")
def expedition_group():
    """Expedition management — create, monitor, steer autonomous projects."""
    pass


@expedition_group.command(name="list")
async def list_expeditions(ctx: ShellContext):
    """List all active expeditions with status."""
    expeditions = await _fetch_expeditions()
    if not expeditions:
        console.print("[yellow]No expeditions found[/yellow]")
        return

    table = Table(title="Active Expeditions")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Phase", style="green")
    table.add_column("Progress", style="blue")
    table.add_column("Gates", style="red")

    for exp in expeditions:
        exp_id = exp.get("id", "unknown")[:8]
        title = exp.get("title", "Untitled")[:50]
        status = exp.get("status", "unknown")
        phase = exp.get("current_phase", "—")
        completed = exp.get("completed_tasks", 0)
        total = exp.get("total_tasks", 0)
        progress = f"{completed}/{total}" if total > 0 else "—"
        pending_gates = exp.get("pending_gates", 0)
        gates_str = f"[red]{pending_gates}[/red]" if pending_gates > 0 else "✓"

        table.add_row(exp_id, title, status, phase, progress, gates_str)

    console.print(table)


@expedition_group.command(name="status")
@click.argument("expedition_id")
async def show_status(ctx: ShellContext, expedition_id: str):
    """Show detailed expedition status."""
    exp = await _fetch_expedition(expedition_id)
    if not exp:
        console.print(f"[red]Expedition {expedition_id} not found[/red]")
        return

    # Build status panel
    lines = [
        f"[cyan]ID[/cyan]: {exp.get('id', '?')}",
        f"[cyan]Title[/cyan]: {exp.get('title', '?')}",
        f"[cyan]Status[/cyan]: {exp.get('status', '?')}",
        f"[cyan]Owner[/cyan]: {exp.get('owner', 'system')}",
        f"[cyan]Created[/cyan]: {exp.get('created_at', '?')}",
    ]

    console.print(Panel("\n".join(lines), title="Expedition Overview", expand=False))

    # Phase progress
    phase = exp.get("current_phase", {})
    if isinstance(phase, dict):
        console.print(f"\n[bold green]Current Phase[/bold green]: {phase.get('title', '?')}")
        console.print(f"  Description: {phase.get('description', '?')}")
        console.print(f"  Order: {phase.get('order', '?')}")

    # Task progress
    tasks = exp.get("tasks", [])
    if tasks:
        completed = sum(1 for t in tasks if t.get("status") == "completed")
        total = len(tasks)
        console.print(f"\n[bold blue]Tasks[/bold blue]: {completed}/{total} completed")

        # Show next pending task
        pending = [t for t in tasks if t.get("status") == "pending"]
        if pending:
            next_task = pending[0]
            console.print(f"  [yellow]Next[/yellow]: {next_task.get('title', '?')}")

    # Pending gates
    gates = await _fetch_gates(expedition_id)
    if gates:
        pending_gates = [g for g in gates if g.get("status") == "pending"]
        console.print(f"\n[bold red]{len(pending_gates)} Pending Gate(s)[/bold red]")
        for gate in pending_gates:
            gate_type = gate.get("gate_type", "decision")
            gate_id = gate.get("id", "?")[:8]
            desc = gate.get("description", "Approval pending")[:60]
            console.print(f"  [{gate_id}] {gate_type}: {desc}")


@expedition_group.command(name="gate")
@click.argument("expedition_id")
async def show_gates(ctx: ShellContext, expedition_id: str):
    """Show pending gates (decisions) for expedition."""
    gates = await _fetch_gates(expedition_id)
    if not gates:
        console.print(f"[green]✓ No pending gates for {expedition_id[:8]}[/green]")
        return

    pending_gates = [g for g in gates if g.get("status") == "pending"]
    if not pending_gates:
        console.print(f"[green]✓ No pending gates[/green]")
        return

    table = Table(title="Pending Gates")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="yellow")
    table.add_column("Required Role", style="green")

    for gate in pending_gates:
        gate_id = gate.get("id", "?")[:12]
        gate_type = gate.get("gate_type", "decision")
        desc = gate.get("description", "—")[:60]
        role = gate.get("required_role", "any")
        table.add_row(gate_id, gate_type, desc, role)

    console.print(table)
    console.print("\n[cyan]Hint:[/cyan] Use decision cards to answer gates (no CLI approve yet)")


@expedition_group.command(name="open")
@click.argument("expedition_id")
async def open_in_portal(ctx: ShellContext, expedition_id: str):
    """Open expedition in web portal."""
    portal_url = "https://portal.aitherium.com"
    full_url = f"{portal_url}/workspace/expeditions/{expedition_id}"
    console.print(f"[cyan]Opening expedition in portal...[/cyan]")
    console.print(f"[blue]{full_url}[/blue]")

    # Try to open in browser if available
    try:
        import webbrowser
        webbrowser.open(full_url)
    except Exception:
        console.print("[yellow]Could not open browser automatically[/yellow]")


@expedition_group.command(name="watch")
@click.argument("expedition_id")
@click.option("--interval", default=30, help="Polling interval in seconds")
async def watch_expedition(ctx: ShellContext, expedition_id: str, interval: int):
    """Watch expedition in real-time (polling)."""
    console.print(f"[cyan]Watching expedition {expedition_id[:8]}... (Ctrl+C to stop)[/cyan]\n")

    try:
        while True:
            exp = await _fetch_expedition(expedition_id)
            if exp:
                status = exp.get("status", "?")
                phase = exp.get("current_phase", {})
                phase_title = phase.get("title", "?") if isinstance(phase, dict) else "?"
                completed = exp.get("completed_tasks", 0)
                total = exp.get("total_tasks", 0)
                progress = f"{completed}/{total}"

                # Build compact status line
                status_icon = {
                    "planning": "📋",
                    "active": "▶️",
                    "blocked": "⏸️",
                    "review": "👀",
                    "completed": "✅",
                    "failed": "❌",
                }.get(status, "?")

                console.print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{status_icon} {status:10s} | {phase_title:40s} | "
                    f"{progress:10s}"
                )

                # Show pending gates if any
                gates = await _fetch_gates(expedition_id)
                pending = [g for g in gates if g.get("status") == "pending"]
                if pending:
                    console.print(f"  [red]⚠️  {len(pending)} gate(s) pending[/red]")

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching[/yellow]")


def register_commands(shell: "ShellContext") -> None:
    """Register expedition commands with the shell."""
    # Add the group to the shell's command registry
    shell.add_command_group(expedition_group)
