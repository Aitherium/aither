"""Interactive turn runner for the ADK platform shell.

Native, google-free. Drives the core :class:`adk.agent.AitherAgent` streaming
loop and renders it with Rich. Replaces the previous google-adk ``Runner``
event-stream consumer — there is now a single agent runtime (AitherAgent).

Public surface (imported by the platform UI):
    process_turn, add_attachment, get_show_reasoning, set_show_reasoning
"""

from __future__ import annotations

import logging
import os
import time

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from adk.platform.infrastructure.utils import (
    extract_thinking,
    strip_thinking,
)
from adk.platform.ui.console import console, safe_print

logger = logging.getLogger(__name__)


# ── Real-time context (optional, best-effort over HTTP) ──────────────────────


async def get_flux_context() -> str:
    """Return real-time context from a running AitherFlux service, or ''.

    Best-effort: resolves the URL from env and never raises. No monorepo import.
    """
    url = os.getenv("AITHER_FLUX_URL")
    if not url:
        return ""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/context")
            if resp.status_code == 200:
                return resp.json().get("context", "") or ""
    except Exception:
        pass
    return ""


async def emit_flux_event(event_type: str, **kwargs) -> None:
    """Emit an event to AitherFlux if configured (best-effort, never raises)."""
    url = os.getenv("AITHER_FLUX_URL")
    if not url:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{url.rstrip('/')}/emit", json={"type": event_type, **kwargs})
    except Exception:
        pass


# ── UI display state ─────────────────────────────────────────────────────────

SHOW_REASONING_TRACES = False
SHOW_SUBTHOUGHTS = True
ECOSYSTEM_INJECTION_LEVEL = "standard"


def set_show_reasoning(show: bool) -> None:
    global SHOW_REASONING_TRACES
    SHOW_REASONING_TRACES = show


def get_show_reasoning() -> bool:
    return SHOW_REASONING_TRACES


def set_show_subthoughts(show: bool) -> None:
    global SHOW_SUBTHOUGHTS
    SHOW_SUBTHOUGHTS = show


def get_show_subthoughts() -> bool:
    return SHOW_SUBTHOUGHTS


def set_ecosystem_level(level: str) -> None:
    global ECOSYSTEM_INJECTION_LEVEL
    ECOSYSTEM_INJECTION_LEVEL = level


def get_ecosystem_level() -> str:
    return ECOSYSTEM_INJECTION_LEVEL


_AGENT_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "bright_cyan"]


def get_agent_color(name: str) -> str:
    if not name:
        return "cyan"
    return _AGENT_COLORS[sum(ord(c) for c in name) % len(_AGENT_COLORS)]


# ── Attachments ───────────────────────────────────────────────────────────────
# Native attachment records: {"mime_type": str, "data": bytes|None, "path": str|None}.
PENDING_ATTACHMENTS: list[dict] = []


def add_attachment(attachment) -> None:
    """Queue an attachment for the next turn.

    Accepts a native dict ``{"mime_type", "data"/"path"}`` or a filesystem path.
    """
    if isinstance(attachment, str):
        PENDING_ATTACHMENTS.append({"path": attachment, "mime_type": "application/octet-stream"})
    elif isinstance(attachment, dict):
        PENDING_ATTACHMENTS.append(attachment)
    else:  # tolerate objects exposing .data/.mime_type
        PENDING_ATTACHMENTS.append({
            "data": getattr(attachment, "data", None),
            "mime_type": getattr(attachment, "mime_type", "application/octet-stream"),
        })


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60)}s"


def _attachment_note() -> str:
    if not PENDING_ATTACHMENTS:
        return ""
    names = []
    for a in PENDING_ATTACHMENTS:
        names.append(a.get("path") or f"<{a.get('mime_type', 'data')}>")
    return "\n\n[attachments: " + ", ".join(names) + "]"


# ── The turn ──────────────────────────────────────────────────────────────────


async def process_turn(
    agent,
    user_id: str = "user",
    session_id: str | None = None,
    user_input: str = "",
    model_name: str = "",
    session_stats: dict | None = None,
    root_agent=None,
    debug_mode: bool = False,
    memory_manager=None,
    show_spinner: bool = True,
    toolbar_renderer=None,
    mailbox=None,
) -> str:
    """Run one conversational turn through the native AitherAgent and render it.

    ``agent`` (or ``root_agent``) is an :class:`adk.agent.AitherAgent`. Streams
    the response with a live Markdown view, shows the reasoning trace when
    enabled, updates ``session_stats`` from the agent's meter, and returns the
    final assistant text.
    """
    agent = root_agent or agent
    if agent is None:
        safe_print("[bold red]No agent available for this turn.[/]")
        return ""

    session_stats = session_stats if session_stats is not None else {
        "total_cost": 0.0, "total_input": 0, "total_output": 0,
    }

    message = user_input + _attachment_note()
    PENDING_ATTACHMENTS.clear()

    color = get_agent_color(getattr(agent, "name", "agent"))
    started = time.monotonic()
    full = ""
    used_stream = True

    try:
        spinner = Spinner("dots", text=Text(" thinking…", style="dim"))
        with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
            first = True
            async for chunk in agent.chat_stream(message, session_id=session_id):
                if not chunk:
                    continue
                full += chunk
                visible = full if get_show_reasoning() else strip_thinking(full)
                try:
                    live.update(Markdown(visible) if visible.strip() else spinner)
                except Exception:
                    live.update(Text(visible))
                if first and toolbar_renderer:
                    try:
                        toolbar_renderer()
                    except Exception:
                        pass
                    first = False
    except Exception as exc:
        # Stream not supported / failed — fall back to a single non-streamed turn.
        used_stream = False
        logger.debug("chat_stream failed, falling back to chat(): %s", exc)
        try:
            resp = await agent.chat(message, session_id=session_id)
            full = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        except Exception as exc2:
            safe_print(f"[bold red]Turn failed: {exc2}[/]")
            return ""

    elapsed = time.monotonic() - started

    # Render the final answer (and reasoning trace if requested).
    answer = strip_thinking(full).strip()
    thinking = extract_thinking(full) if get_show_reasoning() else ""
    if thinking.strip():
        console.print(Panel(Markdown(thinking.strip()), title="[dim]reasoning[/]", border_style="dim"))
    if answer:
        console.print(Markdown(answer))

    # Update stats from the agent's meter when available.
    try:
        meter = getattr(agent, "meter", None)
        if meter is not None and hasattr(meter, "last_turn"):
            lt = meter.last_turn or {}
            session_stats["total_input"] += int(lt.get("input_tokens", 0) or 0)
            session_stats["total_output"] += int(lt.get("output_tokens", 0) or 0)
            session_stats["total_cost"] += float(lt.get("cost", 0.0) or 0.0)
    except Exception:
        pass

    if debug_mode:
        safe_print(f"[dim]turn: {_format_duration(elapsed)}, stream={used_stream}, "
                   f"chars={len(answer)}[/]")

    # Best-effort mailbox hook (non-fatal).
    if mailbox is not None:
        try:
            await emit_flux_event("turn_complete", agent=getattr(agent, "name", ""), chars=len(answer))
        except Exception:
            pass

    return answer
