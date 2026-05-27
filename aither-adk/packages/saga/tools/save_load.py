"""Save/Load/Branch Tool — Project management for story saves."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from adk.tools import tool

logger = logging.getLogger("saga.tools.save_load")

SAGA_HOME = Path.home() / ".saga"
PROJECTS_DIR = SAGA_HOME / "projects"


def _ensure_dirs():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


@tool(
    name="save_project",
    description="Save the current story project (world graph + memories) to a named save slot.",
)
def save_project(name: str, description: str = "") -> dict:
    """Save the current project state.

    Args:
        name: Save name (used as directory name)
        description: Optional description of this save point
    """
    _ensure_dirs()

    from .story_turn import _get_engine
    graph, memory, _ = _get_engine()

    save_dir = PROJECTS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save graph and memory
    graph._data_dir = save_dir
    graph.save()
    memory._data_dir = save_dir
    memory.save()

    # Save metadata
    meta = {
        "name": name,
        "description": description,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "story_name": graph.world.story_name,
        "turn_number": graph.world.turn_number,
        "node_count": graph.node_count,
        "memory_count": memory.count,
    }
    (save_dir / "project.json").write_text(json.dumps(meta, indent=2))

    return {"saved": name, "path": str(save_dir), **meta}


@tool(
    name="load_project",
    description="Load a saved story project by name.",
)
def load_project(name: str) -> dict:
    """Load a previously saved project.

    Args:
        name: The save name to load
    """
    _ensure_dirs()
    save_dir = PROJECTS_DIR / name

    if not save_dir.exists():
        return {"error": f"Project '{name}' not found"}

    from .story_turn import _get_engine
    graph, memory, _ = _get_engine()

    # Point to the save directory and load
    graph._data_dir = save_dir
    loaded_graph = graph.load()
    memory._data_dir = save_dir
    loaded_memory = memory.load()

    meta_path = save_dir / "project.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "loaded": name,
        "graph_loaded": loaded_graph,
        "memory_loaded": loaded_memory,
        "story_name": graph.world.story_name,
        "turn_number": graph.world.turn_number,
        **meta,
    }


@tool(
    name="list_projects",
    description="List all saved story projects.",
)
def list_projects() -> dict:
    """List all saved projects in ~/.saga/projects/."""
    _ensure_dirs()

    projects = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        if d.is_dir():
            meta_path = d / "project.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    projects.append(meta)
                except Exception:
                    projects.append({"name": d.name, "error": "corrupt metadata"})
            else:
                projects.append({"name": d.name, "description": "(no metadata)"})

    return {"count": len(projects), "projects": projects}


@tool(
    name="branch_project",
    description="Create a branch (copy) of the current project at a new save point, for exploring alternate storylines.",
)
def branch_project(branch_name: str, description: str = "") -> dict:
    """Branch the current project into a new save.

    Args:
        branch_name: Name for the branch
        description: Description of why this branch was created
    """
    _ensure_dirs()

    from .story_turn import _get_engine
    graph, memory, _ = _get_engine()

    # First save current state
    if graph._data_dir:
        graph.save()
        memory.save()

        # Copy to new directory
        branch_dir = PROJECTS_DIR / branch_name
        if branch_dir.exists():
            return {"error": f"Branch '{branch_name}' already exists"}

        shutil.copytree(graph._data_dir, branch_dir)

        # Update metadata
        meta_path = branch_dir / "project.json"
        meta = {
            "name": branch_name,
            "description": description or f"Branch from turn {graph.world.turn_number}",
            "branched_from": graph._data_dir.name,
            "branch_turn": graph.world.turn_number,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "story_name": graph.world.story_name,
            "turn_number": graph.world.turn_number,
            "node_count": graph.node_count,
            "memory_count": memory.count,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return {"branched": branch_name, "from_turn": graph.world.turn_number}

    return {"error": "No active project to branch from"}
