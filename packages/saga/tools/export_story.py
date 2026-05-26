"""Export Story Tool — Export stories to markdown, JSON, or SillyTavern format."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from adk.tools import tool

logger = logging.getLogger("saga.tools.export")


def _get_engine():
    from .story_turn import _get_engine
    return _get_engine()


@tool(
    name="export_story",
    description="Export the current story to a file format (markdown, json, or sillytavern).",
)
def export_story(
    format: str = "markdown",
    output_path: str = "",
    include_world: bool = True,
    include_memories: bool = False,
) -> dict:
    """Export the story.

    Args:
        format: 'markdown', 'json', or 'sillytavern'
        output_path: Where to save (default: ~/Documents/saga-export-{timestamp}.{ext})
        include_world: Include world/character info in export
        include_memories: Include memory store in export
    """
    graph, memory, _ = _get_engine()
    from saga_engine.models import NodeType

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if format == "markdown":
        content = _export_markdown(graph, memory, include_world, include_memories)
        ext = "md"
    elif format == "json":
        content = _export_json(graph, memory, include_world, include_memories)
        ext = "json"
    elif format == "sillytavern":
        content = _export_sillytavern(graph)
        ext = "json"
    else:
        return {"error": f"Unknown format '{format}'. Use 'markdown', 'json', or 'sillytavern'."}

    if not output_path:
        docs = Path.home() / "Documents"
        docs.mkdir(exist_ok=True)
        output_path = str(docs / f"saga-export-{timestamp}.{ext}")

    Path(output_path).write_text(content, encoding="utf-8")
    return {"exported": output_path, "format": format, "size": len(content)}


def _export_markdown(graph, memory, include_world, include_memories):
    from saga_engine.models import NodeType

    lines = [f"# {graph.world.story_name}", ""]

    if include_world:
        chars = graph.find_nodes_by_type(NodeType.CHARACTER)
        if chars:
            lines.append("## Characters")
            for c in chars:
                lines.append(f"### {c.name}")
                lines.append(c.description)
                lines.append("")

        locs = graph.find_nodes_by_type(NodeType.LOCATION)
        if locs:
            lines.append("## Locations")
            for loc in locs:
                lines.append(f"### {loc.name}")
                lines.append(loc.description)
                lines.append("")

    if include_memories:
        mems = memory.get_all()
        if mems:
            lines.append("## Story Memories")
            for m in sorted(mems, key=lambda x: x.turn_number):
                lines.append(f"- **[{m.type.value}]** {m.summary}")
            lines.append("")

    lines.append("---")
    lines.append(f"*Exported from Saga on {datetime.now().isoformat()}*")
    return "\n".join(lines)


def _export_json(graph, memory, include_world, include_memories):
    data = {
        "story_name": graph.world.story_name,
        "world_state": graph.world.model_dump(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    if include_world:
        data["nodes"] = [n.model_dump() for n in graph.get_all_nodes()]
        data["edges"] = [e.model_dump() for e in graph.get_all_edges()]
    if include_memories:
        data["memories"] = [m.model_dump() for m in memory.get_all()]
    return json.dumps(data, indent=2, default=str)


def _export_sillytavern(graph):
    from saga_engine.models import NodeType
    chars = graph.find_nodes_by_type(NodeType.CHARACTER)

    # SillyTavern character card format (V2)
    if chars:
        main = chars[0]
        card = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": main.name,
                "description": main.description,
                "personality": main.properties.get("personality", ""),
                "scenario": graph.world.story_name,
                "first_mes": f"*{main.name} stands before you in {graph.world.story_name}.*",
                "mes_example": "",
                "creator_notes": f"Exported from Saga - {graph.world.story_name}",
                "system_prompt": "",
                "tags": main.tags,
            },
        }
        return json.dumps(card, indent=2)

    return json.dumps({"error": "No characters to export"})
