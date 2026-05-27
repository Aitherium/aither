"""Saga Server — ADK server + StoryGraph REST endpoints.

Usage:
    python saga_server.py              # Start on port 8080
    python saga_server.py --port 9000  # Custom port
    adk run packages/saga              # Via ADK CLI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add package root to path for saga_engine imports
PACKAGE_DIR = Path(__file__).parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

logger = logging.getLogger("saga.server")


def create_saga_app(port: int = 8080):
    """Create the Saga app combining ADK agent + StoryGraph endpoints."""
    from adk.server import create_app
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    # Create ADK agent app
    app = create_app(identity="saga")

    # Mount StoryGraph REST API
    _mount_storygraph_api(app)

    # Mount Simulation API
    _mount_simulation_api(app)

    # Discover and activate addons
    _mount_addon_api(app)

    # Serve static UI if built
    ui_dist = PACKAGE_DIR / "ui" / "dist"
    if ui_dist.exists():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        logger.info(f"Serving UI from {ui_dist}")

    return app


def _mount_storygraph_api(app):
    """Add StoryGraph REST endpoints to the FastAPI app."""
    from fastapi import APIRouter
    from saga_engine.graph import StoryGraph
    from saga_engine.memory import MemoryManager
    from saga_engine.context import ContextAssembler
    from saga_engine.models import (
        CreateNodeRequest, CreateEdgeRequest, CreateMemoryRequest,
        CreateCharacterRequest, StoryTurnRequest, NodeType,
    )

    router = APIRouter(prefix="/storygraph", tags=["storygraph"])
    data_dir = Path.home() / ".saga" / "active_project"
    data_dir.mkdir(parents=True, exist_ok=True)

    graph = StoryGraph(data_dir=data_dir)
    graph.load()
    mem_mgr = MemoryManager(data_dir=data_dir)
    mem_mgr.load()
    ctx_asm = ContextAssembler()

    @router.get("/stats")
    async def get_stats():
        return {
            "graph": graph.stats(),
            "memories": mem_mgr.stats(),
        }

    @router.get("/nodes")
    async def list_nodes(type: str = ""):
        if type:
            try:
                nt = NodeType(type)
                nodes = graph.find_nodes_by_type(nt)
            except ValueError:
                nodes = graph.get_all_nodes()
        else:
            nodes = graph.get_all_nodes()
        return {"nodes": [n.model_dump() for n in nodes]}

    @router.get("/nodes/{node_id}")
    async def get_node(node_id: str):
        node = graph.get_node(node_id)
        if not node:
            return {"error": "Not found"}, 404
        neighbors = graph.get_neighbors(node_id)
        return {
            "node": node.model_dump(),
            "neighbors": [
                {"node": n.model_dump(), "edge": e.model_dump()}
                for n, e in neighbors
            ],
        }

    @router.post("/nodes")
    async def create_node(req: CreateNodeRequest):
        from saga_engine.models import StoryNode
        node = StoryNode(
            type=req.type, name=req.name, description=req.description,
            short_description=req.short_description, properties=req.properties,
            tags=req.tags, aliases=req.aliases, icon=req.icon,
            created_by=req.display_name,
        )
        graph.add_node(node)
        graph.save()
        return node.model_dump()

    @router.post("/edges")
    async def create_edge(req: CreateEdgeRequest):
        from saga_engine.models import StoryEdge
        edge = StoryEdge(
            type=req.type, source_id=req.source_id, target_id=req.target_id,
            label=req.label, weight=req.weight, bidirectional=req.bidirectional,
            created_by=req.display_name,
        )
        graph.add_edge(edge)
        graph.save()
        return edge.model_dump()

    @router.get("/memories")
    async def list_memories():
        return {"memories": [m.model_dump() for m in mem_mgr.get_all()]}

    @router.post("/memories")
    async def create_memory(req: CreateMemoryRequest):
        mem = mem_mgr.create(
            memory_type=req.type, content=req.content, summary=req.summary,
            importance=req.importance, related_nodes=req.related_nodes,
            tags=req.tags, story_time=req.story_time, pinned=req.pinned,
            turn_number=graph.world.turn_number, created_by=req.display_name,
        )
        mem_mgr.save()
        return mem.model_dump()

    @router.post("/context")
    async def assemble_context(req: StoryTurnRequest):
        assembly = ctx_asm.assemble(
            user_input=req.message, graph=graph,
            memory_manager=mem_mgr, world_state=graph.world,
            token_budget=req.context_budget,
        )
        return assembly.model_dump()

    @router.get("/world")
    async def get_world():
        return graph.world.model_dump()

    @router.post("/save")
    async def save_all():
        graph.save()
        mem_mgr.save()
        return {"saved": True}

    app.include_router(router)


def _mount_simulation_api(app):
    """Add simulation control endpoints to the FastAPI app."""
    from fastapi import APIRouter
    from pathlib import Path

    sim_router = APIRouter(prefix="/simulation", tags=["simulation"])
    data_dir = Path.home() / ".saga" / "active_project"

    @sim_router.post("/start")
    async def start_simulation():
        from saga_engine.simulation import get_simulation_client
        from saga_engine.graph import StoryGraph

        graph = StoryGraph(data_dir=data_dir)
        graph.load()

        sim = get_simulation_client()
        project_data = {
            "story_name": graph.world.story_name,
            "nodes": [n.model_dump() for n in graph.get_all_nodes()],
        }
        state = await sim.start(project_data)
        sim.sync_to_world_state(graph.world, graph)
        return {
            "world_id": state.world_id,
            "backend": sim.backend_name,
            "game_time": state.game_time,
            "weather": state.weather,
        }

    @sim_router.post("/tick")
    async def tick_simulation(minutes: int = 60):
        from saga_engine.simulation import get_simulation_client
        from saga_engine.sim_event_mapper import SimEventMapper
        from saga_engine.graph import StoryGraph
        from saga_engine.memory import MemoryManager

        graph = StoryGraph(data_dir=data_dir)
        graph.load()
        mem_mgr = MemoryManager(data_dir=data_dir)
        mem_mgr.load()

        sim = get_simulation_client()
        cached = sim.last_state
        if not cached or not cached.world_id:
            return {"error": "No simulation running. Call POST /simulation/start first."}

        result = await sim.tick(cached.world_id, minutes)
        mapper = SimEventMapper()
        mapper.tick_result_to_state_update(result, graph)
        memories = mapper.events_to_memories(result.events, graph, mem_mgr)

        return {
            "game_time": result.game_time,
            "weather": result.weather,
            "events": [{"type": e.event_type, "description": e.description, "severity": e.severity}
                       for e in result.events],
            "npc_movements": result.npc_movements,
            "faction_actions": result.faction_actions,
            "memories_stored": len(memories),
        }

    @sim_router.get("/state")
    async def get_simulation_state():
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        cached = sim.last_state
        if not cached or not cached.world_id:
            return {"error": "No simulation running"}
        state = await sim.get_state(cached.world_id)
        return {
            "world_id": state.world_id,
            "game_time": state.game_time,
            "weather": state.weather,
            "season": state.season,
            "npc_count": len(state.npc_states),
            "faction_count": len(state.faction_states),
            "awi_metrics": state.awi_metrics,
            "connected": state.connected,
        }

    @sim_router.get("/events")
    async def get_simulation_events(count: int = 20):
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        cached = sim.last_state
        if not cached or not cached.world_id:
            return {"error": "No simulation running"}
        events = await sim.get_recent_events(cached.world_id, count)
        return {
            "events": [
                {"type": e.event_type, "description": e.description, "severity": e.severity}
                for e in events
            ]
        }

    @sim_router.delete("")
    async def stop_simulation():
        from saga_engine.simulation import get_simulation_client
        from saga_engine.graph import StoryGraph

        sim = get_simulation_client()
        cached = sim.last_state
        if not cached or not cached.world_id:
            return {"stopped": False, "reason": "No simulation running"}

        await sim.stop(cached.world_id)

        graph = StoryGraph(data_dir=data_dir)
        graph.load()
        graph.world.prometheus_active = False
        graph.world.prometheus_world_id = None
        graph.world.simulation_time = None
        graph.world.simulation_weather = None
        graph.world.last_sim_events = []
        graph.save()

        return {"stopped": True}

    @sim_router.get("/awi")
    async def get_awi_metrics():
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        cached = sim.last_state
        if not cached or not cached.world_id:
            return {"error": "No simulation running"}
        return await sim.get_awi_metrics(cached.world_id)

    @sim_router.get("/backend")
    async def get_simulation_backend():
        from saga_engine.simulation import get_simulation_client
        sim = get_simulation_client()
        return {
            "backend": sim.backend_name,
            "active": sim.last_state is not None and sim.last_state.connected,
            "world_id": sim.last_state.world_id if sim.last_state else None,
        }

    app.include_router(sim_router)


def _mount_addon_api(app):
    """Discover addons and mount their endpoints."""
    from fastapi import APIRouter

    addon_router = APIRouter(prefix="/addons", tags=["addons"])

    try:
        from saga_engine.addons import get_addon_registry
        registry = get_addon_registry()
        registry.discover()
        registry.activate_all(app)
    except Exception:
        pass

    @addon_router.get("/installed")
    async def list_installed_addons():
        from saga_engine.addons import get_addon_registry
        return {"addons": get_addon_registry().list_addons()}

    @addon_router.get("/available")
    async def list_available_addons():
        from saga_engine.addons import get_addon_registry
        return {"addons": get_addon_registry().list_available()}

    app.include_router(addon_router)


def main():
    parser = argparse.ArgumentParser(description="Saga Standalone Server")
    parser.add_argument("--port", type=int, default=int(os.getenv("SAGA_PORT", "8080")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import uvicorn
    app = create_saga_app(port=args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
