"""Simulation Event Mapper — Converts sim events into Saga memories and graph updates.

Bridges the gap between simulation output (SimulationEvent, TickResult)
and Saga's StoryGraph/MemoryManager data structures.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .models import EdgeType, MemoryType, NodeType, StoryMemory
from .simulation import SimulationEvent, SimulationState, TickResult

logger = logging.getLogger("saga.sim_event_mapper")

# Severity → memory importance mapping
SEVERITY_IMPORTANCE = {
    "trivial": 0.0,    # Skip
    "normal": 0.3,
    "notable": 0.6,
    "important": 0.8,
    "critical": 1.0,
}


class SimEventMapper:
    """Maps simulation events to Saga memories and graph updates."""

    def events_to_memories(
        self,
        events: List[SimulationEvent],
        graph,
        memory,
    ) -> List[StoryMemory]:
        """Convert simulation events into stored StoryMemory objects.

        Args:
            events: Simulation events from a tick
            graph: StoryGraph instance
            memory: MemoryManager instance

        Returns:
            List of created StoryMemory objects
        """
        created = []
        for event in events:
            importance = SEVERITY_IMPORTANCE.get(event.severity, 0.3)
            if importance <= 0.0:
                continue  # Skip trivial events

            # Find related nodes by matching actor names to graph
            related_nodes = []
            for actor in event.actors:
                node = graph.find_node_by_name(actor)
                if node:
                    related_nodes.append(node.id)

            pinned = importance >= 0.8

            mem = memory.create(
                memory_type=MemoryType.EPISODIC,
                content=f"[SIMULATION] {event.description}",
                summary=event.description[:120],
                importance=importance,
                pinned=pinned,
                related_nodes=related_nodes,
                tags=["simulation", event.event_type],
                turn_number=graph.world.turn_number,
                created_by="simulation",
            )
            created.append(mem)

        if created:
            memory.save()
            logger.info("Stored %d simulation events as memories", len(created))

        return created

    def update_world_state(self, state: SimulationState, graph):
        """Update Saga WorldState and graph from simulation state.

        Args:
            state: Current SimulationState
            graph: StoryGraph instance
        """
        world = graph.world
        world.prometheus_world_id = state.world_id
        world.prometheus_active = state.connected
        world.simulation_time = state.game_time
        world.simulation_weather = state.weather
        world.last_sim_events = [
            e.description for e in state.events[-5:]
        ]

        # Update NPC locations from simulation
        for npc_id, npc_data in state.npc_states.items():
            sim_location = npc_data.get("location", "")
            if not sim_location:
                continue

            # Find the NPC node by name or ID
            npc_node = graph.get_node(npc_id)
            if not npc_node:
                npc_node = graph.find_node_by_name(npc_data.get("name", ""))
            if not npc_node:
                continue

            # Find or create location node
            loc_node = graph.find_node_by_name(sim_location)
            if not loc_node:
                continue

            # Update LOCATED_AT edge
            existing_loc_edges = [
                (n, e) for n, e in graph.get_neighbors(npc_node.id)
                if e.type == EdgeType.LOCATED_AT
            ]
            for _, edge in existing_loc_edges:
                if edge.target_id != loc_node.id:
                    graph.remove_edge(edge.id)

            if not any(e.target_id == loc_node.id for _, e in existing_loc_edges):
                from .models import StoryEdge
                graph.add_edge(StoryEdge(
                    type=EdgeType.LOCATED_AT,
                    source_id=npc_node.id,
                    target_id=loc_node.id,
                    label=f"{npc_node.name} is at {loc_node.name}",
                    created_by="simulation",
                ))

        # Update faction relationships
        for faction_id, faction_data in state.faction_states.items():
            faction_node = graph.get_node(faction_id)
            if not faction_node:
                faction_node = graph.find_node_by_name(faction_data.get("name", ""))
            if not faction_node:
                continue

            for rel in faction_data.get("relationships", []):
                target_name = rel.get("faction", rel.get("with", ""))
                rel_type = rel.get("type", "")
                target_node = graph.find_node_by_name(target_name)
                if not target_node:
                    continue

                if rel_type in ("hostile", "enemy", "at_war"):
                    edge_type = EdgeType.HOSTILE_TO
                elif rel_type in ("allied", "ally", "friendly"):
                    edge_type = EdgeType.ALLIED_WITH
                else:
                    continue

                # Check if edge already exists
                existing = False
                for neighbor, edge in graph.get_neighbors(faction_node.id):
                    if neighbor.id == target_node.id and edge.type == edge_type:
                        existing = True
                        break

                if not existing:
                    from .models import StoryEdge
                    graph.add_edge(StoryEdge(
                        type=edge_type,
                        source_id=faction_node.id,
                        target_id=target_node.id,
                        label=f"{faction_node.name} is {rel_type} with {target_node.name}",
                        bidirectional=True,
                        created_by="simulation",
                    ))

        graph.save()

    def tick_result_to_state_update(self, result: TickResult, graph):
        """Apply tick result changes to the world state.

        Args:
            result: TickResult from a simulation tick
            graph: StoryGraph instance
        """
        world = graph.world
        if result.game_time:
            world.simulation_time = result.game_time
        if result.weather:
            world.simulation_weather = result.weather
            world.weather = result.weather
        world.last_sim_events = [e.description for e in result.events[-5:]]
        graph.save()

    def diary_to_memory(
        self,
        npc_id: str,
        diary_text: str,
        graph,
        memory,
    ) -> Optional[StoryMemory]:
        """Convert a citizen diary entry into an EMOTIONAL memory.

        Args:
            npc_id: NPC identifier
            diary_text: Diary text content
            graph: StoryGraph instance
            memory: MemoryManager instance

        Returns:
            Created StoryMemory or None
        """
        npc_node = graph.get_node(npc_id)
        if not npc_node:
            npc_node = graph.find_node_by_name(npc_id)

        related_nodes = [npc_node.id] if npc_node else []
        npc_name = npc_node.name if npc_node else npc_id

        mem = memory.create(
            memory_type=MemoryType.EMOTIONAL,
            content=f"[DIARY: {npc_name}] {diary_text}",
            summary=f"{npc_name}'s diary: {diary_text[:80]}",
            importance=0.5,
            related_nodes=related_nodes,
            tags=["simulation", "diary", "emotional"],
            turn_number=graph.world.turn_number,
            created_by="simulation:diary",
        )
        memory.save()
        return mem
