"""
StoryGraph Engine
==================

The core graph data structure — nodes, edges, indices, traversal, persistence.
No external graph database needed. In-memory with JSON persistence.
Designed for one shared community story (Elysium).

Operations:
    - Add/update/delete nodes and edges
    - Query by type, name, tags
    - Traverse edges (BFS with depth limit)
    - Subgraph extraction (all nodes within N hops)
    - JSON persistence (save/load)
"""

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    EdgeType,
    NodeStatus,
    NodeType,
    StoryEdge,
    StoryNode,
    WorldState,
    _make_id,
    _now,
)

logger = logging.getLogger("Saga.StoryGraph")


class StoryGraph:
    """
    A story-world graph with indexed lookups and BFS traversal.

    Indices:
        _nodes:      Dict[id → StoryNode]
        _edges:      Dict[id → StoryEdge]
        _outgoing:   Dict[node_id → [edge_ids]]  (edges FROM this node)
        _incoming:   Dict[node_id → [edge_ids]]  (edges TO this node)
        _name_index: Dict[lowercase_name → node_id]
        _type_index: Dict[NodeType → {node_ids}]
    """

    def __init__(self, data_dir: Optional[Path] = None):
        # Storage
        self._nodes: Dict[str, StoryNode] = {}
        self._edges: Dict[str, StoryEdge] = {}

        # Indices — rebuilt on load
        self._outgoing: Dict[str, List[str]] = defaultdict(list)
        self._incoming: Dict[str, List[str]] = defaultdict(list)
        self._name_index: Dict[str, str] = {}          # lowercase → node_id
        self._type_index: Dict[str, Set[str]] = defaultdict(set)

        # World state
        self.world: WorldState = WorldState()

        # Persistence
        self._data_dir = data_dir
        if data_dir:
            data_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # NODE OPERATIONS
    # ========================================================================

    def add_node(self, node: StoryNode) -> StoryNode:
        """Add a node to the graph. Updates indices."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} already exists. Use update_node().")

        self._nodes[node.id] = node
        self._index_node(node)
        logger.debug(f"Added node: {node.type.value}/{node.name} ({node.id})")
        return node

    def update_node(self, node_id: str, **updates) -> StoryNode:
        """Update fields on an existing node."""
        node = self.get_node(node_id)
        if not node:
            raise KeyError(f"Node {node_id} not found")

        # Remove old index entries
        self._unindex_node(node)

        # Apply updates
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        node.updated_at = _now()

        # Re-index
        self._index_node(node)
        return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges."""
        node = self._nodes.pop(node_id, None)
        if not node:
            return False

        self._unindex_node(node)

        # Remove connected edges
        edge_ids_to_remove = set()
        edge_ids_to_remove.update(self._outgoing.pop(node_id, []))
        edge_ids_to_remove.update(self._incoming.pop(node_id, []))
        for eid in edge_ids_to_remove:
            edge = self._edges.pop(eid, None)
            if edge:
                # Clean up the other side's adjacency
                if edge.source_id != node_id:
                    self._outgoing.get(edge.source_id, [])
                    if eid in self._outgoing.get(edge.source_id, []):
                        self._outgoing[edge.source_id].remove(eid)
                if edge.target_id != node_id:
                    if eid in self._incoming.get(edge.target_id, []):
                        self._incoming[edge.target_id].remove(eid)

        # Remove from world state references
        if node_id in self.world.present_characters:
            self.world.present_characters.remove(node_id)
        if node_id in self.world.active_plot_threads:
            self.world.active_plot_threads.remove(node_id)
        if self.world.current_location == node_id:
            self.world.current_location = None
        if self.world.current_scene == node_id:
            self.world.current_scene = None

        logger.debug(f"Deleted node: {node.name} ({node_id}) + {len(edge_ids_to_remove)} edges")
        return True

    def get_node(self, node_id: str) -> Optional[StoryNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def find_node_by_name(self, name: str) -> Optional[StoryNode]:
        """Find a node by name (case-insensitive)."""
        node_id = self._name_index.get(name.lower())
        return self._nodes.get(node_id) if node_id else None

    def find_nodes_by_type(self, node_type: NodeType) -> List[StoryNode]:
        """Get all nodes of a given type."""
        ids = self._type_index.get(node_type.value, set())
        return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def find_nodes_by_tag(self, tag: str) -> List[StoryNode]:
        """Find all nodes with a given tag."""
        tag_lower = tag.lower()
        return [n for n in self._nodes.values() if tag_lower in [t.lower() for t in n.tags]]

    def search_nodes(self, query: str, node_type: Optional[NodeType] = None) -> List[StoryNode]:
        """Search nodes by name/description substring."""
        q = query.lower()
        results = []
        candidates = (
            self.find_nodes_by_type(node_type) if node_type
            else list(self._nodes.values())
        )
        for node in candidates:
            if (q in node.name.lower()
                    or q in node.description.lower()
                    or any(q in a.lower() for a in node.aliases)):
                results.append(node)
        return results

    def get_all_nodes(self) -> List[StoryNode]:
        """Return all nodes."""
        return list(self._nodes.values())

    def get_pinned_nodes(self) -> List[StoryNode]:
        """Return all pinned nodes (always included in context)."""
        return [n for n in self._nodes.values() if n.pinned]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    # ========================================================================
    # EDGE OPERATIONS
    # ========================================================================

    def add_edge(self, edge: StoryEdge) -> StoryEdge:
        """Add an edge between two nodes."""
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id} not found")

        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)

        # If bidirectional, also add the reverse
        if edge.bidirectional:
            self._outgoing[edge.target_id].append(edge.id)
            self._incoming[edge.source_id].append(edge.id)

        src = self._nodes[edge.source_id].name
        tgt = self._nodes[edge.target_id].name
        logger.debug(f"Added edge: {src} --[{edge.type.value}]--> {tgt}")
        return edge

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        edge = self._edges.pop(edge_id, None)
        if not edge:
            return False

        if edge_id in self._outgoing.get(edge.source_id, []):
            self._outgoing[edge.source_id].remove(edge_id)
        if edge_id in self._incoming.get(edge.target_id, []):
            self._incoming[edge.target_id].remove(edge_id)

        if edge.bidirectional:
            if edge_id in self._outgoing.get(edge.target_id, []):
                self._outgoing[edge.target_id].remove(edge_id)
            if edge_id in self._incoming.get(edge.source_id, []):
                self._incoming[edge.source_id].remove(edge_id)

        return True

    def get_edge(self, edge_id: str) -> Optional[StoryEdge]:
        return self._edges.get(edge_id)

    def get_edges_from(self, node_id: str) -> List[StoryEdge]:
        """All edges originating from a node."""
        return [self._edges[eid] for eid in self._outgoing.get(node_id, [])
                if eid in self._edges]

    def get_edges_to(self, node_id: str) -> List[StoryEdge]:
        """All edges pointing to a node."""
        return [self._edges[eid] for eid in self._incoming.get(node_id, [])
                if eid in self._edges]

    def get_all_edges(self) -> List[StoryEdge]:
        return list(self._edges.values())

    def get_neighbors(self, node_id: str) -> List[Tuple[StoryNode, StoryEdge]]:
        """Get all directly connected nodes with their edge."""
        neighbors = []
        for edge in self.get_edges_from(node_id):
            target = self._nodes.get(edge.target_id)
            if target:
                neighbors.append((target, edge))
        for edge in self.get_edges_to(node_id):
            source = self._nodes.get(edge.source_id)
            if source and source.id != node_id:
                neighbors.append((source, edge))
        return neighbors

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # ========================================================================
    # TRAVERSAL — BFS with depth limit and hop decay
    # ========================================================================

    def bfs(
        self,
        start_ids: List[str],
        max_depth: int = 2,
        edge_types: Optional[List[EdgeType]] = None,
        node_types: Optional[List[NodeType]] = None,
    ) -> List[Tuple[StoryNode, int, str]]:
        """
        BFS from start nodes up to max_depth hops.

        Returns: [(node, depth, reason_string), ...]
        Reason explains the path: "1-hop: knows Kael", "2-hop: located_at Ironhold → contains Throne Room"
        """
        visited: Set[str] = set(start_ids)
        results: List[Tuple[StoryNode, int, str]] = []
        queue: deque = deque()

        # Seed the queue
        for sid in start_ids:
            node = self._nodes.get(sid)
            if node:
                for neighbor, edge in self.get_neighbors(sid):
                    if neighbor.id not in visited:
                        if edge_types and edge.type not in edge_types:
                            continue
                        if node_types and neighbor.type not in node_types:
                            continue
                        reason = f"1-hop: {edge.type.value} {node.name}"
                        queue.append((neighbor, 1, reason))

        while queue:
            current_node, depth, reason = queue.popleft()
            if current_node.id in visited:
                continue
            visited.add(current_node.id)
            results.append((current_node, depth, reason))

            if depth < max_depth:
                for neighbor, edge in self.get_neighbors(current_node.id):
                    if neighbor.id not in visited:
                        if edge_types and edge.type not in edge_types:
                            continue
                        if node_types and neighbor.type not in node_types:
                            continue
                        new_reason = f"{depth + 1}-hop: {reason} → {edge.type.value} {current_node.name}"
                        queue.append((neighbor, depth + 1, new_reason))

        return results

    def subgraph(
        self, center_ids: List[str], max_depth: int = 2
    ) -> Tuple[List[StoryNode], List[StoryEdge]]:
        """Extract a subgraph around center nodes."""
        node_ids = set(center_ids)
        reachable = self.bfs(center_ids, max_depth=max_depth)
        for node, _, _ in reachable:
            node_ids.add(node.id)

        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
        edges = [
            e for e in self._edges.values()
            if e.source_id in node_ids and e.target_id in node_ids
        ]
        return nodes, edges

    # ========================================================================
    # NAME MATCHING — for entity extraction
    # ========================================================================

    def match_names_in_text(self, text: str) -> List[Tuple[StoryNode, str]]:
        """
        Find all graph node names mentioned in text.
        Returns: [(node, matched_name), ...]
        """
        text_lower = text.lower()
        matches = []
        seen_ids = set()

        # Sort by name length (longest first to avoid partial matches)
        all_names = []
        for node in self._nodes.values():
            for name in node.search_names:
                if len(name) >= 3:  # Skip very short names
                    all_names.append((name, node))

        all_names.sort(key=lambda x: len(x[0]), reverse=True)

        for name, node in all_names:
            if node.id not in seen_ids and name in text_lower:
                matches.append((node, name))
                seen_ids.add(node.id)

        return matches

    # ========================================================================
    # INDEXING
    # ========================================================================

    def _index_node(self, node: StoryNode):
        """Add a node to all indices."""
        for name in node.search_names:
            self._name_index[name] = node.id
        self._type_index[node.type.value].add(node.id)

    def _unindex_node(self, node: StoryNode):
        """Remove a node from all indices."""
        for name in node.search_names:
            self._name_index.pop(name, None)
        self._type_index.get(node.type.value, set()).discard(node.id)

    def _rebuild_indices(self):
        """Rebuild all indices from scratch (after load)."""
        self._name_index.clear()
        self._type_index.clear()
        self._outgoing.clear()
        self._incoming.clear()

        for node in self._nodes.values():
            self._index_node(node)

        for edge in self._edges.values():
            self._outgoing[edge.source_id].append(edge.id)
            self._incoming[edge.target_id].append(edge.id)
            if edge.bidirectional:
                self._outgoing[edge.target_id].append(edge.id)
                self._incoming[edge.source_id].append(edge.id)

    # ========================================================================
    # PERSISTENCE — JSON save/load
    # ========================================================================

    def _graph_path(self) -> Path:
        assert self._data_dir, "No data_dir configured"
        return self._data_dir / "storygraph.json"

    def save(self):
        """Persist the full graph to disk."""
        if not self._data_dir:
            logger.warning("No data_dir — skipping save")
            return

        data = {
            "world": self.world.model_dump(),
            "nodes": {nid: n.model_dump() for nid, n in self._nodes.items()},
            "edges": {eid: e.model_dump() for eid, e in self._edges.items()},
        }
        path = self._graph_path()
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"Graph saved: {self.node_count} nodes, {self.edge_count} edges → {path}")

    def load(self) -> bool:
        """Load graph from disk. Returns True if loaded."""
        if not self._data_dir:
            return False

        path = self._graph_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            self.world = WorldState(**data.get("world", {}))
            self._nodes = {
                nid: StoryNode(**ndata)
                for nid, ndata in data.get("nodes", {}).items()
            }
            self._edges = {
                eid: StoryEdge(**edata)
                for eid, edata in data.get("edges", {}).items()
            }
            self._rebuild_indices()
            logger.info(
                f"Graph loaded: {self.node_count} nodes, {self.edge_count} edges ← {path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return False

    # ========================================================================
    # STATS
    # ========================================================================

    def stats(self) -> Dict[str, Any]:
        """Graph statistics for the frontend."""
        type_counts = {}
        for node_type in NodeType:
            count = len(self._type_index.get(node_type.value, set()))
            if count > 0:
                type_counts[node_type.value] = count

        return {
            "story_name": self.world.story_name,
            "turn_number": self.world.turn_number,
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "node_types": type_counts,
            "pinned_nodes": len(self.get_pinned_nodes()),
            "active_characters": len(self.world.present_characters),
            "active_plot_threads": len(self.world.active_plot_threads),
            "current_location": (
                self._nodes[self.world.current_location].name
                if self.world.current_location and self.world.current_location in self._nodes
                else None
            ),
        }
