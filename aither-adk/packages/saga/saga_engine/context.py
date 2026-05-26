"""
Context Assembler — The Glass Box
===================================

A 6-stage pipeline that builds the LLM context window for each story turn.
Every decision is logged and visible to the user. No black magic.

Pipeline Stages:
    1. EXTRACT  🔍  Parse user input for entity references + intents
    2. ACTIVATE ⚡  Match extracted entities to graph nodes + scene state
    3. EXPAND   🌐  Walk edges 1-2 hops for related context
    4. RECALL   🧠  Query memory store for relevant memories
    5. RANK     📊  Score everything by relevance
    6. ASSEMBLE 📦  Build final context, log what's in and what's out

The output is a ContextAssembly object that the frontend renders as a
transparent view of every piece of context and WHY it's there.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    ActivatedNode,
    ContextAssembly,
    ContextStage,
    EdgeType,
    NodeType,
    PrunedItem,
    RecalledMemory,
    StoryNode,
    _make_id,
)

logger = logging.getLogger("Saga.ContextAssembler")

# Rough token estimator: ~4 chars per token for English
CHARS_PER_TOKEN = 4

# Relevance thresholds
DIRECT_MENTION_SCORE = 0.95
CURRENT_SCENE_SCORE = 0.90
CURRENT_LOCATION_SCORE = 0.85
PRESENT_CHARACTER_SCORE = 0.80
ACTIVE_PLOT_SCORE = 0.75
PINNED_SCORE = 0.99
HOP_1_DECAY = 0.70
HOP_2_DECAY = 0.45
MEMORY_BASE_SCORE = 0.60
PRUNING_THRESHOLD = 0.20

# Intent keywords (rough detection for context expansion)
INTENT_COMBAT = {"fight", "attack", "battle", "strike", "defend", "sword", "weapon", "kill"}
INTENT_SOCIAL = {"talk", "speak", "ask", "tell", "persuade", "negotiate", "greet", "meet"}
INTENT_EXPLORE = {"go", "travel", "explore", "enter", "leave", "approach", "walk", "visit"}
INTENT_EXAMINE = {"look", "examine", "inspect", "study", "read", "search", "investigate"}
INTENT_MAGIC = {"cast", "spell", "magic", "enchant", "summon", "ritual", "invoke"}
INTENT_CREATE = {"craft", "build", "make", "forge", "create", "write", "paint"}

ALL_INTENTS = {
    "combat": INTENT_COMBAT,
    "social": INTENT_SOCIAL,
    "explore": INTENT_EXPLORE,
    "examine": INTENT_EXAMINE,
    "magic": INTENT_MAGIC,
    "create": INTENT_CREATE,
}


class ContextAssembler:
    """
    Builds the LLM context window from the StoryGraph, with full transparency.

    Usage:
        assembler = ContextAssembler(graph, memory_manager)
        assembly = assembler.assemble("I want to talk to Kael about the siege", budget=4096)
        # assembly.stages shows every decision
        # assembly.activated_nodes shows what was pulled in
        # assembly.pruned shows what was cut and why
    """

    def __init__(self, graph, memory_manager):
        """
        Args:
            graph: StoryGraph instance
            memory_manager: MemoryManager instance
        """
        self.graph = graph
        self.memory = memory_manager
        self._last_assembly: Optional[ContextAssembly] = None

    def assemble(
        self,
        user_input: str,
        token_budget: int = 4096,
        pin_nodes: Optional[List[str]] = None,
        exclude_nodes: Optional[List[str]] = None,
    ) -> ContextAssembly:
        """
        Run the 6-stage pipeline and return the full transparent assembly.
        """
        assembly = ContextAssembly(
            user_input=user_input,
            turn_number=self.graph.world.turn_number,
            token_budget=token_budget,
        )
        pin_nodes = set(pin_nodes or [])
        exclude_nodes = set(exclude_nodes or [])

        # Collect candidates through the pipeline
        candidates: Dict[str, Dict[str, Any]] = {}   # node_id → {node, score, reason, stage}
        memory_candidates: List[Dict[str, Any]] = []

        # ── Stage 1: EXTRACT ──
        stage1_start = time.monotonic()
        extracted_entities, detected_intents = self._stage_extract(user_input)
        stage1 = ContextStage(
            name="EXTRACT", icon="🔍",
            duration_ms=int((time.monotonic() - stage1_start) * 1000),
            items_in=0,
            items_out=len(extracted_entities) + len(detected_intents),
            details=[
                f"Found entity: \"{name}\" → {node.name} ({node.type.value})"
                for node, name in extracted_entities
            ] + [
                f"Detected intent: {intent}"
                for intent in detected_intents
            ],
        )
        if not extracted_entities and not detected_intents:
            stage1.details.append("No specific entities or intents detected in input")
        assembly.stages.append(stage1)

        # ── Stage 2: ACTIVATE ──
        stage2_start = time.monotonic()
        activated = self._stage_activate(extracted_entities, pin_nodes, exclude_nodes)
        for node_id, info in activated.items():
            candidates[node_id] = info
        stage2 = ContextStage(
            name="ACTIVATE", icon="⚡",
            duration_ms=int((time.monotonic() - stage2_start) * 1000),
            items_in=len(extracted_entities),
            items_out=len(activated),
            details=[
                f"Activated: {info['node'].name} ({info['reason']}, score={info['score']:.2f})"
                for info in activated.values()
            ],
        )
        assembly.stages.append(stage2)

        # ── Stage 3: EXPAND ──
        stage3_start = time.monotonic()
        expanded = self._stage_expand(list(candidates.keys()), exclude_nodes)
        for node_id, info in expanded.items():
            if node_id not in candidates:
                candidates[node_id] = info
        stage3 = ContextStage(
            name="EXPAND", icon="🌐",
            duration_ms=int((time.monotonic() - stage3_start) * 1000),
            items_in=len(candidates) - len(expanded),
            items_out=len(candidates),
            details=[
                f"Expanded: {info['node'].name} ({info['reason']}, score={info['score']:.2f})"
                for info in expanded.values()
            ],
        )
        if not expanded:
            stage3.details.append("No additional nodes found within 2 hops")
        assembly.stages.append(stage3)

        # ── Stage 4: RECALL ──
        stage4_start = time.monotonic()
        memory_candidates = self._stage_recall(
            list(candidates.keys()), user_input, detected_intents
        )
        stage4 = ContextStage(
            name="RECALL", icon="🧠",
            duration_ms=int((time.monotonic() - stage4_start) * 1000),
            items_in=0,
            items_out=len(memory_candidates),
            details=[
                f"Recalled: \"{m['summary'][:60]}\" ({m['reason']}, score={m['score']:.2f})"
                for m in memory_candidates
            ],
        )
        if not memory_candidates:
            stage4.details.append("No relevant memories found")
        assembly.stages.append(stage4)

        # ── Stage 4.5: RECURSIVE REFINEMENT ──
        # When context is sparse, expand queries using activated entity names
        if len(memory_candidates) < 3 and len(candidates) < 5:
            stage45_start = time.monotonic()
            extra_count = 0
            for node_id, info in list(candidates.items()):
                node_name = info["node"].name
                extra = self._stage_recall([node_id], node_name, [])
                for em in extra:
                    # Avoid duplicates
                    if not any(m["memory"].id == em["memory"].id for m in memory_candidates):
                        em["reason"] = f"recursive refinement via {node_name}"
                        em["score"] *= 0.9  # slight penalty for secondary recall
                        memory_candidates.append(em)
                        extra_count += 1
            if extra_count > 0:
                stage45 = ContextStage(
                    name="REFINE", icon="🔄",
                    duration_ms=int((time.monotonic() - stage45_start) * 1000),
                    items_in=len(candidates),
                    items_out=extra_count,
                    details=[
                        f"Sparse context detected — recursive refinement added {extra_count} memories",
                    ],
                )
                assembly.stages.append(stage45)

        # ── Stage 5: RANK ──
        stage5_start = time.monotonic()
        ranked_nodes, ranked_memories = self._stage_rank(candidates, memory_candidates)
        stage5 = ContextStage(
            name="RANK", icon="📊",
            duration_ms=int((time.monotonic() - stage5_start) * 1000),
            items_in=len(candidates) + len(memory_candidates),
            items_out=len(ranked_nodes) + len(ranked_memories),
            details=[
                f"#{i + 1}: {info['node'].name} (score={info['score']:.2f})"
                for i, (_, info) in enumerate(ranked_nodes[:10])
            ],
        )
        assembly.stages.append(stage5)

        # ── Stage 6: ASSEMBLE ──
        stage6_start = time.monotonic()
        context_text, sections, included_nodes, included_memories, pruned = (
            self._stage_assemble(ranked_nodes, ranked_memories, token_budget)
        )
        stage6 = ContextStage(
            name="ASSEMBLE", icon="📦",
            duration_ms=int((time.monotonic() - stage6_start) * 1000),
            items_in=len(ranked_nodes) + len(ranked_memories),
            items_out=len(included_nodes) + len(included_memories),
            details=[
                f"Included {len(included_nodes)} nodes, {len(included_memories)} memories",
                f"Pruned {len(pruned)} items",
                f"Token estimate: {len(context_text) // CHARS_PER_TOKEN}/{token_budget}",
            ],
        )
        assembly.stages.append(stage6)

        # Build the assembly output
        assembly.activated_nodes = included_nodes
        assembly.recalled_memories = included_memories
        assembly.pruned = pruned
        assembly.context_sections = sections
        assembly.context_text = context_text
        assembly.token_estimate = len(context_text) // CHARS_PER_TOKEN

        self._last_assembly = assembly
        return assembly

    @property
    def last_assembly(self) -> Optional[ContextAssembly]:
        return self._last_assembly

    # ========================================================================
    # STAGE 1: EXTRACT — Parse user input
    # ========================================================================

    def _stage_extract(
        self, text: str
    ) -> Tuple[List[Tuple[StoryNode, str]], List[str]]:
        """Extract entity references and intents from user text."""

        # Find named entities by matching against graph node names
        entities = self.graph.match_names_in_text(text)

        # Detect intents
        words = set(re.findall(r'\b\w+\b', text.lower()))
        intents = []
        for intent_name, keywords in ALL_INTENTS.items():
            if words & keywords:
                intents.append(intent_name)

        return entities, intents

    # ========================================================================
    # STAGE 2: ACTIVATE — Match to graph nodes
    # ========================================================================

    def _stage_activate(
        self,
        entities: List[Tuple[StoryNode, str]],
        pin_nodes: Set[str],
        exclude_nodes: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Activate graph nodes based on extracted entities + current scene state."""
        activated: Dict[str, Dict[str, Any]] = {}
        world = self.graph.world

        # Direct mentions from extraction
        for node, matched_name in entities:
            if node.id not in exclude_nodes:
                activated[node.id] = {
                    "node": node,
                    "score": DIRECT_MENTION_SCORE,
                    "reason": f"directly mentioned as \"{matched_name}\"",
                    "stage": "ACTIVATE",
                }

        # Current scene
        if world.current_scene and world.current_scene not in exclude_nodes:
            scene = self.graph.get_node(world.current_scene)
            if scene and scene.id not in activated:
                activated[scene.id] = {
                    "node": scene, "score": CURRENT_SCENE_SCORE,
                    "reason": "current scene", "stage": "ACTIVATE",
                }

        # Current location
        if world.current_location and world.current_location not in exclude_nodes:
            loc = self.graph.get_node(world.current_location)
            if loc and loc.id not in activated:
                activated[loc.id] = {
                    "node": loc, "score": CURRENT_LOCATION_SCORE,
                    "reason": "current location", "stage": "ACTIVATE",
                }

        # Present characters
        for char_id in world.present_characters:
            if char_id not in exclude_nodes and char_id not in activated:
                char = self.graph.get_node(char_id)
                if char:
                    activated[char_id] = {
                        "node": char, "score": PRESENT_CHARACTER_SCORE,
                        "reason": "present in scene", "stage": "ACTIVATE",
                    }

        # Active plot threads
        for thread_id in world.active_plot_threads:
            if thread_id not in exclude_nodes and thread_id not in activated:
                thread = self.graph.get_node(thread_id)
                if thread:
                    activated[thread_id] = {
                        "node": thread, "score": ACTIVE_PLOT_SCORE,
                        "reason": "active plot thread", "stage": "ACTIVATE",
                    }

        # Pinned nodes (user or system pinned)
        for node in self.graph.get_pinned_nodes():
            if node.id not in exclude_nodes and node.id not in activated:
                activated[node.id] = {
                    "node": node, "score": PINNED_SCORE,
                    "reason": "pinned", "stage": "ACTIVATE",
                }

        # Explicitly pinned by request
        for pid in pin_nodes:
            if pid not in exclude_nodes and pid not in activated:
                pnode = self.graph.get_node(pid)
                if pnode:
                    activated[pid] = {
                        "node": pnode, "score": PINNED_SCORE,
                        "reason": "pinned by user for this turn", "stage": "ACTIVATE",
                    }

        return activated

    # ========================================================================
    # STAGE 3: EXPAND — Walk edges
    # ========================================================================

    def _stage_expand(
        self,
        seed_ids: List[str],
        exclude_nodes: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Expand context by walking edges 1-2 hops from activated nodes."""
        expanded: Dict[str, Dict[str, Any]] = {}
        seed_set = set(seed_ids)

        reachable = self.graph.bfs(seed_ids, max_depth=2)
        for node, depth, reason in reachable:
            if node.id in seed_set or node.id in exclude_nodes:
                continue
            score = HOP_1_DECAY if depth == 1 else HOP_2_DECAY
            # Don't overwrite with lower score
            if node.id in expanded and expanded[node.id]["score"] >= score:
                continue
            expanded[node.id] = {
                "node": node,
                "score": score,
                "reason": reason,
                "stage": "EXPAND",
            }

        return expanded

    # ========================================================================
    # STAGE 4: RECALL — Query memories
    # ========================================================================

    def _stage_recall(
        self,
        activated_ids: List[str],
        user_text: str,
        intents: List[str],
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories from the memory store."""
        if not self.memory:
            return []

        recalled = self.memory.recall(
            related_node_ids=activated_ids,
            query_text=user_text,
            tags=intents,
            limit=15,
            current_turn=self.graph.world.turn_number,
        )

        return [
            {
                "memory": mem,
                "summary": mem.summary,
                "memory_type": mem.type.value,
                "score": MEMORY_BASE_SCORE * mem.effective_importance(
                    self.graph.world.turn_number
                ),
                "reason": reason,
            }
            for mem, reason in recalled
        ]

    # ========================================================================
    # STAGE 5: RANK — Score and sort
    # ========================================================================

    def _stage_rank(
        self,
        candidates: Dict[str, Dict[str, Any]],
        memory_candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[str, Dict]], List[Dict]]:
        """Rank all candidates by relevance score."""
        # Sort nodes by score descending
        ranked_nodes = sorted(
            candidates.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )

        # Sort memories by score descending
        ranked_memories = sorted(
            memory_candidates,
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked_nodes, ranked_memories

    # ========================================================================
    # STAGE 6: ASSEMBLE — Build the final context
    # ========================================================================

    def _stage_assemble(
        self,
        ranked_nodes: List[Tuple[str, Dict]],
        ranked_memories: List[Dict],
        token_budget: int,
    ) -> Tuple[str, Dict[str, str], List[ActivatedNode], List[RecalledMemory], List[PrunedItem]]:
        """
        Build the final context string from ranked candidates.
        Fits within token budget. Logs everything included and pruned.
        """
        included_nodes: List[ActivatedNode] = []
        included_memories: List[RecalledMemory] = []
        pruned: List[PrunedItem] = []

        # Context sections (ordered for the LLM)
        sections: Dict[str, List[str]] = {
            "world_state": [],
            "current_scene": [],
            "characters": [],
            "lore_and_rules": [],
            "plot_threads": [],
            "memories": [],
            "recent_events": [],
        }

        char_budget = token_budget * CHARS_PER_TOKEN
        chars_used = 0

        # Add world state header (always included, doesn't count nodes)
        world = self.graph.world
        world_header = (
            f"Story: {world.story_name} | Turn: {world.turn_number} | "
            f"Time: {world.time_of_day} | Mood: {world.mood}"
        )
        if world.last_turn_summary:
            world_header += f"\nPreviously: {world.last_turn_summary}"
        sections["world_state"].append(world_header)
        chars_used += len(world_header)

        # ── Prometheus: Inject living world simulation data ──
        if world.prometheus_active and world.prometheus_world_id:
            sim_parts = []
            if world.simulation_time:
                sim_parts.append(f"Simulation Time: {world.simulation_time}")
            if world.simulation_weather:
                sim_parts.append(f"Simulated Weather: {world.simulation_weather}")
            if world.last_sim_events:
                sim_parts.append("Recent world events:")
                for ev in world.last_sim_events[:5]:
                    sim_parts.append(f"  - {ev}")

            # Rich simulation context from cached state
            try:
                from saga_engine.simulation import get_simulation_client
                sim_client = get_simulation_client()
                cached = sim_client.last_state
                if cached and cached.world_id == world.prometheus_world_id:
                    # NPC emotional states for present characters
                    if cached.npc_states and world.present_characters:
                        npc_lines = []
                        for char_id in world.present_characters:
                            npc = cached.npc_states.get(char_id, {})
                            if not npc:
                                char_node = self.graph.get_node(char_id)
                                if char_node:
                                    npc = cached.npc_states.get(char_node.name, {})
                            if npc and npc.get("emotional_state"):
                                name = npc.get("name", char_id)
                                npc_lines.append(f"  {name}: {npc['emotional_state']}")
                        if npc_lines:
                            sim_parts.append("Character emotional states:")
                            sim_parts.extend(npc_lines)

                    # Faction tension levels
                    if cached.faction_states:
                        tense_factions = []
                        for fid, fdata in cached.faction_states.items():
                            tension = fdata.get("tension", 0)
                            if tension > 60:
                                fname = fdata.get("name", fid)
                                tense_factions.append(f"  {fname}: tension {tension}%")
                        if tense_factions:
                            sim_parts.append("Faction tensions (high):")
                            sim_parts.extend(tense_factions)

                    # Economic conditions
                    if cached.economy:
                        econ_parts = []
                        if cached.economy.get("inflation"):
                            econ_parts.append(f"inflation={cached.economy['inflation']}")
                        if cached.economy.get("market_mood"):
                            econ_parts.append(f"market={cached.economy['market_mood']}")
                        if econ_parts:
                            sim_parts.append(f"Economy: {', '.join(econ_parts)}")

                    # AWI metrics — flag declining indicators
                    if cached.awi_metrics:
                        declining = [
                            f"{k}: {v:.0f}%"
                            for k, v in cached.awi_metrics.items()
                            if v < 30
                        ]
                        if declining:
                            sim_parts.append(f"World health warnings: {', '.join(declining)}")
            except Exception:
                pass  # Simulation module may not be available

            if sim_parts:
                sim_block = "\n".join(sim_parts)
                sections["world_state"].append(sim_block)
                chars_used += len(sim_block)

        # Fit nodes into budget
        for node_id, info in ranked_nodes:
            node: StoryNode = info["node"]
            line = node.context_line()
            line_chars = len(line) + 2  # newline overhead

            if info["score"] < PRUNING_THRESHOLD:
                pruned.append(PrunedItem(
                    item_id=node_id, item_name=node.name,
                    item_type="node",
                    reason=f"below threshold ({info['score']:.2f} < {PRUNING_THRESHOLD})",
                    relevance=info["score"],
                ))
                continue

            if chars_used + line_chars > char_budget:
                pruned.append(PrunedItem(
                    item_id=node_id, item_name=node.name,
                    item_type="node",
                    reason="token budget exceeded",
                    relevance=info["score"],
                ))
                continue

            # Add to appropriate section
            section = self._section_for_node(node)
            sections[section].append(line)
            chars_used += line_chars

            included_nodes.append(ActivatedNode(
                node_id=node_id,
                node_name=node.name,
                node_type=node.type.value,
                reason=info["reason"],
                relevance=info["score"],
                source_stage=info["stage"],
                included=True,
            ))

        # Fit memories into remaining budget
        for minfo in ranked_memories:
            mem = minfo["memory"]
            line = f"[{mem.type.value}] {mem.summary}"
            line_chars = len(line) + 2

            if minfo["score"] < PRUNING_THRESHOLD:
                pruned.append(PrunedItem(
                    item_id=mem.id, item_name=mem.summary[:40],
                    item_type="memory",
                    reason=f"below threshold ({minfo['score']:.2f})",
                    relevance=minfo["score"],
                ))
                continue

            if chars_used + line_chars > char_budget:
                pruned.append(PrunedItem(
                    item_id=mem.id, item_name=mem.summary[:40],
                    item_type="memory",
                    reason="token budget exceeded",
                    relevance=minfo["score"],
                ))
                continue

            sections["memories"].append(line)
            chars_used += line_chars

            included_memories.append(RecalledMemory(
                memory_id=mem.id,
                summary=mem.summary,
                memory_type=mem.type.value,
                reason=minfo["reason"],
                relevance=minfo["score"],
                included=True,
            ))

            # Mark as accessed in memory store
            if self.memory:
                self.memory.mark_accessed(mem.id)

        # Build the final context text
        context_parts = []
        section_labels = {
            "world_state": "=== WORLD STATE ===",
            "current_scene": "=== CURRENT SCENE ===",
            "characters": "=== CHARACTERS PRESENT ===",
            "lore_and_rules": "=== LORE & RULES ===",
            "plot_threads": "=== ACTIVE PLOT THREADS ===",
            "memories": "=== MEMORIES ===",
            "recent_events": "=== RECENT EVENTS ===",
        }
        for section_key, label in section_labels.items():
            lines = sections[section_key]
            if lines:
                context_parts.append(label)
                context_parts.extend(lines)

        context_text = "\n".join(context_parts)

        # Convert sections dict to str→str for the model
        flat_sections = {k: "\n".join(v) for k, v in sections.items() if v}

        return context_text, flat_sections, included_nodes, included_memories, pruned

    def _section_for_node(self, node: StoryNode) -> str:
        """Determine which context section a node belongs in."""
        mapping = {
            NodeType.CHARACTER: "characters",
            NodeType.LOCATION: "current_scene",
            NodeType.ITEM: "current_scene",
            NodeType.FACTION: "lore_and_rules",
            NodeType.EVENT: "recent_events",
            NodeType.LORE: "lore_and_rules",
            NodeType.PLOT_THREAD: "plot_threads",
            NodeType.SCENE: "current_scene",
        }
        return mapping.get(node.type, "lore_and_rules")

    # ========================================================================
    # PER-CHARACTER CONTEXT — Character-POV filtered assembly
    # ========================================================================

    def assemble_for_character(
        self,
        character_id: str,
        scene_prompt: str,
        other_characters: list,
        token_budget: int = 2048,
    ) -> ContextAssembly:
        """Build context from a specific character's point of view.

        Wraps the 6-stage pipeline with character-POV filtering:
        1. Run full assembly with the character pinned
        2. Filter nodes to character's knowledge graph (2-hop)
        3. Filter memories to character-related ones
        4. Inject character's private data (personality, backstory, goals)
        5. Inject relationship info for other present characters
        """
        # Run standard pipeline with character pinned
        assembly = self.assemble(
            scene_prompt,
            token_budget=token_budget,
            pin_nodes=[character_id],
        )

        # Get character's knowledge scope (nodes within 2 hops)
        knowledge_edge_types = [
            EdgeType.KNOWS, EdgeType.LOCATED_AT, EdgeType.MEMBER_OF,
            EdgeType.PARTICIPATED_IN, EdgeType.OWNS, EdgeType.SEEKS,
        ]
        reachable = self.graph.bfs(
            start_ids=[character_id],
            max_depth=2,
            edge_types=knowledge_edge_types,
        )
        reachable_ids = {character_id} | {n.id for n, _, _ in reachable}

        # Filter activated nodes to character's knowledge
        for node in assembly.activated_nodes:
            if node.node_id not in reachable_ids:
                node.included = False

        # Filter memories to character-related ones
        for mem in assembly.recalled_memories:
            mem_obj = self.memory.get(mem.memory_id)
            if mem_obj and not (set(mem_obj.related_nodes) & reachable_ids):
                mem.included = False

        # Get character node data
        char_node = self.graph.get_node(character_id)
        private_lines = []
        if char_node:
            props = char_node.properties
            if props.get("personality"):
                private_lines.append(f"Your personality: {props['personality']}")
            if props.get("backstory"):
                private_lines.append(f"Your backstory: {props['backstory']}")
            if props.get("goals"):
                private_lines.append(f"Your current goals: {props['goals']}")

        # Inject relationship context for other present characters
        rel_lines = []
        for other_id in other_characters:
            if other_id == character_id:
                continue
            other_node = self.graph.get_node(other_id)
            if not other_node:
                continue
            for neighbor, edge in self.graph.get_neighbors(character_id):
                if neighbor.id == other_id:
                    rel_lines.append(
                        f"Your relationship with {other_node.name}: {edge.label or edge.type.value}"
                    )
                    break

        # Rebuild context text with private data prepended
        extra_sections = []
        if private_lines:
            extra_sections.append("=== YOUR CHARACTER ===")
            extra_sections.extend(private_lines)
        if rel_lines:
            extra_sections.append("=== YOUR RELATIONSHIPS ===")
            extra_sections.extend(rel_lines)

        if extra_sections:
            assembly.context_text = "\n".join(extra_sections) + "\n\n" + assembly.context_text
            assembly.token_estimate = len(assembly.context_text) // CHARS_PER_TOKEN

        return assembly
