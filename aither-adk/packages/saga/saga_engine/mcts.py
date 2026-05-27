"""
Narrative MCTS — Monte Carlo Tree Search for Story Branching
==============================================================

Explores multiple possible story continuations, scores them on narrative
quality dimensions, and selects the best path. All decisions are transparent
— the full exploration tree is returned for the frontend to visualize.

Modes:
    lightweight — 3 candidates, depth 1 (best-of-N). 3 LLM calls.
    standard   — 5 candidates, depth 1. 5 LLM calls. Good balance.
    deep       — 3 candidates/node, depth 2, 4 iterations. ~12 LLM calls.

Scoring dimensions (heuristic, no extra LLM calls):
    coherence             — references known entities from assembled context
    character_consistency — uses known characters correctly, no hallucinated names
    tension               — action verbs, emotional language, conflict indicators
    player_agency         — directly addresses the player's stated action
    novelty               — diversity vs sibling candidates (Jaccard distance)
    prose_quality         — sentence variety, vocabulary richness, balance

Usage:
    mcts = NarrativeMCTS(graph, memory, assembler)
    exploration = await mcts.search(user_input, assembly, generate_fn)
    # exploration.selected_text   — the best continuation
    # exploration.alternatives    — runner-ups with scores
    # exploration.tree            — full tree for visualization

The generate_fn is decoupled from the LLM backend:
    async def generate_fn(prompt: str, n: int) -> List[str]

With vLLM's prefix caching, N completions from the same prompt are efficient
(the context is processed once, only completions diverge).
"""

import logging
import math
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from .models import ContextAssembly, NodeType, _make_id, _now

logger = logging.getLogger("Saga.MCTS")

# Type alias for the LLM generation function
GenerateFn = Callable[[str, int], Awaitable[List[str]]]


# ============================================================================
# CONFIGURATION
# ============================================================================


class MCTSConfig(BaseModel):
    """
    Search parameters. Use the class methods for sensible presets.
    """

    mode: Literal["lightweight", "standard", "deep"] = "standard"
    candidates_per_expansion: int = 5
    max_depth: int = 1
    max_iterations: int = 1
    exploration_constant: float = 1.414  # UCB1 C parameter (sqrt(2))
    temperature: float = 0.9
    min_score_threshold: float = 0.2
    scoring_weights: Dict[str, float] = Field(default_factory=lambda: {
        "coherence": 0.25,
        "character_consistency": 0.20,
        "tension": 0.15,
        "player_agency": 0.20,
        "novelty": 0.10,
        "prose_quality": 0.10,
    })

    @classmethod
    def lightweight(cls) -> "MCTSConfig":
        """3 candidates, 1 iteration, depth 1. Fastest — 3 LLM calls."""
        return cls(
            mode="lightweight",
            candidates_per_expansion=3,
            max_iterations=1,
            max_depth=1,
            temperature=0.85,
        )

    @classmethod
    def standard(cls) -> "MCTSConfig":
        """5 candidates, 1 iteration, depth 1. Good balance — 5 LLM calls."""
        return cls(
            mode="standard",
            candidates_per_expansion=5,
            max_iterations=1,
            max_depth=1,
            temperature=0.9,
        )

    @classmethod
    def deep(cls) -> "MCTSConfig":
        """3 candidates/node, depth 2, 4 iterations. Full tree — ~12 LLM calls."""
        return cls(
            mode="deep",
            candidates_per_expansion=3,
            max_iterations=4,
            max_depth=2,
            exploration_constant=1.8,  # More exploration at depth
            temperature=0.95,
        )


# ============================================================================
# SCORING CONTEXT (built from ContextAssembly)
# ============================================================================


class ScoringContext(BaseModel):
    """Everything the scorer needs to evaluate a candidate continuation."""

    user_input: str = ""
    character_names: Set[str] = Field(default_factory=set)
    location_names: Set[str] = Field(default_factory=set)
    active_themes: List[str] = Field(default_factory=list)
    recent_text: str = ""
    world_mood: str = ""


# ============================================================================
# MCTS TREE NODE (not a StoryGraph node — this is a search-tree node)
# ============================================================================


class MCTSNode:
    """
    One node in the MCTS search tree.

    - depth 0 = root (current story state, no generated text)
    - depth 1 = candidate responses to the player's input
    - depth 2+ = lookahead continuations (deep mode only)
    """

    __slots__ = (
        "id", "text", "prompt", "depth", "scores", "total_score",
        "visits", "value_sum", "parent", "children", "expanded",
    )

    def __init__(
        self,
        text: str = "",
        depth: int = 0,
        prompt: str = "",
        parent: Optional["MCTSNode"] = None,
    ):
        self.id: str = _make_id("mcts")
        self.text = text
        self.prompt = prompt
        self.depth = depth
        self.scores: Dict[str, float] = {}
        self.total_score: float = 0.0
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.expanded: bool = False

    @property
    def avg_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits: int, c: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.avg_value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def add_child(self, child: "MCTSNode"):
        child.parent = self
        self.children.append(child)


# ============================================================================
# NARRATIVE SCORER — Heuristic multi-dimensional evaluation
# ============================================================================


# Word sets for tension and emotion detection
_TENSION_WORDS = frozenset({
    "fight", "battle", "clash", "struggle", "attack", "defend", "strike",
    "danger", "threat", "shadow", "darkness", "blood", "wound", "scream",
    "chase", "flee", "trap", "ambush", "siege", "storm", "shatter",
    "betray", "reveal", "secret", "forbid", "curse", "doom", "dread",
    "confront", "challenge", "defy", "surge", "erupt", "collapse",
    "desperate", "urgent", "peril", "ruin", "destroy", "sacrifice",
})

_EMOTION_WORDS = frozenset({
    "fear", "hope", "anger", "joy", "grief", "love", "hatred", "sorrow",
    "awe", "wonder", "dread", "peace", "rage", "terror", "longing",
    "despair", "triumph", "shame", "pride", "compassion", "fury",
    "tenderness", "resolve", "doubt", "courage", "regret", "yearning",
    "warmth", "chill", "ache", "bliss", "agony", "relief", "anxiety",
})

_DIALOGUE_PATTERN = re.compile(r'"[^"]{3,}"')
_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+")


class NarrativeScorer:
    """
    Scores a candidate continuation on multiple narrative dimensions.
    Pure heuristic — no LLM calls. Fast enough to score dozens of candidates.
    """

    def score(self, text: str, ctx: ScoringContext) -> Dict[str, float]:
        """Score a single candidate on all dimensions (except novelty)."""
        return {
            "coherence": self._coherence(text, ctx),
            "character_consistency": self._character_consistency(text, ctx.character_names),
            "tension": self._tension(text),
            "player_agency": self._player_agency(text, ctx.user_input),
            "prose_quality": self._prose_quality(text),
        }

    def novelty_scores(self, texts: List[str]) -> List[float]:
        """
        Score each text's uniqueness relative to its siblings.
        Called AFTER all candidates are generated so we can compare.
        """
        if len(texts) <= 1:
            return [0.7] * len(texts)

        word_sets = [set(t.lower().split()) for t in texts]
        scores = []
        for i, ws in enumerate(word_sets):
            similarities = []
            for j, other in enumerate(word_sets):
                if i != j:
                    intersection = len(ws & other)
                    union = len(ws | other)
                    jaccard = intersection / union if union > 0 else 0.0
                    similarities.append(jaccard)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            scores.append(max(0.0, min(1.0, 1.0 - avg_sim)))
        return scores

    @staticmethod
    def weighted_total(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted sum of dimension scores."""
        total = 0.0
        weight_sum = 0.0
        for dim, score in scores.items():
            w = weights.get(dim, 0.0)
            total += score * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0

    # ── Dimension scorers ──

    def _coherence(self, text: str, ctx: ScoringContext) -> float:
        """Does the text reference known entities from the assembled context?"""
        if not text:
            return 0.0

        text_lower = text.lower()
        all_names = ctx.character_names | ctx.location_names
        if not all_names:
            return 0.5  # Neutral if no context entities

        mentioned = sum(1 for name in all_names if name in text_lower)
        ratio = mentioned / len(all_names)

        # Also check if response references recent context themes
        theme_bonus = 0.0
        for theme in ctx.active_themes[:5]:
            if theme.lower() in text_lower:
                theme_bonus += 0.05

        return min(1.0, 0.3 + ratio * 0.5 + theme_bonus)

    def _character_consistency(self, text: str, known_chars: Set[str]) -> float:
        """Does the text use known characters correctly?"""
        if not text:
            return 0.0
        if not known_chars:
            return 0.5

        text_lower = text.lower()
        mentioned = sum(1 for name in known_chars if name in text_lower)
        ratio = mentioned / len(known_chars)

        # At least one character should be doing something
        base = 0.3 if mentioned > 0 else 0.1
        return min(1.0, base + ratio * 0.7)

    def _tension(self, text: str) -> float:
        """Does the text maintain narrative tension?"""
        if not text:
            return 0.0

        words = set(text.lower().split())
        tension_hits = len(words & _TENSION_WORDS)
        emotion_hits = len(words & _EMOTION_WORDS)
        word_count = len(text.split())

        if word_count == 0:
            return 0.0

        # Tension density (not too low, not too high)
        density = (tension_hits + emotion_hits) / word_count
        # Sweet spot: 2-8% tension/emotion words
        if density < 0.01:
            return 0.2  # Too flat
        elif density < 0.03:
            return 0.5 + density * 10
        elif density <= 0.08:
            return 0.8  # Sweet spot
        else:
            return max(0.4, 0.8 - (density - 0.08) * 5)  # Melodramatic

    def _player_agency(self, text: str, user_input: str) -> float:
        """Does the response address what the player actually said?"""
        if not text or not user_input:
            return 0.3

        user_words = set(user_input.lower().split())
        # Remove very common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "i", "to",
                     "and", "of", "in", "it", "my", "me", "at", "on", "for"}
        user_keywords = user_words - stopwords
        if not user_keywords:
            return 0.5

        text_lower = text.lower()
        matched = sum(1 for w in user_keywords if w in text_lower)
        ratio = matched / len(user_keywords)

        # Higher score if the response starts by addressing the player's action
        first_100 = text_lower[:min(200, len(text_lower))]
        early_match = sum(1 for w in user_keywords if w in first_100)
        early_bonus = min(0.2, early_match * 0.05)

        return min(1.0, 0.2 + ratio * 0.6 + early_bonus)

    def _prose_quality(self, text: str) -> float:
        """Basic prose quality: sentence variety, vocabulary, balance."""
        if not text or len(text) < 20:
            return 0.1

        # Sentence length variety
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) < 2:
            return 0.3

        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        if avg_len == 0:
            return 0.2

        # Standard deviation of sentence lengths (variety is good)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        variety_score = min(1.0, std_dev / 8.0)  # 8 words std dev is good variety

        # Vocabulary richness (unique words / total words)
        all_words = text.lower().split()
        unique_ratio = len(set(all_words)) / len(all_words) if all_words else 0
        vocab_score = min(1.0, unique_ratio * 1.5)  # ~67% unique is 1.0

        # Dialogue balance (some dialogue is good for narrative)
        dialogue_count = len(_DIALOGUE_PATTERN.findall(text))
        dialogue_score = 0.5  # Neutral base
        if 1 <= dialogue_count <= 5:
            dialogue_score = 0.8  # Good mix
        elif dialogue_count > 5:
            dialogue_score = 0.5  # Too talky

        # Length check (too short or too long is bad)
        word_count = len(all_words)
        length_score = 0.5
        if 60 <= word_count <= 400:
            length_score = 0.9  # Sweet spot
        elif 30 <= word_count < 60:
            length_score = 0.6
        elif word_count > 400:
            length_score = 0.6

        return (variety_score * 0.3 + vocab_score * 0.3 +
                dialogue_score * 0.2 + length_score * 0.2)


# ============================================================================
# GLASS-BOX OUTPUT MODELS
# ============================================================================


class MCTSBranch(BaseModel):
    """One explored branch — for tree visualization."""

    node_id: str
    text_preview: str = ""
    full_text: str = ""
    scores: Dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    visits: int = 0
    depth: int = 0
    selected: bool = False
    children: List["MCTSBranch"] = Field(default_factory=list)


class MCTSExploration(BaseModel):
    """
    Full glass-box view of an MCTS search.
    The frontend renders this as a branching tree with scores.
    """

    mode: str = "standard"
    config_used: Dict[str, Any] = Field(default_factory=dict)
    iterations_run: int = 0
    total_candidates: int = 0
    max_depth_reached: int = 0
    duration_ms: float = 0.0

    # The winner
    selected_text: str = ""
    selected_scores: Dict[str, float] = Field(default_factory=dict)
    selected_total: float = 0.0

    # Alternatives (top N, sorted by score)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)

    # Full tree for visualization
    tree: Optional[MCTSBranch] = None

    # Scoring info
    scoring_weights: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_now)


# ============================================================================
# THE ENGINE
# ============================================================================


STORY_PROMPT_TEMPLATE = """\
Continue the story.

{context}

The player says: {user_input}

Write 2-4 paragraphs of vivid narrative that responds to the player's action, \
maintains character voices, respects world rules, and ends at an interesting \
moment or choice. Write ONLY the narrative continuation."""

DEPTH2_PROMPT_TEMPLATE = """\
Continue the story.

{context}

The player said: {user_input}

The story continued:
{previous_continuation}

The player continues forward.

Write 2-4 paragraphs of what happens next. Maintain continuity with the \
previous scene. Write ONLY the narrative continuation."""


class NarrativeMCTS:
    """
    Monte Carlo Tree Search for story branching.

    At depth=1, this is efficient best-of-N with narrative scoring.
    At depth=2+, it's full tree search with lookahead.
    All decisions are transparent — the full tree is returned.
    """

    # When True, depth-1 searches delegate to UnifiedMCTS via NarrativeEnv
    _use_unified_engine: bool = True

    def __init__(
        self,
        graph,           # StoryGraph instance (for entity extraction)
        memory,          # MemoryManager instance
        assembler,       # ContextAssembler instance
        config: Optional[MCTSConfig] = None,
    ):
        self.graph = graph
        self.memory = memory
        self.assembler = assembler
        self.config = config or MCTSConfig.standard()
        self.scorer = NarrativeScorer()
        self.last_exploration: Optional[MCTSExploration] = None
        self.generate_fn: Optional[GenerateFn] = None  # Set by service at startup

    def update_config(self, config: MCTSConfig):
        """Update search config (e.g., switch from lightweight to deep)."""
        self.config = config
        logger.info(f"MCTS config updated: mode={config.mode}")

    async def search(
        self,
        user_input: str,
        context_assembly: ContextAssembly,
        generate_fn: GenerateFn,
        config_override: Optional[MCTSConfig] = None,
    ) -> MCTSExploration:
        """
        Run MCTS search and return the full exploration.

        Args:
            user_input:        The player's message
            context_assembly:  Output from ContextAssembler (includes context_text)
            generate_fn:       async (prompt, n) -> List[str]
            config_override:   Optional one-time config override

        Returns:
            MCTSExploration with selected text, alternatives, and full tree.
        """
        cfg = config_override or self.config
        t0 = time.perf_counter()

        # Build scoring context from assembly
        scoring_ctx = self._build_scoring_context(user_input, context_assembly)

        # ── UnifiedMCTS delegation (Gap 13) ──
        # For depth-1 searches, optionally delegate to UnifiedMCTS via NarrativeEnv.
        # This lets the unified engine handle scoring and selection while we handle
        # candidate generation via the LLM.
        if self._use_unified_engine and cfg.max_depth == 1:
            try:
                exploration = await self._search_via_unified(
                    user_input, context_assembly, generate_fn, cfg, scoring_ctx,
                )
                if exploration:
                    self.last_exploration = exploration
                    return exploration
            except Exception as e:
                logger.debug(f"UnifiedMCTS delegation failed, using native: {e}")

        # Build the base prompt
        base_prompt = STORY_PROMPT_TEMPLATE.format(
            context=context_assembly.context_text,
            user_input=user_input,
        )

        # Create root node
        root = MCTSNode(text="[current story state]", depth=0, prompt="")

        if cfg.max_depth == 1:
            # ── BEST-OF-N MODE ── (practical for single GPU)
            await self._expand_node(root, base_prompt, cfg, generate_fn, scoring_ctx)
        else:
            # ── FULL MCTS MODE ── (tree search with lookahead)
            for iteration in range(cfg.max_iterations):
                leaf = self._select(root, cfg.exploration_constant)

                if leaf.depth >= cfg.max_depth:
                    # At max depth — just backpropagate existing value
                    self._backpropagate(leaf, leaf.total_score)
                    continue

                # Build prompt for this depth
                if leaf.depth == 0:
                    prompt = base_prompt
                else:
                    # Depth 2+ prompt includes the path so far
                    prompt = DEPTH2_PROMPT_TEMPLATE.format(
                        context=context_assembly.context_text,
                        user_input=user_input,
                        previous_continuation=leaf.text,
                    )

                await self._expand_node(leaf, prompt, cfg, generate_fn, scoring_ctx)

                # Backpropagate best child's value
                if leaf.children:
                    best_val = max(c.total_score for c in leaf.children)
                    self._backpropagate(leaf, best_val)

        # Build the glass-box exploration result
        duration_ms = (time.perf_counter() - t0) * 1000
        exploration = self._build_exploration(root, cfg, duration_ms)
        self.last_exploration = exploration

        logger.info(
            f"MCTS {cfg.mode}: {exploration.total_candidates} candidates, "
            f"best={exploration.selected_total:.3f}, "
            f"{duration_ms:.0f}ms"
        )
        return exploration

    # ── UnifiedMCTS delegation ──

    async def _search_via_unified(
        self,
        user_input: str,
        context_assembly: ContextAssembly,
        generate_fn: GenerateFn,
        cfg: MCTSConfig,
        scoring_ctx: ScoringContext,
    ) -> Optional[MCTSExploration]:
        """Delegate depth-1 search to UnifiedMCTS via NarrativeEnv (Gap 13).

        Generates candidates via the LLM first, then uses UnifiedMCTS +
        NarrativeEnv to score and select the best one.  Returns None if
        the unified engine is unavailable.
        """
        from lib.cognitive.UnifiedMCTS import (
            UnifiedMCTS,
            MCTSConfig as UnifiedConfig,
        )
        from lib.cognitive.environments.narrative import (
            NarrativeEnv,
            NarrativeScoringContext,
        )

        # Generate candidates using the LLM (same as native path)
        base_prompt = STORY_PROMPT_TEMPLATE.format(
            context=context_assembly.context_text,
            user_input=user_input,
        )
        try:
            texts = await generate_fn(base_prompt, cfg.candidates_per_expansion)
        except Exception as e:
            logger.warning(f"UnifiedMCTS candidate generation failed: {e}")
            return None

        texts = [t.strip() for t in texts if t and len(t.strip()) > 20]
        if not texts:
            return None

        # Build NarrativeEnv for unified search
        nav_ctx = NarrativeScoringContext(
            user_input=scoring_ctx.user_input,
            character_names=scoring_ctx.character_names,
            location_names=scoring_ctx.location_names,
            active_themes=scoring_ctx.active_themes,
            recent_text=scoring_ctx.recent_text,
            world_mood=scoring_ctx.world_mood,
        )
        env = NarrativeEnv(
            candidates=texts,
            scoring_context=nav_ctx,
            scoring_weights=dict(cfg.scoring_weights),
        )
        engine = UnifiedMCTS(UnifiedConfig(
            iterations=max(len(texts) * 3, 20),
            time_limit=1.0,
        ))
        result = await engine.search(env)

        if result.best_action is None:
            return None

        # Convert to MCTSExploration for compatibility with existing Saga API
        best_idx = result.best_action.index
        best_text = texts[best_idx]

        # Score all texts for the exploration result
        scored = []
        for i, text in enumerate(texts):
            score_val = env._score_candidate(i)
            scored.append((i, text, score_val))
        scored.sort(key=lambda x: x[2], reverse=True)

        alternatives = []
        for idx, text, score_val in scored:
            if idx == best_idx:
                continue
            alternatives.append({
                "text": text,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "scores": {},
                "total_score": round(score_val, 4),
                "visits": 1,
            })

        duration_ms = (time.perf_counter() - (result.elapsed_seconds or 0)) * 1000

        return MCTSExploration(
            mode=cfg.mode,
            config_used={
                "candidates_per_expansion": cfg.candidates_per_expansion,
                "max_iterations": cfg.max_iterations,
                "max_depth": cfg.max_depth,
                "exploration_constant": cfg.exploration_constant,
                "temperature": cfg.temperature,
                "engine": "unified_mcts",
            },
            iterations_run=result.iterations_used,
            total_candidates=len(texts),
            max_depth_reached=1,
            duration_ms=round(result.elapsed_seconds * 1000, 1),
            selected_text=best_text,
            selected_scores={},
            selected_total=round(result.best_value, 4),
            alternatives=alternatives[:5],
            tree=MCTSBranch(
                node_id="unified_root",
                text_preview="[unified MCTS root]",
                full_text="",
                scores={},
                total_score=0.0,
                visits=result.iterations_used,
                depth=0,
                selected=False,
                children=[],
            ),
            scoring_weights=dict(cfg.scoring_weights),
        )

    # ── Internal methods ──

    def _build_scoring_context(
        self,
        user_input: str,
        assembly: ContextAssembly,
    ) -> ScoringContext:
        """Extract entity names and themes from the context assembly."""
        char_names: Set[str] = set()
        loc_names: Set[str] = set()

        for an in assembly.activated_nodes:
            if not an.included:
                continue
            node = self.graph.get_node(an.node_id)
            if node is None:
                continue
            names = {node.name.lower()}
            names.update(a.lower() for a in node.aliases)

            if node.type == NodeType.CHARACTER:
                char_names.update(names)
            elif node.type == NodeType.LOCATION:
                loc_names.update(names)

        # Active themes from plot threads and tags
        themes = []
        for an in assembly.activated_nodes:
            node = self.graph.get_node(an.node_id)
            if node and node.type == NodeType.PLOT_THREAD:
                themes.append(node.name.lower())
                themes.extend(t.lower() for t in node.tags)

        return ScoringContext(
            user_input=user_input,
            character_names=char_names,
            location_names=loc_names,
            active_themes=themes[:10],
            recent_text=assembly.context_text[-500:] if assembly.context_text else "",
            world_mood=self.graph.world.mood or "",
        )

    async def _expand_node(
        self,
        node: MCTSNode,
        prompt: str,
        cfg: MCTSConfig,
        generate_fn: GenerateFn,
        scoring_ctx: ScoringContext,
    ):
        """Generate N candidates from this node and score them."""
        try:
            texts = await generate_fn(prompt, cfg.candidates_per_expansion)
        except Exception as e:
            logger.error(f"MCTS generate_fn failed: {e}")
            texts = []

        # Filter empty/too-short responses
        texts = [t.strip() for t in texts if t and len(t.strip()) > 20]
        if not texts:
            logger.warning("MCTS: no valid candidates generated")
            return

        # Score each candidate on 5 dimensions (no novelty yet)
        children = []
        for text in texts:
            child = MCTSNode(
                text=text,
                depth=node.depth + 1,
                prompt=prompt,
                parent=node,
            )
            child.scores = self.scorer.score(text, scoring_ctx)
            children.append(child)

        # Compute novelty scores across all siblings
        novelty_scores = self.scorer.novelty_scores(texts)

        # Combine scores and add children to node
        for i, child in enumerate(children):
            child.scores["novelty"] = novelty_scores[i]
            child.total_score = self.scorer.weighted_total(
                child.scores, cfg.scoring_weights
            )
            child.visits = 1
            child.value_sum = child.total_score
            node.add_child(child)

        node.expanded = True

    def _select(self, root: MCTSNode, c: float) -> MCTSNode:
        """UCB1 tree descent to find the best leaf to expand."""
        node = root
        while node.expanded and node.children:
            # Pick child with highest UCB1
            parent_visits = max(1, node.visits)
            node = max(node.children, key=lambda ch: ch.ucb1(parent_visits, c))
        return node

    def _backpropagate(self, node: MCTSNode, value: float):
        """Walk up the tree updating visit counts and value sums."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value_sum += value
            current = current.parent

    def _build_exploration(
        self,
        root: MCTSNode,
        cfg: MCTSConfig,
        duration_ms: float,
    ) -> MCTSExploration:
        """Convert the MCTS tree into a glass-box MCTSExploration."""

        # Collect all leaf candidates (depth=1 children of root for depth-1 search)
        all_candidates = self._collect_leaves(root)
        all_candidates.sort(key=lambda n: n.total_score, reverse=True)

        # Find max depth reached
        max_depth = max((n.depth for n in all_candidates), default=0)

        # Build the winner
        winner = all_candidates[0] if all_candidates else None

        # Build alternatives (top 5 runner-ups)
        alternatives = []
        for node in all_candidates[1:6]:
            alternatives.append({
                "text": node.text,
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "scores": dict(node.scores),
                "total_score": round(node.total_score, 4),
                "visits": node.visits,
            })

        # Build tree for visualization
        tree = self._node_to_branch(root, winner.id if winner else "")

        return MCTSExploration(
            mode=cfg.mode,
            config_used={
                "candidates_per_expansion": cfg.candidates_per_expansion,
                "max_iterations": cfg.max_iterations,
                "max_depth": cfg.max_depth,
                "exploration_constant": cfg.exploration_constant,
                "temperature": cfg.temperature,
            },
            iterations_run=cfg.max_iterations,
            total_candidates=len(all_candidates),
            max_depth_reached=max_depth,
            duration_ms=round(duration_ms, 1),
            selected_text=winner.text if winner else "",
            selected_scores={k: round(v, 4) for k, v in winner.scores.items()} if winner else {},
            selected_total=round(winner.total_score, 4) if winner else 0.0,
            alternatives=alternatives,
            tree=tree,
            scoring_weights=dict(cfg.scoring_weights),
        )

    def _collect_leaves(self, node: MCTSNode) -> List[MCTSNode]:
        """Collect all scored leaf nodes in the tree."""
        leaves: List[MCTSNode] = []
        if node.depth > 0 and node.text:
            leaves.append(node)
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _node_to_branch(self, node: MCTSNode, selected_id: str) -> MCTSBranch:
        """Recursively convert MCTSNode tree to MCTSBranch for serialization."""
        return MCTSBranch(
            node_id=node.id,
            text_preview=node.text[:200] + "..." if len(node.text) > 200 else node.text,
            full_text=node.text,
            scores={k: round(v, 4) for k, v in node.scores.items()},
            total_score=round(node.total_score, 4),
            visits=node.visits,
            depth=node.depth,
            selected=(node.id == selected_id),
            children=[
                self._node_to_branch(child, selected_id)
                for child in node.children
            ],
        )
