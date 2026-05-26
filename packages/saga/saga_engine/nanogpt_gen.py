"""NanoGPT Asset Generator — Procedural name/item/quest generation.

Uses adk.nanogpt.NanoGPT for character-level text generation across
6 domains: names, items, quests, rumors, visual profiles, flavor text.

Lazy-trains on first use from StoryGraph data + built-in corpus.
Models persist to disk and only retrain when corpus hash changes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("saga.nanogpt_gen")

# Built-in fantasy name corpus for bootstrap training
BUILTIN_NAMES = [
    "Aldric", "Brynn", "Caelum", "Dahlia", "Elara", "Fenris", "Gael", "Helena",
    "Idris", "Juniper", "Kael", "Lysandra", "Magnus", "Nyx", "Orion", "Petra",
    "Quinn", "Rowan", "Sable", "Theron", "Ursa", "Vesper", "Wren", "Xander",
    "Yvaine", "Zephyr", "Ashwin", "Beira", "Caspian", "Darian", "Elowen",
    "Florian", "Gareth", "Haldor", "Isolde", "Jasper", "Kieran", "Liora",
    "Maren", "Niall", "Ondine", "Priya", "Ragnar", "Selene", "Tavian",
    "Ulric", "Vivienne", "Wynne", "Ximena", "Yara", "Zan", "Alaric",
    "Basira", "Corvin", "Delphine", "Erasmus", "Freya", "Gideon", "Helios",
    "Ingrid", "Jareth", "Kira", "Leander", "Mirabel", "Nerissa", "Oberon",
    "Pandora", "Quill", "Rhea", "Silas", "Tamsin", "Una", "Valor",
    "Wisteria", "Xerxes", "Ysolde", "Zara", "Artemis", "Bastian",
    "Celestine", "Dorian", "Ember", "Fable", "Griffin", "Haven", "Ivy",
    "Juno", "Knox", "Lark", "Moss", "Nova", "Opal", "Phoenix",
    "Rune", "Storm", "Sage", "Thorn", "Umbra", "Vine", "Wolf",
    "Ash", "Blade", "Crow", "Drake", "Echo", "Frost", "Ghost",
    "Hawk", "Iron", "Jade", "Kai", "Lion", "Mist", "Night",
    "Oak", "Pike", "Rain", "Shadow", "Steel", "Tusk", "Vex",
    "Wave", "Onyx", "Pearl", "Reed", "Slate", "Vale", "Zen",
    "Arin", "Brin", "Cyra", "Dax", "Eira", "Fynn", "Gwen",
    "Hale", "Isla", "Jace", "Kova", "Lena", "Mira", "Nola",
    "Oren", "Pax", "Ren", "Sora", "Tael", "Uri", "Vera",
    "Wynn", "Xara", "Yael", "Zev", "Alder", "Brook", "Cedar",
    "Dove", "Elm", "Fern", "Glen", "Heath", "Ivy", "Jay",
    "Kestrel", "Linden", "Maple", "Nettle", "Olive", "Poppy",
    "Robin", "Sorrel", "Thistle", "Violet", "Willow", "Yarrow",
]

BUILTIN_VISUAL_CORPUS = [
    "tall muscular scarred brown-eyes black-hair",
    "short stocky red-beard blue-eyes bald",
    "slender pale silver-hair violet-eyes elegant",
    "average weathered tan-skin green-eyes grey-hair",
    "imposing broad dark-skin amber-eyes shaved-head",
    "petite agile freckled hazel-eyes auburn-hair",
    "gaunt hollow-cheeks white-hair ice-blue-eyes scarred",
    "youthful bright blonde-hair emerald-eyes dimpled",
    "elderly stooped long-white-beard cloudy-eyes wrinkled",
    "athletic toned bronze-skin braided-hair dark-eyes",
]


def _corpus_hash(docs: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(docs)).encode()).hexdigest()[:16]


class AssetGenerator:
    """NanoGPT-powered procedural asset generation across 6 domains."""

    DOMAINS = {
        "names": {"block_size": 24, "n_embd": 16, "steps": 300},
        "items": {"block_size": 64, "n_embd": 24, "steps": 200},
        "quests": {"block_size": 64, "n_embd": 24, "steps": 200},
        "rumors": {"block_size": 64, "n_embd": 24, "steps": 200},
        "visual": {"block_size": 48, "n_embd": 16, "steps": 200},
        "flavor": {"block_size": 64, "n_embd": 24, "steps": 200},
    }

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._model_dir = data_dir / "nanogpt_models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Any] = {}
        self._corpus_hashes: Dict[str, str] = {}
        self._hash_file = self._model_dir / "corpus_hashes.json"
        self._load_hashes()

    def _load_hashes(self):
        if self._hash_file.exists():
            try:
                self._corpus_hashes = json.loads(self._hash_file.read_text())
            except Exception:
                self._corpus_hashes = {}

    def _save_hashes(self):
        self._hash_file.write_text(json.dumps(self._corpus_hashes))

    def _get_model(self, domain: str):
        if domain in self._models:
            return self._models[domain]
        try:
            from adk.nanogpt import NanoGPT
        except ImportError:
            logger.warning("NanoGPT not available — asset generation disabled")
            return None

        config = self.DOMAINS.get(domain, self.DOMAINS["names"])
        model = NanoGPT(
            block_size=config["block_size"],
            n_embd=config["n_embd"],
        )
        # Try loading persisted model
        model_path = self._model_dir / f"{domain}.json"
        if model_path.exists():
            try:
                model.load(model_path)
                self._models[domain] = model
                return model
            except Exception:
                logger.warning("Failed to load %s model, will retrain", domain)

        self._models[domain] = model
        return model

    def _build_corpus(self, domain: str, graph=None) -> List[str]:
        from saga_engine.models import NodeType, MemoryType
        docs = []

        if domain == "names":
            docs.extend(BUILTIN_NAMES)
            if graph:
                for node in graph.find_nodes_by_type(NodeType.CHARACTER):
                    docs.append(node.name)
                    docs.extend(node.aliases)

        elif domain == "items":
            if graph:
                for node in graph.find_nodes_by_type(NodeType.ITEM):
                    if node.description:
                        docs.append(node.description[:120])
            if not docs:
                docs = [
                    "A rusty iron sword with notches in the blade",
                    "An ornate silver ring set with a pale moonstone",
                    "A vial of shimmering blue potion that hums faintly",
                    "A leather-bound journal filled with coded entries",
                    "A black feather that never touches the ground",
                ]

        elif domain == "quests":
            if graph:
                for node in graph.find_nodes_by_type(NodeType.PLOT_THREAD):
                    if node.description:
                        docs.append(node.description[:120])
            if not docs:
                docs = [
                    "Find the missing merchant who vanished on the north road",
                    "Retrieve the stolen relic from the sunken temple",
                    "Negotiate a truce between the warring river clans",
                    "Investigate strange disappearances in the fog quarter",
                    "Escort the oracle safely through bandit territory",
                ]

        elif domain == "rumors":
            if graph:
                from saga_engine.memory import MemoryManager
                # Look for memories tagged as rumors
                mem_dir = graph._data_dir if hasattr(graph, '_data_dir') else None
                if mem_dir:
                    try:
                        mm = MemoryManager(data_dir=mem_dir)
                        mm.load()
                        for m in mm.get_all():
                            if "rumor" in m.tags:
                                docs.append(m.summary)
                    except Exception:
                        pass
            if not docs:
                docs = [
                    "They say the old tower glows at midnight",
                    "The blacksmith has been seen talking to shadows",
                    "A dragon was spotted near the eastern peaks",
                    "The king's advisor serves a darker master",
                    "Gold coins from a dead kingdom appeared in the market",
                ]

        elif domain == "visual":
            docs.extend(BUILTIN_VISUAL_CORPUS)
            if graph:
                for node in graph.find_nodes_by_type(NodeType.CHARACTER):
                    appearance = node.properties.get("appearance", "")
                    if appearance:
                        docs.append(appearance[:100])

        elif domain == "flavor":
            if graph:
                for node in graph.find_nodes_by_type(NodeType.FACTION):
                    if node.description:
                        docs.append(node.description[:120])
                for node in graph.find_nodes_by_type(NodeType.LORE):
                    if node.description:
                        docs.append(node.description[:120])
            if not docs:
                docs = [
                    "The ancient order guards secrets older than the mountains",
                    "Crimson banners fly above walls scarred by siege",
                    "In the deep places the old songs still echo",
                    "The market bustles with traders from distant shores",
                    "Moonlight spills through the broken dome of the temple",
                ]

        return [d for d in docs if d and len(d) > 2]

    async def ensure_trained(self, domain: str, graph=None):
        """Train or retrain model if corpus changed."""
        model = self._get_model(domain)
        if model is None:
            return False

        corpus = self._build_corpus(domain, graph)
        if not corpus:
            return False

        new_hash = _corpus_hash(corpus)
        old_hash = self._corpus_hashes.get(domain, "")

        if model.is_trained and new_hash == old_hash:
            return True  # Already trained on same data

        config = self.DOMAINS.get(domain, self.DOMAINS["names"])
        await model.train(corpus, num_steps=config["steps"])

        # Persist
        model_path = self._model_dir / f"{domain}.json"
        model.save(model_path)
        self._corpus_hashes[domain] = new_hash
        self._save_hashes()
        logger.info("Trained %s model: %d docs, hash=%s", domain, len(corpus), new_hash)
        return True

    async def generate_name(self, count: int = 5, graph=None) -> List[str]:
        if not await self.ensure_trained("names", graph):
            return random.sample(BUILTIN_NAMES, min(count, len(BUILTIN_NAMES)))
        model = self._models.get("names")
        if not model:
            return random.sample(BUILTIN_NAMES, min(count, len(BUILTIN_NAMES)))
        raw = await model.generate(num_samples=count * 2, temperature=0.6)
        # Filter: capitalize, remove empty, deduplicate
        names = []
        seen = set()
        for name in raw:
            name = name.strip().title()
            if name and len(name) > 1 and name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        return names[:count] if names else random.sample(BUILTIN_NAMES, min(count, 5))

    async def generate_item_description(self, rarity: str = "common", graph=None) -> str:
        if not await self.ensure_trained("items", graph):
            return f"A {rarity} item of unknown origin"
        model = self._models.get("items")
        if not model:
            return f"A {rarity} item of unknown origin"
        temps = {"common": 0.5, "uncommon": 0.6, "rare": 0.7, "epic": 0.8, "legendary": 0.9}
        results = await model.generate(num_samples=3, temperature=temps.get(rarity, 0.5))
        return max(results, key=len) if results else f"A {rarity} item of unknown origin"

    async def generate_quest_hook(self, graph=None) -> str:
        if not await self.ensure_trained("quests", graph):
            return "A mysterious task awaits"
        model = self._models.get("quests")
        if not model:
            return "A mysterious task awaits"
        results = await model.generate(num_samples=3, temperature=0.7)
        return max(results, key=len) if results else "A mysterious task awaits"

    async def generate_rumor(self, graph=None) -> str:
        if not await self.ensure_trained("rumors", graph):
            return "Whispers of something strange"
        model = self._models.get("rumors")
        if not model:
            return "Whispers of something strange"
        results = await model.generate(num_samples=3, temperature=0.7)
        return max(results, key=len) if results else "Whispers of something strange"

    async def generate_visual_profile(self, npc_name: str = "", graph=None) -> dict:
        if not await self.ensure_trained("visual", graph):
            return self._random_visual()
        model = self._models.get("visual")
        if not model:
            return self._random_visual()
        results = await model.generate(num_samples=3, temperature=0.6)
        best = max(results, key=len) if results else ""
        return self._parse_visual(best) if best else self._random_visual()

    def _random_visual(self) -> dict:
        return {
            "hair": random.choice(["black", "brown", "blonde", "red", "grey", "white", "auburn"]),
            "eyes": random.choice(["brown", "blue", "green", "hazel", "grey", "amber", "violet"]),
            "build": random.choice(["slender", "average", "muscular", "stocky", "tall", "petite"]),
            "distinguishing": random.choice(["a scar", "freckles", "a tattoo", "a limp", "a piercing gaze"]),
        }

    def _parse_visual(self, text: str) -> dict:
        profile = {"raw": text}
        words = text.lower().split()
        hair_words = ["black-hair", "brown-hair", "blonde", "red-hair", "grey-hair",
                      "white-hair", "auburn-hair", "silver-hair", "braided-hair", "bald", "shaved-head"]
        eye_words = ["brown-eyes", "blue-eyes", "green-eyes", "hazel-eyes", "grey-eyes",
                     "amber-eyes", "violet-eyes", "ice-blue-eyes", "dark-eyes", "emerald-eyes", "cloudy-eyes"]
        build_words = ["tall", "short", "slender", "muscular", "stocky", "petite",
                       "average", "imposing", "broad", "gaunt", "athletic", "youthful", "elderly"]
        for w in words:
            if w in hair_words:
                profile["hair"] = w.replace("-hair", "").replace("-", " ")
            elif w in eye_words:
                profile["eyes"] = w.replace("-eyes", "").replace("-", " ")
            elif w in build_words:
                profile["build"] = w
        for key in ["hair", "eyes", "build"]:
            if key not in profile:
                profile[key] = self._random_visual()[key]
        return profile

    async def generate_flavor_text(self, theme: str = "", graph=None) -> str:
        if not await self.ensure_trained("flavor", graph):
            return "The world holds many secrets"
        model = self._models.get("flavor")
        if not model:
            return "The world holds many secrets"
        results = await model.generate(num_samples=3, temperature=0.7)
        return max(results, key=len) if results else "The world holds many secrets"


# Module-level singleton
_generator: Optional[AssetGenerator] = None


def get_asset_generator(data_dir: Path) -> AssetGenerator:
    global _generator
    if _generator is None:
        _generator = AssetGenerator(data_dir)
    return _generator
