"""
AitherOS Speculative Image Generator

Predicts what images the user might want next and pre-generates them
in the background, giving the appearance of instant generation.

Features:
- Monitors conversation context
- Predicts likely next image requests
- Background generation queue
- Scene state tracking
- Smart cache management
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Paths — use paths.py for Docker-safe writable directory resolution
try:
    from aither_adk.paths import get_saga_data_dir, get_saga_subdir
    NARRATIVE_AGENT_DIR = get_saga_data_dir()
    CACHE_DIR = get_saga_subdir("memory", "image_cache", create=True)
except ImportError:
    AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NARRATIVE_AGENT_DIR = os.path.join(AGENT_DIR, "Saga")
    CACHE_DIR = os.path.join(NARRATIVE_AGENT_DIR, "memory", "image_cache")
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except OSError:
        pass  # Read-only filesystem (Docker)
STATE_FILE = os.path.join(NARRATIVE_AGENT_DIR, "memory", "scene_state.json")


@dataclass
class SceneState:
    """Current state of the visual scene."""
    # Characters present
    active_personas: List[str] = field(default_factory=list)

    # What's happening
    current_action: str = ""
    current_pose: str = ""

    # Environment
    location: str = "high-tech office"
    lighting: str = "soft ambient lighting"

    # Character states
    clothing_state: Dict[str, str] = field(default_factory=dict)  # persona -> "nude", "clothed", etc.

    # Conversation hints
    nsfw_level: int = 0  # 0-5, how NSFW the conversation is getting
    visual_intent: float = 0.0  # 0-1, how likely user wants an image

    # Recent context
    recent_messages: deque = field(default_factory=lambda: deque(maxlen=10))

    def to_dict(self) -> dict:
        return {
            "active_personas": self.active_personas,
            "current_action": self.current_action,
            "current_pose": self.current_pose,
            "location": self.location,
            "lighting": self.lighting,
            "clothing_state": self.clothing_state,
            "nsfw_level": self.nsfw_level,
            "visual_intent": self.visual_intent,
        }


@dataclass
class CachedImage:
    """A pre-generated image in the cache."""
    path: str
    prompt: str
    personas: List[str]
    action: str
    timestamp: float
    relevance_score: float = 1.0  # How relevant to current scene


class SpeculativeGenerator:
    """
    Pre-generates images based on conversation context.

    The goal is to have likely images ready BEFORE the user asks,
    giving the appearance of instant generation.
    """

    def __init__(self):
        self.scene = SceneState()
        self.cache: Dict[str, CachedImage] = {}  # hash -> CachedImage
        self.generation_queue: Queue = Queue()
        self.is_generating = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load_state()

    def _load_state(self):
        """Load scene state from disk."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.scene, key):
                        setattr(self.scene, key, value)
            except Exception as exc:
                logger.debug(f"Scene state load failed: {exc}")

        # Load cached images
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(CACHE_DIR, f), 'r') as file:
                        data = json.load(file)
                        img_hash = f[:-5]  # Remove .json
                        self.cache[img_hash] = CachedImage(**data)
                except Exception as exc:
                    logger.debug(f"Cached image load failed: {exc}")

    def _save_state(self):
        """Save scene state to disk."""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.scene.to_dict(), f, indent=2)
        except Exception as exc:
            logger.debug(f"Scene state save failed: {exc}")

    def _save_cached_image(self, img_hash: str, cached: CachedImage):
        """Save cached image metadata."""
        try:
            with open(os.path.join(CACHE_DIR, f"{img_hash}.json"), 'w') as f:
                json.dump(asdict(cached), f)
        except Exception as exc:
            logger.debug(f"Cached image save failed: {exc}")

    # ========== Context Analysis ==========

    def analyze_message(self, message: str, sender: str = "user") -> Dict[str, Any]:
        """
        Analyze a message and update scene state.
        Returns predictions about what images might be needed.
        """
        text = message.lower()

        # Update recent messages
        self.scene.recent_messages.append({"sender": sender, "text": text, "time": time.time()})

        # Detect personas mentioned
        import re
        mentions = re.findall(r"@(\w+)", text)
        if mentions:
            self.scene.active_personas = list(set(self.scene.active_personas + mentions))

        # Detect NSFW level
        nsfw_keywords = {
            1: ["sexy", "hot", "beautiful", "cute"],
            2: ["nude", "naked", "undress", "strip", "topless"],
            3: ["exposed", "revealing", "suggestive", "provocative"],
            4: ["fuck", "sex", "penetrat", "anal", "oral"],
            5: ["forced", "extreme", "violent"],
        }

        for level, keywords in nsfw_keywords.items():
            if any(kw in text for kw in keywords):
                self.scene.nsfw_level = max(self.scene.nsfw_level, level)

        # Detect visual intent (likelihood user wants an image)
        visual_signals = [
            ("show", 0.7), ("see", 0.5), ("look", 0.4), ("pic", 0.9), ("photo", 0.9),
            ("selfie", 0.95), ("image", 0.9), ("visualize", 0.8), ("imagine", 0.3),
            ("what do you look like", 0.9), ("send", 0.6),
        ]

        for signal, weight in visual_signals:
            if signal in text:
                self.scene.visual_intent = max(self.scene.visual_intent, weight)

        # Detect actions/poses
        action_keywords = {
            "from behind": "rear view, looking back",
            "riding": "dynamic pose, action",
            "lying": "lying down, reclining",
            "leaning": "leaning forward",
            "selfie": "selfie, holding phone, pov",
            "kneeling": "kneeling, looking up",
        }

        for keyword, action in action_keywords.items():
            if keyword in text:
                self.scene.current_action = action
                self.scene.current_pose = keyword

        # Detect clothing changes
        if "nude" in text or "naked" in text:
            for persona in self.scene.active_personas:
                self.scene.clothing_state[persona] = "nude"
        elif "clothed" in text or "dress" in text:
            for persona in self.scene.active_personas:
                self.scene.clothing_state[persona] = "clothed"

        self._save_state()

        # Return predictions
        predictions = self._predict_next_images()

        return {
            "nsfw_level": self.scene.nsfw_level,
            "visual_intent": self.scene.visual_intent,
            "active_personas": self.scene.active_personas,
            "current_action": self.scene.current_action,
            "predictions": predictions,
        }

    def _predict_next_images(self) -> List[Dict[str, Any]]:
        """
        Predict what images the user might want next.
        Returns a list of prompt specs to pre-generate.
        """
        predictions = []

        # If visual intent is high, predict variations
        if self.scene.visual_intent > 0.5:
            personas = self.scene.active_personas or ["aither"]

            for persona in personas:
                # Current state image
                predictions.append({
                    "personas": [persona],
                    "action": self.scene.current_action or "standing, looking at viewer",
                    "clothing": self.scene.clothing_state.get(persona, "clothed"),
                    "priority": 1.0,
                })

                # If NSFW, predict nude version
                if self.scene.nsfw_level >= 2:
                    predictions.append({
                        "personas": [persona],
                        "action": self.scene.current_action or "standing, looking at viewer",
                        "clothing": "nude",
                        "priority": 0.8,
                    })

                # If action-oriented, predict action shots
                if self.scene.nsfw_level >= 3 and self.scene.current_pose:
                    predictions.append({
                        "personas": [persona],
                        "action": self.scene.current_action,
                        "clothing": "nude",
                        "priority": 0.9,
                    })

        return predictions

    # ========== Cache Management ==========

    def _generate_cache_key(self, personas: List[str], action: str, clothing: str) -> str:
        """Generate a unique hash for an image configuration."""
        key_str = f"{sorted(personas)}_{action}_{clothing}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def find_cached_image(self, personas: List[str], action: str = None, clothing: str = None) -> Optional[CachedImage]:
        """
        Find a cached image that matches the request.
        Returns the most relevant cached image or None.
        """
        best_match = None
        best_score = 0

        for img_hash, cached in self.cache.items():
            # Check if personas match
            if not set(personas).issubset(set(cached.personas)):
                continue

            score = 0.5  # Base score for persona match

            # Action match
            if action and action.lower() in cached.action.lower():
                score += 0.3

            # Recency bonus
            age = time.time() - cached.timestamp
            if age < 60:  # Less than 1 minute old
                score += 0.2
            elif age < 300:  # Less than 5 minutes
                score += 0.1

            if score > best_score:
                best_score = score
                best_match = cached

        return best_match if best_score > 0.5 else None

    def add_to_cache(self, path: str, prompt: str, personas: List[str], action: str):
        """Add a generated image to the cache."""
        img_hash = self._generate_cache_key(personas, action, "")

        cached = CachedImage(
            path=path,
            prompt=prompt,
            personas=personas,
            action=action,
            timestamp=time.time(),
        )

        self.cache[img_hash] = cached
        self._save_cached_image(img_hash, cached)

        # Clean old cache entries (keep last 20)
        if len(self.cache) > 20:
            oldest = sorted(self.cache.items(), key=lambda x: x[1].timestamp)[:5]
            for h, _ in oldest:
                del self.cache[h]
                try:
                    os.remove(os.path.join(CACHE_DIR, f"{h}.json"))
                except Exception as exc:
                    logger.debug(f"Cache file removal failed: {exc}")

    # ========== Background Generation ==========

    def start_background_worker(self):
        """Start the background generation worker."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_background_worker(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def _worker_loop(self):
        """Background worker that processes the generation queue."""
        while not self._stop_event.is_set():
            try:
                # Check for work
                try:
                    task = self.generation_queue.get(timeout=1)
                except Empty:
                    continue

                if task is None:
                    break

                # Generate the image
                self._generate_speculative(task)

            except Exception as e:
                print(f"[SpecGen] Worker error: {e}")

    def _generate_speculative(self, task: Dict[str, Any]):
        """Generate a speculative image."""
        try:
            # Build prompt from task
            personas = task.get("personas", ["aither"])
            action = task.get("action", "standing, looking at viewer")
            clothing = task.get("clothing", "clothed")

            # Check if already cached
            cache_key = self._generate_cache_key(personas, action, clothing)
            if cache_key in self.cache:
                return  # Already have this one

            print(f"[SpecGen] Pre-generating: {personas} - {action}")

            # Build the prompt
            # This is a simplified version - would use the full prompt builder in production
            persona_tags = ", ".join([f"({p}:1.2)" for p in personas])
            clothing_tags = "(nude:1.3)" if clothing == "nude" else ""

            prompt = f"1girl, solo, {persona_tags}, {clothing_tags}, {action}, anime style, masterpiece, best quality"

            # Generate using ComfyUI
            import asyncio

            from AitherOS.AitherNode.AitherCanvas import generate_local

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                paths = loop.run_until_complete(
                    generate_local(prompt, None, negative_prompt="bad anatomy, blurry, low quality")
                )

                if paths:
                    self.add_to_cache(paths[0], prompt, personas, action)
                    print(f"[SpecGen] Cached: {paths[0]}")
            finally:
                loop.close()

        except Exception as e:
            print(f"[SpecGen] Generation error: {e}")

    def queue_speculative(self, predictions: List[Dict[str, Any]]):
        """Queue predicted images for background generation."""
        # Sort by priority
        sorted_preds = sorted(predictions, key=lambda x: x.get("priority", 0), reverse=True)

        for pred in sorted_preds[:3]:  # Max 3 at a time
            self.generation_queue.put(pred)

    # ========== Instant Serving ==========

    def get_instant_image(self, request: str, personas: List[str] = None) -> Optional[str]:
        """
        Try to instantly serve a pre-generated image.
        Returns the path if available, None otherwise.
        """
        if personas is None:
            personas = self.scene.active_personas or ["aither"]

        # Parse request for action
        action = self._extract_action(request)

        # Try to find in cache
        cached = self.find_cached_image(personas, action)

        if cached and os.path.exists(cached.path):
            print(f"[SpecGen] [ZAP] Serving from cache: {cached.path}")
            return cached.path

        return None

    def _extract_action(self, request: str) -> str:
        """Extract action from request."""
        text = request.lower()

        actions = {
            "selfie": "selfie, holding phone, pov",
            "rear": "rear view, looking back",
            "lying": "lying down, reclining",
            "riding": "dynamic pose, action",
            "kneeling": "kneeling, looking up",
            "nude": "nude, artistic, standing",
        }

        for keyword, action in actions.items():
            if keyword in text:
                return action

        return "standing, looking at viewer"


# Singleton
_generator: Optional[SpeculativeGenerator] = None


def get_speculative_generator() -> SpeculativeGenerator:
    """Get the singleton speculative generator."""
    global _generator
    if _generator is None:
        _generator = SpeculativeGenerator()
    return _generator


# Convenience functions

def analyze_and_predict(message: str) -> Dict[str, Any]:
    """Analyze message and start pre-generating likely images."""
    gen = get_speculative_generator()
    result = gen.analyze_message(message)

    # If high visual intent, start background generation
    if result["visual_intent"] > 0.6:
        gen.start_background_worker()
        gen.queue_speculative(result["predictions"])

    return result


def try_instant_serve(request: str, personas: List[str] = None) -> Optional[str]:
    """Try to serve an image instantly from cache."""
    return get_speculative_generator().get_instant_image(request, personas)

