import json
import os

import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

# Default state file location (can be overridden)
# If running from agent, it might be relative to agent root.
# We'll try to use a shared location or environment variable.
STATE_FILE = os.getenv("AITHER_VISUAL_STATE_FILE", os.path.join(os.getcwd(), "memory", "visual_state.json"))

class StateManager:
    def __init__(self, state_file=None):
        self.state_file = state_file or STATE_FILE
        self.state = self._load_state()
        self.current_persona = "unknown"
        self.base_seed = None

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as exc:
                logger.debug(f"Visual state load failed: {exc}")
        return {
            "clothing": "clothed", # clothed, nude, lingerie, etc.
            "location": "high tech office",
            "pose": "standing",
            "camera": "front view",
            "expression": "neutral",
            "scene_context": "", # Shared scene description for all agents
            "active_characters": [], # Who's in the scene
            "lighting": "studio lighting",
            "time_of_day": "daytime"
        }

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update_state(self, prompt: str):
        """Updates the state based on keywords in the prompt."""
        prompt_lower = prompt.lower()

        # Clothing State
        if any(w in prompt_lower for w in ["nude", "naked", "remove clothes", "undress", "no clothes", "strip"]):
            self.state["clothing"] = "nude"
        elif any(w in prompt_lower for w in ["lingerie", "underwear"]):
            self.state["clothing"] = "lingerie"
        elif any(w in prompt_lower for w in ["dress", "suit", "outfit", "clothed", "wear", "clothes", "put on", "dressed", "cover up", "jeans", "shirt", "skirt"]):
            self.state["clothing"] = "clothed"

        # Location State (Enhanced)
        if "office" in prompt_lower or "server room" in prompt_lower or "workspace" in prompt_lower:
            self.state["location"] = "high tech office, neon lights, glass walls"
        elif "bedroom" in prompt_lower:
            self.state["location"] = "bedroom, bed, dim lighting"
        elif "beach" in prompt_lower:
            self.state["location"] = "beach, sand, ocean, sunlight"
        elif "gym" in prompt_lower:
            self.state["location"] = "gym, workout equipment, mirrors"
        elif "dungeon" in prompt_lower:
            self.state["location"] = "dungeon, chains, dark, stone walls"
        elif "shower" in prompt_lower or "bathroom" in prompt_lower:
            self.state["location"] = "bathroom, shower, wet tiles, steam"

        # Lighting State (NEW)
        if "neon" in prompt_lower or "cyberpunk" in prompt_lower:
            self.state["lighting"] = "neon lights, cyan and magenta, cyberpunk"
        elif "dark" in prompt_lower or "dim" in prompt_lower or "night" in prompt_lower:
            self.state["lighting"] = "dim lighting, shadows, low key"
        elif "bright" in prompt_lower or "sunlight" in prompt_lower or "golden hour" in prompt_lower:
            self.state["lighting"] = "bright sunlight, golden hour, warm tones"
        elif "cinematic" in prompt_lower:
            self.state["lighting"] = "cinematic lighting, dramatic shadows"
        elif "studio" in prompt_lower:
            self.state["lighting"] = "studio lighting, professional"

        # Time of Day (NEW)
        if "morning" in prompt_lower or "sunrise" in prompt_lower:
            self.state["time_of_day"] = "morning, sunrise"
        elif "afternoon" in prompt_lower or "midday" in prompt_lower:
            self.state["time_of_day"] = "afternoon, daytime"
        elif "evening" in prompt_lower or "sunset" in prompt_lower:
            self.state["time_of_day"] = "evening, sunset, golden hour"
        elif "night" in prompt_lower or "midnight" in prompt_lower:
            self.state["time_of_day"] = "night, dark sky"

        # Pose State (Enhanced)
        if "standing" in prompt_lower: self.state["pose"] = "standing"
        elif "sitting" in prompt_lower: self.state["pose"] = "sitting"
        elif "lying" in prompt_lower or "laying" in prompt_lower: self.state["pose"] = "lying down"
        elif "kneeling" in prompt_lower: self.state["pose"] = "kneeling"
        elif "leaning" in prompt_lower or "bent" in prompt_lower:
            self.state["pose"] = "leaning forward"
        elif "riding" in prompt_lower:
            self.state["pose"] = "riding"
        elif "lying" in prompt_lower or "laying" in prompt_lower:
            self.state["pose"] = "lying down, reclining"

        # Camera Angle (NEW)
        if "pov" in prompt_lower or "first person" in prompt_lower:
            self.state["camera"] = "pov, first person view"
        elif "from behind" in prompt_lower or "back view" in prompt_lower:
            self.state["camera"] = "from behind, back view"
        elif "side view" in prompt_lower or "profile" in prompt_lower:
            self.state["camera"] = "side view, profile"
        elif "from above" in prompt_lower or "top down" in prompt_lower:
            self.state["camera"] = "from above, high angle"
        elif "from below" in prompt_lower or "low angle" in prompt_lower:
            self.state["camera"] = "from below, low angle"
        else:
            self.state["camera"] = "front view"

        # Expression (NEW)
        if "angry" in prompt_lower or "mad" in prompt_lower:
            self.state["expression"] = "angry, furrowed brow"
        elif "happy" in prompt_lower or "smile" in prompt_lower or "grin" in prompt_lower:
            self.state["expression"] = "smiling, happy"
        elif "sad" in prompt_lower or "crying" in prompt_lower:
            self.state["expression"] = "sad, crying, tears"
        elif "scared" in prompt_lower or "fear" in prompt_lower:
            self.state["expression"] = "scared, fearful"
        elif "neutral" in prompt_lower:
            self.state["expression"] = "neutral expression"

        self._save_state()

    def set_persona(self, persona_name: str):
        """Sets the current persona generating the image."""
        self.current_persona = persona_name.lower()

    def get_deterministic_seed(self, prompt: str) -> int:
        """Generates a deterministic seed based on prompt + state."""
        import hashlib
        # Create a hash from prompt + current state
        state_str = json.dumps(self.state, sort_keys=True)
        combined = f"{prompt}{state_str}{self.current_persona}"
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        # Convert first 8 hex chars to int (32-bit seed)
        seed = int(hash_val[:8], 16) & 0x7FFFFFFF  # Keep positive
        return seed

    def inject_state(self, prompt: str, force_scene: bool = False) -> str:
        """Injects state information into the prompt if not explicitly overridden."""
        prompt_lower = prompt.lower()

        # FORCE inject scene context for group chats (shared consistency)
        if force_scene and self.state.get("scene_context"):
            scene_desc = self.state["scene_context"]
            # Prepend scene context for consistency
            prompt = f"{scene_desc}, {prompt}"
            return prompt

        # Check if the prompt implies a person/character
        person_keywords = ["woman", "man", "girl", "boy", "person", "lady", "guy", "female", "male", "she", "he", "her", "his", "self", "me", "myself", "aither", "assistant", "character", "actor", "model", "human", "someone", "somebody", "portrait", "face", "eyes", "hair", "body"]
        has_person = any(k in prompt_lower for k in person_keywords)

        # Inject Lighting (if not overridden)
        lighting_keywords = ["lighting", "light", "neon", "dark", "bright", "dim", "cinematic", "studio", "shadow"]
        if not any(k in prompt_lower for k in lighting_keywords):
            prompt += f", {self.state['lighting']}"

        # Inject Location (if not overridden)
        location_override_keywords = ["office", "bedroom", "beach", "gym", "background", "setting", "street", "city", "outdoors", "indoors", "landscape", "view", "scene", "room", "park", "forest", "mountain", "sea", "ocean", "space", "sky", "dungeon", "bathroom", "shower"]
        if not any(w in prompt_lower for w in location_override_keywords):
            prompt += f", {self.state['location']}"

        # Inject Clothing State (only if a person is implied and not overridden)
        if has_person:
            if self.state["clothing"] == "nude" and not any(w in prompt_lower for w in ["clothed", "wear", "dress", "suit", "outfit", "shirt", "pants"]):
                if "nude" not in prompt_lower:
                    prompt += ", nude, artistic"
            elif self.state["clothing"] == "clothed" and not any(w in prompt_lower for w in ["nude", "naked", "undress", "lingerie", "bikini"]):
                # If state is clothed but prompt doesn't specify what to wear, add generic clothing or specific items if known
                if not any(w in prompt_lower for w in ["dress", "suit", "outfit", "clothes", "shirt", "skirt", "jeans", "wearing"]):
                    prompt += ", wearing clothes, fully clothed, professional outfit"

        # Inject Camera Angle (if not overridden)
        camera_keywords = ["pov", "from behind", "from above", "from below", "side view", "back view", "front view", "angle", "perspective"]
        if not any(k in prompt_lower for k in camera_keywords):
            prompt += f", {self.state['camera']}"

        return prompt

state_manager = StateManager()
