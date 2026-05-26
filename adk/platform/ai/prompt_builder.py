"""
AitherOS Prompt Builder

Prompt engineering for image generation.
Combines character descriptions, scene context, user preferences,
pose/action detection, and quality tags.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """User preferences for prompt generation."""
    preferred_style: str = "anime"  # anime, realistic, semi-realistic
    quality_level: str = "high"  # high, ultra, fast

    # Weight preferences (1.0 = default, higher = stronger)
    body_weight: float = 1.1
    face_weight: float = 1.1
    pose_weight: float = 1.15
    lighting_weight: float = 1.0

    always_include: List[str] = field(default_factory=lambda: [
        "masterpiece", "best quality", "highly detailed"
    ])

    always_exclude: List[str] = field(default_factory=lambda: [
        "bad anatomy", "bad hands", "missing fingers", "extra digits",
        "blurry", "low quality", "text", "watermark", "signature",
        "ugly", "deformed", "disfigured", "mutated"
    ])

    character_exclusions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Character:
    """Character visual description."""
    name: str
    body: str = ""
    face: str = ""
    hair: str = ""
    outfit_default: str = ""
    accessories: str = ""
    personality_tags: List[str] = field(default_factory=list)


# Pre-defined characters
CHARACTERS = {
    "aither": Character(
        name="Aither",
        body="1girl, solo, athletic build",
        face="beautiful face, confident expression, intelligent eyes, thin-framed glasses",
        hair="long sleek dark hair, high ponytail",
        outfit_default="office attire, blouse, pencil skirt",
        accessories="glasses",
        personality_tags=["confident", "professional"]
    ),
}


@dataclass
class Scene:
    """Scene/setting context."""
    location: str = "high-tech office"
    lighting: str = "neon ambient lighting, cyan and magenta accents"
    atmosphere: str = "futuristic, sleek"
    time_of_day: str = ""
    additional_details: str = ""


# Pre-defined scenes
SCENES = {
    "office": Scene(
        location="high-tech futuristic office",
        lighting="neon ambient lighting, cyan and magenta accents",
        atmosphere="holographic displays, server racks background"
    ),
    "outdoors": Scene(
        location="outdoor setting",
        lighting="natural sunlight, golden hour",
        atmosphere="nature, sky, trees"
    ),
    "studio": Scene(
        location="photo studio",
        lighting="studio lighting, softbox, rim light",
        atmosphere="professional backdrop, clean"
    ),
    "city": Scene(
        location="cyberpunk city street",
        lighting="neon signs, rain reflections",
        atmosphere="futuristic cityscape, night"
    ),
}


class PromptBuilder:
    """Prompt builder that combines character, scene, and style context."""

    POSE_BOOSTS = {
        "selfie": ["selfie", "POV shot", "holding phone", "looking at viewer", "close up"],
        "pov": ["POV", "first person view", "looking at viewer", "eye contact"],
        "standing": ["standing", "full body", "upright pose"],
        "sitting": ["sitting", "seated", "on chair"],
        "lying": ["lying down", "reclined"],
        "kneeling": ["kneeling"],
        "back_view": ["from behind", "back view", "looking over shoulder"],
        "action": ["dynamic pose", "action pose", "motion"],
        "smiling": ["smiling", "happy expression", "warm smile"],
        "serious": ["serious expression", "intense gaze", "focused"],
        "playful": ["playful expression", "wink"],
    }

    STYLE_PRESETS = {
        "anime": "anime style, 2d, cel shading, vibrant colors",
        "realistic": "photorealistic, 8k, detailed skin texture, professional photo",
        "semi-realistic": "semi-realistic, detailed, soft lighting, artistic",
    }

    QUALITY_PRESETS = {
        "fast": "good quality",
        "high": "masterpiece, best quality, highly detailed",
        "ultra": "masterpiece, best quality, extremely detailed, 8k, ultra high resolution",
    }

    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()

    def detect_character(self, text: str) -> Optional[Character]:
        """Detect which character is being referenced."""
        text_lower = text.lower()

        for name, char in CHARACTERS.items():
            if name in text_lower or char.name.lower() in text_lower:
                return char

        personal_indicators = ["your", "you", "me", "send", "show me", "selfie"]
        if any(ind in text_lower for ind in personal_indicators):
            return CHARACTERS.get("aither")

        return None

    def detect_scene(self, text: str) -> Scene:
        """Detect scene/setting from text."""
        text_lower = text.lower()

        scene_keywords = {
            "outdoors": ["outside", "outdoor", "park", "beach", "nature", "forest"],
            "studio": ["studio", "photoshoot", "modeling"],
            "office": ["office", "work", "desk"],
            "city": ["city", "street", "cyberpunk", "urban", "night"],
        }

        for scene_name, keywords in scene_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return SCENES.get(scene_name, SCENES["office"])

        return SCENES["office"]

    def detect_poses_and_actions(self, text: str) -> List[str]:
        """Detect poses and actions from text."""
        text_lower = text.lower()
        detected = []

        for keyword, boosts in self.POSE_BOOSTS.items():
            if keyword in text_lower:
                detected.extend(boosts)

        return list(set(detected))

    def apply_weights(self, tag: str, weight: float) -> str:
        """Apply weight to a tag: tag -> (tag:weight)"""
        if weight == 1.0:
            return tag
        return f"({tag}:{weight:.2f})"

    def build_prompt(
        self,
        user_request: str,
        character: Character = None,
        scene: Scene = None,
        extra_tags: List[str] = None,
    ) -> Tuple[str, str]:
        """
        Build a complete prompt from all context.

        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        parts = []

        if character is None:
            character = self.detect_character(user_request)

        if character:
            body_weighted = self.apply_weights(character.body, self.config.body_weight)
            parts.append(body_weighted)
            face_weighted = self.apply_weights(character.face, self.config.face_weight)
            parts.append(face_weighted)
            parts.append(character.hair)
        else:
            parts.append("1girl, solo")

        poses = self.detect_poses_and_actions(user_request)
        if poses:
            pose_str = ", ".join(poses)
            pose_weighted = self.apply_weights(pose_str, self.config.pose_weight)
            parts.append(pose_weighted)

        if scene is None:
            scene = self.detect_scene(user_request)

        parts.append(scene.location)
        lighting_weighted = self.apply_weights(scene.lighting, self.config.lighting_weight)
        parts.append(lighting_weighted)
        if scene.atmosphere:
            parts.append(scene.atmosphere)

        style = self.STYLE_PRESETS.get(self.config.preferred_style, self.STYLE_PRESETS["anime"])
        parts.append(style)

        quality = self.QUALITY_PRESETS.get(self.config.quality_level, self.QUALITY_PRESETS["high"])
        parts.append(quality)

        parts.extend(self.config.always_include)

        if extra_tags:
            parts.extend(extra_tags)

        positive = ", ".join(parts)

        negatives = list(self.config.always_exclude)
        if character and character.name.lower() in self.config.character_exclusions:
            negatives.extend(self.config.character_exclusions[character.name.lower()])

        negative = ", ".join(negatives)

        return positive, negative


_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get the singleton prompt builder."""
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder


def build_image_prompt(
    user_request: str,
    character_name: str = None,
    scene_name: str = None,
    extra_tags: List[str] = None,
    use_persona_system: bool = True
) -> Tuple[str, str]:
    """
    Convenience function to build an image prompt.

    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    if use_persona_system:
        try:
            from aither_adk.ai.persona_image_system import generate_persona_prompt

            persona = character_name or "aither"
            result = generate_persona_prompt(persona, user_request)

            prompt = result.get("prompt", "")
            negative = result.get("negative_prompt", "")

            if prompt:
                if extra_tags:
                    prompt = f"{prompt}, {', '.join(extra_tags)}"
                return prompt, negative

        except Exception as e:
            print(f"[PromptBuilder] PersonaImageSystem failed ({e}), using legacy builder")

    builder = get_prompt_builder()

    character = CHARACTERS.get(character_name) if character_name else None
    scene = SCENES.get(scene_name) if scene_name else None

    return builder.build_prompt(
        user_request=user_request,
        character=character,
        scene=scene,
        extra_tags=extra_tags
    )


if __name__ == "__main__":
    test_requests = [
        "send a selfie",
        "show me standing in the office",
        "sitting at the desk",
        "action pose in the city",
    ]

    for req in test_requests:
        pos, neg = build_image_prompt(req)
        print(f"\n=== {req} ===")
        print(f"POSITIVE: {pos[:200]}...")
        print(f"NEGATIVE: {neg[:100]}...")
