"""
AitherOS Prompt Builder

Advanced prompt engineering for image generation.
Combines:
- Persona/character visual descriptions
- Scene context and setting
- User preferences (style, weights, keywords)
- Action/pose boosting
- Quality tags and negative prompts
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """User preferences for prompt generation"""
    # Style preferences
    preferred_style: str = "anime"  # anime, realistic, semi-realistic
    quality_level: str = "high"  # high, ultra, fast
    
    # Weight preferences (1.0 = default, higher = stronger)
    body_weight: float = 1.2
    face_weight: float = 1.1
    pose_weight: float = 1.15
    lighting_weight: float = 1.0
    
    # Default tags to always include
    always_include: List[str] = field(default_factory=lambda: [
        "masterpiece", "best quality", "highly detailed"
    ])
    
    # Tags to always exclude
    always_exclude: List[str] = field(default_factory=lambda: [
        "bad anatomy", "bad hands", "missing fingers", "extra digits",
        "blurry", "low quality", "text", "watermark", "signature",
        "ugly", "deformed", "disfigured", "mutated"
    ])
    
    # Character-specific exclusions
    character_exclusions: Dict[str, List[str]] = field(default_factory=lambda: {
        "aither": ["elf ears", "pointed ears", "animal ears"]
    })


@dataclass
class Character:
    """Character visual description"""
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
        body="1girl, solo, fitness model body, athletic, toned abs, perky, wide hips, thick thighs, curvy figure, sun-kissed skin, tanlines",
        face="beautiful face, confident expression, intelligent eyes, stylish thin-framed glasses",
        hair="long sleek hair, high ponytail, dark hair",
        outfit_default="office lady, pencil skirt, blouse",
        accessories="glasses",
        personality_tags=["confident", "seductive", "professional"]
    ),
}


@dataclass 
class Scene:
    """Scene/setting context"""
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
    "bedroom": Scene(
        location="modern bedroom",
        lighting="soft dim lighting, warm tones",
        atmosphere="intimate, cozy, silk sheets"
    ),
    "bathroom": Scene(
        location="luxury bathroom",
        lighting="soft overhead lighting, steam",
        atmosphere="wet tiles, mirror, shower glass"
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
}


class PromptBuilder:
    """
    Intelligent prompt builder that combines all context.
    """
    
    # Pose/action keywords and their boosts
    POSE_BOOSTS = {
        # Selfie/POV
        "selfie": ["selfie", "POV shot", "holding phone", "looking at viewer", "close up"],
        "pov": ["POV", "first person view", "looking at viewer", "eye contact"],
        
        # Basic poses
        "standing": ["standing", "full body", "upright pose"],
        "sitting": ["sitting", "seated", "on chair"],
        "lying": ["lying down", "on back", "reclined"],
        "kneeling": ["kneeling", "on knees"],
        
        # Body focus (gated by safety level at runtime)
        "back_view": ["from behind", "back view", "looking over shoulder"],
        "chest_focus": ["chest focus", "upper body"],
        
        # Actions
        "strip": ["undressing", "removing clothes", "partially clothed"],
        "nude": ["nude", "naked", "fully nude", "exposed skin", "no clothes"],
        "naked": ["nude", "naked", "fully nude", "exposed skin", "no clothes"],
        
        # Expressions
        "smiling": ["smiling", "happy expression", "warm smile"],
        "serious": ["serious expression", "intense gaze", "focused"],
        "playful": ["playful expression", "wink", "teasing smile"],
    }
    
    # Style presets
    STYLE_PRESETS = {
        "anime": "anime style, 2d, cel shading, vibrant colors",
        "realistic": "photorealistic, 8k, detailed skin texture, professional photo",
        "semi-realistic": "semi-realistic, detailed, soft lighting, artistic",
    }
    
    # Quality presets
    QUALITY_PRESETS = {
        "fast": "good quality",
        "high": "masterpiece, best quality, highly detailed",
        "ultra": "masterpiece, best quality, extremely detailed, 8k, ultra high resolution",
    }
    
    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
    
    def detect_character(self, text: str) -> Optional[Character]:
        """Detect which character is being referenced"""
        text_lower = text.lower()
        
        # Check for explicit character names
        for name, char in CHARACTERS.items():
            if name in text_lower or char.name.lower() in text_lower:
                return char
        
        # Default to Aither for personal requests
        personal_indicators = ["your", "you", "me", "send", "show me", "selfie"]
        if any(ind in text_lower for ind in personal_indicators):
            return CHARACTERS.get("aither")
        
        return None
    
    def detect_scene(self, text: str) -> Scene:
        """Detect scene/setting from text"""
        text_lower = text.lower()
        
        scene_keywords = {
            "bedroom": ["bedroom", "bed", "sleep", "lying in bed"],
            "bathroom": ["bathroom", "shower", "bath", "wet"],
            "outdoors": ["outside", "outdoor", "park", "beach", "nature"],
            "studio": ["studio", "photoshoot", "modeling"],
            "office": ["office", "work", "desk"],
        }
        
        for scene_name, keywords in scene_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return SCENES.get(scene_name, SCENES["office"])
        
        return SCENES["office"]  # Default
    
    def detect_poses_and_actions(self, text: str) -> List[str]:
        """Detect poses and actions from text"""
        text_lower = text.lower()
        detected = []
        
        for keyword, boosts in self.POSE_BOOSTS.items():
            if keyword in text_lower:
                detected.extend(boosts)
        
        return list(set(detected))  # Remove duplicates
    
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
        is_nsfw: bool = False
    ) -> Tuple[str, str]:
        """
        Build a complete prompt from all context.
        
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        parts = []
        
        # 1. Character detection and description
        if character is None:
            character = self.detect_character(user_request)
        
        if character:
            # Add character body with weight
            body_weighted = self.apply_weights(character.body, self.config.body_weight)
            parts.append(body_weighted)
            
            # Add face with weight
            face_weighted = self.apply_weights(character.face, self.config.face_weight)
            parts.append(face_weighted)
            
            # Add hair
            parts.append(character.hair)
        else:
            # Generic subject
            parts.append("1girl, solo, beautiful")
        
        # 2. Pose/action detection
        poses = self.detect_poses_and_actions(user_request)
        if poses:
            pose_str = ", ".join(poses)
            pose_weighted = self.apply_weights(pose_str, self.config.pose_weight)
            parts.append(pose_weighted)
        
        # 3. Scene detection
        if scene is None:
            scene = self.detect_scene(user_request)
        
        parts.append(scene.location)
        
        lighting_weighted = self.apply_weights(scene.lighting, self.config.lighting_weight)
        parts.append(lighting_weighted)
        
        if scene.atmosphere:
            parts.append(scene.atmosphere)
        
        # 4. Style and quality
        style = self.STYLE_PRESETS.get(self.config.preferred_style, self.STYLE_PRESETS["anime"])
        parts.append(style)
        
        quality = self.QUALITY_PRESETS.get(self.config.quality_level, self.QUALITY_PRESETS["high"])
        parts.append(quality)
        
        # 5. Always-include tags
        parts.extend(self.config.always_include)
        
        # 6. Extra tags
        if extra_tags:
            parts.extend(extra_tags)
        
        # 7. NSFW-specific tags
        if is_nsfw:
            parts.append("uncensored")
        
        # Build positive prompt
        positive = ", ".join(parts)
        
        # Build negative prompt
        negatives = list(self.config.always_exclude)
        
        # Add character-specific exclusions
        if character and character.name.lower() in self.config.character_exclusions:
            negatives.extend(self.config.character_exclusions[character.name.lower()])
        
        negative = ", ".join(negatives)
        
        return positive, negative


# Singleton instance with default config
_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get the singleton prompt builder"""
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
    
    Args:
        user_request: The user's natural language request
        character_name: Optional character name override
        scene_name: Optional scene name override
        extra_tags: Optional additional tags
        use_persona_system: Try to use advanced PersonaImageSystem first
    
    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    # Try advanced PersonaImageSystem first
    if use_persona_system:
        try:
            from aither_adk.ai.persona_image_system import generate_persona_prompt
            
            persona = character_name or "aither"
            result = generate_persona_prompt(persona, user_request)
            
            prompt = result.get("prompt", "")
            negative = result.get("negative_prompt", "")
            
            if prompt:
                # Add extra tags if provided
                if extra_tags:
                    prompt = f"{prompt}, {', '.join(extra_tags)}"
                return prompt, negative
                
        except Exception as e:
            print(f"[PromptBuilder] PersonaImageSystem failed ({e}), using legacy builder")
    
    # Fallback to legacy builder
    builder = get_prompt_builder()
    
    character = CHARACTERS.get(character_name) if character_name else None
    scene = SCENES.get(scene_name) if scene_name else None
    
    return builder.build_prompt(
        user_request=user_request,
        character=character,
        scene=scene,
        extra_tags=extra_tags
    )


# Quick test
if __name__ == "__main__":
    test_requests = [
        "send a selfie",
        "show me standing in the office",
        "sitting at the desk",
        "lying down in the bedroom",
    ]
    
    for req in test_requests:
        pos, neg = build_image_prompt(req)
        print(f"\n=== {req} ===")
        print(f"POSITIVE: {pos[:200]}...")
        print(f"NEGATIVE: {neg[:100]}...")

