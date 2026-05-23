"""
AitherOS Persona Image Generation System

Consistent character image generation with:
- Anchor reference images for face/style consistency
- ControlNet integration for pose/face consistency
- Vision-based detail extraction
- Multi-persona group scene support
- Deterministic prompt synthesis based on scene analysis
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# Base paths
try:
    from aither_adk.paths import get_saga_data_dir, get_saga_subdir
    NARRATIVE_AGENT_DIR = get_saga_data_dir()
    ANCHORS_DIR = get_saga_subdir("memory", "anchors", create=True)
except ImportError:
    AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    NARRATIVE_AGENT_DIR = os.path.join(AGENT_DIR, "Saga")
    ANCHORS_DIR = os.path.join(NARRATIVE_AGENT_DIR, "memory", "anchors")
    try:
        os.makedirs(ANCHORS_DIR, exist_ok=True)
    except OSError:
        pass
STATE_FILE = os.path.join(NARRATIVE_AGENT_DIR, "memory", "persona_image_state.json")


@dataclass
class VisualIdentity:
    """Core visual identity for consistent character generation."""
    # Face
    face_shape: str = ""
    eye_color: str = ""
    eye_style: str = ""
    eyebrows: str = ""
    nose: str = ""
    lips: str = ""
    skin_tone: str = ""
    facial_features: str = ""  # e.g., "beauty mark", "freckles", "glasses"

    # Hair
    hair_color: str = ""
    hair_style: str = ""
    hair_length: str = ""
    hair_details: str = ""

    # Body
    body_type: str = ""
    height: str = ""
    distinguishing_marks: str = ""  # e.g., "tattoo on shoulder", "scar"

    # Style
    art_style: str = "anime"
    quality_tags: str = "masterpiece, best quality, highly detailed"

    def to_prompt_tags(self) -> str:
        """Convert identity to prompt tags."""
        tags = []

        if self.face_shape:
            tags.append(self.face_shape)
        if self.eye_color:
            tags.append(f"{self.eye_color} eyes")
        if self.eye_style:
            tags.append(self.eye_style)
        if self.skin_tone:
            tags.append(f"{self.skin_tone} skin")
        if self.facial_features:
            tags.append(self.facial_features)

        if self.hair_color:
            tags.append(f"{self.hair_color} hair")
        if self.hair_style:
            tags.append(self.hair_style)
        if self.hair_length:
            tags.append(f"{self.hair_length} hair")

        if self.body_type:
            tags.append(self.body_type)
        if self.distinguishing_marks:
            tags.append(self.distinguishing_marks)

        return ", ".join(tags)


@dataclass
class PersonaAnchor:
    """Reference data for a persona's visual consistency."""
    name: str
    display_name: str = ""

    identity: VisualIdentity = field(default_factory=VisualIdentity)

    # Reference images
    face_reference: str = ""
    body_reference: str = ""
    style_reference: str = ""

    # Extracted descriptions (from vision model analysis)
    face_description: str = ""
    body_description: str = ""
    style_description: str = ""

    base_prompt_template: str = ""
    negative_prompt: str = "bad anatomy, bad hands, missing fingers, extra digits, blurry, low quality"
    exclusions: List[str] = field(default_factory=list)
    preferred_model: str = "flux"
    default_style: str = "anime"

    updated_at: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.title()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


@dataclass
class SceneContext:
    """Current scene context for consistency."""
    location: str = ""
    lighting: str = ""
    time_of_day: str = ""
    atmosphere: str = ""
    camera_angle: str = ""

    active_personas: List[str] = field(default_factory=list)
    shared_elements: str = ""
    character_states: Dict[str, str] = field(default_factory=dict)

    def to_prompt_tags(self) -> str:
        """Convert scene context to prompt tags."""
        tags = []
        if self.location:
            tags.append(self.location)
        if self.lighting:
            tags.append(self.lighting)
        if self.time_of_day:
            tags.append(self.time_of_day)
        if self.atmosphere:
            tags.append(self.atmosphere)
        if self.shared_elements:
            tags.append(self.shared_elements)
        return ", ".join(tags)


class PersonaImageSystem:
    """
    Main system for persona-consistent image generation.

    Features:
    - Anchor image management (face/style references)
    - Vision-based identity extraction
    - LLM-powered prompt synthesis
    - ControlNet integration for consistency
    - Multi-persona scene management
    """

    def __init__(self):
        self.anchors: Dict[str, PersonaAnchor] = {}
        self.scene: SceneContext = SceneContext()
        self._load_state()

    def _load_state(self):
        """Load saved anchors and scene state."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)

                for name, anchor_data in data.get("anchors", {}).items():
                    identity_data = anchor_data.pop("identity", {})
                    identity = VisualIdentity(**identity_data)
                    self.anchors[name] = PersonaAnchor(identity=identity, **anchor_data)

                scene_data = data.get("scene", {})
                self.scene = SceneContext(**scene_data)

            except Exception as e:
                print(f"[PersonaImageSystem] Error loading state: {e}")

    def _save_state(self):
        """Save anchors and scene state."""
        try:
            data = {
                "anchors": {},
                "scene": asdict(self.scene)
            }

            for name, anchor in self.anchors.items():
                data["anchors"][name] = asdict(anchor)

            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[PersonaImageSystem] Error saving state: {e}")

    # ========== Anchor Management ==========

    def set_anchor_image(self, persona_name: str, image_path: str, anchor_type: str = "face") -> bool:
        """Set a reference anchor image for a persona."""
        if not os.path.exists(image_path):
            print(f"[PersonaImageSystem] Image not found: {image_path}")
            return False

        name_lower = persona_name.lower()
        if name_lower not in self.anchors:
            self.anchors[name_lower] = PersonaAnchor(name=name_lower)

        anchor = self.anchors[name_lower]

        ext = os.path.splitext(image_path)[1]
        dest_filename = f"{name_lower}_{anchor_type}{ext}"
        dest_path = os.path.join(ANCHORS_DIR, dest_filename)

        import shutil
        shutil.copy2(image_path, dest_path)

        if anchor_type == "face":
            anchor.face_reference = dest_path
        elif anchor_type == "body":
            anchor.body_reference = dest_path
        elif anchor_type == "style":
            anchor.style_reference = dest_path

        anchor.updated_at = datetime.now().isoformat()
        self._analyze_anchor_image(name_lower, anchor_type)
        self._save_state()
        print(f"[PersonaImageSystem] Set {anchor_type} anchor for {persona_name}")
        return True

    def _analyze_anchor_image(self, persona_name: str, anchor_type: str):
        """Use vision model to extract description from anchor image."""
        anchor = self.anchors.get(persona_name)
        if not anchor:
            return

        image_path = None
        if anchor_type == "face" and anchor.face_reference:
            image_path = anchor.face_reference
        elif anchor_type == "body" and anchor.body_reference:
            image_path = anchor.body_reference
        elif anchor_type == "style" and anchor.style_reference:
            image_path = anchor.style_reference

        if not image_path or not os.path.exists(image_path):
            return

        try:
            from AitherOS.AitherNode.vision_tools import analyze_with_ollama

            prompts = {
                "face": """Analyze this character's face. Output ONLY comma-separated tags.

EXTRACT:
- Eye color and shape
- Hair color, style, length
- Skin tone
- Face shape
- Distinguishing features (glasses, marks, etc.)
- Accessories

OUTPUT ONLY TAGS, NO SENTENCES.""",
                "body": """Analyze this character's body type. Output ONLY comma-separated tags.

EXTRACT:
- Body type (athletic, slim, etc.)
- Height impression
- Distinguishing marks (tattoo, scar, etc.)

OUTPUT ONLY TAGS, NO SENTENCES.""",
                "style": """Analyze this image's art style. Output ONLY comma-separated tags.

EXTRACT:
- Art style (anime, realistic, etc.)
- Color palette
- Lighting style
- Overall aesthetic

OUTPUT ONLY TAGS, NO SENTENCES."""
            }

            prompt = prompts.get(anchor_type, "Describe this image as comma-separated tags.")

            result = analyze_with_ollama(image_path, prompt, auto_unload=True)

            if result:
                if anchor_type == "face":
                    anchor.face_description = result
                    self._parse_face_to_identity(anchor, result)
                elif anchor_type == "body":
                    anchor.body_description = result
                    self._parse_body_to_identity(anchor, result)
                elif anchor_type == "style":
                    anchor.style_description = result

                self._save_state()

        except Exception as e:
            print(f"[PersonaImageSystem] Vision analysis failed: {e}")

    def _parse_face_to_identity(self, anchor: PersonaAnchor, description: str):
        """Parse face description into structured identity."""
        desc_lower = description.lower()

        eye_colors = ["blue", "green", "brown", "amber", "hazel", "gray", "purple", "red", "golden"]
        for color in eye_colors:
            if color in desc_lower:
                anchor.identity.eye_color = color
                break

        hair_colors = ["black", "brown", "blonde", "red", "white", "silver", "pink", "blue", "purple", "auburn"]
        for color in hair_colors:
            if color in desc_lower and "hair" in desc_lower:
                anchor.identity.hair_color = color
                break

        hair_styles = ["ponytail", "braids", "bun", "twintails", "bob", "long", "short", "messy", "straight", "wavy", "curly"]
        for style in hair_styles:
            if style in desc_lower:
                anchor.identity.hair_style = style
                break

        skin_tones = ["pale", "fair", "tan", "dark", "olive", "porcelain"]
        for tone in skin_tones:
            if tone in desc_lower:
                anchor.identity.skin_tone = tone
                break

        if "glasses" in desc_lower:
            anchor.identity.facial_features = "glasses, " + anchor.identity.facial_features

    def _parse_body_to_identity(self, anchor: PersonaAnchor, description: str):
        """Parse body description into structured identity."""
        desc_lower = description.lower()

        body_types = ["athletic", "slender", "petite", "muscular", "slim", "fit", "toned", "stocky", "tall"]
        for btype in body_types:
            if btype in desc_lower:
                anchor.identity.body_type = btype
                break

        marks = ["tattoo", "scar", "birthmark", "freckles"]
        found_marks = [m for m in marks if m in desc_lower]
        if found_marks:
            anchor.identity.distinguishing_marks = ", ".join(found_marks)

    def get_anchor(self, persona_name: str) -> Optional[PersonaAnchor]:
        """Get anchor data for a persona."""
        return self.anchors.get(persona_name.lower())

    def create_persona_from_yaml(self, persona_name: str, auto_generate_anchor: bool = False) -> Optional[PersonaAnchor]:
        """Create a PersonaAnchor from persona YAML config."""
        try:
            import yaml

            persona_path = os.path.join(
                NARRATIVE_AGENT_DIR, "config", "personas", f"{persona_name.lower()}.yaml"
            )

            if not os.path.exists(persona_path):
                return None

            with open(persona_path, 'r') as f:
                data = yaml.safe_load(f)

            instruction = data.get("instruction", "")

            visual_desc = ""
            if "[VISUAL DESCRIPTION]" in instruction:
                parts = instruction.split("[VISUAL DESCRIPTION]")
                if len(parts) > 1:
                    visual_section = parts[1]
                    if "[" in visual_section:
                        visual_section = visual_section.split("[")[0]
                    visual_desc = visual_section.strip()

            anchor = PersonaAnchor(
                name=persona_name.lower(),
                display_name=persona_name.title(),
                base_prompt_template=visual_desc
            )

            self._parse_visual_description_to_identity(anchor, visual_desc)

            self.anchors[persona_name.lower()] = anchor
            self._save_state()

            if auto_generate_anchor and not anchor.face_reference:
                self._auto_generate_anchor(anchor)

            return anchor

        except Exception as e:
            print(f"[PersonaImageSystem] Failed to load persona YAML: {e}")
            return None

    def _auto_generate_anchor(self, anchor: PersonaAnchor) -> bool:
        """Auto-generate an anchor reference image."""
        try:
            identity_tags = anchor.identity.to_prompt_tags()

            ref_prompt = (
                f"1girl, solo, {anchor.display_name}, portrait, face focus, "
                f"{identity_tags}, looking at viewer, neutral expression, soft smile, "
                f"studio lighting, clean background, white background, "
                f"anime style, masterpiece, best quality, highly detailed face, sharp focus"
            )

            ref_negative = (
                "bad anatomy, bad face, ugly, deformed, blurry, "
                "low quality, multiple people, extra limbs, watermark, text, "
                "complex background, busy background"
            )

            import asyncio
            from AitherOS.AitherNode.AitherCanvas import generate_local

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            paths = loop.run_until_complete(
                generate_local(ref_prompt, None, negative_prompt=ref_negative)
            )

            if paths and len(paths) > 0:
                import shutil
                dest_path = os.path.join(ANCHORS_DIR, f"{anchor.name}_face_auto.png")
                shutil.copy2(paths[0], dest_path)
                anchor.face_reference = dest_path
                anchor.updated_at = datetime.now().isoformat()
                self._analyze_anchor_image(anchor.name, "face")
                self._save_state()
                return True

            return False

        except Exception as e:
            print(f"[PersonaImageSystem] Auto-anchor generation failed: {e}")
            return False

    def initialize_all_personas(self, auto_generate: bool = False) -> Dict[str, bool]:
        """Initialize all personas from YAML files."""
        personas_dir = os.path.join(NARRATIVE_AGENT_DIR, "config", "personas")
        results = {}

        if not os.path.exists(personas_dir):
            return results

        skip = {"router", "security", "debugger", "architect"}
        for filename in os.listdir(personas_dir):
            if filename.endswith(".yaml"):
                persona_name = filename[:-5]
                if persona_name in skip:
                    continue
                try:
                    anchor = self.create_persona_from_yaml(persona_name, auto_generate_anchor=auto_generate)
                    results[persona_name] = anchor is not None
                except Exception as e:
                    print(f"[PersonaImageSystem] Failed to init {persona_name}: {e}")
                    results[persona_name] = False

        return results

    def _parse_visual_description_to_identity(self, anchor: PersonaAnchor, description: str):
        """Parse YAML visual description into structured identity."""
        desc_lower = description.lower()

        if "ponytail" in desc_lower:
            anchor.identity.hair_style = "high ponytail"
        if "long" in desc_lower and "hair" in desc_lower:
            anchor.identity.hair_length = "long"

        if "athletic" in desc_lower:
            anchor.identity.body_type = "athletic"
        elif "slim" in desc_lower:
            anchor.identity.body_type = "slim"

        if "glasses" in desc_lower:
            anchor.identity.facial_features = "thin-framed glasses"

    # ========== Prompt Generation ==========

    def build_prompt(
        self,
        persona_name: str,
        user_request: str,
        include_scene: bool = True,
        override_pose: str = None,
        override_clothes: str = None,
        override_expression: str = None,
        use_controlnet: bool = True
    ) -> Dict[str, Any]:
        """
        Build a complete, consistent prompt for image generation.

        Returns:
            Dict with prompt, negative_prompt, controlnet_image, controlnet_model,
            model_preference, seed, persona_name, modifications
        """
        name_lower = persona_name.lower()

        anchor = self.anchors.get(name_lower)
        if not anchor:
            anchor = self.create_persona_from_yaml(name_lower)
        if not anchor:
            anchor = PersonaAnchor(name=name_lower)
            self.anchors[name_lower] = anchor

        modifications = self._analyze_request(user_request)

        parts = []

        # 1. Subject
        parts.append("1girl, solo")
        parts.append(anchor.display_name)

        # 2. Core identity
        identity_tags = anchor.identity.to_prompt_tags()
        if identity_tags:
            parts.append(f"({identity_tags}:1.15)")

        # 3. Pose
        pose = override_pose or modifications.get("pose", "")
        if pose:
            parts.append(pose)

        # 4. Clothing/outfit
        clothes = override_clothes or modifications.get("clothing", "")
        if clothes:
            parts.append(clothes)

        # 5. Expression
        expression = override_expression or modifications.get("expression", "")
        if expression:
            parts.append(expression)

        # 6. Camera/framing
        camera = modifications.get("camera", "")
        if camera:
            parts.append(camera)

        # 7. Scene context
        if include_scene:
            if self.scene.location:
                scene_tags = self.scene.to_prompt_tags()
                if scene_tags:
                    parts.append(scene_tags)
            else:
                parts.append("high-tech office, neon ambient lighting, cyan and magenta")

        # 8. Style and quality
        parts.append(anchor.identity.art_style or "anime style")
        parts.append("masterpiece, best quality, highly detailed, detailed face")

        # Build negative
        negative_parts = [
            "bad anatomy, bad hands, missing fingers, extra digits, extra limbs, missing limbs",
            "fused fingers, too many fingers, poorly drawn hands, poorly drawn face",
            "ugly, deformed, disfigured, mutated",
            "2girls, multiple girls, 3girls, multiple people, duo, group",
            "blurry, low quality, text, watermark, signature",
            "multiple views, split screen, collage, comic, panel",
            "cropped, out of frame"
        ]

        for exclusion in anchor.exclusions:
            negative_parts.append(exclusion)

        # Pose-specific negatives
        if "from behind" in (pose or "").lower() or "back view" in (pose or "").lower():
            negative_parts.append("front view, facing viewer")
        if "front" in (pose or "").lower() or "looking at viewer" in (pose or "").lower():
            negative_parts.append("from behind, back view")

        # ControlNet
        controlnet_image = None
        controlnet_model = None
        if use_controlnet:
            if anchor.face_reference and os.path.exists(anchor.face_reference):
                controlnet_image = anchor.face_reference
                controlnet_model = "ip_adapter_face"
            elif anchor.body_reference and os.path.exists(anchor.body_reference):
                controlnet_image = anchor.body_reference
                controlnet_model = "ip_adapter"

        seed = self._generate_identity_seed(anchor)

        prompt_parts = [p for p in parts if p and p.strip()]
        negative_prompt_parts = [p for p in negative_parts if p and p.strip()]

        return {
            "prompt": ", ".join(prompt_parts),
            "negative_prompt": ", ".join(negative_prompt_parts),
            "controlnet_image": controlnet_image,
            "controlnet_model": controlnet_model,
            "model_preference": anchor.preferred_model,
            "seed": seed,
            "persona_name": persona_name,
            "modifications": modifications
        }

    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """Analyze user request to extract pose, clothing, expression, camera."""
        request_lower = request.lower()
        result = {
            "pose": "",
            "clothing": "",
            "expression": "",
            "camera": "",
            "extra_tags": []
        }

        # === POSES ===
        pose_map = {
            "from behind": "(from behind:1.3), (back view:1.2), (looking back:1.1)",
            "rear view": "(from behind:1.3), (back view:1.2), (looking back:1.1)",
            "selfie": "(selfie:1.3), (holding phone:1.2), (POV:1.2), (looking at viewer:1.2)",
            "leaning": "(leaning forward:1.2), (casual pose:1.1)",
            "kneeling": "(kneeling:1.3)",
            "lying": "(lying down:1.2)",
            "reclining": "(reclining:1.2), (relaxed pose:1.1)",
            "sitting": "(sitting:1.2), (seated:1.1)",
            "standing": "(standing:1.2), (full body:1.1)",
            "action": "(dynamic pose:1.3), (action:1.2), (motion:1.1)",
            "fighting": "(fighting stance:1.3), (combat pose:1.2)",
            "running": "(running:1.3), (dynamic:1.2)",
            "jumping": "(jumping:1.3), (mid-air:1.2)",
            "dancing": "(dancing:1.3), (graceful pose:1.2)",
        }

        for keyword, tags in pose_map.items():
            if keyword in request_lower and not result["pose"]:
                result["pose"] = tags
                break

        # === CLOTHING ===
        clothing_map = {
            "armor": "(armor:1.3), (plate armor:1.2)",
            "dress": "(dress:1.2), (elegant:1.1)",
            "casual": "(casual clothes:1.1), (t-shirt:1.0)",
            "formal": "(formal attire:1.2), (suit:1.1)",
            "uniform": "(uniform:1.2)",
            "swimsuit": "(swimsuit:1.2)",
            "jacket": "(jacket:1.2), (coat:1.1)",
        }

        for keyword, tags in clothing_map.items():
            if keyword in request_lower:
                result["clothing"] = tags
                break

        # === EXPRESSIONS ===
        expression_map = {
            "smile": "(smiling:1.2), (happy:1.1)",
            "smirk": "(smirking:1.2), (confident:1.1)",
            "angry": "(angry:1.2), (frowning:1.1)",
            "surprised": "(surprised:1.2), (wide eyes:1.1)",
            "sad": "(sad:1.2), (melancholy:1.1)",
            "determined": "(determined expression:1.2), (focused:1.1)",
            "laughing": "(laughing:1.2), (joyful:1.1)",
            "serious": "(serious expression:1.2), (stern:1.1)",
            "crying": "(crying:1.3), (tears:1.2)",
            "blushing": "(blushing:1.2), (embarrassed:1.1)",
        }

        for keyword, tags in expression_map.items():
            if keyword in request_lower:
                result["expression"] = tags
                break

        # === CAMERA ===
        camera_map = {
            "pov": "(POV:1.3), (first person view:1.2)",
            "close up": "(close up:1.3), (face focus:1.2)",
            "closeup": "(close up:1.3), (face focus:1.2)",
            "portrait": "(portrait:1.2), (upper body:1.1)",
            "full body": "(full body:1.2), (wide shot:1.1)",
            "from above": "(from above:1.3), (high angle:1.2)",
            "from below": "(from below:1.3), (low angle:1.2)",
            "side view": "(side view:1.2), (profile:1.1)",
        }

        for keyword, tags in camera_map.items():
            if keyword in request_lower:
                result["camera"] = tags
                break

        return result

    def _generate_identity_seed(self, anchor: PersonaAnchor) -> int:
        """Generate a deterministic seed based on persona identity."""
        identity_string = f"{anchor.name}_{anchor.identity.to_prompt_tags()}"
        hash_obj = hashlib.sha256(identity_string.encode())
        return int(hash_obj.hexdigest()[:8], 16)

    # ========== Vision-Enhanced Prompt Generation ==========

    def enhance_prompt_with_vision(
        self,
        base_image_path: str,
        modification_request: str,
        persona_name: str = None
    ) -> Dict[str, Any]:
        """
        Use vision model to analyze an existing image and build a prompt
        that preserves its details while applying requested modifications.
        """
        try:
            from AitherOS.AitherNode.vision_tools import analyze_with_ollama

            extraction_prompt = """Analyze this image and output ONLY comma-separated tags.

EXTRACT:
1. Character appearance: hair color, hair style, eye color, skin tone, body type
2. Clothing/outfit
3. Pose: body position, where they're looking
4. Expression: facial expression, emotion
5. Setting: location, background, lighting
6. Camera: angle, framing, composition

OUTPUT ONLY TAGS. NO SENTENCES."""

            current_description = analyze_with_ollama(base_image_path, extraction_prompt, auto_unload=True)

            if not current_description:
                return self.build_prompt(persona_name or "aither", modification_request)

            keep_tags, _ = self._parse_modification_intent(
                current_description, modification_request
            )

            parts = []
            parts.append(keep_tags)

            modifications = self._analyze_request(modification_request)

            if modifications.get("pose"):
                parts.append(f"({modifications['pose']}:1.2)")
            if modifications.get("clothing"):
                parts.append(modifications["clothing"])
            if modifications.get("expression"):
                parts.append(f"({modifications['expression']}:1.1)")
            if modifications.get("camera"):
                parts.append(modifications["camera"])

            parts.append("masterpiece, best quality, highly detailed, anime style")

            return {
                "prompt": ", ".join(parts),
                "negative_prompt": "bad anatomy, bad hands, blurry, low quality, multiple views",
                "controlnet_image": base_image_path,
                "controlnet_model": "ip_adapter",
                "original_description": current_description,
                "modifications": modifications
            }

        except Exception as e:
            print(f"[PersonaImageSystem] Vision enhancement failed: {e}")
            return self.build_prompt(persona_name or "aither", modification_request)

    def _parse_modification_intent(
        self,
        current_description: str,
        modification_request: str
    ) -> Tuple[str, str]:
        """Determine which tags to keep and which to change."""
        request_lower = modification_request.lower()
        desc_tags = [t.strip() for t in current_description.split(",")]

        pose_keywords = ["standing", "sitting", "lying", "kneeling", "leaning", "from behind", "reclining"]
        clothing_keywords = ["dress", "shirt", "pants", "armor", "uniform", "jacket", "suit"]
        expression_keywords = ["smile", "smirk", "angry", "sad", "neutral", "surprised"]

        keep_tags = []
        remove_categories = set()

        if any(kw in request_lower for kw in ["pose", "position"] + pose_keywords):
            remove_categories.add("pose")
        if any(kw in request_lower for kw in ["clothes", "clothing", "wear", "outfit"] + clothing_keywords):
            remove_categories.add("clothing")
        if any(kw in request_lower for kw in ["expression", "face", "emotion"] + expression_keywords):
            remove_categories.add("expression")

        for tag in desc_tags:
            tag_lower = tag.lower()
            should_remove = False

            if "pose" in remove_categories and any(kw in tag_lower for kw in pose_keywords):
                should_remove = True
            if "clothing" in remove_categories and any(kw in tag_lower for kw in clothing_keywords):
                should_remove = True
            if "expression" in remove_categories and any(kw in tag_lower for kw in expression_keywords):
                should_remove = True

            if not should_remove:
                keep_tags.append(tag)

        return ", ".join(keep_tags), ", ".join(remove_categories)

    # ========== Scene Management ==========

    def set_scene(self, location: str = None, lighting: str = None,
                  time_of_day: str = None, atmosphere: str = None,
                  personas: List[str] = None):
        """Update the current scene context."""
        if location:
            self.scene.location = location
        if lighting:
            self.scene.lighting = lighting
        if time_of_day:
            self.scene.time_of_day = time_of_day
        if atmosphere:
            self.scene.atmosphere = atmosphere
        if personas:
            self.scene.active_personas = personas
        self._save_state()

    def update_character_state(self, persona_name: str, state: str):
        """Update a character's state in the current scene."""
        self.scene.character_states[persona_name.lower()] = state
        self._save_state()

    def clear_scene(self):
        """Clear the current scene context."""
        self.scene = SceneContext()
        self._save_state()

    # ========== Group Generation ==========

    def build_group_prompt(self, persona_names: List[str], user_request: str,
                           shared_pose: str = None) -> Dict[str, Any]:
        """Build a prompt for multiple characters in the same scene."""
        parts = []

        count = len(persona_names)
        count_tag = {1: "1girl", 2: "2girls", 3: "3girls"}.get(count, f"{count}girls")
        parts.append(count_tag)

        for name in persona_names:
            anchor = self.get_anchor(name) or self.create_persona_from_yaml(name)
            if anchor:
                identity_tags = anchor.identity.to_prompt_tags()
                if identity_tags:
                    parts.append(f"({anchor.display_name}: {identity_tags})")

        if self.scene.location:
            parts.append(self.scene.to_prompt_tags())

        if shared_pose:
            parts.append(f"({shared_pose}:1.2)")

        modifications = self._analyze_request(user_request)
        if modifications.get("pose") and not shared_pose:
            parts.append(f"({modifications['pose']}:1.2)")

        parts.append("masterpiece, best quality, highly detailed, anime style")

        return {
            "prompt": ", ".join(parts),
            "negative_prompt": "bad anatomy, bad hands, blurry, low quality, solo, 1girl, multiple views",
            "personas": persona_names,
            "scene": asdict(self.scene)
        }


# Singleton
_system: Optional[PersonaImageSystem] = None


def get_persona_image_system() -> PersonaImageSystem:
    """Get the singleton PersonaImageSystem instance."""
    global _system
    if _system is None:
        _system = PersonaImageSystem()
    return _system


# Convenience functions

def set_persona_anchor(persona_name: str, image_path: str, anchor_type: str = "face") -> bool:
    return get_persona_image_system().set_anchor_image(persona_name, image_path, anchor_type)


def generate_persona_prompt(persona_name: str, request: str, **kwargs) -> Dict[str, Any]:
    return get_persona_image_system().build_prompt(persona_name, request, **kwargs)


def enhance_with_vision(image_path: str, request: str, persona: str = None) -> Dict[str, Any]:
    return get_persona_image_system().enhance_prompt_with_vision(image_path, request, persona)


def set_scene_context(location: str = None, lighting: str = None, **kwargs):
    get_persona_image_system().set_scene(location=location, lighting=lighting, **kwargs)
