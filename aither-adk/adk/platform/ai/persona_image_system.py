"""
AitherOS Persona Image Generation System

Advanced LLM-reasoning enhanced prompt generation with:
- Anchor reference images for consistent face/body
- ControlNet integration for pose/face consistency
- Vision-based detail extraction
- Multi-persona group scene support
- Deterministic prompt synthesis based on scene analysis
"""

import os
import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# Base paths — use paths.py for Docker-safe writable directory resolution
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
        pass  # Read-only filesystem (Docker)
STATE_FILE = os.path.join(NARRATIVE_AGENT_DIR, "memory", "persona_image_state.json")


@dataclass
class VisualIdentity:
    """Core visual identity that should remain consistent across generations."""
    # Face
    face_shape: str = ""
    eye_color: str = ""
    eye_style: str = ""  # e.g., "almond shaped", "large anime eyes"
    eyebrows: str = ""
    nose: str = ""
    lips: str = ""
    skin_tone: str = ""
    facial_features: str = ""  # e.g., "beauty mark", "freckles"
    
    # Hair
    hair_color: str = ""
    hair_style: str = ""
    hair_length: str = ""
    hair_details: str = ""  # e.g., "highlights", "gradient"
    
    # Body
    body_type: str = ""
    height: str = ""
    bust_size: str = ""
    hip_width: str = ""
    distinguishing_marks: str = ""  # e.g., "tanlines", "tattoo on shoulder"
    
    # Style
    art_style: str = "anime"
    quality_tags: str = "masterpiece, best quality, highly detailed"
    
    def to_prompt_tags(self) -> str:
        """Convert identity to prompt tags."""
        tags = []
        
        # Face tags
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
        
        # Hair tags
        if self.hair_color:
            tags.append(f"{self.hair_color} hair")
        if self.hair_style:
            tags.append(self.hair_style)
        if self.hair_length:
            tags.append(f"{self.hair_length} hair")
        
        # Body tags
        if self.body_type:
            tags.append(self.body_type)
        if self.bust_size:
            tags.append(self.bust_size)
        if self.hip_width:
            tags.append(self.hip_width)
        if self.distinguishing_marks:
            tags.append(self.distinguishing_marks)
        
        return ", ".join(tags)


@dataclass
class PersonaAnchor:
    """Reference data for a persona's visual consistency."""
    name: str
    display_name: str = ""
    
    # Core visual identity (extracted from reference images)
    identity: VisualIdentity = field(default_factory=VisualIdentity)
    
    # Reference images
    face_reference: str = ""  # Path to face anchor image
    body_reference: str = ""  # Path to body anchor image
    style_reference: str = ""  # Path to style/aesthetic anchor
    
    # Extracted descriptions (from vision model analysis)
    face_description: str = ""
    body_description: str = ""
    style_description: str = ""
    
    # Full prompt template
    base_prompt_template: str = ""
    
    # Negative prompt (things to always exclude)
    negative_prompt: str = "bad anatomy, bad hands, missing fingers, extra digits, blurry, low quality"
    
    # Character-specific exclusions
    exclusions: List[str] = field(default_factory=list)
    
    # Generation settings
    preferred_model: str = "pony"  # pony, flux, sdxl
    default_style: str = "anime"
    
    # Last updated
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.title()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


@dataclass
class SceneContext:
    """Current scene context for group consistency."""
    location: str = ""
    lighting: str = ""
    time_of_day: str = ""
    atmosphere: str = ""
    camera_angle: str = ""
    
    # Active characters in scene
    active_personas: List[str] = field(default_factory=list)
    
    # Shared scene elements
    shared_elements: str = ""
    
    # Last action/pose per character
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
    - Anchor image management (face/body references)
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
                
                # Load anchors
                for name, anchor_data in data.get("anchors", {}).items():
                    identity_data = anchor_data.pop("identity", {})
                    identity = VisualIdentity(**identity_data)
                    self.anchors[name] = PersonaAnchor(identity=identity, **anchor_data)
                
                # Load scene
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
                anchor_dict = asdict(anchor)
                data["anchors"][name] = anchor_dict
            
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[PersonaImageSystem] Error saving state: {e}")
    
    # ========== Anchor Management ==========
    
    def set_anchor_image(self, persona_name: str, image_path: str, anchor_type: str = "face") -> bool:
        """
        Set a reference anchor image for a persona.
        
        Args:
            persona_name: Name of the persona (e.g., "aither", "leo")
            image_path: Path to the reference image
            anchor_type: "face", "body", or "style"
        
        Returns:
            True if successful
        """
        if not os.path.exists(image_path):
            print(f"[PersonaImageSystem] Image not found: {image_path}")
            return False
        
        # Get or create anchor
        name_lower = persona_name.lower()
        if name_lower not in self.anchors:
            self.anchors[name_lower] = PersonaAnchor(name=name_lower)
        
        anchor = self.anchors[name_lower]
        
        # Copy image to anchors directory
        ext = os.path.splitext(image_path)[1]
        dest_filename = f"{name_lower}_{anchor_type}{ext}"
        dest_path = os.path.join(ANCHORS_DIR, dest_filename)
        
        import shutil
        shutil.copy2(image_path, dest_path)
        
        # Update anchor
        if anchor_type == "face":
            anchor.face_reference = dest_path
        elif anchor_type == "body":
            anchor.body_reference = dest_path
        elif anchor_type == "style":
            anchor.style_reference = dest_path
        
        anchor.updated_at = datetime.now().isoformat()
        
        # Auto-analyze the image
        self._analyze_anchor_image(name_lower, anchor_type)
        
        self._save_state()
        print(f"[PersonaImageSystem] Set {anchor_type} anchor for {persona_name}")
        return True
    
    def _analyze_anchor_image(self, persona_name: str, anchor_type: str):
        """
        Use vision model to extract detailed description from anchor image.
        """
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
            
            # Detailed prompts for each anchor type
            prompts = {
                "face": """Analyze this character's face in EXTREME detail. Output ONLY comma-separated tags.

REQUIRED DETAILS:
- Eye color (exact shade: e.g., "bright blue eyes", "amber eyes", "heterochromia")
- Eye shape (e.g., "almond eyes", "large round eyes", "narrow eyes")
- Hair color (exact: e.g., "dark brown hair", "platinum blonde", "black hair with blue highlights")
- Hair style (e.g., "high ponytail", "twin braids", "messy bob")
- Skin tone (e.g., "pale skin", "tan skin", "sun-kissed skin")
- Face shape (e.g., "oval face", "heart-shaped face")
- Distinguishing features (e.g., "beauty mark under eye", "thin eyebrows", "full lips")
- Accessories (e.g., "thin-framed glasses", "earrings", "choker")

OUTPUT ONLY TAGS, NO SENTENCES.""",
                
                "body": """Analyze this character's body in EXTREME detail. Output ONLY comma-separated tags.

REQUIRED DETAILS:
- Body type (e.g., "athletic build", "curvy figure", "slender")
- Bust size (e.g., "small breasts", "large breasts", "flat chest")
- Hip/waist (e.g., "wide hips", "narrow waist", "hourglass figure")
- Skin details (e.g., "bikini tanlines", "freckled shoulders", "tattoo on arm")
- Height impression (e.g., "tall", "petite")
- Muscle definition (e.g., "toned abs", "defined arms")

OUTPUT ONLY TAGS, NO SENTENCES.""",
                
                "style": """Analyze this image's art style and aesthetic. Output ONLY comma-separated tags.

REQUIRED DETAILS:
- Art style (e.g., "anime style", "semi-realistic", "cel shaded")
- Color palette (e.g., "vibrant colors", "muted tones", "neon accents")
- Lighting style (e.g., "soft lighting", "dramatic shadows", "rim lighting")
- Line quality (e.g., "clean lines", "sketchy", "detailed linework")
- Overall aesthetic (e.g., "cyberpunk aesthetic", "pastel aesthetic")

OUTPUT ONLY TAGS, NO SENTENCES."""
            }
            
            prompt = prompts.get(anchor_type, "Describe this image in detail as comma-separated tags.")
            
            # Analyze with vision model
            result = analyze_with_ollama(image_path, prompt, auto_unload=True)
            
            if result:
                # Store the description
                if anchor_type == "face":
                    anchor.face_description = result
                    self._parse_face_to_identity(anchor, result)
                elif anchor_type == "body":
                    anchor.body_description = result
                    self._parse_body_to_identity(anchor, result)
                elif anchor_type == "style":
                    anchor.style_description = result
                
                print(f"[PersonaImageSystem] Analyzed {anchor_type} for {persona_name}: {result[:100]}...")
                self._save_state()
                
        except Exception as e:
            print(f"[PersonaImageSystem] Vision analysis failed: {e}")
    
    def _parse_face_to_identity(self, anchor: PersonaAnchor, description: str):
        """Parse face description into structured identity."""
        desc_lower = description.lower()
        
        # Eye color
        eye_colors = ["blue", "green", "brown", "amber", "hazel", "gray", "purple", "red", "golden", "heterochromia"]
        for color in eye_colors:
            if color in desc_lower:
                anchor.identity.eye_color = color
                break
        
        # Hair color
        hair_colors = ["black", "brown", "blonde", "red", "white", "silver", "pink", "blue", "purple", "green", "auburn", "platinum"]
        for color in hair_colors:
            if color in desc_lower and "hair" in desc_lower:
                anchor.identity.hair_color = color
                break
        
        # Hair style
        hair_styles = ["ponytail", "braids", "bun", "twintails", "pigtails", "bob", "long", "short", "messy", "straight", "wavy", "curly"]
        for style in hair_styles:
            if style in desc_lower:
                anchor.identity.hair_style = style
                break
        
        # Skin tone
        skin_tones = ["pale", "fair", "tan", "dark", "sun-kissed", "olive", "porcelain"]
        for tone in skin_tones:
            if tone in desc_lower:
                anchor.identity.skin_tone = tone
                break
        
        # Glasses
        if "glasses" in desc_lower:
            anchor.identity.facial_features = "glasses, " + anchor.identity.facial_features
    
    def _parse_body_to_identity(self, anchor: PersonaAnchor, description: str):
        """Parse body description into structured identity."""
        desc_lower = description.lower()
        
        # Body type
        body_types = ["athletic", "curvy", "slender", "petite", "muscular", "voluptuous", "slim", "fit", "toned"]
        for btype in body_types:
            if btype in desc_lower:
                anchor.identity.body_type = btype
                break
        
        # Bust
        bust_terms = ["small breasts", "medium breasts", "large breasts", "flat chest", "busty", "perky"]
        for term in bust_terms:
            if term in desc_lower:
                anchor.identity.bust_size = term
                break
        
        # Hips
        hip_terms = ["wide hips", "narrow hips", "thick thighs", "hourglass"]
        for term in hip_terms:
            if term in desc_lower:
                anchor.identity.hip_width = term
                break
        
        # Distinguishing marks
        marks = ["tanlines", "tattoo", "scar", "birthmark", "freckles"]
        found_marks = [m for m in marks if m in desc_lower]
        if found_marks:
            anchor.identity.distinguishing_marks = ", ".join(found_marks)
    
    def get_anchor(self, persona_name: str) -> Optional[PersonaAnchor]:
        """Get anchor data for a persona."""
        return self.anchors.get(persona_name.lower())
    
    def create_persona_from_yaml(self, persona_name: str, auto_generate_anchor: bool = False) -> Optional[PersonaAnchor]:
        """
        Create a PersonaAnchor from the existing YAML persona config.
        
        Args:
            persona_name: Name of the persona
            auto_generate_anchor: If True, generate an anchor image automatically
        """
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
            
            # Extract visual description section
            visual_desc = ""
            if "[VISUAL DESCRIPTION]" in instruction:
                parts = instruction.split("[VISUAL DESCRIPTION]")
                if len(parts) > 1:
                    visual_section = parts[1]
                    if "[" in visual_section:
                        visual_section = visual_section.split("[")[0]
                    visual_desc = visual_section.strip()
            
            # Create anchor with extracted data
            anchor = PersonaAnchor(
                name=persona_name.lower(),
                display_name=persona_name.title(),
                base_prompt_template=visual_desc
            )
            
            # Parse the visual description into identity
            self._parse_visual_description_to_identity(anchor, visual_desc)
            
            self.anchors[persona_name.lower()] = anchor
            self._save_state()
            
            # Auto-generate anchor image if requested
            if auto_generate_anchor and not anchor.face_reference:
                self._auto_generate_anchor(anchor)
            
            return anchor
            
        except Exception as e:
            print(f"[PersonaImageSystem] Failed to load persona YAML: {e}")
            return None
    
    def _auto_generate_anchor(self, anchor: PersonaAnchor) -> bool:
        """
        Automatically generate an anchor reference image for a persona.
        Uses ComfyUI to create a consistent reference.
        """
        try:
            print(f"[PersonaImageSystem] Auto-generating anchor for {anchor.display_name}...")
            
            # Build a reference prompt (neutral pose, clear face)
            identity_tags = anchor.identity.to_prompt_tags()
            
            ref_prompt = f"""1girl, solo, {anchor.display_name}, portrait, face focus, 
{identity_tags}, 
looking at viewer, neutral expression, soft smile, 
studio lighting, clean background, white background,
anime style, masterpiece, best quality, highly detailed face, 
sharp focus, high resolution"""
            
            ref_prompt = ref_prompt.replace("\n", " ").replace("  ", " ")
            
            ref_negative = """bad anatomy, bad face, ugly, deformed, blurry, 
low quality, multiple people, extra limbs, watermark, text,
complex background, busy background"""
            
            ref_negative = ref_negative.replace("\n", " ")
            
            # Generate using ComfyUI
            import asyncio
            from AitherOS.AitherNode.AitherCanvas import generate_local
            
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Generate the image
            paths = loop.run_until_complete(
                generate_local(ref_prompt, None, negative_prompt=ref_negative)
            )
            
            if paths and len(paths) > 0:
                # Copy to anchors directory
                import shutil
                src_path = paths[0]
                dest_path = os.path.join(ANCHORS_DIR, f"{anchor.name}_face_auto.png")
                shutil.copy2(src_path, dest_path)
                
                anchor.face_reference = dest_path
                anchor.updated_at = datetime.now().isoformat()
                
                # Analyze the generated image to refine identity
                self._analyze_anchor_image(anchor.name, "face")
                
                self._save_state()
                print(f"[PersonaImageSystem] [DONE] Auto-generated anchor: {dest_path}")
                return True
            else:
                print(f"[PersonaImageSystem] [FAIL] Failed to generate anchor image")
                return False
                
        except Exception as e:
            print(f"[PersonaImageSystem] Auto-anchor generation failed: {e}")
            return False
    
    def initialize_all_personas(self, auto_generate: bool = False) -> Dict[str, bool]:
        """
        Initialize all personas from YAML files.
        
        Args:
            auto_generate: Generate anchor images for personas without them
        
        Returns:
            Dict of persona_name -> success
        """
        personas_dir = os.path.join(NARRATIVE_AGENT_DIR, "config", "personas")
        results = {}
        
        if not os.path.exists(personas_dir):
            return results
        
        for filename in os.listdir(personas_dir):
            if filename.endswith(".yaml"):
                persona_name = filename[:-5]  # Remove .yaml
                
                # Skip system personas
                if persona_name in ["router", "security", "debugger", "architect"]:
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
        
        # Extract from common YAML format
        # Hair
        if "ponytail" in desc_lower:
            anchor.identity.hair_style = "high ponytail"
        if "long" in desc_lower and "hair" in desc_lower:
            anchor.identity.hair_length = "long"
        if "sleek" in desc_lower:
            anchor.identity.hair_details = "sleek"
        
        # Body
        if "fitness model" in desc_lower or "athletic" in desc_lower:
            anchor.identity.body_type = "fitness model, athletic"
        if "toned abs" in desc_lower or "abs" in desc_lower:
            anchor.identity.body_type += ", toned abs"
        if "small" in desc_lower and "breast" in desc_lower:
            anchor.identity.bust_size = "small perky breasts"
        if "wide hips" in desc_lower or "child-bearing hips" in desc_lower:
            anchor.identity.hip_width = "wide hips, thick thighs"
        if "curvy" in desc_lower:
            anchor.identity.hip_width += ", curvy figure"
        
        # Skin/marks
        if "tanlines" in desc_lower or "tan lines" in desc_lower:
            anchor.identity.distinguishing_marks = "bikini tanlines"
        if "sun-kissed" in desc_lower:
            anchor.identity.skin_tone = "sun-kissed"
        
        # Face
        if "glasses" in desc_lower:
            anchor.identity.facial_features = "stylish thin-framed glasses"
        if "confident" in desc_lower:
            anchor.identity.facial_features += ", confident expression"
    
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
        
        Args:
            persona_name: Which persona to generate
            user_request: The user's natural language request
            include_scene: Include current scene context
            override_pose: Specific pose to use (ignores extraction from request)
            override_clothes: Specific clothing (ignores current state)
            override_expression: Specific expression
            use_controlnet: Whether to include ControlNet references
        
        Returns:
            Dict with:
            - prompt: The final positive prompt
            - negative_prompt: Negative prompt
            - controlnet_image: Path to face/pose reference (if available)
            - controlnet_model: Which ControlNet to use
            - model_preference: Suggested diffusion model
            - seed: Deterministic seed based on identity
        """
        name_lower = persona_name.lower()
        
        # Get or create anchor
        anchor = self.anchors.get(name_lower)
        if not anchor:
            anchor = self.create_persona_from_yaml(name_lower)
        if not anchor:
            # Fallback: create minimal anchor
            anchor = PersonaAnchor(name=name_lower)
            self.anchors[name_lower] = anchor
        
        # Analyze user request FIRST to understand what they want
        modifications = self._analyze_request(user_request)
        
        # Start building prompt parts
        parts = []
        
        # 1. Subject identifier (ALWAYS first, enforces single character)
        parts.append("1girl, solo")
        parts.append(anchor.display_name)
        
        # 2. Core identity (face/hair/body - always for consistency)
        identity_tags = anchor.identity.to_prompt_tags()
        if identity_tags:
            parts.append(f"({identity_tags}:1.15)")
        
        # 3. SEXUAL ACTION (if detected - this is the main thing user wants!)
        action = modifications.get("action", "")
        if action:
            parts.append(action)
        
        # 4. Pose (from request or override)
        pose = override_pose or modifications.get("pose", "")
        if pose:
            parts.append(pose)
        
        # 5. Clothing/nudity state
        clothes = override_clothes or modifications.get("clothing", "")
        if clothes:
            parts.append(clothes)
        
        # 6. Fluids
        fluids = modifications.get("fluids", "")
        if fluids:
            parts.append(fluids)
        
        # 7. Expression
        expression = override_expression or modifications.get("expression", "")
        if expression:
            parts.append(expression)
        
        # 8. Camera/framing
        camera = modifications.get("camera", "")
        if camera:
            parts.append(camera)
        
        # 9. Scene context
        if include_scene:
            if self.scene.location:
                scene_tags = self.scene.to_prompt_tags()
                if scene_tags:
                    parts.append(scene_tags)
            else:
                # Default scene
                parts.append("high-tech office, neon ambient lighting, cyan and magenta")
        
        # 10. Style and quality (always last)
        parts.append(anchor.identity.art_style or "anime style")
        parts.append("masterpiece, best quality, highly detailed, detailed face, detailed body")
        
        # 11. Creative mode flag
        if modifications.get("is_nsfw"):
            parts.append("creative, detailed")
        
        # Build STRONG negative prompt
        negative_parts = [
            # Bad anatomy
            "bad anatomy, bad hands, missing fingers, extra digits, extra limbs, missing limbs",
            "fused fingers, too many fingers, poorly drawn hands, poorly drawn face",
            "ugly, deformed, disfigured, mutated",
            # Prevent multiple characters (critical!)
            "2girls, multiple girls, 3girls, multiple people, duo, group",
            # Prevent unwanted features
            "elf ears, pointed ears, animal ears, furry, tail",
            # Quality
            "blurry, low quality, text, watermark, signature",
            "multiple views, split screen, collage, comic, panel",
            "cropped, out of frame"
        ]
        
        # Add persona-specific exclusions
        for exclusion in anchor.exclusions:
            negative_parts.append(exclusion)
        
        # Pose-specific negatives
        if "from behind" in (pose or "").lower() or "back view" in (pose or "").lower():
            negative_parts.append("front view, facing viewer, looking at viewer")
        if "front" in (pose or "").lower() or "looking at viewer" in (pose or "").lower():
            negative_parts.append("from behind, back view, back turned")
        
        # If creative mode, make sure clothed is in negative when appropriate
        if modifications.get("is_nsfw") and "nude" in modifications.get("clothing", ""):
            negative_parts.append("clothed, dressed, wearing clothes")
        
        # Prepare ControlNet
        controlnet_image = None
        controlnet_model = None
        
        if use_controlnet:
            if anchor.face_reference and os.path.exists(anchor.face_reference):
                controlnet_image = anchor.face_reference
                controlnet_model = "ip_adapter_face"
            elif anchor.body_reference and os.path.exists(anchor.body_reference):
                controlnet_image = anchor.body_reference
                controlnet_model = "ip_adapter"
        
        # Generate deterministic seed from identity
        seed = self._generate_identity_seed(anchor)
        
        # Clean up and join parts (remove empty strings)
        prompt_parts = [p for p in parts if p and p.strip()]
        negative_prompt_parts = [p for p in negative_parts if p and p.strip()]
        
        return {
            "prompt": ", ".join(prompt_parts),
            "negative_prompt": ", ".join(negative_prompt_parts),
            "controlnet_image": controlnet_image,
            "controlnet_model": controlnet_model,
            "model_preference": "pony" if modifications.get("is_nsfw") else anchor.preferred_model,
            "seed": seed,
            "persona_name": persona_name,
            "modifications": modifications
        }
    
    def _analyze_request(self, request: str) -> Dict[str, Any]:
        """
        Analyze user request to extract pose, clothing, expression, action, etc.
        This is the core understanding of what the user wants.
        """
        request_lower = request.lower()
        result = {
            "pose": "",
            "action": "",
            "clothing": "",
            "expression": "",
            "camera": "",
            "fluids": "",
            "is_nsfw": False,
            "extra_tags": []
        }
        
        # === ACTIONS (highest priority) ===
        # Action detection is handled by safety-gated runtime.
        # This map provides clean generic action tags only.
        action_map = {
            "lick": "(licking:1.3), (tongue out:1.2)",
            "suck": "(sucking:1.3), (oral:1.2), (mouth:1.1)",
            "kiss": "(kissing:1.3), (lips:1.2)",
            "hug": "(hugging:1.3), (embrace:1.2)",
            "touch": "(touching:1.2), (hand contact:1.1)",
        }
        
        for keyword, tags in action_map.items():
            if keyword in request_lower:
                result["action"] = tags
                result["is_nsfw"] = True
                break
        
        # === POSES ===
        pose_map = {
            # Camera angles
            "from behind": "(from behind:1.3), (back view:1.2), (looking back:1.1)",
            "rear view": "(from behind:1.3), (back view:1.2), (looking back:1.1)",
            
            # General poses
            "selfie": "(selfie:1.3), (holding phone:1.2), (POV:1.2), (looking at viewer:1.2), (slight smile:1.1)",
            "leaning": "(leaning forward:1.2), (casual pose:1.1)",
            "kneeling": "(kneeling:1.3), (on knees:1.2)",
            "lying": "(lying down:1.2), (on back:1.1)",
            "reclining": "(reclining:1.2), (relaxed pose:1.1)",
            "sitting": "(sitting:1.2), (seated:1.1)",
            "standing": "(standing:1.2), (full body:1.1)",
            "riding": "(dynamic pose:1.3), (action:1.2)",
            "action": "(dynamic pose:1.3), (action:1.2), (motion:1.1)",
        }
        
        for keyword, tags in pose_map.items():
            if keyword in request_lower and not result["pose"]:
                result["pose"] = tags
                break
        
        # === CLOTHING/STATE ===
        clothing_map = {
            "nude": "(nude:1.3), (artistic nude:1.2)",
            "naked": "(nude:1.3), (artistic nude:1.2)",
            "topless": "(topless:1.3), (bare chest:1.2)",
            "lingerie": "(lingerie:1.3), (lace:1.2), (elegant underwear:1.1)",
            "bikini": "(bikini:1.3), (swimwear:1.2)",
            "underwear": "(underwear:1.2), (undergarments:1.1)",
            "undressing": "(undressing:1.3), (clothes coming off:1.2)",
            "clothed": "(clothed:1.1), (dressed:1.0)",
        }
        
        for keyword, tags in clothing_map.items():
            if keyword in request_lower:
                result["clothing"] = tags
                result["is_nsfw"] = True if keyword in ["nude", "naked", "topless", "bottomless"] else result["is_nsfw"]
                break
        
        # If creative mode request but no clothing specified, use default
        if result["is_nsfw"] and not result["clothing"]:
            result["clothing"] = "(nude:1.2), (artistic:1.1)"
        
        # === APPEARANCE STATE ===
        fluid_map = {
            "messy": "(disheveled:1.2), (messy appearance:1.1)",
            "sweat": "(sweaty:1.2), (glistening skin:1.1)",
            "wet": "(wet:1.2), (wet skin:1.1)",
            "dripping": "(wet:1.2), (dripping water:1.1)",
        }
        
        for keyword, tags in fluid_map.items():
            if keyword in request_lower:
                result["fluids"] = tags
                result["is_nsfw"] = True
                break
        
        # === EXPRESSIONS ===
        expression_map = {
            "crying": "(crying:1.3), (tears:1.2), (teary eyes:1.1)",
            "scared": "(scared:1.3), (frightened:1.2), (fearful expression:1.1)",
            "pleasure": "(pleasure:1.3), (ecstasy:1.2), (bliss:1.1)",
            "smile": "(smiling:1.2), (happy:1.1)",
            "smirk": "(smirking:1.2), (confident:1.1)",
            "blush": "(blushing:1.2), (embarrassed:1.1), (red cheeks:1.0)",
            "angry": "(angry:1.2), (frowning:1.1)",
            "seductive": "(seductive:1.2), (alluring gaze:1.1), (sultry:1.1)",
            "surprised": "(surprised:1.2), (wide eyes:1.1)",
            "sad": "(sad:1.2), (melancholy:1.1), (downcast eyes:1.0)",
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
            "from below": "(from below:1.3), (low angle:1.2), (looking down at viewer:1.1)",
        }
        
        for keyword, tags in camera_map.items():
            if keyword in request_lower:
                result["camera"] = tags
                break
        
        # === EXTRA NSFW DETECTION ===
        nsfw_keywords = ["nude", "naked", "sex", "explicit", "nsfw", "adult",
                        "mature content", "hentai", "ecchi"]
        if any(kw in request_lower for kw in nsfw_keywords):
            result["is_nsfw"] = True
        
        return result
    
    def _generate_identity_seed(self, anchor: PersonaAnchor) -> int:
        """
        Generate a deterministic seed based on persona identity.
        This helps maintain consistency across generations.
        """
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
        
        Args:
            base_image_path: Path to the image to modify
            modification_request: What to change (e.g., "change pose to sitting")
            persona_name: Optional persona to associate with
        
        Returns:
            Dict with enhanced prompt data
        """
        try:
            from AitherOS.AitherNode.vision_tools import analyze_with_ollama
            
            # First, extract detailed description of current image
            extraction_prompt = """Analyze this image and output ONLY comma-separated tags.

EXTRACT EXACTLY:
1. Character appearance: hair color, hair style, eye color, skin tone, body type
2. Clothing/state: what they're wearing or if nude
3. Pose: body position, where they're looking
4. Expression: facial expression, emotion
5. Setting: location, background, lighting
6. Camera: angle, framing, composition

OUTPUT ONLY TAGS. NO SENTENCES. NO EXPLANATIONS."""

            current_description = analyze_with_ollama(base_image_path, extraction_prompt, auto_unload=True)
            
            if not current_description:
                print("[PersonaImageSystem] Vision analysis failed")
                return self.build_prompt(persona_name or "aither", modification_request)
            
            print(f"[PersonaImageSystem] Extracted: {current_description[:100]}...")
            
            # Parse what to keep vs what to change
            keep_tags, change_tags = self._parse_modification_intent(
                current_description, modification_request
            )
            
            # Build new prompt
            parts = []
            
            # Keep character identity tags
            parts.append(keep_tags)
            
            # Apply modifications
            modifications = self._analyze_request(modification_request)
            
            if modifications.get("pose"):
                parts.append(f"({modifications['pose']}:1.2)")
            if modifications.get("clothing"):
                parts.append(modifications["clothing"])
            if modifications.get("expression"):
                parts.append(f"({modifications['expression']}:1.1)")
            if modifications.get("camera"):
                parts.append(modifications["camera"])
            
            # Quality tags
            parts.append("masterpiece, best quality, highly detailed, anime style")
            
            if modifications.get("is_nsfw"):
                parts.append("creative")
            
            return {
                "prompt": ", ".join(parts),
                "negative_prompt": "bad anatomy, bad hands, blurry, low quality, multiple views",
                "controlnet_image": base_image_path,  # Use original as reference
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
        """
        Determine which tags to keep and which to change based on the modification request.
        """
        request_lower = modification_request.lower()
        desc_tags = [t.strip() for t in current_description.split(",")]
        
        # Categories of tags
        pose_keywords = ["standing", "sitting", "lying", "kneeling", "leaning", "from behind", "riding", "reclining"]
        clothing_keywords = ["nude", "naked", "dress", "shirt", "pants", "bikini", "lingerie", "clothed", "topless"]
        expression_keywords = ["smile", "smirk", "angry", "sad", "blush", "neutral", "surprised"]
        
        keep_tags = []
        remove_categories = set()
        
        # Determine what's being changed
        if any(kw in request_lower for kw in ["pose", "position"] + pose_keywords):
            remove_categories.add("pose")
        if any(kw in request_lower for kw in ["clothes", "clothing", "wear", "nude", "naked"] + clothing_keywords):
            remove_categories.add("clothing")
        if any(kw in request_lower for kw in ["expression", "face", "emotion"] + expression_keywords):
            remove_categories.add("expression")
        
        # Filter tags
        for tag in desc_tags:
            tag_lower = tag.lower()
            
            should_remove = False
            
            if "pose" in remove_categories:
                if any(kw in tag_lower for kw in pose_keywords):
                    should_remove = True
            
            if "clothing" in remove_categories:
                if any(kw in tag_lower for kw in clothing_keywords):
                    should_remove = True
            
            if "expression" in remove_categories:
                if any(kw in tag_lower for kw in expression_keywords):
                    should_remove = True
            
            if not should_remove:
                keep_tags.append(tag)
        
        return ", ".join(keep_tags), ", ".join(remove_categories)
    
    # ========== Scene Management ==========
    
    def set_scene(
        self,
        location: str = None,
        lighting: str = None,
        time_of_day: str = None,
        atmosphere: str = None,
        personas: List[str] = None
    ):
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
    
    def build_group_prompt(
        self,
        persona_names: List[str],
        user_request: str,
        shared_pose: str = None
    ) -> Dict[str, Any]:
        """
        Build a prompt for multiple characters in the same scene.
        """
        parts = []
        
        # Character count
        count = len(persona_names)
        count_tag = {1: "1girl", 2: "2girls", 3: "3girls"}.get(count, f"{count}girls")
        parts.append(count_tag)
        
        # Individual character descriptions
        for name in persona_names:
            anchor = self.get_anchor(name) or self.create_persona_from_yaml(name)
            if anchor:
                identity_tags = anchor.identity.to_prompt_tags()
                if identity_tags:
                    parts.append(f"({anchor.display_name}: {identity_tags})")
        
        # Shared scene
        if self.scene.location:
            parts.append(self.scene.to_prompt_tags())
        
        # Shared pose
        if shared_pose:
            parts.append(f"({shared_pose}:1.2)")
        
        # Request-based modifications
        modifications = self._analyze_request(user_request)
        if modifications.get("pose") and not shared_pose:
            parts.append(f"({modifications['pose']}:1.2)")
        
        # Quality
        parts.append("masterpiece, best quality, highly detailed, anime style")
        
        if modifications.get("is_nsfw"):
            parts.append("uncensored")
        
        return {
            "prompt": ", ".join(parts),
            "negative_prompt": "bad anatomy, bad hands, blurry, low quality, solo, 1girl, multiple views",
            "personas": persona_names,
            "scene": asdict(self.scene)
        }


# Singleton instance
_system: Optional[PersonaImageSystem] = None


def get_persona_image_system() -> PersonaImageSystem:
    """Get the singleton PersonaImageSystem instance."""
    global _system
    if _system is None:
        _system = PersonaImageSystem()
    return _system


# Convenience functions

def set_persona_anchor(persona_name: str, image_path: str, anchor_type: str = "face") -> bool:
    """Quick function to set an anchor image."""
    return get_persona_image_system().set_anchor_image(persona_name, image_path, anchor_type)


def generate_persona_prompt(persona_name: str, request: str, **kwargs) -> Dict[str, Any]:
    """Quick function to generate a prompt for a persona."""
    return get_persona_image_system().build_prompt(persona_name, request, **kwargs)


def enhance_with_vision(image_path: str, request: str, persona: str = None) -> Dict[str, Any]:
    """Quick function to enhance prompt with vision analysis."""
    return get_persona_image_system().enhance_prompt_with_vision(image_path, request, persona)


def set_scene_context(location: str = None, lighting: str = None, **kwargs):
    """Quick function to set scene context."""
    get_persona_image_system().set_scene(location=location, lighting=lighting, **kwargs)

