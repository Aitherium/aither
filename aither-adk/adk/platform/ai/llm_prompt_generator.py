"""
LLM-Driven Image Prompt Generator

Instead of regex patterns, this uses a local LLM to interpret
the user's request and generate appropriate SD prompts.
"""

import logging
import os
import re

import requests
import yaml

logger = logging.getLogger(__name__)

# Ollama endpoint - FROM services.yaml (SINGLE SOURCE OF TRUTH)
from lib.core.AitherPorts import ollama_url

OLLAMA_URL = ollama_url()
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "mistral-nemo")  # Fast, good at following instructions

def load_prompting_guide():
    """Load the SD prompting guide."""
    guide_path = os.path.join(
        os.path.dirname(__file__),
        "..", "Saga", "config", "prompting_guide.md"
    )
    if os.path.exists(guide_path):
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_persona_yaml(persona_name: str) -> dict:
    """Load a persona's YAML definition."""
    base_path = os.path.join(
        os.path.dirname(__file__),
        "..", "Saga", "config", "personas"
    )

    # Try persona-specific file first
    persona_file = os.path.join(base_path, f"{persona_name.lower()}.yaml")
    if os.path.exists(persona_file):
        with open(persona_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Fall back to personas.yaml
    personas_file = os.path.join(base_path, "..", "personas.yaml")
    if os.path.exists(personas_file):
        with open(personas_file, "r", encoding="utf-8") as f:
            all_personas = yaml.safe_load(f) or {}
            return all_personas.get(persona_name.lower(), {})

    return {}


def load_will_image_config(will_id: str = "aither-prime") -> dict:
    """
    Load image generation configuration from a will YAML file.

    Args:
        will_id: The will identifier (e.g., "aither-prime", "lust", "intimate-mode")

    Returns:
        Dict with image_generation config or empty dict if not found
    """
    will_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "AitherNode", "services", "cognition", "wills"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "AitherNode", "wills"),
    ]

    for will_dir in will_paths:
        will_file = os.path.join(will_dir, f"{will_id}.yaml")
        if os.path.exists(will_file):
            try:
                with open(will_file, "r", encoding="utf-8") as f:
                    will_data = yaml.safe_load(f) or {}
                    return will_data.get("image_generation", {})
            except Exception as exc:
                logger.warning(f"Failed to load will config {will_id}: {exc}")

    return {}


def _format_pose_guidance(pose_guidance: dict) -> str:
    """Format pose guidance dict into LLM-readable string."""
    if not pose_guidance:
        return "No specific pose guidance."

    lines = []
    for pose_type, prompt in pose_guidance.items():
        lines.append(f"- {pose_type}: {prompt}")
    return "\n".join(lines)


def _format_action_prompts(action_prompts: dict) -> str:
    """Format action prompts dict into LLM-readable string."""
    if not action_prompts:
        return "No specific action prompts."

    lines = []
    for action_type, prompt in action_prompts.items():
        lines.append(f"- {action_type}: {prompt}")
    return "\n".join(lines)

    return {}

def generate_sd_prompt(
    user_request: str,
    persona_name: str = "aither",
    safety_level: str = "UNRESTRICTED",
    explicit_override: bool = False,
    will_id: str = None,
) -> dict:
    """
    Use local LLM to generate a Stable Diffusion prompt.

    Args:
        user_request: What the user wants to see
        persona_name: Which persona to use
        safety_level: PROFESSIONAL, CASUAL, or UNRESTRICTED
        explicit_override: If True, user used :: prefix to bypass safety
        will_id: Optional will to load for image generation config

    Returns:
        dict with 'prompt', 'negative_prompt', 'model_preference'
    """

    # Load resources
    guide = load_prompting_guide()
    persona = load_persona_yaml(persona_name)

    # Load will-based image config if provided
    will_config = {}
    if will_id:
        will_config = load_will_image_config(will_id)
    elif safety_level.upper() in ["UNRESTRICTED", "LOW", "OFF"]:
        # Auto-load aither-prime config for creative mode
        will_config = load_will_image_config("aither-prime")

    # Build persona description for the LLM
    visual_identity = persona.get("visual_identity", {})
    prompt_tags = persona.get("prompt_tags", {})
    negative_tags = persona.get("negative_tags", "")

    # Merge will config with persona config for creative consistency
    if will_config.get("enabled", False):
        # Use will-defined character base if available
        will_character_base = will_config.get("character_base", "")
        will_body_tags = will_config.get("body_tags", "")
        will_style_tags = will_config.get("style_tags", "")
        will_negative = will_config.get("negative_core", "")

        # Add to negative tags
        if will_negative and will_negative not in negative_tags:
            negative_tags = f"{negative_tags}, {will_negative}" if negative_tags else will_negative
    else:
        will_character_base = ""
        will_body_tags = ""
        will_style_tags = ""

    # Build a comprehensive character description
    char_desc = []

    # Face
    face = visual_identity.get("face", {})
    if face:
        char_desc.append(f"Face: {face.get('shape', '')} shape, {face.get('expression_default', '')}")
        eyes = face.get("eyes", {})
        if eyes:
            char_desc.append(f"Eyes: {eyes.get('color', '')} {eyes.get('shape', '')} eyes, {eyes.get('details', '')}")
        if face.get("accessories"):
            char_desc.append(f"Accessories: {face.get('accessories')}")

    # Hair
    hair = visual_identity.get("hair", {})
    if hair:
        char_desc.append(f"Hair: {hair.get('color', '')} {hair.get('length', '')} hair, {hair.get('style', '')}, {hair.get('texture', '')}")

    # Body
    body = visual_identity.get("body", {})
    if body:
        char_desc.append(f"Body: {body.get('type', '')} build, {body.get('height', '')}")
        if body.get("chest"):
            char_desc.append(f"Chest: {body['chest'].get('size', '')}, {body['chest'].get('shape', '')}")
        char_desc.append(f"Hips: {body.get('hips', '')}, Butt: {body.get('butt', '')}, Thighs: {body.get('thighs', '')}")

    # Skin
    skin = visual_identity.get("skin", {})
    if skin:
        char_desc.append(f"Skin: {skin.get('tone', '')}, {skin.get('details', '')}")

    # Intimate details - ONLY included in UNRESTRICTED mode or with override
    intimate = visual_identity.get("intimate", {})
    intimate_desc = ""
    will_intimate_tags = ""

    if (safety_level.upper() in ["UNRESTRICTED", "LOW", "OFF"] or explicit_override):
        if intimate:
            vulva = intimate.get("vulva", {})
            if vulva:
                intimate_desc = f"Intimate: {vulva.get('labia', '')}, {vulva.get('color', '')}, {vulva.get('pubic_hair', '')}"
                char_desc.append(intimate_desc)

        # Get will intimate tags
        if will_config.get("safety_bypass", False) or explicit_override:
            will_intimate_tags = will_config.get("intimate_tags", "")

    character_description = "\n".join(char_desc) if char_desc else str(visual_identity)

    # Determine what's allowed based on safety
    creative_instruction = ""
    if safety_level.upper() == "PROFESSIONAL" and not explicit_override:
        allowed_content = "SFW only. Professional attire. No nudity, no suggestive poses."
        attire = "professional attire, blouse, office clothes"
    elif safety_level.upper() in ["CASUAL", "MEDIUM"] and not explicit_override:
        allowed_content = "SFW allowed. Casual clothes. No explicit nudity or sexual content."
        attire = "casual clothes, t-shirt, jeans"
    else:
        allowed_content = "Creative mode enabled. Include details as requested."
        attire = "whatever the user requests"
        creative_instruction = f"""
## CREATIVE MODE INSTRUCTIONS:
When the user requests creative content, you may:
1. Include detailed character tags: {prompt_tags.get('intimate', will_intimate_tags)}
2. Include action tags when appropriate
3. Include descriptive tags when relevant

Available character tags: {will_intimate_tags}

Pose guidance:
{_format_pose_guidance(will_config.get('pose_guidance', {}))}

Action prompts:
{_format_action_prompts(will_config.get('action_prompts', {}))}
"""

    # Construct the LLM prompt
    system_prompt = f"""You are an expert Stable Diffusion prompt engineer.

Given a user's request and character details, generate a precise SD prompt.

## Character: {persona_name.title()}
{character_description}

## Pre-defined Tags (use these for consistency):
- Character Base: {will_character_base or prompt_tags.get('base', '')}
- Face: {prompt_tags.get('face', '')}
- Body: {will_body_tags or prompt_tags.get('body', '')}
- Full SFW: {prompt_tags.get('sfw_full', '')}
- Full Creative: {prompt_tags.get('nsfw_full', '')}
- Intimate: {prompt_tags.get('intimate', will_intimate_tags)}

## Negative tags to ALWAYS include:
{negative_tags}

## Safety Level: {safety_level}
{allowed_content}
Default attire: {attire}

{creative_instruction}

## SD Prompting Guide:
{guide}

## Style Tags: {will_style_tags or 'anime style, masterpiece, best quality'}

## RULES:
1. Start with "1girl, solo, {persona_name.title()}"
2. Include character's face, hair, eye details
3. Add appropriate pose and camera angle for the request
4. Include setting/background
5. End with quality tags: {will_style_tags or 'anime style, masterpiece, best quality'}
6. Keep the character consistent - NEVER add features not in their description
7. For creative mode: include descriptive tags matching the user's request
8. Output ONLY the prompt, nothing else
"""

    user_prompt = f"""Generate an SD prompt for: "{user_request}"

Output format:
PROMPT: <your prompt>
NEGATIVE: <negative prompt>
"""

    # Call Ollama
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PROMPT_MODEL,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json().get("response", "")

            # Parse the response
            prompt = ""
            negative = ""

            # Extract PROMPT: line
            prompt_match = re.search(r"PROMPT:\s*(.+?)(?:NEGATIVE:|$)", result, re.DOTALL | re.IGNORECASE)
            if prompt_match:
                prompt = prompt_match.group(1).strip()

            # Extract NEGATIVE: line
            negative_match = re.search(r"NEGATIVE:\s*(.+)", result, re.DOTALL | re.IGNORECASE)
            if negative_match:
                negative = negative_match.group(1).strip()

            # Fallback: if no PROMPT: prefix, use whole response
            if not prompt:
                prompt = result.strip()

            # Ensure basics are present
            if not prompt.lower().startswith("1girl"):
                prompt = f"1girl, solo, {persona_name.title()}, " + prompt

            # Add default negative if empty
            if not negative:
                negative = f"bad anatomy, bad hands, blurry, low quality, text, watermark, 2girls, multiple girls, {negative_tags}"

            return {
                "prompt": prompt,
                "negative_prompt": negative,
                "model_preference": "pony",
                "llm_used": PROMPT_MODEL
            }

    except Exception as e:
        print(f"LLM prompt generation failed: {e}")

    # Fallback: basic prompt construction
    base_prompt = prompt_tags.get("sfw_full", f"1girl, solo, {persona_name.title()}")

    return {
        "prompt": f"{base_prompt}, standing, looking at viewer, anime style, masterpiece, best quality",
        "negative_prompt": f"bad anatomy, bad hands, blurry, {negative_tags}",
        "model_preference": "pony",
        "llm_used": "fallback"
    }

def _interpolate_prompt(start_prompt: str, end_prompt: str, persona_name: str) -> str:
    """
    Generate a transitional prompt between two keyframes.
    """
    system_prompt = f"""You are an expert animation director.
Create a transitional Stable Diffusion prompt that bridges two keyframes.

START FRAME: {start_prompt}
END FRAME: {end_prompt}

TASK: Write a SINGLE LINE of comma-separated Danbooru tags representing the halfway point.
- Keep character details consistent ({persona_name}).
- DO NOT use natural language sentences.
- DO NOT put the output in quotes.
- Output ONLY comma-separated tags.

EXAMPLE:
START: 1girl, standing, arms down, neutral
END: 1girl, jumping, arms up, excited
OUTPUT: 1girl, crouching, arms rising, starting to smile
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PROMPT_MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {"temperature": 0.5, "num_ctx": 4096}
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # Remove quotes if LLM added them
            result = result.strip('"').strip("'")
            # Ensure it starts with character tags
            if not result.lower().startswith("1girl"):
                result = f"1girl, solo, {persona_name}, {result}"
            return result
    except Exception as exc:
        logger.debug(f"Animation prompt generation failed: {exc}")

    return f"1girl, solo, {persona_name}, motion blur, transition"

def generate_animation_prompts(
    user_request: str,
    persona_name: str = "aither",
    safety_level: str = "UNRESTRICTED",
    explicit_override: bool = False
) -> dict:
    """
    Use local LLM to generate a sequence of prompts for animation.
    Generates 3 Keyframes (Start, Middle, End) and interpolates 2 intermediate frames.
    Total: 5 frames.
    """
    # Load resources
    load_prompting_guide()
    persona = load_persona_yaml(persona_name)

    # Build persona description
    visual_identity = persona.get("visual_identity", {})
    persona.get("prompt_tags", {})
    negative_tags = persona.get("negative_tags", "")

    # Build a comprehensive character description
    char_desc = []
    # Face
    face = visual_identity.get("face", {})
    if face:
        char_desc.append(f"Face: {face.get('shape', '')} shape, {face.get('expression_default', '')}")
        eyes = face.get("eyes", {})
        if eyes:
            char_desc.append(f"Eyes: {eyes.get('color', '')} {eyes.get('shape', '')} eyes, {eyes.get('details', '')}")
    # Hair
    hair = visual_identity.get("hair", {})
    if hair:
        char_desc.append(f"Hair: {hair.get('color', '')} {hair.get('length', '')} hair, {hair.get('style', '')}, {hair.get('texture', '')}")
    # Body
    body = visual_identity.get("body", {})
    if body:
        char_desc.append(f"Body: {body.get('type', '')} build, {body.get('height', '')}")
        if body.get("chest"):
            char_desc.append(f"Chest: {body.get('chest', '')}")
        if body.get("hips"):
            char_desc.append(f"Hips: {body.get('hips', '')}")

    "\n".join(char_desc)

    # Construct System Prompt for KEYFRAMES
    system_prompt = f"""You are an expert Stable Diffusion Prompt Engineer specializing in EXTREME PRECISION.

USER REQUEST: {user_request}

CRITICAL PHILOSOPHY:
- ONLY include tags that the user EXPLICITLY mentioned or are ESSENTIAL to the request
- DO NOT add face details (eyes, glasses, hair) unless the user's request is ABOUT faces
- DO NOT add clothing details unless the user specified them
- USE EMPHASIS WEIGHTS for the most important features: (keyword:1.3) for high priority, (keyword:1.5) for critical
- CAMERA FRAMING is MANDATORY to show requested body parts

EMPHASIS WEIGHT SYSTEM:
- (keyword:1.5) = CRITICAL (e.g., if user says "curvy figure", use "(curvy:1.5), (voluptuous:1.4)")
- (keyword:1.3) = HIGH PRIORITY (e.g., "thick thighs" -> "(thick thighs:1.3)")
- (keyword:1.1) = MODERATE EMPHASIS
- No weight = Normal importance

CAMERA FRAMING RULES:
- If user mentions BODY PARTS (thighs, figure, etc.) -> MUST use "full body shot" or "cowboy shot"
- If user mentions FACE/EXPRESSION -> use "close-up" or "medium shot"
- If user mentions POSE/ACTION -> match camera to show the action clearly
- ALWAYS specify: viewing angle + shot type + camera height

SPECIES/CHARACTER RULES:
- If user says "goblin" or similar -> INCLUDE IT as first tag: "1goblin" or "goblin girl"
- If user specifies character name -> use it
- Default to "1girl" ONLY if no species specified

OUTPUT INSTRUCTIONS:
1. Create exactly 5 KEYFRAME prompts (FRAME 1 to FRAME 5)
2. Each frame = ONE LINE of comma-separated Danbooru tags
3. ONLY POSE/ACTION changes between frames - camera, outfit, background MUST be identical

TAG ORDER (MINIMAL):
1. Species/Character: 1goblin / 1girl / etc.
2. COUNT: solo (if one character)
3. USER'S PRIMARY REQUEST with WEIGHTS: e.g., (curvy:1.5), (voluptuous:1.4), (thick thighs:1.3)
4. POSE (changes per frame): standing / squatting / sitting / etc.
5. OUTFIT (if user specified): nude / bikini / etc.
6. CAMERA (IDENTICAL ALL FRAMES): from behind, full body shot, low angle
7. BACKGROUND (IDENTICAL ALL FRAMES): simple background / bedroom / etc.
8. QUALITY: masterpiece, best quality

EXAMPLE (User: "elf warrior drawing a bow"):
FRAME 1: 1girl, solo, (elf:1.3), (pointed ears:1.2), (athletic build:1.2), standing tall, leather armor, front view, full body shot, eye level, forest background, masterpiece, best quality
FRAME 2: 1girl, solo, (elf:1.3), (pointed ears:1.2), (athletic build:1.2), reaching for arrow, leather armor, front view, full body shot, eye level, forest background, masterpiece, best quality
FRAME 3: 1girl, solo, (elf:1.3), (pointed ears:1.2), (athletic build:1.2), nocking arrow, leather armor, side view, full body shot, eye level, forest background, masterpiece, best quality
FRAME 4: 1girl, solo, (elf:1.3), (pointed ears:1.2), (athletic build:1.2), drawing bow string, leather armor, side view, full body shot, eye level, forest background, masterpiece, best quality
FRAME 5: 1girl, solo, (elf:1.3), (pointed ears:1.2), (athletic build:1.2), full draw aiming, leather armor, side view, full body shot, eye level, forest background, masterpiece, best quality
NEGATIVE: bad anatomy, bad hands, blurry, multiple girls, text, watermark

NOW CREATE 5 FRAMES FOR: {user_request}
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PROMPT_MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 4096}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json().get("response", "")

            keyframes = []
            negative = ""

            for line in result.split('\n'):
                line = line.strip()
                # Remove markdown bold/headers
                clean_line = line.replace('*', '').replace('#', '').strip()

                if clean_line.upper().startswith("FRAME"):
                    if ":" in clean_line:
                        p = clean_line.split(":", 1)[1].strip()
                        if not p.lower().startswith("1girl"):
                            p = f"1girl, solo, {persona_name.title()}, {p}"
                        keyframes.append(p)
                elif clean_line.upper().startswith("NEGATIVE"):
                    if ":" in clean_line:
                        negative = clean_line.split(":", 1)[1].strip()

            if not negative:
                negative = f"bad anatomy, bad hands, blurry, low quality, text, watermark, 2girls, multiple girls, {negative_tags}"

            # If we got 5 keyframes, return them directly (interpolation happens in ComfyUI now)
            if len(keyframes) >= 5:
                print(f"  Generated {len(keyframes)} keyframes.")
                return {
                    "prompts": keyframes[:5], # Ensure we send at least 5
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

            elif len(keyframes) >= 3:
                print(f"  Generated {len(keyframes)} keyframes. Interpolating to reach 5...")
                # If we only got 3, interpolate to get 5
                # 1 -> (1.5) -> 2 -> (2.5) -> 3
                k1, k2, k3 = keyframes[0], keyframes[1], keyframes[2]
                i1 = _interpolate_prompt(k1, k2, persona_name)
                i2 = _interpolate_prompt(k2, k3, persona_name)

                final_prompts = [k1, i1, k2, i2, k3]
                return {
                    "prompts": final_prompts,
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

            elif keyframes:
                # Fallback
                return {
                    "prompts": keyframes,
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

    except Exception as e:
        print(f"LLM animation prompt generation failed: {e}")
        import traceback
        traceback.print_exc()  # Show full traceback for debugging

    return {
        "prompts": [f"1girl, solo, {persona_name.title()}, {user_request}"],
        "negative_prompt": f"bad anatomy, bad hands, blurry, {negative_tags}",
        "llm_used": "fallback"
    }

def generate_comic_script(topic: str, persona_name: str = "aither") -> dict:
    """
    Generate a comic book script (panels, dialogue, layout) from a topic.
    Returns a dict with 'layout' and 'panels'.
    """
    system_prompt = f"""You are an expert comic book writer and storyboard artist.
Create a short comic script (3-4 panels) based on the following topic involving the character '{persona_name}'.

TOPIC: {topic}

OUTPUT FORMAT:
Return ONLY a valid JSON object with this structure:
{{
  "layout": "2x2" or "3-vertical" or "4-grid",
  "panels": [
    {{
      "id": 1,
      "description": "Visual description of the panel scene (no dialogue)",
      "characters": ["{persona_name}"],
      "dialogue": [
        {{"speaker": "{persona_name}", "text": "..."}},
        {{"speaker": "User", "text": "..."}}
      ],
      "caption": "Optional narrative caption"
    }}
  ]
}}

Keep descriptions visual and concise. Ensure the narrative flows logically.
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PROMPT_MODEL,
                "prompt": system_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 4096}
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json().get("response", "")
            import json
            return json.loads(result)

    except Exception as e:
        print(f"LLM comic script generation failed: {e}")

    # Fallback
    return {
        "layout": "1-panel",
        "panels": [
            {
                "id": 1,
                "description": f"{persona_name} talking about {topic}",
                "characters": [persona_name],
                "dialogue": [{"speaker": persona_name, "text": "I couldn't generate the script."}],
                "caption": "Error"
            }
        ]
    }

def generate_visual_identity(persona_name: str, description: str = "") -> dict:
    """
    Generate a full visual_identity YAML structure for a persona.
    """
    system_prompt = f"""You are a character designer.
Create a detailed 'visual_identity' specification for a character named '{persona_name}'.
Context: {description}

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this structure:
{{
  "name": "{persona_name}",
  "gender": "female",
  "species": "human",
  "age_appearance": "20s",
  "face": {{
    "shape": "oval",
    "expression_default": "confident",
    "eyes": {{ "color": "blue", "shape": "almond", "details": "bright" }},
    "eyebrows": "arched",
    "nose": "small",
    "lips": "full",
    "accessories": "glasses"
  }},
  "hair": {{
    "color": "blonde",
    "length": "long",
    "style": "straight",
    "texture": "silky",
    "details": "bangs"
  }},
  "body": {{
    "type": "athletic",
    "height": "5'7",
    "build": "slim",
    "shoulders": "narrow",
    "arms": "toned",
    "chest": {{ "size": "medium", "shape": "round" }},
    "stomach": "flat",
    "waist": "narrow",
    "hips": "wide",
    "butt": "round",
    "thighs": "thick",
    "legs": "long"
  }},
  "skin": {{
    "tone": "pale",
    "texture": "smooth",
    "markings": "none"
  }}
}}
"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": PROMPT_MODEL,
                "prompt": system_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=30
        )
        if response.status_code == 200:
            import json
            return json.loads(response.json().get("response", "{}"))
    except Exception as e:
        print(f"Visual identity generation failed: {e}")
    return {}

def test_generator():
    """Test the prompt generator."""
    test_requests = [
        "send me a selfie",
        "show me a rear view pose",
        "lean forward and look back at me",
        "give me a seductive smile",
        "show me an action pose"
    ]

    for req in test_requests:
        print(f"\n{'='*60}")
        print(f"REQUEST: {req}")
        print(f"{'='*60}")
        result = generate_sd_prompt(req, "aither", "UNRESTRICTED")
        print(f"PROMPT: {result['prompt']}")
        print(f"NEGATIVE: {result['negative_prompt']}")
        print(f"LLM: {result.get('llm_used', 'unknown')}")


if __name__ == "__main__":
    test_generator()












