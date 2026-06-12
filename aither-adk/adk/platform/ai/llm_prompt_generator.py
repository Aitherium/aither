"""
LLM-Driven Image Prompt Generator

Uses a local LLM to interpret user requests and generate
Stable Diffusion prompts with character consistency.
"""

import logging
import os
import re

import requests
import yaml

logger = logging.getLogger(__name__)

# Ollama endpoint - FROM services.yaml (SINGLE SOURCE OF TRUTH)
try:
    from adk.ports import ollama_url
    OLLAMA_URL = ollama_url()
except ImportError:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

PROMPT_MODEL = os.getenv("PROMPT_MODEL", "mistral-nemo")


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

    persona_file = os.path.join(base_path, f"{persona_name.lower()}.yaml")
    if os.path.exists(persona_file):
        with open(persona_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    personas_file = os.path.join(base_path, "..", "personas.yaml")
    if os.path.exists(personas_file):
        with open(personas_file, "r", encoding="utf-8") as f:
            all_personas = yaml.safe_load(f) or {}
            return all_personas.get(persona_name.lower(), {})

    return {}


def generate_sd_prompt(
    user_request: str,
    persona_name: str = "aither",
    style: str = "anime",
) -> dict:
    """
    Use local LLM to generate a Stable Diffusion prompt.

    Args:
        user_request: What the user wants to see
        persona_name: Which character to use
        style: Art style (anime, realistic, semi-realistic)

    Returns:
        dict with 'prompt', 'negative_prompt', 'model_preference'
    """
    guide = load_prompting_guide()
    persona = load_persona_yaml(persona_name)

    visual_identity = persona.get("visual_identity", {})
    prompt_tags = persona.get("prompt_tags", {})
    negative_tags = persona.get("negative_tags", "")

    # Build character description from visual identity
    char_desc = []

    face = visual_identity.get("face", {})
    if face:
        char_desc.append(f"Face: {face.get('shape', '')} shape, {face.get('expression_default', '')}")
        eyes = face.get("eyes", {})
        if eyes:
            char_desc.append(f"Eyes: {eyes.get('color', '')} {eyes.get('shape', '')} eyes")
        if face.get("accessories"):
            char_desc.append(f"Accessories: {face.get('accessories')}")

    hair = visual_identity.get("hair", {})
    if hair:
        char_desc.append(f"Hair: {hair.get('color', '')} {hair.get('length', '')} hair, {hair.get('style', '')}")

    body = visual_identity.get("body", {})
    if body:
        char_desc.append(f"Body: {body.get('type', '')} build, {body.get('height', '')}")

    skin = visual_identity.get("skin", {})
    if skin:
        char_desc.append(f"Skin: {skin.get('tone', '')}")

    character_description = "\n".join(char_desc) if char_desc else str(visual_identity)

    style_tags = {
        "anime": "anime style, masterpiece, best quality",
        "realistic": "photorealistic, 8k, professional photo",
        "semi-realistic": "semi-realistic, detailed, soft lighting",
    }.get(style, "anime style, masterpiece, best quality")

    system_prompt = f"""You are an expert Stable Diffusion prompt engineer.

Given a user's request and character details, generate a precise SD prompt.

## Character: {persona_name.title()}
{character_description}

## Pre-defined Tags (use these for consistency):
- Character Base: {prompt_tags.get('base', '')}
- Face: {prompt_tags.get('face', '')}
- Body: {prompt_tags.get('body', '')}

## Negative tags to ALWAYS include:
{negative_tags}

## Style Tags: {style_tags}

## SD Prompting Guide:
{guide}

## RULES:
1. Start with "1girl, solo, {persona_name.title()}" (or appropriate subject)
2. Include character's face, hair, eye details
3. Add appropriate pose and camera angle for the request
4. Include setting/background
5. End with quality tags: {style_tags}
6. Keep the character consistent - NEVER add features not in their description
7. Output ONLY the prompt, nothing else
"""

    user_prompt = f"""Generate an SD prompt for: "{user_request}"

Output format:
PROMPT: <your prompt>
NEGATIVE: <negative prompt>
"""

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

            prompt = ""
            negative = ""

            prompt_match = re.search(r"PROMPT:\s*(.+?)(?:NEGATIVE:|$)", result, re.DOTALL | re.IGNORECASE)
            if prompt_match:
                prompt = prompt_match.group(1).strip()

            negative_match = re.search(r"NEGATIVE:\s*(.+)", result, re.DOTALL | re.IGNORECASE)
            if negative_match:
                negative = negative_match.group(1).strip()

            if not prompt:
                prompt = result.strip()

            if not negative:
                negative = f"bad anatomy, bad hands, blurry, low quality, text, watermark, {negative_tags}"

            return {
                "prompt": prompt,
                "negative_prompt": negative,
                "model_preference": "flux",
                "llm_used": PROMPT_MODEL
            }

    except Exception as e:
        logger.warning(f"LLM prompt generation failed: {e}")

    # Fallback
    base_prompt = prompt_tags.get("base", f"1girl, solo, {persona_name.title()}")

    return {
        "prompt": f"{base_prompt}, standing, looking at viewer, {style_tags}",
        "negative_prompt": f"bad anatomy, bad hands, blurry, {negative_tags}",
        "model_preference": "flux",
        "llm_used": "fallback"
    }


def _interpolate_prompt(start_prompt: str, end_prompt: str, persona_name: str) -> str:
    """Generate a transitional prompt between two keyframes."""
    system_prompt = f"""You are an expert animation director.
Create a transitional Stable Diffusion prompt that bridges two keyframes.

START FRAME: {start_prompt}
END FRAME: {end_prompt}

TASK: Write a SINGLE LINE of comma-separated tags representing the halfway point.
- Keep character details consistent ({persona_name}).
- DO NOT use natural language sentences.
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
            result = result.strip('"').strip("'")
            return result
    except Exception as exc:
        logger.debug(f"Animation prompt generation failed: {exc}")

    return f"1girl, solo, {persona_name}, motion blur, transition"


def generate_animation_prompts(
    user_request: str,
    persona_name: str = "aither",
    style: str = "anime",
) -> dict:
    """
    Use local LLM to generate a sequence of prompts for animation.
    Generates 5 keyframes with consistent character and scene.
    """
    persona = load_persona_yaml(persona_name)

    visual_identity = persona.get("visual_identity", {})
    negative_tags = persona.get("negative_tags", "")

    # Build character description
    char_desc = []
    face = visual_identity.get("face", {})
    if face:
        char_desc.append(f"Face: {face.get('shape', '')} shape, {face.get('expression_default', '')}")
        eyes = face.get("eyes", {})
        if eyes:
            char_desc.append(f"Eyes: {eyes.get('color', '')} {eyes.get('shape', '')} eyes")

    hair = visual_identity.get("hair", {})
    if hair:
        char_desc.append(f"Hair: {hair.get('color', '')} {hair.get('length', '')} hair, {hair.get('style', '')}")

    body = visual_identity.get("body", {})
    if body:
        char_desc.append(f"Body: {body.get('type', '')} build")

    system_prompt = f"""You are an expert Stable Diffusion Prompt Engineer.

USER REQUEST: {user_request}

CHARACTER: {persona_name.title()}
{chr(10).join(char_desc)}

OUTPUT INSTRUCTIONS:
1. Create exactly 5 KEYFRAME prompts (FRAME 1 to FRAME 5)
2. Each frame = ONE LINE of comma-separated tags
3. ONLY POSE/ACTION changes between frames - camera, outfit, background MUST be identical

TAG ORDER:
1. Subject: 1girl / 1boy / etc.
2. Character details
3. Pose (changes per frame)
4. Outfit
5. Camera angle (IDENTICAL ALL FRAMES)
6. Background (IDENTICAL ALL FRAMES)
7. Quality: masterpiece, best quality

EXAMPLE (User: "elf warrior drawing a bow"):
FRAME 1: 1girl, solo, elf, pointed ears, athletic build, standing tall, leather armor, front view, full body shot, forest background, masterpiece, best quality
FRAME 2: 1girl, solo, elf, pointed ears, athletic build, reaching for arrow, leather armor, front view, full body shot, forest background, masterpiece, best quality
FRAME 3: 1girl, solo, elf, pointed ears, athletic build, nocking arrow, leather armor, side view, full body shot, forest background, masterpiece, best quality
FRAME 4: 1girl, solo, elf, pointed ears, athletic build, drawing bow, leather armor, side view, full body shot, forest background, masterpiece, best quality
FRAME 5: 1girl, solo, elf, pointed ears, athletic build, full draw aiming, leather armor, side view, full body shot, forest background, masterpiece, best quality
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
                clean_line = line.replace('*', '').replace('#', '').strip()

                if clean_line.upper().startswith("FRAME"):
                    if ":" in clean_line:
                        p = clean_line.split(":", 1)[1].strip()
                        keyframes.append(p)
                elif clean_line.upper().startswith("NEGATIVE"):
                    if ":" in clean_line:
                        negative = clean_line.split(":", 1)[1].strip()

            if not negative:
                negative = f"bad anatomy, bad hands, blurry, low quality, text, watermark, {negative_tags}"

            if len(keyframes) >= 5:
                return {
                    "prompts": keyframes[:5],
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

            elif len(keyframes) >= 3:
                k1, k2, k3 = keyframes[0], keyframes[1], keyframes[2]
                i1 = _interpolate_prompt(k1, k2, persona_name)
                i2 = _interpolate_prompt(k2, k3, persona_name)
                return {
                    "prompts": [k1, i1, k2, i2, k3],
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

            elif keyframes:
                return {
                    "prompts": keyframes,
                    "negative_prompt": negative,
                    "llm_used": PROMPT_MODEL
                }

    except Exception as e:
        logger.warning(f"LLM animation prompt generation failed: {e}")

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
        logger.warning(f"LLM comic script generation failed: {e}")

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
    """Generate a visual_identity YAML structure for a persona."""
    system_prompt = f"""You are a character designer.
Create a detailed 'visual_identity' specification for a character named '{persona_name}'.
Context: {description}

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this structure:
{{
  "name": "{persona_name}",
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
    "build": "slim"
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
        logger.warning(f"Visual identity generation failed: {e}")
    return {}


def test_generator():
    """Test the prompt generator."""
    test_requests = [
        "send me a selfie",
        "show me a portrait",
        "action pose with a sword",
        "standing in a cyberpunk city",
    ]

    for req in test_requests:
        print(f"\n{'='*60}")
        print(f"REQUEST: {req}")
        print(f"{'='*60}")
        result = generate_sd_prompt(req, "aither")
        print(f"PROMPT: {result['prompt']}")
        print(f"NEGATIVE: {result['negative_prompt']}")
        print(f"LLM: {result.get('llm_used', 'unknown')}")


if __name__ == "__main__":
    test_generator()
