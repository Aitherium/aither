"""
Fast Image Dispatcher - Direct ComfyUI integration for sub-15-second generation.

This module bypasses the LLM entirely for high-confidence image requests.
Instead of:
  User -> LLM thinks (30s) -> LLM calls tool -> ComfyUI (12s) -> LLM formats (10s)
  
We do:
  User -> Pattern match (0s) -> Template enhance (5ms) -> ComfyUI (12s) -> Done

Target: 15 seconds or less for simple image requests.
"""

import os
import re
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ComfyUI connection settings
COMFY_URL = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = os.getenv("AITHER_OUTPUT_DIR", str(Path.home() / "Pictures" / "AitherCanvas"))

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


@dataclass
class PromptTemplate:
    """Template for different image request types."""
    name: str
    base_prompt: str
    style_tags: str
    quality_tags: str
    negative_prompt: str


# Pre-defined templates for fast prompt building
TEMPLATES = {
    "selfie": PromptTemplate(
        name="selfie",
        base_prompt="1girl, selfie, smartphone, pov, looking at viewer, beautiful face, detailed eyes",
        style_tags="photorealistic, natural lighting, soft focus background",
        quality_tags="masterpiece, best quality, highres, absurdres",
        negative_prompt="bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, text"
    ),
    "portrait": PromptTemplate(
        name="portrait",
        base_prompt="1girl, portrait, upper body, looking at viewer, beautiful face",
        style_tags="professional photography, studio lighting, bokeh",
        quality_tags="masterpiece, best quality, highres",
        negative_prompt="bad anatomy, bad hands, worst quality, low quality, watermark, text"
    ),
    "nude_selfie": PromptTemplate(
        name="nude_selfie",
        base_prompt="1girl, selfie, nude, naked, smartphone, pov, looking at viewer, beautiful face, detailed eyes, breasts, nipples",
        style_tags="photorealistic, natural lighting, bedroom, intimate",
        quality_tags="masterpiece, best quality, highres, absurdres, detailed skin",
        negative_prompt="bad anatomy, bad hands, missing fingers, worst quality, low quality, watermark, censored"
    ),
    "generic": PromptTemplate(
        name="generic",
        base_prompt="",
        style_tags="high quality, detailed",
        quality_tags="masterpiece, best quality",
        negative_prompt="bad anatomy, worst quality, low quality, watermark, text, blurry"
    )
}


def detect_template(user_input: str) -> str:
    """Detect which template to use based on user input."""
    text = user_input.lower()
    
    # Check for NSFW + selfie
    nsfw_words = {"nude", "naked", "nsfw", "explicit", "sexy", "topless", "undressed"}
    selfie_words = {"selfie", "selfies", "self-portrait"}
    
    has_nsfw = any(w in text for w in nsfw_words)
    has_selfie = any(w in text for w in selfie_words)
    
    if has_nsfw and has_selfie:
        return "nude_selfie"
    elif has_selfie:
        return "selfie"
    elif "portrait" in text:
        return "portrait"
    else:
        return "generic"


def build_prompt(user_input: str, template: PromptTemplate, persona_tags: str = "") -> Tuple[str, str]:
    """
    Build enhanced prompt from user input and template.
    
    Returns: (positive_prompt, negative_prompt)
    """
    parts = []
    
    # Quality tags first (for Pony/SDXL models)
    if template.quality_tags:
        parts.append(template.quality_tags)
    
    # Base prompt from template
    if template.base_prompt:
        parts.append(template.base_prompt)
    
    # User's actual request (cleaned up)
    clean_input = user_input.strip()
    # Remove common prefixes
    for prefix in ["send me ", "show me ", "generate ", "create ", "draw ", "make "]:
        if clean_input.lower().startswith(prefix):
            clean_input = clean_input[len(prefix):]
            break
    
    # Add user input if it adds meaningful context
    if clean_input and len(clean_input) > 3:
        parts.append(clean_input)
    
    # Persona-specific tags
    if persona_tags:
        parts.append(persona_tags)
    
    # Style tags
    if template.style_tags:
        parts.append(template.style_tags)
    
    positive = ", ".join(parts)
    negative = template.negative_prompt
    
    return positive, negative


class FastImageDispatcher:
    """Direct ComfyUI dispatcher for fast image generation."""
    
    def __init__(self, comfy_url: str = COMFY_URL):
        self.comfy_url = comfy_url.rstrip("/")
        self.client_id = f"fast_dispatch_{int(time.time())}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._cached_model: Optional[str] = None
        self._failed_models: set = set()  # Track models that failed at runtime
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def check_connection(self) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.comfy_url}/system_stats", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def get_available_model(self, exclude_model: str = None) -> Optional[str]:
        """Get the best available checkpoint model from ComfyUI.
        
        Args:
            exclude_model: If set, skip this model (e.g., if it just failed)
        """
        # Clear cache if we're excluding the cached model
        if exclude_model and self._cached_model == exclude_model:
            self._failed_models.add(exclude_model)
            self._cached_model = None
        
        if self._cached_model:
            return self._cached_model
            
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.comfy_url}/object_info/CheckpointLoaderSimple",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get('CheckpointLoaderSimple', {}).get('input', {}).get('required', {}).get('ckpt_name', [[]])[0]
                    if models:
                        # Skip models that failed at runtime (corrupted files, etc.)
                        valid_models = [m for m in models if m not in self._failed_models]
                        
                        # Filter out obvious video models (by naming convention)
                        image_models = [m for m in valid_models if not any(
                            x in m.lower() for x in ['wan2', 'i2v', 't2v', 'video', 'animatediff']
                        )]
                        
                        if not image_models:
                            image_models = valid_models  # Fall back if filter too aggressive
                        
                        # 1. Pony models (best for anime/photorealistic)
                        pony = [m for m in image_models if 'pony' in m.lower()]
                        if pony:
                            self._cached_model = pony[0]
                            return self._cached_model
                        
                        # 2. Illustrious/wai models (high quality SDXL)
                        wai = [m for m in image_models if 'wai' in m.lower() or 'illustrious' in m.lower()]
                        if wai:
                            self._cached_model = wai[0]
                            return self._cached_model
                        
                        # 3. SDXL models
                        sdxl = [m for m in image_models if 'sdxl' in m.lower() or 'xl' in m.lower()]
                        if sdxl:
                            self._cached_model = sdxl[0]
                            return self._cached_model
                        
                        # 4. Flux models
                        flux = [m for m in image_models if 'flux' in m.lower()]
                        if flux:
                            self._cached_model = flux[0]
                            return self._cached_model
                        
                        # 5. Any remaining image model
                        if image_models:
                            self._cached_model = image_models[0]
                            return self._cached_model
                        
        except Exception as e:
            print(f"Could not get models: {e}")
        return None
    
    async def queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        session = await self._get_session()
        payload = {"prompt": workflow, "client_id": self.client_id}
        
        async with session.post(f"{self.comfy_url}/prompt", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Failed to queue prompt: {resp.status} - {text}")
            data = await resp.json()
            return data["prompt_id"]
    
    async def wait_for_completion(self, prompt_id: str, timeout: float = 120.0) -> dict:
        """Poll history until generation is complete."""
        session = await self._get_session()
        start = time.time()
        
        while (time.time() - start) < timeout:
            async with session.get(f"{self.comfy_url}/history/{prompt_id}") as resp:
                if resp.status != 200:
                    await asyncio.sleep(0.5)
                    continue
                    
                history = await resp.json()
                
                if prompt_id not in history:
                    await asyncio.sleep(0.5)
                    continue
                
                entry = history[prompt_id]
                status = entry.get("status", {}).get("status_str", "")
                
                if status == "success":
                    return entry
                elif status == "error":
                    messages = entry.get("status", {}).get("messages", [])
                    error_msg = "Unknown error"
                    for msg in messages:
                        if msg[0] == "execution_error":
                            error_msg = msg[1].get("exception_message", error_msg)
                            break
                    raise Exception(f"ComfyUI error: {error_msg}")
                
                await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")
    
    async def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Fetch generated image data."""
        session = await self._get_session()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        
        async with session.get(f"{self.comfy_url}/view", params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch image: {resp.status}")
            return await resp.read()
    
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 12,
        seed: int = None,
        _retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Generate an image with minimal overhead.
        
        Uses a simple txt2img workflow optimized for speed.
        If a model fails (corrupted file, etc.), automatically tries the next available model.
        """
        import random
        
        MAX_RETRIES = 3
        
        if seed is None:
            seed = random.randint(1, 2147483647)
        
        # Get available model dynamically
        model_name = await self.get_available_model()
        if not model_name:
            return {"success": False, "error": "No checkpoint models available in ComfyUI"}
        
        print(f"[ZAP] Fast generation: {model_name}, {steps} steps")
        
        # Build minimal workflow - optimized for speed
        # This is a simplified SDXL workflow with 12 steps
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_sde",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model_name
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "fast_gen",
                    "images": ["8", 0]
                }
            }
        }
        
        start_time = time.time()
        
        try:
            # Queue the prompt
            prompt_id = await self.queue_prompt(workflow)
            
            # Wait for completion
            result = await self.wait_for_completion(prompt_id)
        except Exception as e:
            error_msg = str(e)
            # Check if this is a model loading error
            if any(x in error_msg.lower() for x in ['deserializ', 'checkpoint', 'load', 'header', 'corrupt', 'invalid']):
                print(f"[WARN] Model {model_name} failed: {error_msg}")
                if _retry_count < MAX_RETRIES:
                    # Mark this model as failed and try the next one
                    await self.get_available_model(exclude_model=model_name)
                    print(f"[SYNC] Retrying with different model (attempt {_retry_count + 1}/{MAX_RETRIES})")
                    return await self.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        seed=seed,
                        _retry_count=_retry_count + 1
                    )
            return {"success": False, "error": error_msg}
        
        # Extract output images
        outputs = result.get("outputs", {})
        paths = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    # Fetch and save locally
                    img_data = await self.get_image(
                        img_info["filename"],
                        img_info.get("subfolder", ""),
                        img_info.get("type", "output")
                    )
                    
                    # Save to output directory
                    timestamp = int(time.time())
                    clean_name = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
                    filename = f"aither_{timestamp}_{seed}_{clean_name}.png"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    
                    paths.append(filepath)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "paths": paths,
            "seed": seed,
            "elapsed": elapsed,
            "prompt": prompt
        }


# Singleton dispatcher
_dispatcher: Optional[FastImageDispatcher] = None


def get_dispatcher() -> FastImageDispatcher:
    """Get the singleton fast image dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = FastImageDispatcher()
    return _dispatcher


async def fast_generate_image(
    user_input: str,
    safety_config: Any = None,
    persona_tags: str = ""
) -> Dict[str, Any]:
    """
    Fast path for image generation - bypasses LLM entirely.
    
    Args:
        user_input: The user's request (e.g., "send me a selfie")
        safety_config: Safety configuration (for NSFW allowance)
        persona_tags: Optional persona-specific tags to inject
        
    Returns:
        Dict with success, paths, elapsed time, etc.
    """
    try:
        # Detect appropriate template
        template_name = detect_template(user_input)
        
        # Check if NSFW is allowed
        allow_explicit = False
        if safety_config and hasattr(safety_config, 'allow_explicit'):
            allow_explicit = safety_config.allow_explicit
        
        # Downgrade NSFW template if not allowed
        if template_name == "nude_selfie" and not allow_explicit:
            template_name = "selfie"
        
        template = TEMPLATES.get(template_name, TEMPLATES["generic"])
        
        # Build the prompt
        positive, negative = build_prompt(user_input, template, persona_tags)
        
        # Get dispatcher and generate
        dispatcher = get_dispatcher()
        
        # Check connection first
        if not await dispatcher.check_connection():
            return {
                "success": False,
                "error": "ComfyUI not reachable. Is it running on port 8188?"
            }
        
        # Generate!
        result = await dispatcher.generate(
            prompt=positive,
            negative_prompt=negative,
            width=1024,
            height=1024,
            steps=12  # Fast generation
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
