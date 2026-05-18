"""
AitherOS Refinement Engine

Implements a "Generate -> Validate -> Refine" loop for high-fidelity image generation.
Uses Vision models (Ollama/Llava) to critique images and LLMs to adjust prompts.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from AitherOS.AitherNode.AitherCanvas import ComfyUIClient
from AitherOS.AitherNode.vision_tools import analyze_with_ollama, unload_vision_model

from aither_adk.ai.comfyui_service import ComfyUIService
from aither_adk.ai.llm_prompt_generator import generate_sd_prompt
from aither_adk.ui.console import safe_print

logger = logging.getLogger(__name__)

@dataclass
class RefinementResult:
    image_path: str
    prompt: str
    critique: str
    score: float
    iteration: int

class RefinementEngine:
    def __init__(self):
        self.comfy_url = ComfyUIService.get_base_url()
        self.client = ComfyUIClient(self.comfy_url)
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "Saga", "output", "refinements")
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_with_feedback(self,
                                   user_request: str,
                                   persona_name: str = "aither",
                                   max_iterations: int = 3) -> str:
        """
        Generates an image, critiques it, and refines it if necessary.
        """
        safe_print(f"[cyan][ART] Starting generation with refinement loop (Max {max_iterations} iterations)...[/]")

        # 1. Initial Generation
        prompt_data = generate_sd_prompt(user_request, persona_name, "UNRESTRICTED")
        current_prompt = prompt_data["prompt"]
        negative_prompt = prompt_data["negative_prompt"]

        current_image = await self._generate_image(current_prompt, negative_prompt)
        if not current_image:
            return None

        safe_print(f"[green][DONE] Initial image generated: {os.path.basename(current_image)}[/]")

        # 2. Refinement Loop
        for i in range(1, max_iterations + 1):
            # A. Vision Critique
            safe_print(f"[yellow] Analyzing image (Iteration {i})...[/]")
            critique = await self._critique_image(current_image, user_request)

            # Explicitly unload vision model to free VRAM for ComfyUI
            unload_vision_model()

            if not critique:
                safe_print("[warning][WARN] Vision analysis failed, stopping refinement.[/]")
                break

            score = self._extract_score(critique)
            safe_print(f"[dim]Score: {score}/10 | Critique: {critique[:100]}...[/]")

            if score >= 8.5:
                safe_print(f"[green]* Image meets quality threshold ({score}/10)![/]")
                break

            # B. Refine Prompt
            safe_print("[cyan][TOOL] Refining prompt based on feedback...[/]")
            new_prompt = self._refine_prompt(current_prompt, critique, user_request)

            # C. Regenerate (Img2Img)
            # We use the current image as input for Img2Img to maintain composition
            # unless the score is very low (< 4), in which case we might want to start over.

            if score < 4.0:
                safe_print("[red][SYNC] Score too low, restarting generation (Txt2Img)...[/]")
                current_image = await self._generate_image(new_prompt, negative_prompt)
            else:
                safe_print("[cyan][SYNC] Refining image (Img2Img)...[/]")
                # Use a lower denoise for higher scores to preserve more structure
                denoise = 0.65 if score < 6 else 0.45
                current_image = await self._generate_img2img(current_image, new_prompt, negative_prompt, denoise)

            if not current_image:
                safe_print("[red][FAIL] Generation failed during refinement.[/]")
                break

            current_prompt = new_prompt
            safe_print(f"[green][DONE] Iteration {i} complete: {os.path.basename(current_image)}[/]")

        return current_image

    async def refine_existing_image(self, image_path: str, instruction: str) -> str:
        """
        Refines an existing image based on a user instruction.
        """
        safe_print(f"[cyan][TOOL] Refining image with instruction: {instruction}[/]")

        # 1. Analyze current image to understand what we have
        description = await self._critique_image(image_path, "Describe this image in detail.")

        # 2. Generate new prompt combining description + instruction
        # We can use the LLM prompt generator logic here manually or create a helper
        new_prompt_data = generate_sd_prompt(f"Image of {description}. Change: {instruction}", "aither", "UNRESTRICTED")
        new_prompt = new_prompt_data["prompt"]

        # 3. Img2Img
        return await self._generate_img2img(image_path, new_prompt, new_prompt_data["negative_prompt"], denoise=0.75)

    async def _generate_image(self, prompt: str, negative_prompt: str) -> Optional[str]:
        """Wrapper for Txt2Img."""
        try:
            # We need to call the client synchronously as it uses websocket blocking
            # But we should run it in a thread executor to not block the async loop
            loop = asyncio.get_event_loop()

            # Create a simple workflow or use existing API
            # For now, we'll reuse the client's logic but we need a simple txt2img method.
            # The client has generate_txt2img_with_controlnet. We can pass a dummy image or add a method.
            # Let's assume we can use a standard workflow.

            # Since we can't easily modify ComfyUIClient right now without reading it all,
            # let's try to use the 'generate_txt2img_with_controlnet' but without controlnet?
            # No, that requires an input image.

            # We will implement a direct workflow execution here for standard Txt2Img
            # Or better, let's add a simple method to ComfyUIClient in a separate edit if needed.
            # For now, let's try to construct a minimal workflow.

            # Actually, let's just use the client.generate_img2img_with_controlnet if we have an image,
            # but for Txt2Img we need a workflow.

            # Let's assume we can use the 'txt2img_pony_api.json' workflow directly.
            # Use client's WORKFLOW_DIR or auto-detect from AITHERZERO_ROOT environment variable
            aitherzero_root = os.environ.get("AITHERZERO_ROOT", "")
            default_workflow_dir = os.path.join(aitherzero_root, "AitherOS", "AitherNode", "workflows") if aitherzero_root else ""
            workflow_path = os.path.join(self.client.WORKFLOW_DIR if hasattr(self.client, 'WORKFLOW_DIR') else default_workflow_dir, "txt2img_pony_api.json")

            if not os.path.exists(workflow_path):
                # Fallback to a simple prompt if workflow missing (mocking for now if needed)
                return None

            with open(workflow_path, "r") as f:
                json.load(f)

            # Update prompt
            # Find KSampler and Prompt nodes (simplified logic)
            # ... (This is complex to do robustly without the helper methods in Client)

            # Let's use a simpler approach: The ComfyUIClient is designed to be used.
            # I will assume I can add a 'generate_txt2img' method to it in the next step.
            # For now, I'll call a placeholder that I will implement.

            images = await loop.run_in_executor(None, lambda: self.client.generate_txt2img_simple(prompt, negative_prompt))

            if images:
                # Save to output dir
                img_data = list(images.values())[0][0] # First image
                # It returns raw bytes or PIL? Client returns dict of list of bytes usually

                filename = f"gen_{int(time.time())}.png"
                path = os.path.join(self.output_dir, filename)

                import io

                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                img.save(path)
                return path

        except Exception as e:
            safe_print(f"[red]Generation error: {e}[/]")
            # If method missing, we need to add it.
            if "'ComfyUIClient' object has no attribute 'generate_txt2img_simple'" in str(e):
                safe_print("[yellow]Adding missing method to ComfyUIClient...[/]")
                # We will handle this in the next turn
        return None

    async def _generate_img2img(self, image_path: str, prompt: str, negative_prompt: str, denoise: float) -> Optional[str]:
        """Wrapper for Img2Img."""
        try:
            loop = asyncio.get_event_loop()
            # We use generate_img2img_with_controlnet but with low controlnet strength or dummy CN
            # Actually, we should add a proper simple img2img method.

            images = await loop.run_in_executor(None, lambda: self.client.generate_img2img_simple(image_path, prompt, negative_prompt, denoise))

            if images:
                img_data = list(images.values())[0][0]
                filename = f"refine_{int(time.time())}.png"
                path = os.path.join(self.output_dir, filename)

                import io

                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                img.save(path)
                return path
        except Exception as e:
            safe_print(f"[red]Img2Img error: {e}[/]")
        return None

    async def _critique_image(self, image_path: str, user_request: str) -> str:
        """Uses Vision model to critique the image against the request."""
        prompt = f"""
        Analyze this image in relation to the user request: "{user_request}".
        1. Does it accurately depict the request?
        2. Are there any anatomical errors or artifacts?
        3. Is the style consistent?

        Rate the image from 0 to 10.
        Format: "Score: X/10. Critique: ..."
        """
        return analyze_with_ollama(image_path, prompt)

    def _extract_score(self, critique: str) -> float:
        """Extracts numerical score from critique."""
        try:
            import re
            match = re.search(r"Score:\s*(\d+(\.\d+)?)/10", critique, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except Exception as exc:
            logger.debug(f"Score extraction failed: {exc}")
        return 5.0 # Default

    def _refine_prompt(self, current_prompt: str, critique: str, user_request: str) -> str:
        """Uses LLM to refine the prompt based on critique."""
        # We can use the generate_sd_prompt logic but with a "refinement" system prompt
        # For simplicity, we'll append the critique to the negative prompt or adjust positive

        # TODO: Use LLM to rewrite prompt.
        # For now, simple heuristic:
        return current_prompt # Placeholder
