"""
AitherOS Hybrid Executor

This module provides drop-in replacements for the existing dispatch functions
that integrate with the ResourceManager and TaskRouter for intelligent
hybrid cloud/local execution.

Usage:
    from aither_adk.infrastructure.hybrid_executor import (
        delegate_image_smart,
        delegate_text_smart,
        execute_hybrid
    )
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import our new components
try:
    from aither_adk.infrastructure.resource_manager import TaskType, resource_manager
    from aither_adk.infrastructure.task_router import ContentRating, get_router
except ImportError:
    # Handle relative imports when running as script
    from resource_manager import TaskType, resource_manager
    from task_router import get_router


class HybridExecutor:
    """
    Intelligent task executor that coordinates between cloud and local backends.

    Key improvements over original delegate_image_generation:
    - Does NOT spawn competing processes simultaneously
    - Checks GPU availability before launching local tasks
    - Smart content-based routing
    - Sequential execution to prevent VRAM conflicts
    """

    def __init__(self):
        self.router = get_router()
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def execute_text_generation(
        self,
        instruction: str,
        system_prompt: str = None,
        model: str = None,
        mailbox_path: str = None,
        sender_name: str = "Aither"
    ) -> str:
        """
        Execute text generation with smart routing.

        Returns the generated text directly instead of spawning a subprocess.
        """
        # Route the task
        decision = self.router.route_text_generation(instruction, system_prompt)

        print(f"[HybridExecutor] Text routing: {decision.provider}/{decision.model} ({decision.reason})")

        if decision.use_local:
            # Use local Ollama with resource management
            return await self._execute_local_text(instruction, system_prompt, decision)
        else:
            # Use cloud (Gemini)
            return await self._execute_cloud_text(instruction, system_prompt, decision)

    async def _execute_local_text(
        self,
        instruction: str,
        system_prompt: str,
        decision
    ) -> str:
        """Execute text generation on local Ollama"""
        try:
            from AitherOS.AitherNode.async_llm import get_client

            async with resource_manager.acquire_gpu(TaskType.LLM_LOCAL, timeout=120):
                client = get_client()

                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction}
                    ]
                    response = await client.chat(messages, model=decision.model)
                else:
                    response = await client.generate(instruction, model=decision.model)

                return response.text

        except TimeoutError:
            print("[HybridExecutor] Local LLM timeout - resources busy")
            if decision.fallback_provider:
                print("[HybridExecutor] Falling back to cloud")
                return await self._execute_cloud_text(instruction, system_prompt, decision)
            raise

    async def _execute_cloud_text(
        self,
        instruction: str,
        system_prompt: str,
        decision
    ) -> str:
        """Execute text generation on cloud (Gemini)"""
        try:
            from google.genai import Client, types

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("No Google API key configured")

            client = Client(api_key=api_key)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
            )

            response = client.models.generate_content(
                model=decision.model,
                contents=instruction,
                config=config
            )

            # Manually extract text to avoid "non-text parts" warning from SDK
            if response.candidates and response.candidates[0].content.parts:
                text_parts = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return "".join(text_parts)
                # If we have parts but no text (e.g. only function calls), return empty string
                return ""

            return response.text

        except Exception as e:
            print(f"[HybridExecutor] Cloud text generation failed: {e}")
            raise

    async def execute_image_generation(
        self,
        prompt: str,
        style: str = "realistic",
        model: str = None,
        negative_prompt: str = None,
        tool_context=None
    ) -> Dict[str, Any]:
        """
        Execute image generation with smart routing.

        IMPORTANT: This runs SEQUENTIALLY to prevent VRAM conflicts.
        The original delegate_image_generation spawned both ArtistAgent
        AND Aither simultaneously, causing contention.
        """
        # Route the task
        decision = self.router.route_image_generation(prompt, model_preference=model)

        print(f"[HybridExecutor] Image routing: {decision.provider}/{decision.model} ({decision.reason})")

        if decision.use_local:
            return await self._execute_local_image(prompt, style, negative_prompt, decision, tool_context)
        else:
            return await self._execute_cloud_image(prompt, style, decision, tool_context)

    async def execute_refinement(
        self,
        image_path: str,
        prompt: str,
        denoise: float = 0.5,
        negative_prompt: str = ""
    ) -> Dict[str, Any]:
        """Execute image refinement using local ComfyUI"""
        try:
            from AitherOS.AitherNode.AitherCanvas import ComfyUIClient
            from AitherOS.AitherNode.config import COMFYUI_SERVER_ADDRESS

            # Wait for GPU resources
            async with resource_manager.acquire_gpu(TaskType.DIFFUSION_SDXL, timeout=300):
                client = ComfyUIClient(COMFYUI_SERVER_ADDRESS)
                paths = client.refine_image(
                    image_path=image_path,
                    prompt_text=prompt,
                    denoise=denoise,
                    negative_prompt=negative_prompt
                )

                return {
                    "status": "success",
                    "provider": "local_comfyui",
                    "paths": paths,
                    "filename": paths[0] if paths else None
                }
        except Exception as e:
            print(f"[HybridExecutor] Refinement failed: {e}")
            return {"status": "error", "error": str(e)}

    async def execute_animation(
        self,
        prompts: list[str],
        output_dir: str = "generated_animations",
        fps: int = 24
    ) -> Dict[str, Any]:
        """Execute animation generation using local ComfyUI"""
        try:
            from AitherOS.AitherNode.tools.animation_tools import generate_animation

            # Wait for GPU resources (Animation takes longer)
            async with resource_manager.acquire_gpu(TaskType.DIFFUSION_FLUX, timeout=600):
                video_path = generate_animation(
                    prompts=prompts,
                    output_dir=output_dir,
                    fps=fps
                )

                return {
                    "status": "success",
                    "provider": "local_comfyui",
                    "video_path": video_path
                }
        except Exception as e:
            print(f"[HybridExecutor] Animation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_local_image(
        self,
        prompt: str,
        style: str,
        negative_prompt: str,
        decision,
        tool_context
    ) -> Dict[str, Any]:
        """Execute image generation on local ComfyUI"""
        try:
            from AitherOS.AitherNode.AitherCanvas import generate_local

            # Wait for GPU resources
            task_type = TaskType.DIFFUSION_SDXL if decision.model == "pony" else TaskType.DIFFUSION_FLUX

            async with resource_manager.acquire_gpu(task_type, timeout=300):
                paths = await generate_local(
                    prompt,
                    tool_context=tool_context,
                    style=style,
                    negative_prompt=negative_prompt
                )

                return {
                    "status": "success",
                    "provider": "local_comfyui",
                    "model": decision.model,
                    "paths": paths,
                    "filename": paths[0] if paths else None
                }

        except TimeoutError:
            print("[HybridExecutor] Local diffusion timeout - resources busy")
            if decision.fallback_provider:
                print("[HybridExecutor] Falling back to cloud")
                return await self._execute_cloud_image(prompt, style, decision, tool_context)
            raise

    async def _execute_cloud_image(
        self,
        prompt: str,
        style: str,
        decision,
        tool_context
    ) -> Dict[str, Any]:
        """Execute image generation on cloud (Fal.ai)"""
        try:
            from AitherOS.agents.common.tools.fal_tools import generate_image_with_fal

            result = await generate_image_with_fal(
                prompt,
                tool_context,
                style=style,
                model=decision.model
            )

            return {
                "status": "success",
                "provider": "fal",
                "model": decision.model,
                "result": result
            }

        except Exception as e:
            print(f"[HybridExecutor] Cloud image generation failed: {e}")
            raise

    async def execute_combined_generation(
        self,
        instruction: str,
        mailbox_path: str = None
    ) -> Dict[str, Any]:
        """
        Execute a combined text + image generation request.

        This is the SMART replacement for delegate_image_generation.
        Instead of spawning two competing processes:
        1. First routes and executes the image generation
        2. Then (optionally) generates accompanying text
        3. Returns results to mailbox if provided

        This SEQUENTIAL approach prevents VRAM conflicts.
        """
        results = {}

        # 1. Generate image first (resource-intensive)
        print("[HybridExecutor] Starting image generation...")
        try:
            image_result = await self.execute_image_generation(instruction)
            results["image"] = image_result
        except Exception as e:
            print(f"[HybridExecutor] Image generation failed: {e}")
            results["image"] = {"status": "error", "error": str(e)}

        # 2. Generate text response (can use cloud while GPU cools down)
        print("[HybridExecutor] Generating text response...")
        try:
            text_prompt = f"An image was just generated based on: '{instruction}'. Write a brief, engaging response acknowledging the image."
            text_result = await self.execute_text_generation(
                text_prompt,
                system_prompt="You are Aither, a helpful AI assistant. Be concise and friendly."
            )
            results["text"] = text_result
        except Exception as e:
            print(f"[HybridExecutor] Text generation failed: {e}")
            results["text"] = f"Image generated! (Text response unavailable: {e})"

        # 3. Send to mailbox if provided
        if mailbox_path:
            try:
                from aither_adk.communication.mailbox import Mailbox
                mailbox = Mailbox(mailbox_path)

                # Compose message
                image_path = results.get("image", {}).get("filename", "")
                content = results.get("text", "Image generated!")
                if image_path:
                    content += f"\n\n![Generated Image]({image_path})"

                mailbox.send_message(
                    sender="ArtistAgent",
                    recipient="user",
                    subject=f"Generated: {instruction[:50]}...",
                    content=content
                )
            except Exception as e:
                print(f"[HybridExecutor] Mailbox error: {e}")

        return results


# Singleton executor
_executor: Optional[HybridExecutor] = None


def get_executor() -> HybridExecutor:
    """Get the singleton executor"""
    global _executor
    if _executor is None:
        _executor = HybridExecutor()
    return _executor


# Drop-in replacement functions

async def delegate_image_smart(instruction: str, mailbox_path: str = None) -> str:
    """
    Smart replacement for delegate_image_generation.

    Key differences:
    - Executes SEQUENTIALLY instead of spawning parallel processes
    - Checks GPU resources before local execution
    - Routes NSFW to local, SFW to cloud
    - Returns immediately instead of just returning "delegated" message
    """
    executor = get_executor()
    results = await executor.execute_combined_generation(instruction, mailbox_path)

    # Return a summary for the caller
    image_status = results.get("image", {}).get("status", "unknown")
    if image_status == "success":
        path = results.get("image", {}).get("filename", "")
        return f"Image generated successfully: {path}"
    else:
        return f"Image generation completed with status: {image_status}"


async def delegate_text_smart(
    instruction: str,
    system_prompt: str = None,
    model: str = None
) -> str:
    """
    Smart text generation with automatic routing.
    """
    executor = get_executor()
    return await executor.execute_text_generation(
        instruction,
        system_prompt=system_prompt,
        model=model
    )


def delegate_image_sync(instruction: str, mailbox_path: str = None) -> str:
    """
    Synchronous wrapper for delegate_image_smart.

    Use this as a drop-in replacement in non-async code.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(delegate_image_smart(instruction, mailbox_path))


async def delegate_animation_smart(prompts: list[str], output_dir: str = "generated_animations") -> str:
    """Smart delegation for animation generation."""
    executor = get_executor()
    result = await executor.execute_animation(prompts, output_dir)

    if result.get("status") == "success":
        return f"Animation generated successfully: {result.get('video_path')}"
    else:
        return f"Animation generation failed: {result.get('error')}"

async def delegate_refinement_smart(image_path: str, prompt: str, denoise: float = 0.5) -> str:
    """Smart delegation for image refinement."""
    executor = get_executor()
    result = await executor.execute_refinement(image_path, prompt, denoise)

    if result.get("status") == "success":
        path = result.get("filename")
        return f"Image refined successfully: {path}"
    else:
        return f"Refinement failed: {result.get('error')}"


# Quick status check
def print_resource_status():
    """Print current resource status"""
    resource_manager.print_status()


# Pre-flight check
async def preflight_check() -> Dict[str, bool]:
    """
    Run a pre-flight check to verify all backends are available.

    Checks:
        - Ollama: Local LLM inference
        - AitherCanvas: ComfyUI image generation
        - Google API: Cloud API key configured
        - GPU: NVIDIA GPU with sufficient VRAM
    """
    results = {
        "ollama": False,
        "aithercanvas": False,
        "google_api": False,
        "gpu_available": False
    }

    # Check Ollama
    try:
        from AitherOS.AitherNode.async_llm import get_client
        client = get_client()
        results["ollama"] = await client.is_available()
    except Exception as exc:
        logger.debug(f"Ollama availability check failed: {exc}")

    # Check AitherCanvas (ComfyUI)
    try:
        from AitherOS.agents.common.comfyui_service import ComfyUIService
        results["aithercanvas"] = ComfyUIService.is_reachable()
    except Exception as exc:
        logger.debug(f"AitherCanvas reachability check failed: {exc}")

    # Check Google API
    results["google_api"] = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    # Check GPU
    try:
        vram = resource_manager.gpu_monitor.get_vram_info()
        # Check if we have a capable GPU (Total VRAM > 6GB), not just free VRAM
        # The Resource Manager handles unloading, so we don't need free VRAM right now.
        results["gpu_available"] = vram["total"] > 6000
    except Exception as exc:
        logger.debug(f"GPU availability check failed: {exc}")

    # Use Rich formatting for a cleaner look
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console(force_terminal=True)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Status", width=3)
    table.add_column("Service", style="bold")

    # Display names for services (more user-friendly)
    display_names = {
        "ollama": "Ollama",
        "aithercanvas": "AitherCanvas",
        "google_api": "Google API",
        "gpu_available": "GPU Available"
    }

    for service, available in results.items():
        icon = "[green][OK][/]" if available else "[red][X][/]"
        name = display_names.get(service, service.replace("_", " ").title())
        table.add_row(icon, name)

    panel = Panel(
        table,
        title="[bold cyan][ZAP] System Status[/]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    console.print(panel)

    return results

