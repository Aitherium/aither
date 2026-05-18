"""
AitherOS Anchor Automation

Automates the generation of reference (anchor) images for all personas.
Ensures every persona has a consistent Face and Body anchor for the image system.
"""

import os
import yaml
import asyncio
import time
from typing import Dict, Any, List

from aither_adk.ai.persona_image_system import PersonaImageSystem, VisualIdentity
from aither_adk.ai.comfyui_service import ComfyUIService
from AitherOS.AitherNode.AitherCanvas import ComfyUIClient
from aither_adk.ui.console import safe_print

try:
    from aither_adk.paths import get_saga_config_dir
    PERSONAS_DIR = get_saga_config_dir("personas")
except ImportError:
    PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "..", "Saga", "config", "personas")

class AnchorAutomation:
    def __init__(self):
        self.system = PersonaImageSystem()
        self.comfy_url = ComfyUIService.get_base_url()
        self.client = ComfyUIClient(self.comfy_url)
        
    async def initialize_all_personas(self, force_regenerate: bool = False):
        """
        Iterates through all persona YAMLs and generates missing anchors.
        """
        safe_print("[bold cyan][LAUNCH] Starting Persona Anchor Initialization...[/]")
        
        # 1. List all personas
        yaml_files = [f for f in os.listdir(PERSONAS_DIR) if f.endswith(".yaml") and not f.startswith("_")]
        
        for f in yaml_files:
            persona_name = os.path.splitext(f)[0]
            await self._process_persona(persona_name, force_regenerate)
            
        safe_print("[bold green]* Anchor Initialization Complete![/]")

    async def _process_persona(self, name: str, force: bool):
        """Process a single persona."""
        safe_print(f"\n[cyan] Processing Persona: {name.title()}[/]")
        
        # Load YAML
        path = os.path.join(PERSONAS_DIR, f"{name}.yaml")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            safe_print(f"[red]Failed to load {name}.yaml: {e}[/]")
            return

        visual_id = data.get("visual_identity")
        if not visual_id:
            safe_print(f"[yellow][WARN] No visual_identity found for {name}. Skipping.[/]")
            return

        # Check existing anchors
        anchor = self.system.anchors.get(name.lower())
        has_face = anchor and anchor.face_reference and os.path.exists(anchor.face_reference)
        has_body = anchor and anchor.body_reference and os.path.exists(anchor.body_reference)
        
        if has_face and has_body and not force:
            safe_print(f"[green][DONE] Anchors already exist for {name}.[/]")
            return

        # Generate Face
        if not has_face or force:
            safe_print(f"[dim]Generating FACE anchor for {name}...[/]")
            face_prompt = self._build_prompt(visual_id, "face")
            face_path = await self._generate_image(face_prompt, f"{name}_face_gen")
            if face_path:
                self.system.set_anchor_image(name, face_path, "face")
                safe_print(f"[green][DONE] Face anchor set.[/]")
            else:
                safe_print(f"[red][FAIL] Failed to generate face anchor.[/]")

        # Generate Body
        if not has_body or force:
            safe_print(f"[dim]Generating BODY anchor for {name}...[/]")
            body_prompt = self._build_prompt(visual_id, "body")
            body_path = await self._generate_image(body_prompt, f"{name}_body_gen")
            if body_path:
                self.system.set_anchor_image(name, body_path, "body")
                safe_print(f"[green][DONE] Body anchor set.[/]")
            else:
                safe_print(f"[red][FAIL] Failed to generate body anchor.[/]")

    def _build_prompt(self, vid: dict, type: str) -> str:
        """Constructs a prompt from the visual identity dict."""
        # Flatten the dict into a description
        desc_parts = []
        
        # Basic
        desc_parts.append(f"1girl, solo, {vid.get('age_appearance', 'young')} {vid.get('gender', 'female')}")
        
        # Face
        face = vid.get("face", {})
        desc_parts.append(f"{face.get('shape', '')} face")
        eyes = face.get("eyes", {})
        desc_parts.append(f"{eyes.get('color', '')} eyes")
        desc_parts.append(f"{vid.get('hair', {}).get('color', '')} hair, {vid.get('hair', {}).get('style', '')}")
        
        # Body
        body = vid.get("body", {})
        desc_parts.append(f"{body.get('type', '')} body")
        if body.get("chest"):
            desc_parts.append(f"{body.get('chest', {}).get('size', '')} breasts")
            
        # Skin
        desc_parts.append(f"{vid.get('skin', {}).get('tone', '')} skin")
        
        # Type specific
        if type == "face":
            desc_parts.append("close up portrait, face focus, looking at viewer, neutral expression, high quality, masterpiece, 8k")
        else:
            desc_parts.append("full body shot, standing pose, simple background, white background, neutral lighting, masterpiece, best quality")
            
        # Style
        desc_parts.append("anime style, cel shaded")
        
        return ", ".join([p for p in desc_parts if p])

    async def _generate_image(self, prompt: str, prefix: str) -> str:
        """Generates an image using ComfyUI."""
        try:
            loop = asyncio.get_event_loop()
            # Use simple txt2img
            images = await loop.run_in_executor(None, lambda: self.client.generate_txt2img_simple(
                prompt=prompt,
                negative_prompt="bad anatomy, bad hands, missing fingers, extra digits, blurry, low quality, text, watermark, 2girls, multiple views",
                width=1024 if "body" in prefix else 832,
                height=1216,
                seed=None
            ))
            
            if images:
                img_data = list(images.values())[0][0]
                
                # Save to temp (writable path for Docker)
                filename = f"{prefix}_{int(time.time())}.png"
                try:
                    from aither_adk.paths import get_saga_subdir
                    temp_dir = get_saga_subdir("output", "temp", create=True)
                except ImportError:
                    temp_dir = os.path.join(os.path.dirname(__file__), "..", "Saga", "output", "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                path = os.path.join(temp_dir, filename)
                
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                img.save(path)
                return path
                
        except Exception as e:
            safe_print(f"[red]Generation error: {e}[/]")
            
        return None
