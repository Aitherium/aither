"""
AitherOS Anchor Generator

Automates the generation of reference (anchor) images for all personas.
Generates: Headshot, Upper Body, Full Body, Behind View.
Ensures visual consistency using the Refinement Engine.
"""

import os
import yaml
import asyncio
from typing import List, Dict
from aither_adk.ui.console import safe_print
from aither_adk.ai.llm_prompt_generator import generate_sd_prompt, generate_visual_identity
from aither_adk.ai.refinement_engine import RefinementEngine

class AnchorGenerator:
    def __init__(self):
        self.refinement_engine = RefinementEngine()
        try:
            from aither_adk.paths import get_saga_config_dir, get_saga_subdir
            self.personas_dir = get_saga_config_dir("personas")
            self.anchors_dir = get_saga_subdir("memory", "anchors")
        except ImportError:
            self.personas_dir = os.path.join(os.path.dirname(__file__), "..", "Saga", "config", "personas")
            self.anchors_dir = os.path.join(os.path.dirname(__file__), "..", "Saga", "memory", "anchors")
        
    def get_all_personas(self) -> List[str]:
        """List all available persona names based on config files."""
        if not os.path.exists(self.personas_dir):
            return []
        
        personas = []
        for f in os.listdir(self.personas_dir):
            if f.endswith(".yaml") and not f.startswith("_"):
                personas.append(f.replace(".yaml", ""))
        return personas

    def ensure_persona_spec(self, persona_name: str):
        """Ensures the persona YAML has a complete visual_identity."""
        yaml_path = os.path.join(self.personas_dir, f"{persona_name}.yaml")
        if not os.path.exists(yaml_path):
            return

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            # Check if visual_identity is missing or sparse
            vid = data.get("visual_identity", {})
            if not vid or "face" not in vid or "body" not in vid:
                safe_print(f"[yellow][WARN] Incomplete visual identity for {persona_name}. Generating...[/]")
                
                desc = data.get("description", "")
                # Also check safety prompts for clues
                prompts = data.get("safety_prompts", {})
                desc += "\n" + prompts.get("professional", "")
                
                new_vid = generate_visual_identity(persona_name, desc)
                
                if new_vid:
                    data["visual_identity"] = new_vid
                    # Save back
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        yaml.dump(data, f, sort_keys=False)
                    safe_print(f"[green][DONE] Updated visual identity for {persona_name}[/]")
                else:
                    safe_print(f"[red][FAIL] Failed to generate visual identity for {persona_name}[/]")
                    
        except Exception as e:
            safe_print(f"[red]Error updating persona spec: {e}[/]")

    async def generate_anchors_for_persona(self, persona_name: str, force: bool = False):
        """Generates the 4 standard anchor images for a specific persona."""
        safe_print(f"\n[bold cyan][LAUNCH] Generating anchors for: {persona_name.title()}[/]")
        
        # 1. Ensure Spec
        self.ensure_persona_spec(persona_name)
        
        # Ensure output directory
        persona_anchor_dir = os.path.join(self.anchors_dir, persona_name)
        os.makedirs(persona_anchor_dir, exist_ok=True)
        
        views = {
            "headshot": "Headshot portrait, looking at viewer, neutral expression, high detail face",
            "upper_body": "Upper body portrait, waist up, looking at viewer, standing, hands by sides, holding nothing",
            "full_body": "Full body shot, standing, looking at viewer, showing entire outfit, head to toe",
            "behind": "View from behind, walking away, full body from back, detailed hair, detailed outfit"
        }
        
        for view_name, view_prompt in views.items():
            output_path = os.path.join(persona_anchor_dir, f"{view_name}.png")
            
            if os.path.exists(output_path) and not force:
                safe_print(f"[dim]Skipping {view_name} (already exists): {output_path}[/]")
                continue
                
            safe_print(f"[yellow][PHOTO] Generating {view_name}...[/]")
            
            # Use RefinementEngine for high fidelity
            # We pass the view description as the user request
            # The prompt generator will combine this with the persona's visual identity
            
            # We use a lower max_iterations for speed, or higher for quality. 
            # Since these are anchors, quality is paramount.
            image_path = await self.refinement_engine.generate_with_feedback(
                user_request=view_prompt,
                persona_name=persona_name,
                max_iterations=2 
            )
            
            if image_path:
                # Move/Rename to standard location
                import shutil
                shutil.move(image_path, output_path)
                safe_print(f"[green][DONE] Saved {view_name} to {output_path}[/]")
            else:
                safe_print(f"[red][FAIL] Failed to generate {view_name}[/]")

    async def generate_all_anchors(self, force: bool = False):
        """Generates anchors for ALL personas."""
        personas = self.get_all_personas()
        safe_print(f"[bold]Found {len(personas)} personas: {', '.join(personas)}[/]")
        
        for p in personas:
            await self.generate_anchors_for_persona(p, force=force)
            
        safe_print("\n[bold green]* All anchor generations complete![/]")

