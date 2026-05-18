import os
import asyncio
from typing import List, Optional, Any
from aither_adk.ui.console import safe_print

class ScenePrompter:
    """
    Handles the construction of prompts for image generation and narrative direction.
    Supports both heuristic (fast) and LLM-based (smart) generation.
    """

    def __init__(self, state_manager=None):
        self.state_manager = state_manager

    async def generate_scene_prompt(self,
                                  history_text: str,
                                  active_characters: List[str],
                                  mode: str = "heuristic",
                                  model_client: Any = None) -> str:
        """
        Generates a prompt for the image generator based on the scene context.

        Args:
            history_text: The recent conversation history as text.
            active_characters: List of character names currently in the scene.
            mode: "heuristic" (fast, regex) or "llm" (smart, uses model).
            model_client: Optional client/function to call for LLM generation.

        Returns:
            A comma-separated string of tags/description for the image generator.
        """
        if mode == "llm" and model_client:
            try:
                return await self._llm_prompt(history_text, active_characters, model_client)
            except Exception as e:
                safe_print(f"[yellow]LLM prompt generation failed: {e}. Falling back to heuristic.[/]")
                # Fallback

        return self._heuristic_prompt(history_text, active_characters)

    def _heuristic_prompt(self, history_text: str, active_characters: List[str]) -> str:
        """Fast, regex-based prompt construction."""
        scene_lower = history_text.lower()

        # 1. Subject Count
        char_count = len(active_characters)
        if char_count == 2:
            subject = "2girls"
        elif char_count == 3:
            subject = "3girls"
        elif char_count == 4:
            subject = "4girls"
        elif char_count >= 5:
            subject = f"{char_count}girls"
        else:
            subject = "1girl"

        char_names = ", ".join(active_characters)

        # 2. Position / Action
        position = "group interaction"
        if "from behind" in scene_lower:
            position = "rear view, looking back"
        elif "riding" in scene_lower:
            position = "dynamic pose, action"
        elif "lying" in scene_lower or "laying" in scene_lower:
            position = "lying down, reclining"
        elif "fight" in scene_lower or "combat" in scene_lower:
            position = "fighting, dynamic action pose, combat"
        elif "pinned" in scene_lower or "against wall" in scene_lower:
            position = "pinned against wall, standing"

        # 3. State / Appearance
        state_tags = []
        if "nude" in scene_lower or "naked" in scene_lower:
            state_tags.append("nude, artistic")
        if "messy" in scene_lower:
            state_tags.append("disheveled, messy appearance")
        if "sweat" in scene_lower:
            state_tags.append("sweat, glistening skin")
        if "tears" in scene_lower or "crying" in scene_lower:
            state_tags.append("tears, crying")

        state_str = ", ".join(state_tags) if state_tags else "intense scene"

        # 4. Environment (from StateManager if available)
        location = "high tech office, neon lights"
        lighting = "cinematic lighting"

        if self.state_manager:
            location = self.state_manager.state.get("location", location)
            lighting = self.state_manager.state.get("lighting", lighting)

        # Construct Prompt
        prompt = f"{subject}, {char_names}, {position}, {state_str}, multiple characters interacting, group scene, {location}, {lighting}, anime style, masterpiece, best quality, highly detailed, full scene view, wide shot"

        return prompt

    async def _llm_prompt(self, history_text: str, active_characters: List[str], model_client: Any) -> str:
        """Uses an LLM to generate a detailed scene description."""

        system_prompt = (
            "You are an expert Stable Diffusion prompt engineer. "
            "Analyze the following conversation history and create a detailed visual description of the current scene. "
            "Focus on: Characters present, their actions/poses, clothing (or lack thereof), emotional state, setting, and lighting. "
            "Format the output as a comma-separated list of high-quality tags suitable for an anime-style image generator. "
            "Do NOT include conversational text, only visual tags. "
            "Keep it under 75 tokens."
        )

        user_prompt = f"Characters: {', '.join(active_characters)}\n\nConversation History:\n{history_text[-2000:]}"

        # Assuming model_client is a callable that takes (system, user) or just a prompt
        # We'll try to adapt based on what it is

        response_text = ""

        if hasattr(model_client, "generate_content"):
            # Google GenAI Client
            response = await model_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            response_text = response.text
        elif callable(model_client):
            # Simple function
            response_text = await model_client(f"{system_prompt}\n\n{user_prompt}")
        else:
            raise ValueError("Invalid model_client provided")

        return response_text.strip()

    def generate_narrative_trigger(self, history_text: str, target_agent: str, context: str = "") -> str:
        """
        Generates a system trigger to guide the next agent's response.
        Currently heuristic, but could be LLM-based.
        """
        base = f"[System: {target_agent}, please add your perspective."

        if context:
            base += f" Context: {context}"

        base += "]"
        return base
