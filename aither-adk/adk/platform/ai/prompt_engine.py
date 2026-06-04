import os
import asyncio
from typing import List, Optional, Any
from adk.platform.ui.console import safe_print


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

        return self._heuristic_prompt(history_text, active_characters)

    def _heuristic_prompt(self, history_text: str, active_characters: List[str]) -> str:
        """Fast, heuristic-based prompt construction."""
        scene_lower = history_text.lower()

        # 1. Subject Count
        char_count = len(active_characters)
        if char_count == 2:
            subject = "2girls"
        elif char_count == 3:
            subject = "3girls"
        elif char_count >= 4:
            subject = f"{char_count}girls"
        else:
            subject = "1girl"

        char_names = ", ".join(active_characters)

        # 2. Position / Action
        position = "group interaction"
        if "from behind" in scene_lower:
            position = "rear view, looking back"
        elif "lying" in scene_lower or "laying" in scene_lower:
            position = "lying down, reclining"
        elif "fight" in scene_lower or "combat" in scene_lower:
            position = "fighting, dynamic action pose, combat"
        elif "running" in scene_lower or "chase" in scene_lower:
            position = "running, dynamic motion"
        elif "sitting" in scene_lower:
            position = "sitting, seated"

        # 3. State / Appearance
        state_tags = []
        if "crying" in scene_lower or "tears" in scene_lower:
            state_tags.append("tears, crying")
        if "rain" in scene_lower:
            state_tags.append("rain, wet")
        if "wind" in scene_lower:
            state_tags.append("wind, flowing hair")

        state_str = ", ".join(state_tags) if state_tags else "detailed scene"

        # 4. Environment
        location = "high tech office, neon lights"
        lighting = "cinematic lighting"

        if self.state_manager:
            location = self.state_manager.state.get("location", location)
            lighting = self.state_manager.state.get("lighting", lighting)

        prompt = (
            f"{subject}, {char_names}, {position}, {state_str}, "
            f"multiple characters interacting, group scene, "
            f"{location}, {lighting}, "
            f"anime style, masterpiece, best quality, highly detailed, full scene view, wide shot"
        )

        return prompt

    async def _llm_prompt(self, history_text: str, active_characters: List[str], model_client: Any) -> str:
        """Uses an LLM to generate a detailed scene description."""
        system_prompt = (
            "You are an expert Stable Diffusion prompt engineer. "
            "Analyze the following conversation history and create a detailed visual description of the current scene. "
            "Focus on: Characters present, their actions/poses, attire, emotional state, setting, and lighting. "
            "Format the output as a comma-separated list of high-quality tags suitable for an anime-style image generator. "
            "Do NOT include conversational text, only visual tags. "
            "Keep it under 75 tokens."
        )

        user_prompt = f"Characters: {', '.join(active_characters)}\n\nConversation History:\n{history_text[-2000:]}"

        response_text = ""

        if hasattr(model_client, "generate_content"):
            response = await model_client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            response_text = response.text
        elif callable(model_client):
            response_text = await model_client(f"{system_prompt}\n\n{user_prompt}")
        else:
            raise ValueError("Invalid model_client provided")

        return response_text.strip()

    def generate_narrative_trigger(self, history_text: str, target_agent: str, context: str = "") -> str:
        """Generates a system trigger to guide the next agent's response."""
        base = f"[System: {target_agent}, please add your perspective."

        if context:
            base += f" Context: {context}"

        base += "]"
        return base
