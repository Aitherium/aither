"""
AitherOS Smart Task Router

Intelligently routes tasks between cloud and local backends based on:
- Content type (restricted -> local, safe -> cloud preferred)
- Resource availability (GPU VRAM, utilization)
- Task priority and queuing
- Cost optimization

This is the BRAIN of the hybrid system - ensuring deterministic,
efficient operation while maintaining flexible local capabilities.
"""

import os
import asyncio
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Awaitable
from enum import Enum, auto


class ContentRating(Enum):
    """Content rating classification"""
    SFW = auto()       # Safe for work
    SUGGESTIVE = auto() # Borderline
    NSFW = auto()      # Adult content - requires local


class TaskCategory(Enum):
    """High-level task categories"""
    TEXT_GENERATION = auto()
    IMAGE_GENERATION = auto()
    IMAGE_REFINEMENT = auto()
    VIDEO_GENERATION = auto()
    VISION_ANALYSIS = auto()
    CODE_GENERATION = auto()


@dataclass
class TaskDecision:
    """Decision made by the router"""
    use_local: bool
    provider: str
    model: str
    reason: str
    content_rating: ContentRating
    estimated_vram_mb: int = 0
    estimated_duration_sec: float = 0
    fallback_provider: Optional[str] = None


class ContentClassifier:
    """
    Classifies content to determine appropriate backend.
    
    Conservative approach: if ANY trigger is detected, route to local.
    """
    
    # Restricted content triggers - MUST use local
    RESTRICTED_TRIGGERS = [
        # Sexual
        "nude", "naked", "nsfw", "explicit", "sex", "fuck", "cock", "pussy",
        "dick", "penis", "vagina", "anal", "oral", "blowjob", "cum", "semen",
        "hentai", "porn", "erotic", "lewd", "ahegao", "orgasm", "masturbat",
        "penetrat", "intercourse", "gangbang", "threesome", "bukkake",
        "nipple", "areola", "clitoris", "testicle", "scrotum", "genitals",
        "breast", "boob", "tit", "ass", "butt", "cunt", "slutt", "whore",
        "bdsm", "bondage", "fetish", "kink", "domina", "submis", "sadis",
        "rape", "forced", "noncon", "dubcon",
        # Violence
        "gore", "guro", "dismember", "decapitat", "eviscerat", "torture",
        "blood", "wound", "murder", "kill", "death", "corpse", "dead body",
        # Controlled substances (generation)
        "drug", "cocaine", "heroin", "meth", "lsd",
    ]
    
    # Suggestive content - prefer local but cloud can handle
    SUGGESTIVE_TRIGGERS = [
        "bikini", "lingerie", "underwear", "cleavage", "seductive",
        "sensual", "intimate", "passionate", "kiss", "embrace", "caress",
        "provocative", "alluring", "tempting", "sultry", "voluptuous",
        "revealing", "tight clothing", "form-fitting", "skin-tight",
        "wet", "sweat", "glistening", "tanlines", "tan lines",
    ]
    
    @classmethod
    def classify(cls, text: str) -> ContentRating:
        """
        Classify content rating based on text analysis.
        """
        text_lower = text.lower()
        
        # Check for restricted content first
        for trigger in cls.RESTRICTED_TRIGGERS:
            if trigger in text_lower:
                return ContentRating.NSFW
        
        # Check for suggestive content
        for trigger in cls.SUGGESTIVE_TRIGGERS:
            if trigger in text_lower:
                return ContentRating.SUGGESTIVE
        
        return ContentRating.SFW
    
    @classmethod
    def is_roleplay_request(cls, text: str) -> bool:
        """Check if this is a roleplay scenario that might need local"""
        roleplay_patterns = [
            r"\*[^*]+\*",           # *action text*
            r"pretend\s+",
            r"roleplay\s+",
            r"act\s+as\s+",
            r"you\s+are\s+",
            r"in\s+character",
            r"scenario:",
            r"imagine\s+",
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in roleplay_patterns)


class SmartTaskRouter:
    """
    Main router class that makes intelligent decisions about task execution.
    """
    
    def __init__(self, resource_manager=None):
        """
        Initialize with optional resource manager for VRAM-aware routing.
        """
        self._resource_manager = resource_manager
        self._classifier = ContentClassifier()
        
        # Configuration
        self.config = RouterConfig()
    
    @property
    def resource_manager(self):
        """Lazy load resource manager"""
        if self._resource_manager is None:
            try:
                from AitherOS.agents.common.resource_manager import resource_manager
                self._resource_manager = resource_manager
            except ImportError:
                pass
        return self._resource_manager
    
    def route_text_generation(
        self,
        prompt: str,
        system_instruction: str = None,
        force_local: bool = False,
        force_cloud: bool = False
    ) -> TaskDecision:
        """
        Route a text generation task to the appropriate backend.
        """
        # Classify content
        combined_text = f"{system_instruction or ''} {prompt}"
        rating = self._classifier.classify(combined_text)
        is_roleplay = self._classifier.is_roleplay_request(prompt)
        
        # Restricted content MUST use local
        if rating == ContentRating.NSFW or force_local:
            return TaskDecision(
                use_local=True,
                provider="ollama",
                model=self.config.local_llm_model,
                reason="NSFW content or explicit local request",
                content_rating=rating,
                estimated_vram_mb=6000,
                estimated_duration_sec=15,
                fallback_provider=None  # No fallback for restricted content
            )
        
        # Force cloud
        if force_cloud:
            return TaskDecision(
                use_local=False,
                provider="google",
                model=self.config.cloud_llm_model,
                reason="Explicit cloud request",
                content_rating=rating,
                fallback_provider="ollama"
            )
        
        # Check resource availability
        if self.resource_manager:
            status = self.resource_manager.get_status()
            
            # If GPU is busy or low on VRAM, prefer cloud
            if status["llm_busy"] or status["vram_free_mb"] < 4000:
                return TaskDecision(
                    use_local=False,
                    provider="google",
                    model=self.config.cloud_llm_model,
                    reason="Local LLM resources busy/low VRAM",
                    content_rating=rating,
                    fallback_provider="ollama"
                )
        
        # Suggestive content or roleplay - prefer local but cloud is fallback
        if rating == ContentRating.SUGGESTIVE or is_roleplay:
            return TaskDecision(
                use_local=True,
                provider="ollama",
                model=self.config.local_llm_model,
                reason="Suggestive content or roleplay - prefer uncensored local",
                content_rating=rating,
                estimated_vram_mb=6000,
                estimated_duration_sec=15,
                fallback_provider="google"
            )
        
        # SFW content - use cloud for speed/quality
        return TaskDecision(
            use_local=False,
            provider="google",
            model=self.config.cloud_llm_model,
            reason="SFW content - cloud preferred for quality",
            content_rating=rating,
            fallback_provider="ollama"
        )
    
    def route_image_generation(
        self,
        prompt: str,
        force_local: bool = False,
        force_cloud: bool = False,
        model_preference: str = None
    ) -> TaskDecision:
        """
        Route an image generation task to the appropriate backend.
        """
        rating = self._classifier.classify(prompt)
        
        # Restricted content MUST use local
        if rating == ContentRating.NSFW or force_local:
            # Choose model based on content
            is_anime = any(t in prompt.lower() for t in ["anime", "manga", "hentai", "cartoon"])
            model = "pony" if is_anime or rating == ContentRating.NSFW else "flux"
            
            return TaskDecision(
                use_local=True,
                provider="comfyui",
                model=model,
                reason="NSFW content requires local generation",
                content_rating=rating,
                estimated_vram_mb=10000 if model == "pony" else 12000,
                estimated_duration_sec=30,
                fallback_provider=None  # No fallback for restricted content images
            )
        
        # Force cloud
        if force_cloud:
            return TaskDecision(
                use_local=False,
                provider="fal",
                model=model_preference or "fal-ai/flux/schnell",
                reason="Explicit cloud request",
                content_rating=rating,
                fallback_provider="comfyui"
            )
        
        # Check resource availability
        if self.resource_manager:
            status = self.resource_manager.get_status()
            
            # If diffusion is busy or very low VRAM, use cloud
            if status["diffusion_busy"] or status["vram_free_mb"] < 8000:
                return TaskDecision(
                    use_local=False,
                    provider="fal",
                    model="fal-ai/flux/schnell",
                    reason="Local diffusion resources busy/low VRAM",
                    content_rating=rating,
                    fallback_provider="comfyui"
                )
        
        # Suggestive - prefer local for reliability
        if rating == ContentRating.SUGGESTIVE:
            return TaskDecision(
                use_local=True,
                provider="comfyui",
                model="flux",
                reason="Suggestive content - local preferred for reliability",
                content_rating=rating,
                estimated_vram_mb=12000,
                estimated_duration_sec=30,
                fallback_provider="fal"
            )
        
        # SFW - use cloud for speed
        return TaskDecision(
            use_local=False,
            provider="fal",
            model=model_preference or "fal-ai/flux/schnell",
            reason="SFW content - cloud for speed",
            content_rating=rating,
            fallback_provider="comfyui"
        )
    
    def route_image_refinement(
        self,
        prompt: str,
        source_image: str,
        force_local: bool = False
    ) -> TaskDecision:
        """
        Route an image refinement/editing task.
        
        Refinement typically needs more control - prefer local.
        """
        rating = self._classifier.classify(prompt)
        
        # Restricted or suggestive content - local only
        if rating in [ContentRating.NSFW, ContentRating.SUGGESTIVE] or force_local:
            return TaskDecision(
                use_local=True,
                provider="comfyui",
                model="pony",  # Pony is better for img2img
                reason="Refinement with adult content - local required",
                content_rating=rating,
                estimated_vram_mb=10000,
                estimated_duration_sec=25,
                fallback_provider=None
            )
        
        # SFW refinement - still prefer local for quality control
        if self.resource_manager:
            status = self.resource_manager.get_status()
            if not status["diffusion_busy"] and status["vram_free_mb"] >= 8000:
                return TaskDecision(
                    use_local=True,
                    provider="comfyui",
                    model="flux",
                    reason="Local resources available for quality refinement",
                    content_rating=rating,
                    estimated_vram_mb=12000,
                    estimated_duration_sec=25,
                    fallback_provider="fal"
                )
        
        # Fallback to cloud
        return TaskDecision(
            use_local=False,
            provider="fal",
            model="fal-ai/flux/dev",
            reason="Local resources busy - using cloud refinement",
            content_rating=rating,
            fallback_provider="comfyui"
        )
    
    async def execute_with_fallback(
        self,
        decision: TaskDecision,
        local_func: Callable[..., Awaitable[Any]],
        cloud_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a task with automatic fallback.
        
        Tries the primary backend, falls back to secondary on failure.
        """
        primary_func = local_func if decision.use_local else cloud_func
        fallback_func = cloud_func if decision.use_local else local_func
        
        try:
            if decision.use_local and self.resource_manager:
                # Acquire GPU resources for local
                from AitherOS.agents.common.resource_manager import TaskType
                task_type = TaskType.DIFFUSION_SDXL  # Default, could be more specific
                
                async with self.resource_manager.acquire_gpu(task_type, timeout=60):
                    return await primary_func(*args, **kwargs)
            else:
                return await primary_func(*args, **kwargs)
                
        except Exception as e:
            print(f"[TaskRouter] Primary ({decision.provider}) failed: {e}")
            
            if decision.fallback_provider and fallback_func:
                print(f"[TaskRouter] Trying fallback: {decision.fallback_provider}")
                try:
                    return await fallback_func(*args, **kwargs)
                except Exception as e2:
                    print(f"[TaskRouter] Fallback also failed: {e2}")
                    raise
            else:
                raise


class RouterConfig:
    """Configuration for the task router"""
    
    def __init__(self):
        # Model preferences
        self.cloud_llm_model = os.getenv("CLOUD_LLM_MODEL", "gemini-2.5-flash")
        self.local_llm_model = os.getenv("LOCAL_LLM_MODEL", "aither-orchestrator-8b-v4")
        self.cloud_image_model = os.getenv("CLOUD_IMAGE_MODEL", "fal-ai/flux/schnell")
        self.local_image_model_anime = os.getenv("LOCAL_IMAGE_MODEL_ANIME", "pony")
        self.local_image_model_realistic = os.getenv("LOCAL_IMAGE_MODEL_REALISTIC", "flux")
        
        # Thresholds
        self.min_free_vram_for_llm = int(os.getenv("MIN_VRAM_LLM", "4000"))
        self.min_free_vram_for_diffusion = int(os.getenv("MIN_VRAM_DIFFUSION", "8000"))
        
        # Behavior
        self.always_local_nsfw = True  # Never send restricted content to cloud
        self.prefer_local_suggestive = True  # Prefer local for borderline content


# Singleton instance
_router: Optional[SmartTaskRouter] = None


def get_router() -> SmartTaskRouter:
    """Get the singleton router instance"""
    global _router
    if _router is None:
        _router = SmartTaskRouter()
    return _router


# Convenience functions
def route_text(prompt: str, **kwargs) -> TaskDecision:
    """Quick text routing"""
    return get_router().route_text_generation(prompt, **kwargs)


def route_image(prompt: str, **kwargs) -> TaskDecision:
    """Quick image routing"""
    return get_router().route_image_generation(prompt, **kwargs)


def route_refine(prompt: str, source: str, **kwargs) -> TaskDecision:
    """Quick refinement routing"""
    return get_router().route_image_refinement(prompt, source, **kwargs)

