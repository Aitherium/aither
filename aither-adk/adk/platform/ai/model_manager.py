"""
AitherOS Dynamic Model Manager

Intelligently loads and unloads models based on:
- Current task requirements
- Available VRAM
- Model warm-up time vs. VRAM cost tradeoff

Key Principle: Don't keep heavy models loaded when not needed.
- Vision model? Unload after use
- NSFW image? Load Pony, generate, consider unloading
- Simple chat? Use cloud, don't touch local VRAM
- Car photo? Use Imagen/cloud, skip local entirely
"""

import os
import sys
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum, auto
import time
from aither_adk.ai.model_browser import ModelBrowser, ModelSearchResult


# Get URLs from services.yaml (SINGLE SOURCE OF TRUTH)
def _get_ollama_url():
    try:
        _adk_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        _lib_path = os.path.join(_adk_root, "..", "lib", "core")
        if os.path.exists(_lib_path) and _lib_path not in sys.path:
            sys.path.insert(0, _lib_path)
        from AitherPorts import ollama_url
        return ollama_url()
    except ImportError:
        # Fallback: try direct import
        try:
            from lib.core.AitherPorts import ollama_url
            return ollama_url()
        except ImportError:
            raise ImportError("Cannot import AitherPorts. Ensure services.yaml is available.")

OLLAMA_BASE_URL = _get_ollama_url()
COMFY_MODELS_PATH = os.getenv("COMFY_MODELS_PATH", os.path.join(os.getcwd(), "ComfyUI", "models", "checkpoints"))


class ModelTier(Enum):
    """Model tiers based on quality/speed tradeoff"""
    INSTANT = auto()    # Cloud APIs - no local resources
    LIGHT = auto()      # Small models, quick to load
    STANDARD = auto()   # Regular models
    HEAVY = auto()      # Large models, slow to load, high VRAM


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    provider: str  # ollama, comfyui, cloud
    tier: ModelTier
    vram_mb: int
    load_time_sec: float
    capabilities: Set[str]  # chat, vision, image_gen, nsfw, etc.


# Model registry with VRAM estimates and capabilities
# 
# RTX 5090 (32GB VRAM) + 128GB DDR5 RAM Optimization:
# - vLLM-Swap serves deepseek-r1:14b (~14GB) for reasoning
# - Keep orchestrator-8b always loaded (~5GB) for routing
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # ==========================================================================
    # REAL OLLAMA MODELS - Use these for local inference
    # vLLM serves deepseek-r1:14b; Ollama uses these real models:
    # ==========================================================================
    "deepseek-r1:14b": ModelInfo(
        name="deepseek-r1:14b",
        provider="ollama",
        tier=ModelTier.HEAVY,
        vram_mb=9000,
        load_time_sec=8,
        capabilities={"chat", "reasoning", "code", "analysis"}
    ),
    "mistral-nemo:latest": ModelInfo(
        name="mistral-nemo:latest",
        provider="ollama",
        tier=ModelTier.STANDARD,
        vram_mb=7000,
        load_time_sec=6,
        capabilities={"chat", "reasoning", "code", "balanced"}
    ),
    "llama3.2:latest": ModelInfo(
        name="llama3.2:latest",
        provider="ollama",
        tier=ModelTier.LIGHT,
        vram_mb=4000,
        load_time_sec=4,
        capabilities={"chat", "fast", "reflex", "neurons"}
    ),
    # NVIDIA Orchestrator - Keep hot for routing decisions
    "nvidia-orchestrator": ModelInfo(
        name="orchestrator-8b",
        provider="ollama",
        tier=ModelTier.STANDARD,
        vram_mb=5000,
        load_time_sec=5,
        capabilities={"chat", "routing", "tool_calling", "orchestration"}
    ),
    # Ollama LLMs
    "mistral-nemo": ModelInfo(
        name="mistral-nemo",
        provider="ollama",
        tier=ModelTier.STANDARD,
        vram_mb=5500,
        load_time_sec=8,
        capabilities={"chat", "roleplay", "nsfw", "uncensored"}
    ),
    "llama3.2-vision": ModelInfo(
        name="llama3.2-vision",
        provider="ollama",
        tier=ModelTier.HEAVY,
        vram_mb=7000,
        load_time_sec=12,
        capabilities={"vision", "image_analysis", "chat"}
    ),
    "llama3.2": ModelInfo(
        name="llama3.2",
        provider="ollama",
        tier=ModelTier.LIGHT,
        vram_mb=3500,
        load_time_sec=5,
        capabilities={"chat", "code"}
    ),
    "phi3": ModelInfo(
        name="phi3",
        provider="ollama",
        tier=ModelTier.LIGHT,
        vram_mb=2500,
        load_time_sec=4,
        capabilities={"chat", "code", "fast"}
    ),
    
    # ComfyUI Diffusion Models
    "pony": ModelInfo(
        name="pony",  # Pony Diffusion / Unholy Desire
        provider="comfyui",
        tier=ModelTier.HEAVY,
        vram_mb=10000,
        load_time_sec=15,
        capabilities={"image_gen", "anime", "nsfw", "uncensored"}
    ),
    "flux": ModelInfo(
        name="flux",  # Flux.1 Dev
        provider="comfyui",
        tier=ModelTier.HEAVY,
        vram_mb=12000,
        load_time_sec=20,
        capabilities={"image_gen", "photorealistic", "high_quality"}
    ),
    "sdxl": ModelInfo(
        name="sdxl",
        provider="comfyui",
        tier=ModelTier.STANDARD,
        vram_mb=8000,
        load_time_sec=10,
        capabilities={"image_gen", "versatile"}
    ),
    
    # Cloud Models (no local VRAM)
    "gemini-2.5-flash": ModelInfo(
        name="gemini-2.5-flash",
        provider="cloud",
        tier=ModelTier.INSTANT,
        vram_mb=0,
        load_time_sec=0,
        capabilities={"chat", "code", "fast", "vision"}
    ),
    "imagen-4": ModelInfo(
        name="imagen-4",
        provider="cloud",
        tier=ModelTier.INSTANT,
        vram_mb=0,
        load_time_sec=0,
        capabilities={"image_gen", "fast", "sfw_only"}
    ),
    "fal-flux-schnell": ModelInfo(
        name="fal-ai/flux/schnell",
        provider="cloud",
        tier=ModelTier.INSTANT,
        vram_mb=0,
        load_time_sec=0,
        capabilities={"image_gen", "fast", "sfw_only"}
    ),
    "nano-banana": ModelInfo(
        name="fal-ai/nano-banana-pro",
        provider="cloud",
        tier=ModelTier.INSTANT,
        vram_mb=0,
        load_time_sec=0,
        capabilities={"image_gen", "ultra_fast", "sfw_only"}
    ),
}


class OllamaModelManager:
    """
    Manages Ollama model loading/unloading.
    
    Key insight: Ollama's `keep_alive` parameter controls how long a model
    stays in memory after a request. Setting it to 0 unloads immediately.
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._loaded_models: Set[str] = set()
        self._last_check = 0
    
    async def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models (cached for 5s)"""
        now = time.time()
        if now - self._last_check < 5:
            return list(self._loaded_models)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/ps") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._loaded_models = {m["name"] for m in data.get("models", [])}
                        self._last_check = now
        except Exception as e:
            print(f"[ModelManager] Error checking loaded models: {e}")
        
        return list(self._loaded_models)
    
    async def is_loaded(self, model: str) -> bool:
        """Check if a specific model is currently loaded"""
        loaded = await self.get_loaded_models()
        # Handle both "mistral-nemo" and "mistral-nemo:latest"
        return any(model in m or m in model for m in loaded)
    
    async def preload(self, model: str, keep_alive: str = "5m") -> bool:
        """
        Preload a model into VRAM.
        
        Args:
            model: Model name
            keep_alive: How long to keep loaded (e.g. "5m", "1h", "0" for immediate unload)
        """
        print(f"[ModelManager] Preloading {model}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": keep_alive
                    }
                ) as resp:
                    if resp.status == 200:
                        self._loaded_models.add(model)
                        print(f"[ModelManager] [DONE] {model} loaded (keep_alive: {keep_alive})")
                        return True
        except Exception as e:
            print(f"[ModelManager] [FAIL] Failed to load {model}: {e}")
        return False
    
    async def unload(self, model: str) -> bool:
        """
        Immediately unload a model from VRAM.
        """
        print(f"[ModelManager] Unloading {model}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": "0"  # 0 = unload immediately
                    }
                ) as resp:
                    if resp.status == 200:
                        self._loaded_models.discard(model)
                        print(f"[ModelManager] [DONE] {model} unloaded")
                        return True
        except Exception as e:
            print(f"[ModelManager] [FAIL] Failed to unload {model}: {e}")
        return False
    
    async def unload_all(self) -> int:
        """Unload ALL models from Ollama"""
        loaded = await self.get_loaded_models()
        count = 0
        for model in loaded:
            if await self.unload(model):
                count += 1
        return count
    
    async def use_model_temporarily(self, model: str, keep_alive: str = "30s"):
        """
        Context manager for temporary model use.
        
        Usage:
            async with manager.use_model_temporarily("llama3.2-vision"):
                result = await analyze_image(...)
            # Model unloads automatically after 30s of inactivity
        """
        await self.preload(model, keep_alive=keep_alive)
    
    def get_vram_estimate(self) -> int:
        """Estimate VRAM currently used by loaded Ollama models"""
        total = 0
        for model in self._loaded_models:
            # Find in registry
            base_name = model.split(":")[0]
            if base_name in MODEL_REGISTRY:
                total += MODEL_REGISTRY[base_name].vram_mb
            else:
                total += 4000  # Default estimate
        return total


class DynamicModelManager:
    """
    High-level model management with smart loading decisions.
    
    Philosophy:
    - Cloud first for SFW content (instant, no VRAM)
    - Local only when needed (NSFW, specific capabilities)
    - Unload after use, don't hoard VRAM
    - Consider load time in routing decisions
    """

    def __init__(self):
        self.ollama = OllamaModelManager()
        self._session: Optional[aiohttp.ClientSession] = None
        self.browser = ModelBrowser()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def find_and_download_model(self, query: str, source: str = "civitai", nsfw: bool = False) -> Optional[ModelInfo]:
        """
        Search for a model and download it if found.
        """
        print(f"[ModelManager] Searching for '{query}' on {source}...")
        if source == "civitai":
            results = self.browser.search_civitai(query, limit=1, nsfw=nsfw)
        else:
            results = self.browser.search_huggingface(query, limit=1)
            
        if not results:
            print(f"[ModelManager] No models found for '{query}'")
            return None
            
        result = results[0]
        print(f"[ModelManager] Found: {result.name} ({result.filesize_mb:.1f} MB)")
        
        # Check if we have space/resources
        # For now, just check if file exists
        filename = f"{result.name.replace(' ', '_')}.safetensors"
        save_path = os.path.join(COMFY_MODELS_PATH, filename)
        
        if os.path.exists(save_path):
            print(f"[ModelManager] Model already exists at {save_path}")
        else:
            print(f"[ModelManager] Downloading to {save_path}...")
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            success = self.browser.download_model(
                result.download_url, 
                save_path, 
                progress_callback=lambda dl, total: print(f"\rDownloading: {dl/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="")
            )
            print() # Newline
            
            if not success:
                print("[ModelManager] Download failed")
                return None
                
        # Register the new model
        model_info = ModelInfo(
            name=result.name,
            provider="comfyui",
            tier=ModelTier.HEAVY if (result.filesize_mb and result.filesize_mb > 4000) else ModelTier.STANDARD,
            vram_mb=int(result.filesize_mb * 1.2) if result.filesize_mb else 8000, # Rough estimate
            load_time_sec=15,
            capabilities={"image_gen", "downloaded"}
        )
        
        if result.nsfw:
            model_info.capabilities.add("nsfw")
            model_info.capabilities.add("uncensored")
            
        # Add tags as capabilities
        for tag in result.tags:
            model_info.capabilities.add(tag.lower())
            
        MODEL_REGISTRY[result.name] = model_info
        return model_info

    async def recommend_model(self, prompt: str, hardware_constraints: bool = True) -> ModelInfo:
        """
        Recommend a model based on prompt and hardware.
        """
        # 1. Analyze prompt for keywords
        is_anime = "anime" in prompt.lower()
        is_photo = "photo" in prompt.lower() or "realistic" in prompt.lower()
        is_nsfw = "nsfw" in prompt.lower() or "naked" in prompt.lower() # Basic check
        
        # 2. Check hardware
        available_vram = 16000 # Default assumption
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager
            status = resource_manager.get_status()
            available_vram = status["vram_free_mb"]
        except ImportError:
            pass
            
        print(f"[ModelManager] Recommending model for: '{prompt[:30]}...' (VRAM: {available_vram}MB)")
        
        # 3. Filter registry
        candidates = []
        for name, info in MODEL_REGISTRY.items():
            if info.provider == "cloud":
                candidates.append((info, 100)) # Cloud is always a good fallback
                continue
                
            if hardware_constraints and info.vram_mb > available_vram:
                continue
                
            score = 0
            if is_anime and "anime" in info.capabilities:
                score += 50
            if is_photo and ("photorealistic" in info.capabilities or "realism" in info.capabilities):
                score += 50
            if is_nsfw and ("nsfw" in info.capabilities or "uncensored" in info.capabilities):
                score += 100
            elif is_nsfw and "sfw_only" in info.capabilities:
                score = -1000 # Disqualify
                
            candidates.append((info, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_model = candidates[0][0]
            print(f"[ModelManager] Recommended: {best_model.name}")
            return best_model
            
        # Fallback
        return MODEL_REGISTRY.get("sdxl", MODEL_REGISTRY["pony"])

    async def get_status(self) -> Dict:
            await self._session.close()
    
    async def get_status(self) -> Dict:
        """Get current model loading status"""
        loaded_ollama = await self.ollama.get_loaded_models()
        vram_estimate = self.ollama.get_vram_estimate()
        
        return {
            "ollama_loaded": loaded_ollama,
            "estimated_vram_mb": vram_estimate,
            "ollama_available": len(loaded_ollama) > 0,
        }
    
    async def select_model_for_task(
        self,
        task: str,
        is_nsfw: bool = False,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        max_load_time_sec: float = 30,
        available_vram_mb: int = None
    ) -> ModelInfo:
        """
        Intelligently select the best model for a task.
        
        Args:
            task: "chat", "image_gen", "vision", "code"
            is_nsfw: Whether content is NSFW (forces local)
            prefer_speed: Prioritize faster models
            prefer_quality: Prioritize quality over speed
            max_load_time_sec: Maximum acceptable load time
            available_vram_mb: Current free VRAM
        """
        candidates = []
        
        for name, info in MODEL_REGISTRY.items():
            # Must have the capability
            if task not in info.capabilities:
                continue
            
            # NSFW requires uncensored models
            if is_nsfw and "nsfw" not in info.capabilities and "uncensored" not in info.capabilities:
                continue
            
            # Check VRAM constraint
            if available_vram_mb is not None and info.vram_mb > available_vram_mb:
                continue
            
            # Check load time constraint
            if info.load_time_sec > max_load_time_sec:
                continue
            
            candidates.append(info)
        
        if not candidates:
            # Fallback to cloud
            if task == "image_gen":
                return MODEL_REGISTRY.get("nano-banana", MODEL_REGISTRY["fal-flux-schnell"])
            return MODEL_REGISTRY["gemini-2.5-flash"]
        
        # Sort by preference
        def score(m: ModelInfo) -> float:
            s = 0
            if prefer_speed:
                s -= m.load_time_sec * 10
                if m.tier == ModelTier.INSTANT:
                    s += 100
            if prefer_quality:
                if m.tier == ModelTier.HEAVY:
                    s += 50
            # Prefer cloud to save VRAM
            if m.provider == "cloud":
                s += 20
            return s
        
        candidates.sort(key=score, reverse=True)
        return candidates[0]
    
    async def prepare_for_task(
        self,
        task: str,
        is_nsfw: bool = False,
        required_model: str = None
    ) -> ModelInfo:
        """
        Prepare the system for a task by loading required models
        and potentially unloading unnecessary ones.
        """
        # Get current state
        status = await self.get_status()
        
        # Get VRAM info from resource manager if available
        available_vram = 8000  # Default estimate
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager
            rs = resource_manager.get_status()
            available_vram = rs["vram_free_mb"]
        except ImportError:
            pass
        
        # If specific model required, use it
        if required_model and required_model in MODEL_REGISTRY:
            model = MODEL_REGISTRY[required_model]
        else:
            # Select best model
            model = await self.select_model_for_task(
                task=task,
                is_nsfw=is_nsfw,
                available_vram_mb=available_vram
            )
        
        # If it's an Ollama model and not loaded, load it
        if model.provider == "ollama":
            if not await self.ollama.is_loaded(model.name):
                # Maybe unload other models first to free VRAM
                if model.vram_mb > available_vram:
                    print(f"[ModelManager] Need {model.vram_mb}MB but only {available_vram}MB free. Unloading others...")
                    await self.ollama.unload_all()
                    await asyncio.sleep(2)  # Wait for VRAM to free
                
                # Load with short keep_alive for automatic cleanup
                await self.ollama.preload(model.name, keep_alive="2m")
        
        return model
    
    async def cleanup_after_task(self, model: ModelInfo, aggressive: bool = False):
        """
        Cleanup after a task completes.
        
        Args:
            model: The model that was used
            aggressive: If True, unload immediately. If False, let keep_alive handle it.
        """
        if aggressive and model.provider == "ollama":
            await self.ollama.unload(model.name)
    
    async def ensure_vram_available(self, required_mb: int) -> bool:
        """
        Ensure at least `required_mb` VRAM is available.
        Will unload models if necessary.
        """
        try:
            from aither_adk.infrastructure.resource_manager import resource_manager
            status = resource_manager.get_status()
            free = status["vram_free_mb"]
            
            if free >= required_mb:
                return True
            
            # Need to free up space
            print(f"[ModelManager] Need {required_mb}MB, only {free}MB free. Freeing up...")
            await self.ollama.unload_all()
            await asyncio.sleep(3)  # Wait for VRAM to be released
            
            # Check again
            status = resource_manager.get_status()
            return status["vram_free_mb"] >= required_mb
            
        except ImportError:
            # No resource manager, assume OK
            return True


# Singleton instance
_manager: Optional[DynamicModelManager] = None


def get_model_manager() -> DynamicModelManager:
    """Get the singleton model manager"""
    global _manager
    if _manager is None:
        _manager = DynamicModelManager()
    return _manager


# =============================================================================
# ELASTIC MODEL STRATEGY - Optimized for RTX 5090 (32GB) + 128GB DDR5
# =============================================================================

class ElasticModelStrategy:
    """
    Intelligent hot-swapping strategy for GPU models.
    
    vLLM-Swap (port 8176) serves deepseek-r1:14b (~14GB VRAM).
    aither-orchestrator (port 8120) serves the orchestrator model.
    
    VRAM BUDGET (RTX 5090 32GB):
    ===========================
    ALWAYS HOT:
    - orchestrator-8b: ~5GB  (routing decisions - never unload)
    - deepseek-r1:14b: ~14GB (reasoning + writing via vLLM-Swap)
    
    HOT-SWAP POOL (remaining):
    - qwen3-coder-80b: exclusive mode for coding tasks
    - nemotron-3-nano-30b: for large-context tasks
    
    DDR5 OFFLOAD (128GB):
    - For context caching when VRAM is full
    - KV cache can spill to system RAM with minimal latency impact
    """
    
    # Models to keep always loaded (default Ollama set ~17.6GB)
    ALWAYS_HOT = [
        "aither-orchestrator-8b-v5",  # ~5GB  - routing/tool calling
        "mistral-nemo:latest",        # ~5GB  - general chat/balanced
        "nomic-embed-text:latest",    # ~0.6GB - embeddings (always needed)
        "llama3.2-vision:latest",     # ~7GB  - vision/OCR
    ]
    
    # VRAM estimates (MB)
    VRAM_MAP = {
        "aither-orchestrator-8b-v5": 5000,
        "mistral-nemo:latest": 5000,
        "nomic-embed-text:latest": 600,
        "llama3.2-vision:latest": 7000,
        "llama3.2:latest": 4000,
        "deepseek-r1:14b": 9000,
    }
    
    # Task complexity mapping — real Ollama models only
    COMPLEXITY_MAP = {
        "reflex": "llama3.2:latest",          # Fast, simple
        "neurons": "llama3.2:latest",         # Parallel context gathering
        "fast": "llama3.2:latest",            # Quick responses
        "balanced": "mistral-nemo:latest",    # Agent work
        "agent": "mistral-nemo:latest",       # Sub-agent tasks
        "reasoning": "deepseek-r1:14b",       # Complex analysis
        "analysis": "deepseek-r1:14b",        # Deep thinking
        "complex": "deepseek-r1:14b",         # Multi-step problems
    }
    
    def __init__(self, total_vram_mb: int = 32000):
        self.total_vram_mb = total_vram_mb
        self.ollama = OllamaModelManager()
        self._current_elastic: Optional[str] = None
        
    async def initialize(self):
        """Load always-hot models on startup."""
        for model in self.ALWAYS_HOT:
            await self.ollama.preload(model, keep_alive="24h")  # Keep for 24h
        self._current_elastic = "mistral-nemo:latest"
        print(f"[ElasticStrategy] Initialized with: {self.ALWAYS_HOT}")
        
    def get_model_for_complexity(self, complexity: str) -> str:
        """Get the appropriate elastic model for a task complexity."""
        return self.COMPLEXITY_MAP.get(complexity, "llama3.2:latest")
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """
        Ensure an elastic model is loaded, hot-swapping if needed.
        
        Returns True if model is ready.
        """
        # Always-hot models are assumed to be loaded
        if model in self.ALWAYS_HOT:
            return True
        
        # Check if already loaded
        if await self.ollama.is_loaded(model):
            self._current_elastic = model
            return True
        
        # Need to hot-swap - calculate VRAM needed
        required_vram = self.VRAM_MAP.get(model, 10000)
        always_hot_vram = sum(self.VRAM_MAP[m] for m in self.ALWAYS_HOT)
        
        # If current elastic is not in always-hot, unload it
        if self._current_elastic and self._current_elastic not in self.ALWAYS_HOT:
            print(f"[ElasticStrategy] Unloading {self._current_elastic} for {model}")
            await self.ollama.unload(self._current_elastic)
            await asyncio.sleep(1)  # Brief wait for VRAM release
        
        # Load the new model
        success = await self.ollama.preload(model, keep_alive="10m")
        if success:
            self._current_elastic = model
            print(f"[ElasticStrategy] Hot-swapped to {model}")
        return success
    
    async def get_best_model(
        self,
        task_complexity: str = "balanced",
        require_reasoning: bool = False,
        prefer_speed: bool = False
    ) -> str:
        """
        Get the best elastic model for a task.
        
        Args:
            task_complexity: "fast", "balanced", "reasoning", "complex"
            require_reasoning: Force 12b for reasoning quality
            prefer_speed: Prefer 6b for speed even if 9b/12b available
        """
        if prefer_speed:
            return "llama3.2:latest"
        
        if require_reasoning:
            await self.ensure_model_loaded("deepseek-r1:14b")
            return "deepseek-r1:14b"
        
        model = self.get_model_for_complexity(task_complexity)
        await self.ensure_model_loaded(model)
        return model
    
    def get_vram_usage(self) -> Dict[str, int]:
        """Get current VRAM usage by loaded models."""
        loaded = self.ollama._loaded_models
        usage = {}
        for model in loaded:
            for name, vram in self.VRAM_MAP.items():
                if name in model or model in name:
                    usage[model] = vram
                    break
        return usage
    
    def get_available_vram(self) -> int:
        """Estimate available VRAM after loaded models."""
        used = sum(self.get_vram_usage().values())
        return self.total_vram_mb - used


# Singleton elastic strategy
_elastic_strategy: Optional[ElasticModelStrategy] = None


def get_elastic_strategy(total_vram_mb: int = 32000) -> ElasticModelStrategy:
    """Get the singleton elastic model strategy."""
    global _elastic_strategy
    if _elastic_strategy is None:
        _elastic_strategy = ElasticModelStrategy(total_vram_mb)
    return _elastic_strategy


async def select_elastic_model(
    task_complexity: str = "balanced",
    require_reasoning: bool = False
) -> str:
    """
    Quick function to select the best elastic model.
    
    Examples:
        model = await select_elastic_model("fast")      # -> elastic:6b
        model = await select_elastic_model("agent")     # -> elastic:9b  
        model = await select_elastic_model("reasoning") # -> elastic:12b
    """
    strategy = get_elastic_strategy()
    return await strategy.get_best_model(task_complexity, require_reasoning)


# Convenience functions

async def unload_vision_model():
    """Quick function to unload vision model when not needed"""
    manager = get_model_manager()
    await manager.ollama.unload("llama3.2-vision")


async def unload_all_local():
    """Unload all local models to free VRAM"""
    manager = get_model_manager()
    return await manager.ollama.unload_all()


async def prepare_for_image_gen(is_nsfw: bool = False) -> ModelInfo:
    """Prepare system for image generation"""
    manager = get_model_manager()
    return await manager.prepare_for_task("image_gen", is_nsfw=is_nsfw)


async def prepare_for_chat(is_nsfw: bool = False) -> ModelInfo:
    """Prepare system for chat"""
    manager = get_model_manager()
    return await manager.prepare_for_task("chat", is_nsfw=is_nsfw)


async def prepare_for_vision() -> ModelInfo:
    """Prepare system for vision analysis"""
    manager = get_model_manager()
    return await manager.prepare_for_task("vision")


async def smart_model_selection(prompt: str, task: str = "chat") -> ModelInfo:
    """
    Analyze prompt and select the best model.
    
    Examples:
    - "Draw a car" -> nano-banana (fast, SFW)
    - "Draw a sexy anime girl" -> pony (NSFW, local)
    - "Analyze this image" -> llama3.2-vision (vision)
    - "Hello" -> gemini-2.5-flash (fast cloud chat)
    - "Write me a nasty story" -> mistral-nemo (NSFW chat)
    """
    from aither_adk.infrastructure.task_router import ContentClassifier, ContentRating
    
    rating = ContentClassifier.classify(prompt)
    is_nsfw = rating == ContentRating.NSFW
    
    # Check for specific task hints
    vision_hints = ["analyze", "look at", "what's in", "describe this image", "see"]
    is_vision = any(h in prompt.lower() for h in vision_hints)
    
    if is_vision:
        task = "vision"
    
    manager = get_model_manager()
    
    # For simple SFW images, prefer ultra-fast cloud
    if task == "image_gen" and not is_nsfw:
        simple_subjects = ["car", "house", "landscape", "city", "food", "animal", "building", "nature"]
        if any(s in prompt.lower() for s in simple_subjects):
            return MODEL_REGISTRY["nano-banana"]
    
    return await manager.select_model_for_task(
        task=task,
        is_nsfw=is_nsfw,
        prefer_speed=not is_nsfw  # Speed for SFW, quality for NSFW
    )

