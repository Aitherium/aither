"""
AitherOS Hybrid Resource Manager

Manages resource contention between local GPU operations (Ollama, ComfyUI)
and cloud API calls (Gemini, Fal.ai) for optimal performance.

Key Features:
- GPU VRAM monitoring and reservation
- Task queue with priority for local operations
- Smart routing based on content type and resource availability
- Semaphore-based concurrency control
- Deterministic task scheduling for reproducible results
"""

import os
import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any, Dict, List
from queue import PriorityQueue
from contextlib import asynccontextmanager
import subprocess
import json
import requests

# Try to import pynvml for NVIDIA GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("[ResourceManager] pynvml not available - install with: pip install pynvml")


class TaskType(Enum):
    """Categories of GPU-bound tasks with different VRAM requirements"""
    LLM_LOCAL = auto()       # ~4-8GB for mistral-nemo
    DIFFUSION_SDXL = auto()  # ~10-12GB for Pony/SDXL
    DIFFUSION_FLUX = auto()  # ~12-14GB for Flux
    VISION_ANALYSIS = auto() # ~2-4GB for vision models
    CLOUD_API = auto()       # No local GPU, just network


class TaskPriority(Enum):
    """Task priority levels - lower number = higher priority"""
    CRITICAL = 0      # User-facing immediate response
    HIGH = 10         # Interactive tasks
    NORMAL = 20       # Background generation
    LOW = 30          # Batch/async tasks


@dataclass
class ResourceEstimate:
    """Estimated resource requirements for a task"""
    vram_mb: int = 0
    duration_estimate_sec: float = 0
    requires_gpu: bool = True
    can_batch: bool = False


# VRAM estimates for different operations (RTX 5080 has 16GB)
# Lowered estimates to allow ComfyUI to manage its own offloading
VRAM_ESTIMATES: Dict[TaskType, int] = {
    TaskType.LLM_LOCAL: 4000,         # ~4GB for mistral-nemo (quantized)
    TaskType.DIFFUSION_SDXL: 6000,    # ~6GB for Pony/SDXL (can offload)
    TaskType.DIFFUSION_FLUX: 8000,    # ~8GB for Flux (can offload)
    TaskType.VISION_ANALYSIS: 2000,   # ~2GB for vision analysis
    TaskType.CLOUD_API: 0,            # No VRAM
}


@dataclass(order=True)
class QueuedTask:
    """A task waiting to be executed"""
    priority: int
    timestamp: float = field(compare=False)
    task_type: TaskType = field(compare=False)
    coroutine: Any = field(compare=False)
    result_future: asyncio.Future = field(compare=False)
    task_id: str = field(compare=False, default="")


class GPUMonitor:
    """Monitors NVIDIA GPU state"""
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._handle = None
        if HAS_NVML:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                print(f"[GPUMonitor] Failed to get GPU handle: {e}")
    
    def get_vram_info(self) -> Dict[str, int]:
        """Returns VRAM info in MB"""
        if not self._handle:
            return {"total": 16000, "used": 0, "free": 16000}  # Fallback estimate
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "total": info.total // (1024 * 1024),
                "used": info.used // (1024 * 1024),
                "free": info.free // (1024 * 1024)
            }
        except Exception:
            return {"total": 16000, "used": 0, "free": 16000}
    
    def get_gpu_utilization(self) -> int:
        """Returns GPU utilization percentage (0-100)"""
        if not self._handle:
            return 0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return util.gpu
        except Exception:
            return 0
    
    def can_allocate(self, vram_mb: int, buffer_mb: int = 500) -> bool:
        """Check if we can allocate the requested VRAM with buffer"""
        info = self.get_vram_info()
        return info["free"] >= (vram_mb + buffer_mb)

    def unload_all_models(self):
        """Unload all Ollama and ComfyUI models to free VRAM"""
        print("[ResourceManager]  Attempting to unload all models to free VRAM...")
        
        # 1. Unload Ollama (URL from services.yaml)
        try:
            from lib.core.AitherPorts import ollama_url
            base_url = ollama_url().rstrip("/")
            if "/api" in base_url:
                base_url = base_url.replace("/api", "")
                
            # List running models
            try:
                ps_resp = requests.get(f"{base_url}/api/ps", timeout=2)
                if ps_resp.status_code == 200:
                    models = ps_resp.json().get("models", [])
                    if models:
                        for m in models:
                            name = m["name"]
                            print(f"[ResourceManager] Unloading Ollama model: {name}...")
                            # Unload by setting keep_alive to 0
                            requests.post(f"{base_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=5)
            except Exception as e:
                print(f"[ResourceManager] Failed to list/unload Ollama models: {e}")
                
        except Exception as e:
            print(f"[ResourceManager] Error during Ollama cleanup: {e}")

        # 2. Unload ComfyUI
        try:
            comfy_url = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
            if not comfy_url.startswith("http"):
                comfy_url = f"http://{comfy_url}"
            
            print("[ResourceManager] Requesting ComfyUI to free memory...")
            # ComfyUI /free endpoint triggers GC and model unload
            requests.post(f"{comfy_url}/free", json={"unload_models": True, "free_memory": True}, timeout=2)
        except Exception as e:
            print(f"[ResourceManager] Failed to unload ComfyUI models: {e}")


class ResourceManager:
    """
    Singleton resource manager for AitherOS.
    
    Coordinates access to GPU resources between Ollama and ComfyUI,
    ensuring deterministic operation and preventing OOM.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # GPU Monitor
        self.gpu_monitor = GPUMonitor()
        
        # Semaphores for different resource types
        # Only ONE diffusion job at a time (they're heavy)
        self._diffusion_semaphore = asyncio.Semaphore(1)
        # Only ONE local LLM job at a time (VRAM conflict with diffusion)
        self._llm_semaphore = asyncio.Semaphore(1)
        # Cloud API can have multiple concurrent requests
        self._cloud_semaphore = asyncio.Semaphore(5)
        # Master lock - only one GPU-heavy task at a time
        self._gpu_master_lock = asyncio.Lock()
        
        # Task queue for prioritized scheduling
        self._task_queue: PriorityQueue = PriorityQueue()
        self._active_tasks: Dict[str, QueuedTask] = {}
        
        # Statistics
        self._stats = {
            "total_tasks": 0,
            "cloud_tasks": 0,
            "local_tasks": 0,
            "queue_waits": 0,
            "vram_conflicts": 0,
        }
        
        # Configuration
        self.config = HybridConfig()
        
        # Background worker
        self._worker_task = None
        self._shutdown = False
        
        print("[ResourceManager] Initialized")
        print(f"[ResourceManager] GPU VRAM: {self.gpu_monitor.get_vram_info()}")
    
    def unload_all_models(self):
        """Unload all models via the GPU monitor"""
        self.gpu_monitor.unload_all_models()

    async def start(self):
        """Start the background task processor"""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop the background task processor"""
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _process_queue(self):
        """Background worker that processes the task queue"""
        while not self._shutdown:
            try:
                if not self._task_queue.empty():
                    task = self._task_queue.get_nowait()
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[ResourceManager] Queue worker error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: QueuedTask):
        """Execute a queued task with proper resource management"""
        try:
            result = await task.coroutine
            if not task.result_future.done():
                task.result_future.set_result(result)
        except Exception as e:
            if not task.result_future.done():
                task.result_future.set_exception(e)
        finally:
            self._active_tasks.pop(task.task_id, None)
    
    @asynccontextmanager
    async def acquire_gpu(self, task_type: TaskType, timeout: float = 300):
        """
        Context manager to acquire GPU resources.
        
        Usage:
            async with resource_manager.acquire_gpu(TaskType.DIFFUSION_SDXL):
                result = await generate_image(...)
        """
        vram_needed = VRAM_ESTIMATES.get(task_type, 0)
        
        if task_type == TaskType.CLOUD_API:
            async with self._cloud_semaphore:
                self._stats["cloud_tasks"] += 1
                yield
            return
        
        # For GPU tasks, we need proper coordination
        start_time = time.time()
        acquired = False
        
        try:
            # Wait for GPU availability
            while not acquired and (time.time() - start_time) < timeout:
                # Check if we have enough VRAM
                if not self.gpu_monitor.can_allocate(vram_needed):
                    self._stats["vram_conflicts"] += 1
                    print(f"[ResourceManager] Waiting for VRAM... (need {vram_needed}MB)")
                    
                    # Attempt to free VRAM if we are stuck
                    self.unload_all_models()
                    
                    await asyncio.sleep(2)
                    continue
                
                # Try to acquire the appropriate semaphore
                if task_type in [TaskType.DIFFUSION_SDXL, TaskType.DIFFUSION_FLUX]:
                    if self._diffusion_semaphore.locked():
                        await asyncio.sleep(0.5)
                        continue
                    await self._diffusion_semaphore.acquire()
                    acquired = True
                    
                elif task_type == TaskType.LLM_LOCAL:
                    if self._llm_semaphore.locked():
                        await asyncio.sleep(0.5)
                        continue
                    await self._llm_semaphore.acquire()
                    acquired = True
                    
                elif task_type == TaskType.VISION_ANALYSIS:
                    # Vision can run alongside other tasks if VRAM allows
                    acquired = True
                else:
                    acquired = True
            
            if not acquired:
                raise TimeoutError(f"Could not acquire GPU resources within {timeout}s")
            
            self._stats["local_tasks"] += 1
            yield
            
        finally:
            if acquired:
                if task_type in [TaskType.DIFFUSION_SDXL, TaskType.DIFFUSION_FLUX]:
                    self._diffusion_semaphore.release()
                elif task_type == TaskType.LLM_LOCAL:
                    self._llm_semaphore.release()
    
    def should_use_local(self, content_type: str, is_nsfw: bool = False) -> bool:
        """
        Determine if we should use local models based on content and resources.
        
        Args:
            content_type: "text" or "image"
            is_nsfw: Whether the content is NSFW (requires local for uncensored)
        
        Returns:
            True if local model should be used
        """
        # NSFW content MUST use local models
        if is_nsfw:
            return True
        
        # Check GPU availability
        vram_info = self.gpu_monitor.get_vram_info()
        gpu_util = self.gpu_monitor.get_gpu_utilization()
        
        # If GPU is busy (>80% util or <3GB free), prefer cloud
        if gpu_util > 80 or vram_info["free"] < 3000:
            return False
        
        # Follow config preferences
        if content_type == "text":
            return self.config.prefer_local_llm
        elif content_type == "image":
            return self.config.prefer_local_diffusion
        
        return False
    
    def get_best_model(self, task_type: TaskType, is_nsfw: bool = False) -> Dict[str, Any]:
        """
        Get the best model configuration for a task.
        
        Returns:
            Dict with "provider", "model", "reason"
        """
        if task_type == TaskType.LLM_LOCAL or (task_type == TaskType.CLOUD_API and is_nsfw):
            if is_nsfw or self.should_use_local("text", is_nsfw):
                return {
                    "provider": "ollama",
                    "model": self.config.local_llm_model,
                    "reason": "NSFW content or local preference"
                }
            return {
                "provider": "google",
                "model": self.config.cloud_llm_model,
                "reason": "Cloud preferred for SFW content"
            }
        
        if task_type in [TaskType.DIFFUSION_SDXL, TaskType.DIFFUSION_FLUX]:
            if is_nsfw or self.should_use_local("image", is_nsfw):
                return {
                    "provider": "comfyui",
                    "model": "pony" if is_nsfw else "flux",
                    "reason": "NSFW content or local preference"
                }
            return {
                "provider": "fal",
                "model": "fal-ai/flux/schnell",
                "reason": "Cloud preferred for SFW images"
            }
        
        return {
            "provider": "google",
            "model": self.config.cloud_llm_model,
            "reason": "Default"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        vram = self.gpu_monitor.get_vram_info()
        return {
            "vram_total_mb": vram["total"],
            "vram_used_mb": vram["used"],
            "vram_free_mb": vram["free"],
            "gpu_utilization": self.gpu_monitor.get_gpu_utilization(),
            "diffusion_busy": self._diffusion_semaphore.locked(),
            "llm_busy": self._llm_semaphore.locked(),
            "queue_size": self._task_queue.qsize(),
            "active_tasks": len(self._active_tasks),
            "stats": self._stats.copy()
        }
    
    def print_status(self):
        """Print formatted resource status"""
        status = self.get_status()
        print("\n" + "="*50)
        print("[DESKTOP]  AitherOS Resource Status")
        print("="*50)
        print(f"GPU VRAM:      {status['vram_used_mb']:,}MB / {status['vram_total_mb']:,}MB ({status['vram_free_mb']:,}MB free)")
        print(f"GPU Util:      {status['gpu_utilization']}%")
        print(f"Diffusion:     {'[RED] Busy' if status['diffusion_busy'] else '[GREEN] Available'}")
        print(f"Local LLM:     {'[RED] Busy' if status['llm_busy'] else '[GREEN] Available'}")
        print(f"Queue:         {status['queue_size']} pending")
        print(f"Active:        {status['active_tasks']} running")
        print("-"*50)
        print(f"Total Tasks:   {status['stats']['total_tasks']}")
        print(f"Cloud Tasks:   {status['stats']['cloud_tasks']}")
        print(f"Local Tasks:   {status['stats']['local_tasks']}")
        print(f"VRAM Conflicts:{status['stats']['vram_conflicts']}")
        print("="*50 + "\n")


class HybridConfig:
    """Configuration for hybrid cloud/local operation"""
    
    def __init__(self):
        # Load from environment with sensible defaults
        self.cloud_llm_model = os.getenv("CLOUD_LLM_MODEL", "gemini-2.5-flash")
        self.local_llm_model = os.getenv("LOCAL_LLM_MODEL", "aither-orchestrator-8b-v4")
        
        # Preferences
        self.prefer_local_llm = os.getenv("PREFER_LOCAL_LLM", "false").lower() == "true"
        self.prefer_local_diffusion = os.getenv("PREFER_LOCAL_DIFFUSION", "true").lower() == "true"
        
        # VRAM thresholds
        self.min_free_vram_mb = int(os.getenv("MIN_FREE_VRAM_MB", "2000"))
        
        # Timeouts
        self.local_timeout_sec = int(os.getenv("LOCAL_TIMEOUT_SEC", "120"))
        self.cloud_timeout_sec = int(os.getenv("CLOUD_TIMEOUT_SEC", "60"))
        
        # Concurrency
        self.max_concurrent_local = int(os.getenv("MAX_CONCURRENT_LOCAL", "1"))
        self.max_concurrent_cloud = int(os.getenv("MAX_CONCURRENT_CLOUD", "5"))


# Singleton instance
resource_manager = ResourceManager()


# Convenience decorators
def gpu_task(task_type: TaskType):
    """
    Decorator for functions that require GPU resources.
    
    Usage:
        @gpu_task(TaskType.DIFFUSION_SDXL)
        async def generate_image(prompt):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with resource_manager.acquire_gpu(task_type):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def cloud_fallback(local_func, cloud_func, task_type: TaskType):
    """
    Wrapper that tries local first, falls back to cloud on failure or resource constraints.
    
    Usage:
        generate = cloud_fallback(generate_local, generate_cloud, TaskType.DIFFUSION_SDXL)
        result = await generate(prompt=...)
    """
    async def wrapper(*args, is_nsfw: bool = False, **kwargs):
        model_config = resource_manager.get_best_model(task_type, is_nsfw)
        
        if model_config["provider"] in ["ollama", "comfyui"]:
            try:
                async with resource_manager.acquire_gpu(task_type, timeout=30):
                    return await local_func(*args, **kwargs)
            except (TimeoutError, Exception) as e:
                print(f"[Fallback] Local failed ({e}), using cloud")
        
        return await cloud_func(*args, **kwargs)
    
    return wrapper

