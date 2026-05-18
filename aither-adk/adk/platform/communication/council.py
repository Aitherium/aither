"""
AitherCouncil - Intelligent Multi-Model Routing System

╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  DEPRECATED - Use Genesis Instead!                                       ║
║                                                                              ║
║  This module duplicates functionality that exists in:                        ║
║    AitherGenesis (THE BRAIN - Port 8001)                                     ║
║                                                                              ║
║  Genesis (THE BRAIN - Port 8001) provides:                                   ║
║    - /select-llm endpoint for model selection                                ║
║    - /delegate endpoint for multi-agent tasks                                ║
║    - LLMSelector with complexity-based routing                               ║
║    - Elastic model hot-swapping for RTX 5090                                 ║
║    - DeepSeek-R1 in REASONING tier                                           ║
║                                                                              ║
║  To migrate:                                                                 ║
║    from lib.clients.orchestrator import get_orchestrator_client              ║
║    client = get_orchestrator_client()                                        ║
║    result = await client.select_llm(task_description="...")                  ║
║                                                                              ║
║  This file is kept for backward compatibility but should NOT be used         ║
║  for new code. It will be removed in a future version.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module implements the AitherCouncil, a sophisticated model routing system
that delegates tasks to the optimal model based on task characteristics.

The NVIDIA Orchestrator-8B acts as the coordinator, analyzing tasks and routing
them to specialized models:

- DeepSeek R1/V3: Deep reasoning, complex analysis, mathematical problems
- Mistral/Mixtral: General purpose, balanced tasks, conversation
- Creative Models: Storytelling, writing, artistic descriptions  
- Code Models: Programming, debugging, code generation
- Fast Models: Quick responses, simple queries, classification

Architecture:
    User Request -> NVIDIA Orchestrator (classifier) -> Specialized Model -> Response
"""
import warnings
warnings.warn(
    "aither_adk.communication.council is deprecated. "
    "Use Genesis service at port 8001 (/select-llm endpoint) instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import re
import yaml
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from aither_adk.ai.models import (
    get_orchestrator_model,
    is_vllm_available,
    is_ollama_available,
    get_ollama_models,
    get_vllm_models,
)


class TaskType(Enum):
    """Classification of task types for model routing."""
    
    REASONING = "reasoning"           # Complex logic, math, analysis
    CREATIVE_WRITING = "creative"     # Stories, poetry, artistic content
    CODE_GENERATION = "code"          # Programming, debugging, code review
    GENERAL = "general"               # Conversation, Q&A, explanations
    FAST = "fast"                     # Quick responses, classification
    VISION = "vision"                 # Image understanding
    TOOL_USE = "tool_use"             # Tool calling, agent workflows
    ORCHESTRATION = "orchestration"   # Multi-step planning, coordination


@dataclass
class ModelSpec:
    """Specification for a model with its capabilities and priorities."""
    
    name: str
    provider: str                     # ollama, vllm, nvidia, google, openai
    task_types: List[TaskType]        # What this model excels at
    priority: int = 5                 # Higher = preferred (1-10)
    max_tokens: int = 8192
    temperature: float = 0.7
    requires_gpu: bool = False
    requires_api_key: bool = False
    api_key_env: Optional[str] = None
    context_window: int = 32768
    notes: str = ""
    

# =============================================================================
# MODEL REGISTRY - Specialized Models for Each Task Type
# =============================================================================
# 
# GPU MODEL ROUTING (RTX 5090 32GB + 128GB DDR5)
# =====================================================================
# vLLM-Swap (port 8176) serves deepseek-r1:14b (~14GB VRAM)
# aither-orchestrator (port 8120) serves orchestrator-8b (~5GB VRAM)
# Ollama handles embeddings, vision, and fallbacks
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # =========================================================================
    # REAL OLLAMA MODELS — Use these for local inference
    # vLLM serves deepseek-r1:14b; Ollama uses these real models.
    # =========================================================================
    "deepseek-r1-14b": ModelSpec(
        name="deepseek-r1:14b",
        provider="ollama",
        task_types=[TaskType.REASONING, TaskType.GENERAL],
        priority=10,  # Top priority for reasoning
        max_tokens=16384,
        temperature=0.3,
        requires_gpu=True,
        context_window=32768,
        notes="DeepSeek-R1 14B. Strong reasoning. ~9GB VRAM. Best local reasoning model.",
    ),
    "mistral-nemo": ModelSpec(
        name="mistral-nemo:latest",
        provider="ollama",
        task_types=[TaskType.GENERAL, TaskType.TOOL_USE],
        priority=9,
        max_tokens=16384,
        temperature=0.5,
        requires_gpu=True,
        context_window=32768,
        notes="Mistral Nemo. Balanced quality. ~7GB VRAM. Good for agents/subagents.",
    ),
    "llama3-2": ModelSpec(
        name="llama3.2:latest",
        provider="ollama",
        task_types=[TaskType.FAST, TaskType.GENERAL],
        priority=9,  # High priority for fast tasks
        max_tokens=8192,
        temperature=0.5,
        requires_gpu=True,
        context_window=32768,
        notes="Llama 3.2. Fast reflex model. ~4GB VRAM. Keep always hot.",
    ),
    
    # =========================================================================
    # ORCHESTRATION / TOOL USE (NVIDIA - via Ollama)
    # The Orchestrator-8B should ALWAYS be loaded for task routing
    # =========================================================================
    "nvidia-orchestrator-8b": ModelSpec(
        name="orchestrator-8b",  # Ollama model name (NOT HuggingFace path)
        provider="ollama",        # Use Ollama, NOT vLLM
        task_types=[TaskType.ORCHESTRATION, TaskType.TOOL_USE],
        priority=10,  # Highest priority for orchestration
        temperature=0.3,
        notes="Primary orchestrator via Ollama. Routes tasks to elastic models. KEEP ALWAYS HOT (~5GB).",
    ),
    "aither-orchestrator-8b-v4": ModelSpec(
        name="aither-orchestrator-8b-v4",  # v4 context-aware anti-sycophantic
        provider="ollama",
        task_types=[TaskType.ORCHESTRATION, TaskType.TOOL_USE],
        priority=10,
        temperature=0.3,
        notes="Custom fine-tuned orchestrator. Fallback if orchestrator-8b unavailable.",
    ),
    "aither-orchestrator-8b": ModelSpec(
        name="aither-orchestrator-8b-v4",  # Alias to v4
        provider="ollama",
        task_types=[TaskType.ORCHESTRATION, TaskType.TOOL_USE],
        priority=10,
        temperature=0.3,
        notes="Custom fine-tuned orchestrator. Fallback if orchestrator-8b unavailable.",
    ),
    "qwen2.5-14b": ModelSpec(
        name="qwen2.5:14b",
        provider="ollama",
        task_types=[TaskType.TOOL_USE, TaskType.CODE_GENERATION, TaskType.GENERAL],
        priority=8,
        notes="Excellent tool-use fallback when NVIDIA unavailable.",
    ),
    
    # =========================================================================
    # DEEP REASONING (DeepSeek)
    # =========================================================================
    "deepseek-r1": ModelSpec(
        name="deepseek-reasoner",
        provider="deepseek",
        task_types=[TaskType.REASONING],
        priority=10,
        max_tokens=16384,
        temperature=0.0,  # Deterministic for reasoning
        requires_api_key=True,
        api_key_env="DEEPSEEK_API_KEY",
        context_window=65536,
        notes="DeepSeek-R1: Best-in-class reasoning with chain-of-thought.",
    ),
    "deepseek-v3": ModelSpec(
        name="deepseek-chat",
        provider="deepseek",
        task_types=[TaskType.REASONING, TaskType.CODE_GENERATION],
        priority=9,
        max_tokens=16384,
        requires_api_key=True,
        api_key_env="DEEPSEEK_API_KEY",
        context_window=65536,
        notes="DeepSeek-V3: Strong reasoning + code, cost-effective.",
    ),
    "qwq-32b": ModelSpec(
        name="qwq:32b",
        provider="ollama",
        task_types=[TaskType.REASONING],
        priority=8,
        requires_gpu=True,
        notes="Local reasoning model. Requires 24GB+ VRAM.",
    ),
    
    # =========================================================================
    # CREATIVE WRITING
    # =========================================================================
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        provider="google",
        task_types=[TaskType.CREATIVE_WRITING, TaskType.GENERAL],
        priority=9,
        temperature=0.9,
        requires_api_key=True,
        api_key_env="GOOGLE_API_KEY",
        context_window=1000000,
        notes="Excellent for creative writing with huge context.",
    ),
    "claude-sonnet": ModelSpec(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        task_types=[TaskType.CREATIVE_WRITING, TaskType.REASONING],
        priority=9,
        temperature=0.8,
        requires_api_key=True,
        api_key_env="ANTHROPIC_API_KEY",
        notes="Excellent creative writing and nuanced reasoning.",
    ),
    "mistral-nemo": ModelSpec(
        name="mistral-nemo",
        provider="ollama",
        task_types=[TaskType.CREATIVE_WRITING, TaskType.GENERAL],
        priority=7,
        temperature=0.85,
        notes="Local creative model. Good for stories and dialogue.",
    ),
    "llama3.2-vision": ModelSpec(
        name="llama3.2-vision",
        provider="ollama",
        task_types=[TaskType.VISION, TaskType.GENERAL],
        priority=8,
        notes="Vision + text. Good for image descriptions.",
    ),
    
    # =========================================================================
    # CODE GENERATION
    # =========================================================================
    "qwen2.5-coder-32b": ModelSpec(
        name="qwen2.5-coder:32b",
        provider="ollama",
        task_types=[TaskType.CODE_GENERATION],
        priority=10,
        temperature=0.2,
        requires_gpu=True,
        notes="Best local code model. Requires 24GB+ VRAM.",
    ),
    "qwen2.5-coder-7b": ModelSpec(
        name="qwen2.5-coder:7b",
        provider="ollama",
        task_types=[TaskType.CODE_GENERATION],
        priority=7,
        temperature=0.2,
        notes="Lighter code model for smaller GPUs.",
    ),
    "codestral": ModelSpec(
        name="codestral:latest",
        provider="ollama",
        task_types=[TaskType.CODE_GENERATION],
        priority=8,
        temperature=0.2,
        notes="Mistral's code model. Strong for code generation.",
    ),
    
    # =========================================================================
    # GENERAL PURPOSE
    # =========================================================================
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        provider="google",
        task_types=[TaskType.GENERAL, TaskType.FAST],
        priority=8,
        requires_api_key=True,
        api_key_env="GOOGLE_API_KEY",
        notes="Fast, capable general model. Good default.",
    ),
    "llama3.2": ModelSpec(
        name="llama3.2",
        provider="ollama",
        task_types=[TaskType.GENERAL, TaskType.FAST],
        priority=6,
        notes="Local general purpose. Good for basic tasks.",
    ),
    "mistral": ModelSpec(
        name="mistral:latest",
        provider="ollama",
        task_types=[TaskType.GENERAL],
        priority=6,
        notes="Local Mistral 7B. Balanced general use.",
    ),
    
    # =========================================================================
    # FAST MODELS
    # =========================================================================
    "gemini-2.0-flash": ModelSpec(
        name="gemini-2.0-flash",
        provider="google",
        task_types=[TaskType.FAST, TaskType.GENERAL],
        priority=9,
        temperature=0.5,
        requires_api_key=True,
        api_key_env="GOOGLE_API_KEY",
        notes="Very fast. Use for quick classification/responses.",
    ),
    "phi3": ModelSpec(
        name="phi3",
        provider="ollama",
        task_types=[TaskType.FAST],
        priority=5,
        notes="Microsoft Phi-3. Very fast, small model.",
    ),
}


# =============================================================================
# TASK CLASSIFICATION PATTERNS
# =============================================================================

TASK_PATTERNS: Dict[TaskType, List[str]] = {
    TaskType.REASONING: [
        r"(?i)\b(prove|derive|calculate|solve|analyze|explain why|reason|logic)\b",
        r"(?i)\b(math|equation|theorem|proof|hypothesis|deduce)\b",
        r"(?i)\b(step.by.step|chain.of.thought|think through|work through)\b",
        r"(?i)\b(compare and contrast|evaluate|assess|critique)\b",
    ],
    TaskType.CREATIVE_WRITING: [
        r"(?i)\b(write|compose|create|story|poem|narrative|fiction)\b",
        r"(?i)\b(character|dialogue|scene|chapter|plot|setting)\b",
        r"(?i)\b(creative|artistic|imaginative|descriptive|prose)\b",
        r"(?i)\b(roleplay|rp|scenario|world.?building)\b",
    ],
    TaskType.CODE_GENERATION: [
        r"(?i)\b(code|program|function|class|method|implement)\b",
        r"(?i)\b(python|javascript|typescript|powershell|bash|sql)\b",
        r"(?i)\b(debug|fix|refactor|optimize|test|unit test)\b",
        r"(?i)\b(api|endpoint|database|query|algorithm)\b",
    ],
    TaskType.TOOL_USE: [
        r"(?i)\b(execute|run|call|invoke|use tool|tool call)\b",
        r"(?i)\b(script|automation|playbook|workflow)\b",
        r"(?i)\b(search|fetch|get|list|create|delete|update)\b",
    ],
    TaskType.FAST: [
        r"(?i)\b(quick|fast|brief|short|simple|yes or no)\b",
        r"(?i)\b(classify|categorize|label|tag)\b",
        r"(?i)^(what is|who is|when|where|define)\b",
    ],
    TaskType.VISION: [
        r"(?i)\b(image|photo|picture|screenshot|diagram|visual)\b",
        r"(?i)\b(see|look at|analyze this image|describe the)\b",
    ],
}


class AitherCouncil:
    """
    Intelligent model routing system that delegates tasks to optimal models.
    
    The Council uses the NVIDIA Orchestrator-8B as the primary coordinator,
    with specialized models for different task types:
    
    - DeepSeek R1: Deep reasoning, mathematical proofs, complex analysis
    - DeepSeek V3: Code + reasoning, cost-effective alternative
    - Gemini Pro: Creative writing, long-context tasks
    - Qwen Coder: Code generation, debugging
    - Mistral: General conversation, balanced tasks
    - Fast models: Quick responses, classification
    
    Example:
        council = AitherCouncil()
        
        # Classify and route a task
        task_type, model = council.route_task(
            "Prove that the square root of 2 is irrational"
        )
        # Returns: (TaskType.REASONING, "deepseek-r1")
        
        # Get model details
        spec = council.get_model_spec("deepseek-r1")
        print(f"Using {spec.name} via {spec.provider}")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AitherCouncil.
        
        Args:
            config_path: Optional path to council configuration YAML.
        """
        self.config = self._load_config(config_path)
        self._available_models: Optional[Dict[str, ModelSpec]] = None
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load council configuration from YAML."""
        if config_path:
            path = Path(config_path)
        else:
            path = Path(__file__).parent / "config" / "council.yaml"
        
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "orchestrator": "nvidia-orchestrator-8b",
            "fallback_orchestrator": "qwen2.5-14b",
            "default_task_type": "general",
            "enable_auto_routing": True,
            "prefer_local": True,
        }
    
    def get_available_models(self, refresh: bool = False) -> Dict[str, ModelSpec]:
        """
        Get currently available models based on running services and API keys.
        
        Args:
            refresh: Force refresh of availability check.
            
        Returns:
            Dictionary of model_id -> ModelSpec for available models.
        """
        if self._available_models is not None and not refresh:
            return self._available_models
        
        available = {}
        
        # Check vLLM models
        if is_vllm_available():
            vllm_models = get_vllm_models()
            for model_id, spec in MODEL_REGISTRY.items():
                if spec.provider == "vllm":
                    # Check if model is loaded in vLLM
                    for vm in vllm_models:
                        if spec.name in vm or vm in spec.name:
                            available[model_id] = spec
                            break
        
        # Check Ollama models
        if is_ollama_available():
            ollama_models = get_ollama_models()
            for model_id, spec in MODEL_REGISTRY.items():
                if spec.provider == "ollama":
                    # Check if model is pulled in Ollama
                    for om in ollama_models:
                        if spec.name.split(":")[0] in om or om in spec.name:
                            available[model_id] = spec
                            break
        
        # Check cloud models with API keys
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.requires_api_key and spec.api_key_env:
                if os.getenv(spec.api_key_env):
                    available[model_id] = spec
        
        self._available_models = available
        return available
    
    def classify_task(self, prompt: str) -> TaskType:
        """
        Classify a task based on its prompt.
        
        Args:
            prompt: The user's prompt/task description.
            
        Returns:
            The classified TaskType.
        """
        prompt_lower = prompt.lower()
        
        # Check each task type's patterns
        scores: Dict[TaskType, int] = {t: 0 for t in TaskType}
        
        for task_type, patterns in TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt):
                    scores[task_type] += 1
        
        # Get highest scoring type
        if max(scores.values()) > 0:
            return max(scores.keys(), key=lambda t: scores[t])
        
        # Default to general
        return TaskType.GENERAL
    
    def get_model_for_task(
        self, 
        task_type: TaskType, 
        prefer_local: bool = True
    ) -> Optional[str]:
        """
        Get the best available model for a task type.
        
        Args:
            task_type: The type of task to route.
            prefer_local: Prefer local models over cloud APIs.
            
        Returns:
            Model ID of the best available model, or None.
        """
        available = self.get_available_models()
        
        # Filter models that handle this task type
        candidates = [
            (model_id, spec)
            for model_id, spec in available.items()
            if task_type in spec.task_types
        ]
        
        if not candidates:
            # Fall back to general-purpose models
            candidates = [
                (model_id, spec)
                for model_id, spec in available.items()
                if TaskType.GENERAL in spec.task_types
            ]
        
        if not candidates:
            return None
        
        # Sort by priority and local preference
        def sort_key(item):
            model_id, spec = item
            local_bonus = 10 if prefer_local and spec.provider in ("ollama", "vllm") else 0
            return spec.priority + local_bonus
        
        candidates.sort(key=sort_key, reverse=True)
        return candidates[0][0]
    
    def route_task(self, prompt: str) -> Tuple[TaskType, Optional[str]]:
        """
        Classify a task and route it to the optimal model.
        
        Args:
            prompt: The user's prompt/task description.
            
        Returns:
            Tuple of (TaskType, model_id).
        """
        task_type = self.classify_task(prompt)
        model_id = self.get_model_for_task(
            task_type, 
            prefer_local=self.config.get("prefer_local", True)
        )
        return task_type, model_id
    
    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        """
        Get the specification for a model.
        
        Args:
            model_id: The model identifier.
            
        Returns:
            ModelSpec or None if not found.
        """
        return MODEL_REGISTRY.get(model_id)
    
    def get_orchestrator(self) -> Tuple[str, ModelSpec]:
        """
        Get the orchestrator model for task routing.
        
        Returns:
            Tuple of (model_id, ModelSpec) for the orchestrator.
        """
        available = self.get_available_models()
        
        # Try primary orchestrator
        primary = self.config.get("orchestrator", "nvidia-orchestrator-8b")
        if primary in available:
            return primary, available[primary]
        
        # Try fallback
        fallback = self.config.get("fallback_orchestrator", "qwen2.5-14b")
        if fallback in available:
            return fallback, available[fallback]
        
        # Find any available model with orchestration capability
        for model_id, spec in available.items():
            if TaskType.ORCHESTRATION in spec.task_types:
                return model_id, spec
        
        # Last resort: any tool-use capable model
        for model_id, spec in available.items():
            if TaskType.TOOL_USE in spec.task_types:
                return model_id, spec
        
        # Ultimate fallback
        return "gemini-2.5-flash", MODEL_REGISTRY.get("gemini-2.5-flash")
    
    def get_model_endpoint(self, model_id: str) -> Dict[str, Any]:
        """
        Get the API endpoint configuration for a model.
        
        Args:
            model_id: The model identifier.
            
        Returns:
            Dictionary with base_url, api_key, and other connection info.
        """
        spec = self.get_model_spec(model_id)
        if not spec:
            raise ValueError(f"Unknown model: {model_id}")
        
        from lib.core.AitherPorts import ollama_url
        endpoints = {
            "ollama": {
                "base_url": ollama_url(),
                "model": spec.name,
            },
            "vllm": {
                "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                "model": spec.name,
            },
            "google": {
                "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                "model": spec.name,
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": spec.name,
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "model": spec.name,
            },
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": os.getenv("NVIDIA_API_KEY"),
                "model": spec.name,
            },
        }
        
        return endpoints.get(spec.provider, {"model": spec.name})
    
    def format_model_name(self, model_id: str) -> str:
        """
        Format model ID for use with ADK/LiteLLM.
        
        Args:
            model_id: The model identifier.
            
        Returns:
            Formatted model name with provider prefix.
        """
        spec = self.get_model_spec(model_id)
        if not spec:
            return model_id
        
        # Format based on provider
        if spec.provider == "ollama":
            return f"ollama/{spec.name}"
        elif spec.provider == "vllm":
            return f"vllm/{spec.name}"
        elif spec.provider == "deepseek":
            return f"deepseek/{spec.name}"
        elif spec.provider == "anthropic":
            return spec.name  # ADK handles directly
        else:
            return spec.name
    
    def list_models_by_task(self, task_type: TaskType) -> List[Dict[str, Any]]:
        """
        List all models capable of handling a task type.
        
        Args:
            task_type: The task type to query.
            
        Returns:
            List of model info dictionaries.
        """
        available = self.get_available_models()
        
        models = []
        for model_id, spec in MODEL_REGISTRY.items():
            if task_type in spec.task_types:
                models.append({
                    "id": model_id,
                    "name": spec.name,
                    "provider": spec.provider,
                    "priority": spec.priority,
                    "available": model_id in available,
                    "notes": spec.notes,
                })
        
        # Sort by priority (descending), availability first
        models.sort(key=lambda m: (m["available"], m["priority"]), reverse=True)
        return models


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_council() -> AitherCouncil:
    """Get a singleton AitherCouncil instance."""
    if not hasattr(get_council, "_instance"):
        get_council._instance = AitherCouncil()
    return get_council._instance


def route_to_model(prompt: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Quick routing function for common use.
    
    Args:
        prompt: The user's prompt.
        
    Returns:
        Tuple of (task_type_name, model_id, endpoint_config).
    """
    council = get_council()
    task_type, model_id = council.route_task(prompt)
    
    if model_id:
        endpoint = council.get_model_endpoint(model_id)
    else:
        endpoint = {}
    
    return task_type.value, model_id or "gemini-2.5-flash", endpoint


def get_reasoning_model() -> str:
    """Get the best available reasoning model."""
    council = get_council()
    model_id = council.get_model_for_task(TaskType.REASONING)
    return council.format_model_name(model_id) if model_id else "gemini-2.5-flash"


def get_creative_model() -> str:
    """Get the best available creative writing model."""
    council = get_council()
    model_id = council.get_model_for_task(TaskType.CREATIVE_WRITING)
    return council.format_model_name(model_id) if model_id else "gemini-2.5-pro"


def get_code_model() -> str:
    """Get the best available code generation model."""
    council = get_council()
    model_id = council.get_model_for_task(TaskType.CODE_GENERATION)
    return council.format_model_name(model_id) if model_id else "gemini-2.5-flash"


# Export key components
__all__ = [
    "AitherCouncil",
    "TaskType", 
    "ModelSpec",
    "MODEL_REGISTRY",
    "get_council",
    "route_to_model",
    "get_reasoning_model",
    "get_creative_model",
    "get_code_model",
]
