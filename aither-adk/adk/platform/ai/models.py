import logging
import os
import sys
import time

import requests

logger = logging.getLogger(__name__)

# ===============================================================================
# NVIDIA ORCHESTRATOR-8B INTEGRATION
# ===============================================================================
# The Orchestrator-8B model is optimized for tool calling and is the preferred
# model for AitherOS agents when available.

# ===============================================================================
# AITHERLLM UNIFIED GATEWAY
# ===============================================================================
# MicroScheduler (port 8150) provides unified access to all LLM backends with
# automatic fallback: nvidia-orchestrator -> mistral-nemo -> gemini -> claude -> gpt
# Use model prefix "aither/" to route through MicroScheduler (e.g., "aither/nvidia-orchestrator")

AITHERLLM_URL = os.getenv("AITHERLLM_URL", "http://localhost:8150")

# ===============================================================================
# PERFORMANCE CACHES - Avoid redundant network calls
# ===============================================================================
_CACHE_TTL = 30  # Cache validity in seconds
_ollama_models_cache: dict = {"models": [], "timestamp": 0}
_orchestrator_model_cache: dict = {"model": None, "timestamp": 0}
_aitherllm_available_cache: dict = {"available": None, "timestamp": 0}

# Import NVIDIA provider constants
try:
    from aither_adk.infrastructure.nvidia import (
        NVIDIA_MODELS,
        NVIDIA_ORCHESTRATOR_MODEL,
        NvidiaLlm,
        OrchestratorLlm,
        is_nvidia_available,
    )
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_MODELS = {}
    NVIDIA_ORCHESTRATOR_MODEL = "nvidia/Orchestrator-8B"
    NVIDIA_AVAILABLE = False

    def is_nvidia_available():
        return bool(os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_URL"))

# Known local model prefixes/patterns (Ollama models)
LOCAL_MODEL_PATTERNS = [
    "mistral", "llama", "codellama", "vicuna", "orca", "phi",
    "gemma", "qwen", "deepseek", "starcoder", "wizardcoder",
    "neural-chat", "openchat", "nous-hermes", "dolphin", "yi",
    "mixtral", "solar", "stablelm", "zephyr", "tinyllama",
    "aither-orchestrator", "aither-7b", "aither-",  # Custom AitherOS models
    "orchestrator-8b", "orchestrator",  # NVIDIA orchestrator models via Ollama
    "aither-orchestrator-8b-v4",  # v4 context-aware anti-sycophantic model
]

def is_local_model(model_name):
    """Check if a model name appears to be a local (Ollama) model."""
    if not model_name:
        return False
    model_lower = model_name.lower()

    # Check for ollama/ prefix first
    if model_lower.startswith("ollama/"):
        return True

    # Check if it matches known local model patterns
    for pattern in LOCAL_MODEL_PATTERNS:
        if pattern in model_lower:
            return True

    # Check for Ollama-style tag suffixes (:latest, :7b, :q4_0, etc.)
    # This catches custom local models like "my-model:latest"
    if ':' in model_lower:
        # Has a tag suffix - likely an Ollama model
        # But exclude known cloud models with version specifiers
        cloud_prefixes = ('gpt-', 'claude-', 'gemini-', 'o1-', 'openai/')
        if not any(model_lower.startswith(prefix) for prefix in cloud_prefixes):
            return True

    return False


def is_aither_model(model_name):
    """Check if a model name uses AitherLLM gateway (aither/ prefix)."""
    if not model_name:
        return False
    return model_name.lower().startswith("aither/")


def is_aitherllm_available():
    """Check if AitherLLM gateway is running (cached for 30s)."""
    global _aitherllm_available_cache
    now = time.time()
    if now - _aitherllm_available_cache["timestamp"] < _CACHE_TTL:
        return _aitherllm_available_cache["available"]

    try:
        resp = requests.get(f"{AITHERLLM_URL}/health", timeout=1)
        available = resp.status_code == 200
    except requests.RequestException:
        available = False

    _aitherllm_available_cache = {"available": available, "timestamp": now}
    return available


def get_aitherllm_models():
    """Fetch available models from AitherLLM gateway."""
    try:
        resp = requests.get(f"{AITHERLLM_URL}/models", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for model in data.get("models", []):
                # Add aither/ prefix for routing
                name = model.get("id") or model.get("name")
                if name:
                    models.append(f"aither/{name}")
            return models
    except Exception as exc:
        logger.debug(f"AitherLLM models fetch failed: {exc}")
    return []


def is_nvidia_model(model_name):
    """Check if a model name is an NVIDIA NIM model."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return (
        model_lower.startswith("nvidia/") or
        model_lower.startswith("meta/") or
        model_lower.startswith("mistralai/") or
        model_lower in NVIDIA_MODELS or
        "orchestrator" in model_lower or
        "nemotron" in model_lower
    )


def get_orchestrator_model():
    """Get the NVIDIA Orchestrator model name."""
    return NVIDIA_ORCHESTRATOR_MODEL

def _check_cli_for_local_model():
    """Check if --model argument specifies a local model."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            return is_local_model(args[i + 1])
        elif arg.startswith("--model="):
            return is_local_model(arg.split("=", 1)[1])
    return False

def get_available_models(use_local_models=False, local_model_name=None, include_nvidia=True, include_aither=True):
    """Fetches available models dynamically.

    Priority order:
    1. AitherLLM gateway models (unified access with fallback)
    2. NVIDIA Orchestrator-8B (if available) - Best for tool calling
    3. Gemini models (cloud)
    4. Ollama local models
    """
    default_models = []

    # Add AitherLLM gateway models first (unified access)
    if include_aither and is_aitherllm_available():
        aither_models = get_aitherllm_models()
        if aither_models:
            default_models.extend(aither_models)
        else:
            # Add known AitherLLM models if we can't fetch the list
            default_models.extend([
                "aither/nvidia-orchestrator",
                "aither/mistral-nemo",
                "aither/gemini-2.5-flash",
                "aither/claude-sonnet-4",
            ])

    # Add NVIDIA Orchestrator as primary option if available
    if include_nvidia and is_nvidia_available():
        default_models.append(NVIDIA_ORCHESTRATOR_MODEL)
        # Add other NVIDIA models
        for alias, model_name in NVIDIA_MODELS.items():
            if model_name not in default_models:
                default_models.append(model_name)

    # Add Gemini models - prefer stable tool-calling models first
    # Note: gemini-3-pro-preview has UNEXPECTED_TOOL_CALL issues, deprioritized
    gemini_models = [
        "gemini-2.5-pro",                 # Best for tool calling
        "gemini-2.5-flash",               # Fast flash model
        "gemini-2.0-flash",               # Reliable fallback
        "gemini-3-pro-preview",           # Latest but has tool issues
    ]
    default_models.extend(gemini_models)

    # Add local model if configured
    if use_local_models and local_model_name:
        default_models.insert(0, local_model_name)

    # Try to fetch Ollama models (uses cached result for 30s)
    ollama_models = _get_cached_ollama_models()
    for name in ollama_models:
        if name and name not in default_models:
            default_models.append(name)

    # Skip Google API client initialization if using local models only
    if use_local_models and local_model_name:
        return default_models

    # Also skip if CLI specifies a local model (fast path for local-only usage)
    if _check_cli_for_local_model():
        return default_models

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return default_models

    # Skip dynamic model fetching if SKIP_GOOGLE_MODEL_FETCH is set (faster startup)
    # Also skip if AITHERNODE_LAZY_LOAD is set (AitherNode already running)
    if os.environ.get("SKIP_GOOGLE_MODEL_FETCH", "").lower() == "true":
        return default_models
    if os.environ.get("AITHERNODE_LAZY_LOAD", "").lower() in ("1", "true", "yes"):
        return default_models

    try:
        from google.genai import Client
        client = Client(api_key=api_key)
        models = []
        # List models that support generateContent
        for m in client.models.list():
            if "generateContent" in m.supported_generation_methods:
                 # Filter for gemini models to keep the list clean
                 if "gemini" in m.name.lower():
                    name = m.name.split("/")[-1] # remove models/ prefix if present
                    models.append(name)

        if models:
            # Sort to put newer models first (heuristic)
            models.sort(reverse=True)
            # Add local models to the list
            for m in default_models:
                if m not in models and "gemini" not in m:
                     models.append(m)
            return models
    except Exception:
        # Don't print here to avoid cluttering output if it fails silently
        pass

    return default_models

def get_fallback_model(current_model):
    """Returns a fallback model if the current one fails.

    Fallback priority:
    1. Local orchestrator (if available) - lowest latency
    2. Other local models (mistral-nemo)
    3. Cloud models (gemini-2.5-pro > flash > 2.0-flash)
    """
    # First, check if we have a local orchestrator available as ultimate fallback
    local_orchestrator = None
    ollama_models = _get_cached_ollama_models()
    for name in ollama_models:
        if "aither-orchestrator" in name or "orchestrator-8b" in name:
            local_orchestrator = f"ollama/{name.split(':')[0]}"
            break

    # AitherLLM models - they handle fallback internally, but provide a chain
    if is_aither_model(current_model):
        base_model = current_model.replace("aither/", "")
        if "nvidia" in base_model or "orchestrator" in base_model:
            return "aither/mistral-nemo"
        elif "mistral" in base_model:
            return "aither/gemini-2.5-pro"
        elif "gemini" in base_model:
            return "aither/claude-sonnet-4"
        elif "claude" in base_model:
            return "aither/gpt-4o"
        return local_orchestrator or "gemini-2.5-pro"  # Prefer local if available

    # NVIDIA model fallbacks
    if is_nvidia_model(current_model):
        # Try local orchestrator first, then other NVIDIA, then Gemini
        if local_orchestrator:
            return local_orchestrator
        if "Orchestrator" in current_model:
            return "nvidia/llama-3.3-nemotron-super-49b-v1"
        return "gemini-2.5-pro"

    # Local model fallbacks - try other local models first
    if is_local_model(current_model):
        if "orchestrator" in current_model.lower():
            # Orchestrator failed, try mistral-nemo locally
            if "mistral-nemo" in ollama_models:
                return "ollama/mistral-nemo"
        return "gemini-2.5-pro"  # Fall to cloud

    # Gemini model fallbacks - stable cloud chain
    if "gemini-3" in current_model:
        return "gemini-2.5-pro"
    elif "gemini-2.5-pro" in current_model:
        return "gemini-2.5-flash"
    elif "gemini-2.5-flash" in current_model:
        return "gemini-2.0-flash"
    elif "gemini-2.0" in current_model:
        return local_orchestrator or "gemini-2.5-pro"  # Try local before cycling

    return local_orchestrator or "gemini-2.5-pro"  # Ultimate fallback


def _get_cached_ollama_models() -> list:
    """Get Ollama models with caching (30s TTL)."""
    global _ollama_models_cache
    now = time.time()
    if now - _ollama_models_cache["timestamp"] < _CACHE_TTL:
        return _ollama_models_cache["models"]

    try:
        # FROM services.yaml (SINGLE SOURCE OF TRUTH)
        from lib.core.AitherPorts import ollama_url as get_ollama_url
        ollama_url = get_ollama_url()
        resp = requests.get(f"{ollama_url}/api/tags", timeout=1)
        if resp.status_code == 200:
            models = [m.get("name", "").lower() for m in resp.json().get("models", [])]
            _ollama_models_cache = {"models": models, "timestamp": now}
            return models
    except Exception as exc:
        logger.debug(f"Ollama models fetch failed: {exc}")

    _ollama_models_cache = {"models": [], "timestamp": now}
    return []


def invalidate_model_cache():
    """
    Force refresh of all model caches. Call this when you know models have changed
    (e.g., after pulling a new model or starting a server).
    """
    global _ollama_models_cache, _orchestrator_model_cache, _aitherllm_available_cache
    _ollama_models_cache = {"models": [], "timestamp": 0}
    _orchestrator_model_cache = {"model": None, "timestamp": 0}
    _aitherllm_available_cache = {"available": None, "timestamp": 0}


def get_best_orchestration_model():
    """
    Get the best available model for orchestration/tool calling tasks.

    Priority: Local aither-orchestrator-8b > NVIDIA Cloud > Gemini 2.5 Pro
    Results are cached for fast repeated calls.
    """
    global _orchestrator_model_cache
    now = time.time()

    # Return cached result if fresh
    if now - _orchestrator_model_cache["timestamp"] < _CACHE_TTL and _orchestrator_model_cache["model"]:
        return _orchestrator_model_cache["model"]

    result = None

    # First, check for local orchestrator model in Ollama (preferred - lowest latency)
    model_names = _get_cached_ollama_models()
    for name in model_names:
        if "aither-orchestrator" in name or "orchestrator-8b" in name:
            result = f"ollama/{name.split(':')[0]}"  # Return without tag suffix
            break

    # Fall back to NVIDIA cloud API if available
    if not result and is_nvidia_available():
        result = NVIDIA_ORCHESTRATOR_MODEL

    # Final fallback to Gemini
    if not result:
        result = "gemini-2.5-pro"

    _orchestrator_model_cache = {"model": result, "timestamp": now}
    return result

def select_model(models_list, provided_model=None):
    """
    Selects a model from the available list.
    If provided_model is valid, returns it.
    Otherwise, prompts the user to choose.
    """
    if provided_model:
        if provided_model in models_list:
            safe_print(Panel(f"[green]Using provided model:[/green] {provided_model}", title="Model Selection"))
            return provided_model
        else:
            safe_print(f"[yellow]Warning:[/yellow] Provided model '{provided_model}' not found in available models.")

    safe_print(Panel("[bold cyan]Available Models:[/bold cyan]", title="Model Selection"))
    for idx, model in enumerate(models_list):
        print(f"{idx + 1}. {model}")

    while True:
        choice = Prompt.ask("Select a model number", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models_list):
                selected = models_list[idx]
                safe_print(f"[green]Selected model:[/green] {selected}")
                return selected
            else:
                safe_print("[red]Invalid selection. Please try again.[/red]")
        except ValueError:
            safe_print("[red]Please enter a number.[/red]")
