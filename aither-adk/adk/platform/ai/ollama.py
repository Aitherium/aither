import ast
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict

import requests

from adk.platform.ui.console import safe_print

logger = logging.getLogger(__name__)

# ===============================================================================
# PERFORMANCE CACHES & CONFIGURATION
# ===============================================================================
_TOOL_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_PRELOADED: set = set()  # Track which models are already loaded in Ollama

# Adaptive context sizes for quality/speed tradeoff
CONTEXT_SIZES = {
    "minimal": 2048,    # For simple queries, tool calls
    "standard": 4096,   # Default - good balance
    "extended": 8192,   # For complex reasoning, longer contexts
    "maximum": 16384,   # For full document analysis
}

# Default keep_alive - keeps model warm for faster subsequent calls
DEFAULT_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")


def preload_model(model_name: str, base_url: str = None, keep_alive: str = None) -> bool:
    """
    Preload a model into Ollama's memory to eliminate cold start latency.
    base_url defaults to services.yaml value.
    Call this during agent startup while user sees the banner.

    Args:
        model_name: Model to preload (with or without ollama/ prefix)
        base_url: Ollama server URL (defaults to services.yaml value)
        keep_alive: How long to keep in memory (default: 5m). Use "0" to unload.

    Returns True if successful, False otherwise.
    """
    if base_url is None:
        try:
            from adk.ports import ollama_url
            base_url = ollama_url()
        except ImportError:
            raise ImportError("Cannot import AitherPorts. Ensure services.yaml is available.")
    if model_name in _MODEL_PRELOADED:
        return True

    keep_alive = keep_alive or DEFAULT_KEEP_ALIVE

    try:
        # Send a minimal request to load the model with keep_alive
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model_name.replace("ollama/", ""),
                "messages": [{"role": "user", "content": "."}],
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_ctx": 512, "num_predict": 1}  # Minimal context for fast load
            },
            timeout=60  # Allow more time for first load
        )
        if response.ok:
            _MODEL_PRELOADED.add(model_name)
            return True
    except Exception as exc:
        logger.debug(f"Model preload failed: {exc}")
    return False


def estimate_context_size(messages: list, tools_count: int = 0) -> int:
    """
    Estimate optimal context size based on conversation length and complexity.
    Returns an appropriate num_ctx value.
    """
    # Estimate total tokens (rough: ~4 chars per token)
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    estimated_tokens = total_chars // 4

    # Add overhead for tools
    if tools_count > 0:
        estimated_tokens += tools_count * 200  # ~200 tokens per tool definition

    # Select appropriate context size
    if estimated_tokens < 1000:
        return CONTEXT_SIZES["minimal"]
    elif estimated_tokens < 3000:
        return CONTEXT_SIZES["standard"]
    elif estimated_tokens < 6000:
        return CONTEXT_SIZES["extended"]
    else:
        return CONTEXT_SIZES["maximum"]


def get_optimal_temperature(task_type: str = "general") -> float:
    """
    Get optimal temperature based on task type for quality/creativity balance.

    Args:
        task_type: One of "tool_calling", "code", "creative", "factual", "general"

    Returns:
        Optimal temperature value
    """
    temperatures = {
        "tool_calling": 0.3,  # Low - need precise, consistent tool calls
        "code": 0.4,          # Low-medium - accuracy matters, some flexibility
        "factual": 0.5,       # Medium - factual but allow natural phrasing
        "general": 0.7,       # Default - balanced
        "creative": 0.9,      # High - more variety and creativity
        "brainstorm": 1.0,    # Maximum - explore diverse options
    }
    return temperatures.get(task_type, 0.7)

# Models that DON'T support native Ollama tools API (will use [TOOL_CALLS] format)
# These models will get tools injected into system prompt instead
_MODELS_WITHOUT_NATIVE_TOOLS = {
    "aither-orchestrator-8b",
    "aither-orchestrator-8b-v4",
    "aither-orchestrator-8b:v2",
    "aither-orchestrator-8b:latest",
    "orchestrator-8b",
    "mistral-nemo",
    "qwen3",  # Base qwen3 models need prompt-based tools
}

def _model_supports_native_tools(model_name: str) -> bool:
    """Check if model supports Ollama's native tool calling API."""
    base_name = model_name.split(":")[0].lower()
    return base_name not in _MODELS_WITHOUT_NATIVE_TOOLS

# NOTE: The google-adk OllamaLlm provider was removed. Use adk.llm.LLMRouter
# (provider="ollama") for the native, google-free Ollama backend.
