"""
NVIDIA NIM / vLLM LLM Provider for AitherOS Agents

Supports:
- NVIDIA API Cloud (api.nvidia.com) - requires NVIDIA_API_KEY
- Local vLLM Server (localhost:8116) - for local GPU inference
- Local NIM Container (localhost:8116) - for self-hosted Orchestrator-8B

The Orchestrator-8B model is optimized for:
- Tool/function calling with high accuracy
- Structured JSON output
- Multi-step reasoning and planning
- Efficient orchestration of complex workflows

Usage:
    # Cloud API
    export NVIDIA_API_KEY="nvapi-xxx"
    model = "nvidia/llama-3.3-nemotron-super-49b-v1"  # Or other NIM models

    # Local vLLM Server (RECOMMENDED for local GPU)
    # Start vLLM:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model nvidia/Orchestrator-8B \
    #       --port 8116 \
    #       --gpu-memory-utilization 0.90 \
    #       --enable-auto-tool-choice \
    #       --tool-call-parser hermes
    #
    # Then set:
    export NVIDIA_NIM_URL="http://localhost:8116"
    model = "nvidia/Orchestrator-8B"

    # Local NIM Container (alternative)
    export NVIDIA_NIM_URL="http://localhost:8116"
    model = "nvidia/Orchestrator-8B"
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from adk.platform.ui.console import safe_print

logger = logging.getLogger(__name__)

# NVIDIA NIM Model Registry
NVIDIA_MODELS = {
    # === AITHER TIER MODELS (Nemotron Architecture) ===
    # Tier 0 - Reflex/Neuron: Fast responses, neurons
    "aither-reflex": "llama3.2",
    "nemotron-nano": "nvidia/Nemotron-Nano-9B-v2",
    # Tier 1 - Router: Meta-routing, orchestration
    "aither-router": "nvidia/Nemotron-Orchestrator-8B",
    "orchestrator": "nvidia/Nemotron-Orchestrator-8B",
    "orchestrator-8b": "nvidia/Nemotron-Orchestrator-8B",
    # Tier 2 - Agent: Tool calling, MoE (30B with 3.6B active)
    "aither-agent": "nvidia/Nemotron-3-Nano-30B-A3B-FP8",
    # Tier 3 - Reasoning: DeepSeek R1 14B for deep analysis
    "aither-reasoning": "deepseek-r1:14b",

    # === LARGE NEMOTRON MODELS ===
    "nemotron-super": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",

    # === META LLAMA via NIM ===
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta/llama-3.1-405b-instruct",

    # === MISTRAL via NIM ===
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1",
}

# Speculative decoding: Use aither-reflex as draft model for 2-3x speedup
# vLLM supports: --speculative-model nvidia/Nemotron-Nano-9B-v2 --num-speculative-tokens 5

# Default models for each tier
NVIDIA_ORCHESTRATOR_MODEL = "nvidia/Nemotron-Orchestrator-8B"
NVIDIA_REFLEX_MODEL = "llama3.2"  # Fast reflex/neuron model


# NOTE: NvidiaLlm/OrchestratorLlm (google-adk providers) removed. NVIDIA NIM is
# OpenAI-compatible: use adk.llm.LLMRouter(provider="openai", base_url=NVIDIA_NIM_URL).


def get_orchestrator_model() -> str:
    """Get the default orchestrator model name."""
    return NVIDIA_ORCHESTRATOR_MODEL


def is_nvidia_available() -> bool:
    """Check if NVIDIA NIM/vLLM is available (cloud or local)."""
    return bool(os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_URL"))


def is_local_vllm() -> bool:
    """Check if we're configured to use a local vLLM/NIM server."""
    return bool(os.getenv("NVIDIA_NIM_URL"))


async def check_vllm_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Check health of local vLLM or NIM server.

    Works with both vLLM and NIM containers by checking multiple endpoints.
    """
    url = base_url or os.getenv("NVIDIA_NIM_URL", "http://localhost:8116")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try /health first (vLLM)
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return {"status": "healthy", "url": url, "backend": "vllm"}
            except Exception as exc:
                logger.debug(f"vLLM health check failed: {exc}")

            # Try /v1/health/ready (NIM)
            try:
                response = await client.get(f"{url}/v1/health/ready")
                if response.status_code == 200:
                    return {"status": "healthy", "url": url, "backend": "nim"}
            except Exception as exc:
                logger.debug(f"NIM health check failed: {exc}")

            # Try /v1/models (both vLLM and NIM support this)
            response = await client.get(f"{url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = [m["id"] for m in data.get("data", [])]
                return {
                    "status": "healthy",
                    "url": url,
                    "backend": "vllm/nim",
                    "models": models
                }

            return {"status": "unhealthy", "url": url, "code": response.status_code}

    except Exception as e:
        return {"status": "unavailable", "url": url, "error": str(e)}


# Alias for backwards compatibility
check_nim_health = check_vllm_health


def get_vllm_startup_command(
    model: str = NVIDIA_ORCHESTRATOR_MODEL,
    port: int = 8116,
    gpu_memory: float = 0.90,
    max_model_len: int = 8192
) -> str:
    """
    Get the command to start a local vLLM server.

    Useful for displaying instructions to users.
    """
    return f"""python -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --port {port} \\
    --gpu-memory-utilization {gpu_memory} \\
    --max-model-len {max_model_len} \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes"""
