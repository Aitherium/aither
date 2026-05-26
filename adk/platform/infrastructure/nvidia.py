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
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from aither_adk.ui.console import safe_print

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


class NvidiaLlm(BaseLlm):
    """
    NVIDIA NIM LLM implementation for Google ADK.

    Supports both cloud API and local NIM containers.
    Optimized for Orchestrator-8B's tool calling capabilities.
    """
    model: str

    @classmethod
    def supported_models(cls) -> list[str]:
        """Models matching nvidia/* pattern are handled by this provider."""
        return [r"nvidia/.*", r"meta/.*", r"mistralai/.*"]

    def _get_base_url(self) -> str:
        """Get the appropriate base URL for API calls."""
        # Check for local NIM container first
        local_url = os.getenv("NVIDIA_NIM_URL", "").strip()
        if local_url:
            return local_url.rstrip("/")

        # Fall back to cloud API
        return "https://integrate.api.nvidia.com"

    def _get_api_key(self) -> Optional[str]:
        """Get NVIDIA API key for cloud API."""
        return os.getenv("NVIDIA_API_KEY", "").strip() or None

    def _is_local_nim(self) -> bool:
        """Check if we're using a local NIM container."""
        return bool(os.getenv("NVIDIA_NIM_URL", "").strip())

    def _build_messages(self, llm_request: LlmRequest) -> List[Dict[str, Any]]:
        """Convert ADK request to OpenAI-compatible messages format."""
        messages = []

        # System instruction
        if llm_request.config and llm_request.config.system_instruction:
            sys_inst = llm_request.config.system_instruction
            sys_text = ""
            if hasattr(sys_inst, "parts"):
                sys_text = "".join([p.text for p in sys_inst.parts if p.text])
            elif isinstance(sys_inst, list):
                sys_text = "".join([p.text for p in sys_inst if hasattr(p, "text") and p.text])
            elif isinstance(sys_inst, str):
                sys_text = sys_inst

            if sys_text:
                messages.append({"role": "system", "content": sys_text})

        # Chat history
        for content in llm_request.contents:
            role = content.role
            if role == "model":
                role = "assistant"
            elif role == "function":
                role = "tool"

            text_parts = []
            tool_calls_in_msg = []
            tool_response_id = None

            for p in content.parts:
                if p.text:
                    text_parts.append(p.text)
                elif p.function_call:
                    # Assistant made a tool call
                    tool_calls_in_msg.append({
                        "id": f"call_{p.function_call.name}_{len(tool_calls_in_msg)}",
                        "type": "function",
                        "function": {
                            "name": p.function_call.name,
                            "arguments": json.dumps(p.function_call.args) if isinstance(p.function_call.args, dict) else str(p.function_call.args)
                        }
                    })
                elif p.function_response:
                    # Tool response
                    tool_response_id = f"call_{p.function_response.name}_0"
                    resp = p.function_response.response
                    if isinstance(resp, dict):
                        text_parts.append(json.dumps(resp))
                    else:
                        text_parts.append(str(resp))

            msg = {"role": role, "content": "".join(text_parts) or None}

            if tool_calls_in_msg:
                msg["tool_calls"] = tool_calls_in_msg
                msg["content"] = None  # OpenAI format: content is null when tool_calls present

            if role == "tool" and tool_response_id:
                msg["tool_call_id"] = tool_response_id

            messages.append(msg)

        return messages

    def _build_tools(self, llm_request: LlmRequest) -> List[Dict[str, Any]]:
        """Convert ADK tools to OpenAI-compatible format."""
        tools = []

        if not (llm_request.config and llm_request.config.tools):
            return tools

        for tool in llm_request.config.tools:
            if hasattr(tool, 'function_declarations'):
                for func in tool.function_declarations:
                    # Convert Schema
                    def convert_schema(schema):
                        if not schema:
                            return {"type": "object", "properties": {}}

                        type_map = {
                            types.Type.STRING: "string",
                            types.Type.NUMBER: "number",
                            types.Type.INTEGER: "integer",
                            types.Type.BOOLEAN: "boolean",
                            types.Type.ARRAY: "array",
                            types.Type.OBJECT: "object"
                        }
                        t = type_map.get(schema.type, "string")
                        res = {"type": t}

                        if schema.description:
                            res["description"] = schema.description
                        if schema.properties:
                            res["properties"] = {k: convert_schema(v) for k, v in schema.properties.items()}
                        if schema.required:
                            res["required"] = schema.required
                        if schema.items:
                            res["items"] = convert_schema(schema.items)
                        if schema.enum:
                            res["enum"] = schema.enum

                        return res

                    tools.append({
                        "type": "function",
                        "function": {
                            "name": func.name,
                            "description": func.description or "",
                            "parameters": convert_schema(func.parameters)
                        }
                    })

        return tools

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate content using NVIDIA NIM API."""

        base_url = self._get_base_url()
        api_key = self._get_api_key()
        is_local = self._is_local_nim()

        # For cloud API, require API key
        if not is_local and not api_key:
            raise ValueError("NVIDIA_API_KEY required for cloud API. Set NVIDIA_NIM_URL for local NIM.")

        # Build request
        messages = self._build_messages(llm_request)
        tools = self._build_tools(llm_request)

        # Resolve model alias
        model = NVIDIA_MODELS.get(self.model.lower(), self.model)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.6,  # Lower temp for orchestration accuracy
            "max_tokens": 4096,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Build headers
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Determine endpoint
        if is_local:
            endpoint = f"{base_url}/v1/chat/completions"
        else:
            endpoint = f"{base_url}/v1/chat/completions"

        timeout = float(os.getenv("NVIDIA_TIMEOUT", "120"))

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if stream:
                    async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                        response.raise_for_status()

                        full_content = ""
                        tool_calls = []

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue

                            data = line[6:]  # Remove "data: " prefix
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})

                                # Handle content
                                if delta.get("content"):
                                    full_content += delta["content"]
                                    yield LlmResponse(
                                        content=types.Content(
                                            role="model",
                                            parts=[types.Part(text=delta["content"])]
                                        ),
                                        partial=True,
                                        turn_complete=False
                                    )

                                # Handle tool calls
                                if delta.get("tool_calls"):
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)
                                        while len(tool_calls) <= idx:
                                            tool_calls.append({"name": "", "arguments": ""})

                                        if tc.get("function", {}).get("name"):
                                            tool_calls[idx]["name"] = tc["function"]["name"]
                                        if tc.get("function", {}).get("arguments"):
                                            tool_calls[idx]["arguments"] += tc["function"]["arguments"]

                            except json.JSONDecodeError:
                                continue

                        # Final response
                        parts = []
                        if full_content:
                            parts.append(types.Part(text=full_content))

                        for tc in tool_calls:
                            if tc["name"]:
                                try:
                                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                                except json.JSONDecodeError:
                                    args = {}

                                parts.append(types.Part(
                                    function_call=types.FunctionCall(
                                        name=tc["name"],
                                        args=args
                                    )
                                ))

                        if parts:
                            yield LlmResponse(
                                content=types.Content(role="model", parts=parts),
                                partial=False,
                                turn_complete=True,
                                finish_reason=types.FinishReason.STOP
                            )

                else:
                    # Non-streaming
                    response = await client.post(endpoint, json=payload, headers=headers)
                    response.raise_for_status()

                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    parts = []

                    if message.get("content"):
                        parts.append(types.Part(text=message["content"]))

                    for tc in message.get("tool_calls", []):
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}

                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=func.get("name", ""),
                                args=args
                            )
                        ))

                    yield LlmResponse(
                        content=types.Content(role="model", parts=parts),
                        partial=False,
                        turn_complete=True,
                        finish_reason=types.FinishReason.STOP
                    )

        except httpx.HTTPStatusError as e:
            safe_print(f"[red]NVIDIA API Error: {e.response.status_code} - {e.response.text}[/]")
            raise ValueError(f"NVIDIA API request failed: {e}")
        except Exception as e:
            safe_print(f"[red]NVIDIA Error: {e}[/]")
            raise ValueError(f"NVIDIA generation failed: {e}")


class OrchestratorLlm(NvidiaLlm):
    """
    Specialized LLM class for Orchestrator-8B.

    Includes optimizations for tool calling and structured output.
    Uses lower temperature and specific prompting strategies.
    """

    @classmethod
    def supported_models(cls) -> list[str]:
        """Only handle Orchestrator model specifically."""
        return [r"nvidia/Orchestrator.*", r"orchestrator.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate with Orchestrator-specific optimizations."""

        # Force model to Orchestrator-8B
        self.model = NVIDIA_ORCHESTRATOR_MODEL

        # Use parent implementation with overrides applied
        async for response in super().generate_content_async(llm_request, stream):
            yield response


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
