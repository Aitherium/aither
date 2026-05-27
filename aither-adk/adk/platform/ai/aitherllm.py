"""
AitherLLM Integration for CLI Agents

This module provides a Google ADK-compatible LLM implementation that routes
through MicroScheduler (port 8150) instead of direct Ollama calls.

Benefits:
- Unified model routing (nvidia-orchestrator, ollama, gemini, claude, openai)
- Automatic fallback chain
- Consistent behavior between CLI and UI agents

Usage:
    from aither_adk.ai.aitherllm import AitherLlm
    from google.adk.models.registry import LLMRegistry  # stub

    LLMRegistry.register(AitherLlm)  # auto-infers "aither" from supported_models()

    agent = Agent(
        name="MyAgent",
        model="aither/nvidia-orchestrator",  # Uses vLLM
        # model="aither/mistral-nemo",       # Uses Ollama
        # model="aither/gemini-2.5-flash",   # Uses Google API
    )
"""

import json
import os
from typing import AsyncGenerator, Optional

import aiohttp
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# AitherLLM endpoint
AITHERLLM_URL = os.getenv("AITHERLLM_URL", "http://localhost:8150")

# MicroScheduler for centralized VRAM management
try:
    # Try multiple import paths
    try:
        from lib.agents.MicroSchedulerClient import MicroSchedulerClient
        _microscheduler = MicroSchedulerClient()
        MICROSCHEDULER_AVAILABLE = True
    except ImportError:
        from AitherOS.lib.agents.MicroSchedulerClient import MicroSchedulerClient
        _microscheduler = MicroSchedulerClient()
        MICROSCHEDULER_AVAILABLE = True
except ImportError:
    MICROSCHEDULER_AVAILABLE = False
    _microscheduler = None

import logging

logger = logging.getLogger(__name__)


class AitherLlm(BaseLlm):
    """
    AitherLLM integration for Google ADK.

    Routes all LLM requests through AitherLLM unified gateway,
    which handles model routing, fallbacks, and API normalization.

    Model naming convention: "aither/<model-id>"
    Examples:
        - aither/nvidia-orchestrator  (vLLM backend)
        - aither/mistral-nemo         (Ollama backend)
        - aither/gemini-2.5-flash     (Google API)
        - aither/claude-sonnet-4      (Anthropic API)
        - aither/gpt-4o-mini          (OpenAI API)
    """
    model: str
    _session: Optional[aiohttp.ClientSession] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def supported_models(cls) -> list[str]:
        """Match any model prefixed with 'aither/'."""
        return [r"aither/.*"]

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_model_id(self) -> str:
        """Extract model ID from aither/ prefix."""
        return self.model.replace("aither/", "")

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate content via AitherLLM gateway."""
        model_id = self._get_model_id()

        # Build messages array
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
            # Map 'model' to 'assistant'
            if role == "model":
                role = "assistant"
            if role == "function":
                role = "user"  # Tool responses as user role for compatibility

            text_parts = []
            for p in content.parts:
                if p.text:
                    text_parts.append(p.text)
                elif p.function_response:
                    # Format function response
                    resp = p.function_response
                    result = resp.response if hasattr(resp, "response") else {}
                    text_parts.append(f"[Tool Result: {resp.name}]\n{json.dumps(result, indent=2)}")
                elif p.function_call:
                    # Format function call
                    fc = p.function_call
                    args = fc.args if hasattr(fc, "args") else {}
                    text_parts.append(f"[Tool Call: {fc.name}({json.dumps(args)})]\n")

            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

        # Build tools list for function calling
        tools = None
        if llm_request.config and llm_request.config.tools:
            tools = []
            for tool_list in llm_request.config.tools:
                if hasattr(tool_list, "function_declarations"):
                    for func in tool_list.function_declarations:
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": func.name,
                                "description": func.description or "",
                                "parameters": func.parameters or {"type": "object", "properties": {}},
                            }
                        })

        # Request payload
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": getattr(llm_request.config, "temperature", 0.7) if llm_request.config else 0.7,
            "max_tokens": getattr(llm_request.config, "max_output_tokens", 2048) if llm_request.config else 2048,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        # Try centralized queue first if not streaming
        # This prevents multiple agents from OOMing the GPU
        if MICROSCHEDULER_AVAILABLE and _microscheduler and not stream:
            try:
                # Use normal priority for agents
                ms_resp = await _microscheduler.llm_chat(
                    model=model_id,
                    messages=messages,
                    temperature=payload["temperature"],
                    max_tokens=payload["max_tokens"],
                    tools=payload.get("tools"),
                    priority="normal",
                    source="agent_adk"
                )

                if ms_resp and 'choices' in ms_resp:
                    data = ms_resp
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # Check for tool calls in response
                    tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

                    parts = []
                    if content:
                        parts.append(types.Part(text=content))

                    for tc in tool_calls:
                        if tc.get("type") == "function":
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
                        content=types.Content(
                            role="model",
                            parts=parts
                        )
                    )
                    return
            except Exception:
                # Log usage of direct fallback
                pass

        session = await self._get_session()

        try:
            async with session.post(
                f"{AITHERLLM_URL}/v1/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part(text=f"[AitherLLM Error {response.status}]: {error_text}")]
                        )
                    )
                    return

                if stream:
                    full_text = ""
                    async for line in response.content:
                        if line:
                            text = line.decode("utf-8").strip()
                            if text.startswith("data: "):
                                text = text[6:]
                            if text and text != "[DONE]":
                                try:
                                    data = json.loads(text)
                                    delta = data.get("choices", [{}])[0].get("delta", {})
                                    chunk = delta.get("content", "")
                                    if chunk:
                                        full_text += chunk
                                        yield LlmResponse(
                                            content=types.Content(
                                                role="model",
                                                parts=[types.Part(text=chunk)]
                                            ),
                                            partial=True
                                        )
                                except json.JSONDecodeError:
                                    pass

                    # Final response
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part(text=full_text)]
                        ),
                        partial=False
                    )
                else:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # Check for tool calls in response
                    tool_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

                    parts = []
                    if content:
                        parts.append(types.Part(text=content))

                    for tc in tool_calls:
                        if tc.get("type") == "function":
                            func = tc.get("function", {})
                            parts.append(types.Part(
                                function_call=types.FunctionCall(
                                    name=func.get("name", ""),
                                    args=json.loads(func.get("arguments", "{}"))
                                )
                            ))

                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=parts
                        )
                    )

        except aiohttp.ClientConnectorError:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="[Offline] MicroScheduler not running. Start: python -m services.orchestration.AitherMicroScheduler")]
                )
            )
        except Exception as e:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"[AitherLLM Error]: {str(e)}")]
                )
            )


async def check_aitherllm_health() -> dict:
    """Check AitherLLM health status."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{AITHERLLM_URL}/health") as response:
                if response.status == 200:
                    return await response.json()
    except Exception as exc:
        logger.debug(f"AitherLLM health check failed: {exc}")

    return {
        "status": "offline",
        "backends": {},
        "defaultModel": None
    }


async def get_available_models() -> list[dict]:
    """Get list of available models from AitherLLM."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{AITHERLLM_URL}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
    except Exception as exc:
        logger.debug(f"AitherLLM model listing failed: {exc}")

    return []
