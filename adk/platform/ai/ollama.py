import ast
import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict

import requests
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from aither_adk.ui.console import safe_print

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
            from lib.core.AitherPorts import ollama_url
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


class OllamaLlm(BaseLlm):
    """Ollama LLM implementation for ADK."""
    model: str

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"ollama/.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Extract model name (remove ollama/ prefix)
        ollama_model = self.model.replace("ollama/", "")

        # Try to use resource manager for GPU-aware execution
        try:
            from aither_adk.infrastructure.resource_manager import (
                RequestPriority,
                ResourceType,
                get_resource_manager,
            )

            from .resource_integration import route_llm_inference

            # Get config path - os is already imported at module level
            config_path = None
            # Try to find config in common locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "Saga", "config", "resource_config.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "agents", "Saga", "config", "resource_config.json"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

            get_resource_manager(config_path)

            # Check if we should route to cloud instead
            use_local, routing_reason = await route_llm_inference(
                self.model,
                prefer_local=True,
                config_path=config_path
            )

            if not use_local and routing_reason:
                # GPU is busy/low memory, but we're already using Ollama
                # Log warning but proceed (Ollama will handle queuing)
                import sys
                if hasattr(sys, 'stderr'):
                    print(f"[WARN] GPU resource warning: {routing_reason}", file=sys.stderr)
        except ImportError:
            # Resource manager not available, proceed normally
            pass

        # Build messages
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

             # [CRITICAL] Append explicit tool calling instructions for local models
             # Local models often don't use native tool calling even when tools are provided
             if sys_text and llm_request.config and llm_request.config.tools:
                 tool_reminder = """

[TOOL CALLING FORMAT - IMPORTANT]
You have tools available. When you need to call a tool, output EXACTLY this format:
[TOOL_CALLS] {"name": "tool_name", "arguments": {"arg1": "value1"}}

CRITICAL: For current time, weather, news, or any real-time data:
- Current time -> [TOOL_CALLS] {"name": "get_current_time", "arguments": {}}
- Web search -> [TOOL_CALLS] {"name": "web_search", "arguments": {"query": "your search query"}}
- Weather -> [TOOL_CALLS] {"name": "get_weather", "arguments": {"location": "city name"}}

DO NOT answer questions about current events (president, elections, news, time, weather) without calling the appropriate tool first. Your training data is outdated."""
                 sys_text = sys_text + tool_reminder

             if sys_text:
                 messages.append({"role": "system", "content": sys_text})

        # Chat history
        for content in llm_request.contents:
            role = content.role
            # Map 'model' to 'assistant' for Ollama
            if role == "model":
                role = "assistant"

            # Handle function responses in history
            tool_response_name = None
            if role == "function":
                # [FIX] Use 'user' role for tool outputs to avoid ID mismatch issues with local models
                # Many local models (Mistral, Llama) get confused by 'tool' role without IDs
                role = "user"

            text_parts = []
            for p in content.parts:
                if p.text:
                    text_parts.append(p.text)
                elif p.function_response:
                    # Extract response for tool role
                    tool_response_name = p.function_response.name
                    resp = p.function_response.response

                    # Convert dict to JSON string for better model comprehension
                    if isinstance(resp, dict):
                        if 'result' in resp:
                            try:
                                val = json.dumps(resp['result'])
                            except (TypeError, ValueError):
                                val = str(resp['result'])
                        else:
                            try:
                                val = json.dumps(resp)
                            except (TypeError, ValueError):
                                val = str(resp)
                    else:
                        val = str(resp)

                    # Add explicit header for the model to understand this is tool output
                    text_parts.append(f"[TOOL OUTPUT: {tool_response_name}]\n{val}")

            text = "".join(text_parts)

            # Handle function calls in history (if any)
            tool_calls = []
            for p in content.parts:
                if p.function_call:
                    tool_calls.append({
                        "function": {
                            "name": p.function_call.name,
                            "arguments": p.function_call.args
                        }
                    })

            msg = {"role": role, "content": text}
            if tool_calls:
                msg["tool_calls"] = tool_calls

            # Add name for tool responses (Critical for Ollama to map back to call)
            # [FIX] If we use 'user' role, we don't need 'name' field as it might be rejected
            if role == "tool" and tool_response_name:
                msg["name"] = tool_response_name

            messages.append(msg)

        # Convert Tools
        ollama_tools = []
        if llm_request.config and llm_request.config.tools:
            for tool in llm_request.config.tools:
                # DEBUG: What type of tool are we getting?
                # safe_print(f"[dim]Tool type: {type(tool).__name__}, attrs: {[a for a in dir(tool) if not a.startswith('_')][:10]}[/]")

                # Method 1: Standard Tool with function_declarations (from types.Tool)
                if hasattr(tool, 'function_declarations') and tool.function_declarations:
                    for func in tool.function_declarations:
                        # Convert Schema
                        def convert_schema(schema):
                            if not schema: return {}
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
                            if schema.description: res["description"] = schema.description
                            if schema.properties:
                                res["properties"] = {k: convert_schema(v) for k, v in schema.properties.items()}
                            if schema.required: res["required"] = schema.required
                            if schema.items: res["items"] = convert_schema(schema.items)
                            if schema.enum: res["enum"] = schema.enum
                            return res

                        ollama_tool = {
                            "type": "function",
                            "function": {
                                "name": func.name,
                                "description": func.description or "",
                                "parameters": convert_schema(func.parameters)
                            }
                        }
                        ollama_tools.append(ollama_tool)

                # Method 2: FunctionTool from ADK (has .func attribute pointing to the actual function)
                elif hasattr(tool, 'func') and callable(tool.func):
                    func = tool.func
                    func_name = func.__name__
                    func_doc = func.__doc__ or ""

                    # Extract parameters from function signature
                    import inspect
                    sig = inspect.signature(func)
                    properties = {}
                    required = []

                    for param_name, param in sig.parameters.items():
                        if param_name in ('self', 'cls', 'tool_context'):
                            continue

                        # Infer type from annotation
                        param_type = "string"
                        if param.annotation != inspect.Parameter.empty:
                            ann = param.annotation
                            if ann == int:
                                param_type = "integer"
                            elif ann == float:
                                param_type = "number"
                            elif ann == bool:
                                param_type = "boolean"
                            elif ann == list:
                                param_type = "array"
                            elif ann == dict:
                                param_type = "object"

                        properties[param_name] = {"type": param_type}

                        if param.default == inspect.Parameter.empty:
                            required.append(param_name)

                    ollama_tool = {
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "description": func_doc.split('\n')[0] if func_doc else f"Call {func_name}",
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required
                            }
                        }
                    }
                    ollama_tools.append(ollama_tool)

        # Get settings with defaults - FROM services.yaml (SINGLE SOURCE OF TRUTH)
        if base_url is None:
            try:
                from lib.core.AitherPorts import ollama_url
                base_url = ollama_url()
            except ImportError:
                raise ImportError("Cannot import AitherPorts. Ensure services.yaml is available.")
        # Allow timeout override via env var, default to 90s (longer for complex tasks)
        timeout = int(os.getenv('OLLAMA_TIMEOUT', '90'))

        # PERFORMANCE: Adaptive context sizing based on conversation length
        # Override with env var if set, otherwise auto-calculate
        env_ctx = os.getenv('OLLAMA_CONTEXT_SIZE')
        if env_ctx:
            context_window = int(env_ctx)
        else:
            context_window = estimate_context_size(messages, len(ollama_tools))

        # Mark model as preloaded since we're about to use it
        _MODEL_PRELOADED.add(self.model)

        # Check if model supports native tools API
        supports_native_tools = _model_supports_native_tools(ollama_model)

        # Call Ollama
        url = f"{base_url}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "keep_alive": DEFAULT_KEEP_ALIVE,  # Keep model warm for faster subsequent calls
            "options": {
                "temperature": 0.7,  # Good balance of creativity and consistency
                "num_ctx": context_window,
                "num_predict": -1,  # No limit on output tokens (let model decide)
            }
        }

        # PERFORMANCE: Only send native tools to models that support them
        # Models without native support will use [TOOL_CALLS] format via system prompt
        if ollama_tools and supports_native_tools:
            payload["tools"] = ollama_tools
            payload["stream"] = True
            safe_print(f"[dim cyan][TOOL] Sending {len(ollama_tools)} tools to {ollama_model}[/]")
        elif ollama_tools:
            # Model uses prompt-based tool calling - tools already in system prompt
            payload["stream"] = True
            safe_print(f"[dim cyan][TOOL] {ollama_model} uses prompt-based tools ({len(ollama_tools)} available)[/]")
        else:
            payload["stream"] = True

        try:
            # Use requests with stream=True
            response = requests.post(url, json=payload, timeout=timeout, stream=True)

            if response.status_code == 400:
                # Fallback: Try without tools if 400 Bad Request occurs
                if "tools" in payload:
                    del payload["tools"]
                response = requests.post(url, json=payload, timeout=timeout, stream=True)

            response.raise_for_status()

            full_content_text = ""
            full_tool_calls = []

            # Process stream
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)

                    # Handle done
                    if chunk.get("done"):
                        break

                    message = chunk.get("message", {})
                    content_chunk = message.get("content", "")

                    # Accumulate content
                    full_content_text += content_chunk

                    # Yield partial text response for UI feedback
                    if content_chunk:
                        yield LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[types.Part(text=content_chunk)]
                            ),
                            partial=True,
                            turn_complete=False
                        )

                    # Handle tool calls in stream (Ollama sends them usually at the end or in chunks)
                    if "tool_calls" in message:
                        # Accumulate tool calls (they might be partial or full, usually full in Ollama stream)
                        # Ollama streaming tool calls logic: usually sends full tool call in one chunk or accumulates?
                        # Based on docs, tool_calls are sent when ready.
                        new_calls = message.get("tool_calls", [])
                        full_tool_calls.extend(new_calls)

                except json.JSONDecodeError:
                    continue

            # Final processing after stream ends
            content_text = full_content_text
            tool_calls = full_tool_calls

            # safe_print(f"[dim]Ollama response received (Content: {len(content_text)} chars, Tools: {len(tool_calls)})[/]")

            # Strip <think>...</think> tags from qwen3-style reasoning models
            # These are internal reasoning traces that shouldn't be shown to user
            if "<think>" in content_text and "</think>" in content_text:
                # Extract and optionally log thinking for debugging
                think_match = re.search(r"<think>(.*?)</think>", content_text, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    if os.getenv("DEBUG_MODE", "false").lower() == "true":
                        safe_print(f"[dim cyan][BRAIN] Thinking: {thinking[:200]}{'...' if len(thinking) > 200 else ''}[/]")
                    # Remove thinking from output
                    content_text = re.sub(r"<think>.*?</think>\s*", "", content_text, flags=re.DOTALL).strip()

            # DEBUG_MODE check - using env var as fallback
            if os.getenv("DEBUG_MODE", "false").lower() == "true" and content_text:
                 safe_print(f"[dim]Raw Content: {content_text}[/]")

            # Fallback: Manual Tool Parsing for Local Models
            # Some local models output code blocks instead of native tool calls
            if not tool_calls and content_text and ollama_tools:
                # [NEW] Mistral/Nemo [TOOL_CALLS] format fallback
                if "[TOOL_CALLS]" in content_text:
                    try:
                        # Extract JSON after [TOOL_CALLS]
                        json_str = content_text.split("[TOOL_CALLS]")[1].strip()

                        # Simple heuristic: find the matching closing brace/bracket
                        if json_str.startswith("{"):
                            # Find matching }
                            count = 0
                            end_idx = -1
                            for i, char in enumerate(json_str):
                                if char == "{": count += 1
                                elif char == "}": count -= 1
                                if count == 0:
                                    end_idx = i + 1
                                    break
                            if end_idx != -1:
                                json_str = json_str[:end_idx]
                        elif json_str.startswith("["):
                            # Find matching ]
                            count = 0
                            end_idx = -1
                            for i, char in enumerate(json_str):
                                if char == "[": count += 1
                                elif char == "]": count -= 1
                                if count == 0:
                                    end_idx = i + 1
                                    break
                            if end_idx != -1:
                                json_str = json_str[:end_idx]

                        tool_data = json.loads(json_str)

                        calls = []
                        if isinstance(tool_data, list):
                            calls = tool_data
                        elif isinstance(tool_data, dict):
                            calls = [tool_data]

                        for call in calls:
                            if "name" in call and "arguments" in call:
                                tool_calls.append({
                                    "function": {
                                        "name": call["name"],
                                        "arguments": call["arguments"]
                                    }
                                })
                                safe_print(f"[dim]Successfully parsed [TOOL_CALLS]: {call['name']}[/]")

                        if tool_calls:
                            content_text = content_text.split("[TOOL_CALLS]")[0].strip()

                    except Exception as e:
                        safe_print(f"[dim]Failed to parse [TOOL_CALLS]: {e}[/]")

                # Extract code blocks or look for function calls in text
                # Pattern: ```python ... ``` or just text

                # 1. Extract potential code
                code_blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", content_text, re.DOTALL | re.IGNORECASE)
                potential_code = code_blocks if code_blocks else [content_text]

                known_tool_names = {t['function']['name'] for t in ollama_tools}

                for code in potential_code:
                    # [NEW] Regex Fallback for simple calls like tool_name(arg="val")
                    # This catches cases where AST fails due to surrounding text or invalid syntax
                    for tool_name in known_tool_names:
                        # Regex to find tool_name( ... )
                        # This is a simple regex and might fail on complex nested args, but good for simple calls
                        pattern = rf"{tool_name}\s*\((.*?)\)"
                        matches = re.finditer(pattern, code, re.DOTALL)
                        for match in matches:
                            args_str = match.group(1)
                            try:
                                # Try to parse arguments as a python dict content or just eval it wrapped in dict
                                # We construct a fake call to parse args
                                fake_call = f"{tool_name}({args_str})"
                                tree = ast.parse(fake_call)
                                if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                                    call = tree.body[0].value

                                    # Extract args
                                    args = {}
                                    tool_def = next(t for t in ollama_tools if t['function']['name'] == tool_name)
                                    param_names = list(tool_def['function']['parameters'].get('properties', {}).keys())

                                    # Handle Positional Args
                                    for i, arg in enumerate(call.args):
                                        if i < len(param_names):
                                            try:
                                                if isinstance(arg, ast.Constant):
                                                    val = arg.value
                                                else:
                                                    val = ast.literal_eval(arg)
                                                args[param_names[i]] = val
                                            except Exception as exc:
                                                logger.debug(f"Regex+AST positional arg eval failed: {exc}")

                                    # Handle Keyword Args
                                    for keyword in call.keywords:
                                        try:
                                            if isinstance(keyword.value, ast.Constant):
                                                val = keyword.value.value
                                            else:
                                                val = ast.literal_eval(keyword.value)
                                            args[keyword.arg] = val
                                        except Exception as exc:
                                            logger.debug(f"Regex+AST keyword arg eval failed: {exc}")

                                    # Add to tool calls
                                    tool_calls.append({
                                        "function": {
                                            "name": tool_name,
                                            "arguments": args
                                        }
                                    })
                                    safe_print(f"[dim]Successfully parsed tool call via Regex+AST: {tool_name}[/]")

                                    # Suppress text if it was just a tool call
                                    # If the text contains the tool call, we assume the text is just a wrapper.
                                    content_text = ""
                            except Exception:
                                # safe_print(f"[dim]Regex+AST parsing failed: {e}[/]")
                                pass

                    try:
                        # Clean up code (remove non-python text if mixed)
                        # But ast.parse expects valid python.
                        # If the model outputs "Here is the code:\nprint('hi')", ast.parse might fail on the first line.
                        # So we still might need to be careful.
                        # However, mistral-nemo usually outputs clean code blocks.

                        # [FIX] If code is just text, AST parse might fail.
                        # We try to find the first valid python statement if parse fails?
                        # Or just wrap in try/except.

                        tree = ast.parse(code)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                                call = node.value
                            elif isinstance(node, ast.Call):
                                call = node
                            else:
                                continue

                            if isinstance(call.func, ast.Name):
                                tool_name = call.func.id
                                if tool_name in known_tool_names:
                                    safe_print(f"[dim]Detected potential tool call via AST: {tool_name}[/]")

                                    # Extract args
                                    args = {}
                                    tool_def = next(t for t in ollama_tools if t['function']['name'] == tool_name)
                                    param_names = list(tool_def['function']['parameters'].get('properties', {}).keys())

                                    # Handle Positional Args
                                    for i, arg in enumerate(call.args):
                                        if i < len(param_names):
                                            try:
                                                if isinstance(arg, ast.Constant):
                                                    val = arg.value
                                                else:
                                                    val = ast.literal_eval(arg)
                                                args[param_names[i]] = val
                                            except Exception as exc:
                                                logger.debug(f"AST positional arg eval failed: {exc}")

                                    # Handle Keyword Args
                                    for keyword in call.keywords:
                                        try:
                                            if isinstance(keyword.value, ast.Constant):
                                                val = keyword.value.value
                                            else:
                                                val = ast.literal_eval(keyword.value)
                                            args[keyword.arg] = val
                                        except Exception as exc:
                                            logger.debug(f"AST keyword arg eval failed: {exc}")

                                    # Add to tool calls
                                    tool_calls.append({
                                        "function": {
                                            "name": tool_name,
                                            "arguments": args
                                        }
                                    })
                                    safe_print(f"[dim]Successfully parsed tool call: {tool_name}[/]")

                                    # Suppress text if it was just a tool call
                                    # [FIX] Be more aggressive in suppressing text if a tool call is found
                                    # If the text contains the tool call, we assume the text is just a wrapper.
                                    content_text = ""
                    except Exception:
                        # safe_print(f"[dim]AST parsing skipped: {e}[/]")
                        pass

            parts = []
            if content_text:
                parts.append(types.Part(text=content_text))

            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name")
                args = func.get("arguments", {})
                parts.append(types.Part(
                    function_call=types.FunctionCall(
                        name=name,
                        args=args
                    )
                ))

            # Yield response
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=parts
                ),
                partial=False,
                turn_complete=True,
                finish_reason=types.FinishReason.STOP
            )

        except Exception as e:
            safe_print(f"[red]Ollama Error: {e}[/]")
            raise ValueError(f"Ollama generation failed: {e}")
