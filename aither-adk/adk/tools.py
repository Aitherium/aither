"""Tool system — @tool decorator and ToolRegistry for agent function calling."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("adk.tools")


@dataclass
class ToolDef:
    """A registered tool definition."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    fn: Callable
    is_async: bool = False


class ToolRegistry:
    """Registry of tool functions that agents can call."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, fn: Callable, name: str | None = None, description: str | None = None) -> ToolDef:
        """Register a function as a tool."""
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or f"Tool: {tool_name}"
        tool_desc = tool_desc.strip().split("\n")[0]  # First line only

        params = _extract_parameters(fn)
        is_async = inspect.iscoroutinefunction(fn)

        td = ToolDef(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            fn=fn,
            is_async=is_async,
        )
        self._tools[tool_name] = td
        return td

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def to_openai_format(self) -> list[dict]:
        """Export tools in OpenAI function-calling format."""
        result = []
        for td in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            })
        return result

    async def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments. Returns result as string."""
        td = self._tools.get(name)
        if not td:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            if td.is_async:
                result = await td.fn(**arguments)
            else:
                result = td.fn(**arguments)

            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return json.dumps({"error": str(e)})


# Module-level registry for the @tool decorator
_global_registry = ToolRegistry()


def tool(fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator to register a function as an agent tool.

    Usage:
        @tool
        def search_web(query: str) -> str:
            '''Search the web for information.'''
            ...

        @tool(name="calculator", description="Evaluate math expressions")
        def calc(expression: str) -> str:
            ...
    """
    def decorator(f: Callable) -> Callable:
        td = _global_registry.register(f, name=name, description=description)
        f._tool_def = td
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry (populated by @tool decorator)."""
    return _global_registry


def _extract_parameters(fn: Callable) -> dict:
    """Extract JSON Schema parameters from function signature and type hints."""
    sig = inspect.signature(fn)
    hints = getattr(fn, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        hint = hints.get(param_name, str)
        prop = _type_to_schema(hint)
        prop_desc = ""

        # Try to extract from docstring
        doc = fn.__doc__ or ""
        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{param_name}:") or stripped.startswith(f"{param_name} "):
                prop_desc = stripped.split(":", 1)[-1].strip() if ":" in stripped else ""
                break

        if prop_desc:
            prop["description"] = prop_desc

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _type_to_schema(hint) -> dict:
    """Convert a Python type hint to a JSON Schema type."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    if hint in type_map:
        return dict(type_map[hint])

    origin = getattr(hint, "__origin__", None)
    if origin is list:
        args = getattr(hint, "__args__", (str,))
        return {"type": "array", "items": _type_to_schema(args[0] if args else str)}
    if origin is dict:
        return {"type": "object"}

    return {"type": "string"}
