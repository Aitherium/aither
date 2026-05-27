"""
MCP Client Tools - Lightweight HTTP stubs for AitherNode
=========================================================
When AitherNode is already running on port 8080, these functions call
the HTTP API instead of re-initializing heavy modules locally.

This provides:
- Fast startup (~2s instead of ~15s)
- No duplicate service initialization
- Shared state with running AitherNode
"""

import atexit
import json
import logging
import warnings
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

# Suppress unclosed resource warnings at exit (these are harmless)
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")

# Service URLs from services.yaml
try:
    from lib.core.AitherPorts import get_service_url
    AITHERNODE_URL = get_service_url("Node")
except ImportError:
    from lib.core.AitherPorts import get_port
    AITHERNODE_URL = f"http://localhost:{get_port('Node', 8090)}"

# Sync HTTP client for tool calls (reused across calls)
_client: httpx.Client = None

def _get_client() -> httpx.Client:
    """Get or create the HTTP client (lazy init)."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60.0)
    return _client

def _cleanup_client():
    """Close HTTP client on exit."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception as exc:
            logger.debug(f"HTTP client cleanup failed: {exc}")
        _client = None

# Register cleanup on exit
atexit.register(_cleanup_client)

def _call_mcp_tool(tool_name: str, **kwargs) -> str:
    """Call an MCP tool via HTTP REST bridge."""
    try:
        client = _get_client()
        # MCP tools are exposed at /mcp/tools/{tool_name}
        response = client.post(
            f"{AITHERNODE_URL}/mcp/tools/{tool_name}",
            json={"arguments": kwargs},
            timeout=60.0
        )
        if response.status_code == 200:
            # Parse JSON response and extract result
            try:
                data = response.json()
                if data.get("success"):
                    result = data.get("result", "")
                    # If result is a dict/list, return as JSON string
                    if isinstance(result, (dict, list)):
                        return json.dumps(result)
                    return str(result)
                else:
                    return json.dumps({"error": data.get("error", "Unknown error")})
            except json.JSONDecodeError:
                # If not JSON, return raw text
                return response.text
        elif response.status_code == 404:
            return json.dumps({"error": f"Tool '{tool_name}' not found. AitherNode may need restart."})
        else:
            return json.dumps({"error": f"HTTP {response.status_code}: {response.text}"})
    except httpx.ConnectError:
        return json.dumps({"error": "Cannot connect to AitherNode. Is it running on port 8080?"})
    except httpx.TimeoutException:
        return json.dumps({"error": f"Tool '{tool_name}' timed out after 60s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# ESSENTIAL REAL-TIME TOOLS
# =============================================================================

def get_current_time() -> str:
    """
    Get the current date and time.

    ALWAYS use this tool when the user asks:
    - What time is it?
    - What's today's date?
    - What day is it?

    Returns:
        Current date and time string.
    """
    return _call_mcp_tool("get_current_time")

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information.

    Use for current events, news, prices, or any fact that could have changed.

    Args:
        query: Search query
        max_results: Maximum results to return
    """
    return _call_mcp_tool("web_search", query=query, max_results=max_results)


# =============================================================================
# RAG / SEMANTIC SEARCH TOOLS (AitherMind)
# =============================================================================

def search_knowledge(query: str, collection: str = "default", limit: int = 5) -> str:
    """
    Semantic search across the AitherMind knowledge base using vector embeddings.

    This is the PRIMARY TOOL for searching AitherOS documentation, codebase,
    memories, and any indexed knowledge. Uses nomic-embed-text embeddings.

    Use this when you need:
    - To find documentation about AitherOS services
    - To search codebase examples and patterns
    - To retrieve relevant context for a question
    - To find memories or stored facts

    Args:
        query: Natural language search query (semantic, not keyword)
        collection: Knowledge collection to search (default, codebase, teachings, etc.)
        limit: Maximum results to return (default 5)

    Returns:
        JSON with matching documents and relevance scores

    Example:
        >>> search_knowledge("how does the Faculty architecture work")
        >>> search_knowledge("service bootstrap pattern", collection="codebase")
    """
    return _call_mcp_tool("search_knowledge", query=query, collection=collection, limit=limit)


def think(question: str, context: str = "", depth: str = "standard") -> str:
    """
    Engage in deep reasoning about a question using RAG + LLM.

    Uses chain-of-thought reasoning with optional RAG context retrieval.

    Args:
        question: The question or problem to think about
        context: Optional additional context to consider
        depth: Reasoning depth - 'quick', 'standard', 'deep'

    Returns:
        JSON with reasoned analysis and conclusion
    """
    return _call_mcp_tool("think", question=question, context=context, depth=depth)


def summarize_text(text: str, style: str = "concise") -> str:
    """
    Summarize long text content.

    Args:
        text: Text to summarize
        style: Summary style - 'concise', 'bullet_points', 'executive', 'detailed'

    Returns:
        JSON with summary
    """
    return _call_mcp_tool("summarize", text=text, style=style)


# =============================================================================
# MEMORY TOOLS
# =============================================================================

def remember(content: str, category: str = "general", importance: float = 0.5) -> str:
    """Store information in long-term memory."""
    return _call_mcp_tool("remember", content=content, category=category, importance=importance)

def recall(query: str, limit: int = 5, memory_type: str = None) -> str:
    """Search memories using semantic similarity."""
    return _call_mcp_tool("recall", query=query, limit=limit, memory_type=memory_type)

def add_to_working_memory(content: str, category: str = "context") -> str:
    """Add to session working memory."""
    return _call_mcp_tool("add_to_working_memory", content=content, category=category)

def get_current_context() -> str:
    """Get current session context."""
    return _call_mcp_tool("get_current_context")

def clear_context() -> str:
    """Clear session context."""
    return _call_mcp_tool("clear_context")

def list_memory_entries(category: str = None, limit: int = 20) -> str:
    """List recent memory entries."""
    return _call_mcp_tool("list_memory_entries", category=category, limit=limit)


# =============================================================================
# VISION/IMAGE TOOLS
# =============================================================================

def analyze_image_content(image_path: str, question: str = None) -> str:
    """Analyze image using vision model."""
    return _call_mcp_tool("analyze_image_content", image_path=image_path, question=question)

def generate_image(prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, steps: int = 20) -> str:
    """Generate image using ComfyUI."""
    return _call_mcp_tool("generate_image", prompt=prompt, negative_prompt=negative_prompt,
                          width=width, height=height, steps=steps)

def refine_image(image_path: str, prompt: str, strength: float = 0.5) -> str:
    """Refine existing image."""
    return _call_mcp_tool("refine_image", image_path=image_path, prompt=prompt, strength=strength)

def create_animation(prompts: str, output_dir: str = "generated_animations", fps: int = 10) -> str:
    """Create animation from prompts."""
    return _call_mcp_tool("create_animation", prompts=prompts, output_dir=output_dir, fps=fps)

def compare_images(image1_path: str, image2_path: str) -> str:
    """Compare two images."""
    return _call_mcp_tool("compare_images", image1_path=image1_path, image2_path=image2_path)

def ask_about_image(image_path: str, question: str) -> str:
    """Ask a question about an image."""
    return _call_mcp_tool("ask_about_image", image_path=image_path, question=question)

def extract_text_from_image(image_path: str) -> str:
    """OCR - extract text from image."""
    return _call_mcp_tool("extract_text_from_image", image_path=image_path)

def get_vision_status() -> str:
    """Get vision service status."""
    return _call_mcp_tool("get_vision_status")

def unload_vision_model() -> str:
    """Unload vision model from VRAM."""
    return _call_mcp_tool("unload_vision_model")

def list_workflows() -> str:
    """List available ComfyUI workflows."""
    return _call_mcp_tool("list_workflows")


# =============================================================================
# OLLAMA TOOLS
# =============================================================================

def list_ollama_models() -> str:
    """List available Ollama models."""
    return _call_mcp_tool("list_ollama_models")

def chat_ollama(prompt: str, model: str = "llama3", system: str = None) -> str:
    """Chat with local Ollama model."""
    return _call_mcp_tool("chat_ollama", prompt=prompt, model=model, system=system)


# =============================================================================
# PERSONA TOOLS
# =============================================================================

def list_personas() -> str:
    """List available personas."""
    return _call_mcp_tool("list_personas")

def get_persona_details(name: str) -> str:
    """Get persona details."""
    return _call_mcp_tool("get_persona_details", name=name)

def update_persona(name: str, description: str = None, instruction: str = None) -> str:
    """Update persona."""
    return _call_mcp_tool("update_persona", name=name, description=description, instruction=instruction)

def generate_persona_profile_picture(name: str) -> str:
    """Generate profile picture for persona."""
    return _call_mcp_tool("generate_persona_profile_picture", name=name)


# =============================================================================
# INFRASTRUCTURE TOOLS
# =============================================================================

def run_script(script_number: str, args: str = "") -> str:
    """Run automation script."""
    return _call_mcp_tool("run_script", script_number=script_number, args=args)

def mcp_get_service_status(services: str = "", refresh: bool = False) -> str:
    """Get service status."""
    return _call_mcp_tool("get_service_status", services=services, refresh=refresh)

def get_service_summary() -> str:
    """Get service summary."""
    return _call_mcp_tool("get_service_summary")


# =============================================================================
# RBAC TOOLS
# =============================================================================

def rbac_list_users() -> str:
    """List RBAC users."""
    return _call_mcp_tool("rbac_list_users")

def rbac_get_user(user_id: str) -> str:
    """Get RBAC user details."""
    return _call_mcp_tool("rbac_get_user", user_id=user_id)

def rbac_check_permission(user_id: str, resource: str, action: str, scope: str = "*") -> str:
    """Check RBAC permission."""
    return _call_mcp_tool("rbac_check_permission", user_id=user_id, resource=resource,
                          action=action, scope=scope)

def rbac_summary() -> str:
    """Get RBAC summary."""
    return _call_mcp_tool("rbac_summary")


# =============================================================================
# LOCAL GENERATION STUBS (these need local Ollama, not HTTP)
# =============================================================================

def generate_local(prompt: str, model: str = "llama3") -> str:
    """Generate with local Ollama (direct call, not via AitherNode)."""
    try:
        import httpx

        from lib.core.AitherPorts import ollama_url
        resp = httpx.post(
            f"{ollama_url()}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"

def generate_local_response(prompt: str, model: str = "llama3") -> str:
    """Alias for generate_local."""
    return generate_local(prompt, model)

def is_ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        from lib.core.AitherPorts import ollama_url
        resp = httpx.get(f"{ollama_url()}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except (ImportError, httpx.HTTPError):
        return False

def get_vision_backend_status() -> Dict[str, Any]:
    """Get vision backend status via direct HTTP endpoint."""
    try:
        client = _get_client()
        # Use direct vision HTTP endpoint (more reliable than MCP tool route)
        response = client.get(f"{AITHERNODE_URL}/vision/status", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "status" in data:
                return data["status"]
            return data
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Empty tool lists (these tools need local module access)
ollama_tools = []
civitai_tools = []
huggingface_tools = []
video_tools = []
animation_tools = []
dataset_tools = []


# =============================================================================
# CORTEX/NEURONS TOOLS - Parallel intelligence gathering
# =============================================================================

# Cortex URL from services.yaml
try:
    from lib.core.AitherPorts import get_service_url
    CORTEX_URL = get_service_url("Cortex")
except ImportError:
    from lib.core.AitherPorts import get_port
    CORTEX_URL = f"http://localhost:{get_port('Cortex', 8139)}"

def think_with_neurons(
    query: str,
    include_web: bool = False,
    include_memory: bool = True,
    timeout: float = 5.0
) -> str:
    """
    Gather context using AitherCortex neurons for parallel intelligence gathering.

    Fires multiple specialized neurons in parallel:
    - GrepNeuron: Fast codebase pattern matching
    - FileNeuron: Semantic file content analysis
    - DocNeuron: Documentation and README search
    - CodeNeuron: Symbol and function analysis
    - MemoryNeuron: SensoryBuffer recall
    - WebNeuron: Optional web search

    Use this for complex queries that benefit from gathering context from
    multiple sources simultaneously. More efficient than sequential tool calls.

    Args:
        query: The query to gather context for
        include_web: Whether to include web search neuron
        include_memory: Whether to include memory recall neuron
        timeout: Maximum time for neuron execution

    Returns:
        JSON string with synthesis, confidence, sources, and metadata
    """
    try:
        import httpx
        resp = httpx.post(
            f"{CORTEX_URL}/think",
            json={
                "query": query,
                "include_web": include_web,
                "include_memory": include_memory,
                "timeout": timeout
            },
            timeout=timeout + 5.0
        )
        if resp.status_code == 200:
            return resp.text
        return json.dumps({"error": f"Cortex returned {resp.status_code}", "success": False})
    except httpx.TimeoutException:
        return json.dumps({"error": f"Neurons timed out after {timeout}s", "success": False})
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


def fire_neuron(neuron_type: str, query: str, timeout: float = 3.0) -> str:
    """
    Fire a single specialized neuron for targeted context gathering.

    Available neuron types:
    - grep: Fast pattern matching in codebase
    - file: Semantic file content analysis
    - doc: Documentation and README search
    - code: Symbol and function analysis
    - memory: SensoryBuffer recall
    - web: Web search (if enabled)

    Args:
        neuron_type: Type of neuron to fire
        query: The query/pattern to search for
        timeout: Maximum execution time

    Returns:
        JSON string with neuron results
    """
    try:
        import httpx
        resp = httpx.post(
            f"{CORTEX_URL}/fire",
            json={
                "neuron_type": neuron_type,
                "query": query,
                "timeout": timeout
            },
            timeout=timeout + 2.0
        )
        if resp.status_code == 200:
            return resp.text
        return json.dumps({"error": f"Neuron fire failed: {resp.status_code}", "success": False})
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


# =============================================================================
# MCP TOOLS LIST - Built lazily to avoid slow google.adk import at module load
# =============================================================================
_mcp_server_tools = None

def _build_mcp_tools():
    """Build MCP tools list with FunctionTool wrappers (lazy)."""
    from google.adk.tools import FunctionTool
    return [
        # Essential real-time
        FunctionTool(get_current_time),
        FunctionTool(web_search),
        # Memory
        FunctionTool(remember),
        FunctionTool(recall),
        FunctionTool(add_to_working_memory),
        FunctionTool(get_current_context),
        FunctionTool(clear_context),
        FunctionTool(list_memory_entries),
        # RAG/Mind
        FunctionTool(search_knowledge),
        FunctionTool(think),
        FunctionTool(summarize_text),
        # Vision
        FunctionTool(analyze_image_content),
        FunctionTool(compare_images),
        FunctionTool(ask_about_image),
        FunctionTool(extract_text_from_image),
        FunctionTool(get_vision_status),
        # Image Generation
        FunctionTool(generate_image),
        FunctionTool(refine_image),
        FunctionTool(create_animation),
        FunctionTool(list_workflows),
        # Ollama
        FunctionTool(list_ollama_models),
        FunctionTool(chat_ollama),
        # Personas
        FunctionTool(list_personas),
        FunctionTool(get_persona_details),
        FunctionTool(update_persona),
        FunctionTool(generate_persona_profile_picture),
        # Infrastructure
        FunctionTool(run_script),
        FunctionTool(mcp_get_service_status),
        FunctionTool(get_service_summary),
        # RBAC
        FunctionTool(rbac_list_users),
        FunctionTool(rbac_get_user),
        FunctionTool(rbac_check_permission),
        FunctionTool(rbac_summary),
        # Neurons/Cortex
        FunctionTool(think_with_neurons),
        FunctionTool(fire_neuron),
    ]

# Lazy property - only builds when accessed
class _LazyToolList:
    """Lazy tool list that only imports FunctionTool when accessed."""
    _tools = None

    def __iter__(self):
        if self._tools is None:
            self._tools = _build_mcp_tools()
        return iter(self._tools)

    def __len__(self):
        if self._tools is None:
            self._tools = _build_mcp_tools()
        return len(self._tools)

    def __add__(self, other):
        if self._tools is None:
            self._tools = _build_mcp_tools()
        return self._tools + list(other)

    def __radd__(self, other):
        if self._tools is None:
            self._tools = _build_mcp_tools()
        return list(other) + self._tools

mcp_server_tools = _LazyToolList()
