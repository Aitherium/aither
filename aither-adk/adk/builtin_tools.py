"""Built-in tools — core capabilities that work WITHOUT AitherOS/AitherNode.

These give agents real autonomy in standalone mode:
  - File I/O (read, write, edit, list, search)
  - Shell execution (subprocess with timeout + capture)
  - Python REPL (isolated exec with output capture)
  - Web search/fetch (via DuckDuckGo + httpx)
  - Secrets store (local encrypted keyring, no AitherSecrets needed)

When AitherNode is available, these are SUPPLEMENTED (not replaced) by the
449 MCP tools. Built-in tools always work offline.

Usage:
    from adk.builtin_tools import register_builtin_tools

    agent = AitherAgent("demiurge")
    register_builtin_tools(agent, categories=["file_io", "shell", "web"])
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adk.agent import AitherAgent

logger = logging.getLogger("adk.builtin_tools")

# Safety: directories agents can access (expandable via AITHER_ALLOWED_ROOTS)
_DEFAULT_ALLOWED_ROOTS = [os.getcwd()]
_ALLOWED_ROOTS: list[str] | None = None


def _get_allowed_roots() -> list[str]:
    global _ALLOWED_ROOTS
    if _ALLOWED_ROOTS is None:
        extra = os.getenv("AITHER_ALLOWED_ROOTS", "")
        _ALLOWED_ROOTS = _DEFAULT_ALLOWED_ROOTS + [r for r in extra.split(";") if r]
    return _ALLOWED_ROOTS


def _is_safe_path(path: str) -> bool:
    """Check if a path is within allowed roots."""
    try:
        resolved = str(Path(path).resolve())
        return any(resolved.startswith(str(Path(r).resolve())) for r in _get_allowed_roots())
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# File I/O Tools
# ─────────────────────────────────────────────────────────────────────────────

def file_read(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file from disk. Returns file contents.

    path: Absolute or relative file path
    start_line: Start reading from this line (0 = beginning)
    end_line: Stop reading at this line (0 = end of file)
    """
    if not _is_safe_path(path):
        return json.dumps({"error": f"Path outside allowed roots: {path}"})
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        if p.stat().st_size > 10_000_000:  # 10MB limit
            return json.dumps({"error": "File too large (>10MB)"})
        content = p.read_text(encoding="utf-8", errors="replace")
        if start_line or end_line:
            lines = content.split("\n")
            start = max(0, start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            content = "\n".join(lines[start:end])
        return content
    except Exception as e:
        return json.dumps({"error": str(e)})


def file_write(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file on disk.

    path: File path to write to
    content: Content to write
    mode: 'overwrite' or 'append'
    """
    if not _is_safe_path(path):
        return json.dumps({"error": f"Path outside allowed roots: {path}"})
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return json.dumps({"success": True, "path": str(p), "bytes": len(content)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def file_edit(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing old_text with new_text (exact string match).

    path: File path to edit
    old_text: Exact text to find and replace
    new_text: Replacement text
    """
    if not _is_safe_path(path):
        return json.dumps({"error": f"Path outside allowed roots: {path}"})
    try:
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})
        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return json.dumps({"error": "old_text not found in file"})
        count = content.count(old_text)
        if count > 1:
            return json.dumps({"error": f"old_text found {count} times — must be unique. Add more context."})
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return json.dumps({"success": True, "path": str(p)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def file_list(path: str = ".", pattern: str = "*") -> str:
    """List files in a directory.

    path: Directory path to list
    pattern: Glob pattern to filter (default: *)
    """
    try:
        p = Path(path)
        if not p.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"})
        entries = []
        for item in sorted(p.glob(pattern))[:200]:
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return json.dumps({"path": str(p), "entries": entries, "count": len(entries)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def file_search(path: str, pattern: str, content_pattern: str = "") -> str:
    """Search for files by name pattern, optionally grep for content.

    path: Root directory to search
    pattern: Glob pattern for filenames (e.g. '**/*.py')
    content_pattern: Optional text to search for inside matching files
    """
    try:
        p = Path(path)
        matches = []
        for item in p.glob(pattern):
            if not item.is_file():
                continue
            if content_pattern:
                try:
                    text = item.read_text(encoding="utf-8", errors="replace")
                    if content_pattern not in text:
                        continue
                    # Find line numbers
                    lines = []
                    for i, line in enumerate(text.split("\n"), 1):
                        if content_pattern in line:
                            lines.append({"line": i, "text": line.strip()[:200]})
                            if len(lines) >= 5:
                                break
                    matches.append({"path": str(item), "matches": lines})
                except Exception:
                    continue
            else:
                matches.append({"path": str(item)})
            if len(matches) >= 50:
                break
        return json.dumps({"results": matches, "count": len(matches)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Shell & Python Execution
# ─────────────────────────────────────────────────────────────────────────────

def shell_exec(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr.

    command: Shell command to run
    timeout: Maximum execution time in seconds (default 30)
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
        )
        output = {
            "exit_code": result.returncode,
            "stdout": result.stdout[:50_000],
            "stderr": result.stderr[:10_000],
        }
        return json.dumps(output)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def python_exec(code: str) -> str:
    """Execute Python code in an isolated namespace and capture output.

    code: Python code to execute
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    namespace: dict = {"__builtins__": __builtins__}
    result_val = None

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)
            # If there's a 'result' variable, capture it
            if "result" in namespace:
                result_val = namespace["result"]
    except Exception as e:
        stderr_capture.write(f"\n{type(e).__name__}: {e}")

    output = {
        "stdout": stdout_capture.getvalue()[:50_000],
        "stderr": stderr_capture.getvalue()[:10_000],
    }
    if result_val is not None:
        try:
            output["result"] = json.loads(json.dumps(result_val, default=str))
        except Exception:
            output["result"] = str(result_val)
    return json.dumps(output)


# ─────────────────────────────────────────────────────────────────────────────
# Web Tools
# ─────────────────────────────────────────────────────────────────────────────

async def web_search(query: str, limit: int = 5) -> str:
    """Search the web using DuckDuckGo. Returns search results.

    query: Search query string
    limit: Maximum number of results (default 5)
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "AitherADK/1.0"},
            )
            resp.raise_for_status()
            text = resp.text

        # Parse results from HTML (simple extraction)
        results = []
        import re
        links = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', text)
        snippets = re.findall(r'class="result__snippet">(.*?)</a>', text, re.DOTALL)

        for i, (url, title) in enumerate(links[:limit]):
            snippet = snippets[i].strip() if i < len(snippets) else ""
            # Clean HTML tags
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            # Decode DuckDuckGo redirect URL
            if "uddg=" in url:
                from urllib.parse import unquote, parse_qs, urlparse
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                url = unquote(params.get("uddg", [url])[0])
            results.append({"title": title, "url": url, "snippet": snippet[:300]})

        return json.dumps({"query": query, "results": results})
    except ImportError:
        return json.dumps({"error": "httpx required for web search"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a webpage and return its text content.

    url: URL to fetch
    max_chars: Maximum characters to return (default 20000)
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "AitherADK/1.0"},
            )
            resp.raise_for_status()
            content = resp.text

        # Strip HTML tags for cleaner output
        import re
        # Remove script/style blocks
        content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', content, flags=re.DOTALL)
        # Remove tags
        content = re.sub(r'<[^>]+>', ' ', content)
        # Collapse whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        return content[:max_chars]
    except ImportError:
        return json.dumps({"error": "httpx required for web fetch"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Secrets Store (local, standalone)
# ─────────────────────────────────────────────────────────────────────────────

_secrets_cache: dict[str, str] | None = None
_SECRETS_FILE = Path(os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))) / "secrets.json"


def _load_secrets() -> dict[str, str]:
    global _secrets_cache
    if _secrets_cache is not None:
        return _secrets_cache
    if _SECRETS_FILE.exists():
        try:
            _secrets_cache = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            _secrets_cache = {}
    else:
        _secrets_cache = {}
    return _secrets_cache


def _save_secrets(data: dict[str, str]):
    global _secrets_cache
    _secrets_cache = data
    _SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SECRETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Restrict permissions on Unix
    try:
        os.chmod(_SECRETS_FILE, 0o600)
    except (OSError, AttributeError):
        pass


def secret_get(key: str) -> str:
    """Get a secret value by key. Checks env vars first, then local store.

    key: Secret key name
    """
    # Env var takes priority
    env_val = os.getenv(key)
    if env_val:
        return env_val
    secrets = _load_secrets()
    val = secrets.get(key)
    if val is None:
        return json.dumps({"error": f"Secret '{key}' not found"})
    return val


def secret_set(key: str, value: str) -> str:
    """Store a secret value. Persists to ~/.aither/secrets.json.

    key: Secret key name
    value: Secret value to store
    """
    secrets = _load_secrets()
    secrets[key] = value
    _save_secrets(secrets)
    return json.dumps({"success": True, "key": key})


def secret_list() -> str:
    """List all stored secret keys (values are NOT shown)."""
    secrets = _load_secrets()
    return json.dumps({"keys": list(secrets.keys()), "count": len(secrets)})


# ─────────────────────────────────────────────────────────────────────────────
# Creative Tools (AitherCanvas / ComfyUI)
# ─────────────────────────────────────────────────────────────────────────────

_CANVAS_URL = os.getenv("AITHER_CANVAS_URL", "http://localhost:8108")


def image_generate(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
) -> str:
    """Generate an image using AitherCanvas (ComfyUI).

    prompt: Detailed description of the image to generate
    negative_prompt: What to avoid in the image
    width: Image width in pixels (default 1024)
    height: Image height in pixels (default 1024)
    steps: Sampling steps (default 20)
    """
    try:
        import httpx
        resp = httpx.post(
            f"{_CANVAS_URL}/generate",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", data)
        images = result.get("images", [])
        if images:
            import base64 as b64mod
            out_dir = os.path.join(os.getcwd(), "generated")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = int(time.time())
            path = os.path.join(out_dir, f"gen_{timestamp}.png")
            with open(path, "wb") as f:
                f.write(b64mod.b64decode(images[0]))
            return json.dumps({
                "success": True,
                "path": path,
                "base64": images[0][:100] + "...",
                "count": len(images),
            })
        if result.get("paths"):
            return json.dumps({"success": True, "paths": result["paths"]})
        return json.dumps({"success": False, "error": "No images in response"})
    except Exception as e:
        err_msg = str(e)
        if "ConnectError" in type(e).__name__ or "Connection refused" in err_msg:
            return json.dumps({
                "success": False,
                "error": "AitherCanvas not running locally. Use MCP bridge to access "
                         "cloud image generation: MCPBridge(api_key=...).call_tool('generate_image', ...)",
            })
        return json.dumps({"success": False, "error": err_msg})


def image_refine(
    image_path: str,
    prompt: str,
    denoise: float = 0.5,
    negative_prompt: str = "",
) -> str:
    """Refine an existing image using AitherCanvas (Img2Img).

    image_path: Path to the source image
    prompt: Prompt to guide the refinement
    denoise: Denoising strength 0.0-1.0 (lower preserves more)
    negative_prompt: What to avoid
    """
    try:
        import httpx
        resp = httpx.post(
            f"{_CANVAS_URL}/generate",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "source_image_path": image_path,
                "denoise": denoise,
                "mode": "img2img",
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", data)
        images = result.get("images", [])
        if images:
            import base64 as b64mod
            out_dir = os.path.join(os.getcwd(), "generated")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = int(time.time())
            path = os.path.join(out_dir, f"refine_{timestamp}.png")
            with open(path, "wb") as f:
                f.write(b64mod.b64decode(images[0]))
            return json.dumps({"success": True, "path": path, "count": len(images)})
        if result.get("paths"):
            return json.dumps({"success": True, "paths": result["paths"]})
        return json.dumps({"success": False, "error": "No images in response"})
    except Exception as e:
        err_msg = str(e)
        if "ConnectError" in type(e).__name__ or "Connection refused" in err_msg:
            return json.dumps({
                "success": False,
                "error": "AitherCanvas not running locally. Use MCP bridge for cloud access.",
            })
        return json.dumps({"success": False, "error": err_msg})


def image_smart(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
) -> str:
    """Smart generate — auto-detects diagram vs artistic image.

    prompt: Description of what to generate
    negative_prompt: What to avoid
    width: Image width (default 1024)
    height: Image height (default 1024)
    """
    try:
        import httpx
        resp = httpx.post(
            f"{_CANVAS_URL}/smart-generate",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", data)
        images = result.get("images", [])
        if images:
            import base64 as b64mod
            is_diagram = bool(result.get("mermaid_code"))
            out_dir = os.path.join(os.getcwd(), "generated")
            os.makedirs(out_dir, exist_ok=True)
            prefix = "diagram" if is_diagram else "smart"
            timestamp = int(time.time())
            path = os.path.join(out_dir, f"{prefix}_{timestamp}.png")
            with open(path, "wb") as f:
                f.write(b64mod.b64decode(images[0]))
            out = {"success": True, "path": path, "is_diagram": is_diagram}
            if is_diagram:
                out["mermaid_code"] = result.get("mermaid_code", "")
            return json.dumps(out)
        return json.dumps({"success": False, "error": "No images in response"})
    except Exception as e:
        err_msg = str(e)
        if "ConnectError" in type(e).__name__ or "Connection refused" in err_msg:
            return json.dumps({
                "success": False,
                "error": "AitherCanvas not running locally. Use MCP bridge for cloud access.",
            })
        return json.dumps({"success": False, "error": err_msg})


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

# Tool category definitions
TOOL_CATEGORIES = {
    "file_io": [file_read, file_write, file_edit, file_list, file_search],
    "shell": [shell_exec],
    "python": [python_exec],
    "web": [web_search, web_fetch],
    "secrets": [secret_get, secret_set, secret_list],
    "creative": [image_generate, image_refine, image_smart],
}

# Default categories for common identity profiles
IDENTITY_DEFAULTS = {
    "demiurge": ["file_io", "shell", "python", "web"],
    "atlas": ["file_io", "web", "secrets"],
    "aither": ["file_io", "shell", "python", "web", "secrets", "creative"],
    "lyra": ["file_io", "web"],
    "hydra": ["file_io", "shell", "python"],
    "prometheus": ["file_io", "shell", "secrets"],
    "apollo": ["file_io", "shell", "python"],
    "athena": ["file_io", "web", "secrets"],
    "scribe": ["file_io", "web"],
    "iris": ["file_io", "web", "creative"],
    "muse": ["file_io", "web", "creative"],
}


def register_builtin_tools(
    agent: AitherAgent,
    categories: list[str] | None = None,
    auto: bool = True,
) -> int:
    """Register built-in tools on an agent.

    Args:
        agent: The AitherAgent to register tools on.
        categories: Specific categories to register. If None and auto=True,
                    picks based on agent identity name.
        auto: If True and categories is None, auto-detect from identity.

    Returns:
        Number of tools registered.
    """
    if categories is None and auto:
        categories = IDENTITY_DEFAULTS.get(agent.name, ["file_io", "web"])

    if categories is None:
        categories = list(TOOL_CATEGORIES.keys())

    count = 0
    for cat in categories:
        fns = TOOL_CATEGORIES.get(cat, [])
        for fn in fns:
            agent._tools.register(fn)
            count += 1

    if count:
        logger.info("Registered %d built-in tools (%s) on agent %s",
                     count, ", ".join(categories), agent.name)
    return count
