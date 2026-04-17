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
# Git tools — essential for coding agents
# ─────────────────────────────────────────────────────────────────────────────


def git_status(path: str = ".") -> str:
    """Show working tree status (modified, staged, untracked files)."""
    try:
        r = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=10, cwd=path,
        )
        return r.stdout or "(clean)"
    except Exception as e:
        return f"Error: {e}"


def git_diff(path: str = ".", staged: bool = False) -> str:
    """Show file changes. Set staged=true for staged changes only."""
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=path)
        return r.stdout[:20000] or "(no changes)"
    except Exception as e:
        return f"Error: {e}"


def git_log(path: str = ".", count: int = 10) -> str:
    """Show recent commit history."""
    try:
        r = subprocess.run(
            ["git", "log", f"-{count}", "--oneline", "--no-decorate"],
            capture_output=True, text=True, timeout=10, cwd=path,
        )
        return r.stdout or "(no commits)"
    except Exception as e:
        return f"Error: {e}"


def git_add(files: str, path: str = ".") -> str:
    """Stage files for commit. Use '.' for all changes."""
    try:
        r = subprocess.run(
            ["git", "add"] + files.split(),
            capture_output=True, text=True, timeout=10, cwd=path,
        )
        return r.stdout + r.stderr or "Staged"
    except Exception as e:
        return f"Error: {e}"


def git_commit(message: str, path: str = ".") -> str:
    """Create a commit with the given message."""
    try:
        r = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, timeout=15, cwd=path,
        )
        return r.stdout + r.stderr
    except Exception as e:
        return f"Error: {e}"


def git_branch_list(path: str = ".") -> str:
    """List all branches, marking the current one."""
    try:
        r = subprocess.run(
            ["git", "branch", "-a", "--no-color"],
            capture_output=True, text=True, timeout=10, cwd=path,
        )
        return r.stdout or "(no branches)"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Code search — grep/ripgrep for coding agents
# ─────────────────────────────────────────────────────────────────────────────


def code_search(pattern: str, path: str = ".", file_glob: str = "", max_results: int = 50) -> str:
    """Search code for a regex pattern. Uses ripgrep if available, falls back to grep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in.
        file_glob: Optional file pattern filter (e.g. '*.py', '*.ts').
        max_results: Max matching lines to return.
    """
    # Try ripgrep first (much faster)
    for cmd_name in ["rg", "grep"]:
        try:
            cmd = [cmd_name, "-n", "--no-heading"]
            if cmd_name == "rg":
                cmd.extend(["--max-count", str(max_results)])
                if file_glob:
                    cmd.extend(["--glob", file_glob])
            elif cmd_name == "grep":
                cmd.extend(["-r", f"--include={file_glob}" if file_glob else "-r"])
            cmd.extend([pattern, path])

            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            output = r.stdout[:30000]
            lines = output.strip().split("\n")
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"... ({len(output.strip().split(chr(10)))} total matches, showing {max_results})")
            return "\n".join(lines) or "(no matches)"
        except FileNotFoundError:
            continue
        except Exception as e:
            return f"Error: {e}"
    return "Error: neither rg nor grep found"


def repowise_search(query: str, max_results: int = 10) -> str:
    """Search codebase using Repowise semantic + keyword hybrid search.

    Uses the Repowise intelligence service for deep code understanding.
    Falls back to ripgrep code_search if Repowise is unavailable.

    Args:
        query: Natural language or keyword query
        max_results: Maximum results to return
    """
    import json as _json
    repowise_url = os.environ.get("AITHER_REPOWISE_URL", "http://localhost:7337")
    try:
        import httpx
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{repowise_url}/v1/search",
                json={"query": query, "limit": max_results},
            )
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "file": r.get("file", ""),
                        "symbol": r.get("symbol", ""),
                        "snippet": r.get("snippet", "")[:200],
                        "score": round(r.get("score", 0), 3),
                    })
                return _json.dumps({"results": results, "count": len(results), "source": "repowise"})
    except Exception:
        pass
    # Fallback to ripgrep
    return code_search(pattern=query, max_results=max_results)


def swarm_code(problem: str, mode: str = "forge", effort: int = 8) -> str:
    """Dispatch to AitherOS swarm coding engine for complex implementation tasks.

    The swarm runs 11 specialized agents in 4 phases:
    ARCHITECT -> SWARM (8 parallel) -> REVIEW -> JUDGE

    Args:
        problem: Task or feature description to implement
        mode: "llm" (text-only), "forge" (with tools/sandbox), "plan_only" (design only)
        effort: Effort level 1-10 (affects model selection)
    """
    import json as _json
    genesis_url = os.environ.get("AITHER_GENESIS_URL", "http://localhost:8001")
    try:
        import httpx
        resp = httpx.post(
            f"{genesis_url}/swarm/code/sync",
            json={"problem": problem, "mode": mode, "effort": effort},
            timeout=600,
        )
        if resp.status_code == 200:
            data = resp.json()
            return _json.dumps({
                "status": data.get("status", "unknown"),
                "plan": data.get("architect_plan", "")[:2000],
                "code": data.get("code", "")[:5000],
                "tests": data.get("tests", "")[:2000],
                "artifacts": data.get("artifacts", []),
            })
        return _json.dumps({"error": f"Genesis returned {resp.status_code}"})
    except Exception as e:
        return _json.dumps({"error": str(e)})


def code_symbols(path: str, pattern: str = "") -> str:
    """List function/class definitions in a file. Optionally filter by pattern."""
    import ast as _ast
    try:
        source = Path(path).read_text(encoding="utf-8")
        tree = _ast.parse(source)
        symbols = []
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                name = f"def {node.name}() line {node.lineno}"
                if not pattern or pattern.lower() in node.name.lower():
                    symbols.append(name)
            elif isinstance(node, _ast.ClassDef):
                name = f"class {node.name} line {node.lineno}"
                if not pattern or pattern.lower() in node.name.lower():
                    symbols.append(name)
        return "\n".join(symbols) or "(no symbols found)"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Faculty Graph Tools (registered when agent.set_code_graph/set_memory_graph)
# ─────────────────────────────────────────────────────────────────────────────


def _register_code_graph_tools(agent: "AitherAgent", code_graph) -> int:
    """Register CodeGraph-backed tools on an agent.

    Called by agent.set_code_graph(). Adds code_search and code_context tools.
    """
    import asyncio as _asyncio

    def cg_search(query: str, max_results: int = 10) -> str:
        """Search indexed code for functions/classes matching a query.

        query: Natural language or keyword query (e.g. 'authentication middleware')
        max_results: Maximum results to return (default 10)
        """
        try:
            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    results = pool.submit(_asyncio.run, code_graph.query(query, max_results=max_results)).result(timeout=30)
            else:
                results = _asyncio.run(code_graph.query(query, max_results=max_results))
            items = []
            for chunk in results:
                items.append({
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "file": chunk.source_path,
                    "line": chunk.start_line,
                    "signature": chunk.signature,
                    "calls": chunk.calls[:5],
                    "called_by": chunk.called_by[:5],
                })
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def cg_context(chunk_id: str) -> str:
        """Get full context for a code chunk including callers and callees.

        chunk_id: The chunk ID from a code_search result
        """
        try:
            ctx = code_graph.get_context_for_chunk(chunk_id)
            if not ctx:
                return json.dumps({"error": "Chunk not found"})
            chunk = ctx["chunk"]
            result = {
                "name": chunk.name,
                "signature": chunk.signature,
                "docstring": chunk.docstring,
                "file": chunk.source_path,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "callers": [{"name": c.name, "file": c.source_path} for c in ctx.get("callers", [])],
                "callees": [{"name": c.name, "file": c.source_path} for c in ctx.get("callees", [])],
            }
            body = code_graph.get_full_body(chunk_id)
            if body:
                result["body"] = body[:5000]
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    agent._tools.register(cg_search, name="code_search", description="Search indexed code for functions/classes matching a query")
    agent._tools.register(cg_context, name="code_context", description="Get full context for a code chunk including callers and callees")
    logger.info("Registered CodeGraph tools (code_search, code_context) on agent %s", agent.name)
    return 2


def _register_memory_graph_tools(agent: "AitherAgent", memory_graph) -> int:
    """Register MemoryGraph-backed tools on an agent.

    Called by agent.set_memory_graph(). Adds mg_remember, mg_recall, mg_query tools.
    """
    from types import SimpleNamespace
    import hashlib as _hl

    def mg_remember(content: str, title: str = "", tags: str = "") -> str:
        """Store a memory in the agent's knowledge graph.

        content: The memory content to store
        title: Short title for the memory (optional)
        tags: Comma-separated tags (optional)
        """
        try:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            mid = _hl.md5(content[:200].encode()).hexdigest()
            mem = SimpleNamespace(
                id=mid,
                title=title or content[:80],
                content=content,
                memory_type="episodic",
                tags=tag_list,
                source_agent=agent.name,
                importance=0.5,
                embedding=None,
                created_at=time.time(),
                archived=False,
                scope="shared",
            )
            memory_graph.add_node(mem, upsert=True)
            memory_graph.save()
            return json.dumps({"success": True, "id": mid})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def mg_recall(query: str, max_results: int = 5) -> str:
        """Search agent memory for relevant past knowledge.

        query: What to search for in memory
        max_results: Maximum memories to return (default 5)
        """
        try:
            results = memory_graph.hybrid_query(query, max_results=max_results)
            items = []
            for node, score in results:
                mem = node.memory
                items.append({
                    "title": getattr(mem, "title", ""),
                    "content": getattr(mem, "content", "")[:500],
                    "tags": list(getattr(mem, "tags", []) or []),
                    "score": score,
                })
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def mg_stats() -> str:
        """Get memory graph statistics (node count, edges, etc.)."""
        try:
            stats = memory_graph.get_stats()
            return json.dumps(stats)
        except Exception as e:
            return json.dumps({"error": str(e)})

    agent._tools.register(mg_remember, name="remember", description="Store a memory in the agent's knowledge graph")
    agent._tools.register(mg_recall, name="recall", description="Search agent memory for relevant past knowledge")
    agent._tools.register(mg_stats, name="memory_stats", description="Get memory graph statistics")
    logger.info("Registered MemoryGraph tools (remember, recall, memory_stats) on agent %s", agent.name)
    return 3


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
    "git": [git_status, git_diff, git_log, git_add, git_commit, git_branch_list],
    "code": [code_search, code_symbols],
    "repowise": [repowise_search],
    "swarm": [swarm_code],
}

# Default categories for common identity profiles
IDENTITY_DEFAULTS = {
    "demiurge": ["file_io", "shell", "python", "web", "git", "code", "repowise", "swarm"],
    "atlas": ["file_io", "web", "secrets", "code"],
    "aither": ["file_io", "shell", "python", "web", "secrets", "creative", "git", "code", "repowise", "swarm"],
    "lyra": ["file_io", "web"],
    "hydra": ["file_io", "shell", "python", "git", "code", "repowise"],
    "prometheus": ["file_io", "shell", "secrets", "git"],
    "apollo": ["file_io", "shell", "python", "code", "repowise"],
    "athena": ["file_io", "web", "secrets", "code"],
    "scribe": ["file_io", "web", "code", "repowise"],
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
