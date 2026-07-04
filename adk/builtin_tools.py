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
from typing import TYPE_CHECKING, Any

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
    import shlex

    # Block obviously dangerous patterns before execution
    _BLOCKED_PATTERNS = [
        "rm -rf /", "mkfs.", "dd if=", "> /dev/sd",
        ":(){ :|:", "chmod -R 777 /",
    ]
    cmd_lower = command.lower().strip()
    for pat in _BLOCKED_PATTERNS:
        if pat in cmd_lower:
            return json.dumps({"error": f"Blocked: dangerous pattern '{pat}'"})

    try:
        # On Unix, avoid shell=True to prevent injection.
        # On Windows, shell=True is needed for built-in commands.
        use_shell = sys.platform == "win32"
        cmd_arg = command if use_shell else shlex.split(command)
        result = subprocess.run(
            cmd_arg,
            shell=use_shell,
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


_SAFE_BUILTINS = {
    k: getattr(__builtins__, k) if hasattr(__builtins__, k) else __builtins__[k]
    for k in (
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict",
        "dir", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "getattr", "hasattr", "hash", "hex", "id", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip",
        # Exception types — code legitimately raises/catches these
        "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "RuntimeError", "NameError",
        "ZeroDivisionError", "ArithmeticError", "LookupError", "StopIteration",
        "AssertionError", "NotImplementedError", "OSError", "FileNotFoundError",
        "ImportError", "OverflowError",
    )
    if (hasattr(__builtins__, k) if isinstance(__builtins__, type) else k in __builtins__)
}
# Allow safe imports only
_ALLOWED_MODULES = {
    "math", "json", "re", "datetime", "collections", "itertools",
    "functools", "string", "textwrap", "statistics", "random",
    "pathlib", "csv", "io", "base64", "hashlib", "urllib.parse", "sys",
}


def _restricted_import(name, *args, **kwargs):
    if name.split(".")[0] not in _ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' not allowed in sandbox. "
            f"Allowed: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return __import__(name, *args, **kwargs)


def python_exec(code: str) -> str:
    """Execute Python code in a restricted sandbox and capture output.

    code: Python code to execute
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    safe_builtins = {**_SAFE_BUILTINS, "__import__": _restricted_import}
    namespace: dict = {"__builtins__": safe_builtins}
    result_val = None

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)  # noqa: S102
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


def _is_safe_url(url: str) -> bool:
    """Block SSRF: reject private IPs, localhost, metadata endpoints."""
    from urllib.parse import urlparse
    import ipaddress
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Block non-HTTP schemes
        if parsed.scheme not in ("http", "https"):
            return False
        # Block localhost and common internal hostnames
        if hostname in ("localhost", "0.0.0.0", "127.0.0.1", "[::]", "[::1]"):
            return False
        if hostname.startswith("169.254.") or hostname.startswith("fe80:"):
            return False  # Link-local / cloud metadata
        if hostname.endswith(".internal") or hostname.endswith(".local"):
            return False
        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass  # hostname, not IP — OK
        return True
    except Exception:
        return False


async def web_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a webpage and return its text content.

    url: URL to fetch
    max_chars: Maximum characters to return (default 20000)
    """
    if not _is_safe_url(url):
        return json.dumps({
            "error": "URL blocked: private/internal addresses not allowed"
        })
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, max_redirects=5,
        ) as client:
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
_SECRETS_FILE = Path(os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))) / "secrets.enc"
_SECRETS_FILE_LEGACY = _SECRETS_FILE.with_suffix(".json")


def _derive_key() -> bytes:
    """Derive an encryption key from machine-specific data."""
    import hashlib
    # Combine username + home dir + machine-specific salt
    material = f"{os.getlogin()}:{Path.home()}:aither-adk-secrets-v1"
    return hashlib.pbkdf2_hmac("sha256", material.encode(), b"adk-salt-v1", 100_000)


def _encrypt_secrets(data: dict) -> bytes:
    """XOR-based obfuscation with derived key. Not military-grade but prevents
    casual plaintext exposure in backups/cloud sync."""
    import hashlib
    key = _derive_key()
    plaintext = json.dumps(data).encode()
    stream = hashlib.sha256(key).digest()  # Expand key
    result = bytearray()
    for i, b in enumerate(plaintext):
        result.append(b ^ stream[i % len(stream)])
    return bytes(result)


def _decrypt_secrets(data: bytes) -> dict:
    """Reverse XOR obfuscation."""
    import hashlib
    key = _derive_key()
    stream = hashlib.sha256(key).digest()
    result = bytearray()
    for i, b in enumerate(data):
        result.append(b ^ stream[i % len(stream)])
    return json.loads(bytes(result).decode())


def _load_secrets() -> dict[str, str]:
    global _secrets_cache
    if _secrets_cache is not None:
        return _secrets_cache
    # Try encrypted file first
    if _SECRETS_FILE.exists():
        try:
            _secrets_cache = _decrypt_secrets(_SECRETS_FILE.read_bytes())
            return _secrets_cache
        except Exception:
            _secrets_cache = {}
    # Migrate legacy plaintext if exists
    if _SECRETS_FILE_LEGACY.exists():
        try:
            _secrets_cache = json.loads(
                _SECRETS_FILE_LEGACY.read_text(encoding="utf-8")
            )
            _save_secrets(_secrets_cache)  # Re-save encrypted
            _SECRETS_FILE_LEGACY.unlink()  # Remove plaintext
            return _secrets_cache
        except Exception:
            _secrets_cache = {}
    else:
        _secrets_cache = {}
    return _secrets_cache


def _save_secrets(data: dict[str, str]):
    global _secrets_cache
    _secrets_cache = data
    _SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SECRETS_FILE.write_bytes(_encrypt_secrets(data))
    # Restrict permissions
    try:
        os.chmod(_SECRETS_FILE, 0o600)
    except (OSError, AttributeError):
        pass  # Windows — best-effort


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
    # GATED: swarm coding drives the Genesis multi-agent engine — a paid-tier
    # capability. Free agents get a clear upgrade prompt instead of a dispatch.
    try:
        from adk.licensing import get_license_manager
        if not get_license_manager().can_use_swarm():
            return _json.dumps({
                "status": "failed",
                "error": (
                    "Swarm coding requires a Professional tier. Upgrade at "
                    "portal.aitherium.com/portal/marketplace/packs"
                ),
            })
    except ImportError:
        pass
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
# Self-introspection tools (B.3 from .AITHEROS/31-AGENT-LONGRUN-AUDIT.md)
#
# The Reddit pain point: "I asked the agent what it had done and it lied."
# Solution: first-class tools that let an agent read its own audit trail
# instead of hallucinating one. Data lives in `agent._introspection` (a
# bounded deque) and `agent._files_touched`, populated by the dispatch loop
# in adk/agent.py. Tools are closures over `agent` so each agent only sees
# its own history.
# ─────────────────────────────────────────────────────────────────────────────

def register_self_tools(agent: AitherAgent) -> int:
    """Register `self_*` introspection tools that let an agent inspect what it did.

    Tools registered:
      - self_recent_tool_calls(n)   : last N tool calls (tool, args, latency, error)
      - self_files_touched()        : files the agent has read/written/edited this session
      - self_session_summary()      : counts + budget + memory + identity overview
      - self_memory_search(query)   : search the agent's own memory tier only
    """

    def self_recent_tool_calls(n: int = 10) -> str:
        """Return the last N tool calls this agent made in the current process.

        Use this to honestly answer the user's question "what did you just do?"
        instead of guessing from chat context.

        n: how many recent calls to return (default 10, max 200).
        """
        try:
            buf = list(getattr(agent, "_introspection", ()))
            if not buf:
                return json.dumps({"calls": [], "count": 0,
                                    "note": "no tool calls recorded yet"})
            n = max(1, min(int(n or 10), 200))
            return json.dumps({"calls": buf[-n:], "count": len(buf[-n:]),
                                "total_recorded": len(buf)}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def self_files_touched() -> str:
        """Return every file path this agent has read/written/edited this session.

        Output: {path: {first_ts, last_ts, ops: [read|write|edit, ...]}}.
        Use this before claiming you modified a file \u2014 if it isn't here you didn't.
        """
        try:
            touched = dict(getattr(agent, "_files_touched", {}))
            return json.dumps({"files": touched, "count": len(touched)}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def self_session_summary() -> str:
        """Return a structured summary of this agent's current session.

        Includes: identity, session id, total tool calls by name, distinct files
        touched, per-agent meter (tokens/cost so far), memory entry count.
        """
        try:
            buf = list(getattr(agent, "_introspection", ()))
            by_tool: dict[str, int] = {}
            errs = 0
            for rec in buf:
                by_tool[rec.get("tool", "?")] = by_tool.get(rec.get("tool", "?"), 0) + 1
                if rec.get("error"):
                    errs += 1
            meter_snap: dict = {}
            try:
                meter_snap = agent.meter.snapshot() if hasattr(agent.meter, "snapshot") else {}
            except Exception:
                pass
            return json.dumps({
                "agent": agent.name,
                "session_id": getattr(agent, "_session_id", None),
                "tool_calls_total": len(buf),
                "tool_calls_by_name": by_tool,
                "tool_errors": errs,
                "files_touched": len(getattr(agent, "_files_touched", {})),
                "meter": meter_snap,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def self_memory_search(query: str, k: int = 5) -> str:
        """Search this agent's OWN memory tier (not the global graph) for past context.

        query: free-text search
        k: max results (default 5, max 25)
        """
        try:
            mem = getattr(agent, "memory", None)
            if mem is None:
                return json.dumps({"results": [], "note": "no memory backend"})
            k = max(1, min(int(k or 5), 25))
            # memory.search may be sync or async; coerce
            res = mem.search(query, k=k) if hasattr(mem, "search") else []
            if asyncio.iscoroutine(res):
                # We're inside a sync tool; surface the coroutine name and ask caller to use recall.
                return json.dumps({"results": [],
                                    "note": "memory.search is async; use the 'recall' tool"})
            return json.dumps({"results": list(res) if res else [], "count": len(res or [])},
                                default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    agent._tools.register(self_recent_tool_calls, name="self_recent_tool_calls",
        description="Return the last N tool calls this agent made (honest audit, no hallucination).")
    agent._tools.register(self_files_touched, name="self_files_touched",
        description="Return every file path this agent has read/written/edited this session.")
    agent._tools.register(self_session_summary, name="self_session_summary",
        description="Return a structured summary of this session: tool counts, files, meter, identity.")
    agent._tools.register(self_memory_search, name="self_memory_search",
        description="Search this agent's own memory tier for past context.")
    logger.info("Registered 4 self-introspection tools on agent %s", agent.name)
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# AitherGraph tools (calls Genesis /api/graph/ — requires AitherOS running)
# ─────────────────────────────────────────────────────────────────────────────


def _graph_url() -> str:
    return os.getenv("AITHER_GENESIS_URL", "http://localhost:8001") + "/api/graph"


def _graph_request(
    method: str, path: str, params: dict | None = None,
    payload: dict | None = None, timeout: float = 30,
) -> str:
    """HTTP request to Genesis /api/graph/."""
    import httpx
    url = f"{_graph_url()}{path}"
    try:
        _tls = os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false"
        with httpx.Client(timeout=timeout, verify=_tls) as c:
            if method == "GET":
                resp = c.get(url, params=params)
            else:
                resp = c.post(url, json=payload, params=params)
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def graph_search(query: str, domain: str = "", limit: int = 10) -> str:
    """Search across all AitherGraph domains (code, knowledge, events, memory).

    Args:
        query: What to search for.
        domain: Optional domain filter (code, knowledge, events, memory, etc.).
        limit: Max results.
    """
    params: dict = {"q": query, "limit": limit}
    if domain:
        params["domain"] = domain
    return _graph_request("GET", "/search", params=params)


def graph_code_search(query: str, limit: int = 10) -> str:
    """Search CodeGraph for functions, classes, and symbols.

    Args:
        query: Symbol name or search text.
        limit: Max results.
    """
    return _graph_request("GET", "/code/search", params={"q": query, "limit": limit})


def graph_kb_query(base_id: str, query: str, limit: int = 10) -> str:
    """RAG query against a knowledge base.

    Args:
        base_id: Knowledge base ID.
        query: Natural language query.
        limit: Max results.
    """
    return _graph_request(
        "POST", f"/knowledge/bases/{base_id}/query",
        params={"q": query, "limit": limit},
    )


def graph_memory_query(query: str, limit: int = 10) -> str:
    """Query the memory graph.

    Args:
        query: What to recall.
        limit: Max results.
    """
    return _graph_request(
        "POST", "/memory/query",
        payload={"query": query, "limit": limit},
    )


def graph_research(query: str, effort: str = "library_session") -> str:
    """Trigger autonomous research on a topic.

    Args:
        query: Research topic.
        effort: Depth (quick_glance, library_session, deep_dive, leave_no_stone).
    """
    return _graph_request(
        "POST", "/research",
        payload={"query": query, "effort": effort}, timeout=120,
    )


_GRAPH_TOOLS = [
    graph_search, graph_code_search, graph_kb_query,
    graph_memory_query, graph_research,
]


def _init_graph_tools():
    """Lazily populate the graph category."""
    if not TOOL_CATEGORIES.get("graph"):
        TOOL_CATEGORIES["graph"] = _GRAPH_TOOLS


# ─────────────────────────────────────────────────────────────────────────
# Workspace Intelligence tools — people analytics, meetings, email, collab
# ─────────────────────────────────────────────────────────────────────────

def _wi_base_url() -> str:
    """Resolve workspace intelligence base URL."""
    return os.getenv(
        "ADK_APP_PROXY_URL",
        os.getenv("AITHER_GENESIS_URL", "http://localhost:8001"),
    ).rstrip("/")


def _wi_get(path: str, params: dict | None = None) -> str:
    """GET workspace intelligence endpoint with graceful degradation."""
    import httpx
    url = f"{_wi_base_url()}{path}"
    try:
        resp = httpx.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
        return json.dumps({"error": f"HTTP {resp.status_code}", "detail": resp.text[:300]})
    except Exception as e:
        return json.dumps({"error": str(e)})


def workspace_health(days: int = 7) -> str:
    """Get workspace health score and engagement metrics.

    Returns composite health score (0-100), active users, engagement rate,
    meeting/email/message volume, activity trends, and top contributors.

    Args:
        days: Number of days to analyze (default 7)
    """
    return _wi_get("/api/workspace-intelligence/health", {"days": days})


def email_intelligence(days: int = 30) -> str:
    """Analyze email patterns: top senders, categories, busiest hours.

    Provides sender frequency, VIP/urgent/approval/FYI categorization,
    peak activity hours, and thread depth analysis.

    Args:
        days: Number of days to analyze (default 30)
    """
    return _wi_get("/api/workspace-intelligence/email-intelligence", {"days": days})


def meeting_intelligence(days: int = 30) -> str:
    """Analyze meeting patterns: time spent, types, top collaborators, free time.

    Shows total hours in meetings, type breakdown (1:1, team, all-hands),
    top meeting partners, busiest days, and free time percentage.

    Args:
        days: Number of days to analyze (default 30)
    """
    return _wi_get("/api/workspace-intelligence/meeting-intelligence", {"days": days})


def collaboration_signals(days: int = 30) -> str:
    """Get team collaboration metrics: interaction density, pairs, silos.

    Analyzes cross-channel collaboration from meetings, emails, messages.
    Identifies strongest pairs, communication flow, and potential silos.

    Args:
        days: Number of days to analyze (default 30)
    """
    return _wi_get("/api/workspace-intelligence/collaboration", {"days": days})


def person_intelligence(person_id: str, days: int = 30) -> str:
    """Get engagement and activity profile for a specific person.

    Returns engagement score, email volume, meetings, activity trends,
    and top contacts for the specified person.

    Args:
        person_id: Person identifier (email or user ID)
        days: Number of days to analyze (default 30)
    """
    return _wi_get(f"/api/workspace-intelligence/person/{person_id}", {"days": days})


def relationship_insights() -> str:
    """Get network relationship analytics from the social graph.

    Returns relationship type distribution, network density,
    key connectors, and isolated members needing support.
    """
    return _wi_get("/api/workspace-intelligence/relationships")


def post_to_social(text: str, platforms: str = "bluesky,linkedin") -> str:
    """Create a social media post across selected platforms.

    Args:
        text: Post content
        platforms: Comma-separated platform names (default: bluesky,linkedin)
    """
    import httpx
    url = f"{_wi_base_url()}/api/social/posts/draft"
    plat_list = [p.strip() for p in platforms.split(",") if p.strip()]
    try:
        resp = httpx.post(url, json={"text": text, "platforms": plat_list}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            post_id = data.get("id", "")
            if post_id:
                httpx.post(f"{_wi_base_url()}/api/social/posts/{post_id}/publish", timeout=10)
            return json.dumps(data, indent=2)
        return json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def social_analytics(days: int = 30) -> str:
    """Get social media engagement metrics and top posts.

    Args:
        days: Number of days to analyze (default 30)
    """
    return _wi_get("/api/social/analytics", {"days": days})


def executive_briefing() -> str:
    """Get unified morning briefing: calendar, priority emails, tasks, messages.

    Returns today's events, next meeting countdown, priority items
    (VIP emails, urgent tasks, imminent meetings), and unread counts.
    """
    return _wi_get("/api/executive/briefing")


def meeting_prep(event_id: str, title: str = "") -> str:
    """Prepare for a meeting: talking points, related docs, attendee context.

    Args:
        event_id: Calendar event ID to prepare for
        title: Meeting title for additional context
    """
    import httpx
    url = f"{_wi_base_url()}/api/executive/meeting/prep"
    try:
        resp = httpx.post(url, json={"event_id": event_id, "title": title}, timeout=15)
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
        return json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def email_triage(email_id: str, subject: str, sender: str, body: str) -> str:
    """Categorize an email by importance: VIP, urgent, approval, or FYI.

    Args:
        email_id: Email identifier
        subject: Email subject line
        sender: Sender email address
        body: Email body text
    """
    import httpx
    url = f"{_wi_base_url()}/api/executive/email/triage"
    try:
        resp = httpx.post(url, json={
            "email_id": email_id, "subject": subject,
            "sender": sender, "body": body,
        }, timeout=10)
        if resp.status_code == 200:
            return json.dumps(resp.json(), indent=2)
        return json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def rag_search(query: str, top_k: int = 5) -> str:
    """Search document knowledge base for relevant content.

    Args:
        query: Search query
        top_k: Number of results (default 5)
    """
    import httpx
    url = f"{_wi_base_url()}/api/documents/search"
    try:
        resp = httpx.post(url, json={"query": query, "top_k": top_k}, timeout=10)
        return resp.text if resp.status_code == 200 else json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def staff_search(query: str) -> str:
    """Search staff/team members by skills, role, department, or name.

    Args:
        query: Search criteria (name, skill, role, etc.)
    """
    import httpx
    url = f"{_wi_base_url()}/api/people/search"
    try:
        resp = httpx.post(url, json={"query": query}, timeout=10)
        return resp.text if resp.status_code == 200 else json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_document(document_type: str, instructions: str = "", context: str = "") -> str:
    """Generate a formatted document (report, summary, proposal, etc.).

    Args:
        document_type: Type of document (report, summary, proposal, resume, etc.)
        instructions: Additional generation instructions
        context: Background context for the document
    """
    import httpx
    url = f"{_wi_base_url()}/api/content/generate"
    try:
        resp = httpx.post(url, json={
            "document_type": document_type,
            "instructions": instructions, "context": context,
        }, timeout=30)
        return resp.text if resp.status_code == 200 else json.dumps({"error": f"HTTP {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# Domain tags for workspace tools — matches capability_domains.yaml
_WORKSPACE_TOOL_DOMAINS: dict[str, str] = {
    "workspace_health": "workspace_intelligence",
    "email_intelligence": "workspace_intelligence",
    "meeting_intelligence": "workspace_intelligence",
    "collaboration_signals": "workspace_intelligence",
    "person_intelligence": "workspace_intelligence",
    "relationship_insights": "workspace_intelligence",
    "post_to_social": "social_marketing",
    "social_analytics": "social_marketing",
    "executive_briefing": "executive_assistant",
    "meeting_prep": "executive_assistant",
    "email_triage": "executive_assistant",
    "rag_search": "documents",
    "staff_search": "people",
    "generate_document": "documents",
}

_WORKSPACE_TOOLS = [
    # Intelligence
    workspace_health, email_intelligence, meeting_intelligence,
    collaboration_signals, person_intelligence, relationship_insights,
    # Social / Marketing
    post_to_social, social_analytics,
    # Executive Assistant
    executive_briefing, meeting_prep, email_triage,
    # Documents / Knowledge
    rag_search, staff_search, generate_document,
]


def _init_workspace_tools():
    """Lazily populate the workspace category.

    Respects AITHER_ENABLED_DOMAINS env var (comma-separated).
    If unset, all workspace tools are registered.
    """
    if TOOL_CATEGORIES.get("workspace"):
        return
    enabled_raw = os.getenv("AITHER_ENABLED_DOMAINS", "")
    if enabled_raw:
        enabled = {d.strip() for d in enabled_raw.split(",") if d.strip()}
        TOOL_CATEGORIES["workspace"] = [
            fn for fn in _WORKSPACE_TOOLS
            if _WORKSPACE_TOOL_DOMAINS.get(fn.__name__, "") in enabled
        ]
    else:
        TOOL_CATEGORIES["workspace"] = _WORKSPACE_TOOLS


# ─────────────────────────────────────────────────────────────────────────
# Safety & Escalation tools
# ─────────────────────────────────────────────────────────────────────────

async def escalate_to_human(
    reason: str,
    action: str = "",
    urgency: str = "medium",
) -> str:
    """Request human review for uncertain or risky actions.

    Use when you're about to perform an irreversible action, when
    confidence is low, or when the task involves sensitive systems.

    Args:
        reason: Why escalation is needed
        action: The action requiring approval (e.g. deploy.production)
        urgency: low, medium, high, or critical
    """
    # Try AitherOS escalation endpoint first
    try:
        from adk.aither_bridge import get_bridge
        bridge = get_bridge()
        if bridge and bridge.connected:
            result = await bridge.post(
                "/escalations/create",
                {"reason": reason, "action_type": action, "urgency": urgency},
            )
            return json.dumps(result)
    except Exception:
        pass

    # Standalone fallback: log and return guidance
    logger.warning(
        "[ESCALATION] %s | action=%s urgency=%s",
        reason, action, urgency,
    )
    return json.dumps({
        "status": "logged_locally",
        "reason": reason,
        "action": action,
        "urgency": urgency,
        "guidance": (
            "No AitherOS connection — escalation logged locally. "
            "Proceed with caution or wait for human input."
        ),
    })


async def check_safety_gate(
    tool_name: str,
    args: str = "{}",
) -> str:
    """Check if a tool call is allowed by safety gates.

    Call this before performing risky operations to verify
    the action is within your permission level.

    Args:
        tool_name: Name of the tool to check
        args: JSON string of tool arguments
    """
    from adk.safety import ActionGate, GateDecision
    gate = ActionGate()
    action_type = gate.classify_tool(tool_name)
    try:
        parsed_args = json.loads(args) if args else {}
    except json.JSONDecodeError:
        parsed_args = {}
    result = gate.check(action_type, parsed_args)
    return json.dumps({
        "decision": result.decision.value,
        "reason": result.reason,
        "action_type": action_type,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Structured-data ML (TabFM tabular + TimesFM time-series) — zero-shot inference.
# These are NON-LLM foundation models served by a structured-ML inference service
# (default :8192). The tools POST directly to that service, resolved from the
# AITHER_STRUCTURED_ML_URL env var (set this to your deployment's service URL).
# The agent-side gate is pack-domain enablement — only identities with the
# "structured_ml" category get these tools. "Training" here is in-context: hand
# over a labeled support set, get predictions in one forward pass — no gradient step.
# ─────────────────────────────────────────────────────────────────────────────


_STRUCTURED_ML_DEFAULT_URL = "http://localhost:8192"


def _structured_ml_url() -> str:
    """Resolve the structured-ML service base URL from env, http/https only.

    Defense-in-depth: reject any URL whose scheme isn't http/https so a mis-set
    value can't redirect the tool to a file://, gopher:// (etc.) target. A
    private/loopback host is allowed here (unlike web_fetch) — the inference
    service is a trusted deployment endpoint set via AITHER_STRUCTURED_ML_URL.
    """
    url = os.getenv("AITHER_STRUCTURED_ML_URL", _STRUCTURED_ML_DEFAULT_URL).strip()
    if not url.lower().startswith(("http://", "https://")):
        return _STRUCTURED_ML_DEFAULT_URL
    return url


# Cap the JSON we hand back to the model — a big query batch can return a large
# predictions/probabilities matrix that would bloat the agent's context. Beyond
# this, we return a compact summary (counts + a small sample) instead of the raw
# blob, and tell the agent how to narrow the request.
_STRUCTURED_ML_MAX_RESP_CHARS = 20_000


def _structured_ml_post(path: str, payload: dict, timeout: float = 90.0) -> str:
    import httpx

    url = f"{_structured_ml_url()}{path}"
    try:
        _tls = os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false"
        with httpx.Client(timeout=timeout, verify=_tls) as c:
            resp = c.post(url, json=payload)
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:  # noqa: BLE001 - fall back to raw body
                    detail = resp.text
                return json.dumps({"error": f"HTTP {resp.status_code}: {detail}"})
            return _cap_response(resp.json())
    except Exception as e:  # noqa: BLE001 - surface transport errors to the agent
        return json.dumps({"error": str(e)})


def _cap_response(data: dict) -> str:
    """Serialise a service response, summarising it if it would bloat context."""
    full = json.dumps(data)
    if len(full) <= _STRUCTURED_ML_MAX_RESP_CHARS:
        return full
    preds = data.get("predictions") if isinstance(data, dict) else None
    summary: dict = {
        "truncated": True,
        "note": (
            "Response too large to return in full; showing a summary. Re-run on a "
            "smaller batch of query_rows/series to get complete per-row output."
        ),
    }
    if isinstance(preds, list):
        summary["n_predictions"] = len(preds)
        summary["predictions_sample"] = preds[:50]
    else:
        summary["keys"] = list(data.keys()) if isinstance(data, dict) else None
    return json.dumps(summary)


def tabular_classify(
    support_rows: list[dict[str, Any]], target: str, query_rows: list[dict[str, Any]]
) -> str:
    """Classify tabular rows with TabFM (zero-shot in-context; up to 10 classes).

    Hand over a few labeled examples as the support set and get predicted classes +
    probabilities for the query rows — no training step. Use this to adapt to a new
    labeled dataset on the fly instead of getting stuck.

    Args:
        support_rows: Labeled examples; each is a dict of features INCLUDING the target column.
        target: Name of the label column present in support_rows.
        query_rows: Rows (dicts of features) to classify.
    """
    return _structured_ml_post(
        "/tabular/classify",
        {"support_rows": support_rows, "target": target, "query_rows": query_rows},
    )


def tabular_regress(
    support_rows: list[dict[str, Any]], target: str, query_rows: list[dict[str, Any]]
) -> str:
    """Predict a numeric target for tabular rows with TabFM (zero-shot in-context).

    Args:
        support_rows: Labeled examples; each is a dict of features INCLUDING the numeric target column.
        target: Name of the numeric target column present in support_rows.
        query_rows: Rows (dicts of features) to predict.
    """
    return _structured_ml_post(
        "/tabular/regress",
        {"support_rows": support_rows, "target": target, "query_rows": query_rows},
    )


def timeseries_forecast(series: list[float], horizon: int) -> str:
    """Forecast future values of a time series with TimesFM (zero-shot).

    Args:
        series: History as a list of numbers. (Several series at once — a list of
            lists — is also accepted by the service for batch forecasting.)
        horizon: Number of future steps to forecast.
    """
    return _structured_ml_post(
        "/timeseries/forecast", {"series": series, "horizon": horizon}
    )


def tabular_teach(
    task: str, labeled_rows: list[dict[str, Any]], target: str, mode: str = "classify"
) -> str:
    """Teach a tabular task new labeled examples so it adapts to new data.

    Accumulates a per-task support set (kept under YOUR tenant) and only keeps the
    new rows if held-out accuracy doesn't regress (accept-or-rollback). This is how
    an agent gets better at a classification/regression task over time WITHOUT any
    training run — next `tabular_classify` on the same task can reuse what was taught.

    Args:
        task: A stable task name (e.g. "lead-scoring"); namespaced under your tenant.
        labeled_rows: New labeled examples; each a dict of features INCLUDING the target column.
        target: Name of the target column present in labeled_rows.
        mode: "classify" (default) or "regress".
    """
    import httpx

    url = f"{_genesis_url()}/ml/teach"
    payload = {"task": task, "labeled_rows": labeled_rows, "target": target, "mode": mode}
    try:
        _tls = os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false"
        with httpx.Client(timeout=90.0, verify=_tls) as c:
            resp = c.post(url, json=payload)
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:  # noqa: BLE001
                    detail = resp.text
                return json.dumps({"error": f"HTTP {resp.status_code}: {detail}"})
            return _cap_response(resp.json())
    except Exception as e:  # noqa: BLE001 - surface transport errors to the agent
        return json.dumps({"error": str(e)})


def _genesis_url() -> str:
    url = os.getenv("AITHER_GENESIS_URL", "http://localhost:8001").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "http://localhost:8001"
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

# Tool category definitions
TOOL_CATEGORIES: dict = {
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
    "structured_ml": [tabular_classify, tabular_regress, timeseries_forecast, tabular_teach],
    "graph": [],  # populated lazily by _init_graph_tools()
    "safety": [escalate_to_human, check_safety_gate],
    "workspace": [],  # populated lazily by _init_workspace_tools()
    # "self" is registered via register_self_tools(agent) (closures over agent state),
    # not a flat function list — see register_builtin_tools below.
    "self": [],
    "voice": [],  # populated lazily by _init_voice_tools()
    "formbridge": [],  # populated lazily by _init_formbridge_tools()
}

# Default categories for common identity profiles
# Every identity gets "self" by default — self-introspection is universally safe and
# directly addresses Reddit pain ("I asked the agent what it did and it lied").
IDENTITY_DEFAULTS = {
    "demiurge": ["file_io", "shell", "python", "web", "git", "code", "repowise", "swarm", "graph", "workspace", "safety", "self"],
    "analyst": ["file_io", "web", "python", "code", "graph", "structured_ml", "workspace", "safety", "self"],
    "atlas": ["file_io", "web", "secrets", "code", "graph", "workspace", "safety", "self"],
    "aither": ["file_io", "shell", "python", "web", "secrets", "creative", "git", "code", "repowise", "swarm", "graph", "workspace", "safety", "self"],
    "lyra": ["file_io", "web", "graph", "workspace", "voice", "safety", "self"],
    "hydra": ["file_io", "shell", "python", "git", "code", "repowise", "graph", "workspace", "safety", "self"],
    "prometheus": ["file_io", "shell", "secrets", "git", "workspace", "safety", "self"],
    "apollo": ["file_io", "shell", "python", "code", "repowise", "graph", "workspace", "safety", "self"],
    "athena": ["file_io", "web", "secrets", "code", "graph", "workspace", "safety", "self"],
    "scribe": ["file_io", "web", "code", "repowise", "graph", "workspace", "safety", "self"],
    "iris": ["file_io", "web", "creative", "workspace", "voice", "safety", "self"],
    "muse": ["file_io", "web", "creative", "workspace", "voice", "safety", "self"],
}


def _init_voice_tools():
    """Lazily populate the voice category.

    Voice tools are optional (require voice-local or voice-cloud extras).
    This function is called once at tool registration time.
    """
    if TOOL_CATEGORIES.get("voice"):
        return  # Already initialized
    try:
        # say_to_file (returns a path string), NOT say (returns raw bytes that would
        # be JSON-stringified into the model's context).
        from adk.builtin_tools_voice import hear, say_to_file, analyze_voice_emotion
        TOOL_CATEGORIES["voice"] = [hear, say_to_file, analyze_voice_emotion]
        logger.info("Voice tools initialized (hear, say_to_file, analyze_voice_emotion)")
    except ImportError:
        logger.debug("Voice tools not available; voice category remains empty")
        TOOL_CATEGORIES["voice"] = []


def _init_formbridge_tools():
    """Lazily populate the formbridge category (local form automation).

    Redacting-by-construction tools — results carry field NAMES/counts/job
    ids, never captured values (see adk/formbridge/tools.py).
    """
    if TOOL_CATEGORIES.get("formbridge"):
        return  # Already initialized
    try:
        from adk.formbridge.tools import (
            formbridge_fill_form,
            formbridge_list_patients,
            formbridge_pack_health,
            formbridge_purge,
        )
        TOOL_CATEGORIES["formbridge"] = [
            formbridge_list_patients,
            formbridge_fill_form,
            formbridge_pack_health,
            formbridge_purge,
        ]
        logger.info("FormBridge tools initialized (list/fill/health/purge)")
    except ImportError:
        logger.debug("FormBridge tools not available; category remains empty")
        TOOL_CATEGORIES["formbridge"] = []


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
    _init_graph_tools()  # lazily populate graph category
    _init_workspace_tools()  # lazily populate workspace category
    _init_voice_tools()  # lazily populate voice category
    _init_formbridge_tools()  # lazily populate formbridge category

    if categories is None and auto:
        # Unknown identities get a minimal, fully-local default. "workspace"
        # tools call a workspace-intelligence service and aren't useful to an
        # unknown standalone agent, so they're opt-in per named identity only.
        categories = IDENTITY_DEFAULTS.get(agent.name, ["file_io", "web"])

    if categories is None:
        categories = list(TOOL_CATEGORIES.keys())

    count = 0
    for cat in categories:
        if cat == "self":
            # Closure-based registration — each agent gets its own bound copies.
            count += register_self_tools(agent)
            continue
        fns = TOOL_CATEGORIES.get(cat)
        if fns is None:
            # Unknown category — try loading as a tool pack ID
            count += register_tool_packs(agent, pack_ids=[cat])
            continue
        for fn in fns:
            agent._tools.register(fn)
            count += 1

    if count:
        logger.info("Registered %d built-in tools (%s) on agent %s",
                     count, ", ".join(categories), agent.name)
    return count


def register_tool_packs(
    agent: AitherAgent,
    pack_ids: list[str] | None = None,
    packs_dir: str | None = None,
) -> int:
    """Discover and register ToolPack tools on an ADK agent.

    Args:
        agent: The AitherAgent to register tools on.
        pack_ids: Specific pack IDs to load. If None, loads all discovered.
        packs_dir: Additional directory to scan for packs.

    Returns:
        Number of tools registered.
    """
    try:
        from pathlib import Path as _P

        # Tool-pack loading is an optional add-on (adk.tool_pack_loader).
        # Standalone agents without it run with their built-in tools.
        try:
            from adk.tool_pack_loader import get_tool_pack_loader
        except ImportError:
            logger.debug("Tool-pack loader not installed; skipping pack load")
            return 0

        extra = [_P(packs_dir)] if packs_dir else []
        loader = get_tool_pack_loader(extra_dirs=extra)
        loader.discover()
        manifests = loader.load_packs(pack_ids) if pack_ids else list(loader._manifests.values())
        count = 0
        for manifest in manifests:
            allowed, denial = loader.check_license(manifest)
            if not allowed:
                logger.info("Pack %s denied: %s", manifest.id, denial)
                continue
            count += loader.register_on_adk_agent(manifest, agent)
        if count:
            logger.info("Registered %d tool-pack tools on agent %s", count, agent.name)
        return count
    except Exception as exc:
        logger.warning("Tool pack registration failed: %s", exc)
        return 0
