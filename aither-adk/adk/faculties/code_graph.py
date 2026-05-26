#!/usr/bin/env python3
"""
CodeGraph - Python AST Indexer with Call Graph
===============================================

A focused, single-purpose indexer that does ONE thing well:
Parse Python code into semantic chunks with full call graph.

WHAT IT DOES:
1. Real AST parsing (not regex)
2. Extracts functions, classes, methods
3. Builds call graph: what calls what, what is called by what
4. Outputs chunks ready for embedding

WHAT IT DOESN'T DO:
- PDFs, web pages, "universal" anything
- That's for other neurons to handle

USAGE:
------
    from adk.faculties.code_graph import CodeGraph

    graph = CodeGraph()

    # Index a codebase
    await graph.index_codebase("/path/to/code")

    # Query with call graph awareness
    chunks = await graph.query("rate limiter")

    for chunk in chunks:
        print(f"{chunk.name} calls: {chunk.calls}")
        print(f"{chunk.name} called by: {chunk.called_by}")

Author: AitherOS / ADK
Version: 1.0.0
"""

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from adk.faculties.base import BaseFacultyGraph, GraphSyncConfig

# Optional: numpy for fast vector operations
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

# Embedding engine — routes to sentence-transformers, Ollama, Elysium, or feature hash
try:
    from adk.faculties.embeddings import get_embedding_provider
    _HAS_EMBEDDING_ENGINE = True
except ImportError:
    _HAS_EMBEDDING_ENGINE = False

# Optional: vLLM backend (OpenAI-compatible API for generation)
_vllm_url: Optional[str] = None
_vllm_model: Optional[str] = None  # actual model name served by vLLM
_vllm_checked = False

def _detect_vllm() -> Optional[str]:
    """Detect running vLLM instance. Returns base URL or None."""
    global _vllm_url, _vllm_model, _vllm_checked
    if _vllm_checked:
        return _vllm_url
    _vllm_checked = True
    # Env var takes priority
    url = os.environ.get("VLLM_URL") or os.environ.get("NVIDIA_NIM_URL")
    if url:
        _vllm_url = url.rstrip("/")
    # Probe common ports
    if not _vllm_url:
        try:
            import httpx
            for port in (8120, 8116):
                try:
                    r = httpx.get(f"http://localhost:{port}/v1/models", timeout=0.3)
                    if r.status_code == 200:
                        _vllm_url = f"http://localhost:{port}"
                        # Extract the actual model name from vLLM
                        data = r.json()
                        if data.get("data"):
                            _vllm_model = data["data"][0]["id"]
                        break
                except Exception:
                    continue
        except ImportError:
            pass
    return _vllm_url

def _vllm_model_name() -> str:
    """Get the actual model name served by vLLM."""
    return _vllm_model or "deepseek-r1:14b"


# Model mapping — vLLM serves deepseek-r1:14b
ELASTIC_REFLEX = "llama3.2:latest"       # Fast: neurons, rerank, embeddings
ELASTIC_AGENT = "mistral-nemo:latest"    # Balanced: agent tasks, tool calling
ELASTIC_REASON = "deepseek-r1:14b"       # Deep: analysis, complex reasoning


logger = logging.getLogger("adk.faculties.code_graph")


async def _embed_texts(texts: List[str], model: str = "nomic-embed-text") -> List[Optional[list]]:
    """Embed texts via EmbeddingProvider (sentence-transformers / Ollama / Elysium / feature hash)."""
    if _HAS_EMBEDDING_ENGINE:
        try:
            provider = get_embedding_provider()
            return await provider.embed_batch(texts)
        except Exception as e:
            logger.warning(f"EmbeddingProvider failed: {e}")
    return [None] * len(texts)


async def _llm_generate(prompt: str, model: str = ELASTIC_REFLEX,
                         temperature: float = 0.0, max_tokens: int = 200) -> str:
    """Generate text via vLLM."""
    vllm = _detect_vllm()
    if vllm:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    f"{vllm}/v1/chat/completions",
                    json={
                        "model": _vllm_model_name(),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"vLLM generate failed: {e}")

    return ""

_HAS_LLM = True  # At least one backend check succeeded at call time


# ============================================================================
# PICKLE HMAC INTEGRITY HELPERS
# ============================================================================

def _pickle_hmac_key() -> bytes:
    """Get the HMAC key for pickle integrity validation."""
    secret = os.environ.get("AITHER_INTERNAL_SECRET", "")
    if not secret:
        logger.warning(
            "AITHER_INTERNAL_SECRET not set — using default pickle HMAC key. "
            "Set this env var in production!"
        )
        secret = "aither-pickle-hmac-default"
    return secret.encode()


def _compute_file_hmac(filepath: str) -> str:
    """Compute HMAC-SHA256 of a file's contents."""
    import hmac as _hmac_mod, hashlib as _hashlib_mod
    with open(filepath, "rb") as f:
        data = f.read()
    return _hmac_mod.new(_pickle_hmac_key(), data, _hashlib_mod.sha256).hexdigest()


def _verify_pickle_hmac(filepath: str) -> bool:
    """Verify HMAC sidecar for a pickle file. Returns True if valid or legacy (no sidecar)."""
    import hmac as _hmac_mod
    hmac_path = filepath + ".hmac"
    if not os.path.exists(hmac_path):
        logger.warning("[CodeGraph] No HMAC sidecar for %s — refusing to load (rebuild required)", filepath)
        return False
    try:
        with open(hmac_path, "r") as f:
            stored = f.read().strip()
        computed = _compute_file_hmac(filepath)
        if not _hmac_mod.compare_digest(stored, computed):
            logger.error(f"[CodeGraph] HMAC mismatch for {filepath} — cache tampered, deleting")
            os.unlink(filepath)
            os.unlink(hmac_path)
            return False
        return True
    except Exception as e:
        logger.error("[CodeGraph] HMAC verification error for %s: %s — refusing to load", filepath, e)
        return False


def _write_pickle_hmac(filepath: str) -> None:
    """Write HMAC sidecar after saving a pickle file."""
    try:
        hmac_val = _compute_file_hmac(filepath)
        with open(filepath + ".hmac", "w") as f:
            f.write(hmac_val)
    except Exception as e:
        logger.warning(f"[CodeGraph] Failed to write HMAC sidecar: {e}")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ChunkType(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass
class CodeChunk:
    """
    A semantic chunk of Python code with full relationship data.

    This is NOT just "text found at line X" - it's a node in the
    code graph with edges to what it calls and what calls it.
    """
    id: str
    name: str
    chunk_type: ChunkType
    source_path: str

    # Location
    start_line: int
    end_line: int

    # Semantic content
    signature: str = ""  # Full signature: "async def foo(x: int) -> str"
    docstring: str = ""
    body_preview: str = ""  # First N chars of body

    # Imports used by this chunk
    imports: List[str] = field(default_factory=list)
    import_map: Dict[str, str] = field(default_factory=dict)  # local_name -> full_module.name

    # CALL GRAPH - the key insight
    calls: List[str] = field(default_factory=list)  # What functions/methods this code calls
    called_by: List[str] = field(default_factory=list)  # What calls this (backfilled)

    # For classes
    base_classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    # For methods
    parent_class: Optional[str] = None

    # Quality metrics
    complexity: int = 0  # Cyclomatic complexity estimate
    line_count: int = 0

    # Embedding (populated by Mind service later)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.chunk_type.value}_{self.name}_{hashlib.sha256(self.source_path.encode()).hexdigest()[:8]}"


@dataclass
class FileGraph:
    """Graph of all chunks in a file."""
    source_path: str
    chunks: List[CodeChunk] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    import_map: Dict[str, str] = field(default_factory=dict)
    module_docstring: str = ""
    parse_errors: List[str] = field(default_factory=list)
    processing_ms: float = 0.0


# ============================================================================
# AST VISITOR - Extracts calls from function bodies
# ============================================================================

class CallExtractor(ast.NodeVisitor):
    """Extracts all function/method calls from an AST node."""

    def __init__(self):
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call):
        """Extract the name of what's being called or passed as reference."""
        if isinstance(node.func, ast.Name):
            # Simple call: foo()
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.foo()
            if isinstance(node.func.value, ast.Name):
                self.calls.append(f"{node.func.value.id}.{node.func.attr}")
            else:
                self.calls.append(node.func.attr)

        # Also extract any variables/functions passed as arguments
        # (e.g., mcp.tool()(my_target_function) or map(my_func, items))
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self.calls.append(arg.id)
            elif isinstance(arg, ast.Attribute):
                if isinstance(arg.value, ast.Name):
                    self.calls.append(f"{arg.value.id}.{arg.attr}")
                else:
                    self.calls.append(arg.attr)

        # Continue visiting children
        self.generic_visit(node)


def extract_calls(node: ast.AST) -> List[str]:
    """Extract all function/method calls from an AST node."""
    extractor = CallExtractor()
    extractor.visit(node)
    return list(set(extractor.calls))  # Deduplicate


def get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build full function signature from AST node."""
    args = []

    # Regular args
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception as e:
                logger.debug(f"[CallExtractor.get_signature] Operation failed: {e}")
        args.append(arg_str)

    # Return type
    returns = ""
    if node.returns:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception as e:
            logger.debug(f"[CallExtractor.get_signature] Operation failed: {e}")

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args)}){returns}"


def estimate_complexity(node: ast.AST) -> int:
    """Estimate cyclomatic complexity (simplified)."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


# ============================================================================
# PARSER - Single file parsing
# ============================================================================

def parse_file_sync(file_path: str) -> FileGraph:
    """
    Parse a single Python file into a FileGraph.

    This is the CPU-bound work that runs in a process pool.
    """
    start = time.perf_counter()
    graph = FileGraph(source_path=file_path)

    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        graph.imports = []

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            graph.parse_errors.append(str(e))
            graph.processing_ms = (time.perf_counter() - start) * 1000
            return graph

        # Module-level docstring
        graph.module_docstring = ast.get_docstring(tree) or ""

        # Extract imports first
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    graph.imports.append(alias.name)
                    graph.import_map[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    graph.imports.append(node.module)
                    for alias in node.names:
                        graph.import_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        # First pass: extract all functions and classes
        class_methods: Dict[str, List[str]] = defaultdict(list)  # class_name -> [method_names]

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunk = _extract_class(node, lines, file_path, graph.imports, graph.import_map)
                graph.chunks.append(chunk)

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = _extract_method(item, lines, file_path, node.name, graph.imports, graph.import_map)
                        graph.chunks.append(method_chunk)
                        class_methods[node.name].append(item.name)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = _extract_function(node, lines, file_path, graph.imports, graph.import_map)
                graph.chunks.append(chunk)

        # Update class chunks with their methods
        for chunk in graph.chunks:
            if chunk.chunk_type == ChunkType.CLASS:
                chunk.methods = class_methods.get(chunk.name, [])

    except Exception as e:
        graph.parse_errors.append(str(e))

    graph.processing_ms = (time.perf_counter() - start) * 1000
    return graph


def _extract_class(node: ast.ClassDef, lines: List[str], source_path: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a class definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line

    # Get base classes
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            bases.append(base.attr)

    # Get docstring
    docstring = ast.get_docstring(node) or ""

    # Build signature
    signature = f"class {node.name}"
    if bases:
        signature += f"({', '.join(bases)})"

    # Extract calls from class body (decorators, default values, etc.)
    calls = extract_calls(node)

    return CodeChunk(
        id=f"class_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=node.name,
        chunk_type=ChunkType.CLASS,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+10, end_line)])[:500],
        imports=[i for i in imports if i in bases],  # Relevant imports
        import_map=import_map,
        calls=calls,
        base_classes=bases,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], source_path: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a function definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line

    docstring = ast.get_docstring(node) or ""
    signature = get_signature(node)
    calls = extract_calls(node)

    return CodeChunk(
        id=f"func_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=node.name,
        chunk_type=ChunkType.FUNCTION,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+10, end_line)])[:500],
        imports=imports,
        import_map=import_map,
        calls=calls,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


def _extract_method(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], source_path: str, class_name: str, imports: List[str], import_map: Dict[str, str]) -> CodeChunk:
    """Extract a method definition."""
    start_line = node.lineno
    end_line = node.end_lineno or start_line

    docstring = ast.get_docstring(node) or ""
    signature = get_signature(node)
    calls = extract_calls(node)

    return CodeChunk(
        id=f"method_{class_name}_{node.name}_{hashlib.sha256(source_path.encode()).hexdigest()[:8]}",
        name=f"{class_name}.{node.name}",
        chunk_type=ChunkType.METHOD,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        docstring=docstring[:500],
        body_preview="\n".join(lines[start_line-1:min(start_line+5, end_line)])[:300],
        imports=imports,
        import_map=import_map,
        calls=calls,
        parent_class=class_name,
        complexity=estimate_complexity(node),
        line_count=end_line - start_line + 1,
    )


# ============================================================================
# FAST FILE DISCOVERY
# ============================================================================

async def discover_python_files(root: Path) -> Tuple[List[Path], float]:
    """Use ripgrep/fd for fast file discovery."""
    start = time.perf_counter()
    files: List[Path] = []

    # Try fd first (note: fd syntax is `fd <PATTERN> <PATH>`, use "." to match all)
    try:
        result = await asyncio.create_subprocess_exec(
            "fd", ".", str(root),
            "-e", "py", "--type", "f",
            "--exclude", ".git",
            "--exclude", "node_modules",
            "--exclude", "__pycache__",
            "--exclude", ".venv",
            "--exclude", "venv",
            "--exclude", "Worktrees",
            "--exclude", ".worktrees",
            "--exclude", "site-packages",
            "--exclude", "runtime",
            "--exclude", "training-data",
            "--exclude", "test_artifacts",
            "--exclude", "_archive",
            "--exclude", "external",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()

        if result.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    files.append(Path(line))
            return files, (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        logger.debug(f"[CallExtractor.discover_python_files] Operation failed: {e}")

    # Fallback to ripgrep
    try:
        result = await asyncio.create_subprocess_exec(
            "rg", "--files", "-g", "*.py",
            "--glob", "!.git/**",
            "--glob", "!node_modules/**",
            "--glob", "!__pycache__/**",
            "--glob", "!.venv/**",
            "--glob", "!venv/**",
            "--glob", "!Worktrees/**",
            "--glob", "!.worktrees/**",
            "--glob", "!site-packages/**",
            "--glob", "!runtime/**",
            "--glob", "!training-data/**",
            "--glob", "!test_artifacts/**",
            "--glob", "!_archive/**",
            "--glob", "!external/**",
            str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()

        if result.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    p = Path(line)
                    files.append(p if p.is_absolute() else root / p)
            return files, (time.perf_counter() - start) * 1000
    except FileNotFoundError as e:
        logger.debug(f"[CallExtractor.discover_python_files] Operation failed: {e}")

    # Final fallback — rglob is sync I/O; offload to thread so we don't
    # block the event loop (especially dangerous on Docker bind mounts).
    _RGLOB_EXCLUDE_PARTS = frozenset({
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "Worktrees", "worktrees", ".worktrees",
        "site-packages", "dist-packages",
        "runtime", "training-data", "test_artifacts",
        "_archive",
        "Canvas-Studio", "external", "AEON_PORTRAITS",
        "affect_gallery", "simulator-temp",
    })
    _RGLOB_MAX_FILES = 2000  # Safety cap — prevents runaway crawl
    _RGLOB_TIMEOUT_S = 30    # Max seconds for rglob before giving up

    def _sync_rglob():
        found = []
        deadline = time.perf_counter() + _RGLOB_TIMEOUT_S
        for f in root.rglob("*.py"):
            if time.perf_counter() > deadline:
                logger.warning(
                    f"[CodeGraph] rglob timeout after {_RGLOB_TIMEOUT_S}s "
                    f"({len(found)} files found so far) — indexing partial set"
                )
                break
            if _RGLOB_EXCLUDE_PARTS.isdisjoint(f.parts):
                found.append(f)
                if len(found) >= _RGLOB_MAX_FILES:
                    logger.info(
                        f"[CodeGraph] rglob hit {_RGLOB_MAX_FILES} file cap — "
                        f"install fd-find for faster/complete discovery"
                    )
                    break
        return found

    files.extend(await asyncio.to_thread(_sync_rglob))
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms > 5000:
        logger.warning(
            f"[CodeGraph] Slow file discovery: {elapsed_ms:.0f}ms for {len(files)} files. "
            f"Install fd-find in Docker image to fix this."
        )
    return files, elapsed_ms


# ============================================================================
# MAIN CODE GRAPH CLASS
# ============================================================================

class CodeGraph(BaseFacultyGraph):
    """
    Python code indexer with full call graph.

    Indexes Python code and builds a graph of:
    - What each function/method calls
    - What calls each function/method (backfilled)
    """

    _QUERY_CACHE_MAX = 512  # Max cached query embeddings (~1.5MB at 768-dim)

    def __init__(self, max_workers: int = 8):
        super().__init__()
        self._sync_config = GraphSyncConfig(
            enabled=True,
            domain="code",
            source_graph="CodeGraph",
            batch_size=50,
        )
        self.max_workers = max_workers

        # The index
        self.chunks: Dict[str, CodeChunk] = {}  # id -> chunk
        self.by_name: Dict[str, List[str]] = defaultdict(list)  # name -> [chunk_ids]
        self.by_file: Dict[str, List[str]] = defaultdict(list)  # file -> [chunk_ids]
        self.by_class: Dict[str, List[str]] = defaultdict(list)  # parent_class -> [chunk_ids]

        # Query embedding cache: query_text -> embedding vector
        self._query_embed_cache: Dict[str, list] = {}
        self._query_cache_order: List[str] = []  # LRU eviction order

        # Keyword query result cache: "query:max_results" -> List[CodeChunk]
        # Brute-force scan of 28K chunks takes ~60ms; cache turns repeats to ~0ms
        self._keyword_result_cache: Dict[str, List] = {}
        self._keyword_cache_order: List[str] = []  # LRU eviction order
        self._KEYWORD_CACHE_MAX = 64

        # Pre-computed embedding matrix for fast cosine similarity
        # Building np.array from 28K chunk embeddings takes ~500ms — do it ONCE
        self._embedding_matrix: Optional[Any] = None  # np.ndarray or None
        self._embedding_ids: Optional[List[str]] = None  # chunk IDs aligned with matrix rows
        self._embedding_norms: Optional[Any] = None  # pre-computed row norms

        # Stats
        self.total_files = 0
        self.discovery_ms = 0.0
        self.parsing_ms = 0.0
        self.backfill_ms = 0.0

        # Full body cache: chunk_id -> full source text (lazy loaded)
        self._body_cache: Dict[str, str] = {}
        self._root_path: Optional[str] = None

    async def index_codebase(
        self,
        root_path: str,
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Index a codebase with full call graph analysis.

        Returns stats about the indexing operation.
        """
        root = Path(root_path)
        total_start = time.perf_counter()

        # Phase 1: Discovery
        if on_progress:
            on_progress(0.0, "Discovering files...")

        files, self.discovery_ms = await discover_python_files(root)
        self.total_files = len(files)

        logger.info(f"Discovered {len(files)} files in {self.discovery_ms:.0f}ms")

        if on_progress:
            on_progress(0.1, f"Found {len(files)} files")

        # Phase 2: Parse in parallel
        if on_progress:
            on_progress(0.15, "Parsing files...")

        parse_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        # In Docker, ProcessPoolExecutor reliably crashes due to /dev/shm limits.
        # Use ThreadPoolExecutor directly in Docker to avoid BrokenProcessPool spam.
        _in_docker = os.path.exists("/.dockerenv")

        results: List[FileGraph] = []
        pool_type = "thread" if _in_docker else "process"

        if pool_type == "process":
            try:
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    tasks = [
                        loop.run_in_executor(pool, parse_file_sync, str(f))
                        for f in files
                    ]
                    for i, coro in enumerate(asyncio.as_completed(tasks)):
                        result = await coro
                        results.append(result)
                        if on_progress and (i + 1) % 50 == 0:
                            progress = 0.15 + 0.6 * (i + 1) / len(files)
                            on_progress(progress, f"Parsed {i+1}/{len(files)}")
            except (BrokenProcessPool, BrokenPipeError, OSError, RuntimeError) as e:
                # ProcessPool failed (Docker /dev/shm, fork issues, etc.)
                logger.warning(f"ProcessPoolExecutor failed ({e}), falling back to ThreadPoolExecutor")
                results.clear()
                pool_type = "thread"

        if pool_type == "thread":
            if _in_docker:
                logger.info("Using ThreadPoolExecutor (Docker mode)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                # Process in batches to avoid GIL starvation and keep event loop responsive
                # BATCH_SIZE = self.max_workers * 2 keeps the pipeline full without flooding
                batch_size = max(10, self.max_workers * 2)
                total_files = len(files)

                for i in range(0, total_files, batch_size):
                    batch = files[i : i + batch_size]
                    tasks = [
                        loop.run_in_executor(pool, parse_file_sync, str(f))
                        for f in batch
                    ]
                    # Process batch results as they complete
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            results.append(result)
                        except Exception as e:
                            logger.warning(f"Failed to parse file in batch: {e}")

                    # Update progress
                    processed_count = min(i + batch_size, total_files)
                    if on_progress and processed_count % 50 == 0:
                        progress = 0.15 + 0.6 * processed_count / total_files
                        on_progress(progress, f"Parsed {processed_count}/{total_files} (thread)")

                    # Yield to event loop to prevent timeouts
                    await asyncio.sleep(0.05)
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    if on_progress and (i + 1) % 50 == 0:
                        progress = 0.15 + 0.6 * (i + 1) / len(files)
                        on_progress(progress, f"Parsed {i+1}/{len(files)}")

        self.parsing_ms = (time.perf_counter() - parse_start) * 1000

        # Phase 3: Build index
        if on_progress:
            on_progress(0.75, "Building index...")

        for graph in results:
            for chunk in graph.chunks:
                self.chunks[chunk.id] = chunk
                self.by_name[chunk.name].append(chunk.id)
                self.by_file[graph.source_path].append(chunk.id)
                if chunk.parent_class:
                    self.by_class[chunk.parent_class].append(chunk.id)
                # Sync to knowledge graph (fire-and-forget)
                self._queue_sync({
                    "id": chunk.id,
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "properties": {
                        "source_path": chunk.source_path,
                        "signature": chunk.signature,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "parent_class": chunk.parent_class or "",
                        "complexity": chunk.complexity,
                    },
                })
        # Flush remaining sync queue
        self._flush_to_bus()

        # Phase 4: Backfill called_by relationships
        if on_progress:
            on_progress(0.85, "Building call graph...")

        backfill_start = time.perf_counter()
        self._backfill_called_by()
        self._invalidate_keyword_cache()
        self.backfill_ms = (time.perf_counter() - backfill_start) * 1000

        if on_progress:
            on_progress(1.0, "Complete!")

        total_ms = (time.perf_counter() - total_start) * 1000

        stats = {
            "total_files": self.total_files,
            "total_chunks": len(self.chunks),
            "functions": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.FUNCTION),
            "methods": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.METHOD),
            "classes": sum(1 for c in self.chunks.values() if c.chunk_type == ChunkType.CLASS),
            "discovery_ms": self.discovery_ms,
            "parsing_ms": self.parsing_ms,
            "backfill_ms": self.backfill_ms,
            "total_ms": total_ms,
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    async def enrich_with_langextract(
        self,
        root_path: str,
        max_files: int = 50,
        on_progress: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Enrich indexed chunks with LangExtract concept extraction.

        Runs structured extraction on docstrings/comments to identify
        concepts, algorithms, relationships, and entities. Results are
        stored as metadata on each CodeChunk for downstream graph ingest.

        Requires: pip install langextract

        Args:
            root_path: Codebase root (used for relative paths).
            max_files: Max files to run extraction on (most documented first).
            on_progress: Progress callback.

        Returns:
            Stats dict with enrichment counts.
        """
        try:
            from adk.faculties.langextract import (
                LangExtractFaculty, LANGEXTRACT_AVAILABLE,
            )
        except ImportError:
            logger.debug("LangExtractFaculty not available — skipping enrichment")
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "not_installed"}

        if not LANGEXTRACT_AVAILABLE:
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "not_installed"}

        faculty = LangExtractFaculty()
        if not await faculty.initialize():
            return {"enriched_chunks": 0, "extractions": 0, "skipped": "init_failed"}

        if on_progress:
            on_progress(0.0, "Starting LangExtract enrichment...")

        # Prioritize chunks with docstrings (most documentation = most value)
        documented_chunks = [
            c for c in self.chunks.values()
            if c.docstring and len(c.docstring.strip()) > 20
        ]
        # Sort by docstring length descending (richest docs first)
        documented_chunks.sort(key=lambda c: len(c.docstring), reverse=True)
        documented_chunks = documented_chunks[:max_files]

        enriched = 0
        total_extractions = 0

        for i, chunk in enumerate(documented_chunks):
            try:
                text = f"{chunk.name}\n{chunk.signature}\n{chunk.docstring}"
                results = await faculty.extract_from_text(
                    text=text,
                    task="code_concepts",
                    source_file=chunk.source_path,
                )
                if results:
                    # Store extractions as metadata on the chunk
                    chunk.imports = list(set(
                        chunk.imports + [
                            r.text for r in results
                            if r.extraction_class in ("concept", "algorithm")
                        ]
                    ))
                    # Store full extraction data for graph ingest
                    if not hasattr(chunk, 'extractions'):
                        object.__setattr__(chunk, 'extractions', [])
                    for r in results:
                        chunk.extractions.append(r.to_dict())  # type: ignore[attr-defined]
                    enriched += 1
                    total_extractions += len(results)
            except Exception as e:
                logger.debug(f"LangExtract enrichment failed for {chunk.name}: {e}")

            if on_progress and (i + 1) % 10 == 0:
                on_progress(
                    (i + 1) / len(documented_chunks),
                    f"Enriched {enriched}/{i + 1} chunks",
                )

        stats = {
            "enriched_chunks": enriched,
            "extractions": total_extractions,
            "candidates": len(documented_chunks),
        }
        logger.info(f"LangExtract enrichment complete: {stats}")
        return stats

    def _path_to_module_prefix(self, path: str) -> str:
        """Convert a file path to a possible module prefix (e.g. lib.faculties.CodeGraph)."""
        p = path.replace("\\", "/")
        if p.endswith(".py"):
            p = p[:-3]
        if p.endswith("/__init__"):
            p = p[:-9]
        return p.replace("/", ".")

    def _backfill_called_by(self):
        """
        Build the called_by relationships by inverting the calls graph.

        This is the key step that enables "what calls this function?"
        """
        # Build a map of name -> chunk_ids
        name_map: Dict[str, List[str]] = defaultdict(list)
        for chunk_id, chunk in self.chunks.items():
            # Map both full name and short name
            name_map[chunk.name].append(chunk_id)
            if "." in chunk.name:
                short_name = chunk.name.split(".")[-1]
                name_map[short_name].append(chunk_id)

        # For each chunk, add it to the called_by list of what it calls
        for caller_id, caller in self.chunks.items():
            for called_name in caller.calls:
                # Determine expected module if it comes from an import
                expected_module = None
                if getattr(caller, 'import_map', None):
                    if "." in called_name:
                        base_name = called_name.split(".")[0]
                        if base_name in caller.import_map:
                            expected_module = caller.import_map[base_name]
                    elif called_name in caller.import_map:
                        expected_module = caller.import_map[called_name]

                # Find chunks matching this name
                potential_callees = name_map.get(called_name, [])
                for callee_id in potential_callees:
                    callee = self.chunks[callee_id]

                    # Same file calls are always valid
                    if caller.source_path == callee.source_path:
                        if caller.name not in callee.called_by:
                            callee.called_by.append(caller.name)
                        continue

                    # If we expect a specific module origin, enforce it
                    if expected_module:
                        callee_mod = self._path_to_module_prefix(callee.source_path)
                        expected_base = expected_module
                        if "." in expected_module:
                            # Strip off the imported functionality name if it equals the chunk name
                            mod_parts = expected_module.split(".")
                            short_callee_name = callee.name.split(".")[-1] if "." in callee.name else callee.name
                            if mod_parts[-1] == short_callee_name or callee.name.startswith(f"{mod_parts[-1]}."):
                                expected_base = ".".join(mod_parts[:-1])

                        if not callee_mod.endswith(expected_base):
                            continue # Skip: false positive match from a different file

                    if caller.name not in callee.called_by:
                        callee.called_by.append(caller.name)

    def find_orphans(
        self,
        *,
        exclude_tests: bool = True,
        min_lines: int = 5,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find code chunks with zero callers (potential dead code candidates).

        These are CANDIDATES only — dynamic calls (getattr, decorators, framework
        entry points) cannot be tracked by static AST analysis.

        Args:
            exclude_tests: Skip chunks in test files.
            min_lines: Minimum line count to report (filters trivial helpers).
            exclude_patterns: Glob-style patterns for function names to skip.

        Returns:
            List of orphan dicts sorted by line_count descending.
        """
        import fnmatch

        # Names that are always entry points or framework hooks
        _ENTRY_NAMES = frozenset({
            "__init__", "__main__", "main", "app", "router",
            "lifespan", "startup", "shutdown", "on_startup", "on_shutdown",
            "setup", "teardown", "conftest", "pytest_configure",
        })

        exclude_patterns = exclude_patterns or []
        orphans: List[Dict[str, Any]] = []

        for chunk_id, chunk in self.chunks.items():
            # Skip modules — top-level modules always have 0 callers
            if chunk.chunk_type == ChunkType.MODULE:
                continue

            # Skip if has callers
            if chunk.called_by:
                continue

            # Skip entry points and dunder methods
            short_name = chunk.name.split(".")[-1] if "." in chunk.name else chunk.name
            if short_name in _ENTRY_NAMES or short_name.startswith("__"):
                continue

            # Skip test files
            if exclude_tests:
                path_lower = chunk.source_path.lower().replace("\\", "/")
                if "/tests/" in path_lower or "/test_" in path_lower or path_lower.endswith("_test.py"):
                    continue

            # Skip small chunks
            if chunk.line_count < min_lines:
                continue

            # Skip user-specified patterns
            if any(fnmatch.fnmatch(short_name, pat) for pat in exclude_patterns):
                continue

            orphans.append({
                "name": chunk.name,
                "file": chunk.source_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type.value,
                "line_count": chunk.line_count,
                "complexity": chunk.complexity,
                "calls_out": len(chunk.calls),
            })

        orphans.sort(key=lambda o: o["line_count"], reverse=True)
        return orphans

    def _invalidate_keyword_cache(self):
        """Clear keyword result cache (called after index changes)."""
        self._keyword_result_cache.clear()
        self._keyword_cache_order.clear()

    _QUERY_STOP_WORDS = frozenset({
        "how", "does", "the", "what", "is", "a", "an", "and", "or", "to",
        "in", "for", "of", "with", "from", "are", "do", "its", "it", "by",
        "on", "that", "this", "be", "can", "has", "have", "i", "my", "all",
    })

    _TEST_PENALTY = 0.25  # Score multiplier for chunks from test files

    @staticmethod
    def _is_test_path(source_path: str) -> bool:
        """Check if a source path belongs to a test file."""
        p = source_path.replace("\\", "/").lower()
        basename = p.rsplit("/", 1)[-1]
        return (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or "/tests/" in p
            or "/test/" in p
            or basename == "conftest.py"
        )

    async def query(
        self,
        query: str,
        max_results: int = 10,
        include_callers: bool = True,
        include_callees: bool = True,
    ) -> List[CodeChunk]:
        """
        Query the code graph.

        Supports both single-keyword queries (e.g. "inject") and natural
        language queries (e.g. "How does ContextEngine inject items?").
        Tokenizes multi-word queries into keywords and scores chunks by
        how many keywords match across name, signature, docstring, and calls.
        """
        # Check keyword result cache first (saves ~60ms per repeated query)
        cache_key = f"{query}:{max_results}"
        if cache_key in self._keyword_result_cache:
            # Move to end of LRU order
            try:
                self._keyword_cache_order.remove(cache_key)
            except ValueError as e:
                logger.debug(f"[CodeGraph.query] Operation failed: {e}")
            self._keyword_cache_order.append(cache_key)
            return self._keyword_result_cache[cache_key]

        query_lower = query.lower()
        # Tokenize into meaningful keywords
        tokens = [
            w for w in query_lower.replace("?", "").replace(".", "").replace(",", "").split()
            if w not in self._QUERY_STOP_WORDS and len(w) > 1
        ]
        if not tokens:
            tokens = [query_lower]

        results: List[Tuple[float, CodeChunk]] = []

        for chunk in self.chunks.values():
            score = 0.0
            name_lower = chunk.name.lower()
            sig_lower = chunk.signature.lower()
            doc_lower = chunk.docstring.lower() if chunk.docstring else ""
            calls_lower = " ".join(chunk.calls).lower()
            body_lower = chunk.body_preview.lower() if chunk.body_preview else ""

            for token in tokens:
                # Name match (strongest signal)
                if token in name_lower:
                    score += 10.0
                    if name_lower == token:
                        score += 5.0  # Exact match bonus
                # Signature match
                if token in sig_lower:
                    score += 3.0
                # Docstring match
                if token in doc_lower:
                    score += 2.0
                # Call graph match (this chunk calls or references the token)
                if token in calls_lower:
                    score += 1.5
                # Body preview match (weakest but still relevant)
                if token in body_lower:
                    score += 0.5

            # Bonus: fraction of keywords matched (rewards broader coverage)
            if score > 0 and len(tokens) > 1:
                matched_count = sum(
                    1 for t in tokens
                    if t in name_lower or t in sig_lower or t in doc_lower
                )
                coverage = matched_count / len(tokens)
                score *= (1.0 + coverage)  # up to 2x boost

            # Deprioritize test files — they match keywords but aren't useful context
            if score > 0 and self._is_test_path(chunk.source_path):
                score *= self._TEST_PENALTY

            if score > 0:
                results.append((score, chunk))

        # Sort by score descending
        results.sort(key=lambda x: -x[0])

        final = [chunk for _, chunk in results[:max_results]]

        # Store in keyword result cache (LRU, max 64)
        if len(self._keyword_result_cache) >= self._KEYWORD_CACHE_MAX:
            evict_key = self._keyword_cache_order.pop(0)
            self._keyword_result_cache.pop(evict_key, None)
        self._keyword_result_cache[cache_key] = final
        self._keyword_cache_order.append(cache_key)

        return final

    def get_context_for_chunk(self, chunk_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get the full context for a chunk, including its callers and callees.

        This is what you inject into the LLM prompt for surgical context.
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return {}

        context = {
            "chunk": chunk,
            "callers": [],
            "callees": [],
        }

        # Get callers
        for caller_name in chunk.called_by[:10]:  # Limit to 10
            caller_ids = self.by_name.get(caller_name, [])
            for cid in caller_ids[:2]:  # Max 2 per name
                if cid in self.chunks:
                    context["callers"].append(self.chunks[cid])

        # Get callees
        for callee_name in chunk.calls[:10]:
            callee_ids = self.by_name.get(callee_name, [])
            for cid in callee_ids[:2]:
                if cid in self.chunks:
                    context["callees"].append(self.chunks[cid])

        return context

    def export_for_embedding(self) -> List[Dict[str, Any]]:
        """
        Export chunks in a format ready for embedding with MindClient.

        Each chunk becomes a document with metadata for filtering.
        """
        documents = []

        for chunk in self.chunks.values():
            # Build the text to embed
            text_parts = [chunk.signature]
            if chunk.docstring:
                text_parts.append(chunk.docstring)
            text_parts.append(chunk.body_preview)

            doc = {
                "id": chunk.id,
                "text": "\n".join(text_parts),
                "metadata": {
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "file": chunk.source_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "complexity": chunk.complexity,
                    "calls_count": len(chunk.calls),
                    "called_by_count": len(chunk.called_by),
                },
            }
            documents.append(doc)

        return documents

    # ====================================================================
    # EMBEDDING METHODS
    # ====================================================================

    async def embed_chunks(
        self,
        model: str = "nomic-embed-text",
        batch_size: int = 64,
        cache_path: Optional[str] = None,
        on_progress: Optional[callable] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all indexed chunks via EmbeddingProvider.

        Embeddings are stored on CodeChunk.embedding and persisted to disk.
        Incremental: only embeds chunks that don't have embeddings yet.

        Args:
            model: Embedding model (default: nomic-embed-text, 768-dim)
            batch_size: Texts per API call
            cache_path: Where to persist embeddings (default: .aither/data/codegraph_embeddings.pkl)
            on_progress: callback(fraction, message)
            force: Re-embed all chunks even if cached

        Returns:
            Stats dict with total, cached, new, failed, embed_ms
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            raise RuntimeError("No embedding backend available (need EmbeddingProvider or vLLM)")

        if cache_path is None:
            cache_path = str(
                Path(os.path.expanduser("~/.aither/data"))
                / "codegraph_embeddings.pkl"
            )

        # Load cached embeddings from disk (offloaded to thread for 9P safety)
        cached: Dict[str, list] = {}
        if not force and os.path.exists(cache_path):
            if not _verify_pickle_hmac(cache_path):
                logger.warning("Embedding cache HMAC invalid — starting fresh")
            else:
                def _load_pickle():
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
                try:
                    cached = await asyncio.to_thread(_load_pickle)
                    logger.info(f"Loaded {len(cached)} cached embeddings")
                except Exception as e:
                    logger.warning(f"Failed to load embedding cache: {e}")

        # Apply cached embeddings to current chunks
        applied = 0
        for chunk_id, chunk in self.chunks.items():
            if chunk_id in cached and chunk.embedding is None:
                chunk.embedding = cached[chunk_id]
                applied += 1

        # Find chunks needing embeddings
        need_embedding = [
            (cid, chunk) for cid, chunk in self.chunks.items()
            if chunk.embedding is None
        ]

        if not need_embedding:
            logger.info(f"All {len(self.chunks)} chunks already have embeddings")
            return {"total": len(self.chunks), "cached": applied, "new": 0, "embed_ms": 0}

        logger.info(
            f"Embedding {len(need_embedding)} chunks "
            f"({applied} from cache, {len(self.chunks) - applied - len(need_embedding)} in memory)"
        )

        # Build texts to embed — rich representation for semantic matching
        texts = []
        chunk_ids = []
        for cid, chunk in need_embedding:
            parts = [chunk.signature]
            if chunk.docstring:
                parts.append(chunk.docstring[:300])
            if chunk.body_preview:
                parts.append(chunk.body_preview[:300])
            if chunk.calls:
                parts.append(f"calls: {', '.join(chunk.calls[:10])}")
            if chunk.called_by:
                parts.append(f"called by: {', '.join(chunk.called_by[:10])}")
            if chunk.parent_class:
                parts.append(f"class: {chunk.parent_class}")
            texts.append("\n".join(parts))
            chunk_ids.append(cid)

        start = time.perf_counter()
        backend = "vLLM" if _detect_vllm() else "EmbeddingProvider"
        logger.info(f"Embedding {len(need_embedding)} chunks via {backend}")
        total_batches = (len(texts) + batch_size - 1) // batch_size

        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i : i + batch_size])

        results: List[Optional[List[list]]] = [None] * len(batches)
        consecutive_failures = 0

        for idx, batch in enumerate(batches):
            if consecutive_failures >= 3:
                logger.warning(
                    f"Embedding backend down ({consecutive_failures} consecutive failures) "
                    f"— aborting remaining {total_batches - idx} batches"
                )
                break
            try:
                embeddings = await asyncio.wait_for(
                    _embed_texts(batch, model=model), timeout=30.0
                )
                results[idx] = embeddings
                consecutive_failures = 0
            except asyncio.TimeoutError:
                logger.warning(f"Embedding batch {idx+1}/{total_batches} timed out (30s)")
                results[idx] = [None] * len(batch)
                consecutive_failures += 1
            except Exception as e:
                logger.error(f"Embedding batch {idx+1}/{total_batches} failed: {e}")
                results[idx] = [None] * len(batch)
                consecutive_failures += 1
            if on_progress:
                on_progress(
                    (idx + 1) / total_batches,
                    f"Embedded {min((idx + 1) * batch_size, len(texts))}/{len(texts)}",
                )
            await asyncio.sleep(0.05)

        # Flatten results in order
        all_embeddings: List[Optional[list]] = []
        for batch_result in results:
            if batch_result is not None:
                all_embeddings.extend(batch_result)
            else:
                all_embeddings.extend([None])

        embed_ms = (time.perf_counter() - start) * 1000

        # Apply embeddings to chunks + update cache
        new_count = 0
        for cid, emb in zip(chunk_ids, all_embeddings):
            if emb is not None:
                self.chunks[cid].embedding = emb
                cached[cid] = emb
                new_count += 1
        if new_count > 0:
            self._has_embeddings_cached = True  # Invalidate hybrid_query fast-path cache
            self._invalidate_embedding_matrix()  # Force matrix rebuild on next query

        # Persist to disk (offloaded to thread for 9P safety)
        try:
            _cached_copy = dict(cached)  # Snapshot for thread safety
            _cache_path_copy = cache_path  # Capture for closure
            def _save_pickle():
                os.makedirs(os.path.dirname(_cache_path_copy), exist_ok=True)
                with open(_cache_path_copy, "wb") as f:
                    pickle.dump(_cached_copy, f, protocol=pickle.HIGHEST_PROTOCOL)
                _write_pickle_hmac(_cache_path_copy)
                return os.path.getsize(_cache_path_copy) / (1024 * 1024)
            sz = await asyncio.to_thread(_save_pickle)
            logger.info(f"Saved {len(cached)} embeddings ({sz:.1f}MB)")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

        stats = {
            "total": len(self.chunks),
            "cached": applied,
            "new": new_count,
            "failed": len(need_embedding) - new_count,
            "embed_ms": embed_ms,
            "batches": total_batches,
            "model": model,
        }
        logger.info(f"Embedding complete: {stats}")
        return stats

    @property
    def embedding_coverage(self) -> float:
        """Fraction of chunks with embeddings (0.0-1.0)."""
        if not self.chunks:
            return 0.0
        return sum(1 for c in self.chunks.values() if c.embedding is not None) / len(self.chunks)

    async def semantic_query(
        self,
        query: str,
        max_results: int = 10,
        model: str = "nomic-embed-text",
    ) -> List[Tuple[float, "CodeChunk"]]:
        """
        Semantic search using embedding cosine similarity.

        Returns list of (similarity_score, chunk) sorted descending.
        Requires embed_chunks() to have been called first.
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            return []

        embedded = [(cid, c) for cid, c in self.chunks.items() if c.embedding is not None]
        if not embedded:
            logger.warning("No embeddings available — call embed_chunks() first")
            return []

        query_vec = self._query_embed_cache.get(query)
        if query_vec is None:
            try:
                vecs = await _embed_texts([query], model=model)
                query_vec = vecs[0] if vecs else None
                if not query_vec:
                    return []
                # Cache for next time
                self._cache_query_embedding(query, query_vec)
            except Exception as e:
                logger.error(f"Failed to embed query: {e}")
                return []
        else:
            logger.debug(f"[SEMANTIC] Query embedding cache HIT for: {query[:40]}")

        if _HAS_NUMPY:
            # Use pre-computed matrix (avoids ~500ms np.array construction per query)
            self._ensure_embedding_matrix()
            if self._embedding_matrix is None:
                return []

            qv = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(qv)
            if q_norm == 0:
                return []

            valid = (self._embedding_norms > 0)
            sims = np.zeros(len(self._embedding_ids), dtype=np.float32)
            sims[valid] = (self._embedding_matrix[valid] @ qv) / (self._embedding_norms[valid] * q_norm)

            # Apply test file penalty before ranking
            for i, cid in enumerate(self._embedding_ids):
                if self._is_test_path(self.chunks[cid].source_path):
                    sims[i] *= self._TEST_PENALTY

            top_idx = np.argsort(sims)[::-1][:max_results]
            return [
                (float(sims[i]), self.chunks[self._embedding_ids[i]])
                for i in top_idx
                if sims[i] > 0
            ]
        else:
            # Pure-Python fallback
            q_norm = math.sqrt(sum(x * x for x in query_vec))
            if q_norm == 0:
                return []
            results = []
            for cid, chunk in embedded:
                emb = chunk.embedding
                dot = sum(a * b for a, b in zip(emb, query_vec))
                e_norm = math.sqrt(sum(x * x for x in emb))
                if e_norm == 0:
                    continue
                sim = dot / (e_norm * q_norm)
                if sim > 0:
                    if self._is_test_path(chunk.source_path):
                        sim *= self._TEST_PENALTY
                    results.append((sim, chunk))
            results.sort(key=lambda x: -x[0])
            return results[:max_results]

    # -- Embedding Matrix Cache ----------------------------------------

    def _ensure_embedding_matrix(self) -> None:
        """Build pre-computed numpy matrix from chunk embeddings (once).

        The matrix construction (np.array from 28K Python lists) takes ~500ms.
        Cache it so every semantic_query pays only ~50ms for cosine similarity.
        Invalidated when embeddings change (embed_chunks, reindex_files).
        """
        if self._embedding_matrix is not None:
            return
        if not _HAS_NUMPY:
            return
        embedded = [(cid, c) for cid, c in self.chunks.items() if c.embedding is not None]
        if not embedded:
            return
        t0 = time.time()
        self._embedding_ids = [cid for cid, _ in embedded]
        self._embedding_matrix = np.array(
            [c.embedding for _, c in embedded], dtype=np.float32
        )
        self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)
        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"[MATRIX] Pre-computed embedding matrix: {self._embedding_matrix.shape} in {elapsed:.0f}ms"
        )

    def _invalidate_embedding_matrix(self) -> None:
        """Clear cached matrix — call after embeddings change."""
        self._embedding_matrix = None
        self._embedding_ids = None
        self._embedding_norms = None

    # -- Query Embedding Cache -----------------------------------------

    def _cache_query_embedding(self, query: str, vec: list) -> None:
        """Store query embedding in LRU cache, evicting oldest if full."""
        if query in self._query_embed_cache:
            # Move to end (most recent)
            self._query_cache_order.remove(query)
            self._query_cache_order.append(query)
            return
        if len(self._query_embed_cache) >= self._QUERY_CACHE_MAX:
            oldest = self._query_cache_order.pop(0)
            self._query_embed_cache.pop(oldest, None)
        self._query_embed_cache[query] = vec
        self._query_cache_order.append(query)

    def _background_cache_query(self, query: str) -> None:
        """Fire-and-forget: embed query in background and cache result."""
        async def _do():
            try:
                vecs = await _embed_texts([query])
                if vecs and vecs[0]:
                    self._cache_query_embedding(query, vecs[0])
                    logger.debug(f"[CACHE] Background-cached embedding for: {query[:40]}")
            except Exception as e:
                logger.debug(f"[CodeGraph._do] Operation failed: {e}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do())
        except RuntimeError as e:
            logger.debug(f"[CodeGraph._do] Graph operation failed: {e}")

    async def pre_embed_query(self, query: str) -> None:
        """Pre-compute and cache query embedding for future hybrid_query() calls.

        Called during typing speculation. When the actual hybrid_query() fires,
        it finds the embedding already cached and skips the embed step.
        """
        if query in self._query_embed_cache:
            return  # Already cached
        try:
            vecs = await _embed_texts([query])
            if vecs and vecs[0]:
                self._cache_query_embedding(query, vecs[0])
                logger.debug(f"[PREFETCH] Pre-embedded query: {query[:40]}")
        except Exception:
            pass  # Non-critical, graceful skip

    async def warm_query_cache(self, queries: List[str]) -> int:
        """
        Pre-embed multiple queries in parallel.  Batches into a single API
        call so N queries cost ~1x round-trip, not Nx.

        Returns number of newly cached embeddings.
        """
        uncached = [q for q in queries if q not in self._query_embed_cache]
        if not uncached:
            return 0
        try:
            vecs = await _embed_texts(uncached)
            cached = 0
            for q, v in zip(uncached, vecs):
                if v:
                    self._cache_query_embedding(q, v)
                    cached += 1
            logger.info(f"[CACHE] Warmed {cached}/{len(uncached)} query embeddings in batch")
            return cached
        except Exception as e:
            logger.warning(f"[CACHE] Batch warmup failed: {e}")
            return 0

    async def _expand_context(
        self,
        chunk_ids: Set[str],
        query: str = "",
        max_expand: int = 15,
    ) -> List["CodeChunk"]:
        """
        Expand a set of hit chunk IDs with structurally and semantically related chunks.

        Phase 1 -- Structural: parent class, sibling methods, file neighbors, call graph.
        Phase 2 -- Semantic diversity: embedding-similar chunks from DIFFERENT files
                  than the initial hits.  This is what lifts hard/architectural queries.
        """
        expanded: List["CodeChunk"] = []
        seen = set(chunk_ids)
        hit_files = {self.chunks[cid].source_path for cid in chunk_ids if cid in self.chunks}

        # --- Phase 1: structural expansion (same as before) ---
        structural_budget = max_expand // 2 or max_expand

        for cid in list(chunk_ids):
            chunk = self.chunks.get(cid)
            if not chunk or len(expanded) >= structural_budget:
                break

            # 1a. Parent class: if this is a method, pull the class chunk
            if chunk.parent_class:
                for other_id in self.by_name.get(chunk.parent_class, []):
                    if other_id not in seen:
                        seen.add(other_id)
                        expanded.append(self.chunks[other_id])

                # Sibling methods from same class (O(1) via by_class index)
                for other_id in self.by_class.get(chunk.parent_class, []):
                    if other_id not in seen:
                        other = self.chunks.get(other_id)
                        if other and other.chunk_type == ChunkType.METHOD:
                            seen.add(other_id)
                            expanded.append(other)
                            if len(expanded) >= structural_budget:
                                break

            # 1b. Same-file neighbors (functions/classes in the same module)
            file_siblings = self.by_file.get(chunk.source_path, [])
            for sib_id in file_siblings[:8]:
                if sib_id not in seen:
                    seen.add(sib_id)
                    expanded.append(self.chunks[sib_id])
                    if len(expanded) >= structural_budget:
                        break

            # 1c. Call-graph expansion (1 level)
            ctx = self.get_context_for_chunk(cid)
            for related in ctx.get("callers", []) + ctx.get("callees", []):
                if related.id not in seen:
                    seen.add(related.id)
                    expanded.append(related)
                    if len(expanded) >= structural_budget:
                        break

        # --- Phase 2: semantic diversity (cross-file) ---
        diversity_budget = max_expand - len(expanded)
        if diversity_budget > 0 and query and _HAS_EMBEDDING_ENGINE:
            sem_results = await self.semantic_query(query, max_results=diversity_budget * 3)
            for _sim, chunk in sem_results:
                if chunk.id not in seen and chunk.source_path not in hit_files:
                    seen.add(chunk.id)
                    expanded.append(chunk)
                    if len(expanded) >= max_expand:
                        break

        return expanded

    # ========================================================================
    # MULTI-HOP CHAIN EXPANSION -- Architectural query support
    # ========================================================================

    def _multi_hop_expand(
        self,
        seed_chunk_ids: List[str],
        query: str,
        max_depth: int = 3,
        max_chains: int = 5,
    ) -> List[List["CodeChunk"]]:
        """
        BFS on call graph to build multi-hop responsibility chains.

        For architectural queries, follows calls/called_by relationships
        up to max_depth levels, pruning irrelevant branches.

        Returns list of chains (each chain is a list of CodeChunks).
        """
        query_tokens = set(query.lower().split())
        chains: List[List["CodeChunk"]] = []
        visited_chains: set = set()  # Avoid duplicate chain fingerprints

        for seed_id in seed_chunk_ids:
            seed = self.chunks.get(seed_id)
            if not seed:
                continue

            # BFS with path tracking
            queue: List[Tuple[str, List[str], int]] = [(seed_id, [seed_id], 0)]
            seen_in_search: set = {seed_id}

            while queue and len(chains) < max_chains * 3:
                current_id, path, depth = queue.pop(0)
                current = self.chunks.get(current_id)
                if not current or depth >= max_depth:
                    continue

                # Get neighbors: forward calls + reverse callers
                neighbors: List[str] = []
                for call_name in current.calls[:8]:
                    for cid in self.by_name.get(call_name, [])[:2]:
                        if cid not in seen_in_search:
                            neighbors.append(cid)
                for caller_name in current.called_by[:8]:
                    for cid in self.by_name.get(caller_name, [])[:2]:
                        if cid not in seen_in_search:
                            neighbors.append(cid)

                for neighbor_id in neighbors:
                    neighbor = self.chunks.get(neighbor_id)
                    if not neighbor:
                        continue

                    # Relevance check: does this hop relate to the query?
                    hop_text = (neighbor.name + " " + (neighbor.docstring or "")).lower()
                    hop_tokens = set(hop_text.split())
                    overlap = len(query_tokens & hop_tokens)

                    # Prune: zero relevance AND same file = dead branch
                    if overlap == 0 and neighbor.source_path == current.source_path:
                        continue

                    new_path = path + [neighbor_id]
                    seen_in_search.add(neighbor_id)

                    # Record chain if it spans 2+ nodes
                    if len(new_path) >= 2:
                        chain_key = tuple(sorted(new_path))
                        if chain_key not in visited_chains:
                            visited_chains.add(chain_key)
                            chain = [self.chunks[cid] for cid in new_path if cid in self.chunks]
                            if len(chain) >= 2:
                                chains.append(chain)

                    # Continue BFS
                    if depth + 1 < max_depth:
                        queue.append((neighbor_id, new_path, depth + 1))

        return chains

    def _score_chains(
        self,
        chains: List[List["CodeChunk"]],
        query: str,
    ) -> List[Tuple[float, List["CodeChunk"]]]:
        """
        Score and rank chains by relevance to query.

        Score = sum(hop_relevance * 0.7^depth) * cross_file_bonus * sqrt(len)
        """
        query_tokens = set(query.lower().split())
        scored: List[Tuple[float, List["CodeChunk"]]] = []

        for chain in chains:
            hop_score = 0.0
            unique_files: set = set()

            for depth, chunk in enumerate(chain):
                # Relevance of this hop
                hop_text = (chunk.name + " " + (chunk.docstring or "") + " " + chunk.signature).lower()
                hop_tokens = set(hop_text.split())
                overlap = len(query_tokens & hop_tokens)
                relevance = min(1.0, overlap / max(1, len(query_tokens)))

                # Decay by depth
                hop_score += relevance * (0.7 ** depth)
                unique_files.add(chunk.source_path)

            # Cross-file bonus: chains spanning multiple files are more valuable
            cross_file_bonus = 1.0 + 0.1 * len(unique_files)

            # Length bonus: longer chains (that stayed relevant) are richer
            length_bonus = math.sqrt(len(chain))

            total_score = hop_score * cross_file_bonus * length_bonus
            scored.append((total_score, chain))

        scored.sort(key=lambda x: -x[0])
        return scored

    async def _rerank(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
        mode: str = "embedding",
    ) -> List["CodeChunk"]:
        """
        Re-rank candidates by relevance to query.

        Modes:
            'embedding' -- Fast: cosine similarity of query embedding vs chunk embeddings.
                          Zero LLM calls. ~5ms for 20 candidates. Default.
            'llm'       -- Accurate: parallel chunked LLM scoring via nemotron-mini.
                          Splits candidates into groups of 5, scores in parallel.
            'hybrid'    -- Embedding pre-filter -> LLM re-score top candidates.
        """
        if len(chunks) <= top_k:
            return chunks[:top_k]

        if mode == "embedding":
            return await self._rerank_by_embedding(query, chunks, top_k)
        elif mode == "llm":
            return await self._rerank_by_llm(query, chunks, top_k)
        elif mode == "hybrid":
            # Embedding narrows to 2x top_k, then LLM picks final top_k
            narrowed = await self._rerank_by_embedding(query, chunks, top_k * 2)
            return await self._rerank_by_llm(query, narrowed, top_k)
        else:
            return chunks[:top_k]

    async def _rerank_by_embedding(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
    ) -> List["CodeChunk"]:
        """
        Re-rank using embedding cosine similarity. Zero LLM calls.

        Embeds the query once, computes cosine sim against each candidate's
        existing embedding. Falls back to original order for un-embedded chunks.
        """
        if not _HAS_EMBEDDING_ENGINE and not _detect_vllm():
            return chunks[:top_k]

        # Embed the query via EmbeddingProvider
        try:
            vecs = await _embed_texts([query], model="nomic-embed-text")
            query_vec = vecs[0] if vecs else None
            if not query_vec:
                return chunks[:top_k]
        except Exception as e:
            logger.debug(f"Embedding re-rank failed (query embed): {e}")
            return chunks[:top_k]

        if _HAS_NUMPY:
            qv = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(qv)
            if q_norm == 0:
                return chunks[:top_k]

            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    cv = np.array(chunk.embedding, dtype=np.float32)
                    c_norm = np.linalg.norm(cv)
                    sim = float(np.dot(qv, cv) / (q_norm * c_norm)) if c_norm > 0 else 0.0
                else:
                    sim = 0.0
                scored.append((sim, chunk))
        else:
            q_norm = math.sqrt(sum(x * x for x in query_vec))
            if q_norm == 0:
                return chunks[:top_k]
            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    dot = sum(a * b for a, b in zip(chunk.embedding, query_vec))
                    c_norm = math.sqrt(sum(x * x for x in chunk.embedding))
                    sim = dot / (q_norm * c_norm) if c_norm > 0 else 0.0
                else:
                    sim = 0.0
                scored.append((sim, chunk))

        scored.sort(key=lambda x: -x[0])
        return [chunk for _, chunk in scored[:top_k]]

    async def _rerank_by_llm(
        self,
        query: str,
        chunks: List["CodeChunk"],
        top_k: int = 10,
        group_size: int = 5,
    ) -> List["CodeChunk"]:
        """
        Parallel chunked LLM re-ranking via nemotron-mini.

        Splits candidates into groups of `group_size`, scores each group in
        parallel via asyncio.gather. ~4x faster than sequential for 20 candidates.
        """
        if (not _HAS_EMBEDDING_ENGINE and not _detect_vllm()) or len(chunks) <= top_k:
            return chunks[:top_k]

        candidates = chunks[:20]  # Cap at 20

        # Split into groups
        groups = [
            candidates[i : i + group_size]
            for i in range(0, len(candidates), group_size)
        ]

        async def _score_group(group: List["CodeChunk"], offset: int) -> List[Tuple[int, int]]:
            """Score a group of candidates. Returns list of (score, global_index)."""
            items = []
            for j, c in enumerate(group):
                sig = c.signature[:120]
                doc = (c.docstring[:80] if c.docstring else "")
                calls = ", ".join(c.calls[:5]) if c.calls else "none"
                items.append(f"{j}. {c.name} | {sig} | calls: {calls} | {doc}")

            prompt = (
                f"Rate each code function's relevance to the query on a scale 0-9.\n"
                f"Query: {query}\n\n"
                f"Candidates:\n" + "\n".join(items) + "\n\n"
                f"Reply ONLY with one number (0-9) per line, one line per candidate. "
                f"No explanations."
            )
            try:
                text = await _llm_generate(prompt, model=ELASTIC_REFLEX, max_tokens=50)
                lines = text.strip().split("\n")
                results = []
                for j, line in enumerate(lines):
                    digits = [ch for ch in line.strip() if ch.isdigit()]
                    score = int(digits[0]) if digits else 5
                    results.append((score, offset + j))
                # Pad if model returned fewer lines
                while len(results) < len(group):
                    results.append((5, offset + len(results)))
                return results
            except Exception as e:
                logger.debug(f"LLM re-rank group failed: {e}")
                return [(5, offset + j) for j in range(len(group))]

        # Fire all groups in parallel
        group_tasks = [
            _score_group(group, i * group_size)
            for i, group in enumerate(groups)
        ]
        group_results = await asyncio.gather(*group_tasks)

        # Flatten and sort by score descending
        all_scored = []
        for results in group_results:
            all_scored.extend(results)
        all_scored.sort(key=lambda x: -x[0])

        return [candidates[idx] for _, idx in all_scored if idx < len(candidates)][:top_k]

    # Precompiled patterns for query classification
    _RE_CAMELCASE = re.compile(r'[A-Z][a-z]+[A-Z]')       # CamelCase identifiers
    _RE_SNAKE = re.compile(r'[a-z]+_[a-z]+')               # snake_case identifiers
    _RE_DOTPATH = re.compile(r'\w+\.\w+\.\w+')             # dotted paths (a.b.c)
    _RE_FILE_EXT = re.compile(r'\.\w{1,4}(?:\s|$|[,)])')   # file extensions (.py, .yaml)
    _RE_ARCHITECTURAL = re.compile(
        r'\b(?:trace|lifecycle|pipeline|flow|architecture|end.to.end|'
        r'full\s+path|how\s+does\s+\w+\s+handle|what\s+happens\s+when)\b',
        re.IGNORECASE,
    )
    _RE_CROSS_DOMAIN = re.compile(
        r'\b(?:config|yaml|services\.yaml|\.env|port|how\s+do\s+I\s+add|'
        r'set\s*up|deploy|install|register)\b',
        re.IGNORECASE,
    )
    # Relationship patterns: "X interacts with Y", "X links to Y", "across"
    _RE_RELATIONSHIP = re.compile(
        r'\b(?:interact|route\s+\w+\s+to|link|connect|between|across|'
        r'communicate|integrate|layers|and\s+how|query\s+and)\b',
        re.IGNORECASE,
    )

    @staticmethod
    def classify_query(query: str) -> Tuple[float, float, str]:
        """
        Classify a query and return optimal (keyword_weight, semantic_weight, reason).

        Categories (from grid search data, validated by benchmark):
            relationship   -> kw=0.8, sem=0.2  (cross-entity interactions, need exact file matching)
            conceptual     -> kw=0.2, sem=0.8  (natural language, conceptual reasoning)
            architectural  -> kw=0.0, sem=1.0  (cross-abstraction, multi-file tracing)
            cross_domain   -> kw=0.9, sem=0.1  (code+config, literal identifiers across domains)
            focused        -> kw=0.3, sem=0.7  (single entity behavior, embeddings map cleanly)
        """
        q = query.strip()

        # Count signal types
        camel_hits = len(CodeGraph._RE_CAMELCASE.findall(q))
        snake_hits = len(CodeGraph._RE_SNAKE.findall(q))
        dot_hits = len(CodeGraph._RE_DOTPATH.findall(q))
        file_hits = len(CodeGraph._RE_FILE_EXT.findall(q))
        symbol_count = camel_hits + snake_hits + dot_hits + file_hits

        is_architectural = bool(CodeGraph._RE_ARCHITECTURAL.search(q))
        is_cross_domain = bool(CodeGraph._RE_CROSS_DOMAIN.search(q))
        is_relationship = bool(CodeGraph._RE_RELATIONSHIP.search(q))

        words = q.split()
        word_count = len(words)

        # Decision tree (ordered by specificity, tuned by grid search data)

        # 1. Architectural: "trace the full lifecycle of..."
        #    Pure semantic -- needs conceptual reasoning across abstractions
        #    Grid: hard queries peak at kw=0.0/sem=1.0
        if is_architectural and word_count > 6:
            return (0.0, 1.0, "architectural")

        # 2. Cross-domain: mentions config/yaml/ports
        #    Near-pure keyword -- cross-file refs share literal identifiers
        #    Grid: cross-domain peaks at kw=1.0/sem=0.0
        if is_cross_domain:
            return (0.9, 0.1, "cross_domain")

        # 3. Relationship queries: "how does X interact with Y", "X links Y to Z"
        #    Keyword-heavy -- need exact symbol matching across multiple files
        #    Grid: medium queries (all relationship) peak at kw=0.8-0.9/sem=0.1-0.2
        if is_relationship and symbol_count >= 1:
            return (0.8, 0.2, "relationship")

        # 4. Symbol-dense (2+) without relationship -> still keyword-leaning
        if symbol_count >= 2:
            return (0.7, 0.3, "multi_symbol")

        # 5. Single entity focused: "How does ContextEngine inject items?"
        #    Semantic-heavy -- embedding space captures straightforward intent cleanly
        #    Grid: simple queries peak at kw=0.3/sem=0.7
        if symbol_count >= 1:
            return (0.3, 0.7, "focused")

        # 6. Long pure natural language -- conceptual
        if word_count > 6:
            return (0.2, 0.8, "conceptual")

        # 7. Short/ambiguous
        return (0.4, 0.6, "balanced")

    async def hybrid_query(
        self,
        query: str,
        max_results: int = 10,
        keyword_weight: float | None = None,
        semantic_weight: float | None = None,
        expand_context: bool = True,
        rerank: bool | str = False,
        **kwargs,
    ) -> List["CodeChunk"]:
        """
        Hybrid search combining keyword scoring and semantic similarity.

        Falls back to keyword-only if no embeddings are available.

        Pipeline: classify -> keyword + semantic -> merge -> context expansion -> (optional) re-rank

        Args:
            query: Natural language or keyword query
            max_results: Maximum results
            keyword_weight: Weight for keyword score (0-1). None = auto-classify.
            semantic_weight: Weight for semantic score (0-1). None = auto-classify.
            expand_context: Pull parent class, sibling methods, file neighbors
            rerank: False=off, True/'embedding'=fast embedding rerank (~5ms),
                    'llm'=parallel LLM scoring, 'hybrid'=embedding then LLM
        """
        # Adaptive weight classification
        query_type = "balanced"  # default if weights provided explicitly
        if keyword_weight is None or semantic_weight is None:
            keyword_weight, semantic_weight, query_type = self.classify_query(query)
            logger.debug(f"Query classified as '{query_type}': kw={keyword_weight}, sem={semantic_weight}")

        # Always run keyword search
        keyword_results = await self.query(query, max_results=max_results * 3)

        # Check if semantic search is available (cached flag avoids 28K-item scan)
        if not hasattr(self, '_has_embeddings_cached'):
            self._has_embeddings_cached = any(
                c.embedding is not None for c in self.chunks.values()
            )
        if not self._has_embeddings_cached or not _HAS_EMBEDDING_ENGINE:
            return keyword_results[:max_results]

        # Run semantic search.
        # If query embedding is cached: cosine math only (~1ms) -- always succeeds.
        # If uncached: vLLM embed call -- cap at 150ms, fire background cache warmup.
        cache_hit = query in self._query_embed_cache
        timeout = 1.0 if cache_hit else 0.15  # Cache hit = instant; miss = tight cap
        try:
            semantic_results = await asyncio.wait_for(
                self.semantic_query(query, max_results=max_results * 3),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.debug("[HYBRID] Semantic timed out — keyword-only + background cache warmup")
            self._background_cache_query(query)  # Next time will be a cache hit
            semantic_results = None
        if not semantic_results:
            if not cache_hit:
                self._background_cache_query(query)  # Pre-cache for next call
            return keyword_results[:max_results]

        # Build normalized score maps
        # Keyword: reciprocal rank (1st=1.0, 2nd=0.5, 3rd=0.33, ...)
        kw_scores: Dict[str, float] = {}
        for rank, chunk in enumerate(keyword_results):
            kw_scores[chunk.id] = 1.0 / (rank + 1)

        # Semantic: cosine similarity normalized to 0-1 relative to top
        sem_scores: Dict[str, float] = {}
        if semantic_results:
            max_sim = semantic_results[0][0]
            for sim, chunk in semantic_results:
                sem_scores[chunk.id] = sim / max_sim if max_sim > 0 else 0.0

        # Merge and score
        all_ids = set(kw_scores.keys()) | set(sem_scores.keys())
        combined = []
        for cid in all_ids:
            kw = kw_scores.get(cid, 0.0)
            sem = sem_scores.get(cid, 0.0)
            score = keyword_weight * kw + semantic_weight * sem
            # Deprioritize test files in hybrid results too
            chunk = self.chunks.get(cid)
            if chunk and self._is_test_path(chunk.source_path):
                score *= self._TEST_PENALTY
            combined.append((score, cid))

        combined.sort(key=lambda x: -x[0])
        top_ids = [cid for _, cid in combined[:max_results] if cid in self.chunks]
        top_chunks = [self.chunks[cid] for cid in top_ids]

        # Context expansion: pull structurally + semantically related chunks
        if expand_context and top_ids:
            expanded = await self._expand_context(
                set(top_ids[:5]), query=query, max_expand=max_results,
            )
            # Append expanded chunks after the scored results
            top_chunks = top_chunks + expanded

            # Multi-hop chain expansion for architectural queries
            if query_type == "architectural":
                try:
                    chains = self._multi_hop_expand(top_ids[:5], query)
                    if chains:
                        scored_chains = self._score_chains(chains, query)
                        for _score, chain in scored_chains[:3]:
                            top_chunks.extend(chain)
                        logger.debug(
                            f"[MULTI_HOP] {len(chains)} chains found, "
                            f"top 3 injected ({sum(len(c) for _, c in scored_chains[:3])} chunks)"
                        )
                except Exception as e:
                    logger.debug(f"[MULTI_HOP] Chain expansion failed: {e}")

            # Deduplicate while preserving order
            seen: Set[str] = set()
            deduped = []
            for c in top_chunks:
                if c.id not in seen:
                    seen.add(c.id)
                    deduped.append(c)
            top_chunks = deduped[:max_results * 2]

        # Re-ranking: explicit mode or auto-apply embedding rerank when
        # expand_context added extra chunks that need to compete for slots
        if rerank:
            mode = rerank if isinstance(rerank, str) else "embedding"
            top_chunks = await self._rerank(query, top_chunks, top_k=max_results, mode=mode)
        elif expand_context and len(top_chunks) > max_results:
            # Expanded chunks need scoring to compete with originals
            top_chunks = await self._rerank_by_embedding(query, top_chunks, top_k=max_results)

        return top_chunks[:max_results]

    # -- Full body retrieval -------------------------------------------

    def get_full_body(self, chunk_id: str) -> Optional[str]:
        """
        Get the full source text of a chunk by reading the source file.

        Lazy-loads and caches the result. Falls back to body_preview
        if the source file is missing.

        Args:
            chunk_id: The chunk ID to look up.

        Returns:
            Full source text, or body_preview fallback, or None if unknown.
        """
        if chunk_id in self._body_cache:
            return self._body_cache[chunk_id]

        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None

        # Try reading from source file
        root = self._root_path or ""
        source_path = os.path.join(root, chunk.source_path) if root else chunk.source_path

        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            # Extract the chunk's line range (1-indexed start_line/end_line)
            start = max(0, chunk.start_line - 1)
            end = min(len(lines), chunk.end_line)
            body = "".join(lines[start:end])
            self._body_cache[chunk_id] = body
            return body
        except (FileNotFoundError, OSError):
            # Fall back to body_preview
            preview = getattr(chunk, "body_preview", None)
            if preview:
                self._body_cache[chunk_id] = preview
            return preview

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage of the CodeGraph in megabytes."""
        total_bytes = 0
        # Chunks
        for chunk in self.chunks.values():
            total_bytes += len(getattr(chunk, "body_preview", "") or "")
            total_bytes += len(getattr(chunk, "signature", "") or "")
            total_bytes += len(getattr(chunk, "docstring", "") or "")
            if chunk.embedding:
                total_bytes += len(chunk.embedding) * 8  # 64-bit floats
        # Body cache
        for body in self._body_cache.values():
            total_bytes += len(body)
        # Query caches
        for emb in self._query_embed_cache.values():
            total_bytes += len(emb) * 8
        # Embedding matrix
        if self._embedding_matrix is not None and _HAS_NUMPY:
            total_bytes += self._embedding_matrix.nbytes
        return total_bytes / (1024 * 1024)

    # ========================================================================
    # PYTHON METRICS -- instant from in-memory index
    # ========================================================================

    _AREA_PREFIXES = [
        ("lib/", "lib"), ("services/", "services"),
        ("apps/AitherVeil/", "frontend"), ("apps/AitherGenesis/", "genesis"),
        ("apps/AitherNode/", "node"), ("apps/", "apps"),
        ("dev/tests/", "tests"), ("boot/", "boot"),
        ("config/", "config"), ("scripts/", "scripts"),
    ]

    def _classify_area(self, path: str) -> str:
        """Classify a Python file into a project area."""
        normalized = path.replace("\\", "/")
        for prefix, area in self._AREA_PREFIXES:
            if prefix in normalized:
                idx = normalized.find(prefix)
                if idx >= 0 and normalized[idx:].startswith(prefix):
                    return area
        return "other"

    def get_python_metrics(self) -> Dict[str, Any]:
        """
        Instant Python-specific metrics from in-memory CodeGraph index.

        O(n) over chunks, typically <10ms for ~5000 chunks.

        Returns dict with:
            total_py_files, total_chunks, total_py_lines, functions, classes,
            methods, avg_complexity, top_complex_files, by_area, test_functions,
            test_lines
        """
        if not self.chunks:
            return {
                "total_py_files": 0, "total_chunks": 0, "total_py_lines": 0,
                "functions": 0, "classes": 0, "methods": 0,
                "avg_complexity": 0.0, "top_complex_files": [],
                "by_area": {}, "test_functions": 0, "test_lines": 0,
            }

        functions = 0
        classes = 0
        methods = 0
        complexities = []
        file_complexity: Dict[str, List[int]] = {}
        area_lines: Dict[str, int] = {}
        test_functions = 0
        test_lines = 0

        for chunk in self.chunks.values():
            ct = chunk.chunk_type
            if ct == ChunkType.FUNCTION:
                functions += 1
                if chunk.name.startswith("test_"):
                    test_functions += 1
            elif ct == ChunkType.METHOD:
                methods += 1
                if chunk.name.startswith("test_"):
                    test_functions += 1
            elif ct == ChunkType.CLASS:
                classes += 1

            if chunk.complexity:
                complexities.append(chunk.complexity)
                fp = chunk.source_path
                if fp not in file_complexity:
                    file_complexity[fp] = []
                file_complexity[fp].append(chunk.complexity)

            lc = chunk.line_count or (chunk.end_line - chunk.start_line + 1)
            area = self._classify_area(chunk.source_path)
            area_lines[area] = area_lines.get(area, 0) + lc

            # Test lines
            path_lower = chunk.source_path.lower().replace("\\", "/")
            if ("/tests/" in path_lower or "/test/" in path_lower
                    or "/test_" in path_lower):
                test_lines += lc

        # Top complex files
        file_avg: List[Tuple[str, float]] = []
        for fp, cxs in file_complexity.items():
            file_avg.append((fp, sum(cxs) / len(cxs)))
        file_avg.sort(key=lambda x: x[1], reverse=True)

        avg_cx = sum(complexities) / len(complexities) if complexities else 0.0

        return {
            "total_py_files": len(self.by_file),
            "total_chunks": len(self.chunks),
            "total_py_lines": sum(area_lines.values()),
            "functions": functions,
            "classes": classes,
            "methods": methods,
            "avg_complexity": round(avg_cx, 2),
            "top_complex_files": [
                (Path(fp).name, round(cx, 2)) for fp, cx in file_avg[:10]
            ],
            "by_area": area_lines,
            "test_functions": test_functions,
            "test_lines": test_lines,
        }


# ============================================================================
# CLI
# ============================================================================

async def main():
    """CLI entry point."""
    import sys

    root_path = sys.argv[1] if len(sys.argv) > 1 else str(Path.cwd())

    print(f"\n{'='*60}")
    print("CODE GRAPH - Python AST Indexer")
    print(f"{'='*60}")
    print(f"Root: {root_path}")

    graph = CodeGraph(max_workers=8)

    def on_progress(progress: float, message: str):
        print(f"[{progress*100:5.1f}%] {message}")

    stats = await graph.index_codebase(root_path, on_progress=on_progress)

    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"Files:      {stats['total_files']:,}")
    print(f"Chunks:     {stats['total_chunks']:,}")
    print(f"  Functions: {stats['functions']:,}")
    print(f"  Methods:   {stats['methods']:,}")
    print(f"  Classes:   {stats['classes']:,}")
    print(f"\nTiming:")
    print(f"  Discovery:  {stats['discovery_ms']:.0f}ms")
    print(f"  Parsing:    {stats['parsing_ms']:.0f}ms")
    print(f"  Call graph: {stats['backfill_ms']:.0f}ms")
    print(f"  Total:      {stats['total_ms']:.0f}ms")

    # Test query
    print(f"\n{'='*60}")
    print("QUERY TEST: 'IRCEngine'")
    print(f"{'='*60}")

    chunks = await graph.query("IRCEngine", max_results=5)

    for i, chunk in enumerate(chunks):
        print(f"\n{i+1}. {chunk.chunk_type.value}: {chunk.name}")
        print(f"   File: {Path(chunk.source_path).name}:{chunk.start_line}")
        print(f"   Calls: {chunk.calls[:5]}{'...' if len(chunk.calls) > 5 else ''}")
        print(f"   Called by: {chunk.called_by[:5]}{'...' if len(chunk.called_by) > 5 else ''}")

    # Show context for first result
    if chunks:
        print(f"\n{'='*60}")
        print(f"FULL CONTEXT FOR: {chunks[0].name}")
        print(f"{'='*60}")

        ctx = graph.get_context_for_chunk(chunks[0].id)
        print(f"\nCallers ({len(ctx['callers'])}):")
        for c in ctx["callers"][:3]:
            print(f"  - {c.name} ({c.chunk_type.value})")

        print(f"\nCallees ({len(ctx['callees'])}):")
        for c in ctx["callees"][:3]:
            print(f"  - {c.name} ({c.chunk_type.value})")


# ============================================================================
# SINGLETON + PERSISTENT INDEX
# ============================================================================

_codegraph_instance: Optional["CodeGraph"] = None
_codegraph_lock = threading.Lock()
_codegraph_indexing = False  # Guard against concurrent index attempts


def get_codegraph(
    root_path: Optional[str] = None,
    auto_index: bool = True,
    max_workers: int = 8,
) -> "CodeGraph":
    """
    Get or create the CodeGraph singleton.

    First call indexes the codebase and loads embeddings.
    Subsequent calls return the same instance with warm cache.

    Thread-safe: concurrent callers during indexing get the instance
    immediately (possibly with empty chunks) rather than blocking.

    Args:
        root_path: Codebase root to index. Default: current working directory.
                   Can be ANY project path for onboarding external codebases.
        auto_index: If True, index + load embeddings on first call.
        max_workers: Number of threads/processes for indexing.
    """
    global _codegraph_instance, _codegraph_indexing

    if _codegraph_instance is not None and _codegraph_instance.chunks:
        return _codegraph_instance

    with _codegraph_lock:
        if _codegraph_instance is not None and _codegraph_instance.chunks:
            return _codegraph_instance

        # Only create if not exists - preserve instance if warming up (empty chunks)
        if _codegraph_instance is None:
            cg = CodeGraph(max_workers=max_workers)
            _codegraph_instance = cg
        else:
            cg = _codegraph_instance

        # If another thread/task is already indexing, return immediately
        # to avoid stacking concurrent index operations
        if _codegraph_indexing:
            logger.debug("[CodeGraph] Index already in progress — returning instance as-is")
            return cg

        if auto_index:
            _codegraph_indexing = True
            try:
                if root_path is None:
                    root_path = str(Path.cwd())

                # Try loading persistent chunk cache first
                cache_loaded = _load_chunk_cache(cg, root_path)

                if not cache_loaded:
                    # Full index required
                    # Detect if we're inside an already-running event loop (e.g. uvicorn)
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None

                    if running_loop is not None:
                        # We're inside an async context — schedule indexing as a background task
                        # Return the un-indexed instance; callers handle empty .chunks gracefully
                        logger.info("[CodeGraph] Async context detected — scheduling background index")

                        async def _bg_index(cg_ref, rp):
                            global _codegraph_indexing
                            try:
                                await cg_ref.index_codebase(rp)
                                _save_chunk_cache(cg_ref, rp)
                                logger.info(f"[CodeGraph] Background index complete: {len(cg_ref.chunks)} chunks")
                            except Exception as exc:
                                logger.warning(f"[CodeGraph] Background index failed: {exc}")
                            finally:
                                _codegraph_indexing = False

                        running_loop.create_task(_bg_index(cg, root_path))
                    else:
                        # Sync context — safe to create a new loop
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(cg.index_codebase(root_path))
                            _save_chunk_cache(cg, root_path)
                        finally:
                            loop.close()
                            _codegraph_indexing = False
                else:
                    _codegraph_indexing = False
            except Exception:
                _codegraph_indexing = False
                raise

            # Load embedding cache (always fast — just applies dict to chunks)
            embed_path = _get_data_path(root_path, "codegraph_embeddings.pkl")
            if os.path.exists(embed_path):
                if not _verify_pickle_hmac(embed_path):
                    logger.warning("[CodeGraph] Embedding cache HMAC invalid — skipping")
                else:
                    try:
                        with open(embed_path, "rb") as f:
                            cached = pickle.load(f)
                        applied = 0
                        for cid, emb in cached.items():
                            if cid in cg.chunks:
                                cg.chunks[cid].embedding = emb
                                applied += 1
                        logger.info(f"[CodeGraph] Loaded {applied} embeddings from cache")
                    except Exception as e:
                        logger.warning(f"[CodeGraph] Embedding cache load failed: {e}")

        return cg


def _get_data_path(root_path: str, filename: str) -> str:
    """Get path in the .aither/data dir relative to root."""
    data_dir = os.path.join(root_path, ".aither", "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def _save_chunk_cache(cg: "CodeGraph", root_path: str):
    """Persist parsed chunks + file mtimes for instant reload."""
    cache = {
        "chunks": {},
        "mtimes": {},
        "stats": {
            "total_files": cg.total_files,
            "total_chunks": len(cg.chunks),
        },
    }
    for cid, chunk in cg.chunks.items():
        cache["chunks"][cid] = {
            "id": chunk.id,
            "name": chunk.name,
            "chunk_type": chunk.chunk_type.value,
            "source_path": chunk.source_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "signature": chunk.signature,
            "docstring": chunk.docstring,
            "body_preview": chunk.body_preview,
            "calls": chunk.calls,
            "called_by": chunk.called_by,
            "parent_class": chunk.parent_class,
            "base_classes": chunk.base_classes,
            "imports": chunk.imports,
        }

    # Record file mtimes for incremental detection
    for fpath in cg.by_file:
        try:
            full_path = os.path.join(root_path, fpath) if not os.path.isabs(fpath) else fpath
            if os.path.exists(full_path):
                cache["mtimes"][fpath] = os.path.getmtime(full_path)
        except OSError as e:
            logger.debug(f"[CodeGraph._save_chunk_cache] Operation failed: {e}")

    cache_path = _get_data_path(root_path, "codegraph_chunks.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_pickle_hmac(cache_path)
        logger.info(f"[CodeGraph] Saved {len(cache['chunks'])} chunks to cache")
    except Exception as e:
        logger.warning(f"[CodeGraph] Failed to save chunk cache: {e}")


def _load_chunk_cache(cg: "CodeGraph", root_path: str) -> bool:
    """Load persisted chunks from disk. Returns True if cache was valid."""
    cache_path = _get_data_path(root_path, "codegraph_chunks.pkl")
    if not os.path.exists(cache_path):
        return False

    if not _verify_pickle_hmac(cache_path):
        logger.warning("[CodeGraph] Chunk cache HMAC invalid — treating as missing")
        return False

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception as e:
        logger.warning(f"[CodeGraph] Chunk cache corrupted: {e}")
        return False

    chunks_data = cache.get("chunks", {})
    if not chunks_data:
        return False

    # Reconstruct CodeChunk objects
    for cid, data in chunks_data.items():
        try:
            chunk = CodeChunk(
                id=data["id"],
                name=data["name"],
                chunk_type=ChunkType(data["chunk_type"]),
                source_path=data["source_path"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                signature=data["signature"],
                docstring=data.get("docstring"),
                body_preview=data.get("body_preview", data.get("body", "")),
                calls=data.get("calls", []),
                called_by=data.get("called_by", []),
                parent_class=data.get("parent_class"),
                base_classes=data.get("base_classes", []),
                imports=data.get("imports", []),
            )
            cg.chunks[cid] = chunk
            cg.by_name[chunk.name].append(cid)
            cg.by_file[chunk.source_path].append(cid)
            if chunk.parent_class:
                cg.by_class[chunk.parent_class].append(cid)
        except Exception:
            continue

    cg.total_files = cache.get("stats", {}).get("total_files", len(cg.by_file))
    logger.info(
        f"[CodeGraph] Loaded {len(cg.chunks)} chunks from cache "
        f"({cg.total_files} files)"
    )
    return True


async def reindex_files(changed_files: List[str], root_path: Optional[str] = None):
    """
    Incrementally re-index changed files in the live singleton.

    This is the correct way to update the index when files change.
    Uses parse_file_sync() (the same parser as full index), then
    re-embeds only the changed chunks.
    """
    cg = get_codegraph(root_path=root_path)

    if root_path is None:
        root_path = str(Path.cwd())

    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return 0

    reindexed = 0
    new_chunk_ids = []

    for filepath in py_files[:50]:
        try:
            # Remove old chunks for this file
            rel_path = filepath
            if os.path.isabs(filepath):
                try:
                    rel_path = os.path.relpath(filepath, root_path)
                except ValueError:
                    rel_path = filepath

            old_ids = cg.by_file.get(rel_path, [])
            for oid in old_ids:
                cg.chunks.pop(oid, None)
                for name_list in cg.by_name.values():
                    if oid in name_list:
                        name_list.remove(oid)
            cg.by_file[rel_path] = []

            # Re-parse the file
            abs_path = filepath if os.path.isabs(filepath) else os.path.join(root_path, filepath)
            if not os.path.exists(abs_path):
                continue

            file_graph = parse_file_sync(abs_path)
            for chunk in file_graph.chunks:
                cg.chunks[chunk.id] = chunk
                cg.by_name[chunk.name].append(chunk.id)
                cg.by_file[file_graph.source_path].append(chunk.id)
                if chunk.parent_class:
                    cg.by_class[chunk.parent_class].append(chunk.id)
                new_chunk_ids.append(chunk.id)
                # Sync to knowledge graph
                cg._queue_sync({
                    "id": chunk.id,
                    "name": chunk.name,
                    "type": chunk.chunk_type.value,
                    "properties": {
                        "source_path": chunk.source_path,
                        "signature": chunk.signature,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    },
                })

            reindexed += 1
        except Exception as e:
            logger.debug(f"[CodeGraph] Failed to re-index {filepath}: {e}")

    # Backfill call graph for new chunks
    if new_chunk_ids:
        cg._backfill_called_by()

    # Re-embed new chunks (incremental — only the changed ones)
    if new_chunk_ids and cg.embedding_coverage > 0:
        need_embed = [
            (cid, cg.chunks[cid]) for cid in new_chunk_ids
            if cid in cg.chunks and cg.chunks[cid].embedding is None
        ]
        if need_embed:
            texts = [c.signature + "\n" + (c.docstring or "") + "\n" + c.body_preview
                     for _, c in need_embed]
            embeddings = await _embed_texts(texts)
            for (cid, chunk), emb in zip(need_embed, embeddings):
                if emb:
                    chunk.embedding = emb

    # Persist updated cache
    _save_chunk_cache(cg, root_path)

    if reindexed > 0:
        cg._invalidate_embedding_matrix()  # Force matrix rebuild on next query
        cg._invalidate_keyword_cache()  # Clear keyword result cache
        logger.info(f"[CodeGraph] Incrementally re-indexed {reindexed} files "
                     f"({len(new_chunk_ids)} chunks)")

    return reindexed


if __name__ == "__main__":
    asyncio.run(main())
