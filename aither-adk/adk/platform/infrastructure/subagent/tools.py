"""
Sub-Agent Tools - Tools available to spawned sub-agents
========================================================

These tools are designed for research and exploration.
They are READ-ONLY - sub-agents cannot modify the codebase.

Tools:
    - search_codebase: Search for patterns in files
    - read_file_section: Read specific lines from a file
    - list_directory: List contents of a directory
    - get_file_info: Get metadata about a file
    - find_definitions: Find function/class definitions
    - trace_imports: Trace import dependencies
    - get_project_structure: Get project layout

Author: Aitherium
"""

import ast
import fnmatch
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SubAgentTools")


@dataclass
class SearchMatch:
    """A match from a codebase search."""
    file_path: str
    relative_path: str
    line_number: int
    line_content: str
    match_text: str
    context_before: List[str]
    context_after: List[str]


class SubAgentTools:
    """
    Collection of tools for sub-agent codebase exploration.

    All tools are read-only and designed for research.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._default_excludes = [
            "node_modules", ".venv", "__pycache__", ".git",
            ".tox", "dist", "build", "*.egg-info", ".mypy_cache"
        ]

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        return any(
            exc in path_str or fnmatch.fnmatch(path.name, exc)
            for exc in self._default_excludes
        )

    async def search_codebase(
        self,
        pattern: str,
        paths: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
        max_results: int = 50,
        context_lines: int = 2,
        case_sensitive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for a pattern in the codebase.

        Args:
            pattern: Text or regex pattern to search
            paths: Directories to search (relative to workspace root)
            file_patterns: File glob patterns (e.g., "*.py")
            max_results: Maximum matches to return
            context_lines: Lines of context before/after match
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of matches with file, line, and context
        """
        paths = paths or ["."]
        file_patterns = file_patterns or ["*.py", "*.ps1", "*.ts", "*.js", "*.md"]

        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)

        matches = []
        files_searched = 0

        for search_path in paths:
            full_path = self.workspace_root / search_path
            if not full_path.exists():
                continue

            for file_path in full_path.rglob("*"):
                if len(matches) >= max_results:
                    break

                if not file_path.is_file():
                    continue

                if self._should_exclude(file_path):
                    continue

                if not any(fnmatch.fnmatch(file_path.name, fp) for fp in file_patterns):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    files_searched += 1

                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            start = max(0, line_num - 1 - context_lines)
                            end = min(len(lines), line_num + context_lines)

                            matches.append({
                                "file_path": str(file_path),
                                "relative_path": str(file_path.relative_to(self.workspace_root)),
                                "line_number": line_num,
                                "line_content": line,
                                "pattern": pattern,
                                "context_before": lines[start:line_num - 1],
                                "context_after": lines[line_num:end],
                            })

                            if len(matches) >= max_results:
                                break
                except Exception as e:
                    logger.debug(f"Error reading {file_path}: {e}")

        return matches

    async def read_file_section(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_lines: int = 100,
    ) -> Dict[str, Any]:
        """
        Read a section of a file.

        Args:
            file_path: Path to the file (absolute or relative to workspace)
            start_line: First line to read (1-indexed)
            end_line: Last line to read (None = start_line + max_lines)
            max_lines: Maximum lines to read

        Returns:
            Dict with content, line numbers, and metadata
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        if not path.exists():
            return {
                "error": f"File not found: {file_path}",
                "exists": False,
            }

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            # Adjust indices (1-indexed to 0-indexed)
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else start_idx + max_lines
            end_idx = min(end_idx, total_lines)

            section = lines[start_idx:end_idx]

            return {
                "file_path": str(path),
                "relative_path": str(path.relative_to(self.workspace_root)) if path.is_relative_to(self.workspace_root) else str(path),
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "total_lines": total_lines,
                "content": "\n".join(section),
                "lines": [
                    {"number": start_idx + i + 1, "content": line}
                    for i, line in enumerate(section)
                ],
                "exists": True,
                "file_type": self._get_file_type(path),
            }
        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(path),
                "exists": path.exists(),
            }

    async def list_directory(
        self,
        path: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
        file_patterns: Optional[List[str]] = None,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        """
        List contents of a directory.

        Args:
            path: Directory path (relative to workspace root)
            recursive: Whether to list subdirectories
            max_depth: Maximum depth for recursive listing
            file_patterns: Filter by file patterns
            include_hidden: Include hidden files/directories

        Returns:
            Dict with directory contents organized by type
        """
        dir_path = self.workspace_root / path

        if not dir_path.exists():
            return {"error": f"Directory not found: {path}"}

        if not dir_path.is_dir():
            return {"error": f"Not a directory: {path}"}

        def list_dir(p: Path, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {"truncated": True}

            result = {
                "name": p.name,
                "path": str(p.relative_to(self.workspace_root)),
                "type": "directory",
                "children": [],
            }

            try:
                entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

                for entry in entries:
                    if not include_hidden and entry.name.startswith("."):
                        continue

                    if self._should_exclude(entry):
                        continue

                    if entry.is_dir():
                        if recursive:
                            result["children"].append(list_dir(entry, depth + 1))
                        else:
                            result["children"].append({
                                "name": entry.name,
                                "path": str(entry.relative_to(self.workspace_root)),
                                "type": "directory",
                            })
                    else:
                        if file_patterns:
                            if not any(fnmatch.fnmatch(entry.name, fp) for fp in file_patterns):
                                continue

                        try:
                            stat = entry.stat()
                            result["children"].append({
                                "name": entry.name,
                                "path": str(entry.relative_to(self.workspace_root)),
                                "type": "file",
                                "size": stat.st_size,
                                "file_type": self._get_file_type(entry),
                            })
                        except Exception as exc:
                            logger.debug(f"Failed to stat file entry: {exc}")
            except PermissionError:
                result["error"] = "Permission denied"

            return result

        return list_dir(dir_path, 0)

    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dict with file metadata
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        if not path.exists():
            return {"error": f"File not found: {file_path}", "exists": False}

        try:
            stat = path.stat()
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            return {
                "file_path": str(path),
                "relative_path": str(path.relative_to(self.workspace_root)) if path.is_relative_to(self.workspace_root) else str(path),
                "exists": True,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": stat.st_size,
                "line_count": len(lines),
                "file_type": self._get_file_type(path),
                "extension": path.suffix,
                "name": path.name,
                "parent": str(path.parent),
            }
        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(path),
                "exists": path.exists(),
            }

    async def find_definitions(
        self,
        file_path: str,
        definition_type: str = "all",  # "function", "class", "all"
    ) -> Dict[str, Any]:
        """
        Find function and class definitions in a Python file.

        Args:
            file_path: Path to the Python file
            definition_type: Type of definitions to find

        Returns:
            Dict with found definitions
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if path.suffix != ".py":
            return {"error": "Only Python files are supported for definition extraction"}

        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            functions = []
            classes = []

            for node in ast.walk(tree):
                if definition_type in ("function", "all") and isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [
                            ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                            for d in node.decorator_list
                        ],
                        "docstring": ast.get_docstring(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    })

                if definition_type in ("class", "all") and isinstance(node, ast.ClassDef):
                    methods = [
                        {"name": n.name, "line": n.lineno}
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]

                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                        "bases": [
                            ast.unparse(b) if hasattr(ast, 'unparse') else str(b)
                            for b in node.bases
                        ],
                        "methods": methods,
                        "docstring": ast.get_docstring(node),
                    })

            return {
                "file_path": str(path),
                "functions": functions,
                "classes": classes,
                "total_functions": len(functions),
                "total_classes": len(classes),
            }

        except SyntaxError as e:
            return {
                "error": f"Syntax error in file: {e}",
                "file_path": str(path),
            }
        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(path),
            }

    async def trace_imports(
        self,
        file_path: str,
        direction: str = "both",  # "imports", "imported_by", "both"
    ) -> Dict[str, Any]:
        """
        Trace import dependencies for a Python file.

        Args:
            file_path: Path to the Python file
            direction: Which direction to trace

        Returns:
            Dict with import information
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        if path.suffix != ".py":
            return {"error": "Only Python files are supported for import tracing"}

        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            imports = []
            from_imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                        })

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        from_imports.append({
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                            "level": node.level,  # Relative import level
                        })

            result = {
                "file_path": str(path),
                "imports": imports,
                "from_imports": from_imports,
                "total_imports": len(imports) + len(from_imports),
            }

            # Find files that import this one
            if direction in ("imported_by", "both"):
                imported_by = await self._find_files_importing(path)
                result["imported_by"] = imported_by

            return result

        except Exception as e:
            return {
                "error": str(e),
                "file_path": str(path),
            }

    async def _find_files_importing(self, target_file: Path) -> List[Dict[str, Any]]:
        """Find files that import the target file."""
        # Get the module name from the file path
        rel_path = target_file.relative_to(self.workspace_root)
        module_parts = list(rel_path.parts)
        if module_parts[-1].endswith(".py"):
            module_parts[-1] = module_parts[-1][:-3]
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]

        module_name = ".".join(module_parts)
        importing_files = []

        # Search for imports
        for py_file in self.workspace_root.rglob("*.py"):
            if self._should_exclude(py_file):
                continue
            if py_file == target_file:
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                if module_name in content or module_parts[-1] in content:
                    importing_files.append({
                        "file_path": str(py_file),
                        "relative_path": str(py_file.relative_to(self.workspace_root)),
                    })
            except Exception as exc:
                logger.debug(f"Failed to read file for import tracing: {exc}")

        return importing_files

    async def get_project_structure(
        self,
        max_depth: int = 2,
        include_file_count: bool = True,
    ) -> Dict[str, Any]:
        """
        Get an overview of the project structure.

        Args:
            max_depth: How deep to show the structure
            include_file_count: Include file counts per directory

        Returns:
            Dict with project structure and statistics
        """
        structure = await self.list_directory(
            ".",
            recursive=True,
            max_depth=max_depth,
        )

        # Count files by type
        type_counts = {}
        total_files = 0

        def count_files(node: Dict[str, Any]):
            nonlocal total_files
            if node.get("type") == "file":
                total_files += 1
                ft = node.get("file_type", "unknown")
                type_counts[ft] = type_counts.get(ft, 0) + 1
            for child in node.get("children", []):
                count_files(child)

        count_files(structure)

        # Find key files
        key_files = []
        key_patterns = [
            "README.md", "package.json", "pyproject.toml",
            "setup.py", "requirements.txt", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle",
        ]

        for pattern in key_patterns:
            for match in self.workspace_root.glob(pattern):
                key_files.append(str(match.relative_to(self.workspace_root)))

        return {
            "root": str(self.workspace_root),
            "structure": structure,
            "statistics": {
                "total_files": total_files,
                "by_type": type_counts,
            },
            "key_files": key_files,
        }

    def _get_file_type(self, path: Path) -> str:
        """Get the file type from extension."""
        ext_map = {
            ".py": "python",
            ".ps1": "powershell",
            ".psm1": "powershell",
            ".psd1": "powershell",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".sh": "shell",
            ".rs": "rust",
            ".go": "go",
        }
        return ext_map.get(path.suffix.lower(), "text")


# Create a global instance for easy access
_tools: Optional[SubAgentTools] = None


def get_tools(workspace_root: Optional[str] = None) -> SubAgentTools:
    """Get or create the global SubAgentTools instance."""
    global _tools
    if _tools is None:
        _tools = SubAgentTools(workspace_root=workspace_root)
    return _tools
