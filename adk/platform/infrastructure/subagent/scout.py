"""
AitherScout - Research-Focused Sub-Agent for Codebase Exploration
==================================================================

Scouts are the "eyes" of the parent agent. They:
1. Explore file systems methodically
2. Search for patterns in code
3. Build understanding of code structure
4. Report objective findings with evidence

Key Principles:
- OBJECTIVE: Report what exists, not interpretations
- GROUNDED: Every finding has a file path and line number
- TRACEABLE: Show the actual code, not summaries
- THOROUGH: Cover all requested search paths

Events:
- Emits progress events to AitherPulse during exploration
- Emits finding events for important discoveries

Author: Aitherium
"""

import asyncio
import fnmatch
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .forge import SubAgentResult, SubAgentState

logger = logging.getLogger("AitherScout")

# ===============================================================================
# EVENT EMISSION
# ===============================================================================

# Try to import AitherEvents for Pulse integration
_EVENTS_AVAILABLE = False
try:
    _lib_path = Path(__file__).parent.parent.parent.parent / "AitherNode" / "lib"
    if str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))
    
    from AitherEvents import emit_scout_progress, emit_scout_finding
    _EVENTS_AVAILABLE = True
except ImportError:
    # No-op fallbacks
    async def emit_scout_progress(*args, **kwargs): return False
    async def emit_scout_finding(*args, **kwargs): return False


@dataclass
class ScoutFinding:
    """
    A single finding from a scout's exploration.
    
    Every finding is grounded in actual code with:
    - File path (absolute and relative)
    - Line numbers where the pattern was found
    - The actual code snippet
    - Pattern that matched (if any)
    """
    file_path: str
    relative_path: str
    line_start: int
    line_end: int
    snippet: str
    pattern_matched: Optional[str] = None
    context_before: str = ""
    context_after: str = ""
    file_type: str = ""
    importance: str = "normal"  # low, normal, high, critical
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "snippet": self.snippet,
            "pattern_matched": self.pattern_matched,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "file_type": self.file_type,
            "importance": self.importance,
            "tags": self.tags,
        }


@dataclass
class ScoutTask:
    """
    Task definition for a scout.
    
    Defines what the scout should search for and where.
    """
    objective: str
    search_paths: List[str] = field(default_factory=lambda: ["."])
    patterns: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.ps1", "*.ts", "*.md"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["node_modules", ".venv", "__pycache__", ".git"])
    max_depth: int = 10
    max_files: int = 500
    max_findings_per_file: int = 20
    context_lines: int = 3
    workspace_root: str = "."
    
    # Advanced options
    case_sensitive: bool = False
    whole_word: bool = False
    include_binary: bool = False


class AitherScout:
    """
    A scout sub-agent that explores codebases.
    
    Scouts are ephemeral - they're spawned for a specific task,
    explore the codebase, and return findings.
    
    They do NOT:
    - Make assumptions about code intent
    - Generate solutions or suggestions
    - Modify any files
    
    They DO:
    - Find files matching patterns
    - Search for text/regex in files
    - Report exact locations and snippets
    - Build file structure maps
    """
    
    def __init__(self, task: ScoutTask, agent_id: str):
        self.task = task
        self.agent_id = agent_id
        self.workspace_root = Path(task.workspace_root).resolve()
        self.findings: List[ScoutFinding] = []
        self.files_explored: Set[str] = set()
        self.patterns_matched: Dict[str, int] = {}
        self.errors: List[str] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded."""
        path_str = str(path)
        for pattern in self.task.exclude_patterns:
            if pattern in path_str or fnmatch.fnmatch(path.name, pattern):
                return True
        return False
    
    def _matches_file_pattern(self, path: Path) -> bool:
        """Check if a file matches the file patterns."""
        if not self.task.file_patterns:
            return True
        return any(
            fnmatch.fnmatch(path.name, pattern)
            for pattern in self.task.file_patterns
        )
    
    def _compile_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Compile search patterns to regex."""
        compiled = []
        flags = 0 if self.task.case_sensitive else re.IGNORECASE
        
        for pattern in self.task.patterns:
            try:
                if self.task.whole_word:
                    pattern = rf"\b{pattern}\b"
                compiled.append((pattern, re.compile(pattern, flags)))
            except re.error as e:
                # Try as literal string if regex fails
                escaped = re.escape(pattern)
                if self.task.whole_word:
                    escaped = rf"\b{escaped}\b"
                compiled.append((pattern, re.compile(escaped, flags)))
                logger.debug(f"Pattern '{pattern}' compiled as literal: {e}")
        
        return compiled
    
    def _get_file_type(self, path: Path) -> str:
        """Determine file type from extension."""
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
            ".bash": "shell",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
        }
        return ext_map.get(path.suffix.lower(), "text")
    
    async def _search_file(
        self,
        file_path: Path,
        patterns: List[Tuple[str, re.Pattern]]
    ) -> List[ScoutFinding]:
        """Search a single file for patterns."""
        findings = []
        
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            
            self.files_explored.add(str(file_path))
            
            # If no patterns, just note the file exists
            if not patterns:
                return findings
            
            # Search for each pattern
            for original_pattern, regex in patterns:
                finding_count = 0
                
                for line_num, line in enumerate(lines, 1):
                    if finding_count >= self.task.max_findings_per_file:
                        break
                    
                    matches = list(regex.finditer(line))
                    if matches:
                        # Get context lines
                        start_line = max(0, line_num - 1 - self.task.context_lines)
                        end_line = min(len(lines), line_num + self.task.context_lines)
                        
                        context_before = "\n".join(lines[start_line:line_num - 1]) if line_num > 1 else ""
                        context_after = "\n".join(lines[line_num:end_line]) if line_num < len(lines) else ""
                        
                        finding = ScoutFinding(
                            file_path=str(file_path),
                            relative_path=str(file_path.relative_to(self.workspace_root)),
                            line_start=line_num,
                            line_end=line_num,
                            snippet=line,
                            pattern_matched=original_pattern,
                            context_before=context_before,
                            context_after=context_after,
                            file_type=self._get_file_type(file_path),
                        )
                        findings.append(finding)
                        finding_count += 1
                        
                        # Track pattern matches
                        self.patterns_matched[original_pattern] = \
                            self.patterns_matched.get(original_pattern, 0) + 1
        
        except UnicodeDecodeError:
            if not self.task.include_binary:
                pass  # Skip binary files silently
            else:
                self.errors.append(f"Binary file skipped: {file_path}")
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
        
        return findings
    
    async def _walk_directory(
        self,
        directory: Path,
        patterns: List[Tuple[str, re.Pattern]],
        depth: int = 0
    ) -> List[ScoutFinding]:
        """Recursively walk a directory and search files."""
        findings = []
        
        if depth > self.task.max_depth:
            return findings
        
        if not directory.exists() or not directory.is_dir():
            self.errors.append(f"Directory not found: {directory}")
            return findings
        
        try:
            for entry in directory.iterdir():
                if len(self.files_explored) >= self.task.max_files:
                    break
                
                if self._should_exclude(entry):
                    continue
                
                if entry.is_file() and self._matches_file_pattern(entry):
                    file_findings = await self._search_file(entry, patterns)
                    findings.extend(file_findings)
                    
                    # Emit progress every 50 files and emit finding events for important discoveries
                    if len(self.files_explored) % 50 == 0:
                        progress = min(len(self.files_explored) / self.task.max_files, 0.9)
                        await emit_scout_progress(
                            scout_id=self.agent_id,
                            phase="exploring",
                            message=f"Searched {len(self.files_explored)} files, {len(self.findings)} findings",
                            progress=progress,
                            files_explored=len(self.files_explored),
                            findings_count=len(self.findings),
                        )
                        await asyncio.sleep(0)  # Yield control for async
                    
                    # Emit finding events for high-importance findings
                    for finding in file_findings:
                        if finding.importance in ("high", "critical"):
                            await emit_scout_finding(
                                scout_id=self.agent_id,
                                file_path=finding.relative_path,
                                pattern_matched=finding.pattern_matched or "",
                                line_number=finding.line_start,
                                snippet=finding.snippet,
                                importance=finding.importance,
                            )
                
                elif entry.is_dir():
                    subdir_findings = await self._walk_directory(
                        entry, patterns, depth + 1
                    )
                    findings.extend(subdir_findings)
        
        except PermissionError:
            self.errors.append(f"Permission denied: {directory}")
        except Exception as e:
            self.errors.append(f"Error walking {directory}: {e}")
        
        return findings
    
    def _generate_summary(self) -> str:
        """Generate a summary of exploration results."""
        summary_parts = [
            f"Objective: {self.task.objective}",
            f"Files explored: {len(self.files_explored)}",
            f"Findings: {len(self.findings)}",
        ]
        
        if self.patterns_matched:
            summary_parts.append("Pattern matches:")
            for pattern, count in sorted(
                self.patterns_matched.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                summary_parts.append(f"  - '{pattern}': {count} matches")
        
        if self.errors:
            summary_parts.append(f"Errors encountered: {len(self.errors)}")
        
        return "\n".join(summary_parts)
    
    async def explore(self) -> SubAgentResult:
        """
        Execute the exploration task.
        
        Returns:
            SubAgentResult with all findings
        """
        self.started_at = datetime.now()
        
        try:
            # Compile patterns
            patterns = self._compile_patterns()
            
            # Walk each search path
            for search_path in self.task.search_paths:
                if len(self.files_explored) >= self.task.max_files:
                    break
                
                full_path = self.workspace_root / search_path
                path_findings = await self._walk_directory(full_path, patterns)
                self.findings.extend(path_findings)
            
            # Emit analyzing phase
            await emit_scout_progress(
                scout_id=self.agent_id,
                phase="analyzing",
                message=f"Analyzing {len(self.findings)} findings from {len(self.files_explored)} files",
                progress=0.95,
                files_explored=len(self.files_explored),
                findings_count=len(self.findings),
            )
            
            self.completed_at = datetime.now()
            
            return SubAgentResult(
                agent_id=self.agent_id,
                agent_type="scout",
                objective=self.task.objective,
                state=SubAgentState.COMPLETED,
                findings=[f.to_dict() for f in self.findings],
                files_explored=list(self.files_explored),
                patterns_matched=self.patterns_matched.copy(),
                summary=self._generate_summary(),
                started_at=self.started_at,
                completed_at=self.completed_at,
                metadata={
                    "search_paths": self.task.search_paths,
                    "patterns": self.task.patterns,
                    "file_patterns": self.task.file_patterns,
                    "errors": self.errors,
                }
            )
        
        except Exception as e:
            self.completed_at = datetime.now()
            logger.exception(f"Scout exploration failed: {e}")
            
            return SubAgentResult(
                agent_id=self.agent_id,
                agent_type="scout",
                objective=self.task.objective,
                state=SubAgentState.FAILED,
                error=str(e),
                files_explored=list(self.files_explored),
                started_at=self.started_at,
                completed_at=self.completed_at,
            )


class CodebaseMapper:
    """
    Utility to build a map of a codebase structure.
    
    Used by scouts to understand project layout.
    """
    
    def __init__(self, root: Path):
        self.root = root.resolve()
    
    async def get_structure(
        self,
        max_depth: int = 3,
        include_files: bool = True,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build a tree structure of the codebase.
        
        Returns a nested dict representing the file structure.
        """
        exclude = exclude_patterns or ["node_modules", ".venv", "__pycache__", ".git"]
        
        def should_exclude(path: Path) -> bool:
            return any(p in str(path) for p in exclude)
        
        def build_tree(path: Path, depth: int) -> Dict[str, Any]:
            if depth > max_depth or should_exclude(path):
                return {}
            
            result = {
                "name": path.name,
                "type": "directory" if path.is_dir() else "file",
            }
            
            if path.is_dir():
                children = []
                try:
                    for child in sorted(path.iterdir()):
                        if not should_exclude(child):
                            if child.is_dir():
                                children.append(build_tree(child, depth + 1))
                            elif include_files:
                                children.append({
                                    "name": child.name,
                                    "type": "file",
                                    "size": child.stat().st_size,
                                })
                except PermissionError:
                    pass
                result["children"] = children
            
            return result
        
        return build_tree(self.root, 0)
    
    async def find_entry_points(self) -> List[Dict[str, str]]:
        """Find likely entry points in the codebase."""
        entry_patterns = [
            # Python
            ("main.py", "python"),
            ("app.py", "python"),
            ("__main__.py", "python"),
            ("server.py", "python"),
            ("agent.py", "python"),
            
            # JavaScript/TypeScript
            ("index.ts", "typescript"),
            ("index.js", "javascript"),
            ("main.ts", "typescript"),
            ("main.js", "javascript"),
            ("server.ts", "typescript"),
            ("server.js", "javascript"),
            
            # PowerShell
            ("Start-*.ps1", "powershell"),
            ("bootstrap.ps1", "powershell"),
            
            # Config
            ("package.json", "node"),
            ("pyproject.toml", "python"),
            ("Cargo.toml", "rust"),
        ]
        
        found = []
        for pattern, lang in entry_patterns:
            for match in self.root.rglob(pattern):
                if not any(x in str(match) for x in [".venv", "node_modules", ".git"]):
                    found.append({
                        "path": str(match),
                        "relative": str(match.relative_to(self.root)),
                        "language": lang,
                    })
        
        return found
