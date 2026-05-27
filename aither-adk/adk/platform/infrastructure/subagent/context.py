"""
ContextFork - Context Sharing Between Parent and Sub-Agents
============================================================

When spawning sub-agents, the parent needs to share relevant context
without overwhelming the child with irrelevant information.

ContextFork provides:
1. Selective context extraction
2. Context serialization for IPC
3. Result aggregation from multiple sub-agents
4. Context merging for follow-up queries

Author: Aitherium
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger("ContextFork")


@dataclass
class SharedContext:
    """
    Context that can be shared between parent and sub-agents.
    
    This is a lightweight, serializable representation of
    the information a sub-agent needs to do its job.
    """
    # Core identifiers
    session_id: str
    parent_id: Optional[str] = None
    
    # Objective and scope
    objective: str = ""
    scope: str = ""  # "file", "directory", "project", "workspace"
    
    # Files/paths of interest
    target_paths: List[str] = field(default_factory=list)
    exclude_paths: List[str] = field(default_factory=list)
    
    # Search context
    search_patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Code context
    current_file: Optional[str] = None
    current_function: Optional[str] = None
    current_class: Optional[str] = None
    
    # Project context
    project_type: Optional[str] = None  # "python", "powershell", "node", etc.
    project_root: Optional[str] = None
    
    # Constraints
    max_files: int = 100
    max_depth: int = 5
    timeout_seconds: float = 30.0
    
    # Previous findings (for follow-up)
    prior_findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedContext":
        """Create from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "SharedContext":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def hash(self) -> str:
        """Generate a hash for caching/deduplication."""
        key_data = f"{self.objective}:{','.join(sorted(self.target_paths))}:{','.join(sorted(self.search_patterns))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]


class ContextFork:
    """
    Manages context forking for sub-agent spawning.
    
    Usage:
        fork = ContextFork(parent_context)
        
        # Create focused context for a specific task
        scout_context = fork.for_search(
            patterns=["auth", "login"],
            paths=["src/auth", "lib/security"]
        )
        
        # Create context for file exploration
        explorer_context = fork.for_exploration(
            target_dir="src/models",
            file_types=["*.py"]
        )
    """
    
    def __init__(
        self,
        session_id: str,
        project_root: Optional[str] = None,
        parent_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.project_root = project_root or str(Path.cwd())
        self.parent_id = parent_id
        self._forks: List[SharedContext] = []
    
    def for_search(
        self,
        objective: str,
        patterns: List[str],
        paths: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        **kwargs
    ) -> SharedContext:
        """
        Create context for pattern search.
        
        Args:
            objective: What the search is trying to find
            patterns: Regex/text patterns to search
            paths: Directories to search (relative to project root)
            keywords: Additional keywords for context
        """
        ctx = SharedContext(
            session_id=self.session_id,
            parent_id=self.parent_id,
            objective=objective,
            scope="directory" if paths else "project",
            target_paths=paths or ["."],
            search_patterns=patterns,
            keywords=keywords or [],
            project_root=self.project_root,
            **kwargs
        )
        self._forks.append(ctx)
        return ctx
    
    def for_exploration(
        self,
        objective: str,
        target_dir: str,
        file_types: Optional[List[str]] = None,
        max_depth: int = 3,
        **kwargs
    ) -> SharedContext:
        """
        Create context for directory exploration.
        
        Args:
            objective: What to discover
            target_dir: Directory to explore
            file_types: File extensions to include
            max_depth: How deep to traverse
        """
        ctx = SharedContext(
            session_id=self.session_id,
            parent_id=self.parent_id,
            objective=objective,
            scope="directory",
            target_paths=[target_dir],
            search_patterns=file_types or ["*.py", "*.ps1", "*.ts", "*.md"],
            max_depth=max_depth,
            project_root=self.project_root,
            **kwargs
        )
        self._forks.append(ctx)
        return ctx
    
    def for_file_analysis(
        self,
        objective: str,
        file_path: str,
        focus_function: Optional[str] = None,
        focus_class: Optional[str] = None,
        **kwargs
    ) -> SharedContext:
        """
        Create context for single file analysis.
        
        Args:
            objective: What to understand about the file
            file_path: Path to the file
            focus_function: Specific function to analyze
            focus_class: Specific class to analyze
        """
        ctx = SharedContext(
            session_id=self.session_id,
            parent_id=self.parent_id,
            objective=objective,
            scope="file",
            target_paths=[file_path],
            current_file=file_path,
            current_function=focus_function,
            current_class=focus_class,
            project_root=self.project_root,
            **kwargs
        )
        self._forks.append(ctx)
        return ctx
    
    def for_dependency_trace(
        self,
        objective: str,
        starting_point: str,
        trace_direction: str = "both",  # "imports", "usages", "both"
        **kwargs
    ) -> SharedContext:
        """
        Create context for dependency/import tracing.
        
        Args:
            objective: What dependencies to trace
            starting_point: File or symbol to start from
            trace_direction: Which direction to trace
        """
        ctx = SharedContext(
            session_id=self.session_id,
            parent_id=self.parent_id,
            objective=objective,
            scope="project",
            current_file=starting_point,
            project_root=self.project_root,
            metadata={
                "trace_type": "dependency",
                "trace_direction": trace_direction,
            },
            **kwargs
        )
        self._forks.append(ctx)
        return ctx
    
    def with_prior_findings(
        self,
        base_context: SharedContext,
        findings: List[Dict[str, Any]],
        new_objective: Optional[str] = None,
    ) -> SharedContext:
        """
        Create follow-up context that includes previous findings.
        
        Useful for iterative exploration where results inform
        the next round of investigation.
        """
        return SharedContext(
            session_id=self.session_id,
            parent_id=self.parent_id,
            objective=new_objective or base_context.objective,
            scope=base_context.scope,
            target_paths=base_context.target_paths,
            search_patterns=base_context.search_patterns,
            keywords=base_context.keywords,
            project_root=base_context.project_root,
            prior_findings=findings,
        )
    
    def get_all_forks(self) -> List[SharedContext]:
        """Get all contexts that have been forked."""
        return self._forks.copy()


class ResultAggregator:
    """
    Aggregates results from multiple sub-agents.
    
    Provides deduplication, ranking, and summarization.
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self._seen_files: Set[str] = set()
        self._seen_snippets: Set[str] = set()
    
    def add_result(self, result: Dict[str, Any]):
        """Add a sub-agent result to the aggregation."""
        self.results.append(result)
    
    def get_unique_findings(self) -> List[Dict[str, Any]]:
        """Get deduplicated findings across all results."""
        unique = []
        
        for result in self.results:
            for finding in result.get("findings", []):
                # Create a key for deduplication
                file_path = finding.get("file_path", "")
                line_start = finding.get("line_start", 0)
                snippet = finding.get("snippet", "")
                
                key = f"{file_path}:{line_start}:{hashlib.md5(snippet.encode()).hexdigest()[:8]}"
                
                if key not in self._seen_snippets:
                    self._seen_snippets.add(key)
                    unique.append(finding)
        
        return unique
    
    def get_files_explored(self) -> List[str]:
        """Get unique files explored across all results."""
        files = set()
        for result in self.results:
            files.update(result.get("files_explored", []))
        return sorted(files)
    
    def get_pattern_stats(self) -> Dict[str, int]:
        """Aggregate pattern match counts."""
        stats = {}
        for result in self.results:
            for pattern, count in result.get("patterns_matched", {}).items():
                stats[pattern] = stats.get(pattern, 0) + count
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    def rank_findings(
        self,
        by: str = "importance"  # "importance", "frequency", "location"
    ) -> List[Dict[str, Any]]:
        """Rank findings by specified criteria."""
        findings = self.get_unique_findings()
        
        if by == "importance":
            importance_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
            findings.sort(key=lambda f: importance_order.get(f.get("importance", "normal"), 2))
        elif by == "frequency":
            # More pattern matches = higher ranking
            pattern_counts = self.get_pattern_stats()
            findings.sort(
                key=lambda f: pattern_counts.get(f.get("pattern_matched", ""), 0),
                reverse=True
            )
        elif by == "location":
            # Sort by file path then line number
            findings.sort(key=lambda f: (f.get("file_path", ""), f.get("line_start", 0)))
        
        return findings
    
    def summarize(self) -> Dict[str, Any]:
        """Generate a summary of all aggregated results."""
        findings = self.get_unique_findings()
        files = self.get_files_explored()
        patterns = self.get_pattern_stats()
        
        # Group findings by file type
        by_type = {}
        for f in findings:
            file_type = f.get("file_type", "unknown")
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append(f)
        
        return {
            "total_findings": len(findings),
            "total_files_explored": len(files),
            "total_sub_agents": len(self.results),
            "pattern_stats": patterns,
            "findings_by_type": {k: len(v) for k, v in by_type.items()},
            "top_files": self._get_top_files(findings, 10),
        }
    
    def _get_top_files(
        self,
        findings: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get files with most findings."""
        file_counts = {}
        for f in findings:
            path = f.get("relative_path", f.get("file_path", "unknown"))
            file_counts[path] = file_counts.get(path, 0) + 1
        
        sorted_files = sorted(
            file_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [{"file": f, "count": c} for f, c in sorted_files]
