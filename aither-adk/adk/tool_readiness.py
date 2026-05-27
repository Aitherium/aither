"""ADK-side tool readiness adapter.

Checks whether MCP tools can function in the current deployment context
without requiring full ``lib.core`` imports.  Used by AitherAgent to
filter out unavailable tools at registration time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("adk.tool_readiness")


@dataclass
class ADKReadinessReport:
    broken: bool = False
    reason: str = ""


def _is_standalone() -> bool:
    return os.environ.get(
        "AITHER_STANDALONE", ""
    ).lower() in ("1", "true", "yes")


def _get_deployment_mode() -> str:
    mode = os.environ.get("AITHER_DEPLOYMENT_MODE", "").lower()
    if mode in ("cloud", "saas", "local", "spoke", "partner"):
        return "platform"
    if mode in ("sovereign", "airgap", "airgapped"):
        return "sovereign"
    if mode == "standalone" or _is_standalone():
        return "standalone"
    return "standalone"  # ADK defaults to standalone


def check_tool_readiness_adk(
    tool_name: str,
    module_name: str = "",
) -> ADKReadinessReport:
    """Check if a tool can work in the current ADK deployment context.

    Returns an ADKReadinessReport with broken=True if the tool cannot
    function.  Uses _tool_deps declarations when available, otherwise
    falls back to a conservative heuristic.
    """
    mode = _get_deployment_mode()

    # Try to use the full _tool_deps registry
    try:
        from apps.AitherNode.tools.mcp._tool_deps import get_tool_deps
        deps = get_tool_deps(tool_name, module_name)
    except ImportError:
        # _tool_deps not available — use heuristic
        return _heuristic_check(tool_name, module_name, mode)

    if mode not in deps.deployment_modes:
        return ADKReadinessReport(
            broken=True,
            reason=(
                f"Tool '{tool_name}' not available in {mode} mode "
                f"(requires: {', '.join(deps.deployment_modes)})"
            ),
        )

    if mode == "standalone" and not deps.standalone_safe:
        return ADKReadinessReport(
            broken=True,
            reason=(
                f"Tool '{tool_name}' requires platform services "
                f"({', '.join(deps.services)})"
            ),
        )

    return ADKReadinessReport()


# ── Heuristic fallback ─────────────────────────────────────────────────
# Used when _tool_deps is not importable (e.g. vendored ADK without
# the full AitherNode tree).

_KNOWN_STANDALONE = frozenset({
    "fs_read_file", "fs_write_file", "fs_list_directory", "fs_delete_file",
    "fs_create_directory", "fs_copy_file", "fs_move_file", "fs_file_exists",
    "git_status", "git_diff", "git_log", "git_add", "git_commit",
    "git_branch", "git_checkout",
    "codegraph_search", "codegraph_get_context", "explore_code",
})

_KNOWN_STANDALONE_MODULES = frozenset({
    "mcp_filesystem", "mcp_git", "mcp_http_client",
    "mcp_browser_context", "mcp_desktop_context", "mcp_context7",
    "mcp_codegraph_registry", "mcp_codebase_intel", "mcp_exploration",
    "mcp_design_system",
})


def _heuristic_check(
    tool_name: str, module_name: str, mode: str,
) -> ADKReadinessReport:
    """Conservative heuristic when _tool_deps is unavailable."""
    if mode == "platform":
        return ADKReadinessReport()

    if tool_name in _KNOWN_STANDALONE:
        return ADKReadinessReport()

    mod_key = module_name.rsplit(".", 1)[-1] if module_name else ""
    if mod_key in _KNOWN_STANDALONE_MODULES:
        return ADKReadinessReport()

    # Conservative: assume platform-only
    return ADKReadinessReport(
        broken=True,
        reason=f"Tool '{tool_name}' assumed platform-only (no dependency info)",
    )
