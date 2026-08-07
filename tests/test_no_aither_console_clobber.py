"""Test guard: aither-adk must NEVER register `aither` console_script.

The `aither` command is owned by the npm @aitheros/shell-cli (TypeScript REPL).
The Python aithershell package uses `aither-py`. The aither-adk SDK uses `adk`.

This test fails if a future PR re-adds `aither = ...` under [project.scripts].
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_aither_console_script_not_registered() -> None:
    """Reserve the `aither` binary for the npm shell-cli REPL.

    See aither-adk/pyproject.toml [project.scripts] for the rule.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    forbidden = {"aither", "aithershell"}
    collisions = forbidden & set(scripts.keys())
    assert not collisions, (
        f"aither-adk MUST NOT register {sorted(collisions)} as console_scripts. "
        f"The `aither` command belongs to the npm @aitheros/shell-cli REPL. "
        f"Use `adk`, `adk-bug`, `adk-serve` for SDK CLIs. "
        f"Found in [project.scripts]: {dict(scripts)}"
    )


def test_only_adk_prefixed_scripts() -> None:
    """All aither-adk console_scripts must be `adk` or `adk-*`."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data.get("project", {}).get("scripts", {})
    bad = [name for name in scripts if not (name == "adk" or name.startswith("adk-"))]
    assert not bad, (
        f"aither-adk console_scripts must be `adk` or `adk-*` prefix only. "
        f"Found non-conforming entries: {bad}. "
        f"All scripts: {list(scripts.keys())}"
    )
