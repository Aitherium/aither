#!/usr/bin/env python3
"""Verify adk package exports resolve and version is synced.

Catches:
  1. __all__ entries that reference non-existent modules (ghost exports)
  2. __version__ mismatch between __init__.py and pyproject.toml
  3. Top-level .py modules with zero imports from anywhere (dead code)

Run: python scripts/check_exports.py
Exit code 1 on any failure.
"""

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADK = ROOT / "adk"
PYPROJECT = ROOT / "pyproject.toml"

errors: list[str] = []


# ── 1. Verify every __all__ entry resolves via __getattr__ ────────────────

def check_ghost_exports():
    init = ADK / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))

    # Collect __all__ names
    all_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_names.append(elt.value)

    # Collect lazy import targets from __getattr__
    lazy_names: set[str] = set()
    eager_names: set[str] = set()

    for node in ast.walk(tree):
        # Eager imports at module level (from adk.X import Y)
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0:
            for alias in node.names:
                eager_names.add(alias.asname or alias.name)

        # __getattr__ function: find all `if name == "X"` comparisons
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            for child in ast.walk(node):
                if isinstance(child, ast.Compare):
                    for comp in child.comparators:
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                            lazy_names.add(comp.value)

    resolvable = eager_names | lazy_names
    for name in all_names:
        if name not in resolvable:
            errors.append(f"ghost export: '{name}' is in __all__ but has no import or __getattr__ entry")


# ── 2. Verify __version__ matches pyproject.toml ─────────────────────────

def check_version_sync():
    init = ADK / "__init__.py"
    init_version = None
    for line in init.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            init_version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

    with open(PYPROJECT, "rb") as f:
        pyproject_version = tomllib.load(f)["project"]["version"]

    if init_version != pyproject_version:
        errors.append(
            f"version mismatch: __init__.py={init_version} vs pyproject.toml={pyproject_version}"
        )


# ── 3. Detect orphan modules (zero imports from any other file) ──────────

def check_orphan_modules():
    # Get all top-level .py files in adk/ (excluding __init__.py)
    module_files = {
        f.stem: f for f in ADK.glob("*.py")
        if f.name != "__init__.py"
    }

    # Scan all .py files in the package for imports
    all_py = list(ADK.rglob("*.py"))
    # Also check tests, examples, packaging, root scripts, packages, and root .py files
    for subdir in ["tests", "examples", "packaging", "scripts", "packages"]:
        p = ROOT / subdir
        if p.exists():
            all_py.extend(p.rglob("*.py"))
    all_py.extend(ROOT.glob("*.py"))

    imported: set[str] = set()
    for py_file in all_py:
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # from adk.foo import ... or from .foo import ...
                parts = node.module.split(".")
                if len(parts) >= 2 and parts[0] == "adk":
                    imported.add(parts[1])
                elif parts == ["adk"]:
                    # from adk import foo — the imported name is the module
                    for alias in node.names:
                        imported.add(alias.name)
                elif node.level and len(parts) >= 1:
                    imported.add(parts[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) >= 2 and parts[0] == "adk":
                        imported.add(parts[1])

    # Also count __init__.py lazy imports as "used"
    init_source = (ADK / "__init__.py").read_text(encoding="utf-8")
    init_tree = ast.parse(init_source)
    for node in ast.walk(init_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[0] == "adk":
                imported.add(parts[1])

    # Entry points from pyproject.toml count as used
    with open(PYPROJECT, "rb") as f:
        scripts = tomllib.load(f).get("project", {}).get("scripts", {})
    for entry in scripts.values():
        mod = entry.split(":")[0]
        parts = mod.split(".")
        if len(parts) >= 2 and parts[0] == "adk":
            imported.add(parts[1])

    for name, path in sorted(module_files.items()):
        if name not in imported:
            errors.append(f"orphan module: adk/{name}.py is never imported")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_ghost_exports()
    check_version_sync()
    check_orphan_modules()

    if errors:
        print(f"FAIL: {len(errors)} issue(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("OK: all exports resolve, version synced, no orphan modules")
        sys.exit(0)
