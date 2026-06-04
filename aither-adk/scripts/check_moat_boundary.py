#!/usr/bin/env python3
"""Publish-time moat guard — inspect a built wheel before it ships to PyPI.

Fails (non-zero exit) if the wheel leaks the internal moat:
  * imports the proprietary ``aither_adk`` package
  * bundles ``adk/nanogpt.py`` (on-device training IP)
  * bundles any premium identity YAML (only ``aither.yaml`` is free)
  * is missing the ``adk/licensing.py`` entitlement keystone

Usage:
    python scripts/check_moat_boundary.py [dist/aither_adk-*.whl]

With no argument it picks the newest wheel in ``dist/``.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

_AITHER_ADK_IMPORT = re.compile(rb"^\s*(from|import)\s+aither_adk(\.|\s|$)", re.MULTILINE)


def _newest_wheel() -> Path | None:
    dist = Path(__file__).resolve().parent.parent / "dist"
    wheels = sorted(dist.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def main(argv: list[str]) -> int:
    wheel = Path(argv[1]) if len(argv) > 1 else _newest_wheel()
    if not wheel or not wheel.is_file():
        print("MOAT GUARD: no wheel found (build first: python -m build)", file=sys.stderr)
        return 2

    print(f"MOAT GUARD: inspecting {wheel.name}")
    violations: list[str] = []
    saw_licensing = False

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

        for name in names:
            # nanogpt training IP must never ship
            if name.endswith("adk/nanogpt.py") or name.endswith("adk\\nanogpt.py"):
                violations.append(f"LEAK: training IP shipped: {name}")

            # only the free identity may ship
            if "/identities/" in name and name.endswith(".yaml"):
                if not name.endswith("aither.yaml"):
                    violations.append(f"LEAK: premium identity shipped: {name}")

            # the internal moat namespace must never appear
            if name.startswith("aither_adk/") or "/aither_adk/" in name:
                violations.append(f"LEAK: internal moat package shipped: {name}")

            if name.endswith("adk/licensing.py"):
                saw_licensing = True

            # scan python sources for moat imports
            if name.endswith(".py"):
                data = zf.read(name)
                if _AITHER_ADK_IMPORT.search(data):
                    violations.append(f"LEAK: imports internal moat: {name}")

    if not saw_licensing:
        violations.append("MISSING: adk/licensing.py (entitlement keystone) not in wheel")

    if violations:
        print("MOAT GUARD FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("MOAT GUARD PASSED: no leaks, keystone present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
