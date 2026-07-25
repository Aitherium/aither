#!/usr/bin/env python3
"""Sync version from pyproject.toml into all package manifests.

Reads the canonical version from aither-adk/pyproject.toml and updates:
  - packaging/npm/package.json
  - packaging/brew/aither-adk.rb
  - packaging/winget/Aitherium.ADK.yaml

Usage:
    python packaging/sync_versions.py          # Sync versions
    python packaging/sync_versions.py --check  # Check without modifying
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def get_version() -> str:
    """Read version from pyproject.toml."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)


def sync_npm(version: str, check: bool) -> bool:
    """Update packaging/npm/package.json."""
    path = Path(__file__).parent / "npm" / "package.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") == version:
        print(f"  npm: already {version}")
        return True
    if check:
        print(f"  npm: {data.get('version')} -> {version} (needs update)")
        return False
    data["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  npm: updated to {version}")
    return True


def sync_brew(version: str, check: bool) -> bool:
    """Update packaging/brew/aither-adk.rb."""
    path = Path(__file__).parent / "brew" / "aither-adk.rb"
    text = path.read_text(encoding="utf-8")

    old_url = re.search(r'url "https://files\.pythonhosted\.org/.*?aither_adk-([^"]+)\.tar\.gz"', text)
    old_ver = old_url.group(1) if old_url else None

    if old_ver == version:
        print(f"  brew: already {version}")
        return True
    if check:
        print(f"  brew: {old_ver} -> {version} (needs update)")
        return False

    new_url = f'https://files.pythonhosted.org/packages/source/a/aither-adk/aither_adk-{version}.tar.gz'
    # count=1: ONLY the formula's own `url` may be rewritten. Without it this
    # repointed every `resource "<dep>"` block at the aither-adk tarball too, so
    # `brew install` fetched the adk sdist and called it httpx — silently
    # corrupting the formula on every single release.
    text = re.sub(
        r'url "https://files\.pythonhosted\.org/[^"]*"',
        f'url "{new_url}"',
        text,
        count=1,
    )
    text = re.sub(
        r'sha256 "[^"]*"',
        'sha256 "PLACEHOLDER_SHA256"',
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"  brew: updated to {version}")
    print(f"  brew: SHA256 set to PLACEHOLDER — update after PyPI publish")
    return True


def sync_winget(version: str, check: bool) -> bool:
    """Update packaging/winget/Aitherium.ADK.yaml."""
    path = Path(__file__).parent / "winget" / "Aitherium.ADK.yaml"
    text = path.read_text(encoding="utf-8")

    old_match = re.search(r'^PackageVersion:\s*(.+)$', text, re.MULTILINE)
    old_ver = old_match.group(1).strip() if old_match else None

    if old_ver == version:
        print(f"  winget: already {version}")
        return True
    if check:
        print(f"  winget: {old_ver} -> {version} (needs update)")
        return False

    text = re.sub(
        r'^PackageVersion:.*$',
        f'PackageVersion: {version}',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'InstallerUrl:.*$',
        f'InstallerUrl: https://aitherium.com/download/aither-adk-{version}-win64.exe',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'InstallerSha256:.*$',
        'InstallerSha256: PLACEHOLDER_SHA256',
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8")
    print(f"  winget: updated to {version}")
    print(f"  winget: SHA256 set to PLACEHOLDER — update after build")
    return True


def sync_init(version: str, check: bool) -> bool:
    """Sync the source-checkout fallback __version__ in adk/__init__.py.

    The installed package reads pyproject metadata, but a source checkout (and
    the public repo's check_exports.py CI gate) sees the literal fallback —
    leaving it stale broke public CI on v2.27.0 (fallback said 2.24.0)."""
    path = Path(__file__).parent.parent / "adk" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    pattern = r'(__version__ = ")([^"]+)(")'
    m = re.search(pattern, text)
    if not m:
        print("  init: no fallback __version__ found (skipped)")
        return True
    if m.group(2) == version:
        print(f"  init: already {version}")
        return True
    if check:
        print(f"  init: {m.group(2)} -> {version} (needs update)")
        return False
    path.write_text(re.sub(pattern, rf"\g<1>{version}\g<3>", text, count=1), encoding="utf-8")
    print(f"  init: updated to {version}")
    return True


def main():
    check = "--check" in sys.argv
    version = get_version()
    print(f"Canonical version: {version}")
    print()

    all_ok = True
    all_ok &= sync_init(version, check)
    all_ok &= sync_npm(version, check)
    all_ok &= sync_brew(version, check)
    all_ok &= sync_winget(version, check)

    if check and not all_ok:
        print("\nVersion mismatch detected. Run without --check to fix.")
        sys.exit(1)
    elif not check:
        print("\nAll manifests synced.")


if __name__ == "__main__":
    main()
