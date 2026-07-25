#!/usr/bin/env python3
"""Sync version from pyproject.toml into all package manifests.

Reads the canonical version from aither-adk/pyproject.toml and updates:
  - packaging/npm/package.json
  - packaging/brew/aither-adk.rb
  - packaging/winget/Aitherium.ADK.yaml

Usage:
    python packaging/sync_versions.py           # Sync versions
    python packaging/sync_versions.py --check   # Check without modifying
    python packaging/sync_versions.py --digests # Fill brew sha256 from PyPI

Version bumps necessarily precede the PyPI publish, so the brew formula's own
sha256 is written as a PLACEHOLDER at bump time and must be filled once the
sdist exists. Nobody ever did that, so every release shipped a formula that
`brew install` cannot verify. `--digests` fills it from PyPI, and `--check`
FAILS when the version is published but the digest is still a placeholder — so
the omission is now loud instead of silent.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PLACEHOLDER = "PLACEHOLDER_SHA256"


def fetch_sdist_sha256(version: str, *, timeout: float = 15.0) -> str | None:
    """Return the PyPI sdist sha256 for *version*, or None if not published.

    Network failures return None rather than raising: a transient outage must
    not fail a release, it just leaves the digest unfilled (which --check then
    reports).
    """
    url = f"https://pypi.org/pypi/aither-adk/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    except json.JSONDecodeError:
        return None
    for entry in data.get("urls", []):
        if entry.get("packagetype") == "sdist":
            return entry.get("digests", {}).get("sha256")
    return None


def sync_brew_digest(version: str, check: bool, fetch=fetch_sdist_sha256) -> bool:
    """Fill (or verify) the brew formula's own sha256 against PyPI.

    Only the FIRST sha256 in the formula is the package's own; the rest belong
    to `resource` blocks and must never be touched (see sync_brew).
    """
    path = Path(__file__).parent / "brew" / "aither-adk.rb"
    text = path.read_text(encoding="utf-8")
    match = re.search(r'sha256 "([^"]*)"', text)
    if not match:
        print("  brew digest: no sha256 line (skipped)")
        return True

    current = match.group(1)
    published = fetch(version)

    if published is None:
        if current.startswith("PLACEHOLDER"):
            print(f"  brew digest: {version} not on PyPI yet — placeholder retained")
        else:
            print("  brew digest: not on PyPI yet — leaving existing digest")
        return True

    if current == published:
        print("  brew digest: already correct")
        return True
    if check:
        print(f"  brew digest: {current[:16]}… -> {published[:16]}… (needs update)")
        return False

    text = text.replace(f'sha256 "{current}"', f'sha256 "{published}"', 1)
    path.write_text(text, encoding="utf-8")
    print(f"  brew digest: filled from PyPI ({published[:16]}…)")
    return True


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

    digests_only = "--digests" in sys.argv

    all_ok = True
    if not digests_only:
        all_ok &= sync_init(version, check)
        all_ok &= sync_npm(version, check)
        all_ok &= sync_brew(version, check)
        all_ok &= sync_winget(version, check)

    # Runs in --check too: a PUBLISHED version whose formula still carries a
    # placeholder digest is a broken `brew install`, and used to pass silently.
    all_ok &= sync_brew_digest(version, check)

    if check and not all_ok:
        print("\nManifest mismatch detected. Run without --check to fix.")
        sys.exit(1)
    elif not check:
        print("\nAll manifests synced.")


if __name__ == "__main__":
    main()
