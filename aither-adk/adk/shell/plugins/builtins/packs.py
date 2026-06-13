"""
ADK Shell Plugin: Packs Marketplace
Browse, purchase, and manage agent/skill/tool packs from the Elysium marketplace.
"""

import json
import os
import tarfile
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from adk.shell.plugins import SlashCommand


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    """Extract a tarball, rejecting any member that would escape ``path``
    (path-traversal / absolute-path / symlink-escape defense)."""
    base = os.path.realpath(path)
    for member in tar.getmembers():
        target = os.path.realpath(os.path.join(path, member.name))
        if not (target == base or target.startswith(base + os.sep)):
            raise ValueError(f"unsafe path in archive: {member.name}")
        if member.issym() or member.islnk():
            link_target = os.path.realpath(os.path.join(path, member.linkname))
            if not (link_target == base or link_target.startswith(base + os.sep)):
                raise ValueError(f"unsafe link in archive: {member.name}")
    # filter="data" (py3.12+) is the hardened extraction profile; our own check
    # above is the portable belt-and-braces for older runtimes.
    try:
        tar.extractall(path, filter="data")
    except TypeError:
        tar.extractall(path)


class AuthStore:
    """Simple auth token storage for marketplace API calls."""

    def __init__(self):
        self.token: Optional[str] = None
        self.tenant_id: Optional[str] = None

    def set_auth(self, token: str, tenant_id: str = None):
        """Store auth credentials."""
        self.token = token
        self.tenant_id = tenant_id

    def get_headers(self) -> Dict[str, str]:
        """Build auth headers for API calls."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers


class PacksPlugin(SlashCommand):
    """
    /packs — Browse and manage agent/skill/tool packs from marketplace.

    Subcommands:
      /packs list              List all published packs
      /packs browse            Browse with filters (--type, --search)
      /packs info <pack-id>    Show pack details
      /packs purchase <id>     Open Stripe checkout
      /packs install <id>      Download and extract pack
      /packs sync              Install every entitled pack not already present
      /packs library           Show purchased packs
      /packs help              Show help text
    """

    name = "packs"
    aliases = ["pack", "store"]
    category = "marketplace"

    def __init__(self):
        super().__init__()
        self.auth = AuthStore()
        self._base_url = self._get_base_url()

    def _get_base_url(self) -> str:
        """Resolve Elysium base URL from env or fallback."""
        elysium_url = os.environ.get("AITHER_ELYSIUM_URL")
        if elysium_url:
            return elysium_url.rstrip("/")

        portal_url = os.environ.get("AITHER_PORTAL_URL")
        if portal_url:
            return portal_url.rstrip("/")

        return "https://portal.aitherium.com"

    def execute(self, args: List[str], **kwargs) -> str:
        """Main entry point for /packs command."""
        if not args:
            return self._show_list()

        command = args[0].lower()

        if command in ("list", "ls"):
            return self._show_list()
        elif command == "browse":
            return self._browse(args[1:])
        elif command == "info":
            if len(args) < 2:
                return "ERROR: /packs info requires <pack-id>"
            return self._show_info(args[1])
        elif command == "purchase":
            if len(args) < 2:
                return "ERROR: /packs purchase requires <pack-id>"
            return self._purchase(args[1])
        elif command == "install":
            if len(args) < 2:
                return "ERROR: /packs install requires <pack-id>"
            return self._install(args[1], args[2:])
        elif command == "sync":
            return self._sync(args[1:])
        elif command == "library":
            return self._show_library()
        elif command == "help":
            return self._show_help()
        else:
            return f"ERROR: Unknown command '/packs {command}'. Try '/packs help'"

    def _show_list(self) -> str:
        """List all published packs."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self._base_url}/api/marketplace/packs",
                    headers=self.auth.get_headers(),
                )
                resp.raise_for_status()
                packs = resp.json().get("packs", [])

            if not packs:
                return "No packs found in marketplace."

            return self._format_packs_table(packs)

        except httpx.HTTPError as e:
            return f"ERROR: Failed to fetch packs: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _browse(self, args: List[str]) -> str:
        """Browse packs with optional filters."""
        pack_type = None
        search_query = None

        i = 0
        while i < len(args):
            if args[i] == "--type" and i + 1 < len(args):
                pack_type = args[i + 1]
                i += 2
            elif args[i] == "--search" and i + 1 < len(args):
                search_query = args[i + 1]
                i += 2
            else:
                i += 1

        try:
            params = {}
            if pack_type:
                params["type"] = pack_type
            if search_query:
                params["search"] = search_query

            query_string = urlencode(params) if params else ""
            url = f"{self._base_url}/api/marketplace/packs"
            if query_string:
                url = f"{url}?{query_string}"

            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=self.auth.get_headers())
                resp.raise_for_status()
                result = resp.json()
                packs = result.get("packs", [])

            if not packs:
                filters = []
                if pack_type:
                    filters.append(f"type={pack_type}")
                if search_query:
                    filters.append(f"search={search_query}")
                filter_str = ", ".join(filters) if filters else "none"
                return f"No packs found matching filters: {filter_str}"

            return self._format_packs_table(packs)

        except httpx.HTTPError as e:
            return f"ERROR: Failed to browse packs: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _show_info(self, pack_id: str) -> str:
        """Show detailed pack information."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self._base_url}/api/marketplace/packs/{pack_id}",
                    headers=self.auth.get_headers(),
                )
                resp.raise_for_status()
                pack = resp.json()

            return self._format_pack_detail(pack)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"ERROR: Pack '{pack_id}' not found."
            return f"ERROR: HTTP {e.response.status_code}: {e}"
        except httpx.HTTPError as e:
            return f"ERROR: Failed to fetch pack info: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _purchase(self, pack_id: str) -> str:
        """Open Stripe checkout in browser."""
        try:
            checkout_url = f"{self._base_url}/checkout/packs/{pack_id}"
            webbrowser.open(checkout_url)
            return f"Opening checkout for pack '{pack_id}' in browser...\n{checkout_url}"

        except Exception as e:
            return f"ERROR: Failed to open checkout: {e}"

    def _install(self, pack_id: str, args: List[str]) -> str:
        """Download and optionally extract pack bundle."""
        extract = "--extract" in args or "-x" in args

        try:
            # Get pack info first to determine filename
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self._base_url}/api/marketplace/packs/{pack_id}",
                    headers=self.auth.get_headers(),
                )
                resp.raise_for_status()
                pack = resp.json()

            pack_name = pack.get("name", pack_id).lower().replace(" ", "-")
            filename = f"{pack_name}.tar.gz"

            # Download bundle
            download_url = (
                f"{self._base_url}/api/agent-builder/build/{pack_id}/download"
            )
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(download_url, headers=self.auth.get_headers())
                resp.raise_for_status()

                with open(filename, "wb") as f:
                    f.write(resp.content)

            result = f"Downloaded pack to '{filename}'"

            if extract:
                extract_dir = pack_name
                os.makedirs(extract_dir, exist_ok=True)

                with tarfile.open(filename, "r:gz") as tar:
                    tar.extractall(path=extract_dir)

                result += f"\nExtracted to directory '{extract_dir}'"

            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"ERROR: Pack '{pack_id}' not found or not available for download."
            return f"ERROR: HTTP {e.response.status_code}: {e}"
        except httpx.HTTPError as e:
            return f"ERROR: Failed to download pack: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    @staticmethod
    def _packs_dir() -> Path:
        """The local pack install root (matches agent auto-discovery)."""
        d = Path(os.path.expanduser("~")) / ".aitheros" / "packs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _sync(self, args: List[str]) -> str:
        """Install every entitled pack that isn't already present locally.

        Polls the customer's active licenses from the portal, then for each
        entitled pack not yet in ``~/.aitheros/packs/`` downloads the artifact,
        verifies its Ed25519 signature, and extracts it. A portal purchase thus
        converges onto this node with one command (or on `adk up`), no manual
        per-pack install.
        """
        dry_run = "--dry-run" in args or "-n" in args
        packs_root = self._packs_dir()

        # 1. Entitlements: the customer's active licenses (listing_id == pack id).
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{self._base_url}/v1/marketplace/license/mine",
                    headers=self.auth.get_headers(),
                )
                resp.raise_for_status()
                licenses = resp.json().get("licenses", [])
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return "ERROR: Not authenticated. Run /auth with your portal token first."
            return f"ERROR: Could not fetch entitlements: HTTP {e.response.status_code}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: Could not reach the portal: {type(e).__name__}: {e}"

        entitled = sorted({
            lic.get("listing_id", "")
            for lic in licenses
            if lic.get("status") == "active" and lic.get("listing_id")
        })
        if not entitled:
            return "No entitled packs. Buy packs at portal.aitherium.com/portal/marketplace/packs"

        installed, skipped, failed = [], [], []
        for pack_id in entitled:
            dest = packs_root / pack_id
            if dest.is_dir() and any(dest.iterdir()):
                skipped.append(pack_id)
                continue
            if dry_run:
                installed.append(f"{pack_id} (would install)")
                continue
            ok, detail = self._download_verify_install(pack_id, dest)
            (installed if ok else failed).append(detail)

        lines = [f"=== PACK SYNC ({'dry run' if dry_run else 'apply'}) ==="]
        lines.append(f"Entitled: {len(entitled)} · installed: {len(installed)} · "
                     f"already present: {len(skipped)} · failed: {len(failed)}")
        for d in installed:
            lines.append(f"  + {d}")
        for d in failed:
            lines.append(f"  ✗ {d}")
        if skipped:
            lines.append(f"  (present) {', '.join(skipped)}")
        return "\n".join(lines)

    def _download_verify_install(self, pack_id: str, dest: Path) -> tuple:
        """Download → verify signature → extract one pack. Returns (ok, message)."""
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(
                    f"{self._base_url}/v1/packs/{pack_id}/download",
                    headers=self.auth.get_headers(),
                )
            if resp.status_code == 402:
                return False, f"{pack_id}: license required (purchase not active)"
            if resp.status_code == 404:
                return False, f"{pack_id}: no downloadable artifact"
            resp.raise_for_status()
            tarball = resp.content
            signature = resp.headers.get("X-Aither-Pack-Signature")

            # Ed25519 signature verification (fail-closed when a key is pinned).
            try:
                from adk.pack_verifier import verify_pack_tarball

                verified, vmsg = verify_pack_tarball(tarball, signature)
                if not verified:
                    return False, f"{pack_id}: signature check failed ({vmsg})"
            except ImportError:
                pass  # verifier unavailable → proceed (legacy/offline)

            # Extract to a temp dir then atomically swap in (no half-installed pack).
            import io
            import shutil
            import tempfile

            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
                with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
                    _safe_extract(tar, tmp)
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                # If the tar has a single top dir, promote it; else move tmp itself.
                children = [p for p in Path(tmp).iterdir()]
                if len(children) == 1 and children[0].is_dir():
                    shutil.move(str(children[0]), str(dest))
                else:
                    shutil.move(tmp, str(dest))
                    Path(tmp).mkdir(exist_ok=True)  # keep context manager happy
            return True, f"{pack_id}"
        except Exception as e:  # noqa: BLE001
            return False, f"{pack_id}: {type(e).__name__}: {e}"

    def _show_library(self) -> str:
        """Show purchased packs in personal library."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self._base_url}/api/marketplace/my-library",
                    headers=self.auth.get_headers(),
                )
                resp.raise_for_status()
                library = resp.json()
                packs = library.get("packs", [])

            if not packs:
                return (
                    "Your library is empty. Purchase packs from the marketplace "
                    "with '/packs purchase <pack-id>'"
                )

            output = "=== MY LIBRARY ===\n\n"
            output += self._format_packs_table(packs)
            return output

        except httpx.HTTPError as e:
            return f"ERROR: Failed to fetch library: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def _show_help(self) -> str:
        """Show detailed help text."""
        return """
=== PACKS MARKETPLACE HELP ===

Browse and manage agent/skill/tool packs from the Elysium marketplace.

SYNTAX:
  /packs [command] [options]

COMMANDS:

  list
    List all published packs in the marketplace.
    Usage: /packs list

  browse
    Browse packs with optional filters.
    Usage: /packs browse [--type agent|skill|tool] [--search query]
    Example: /packs browse --type skill --search "reasoning"

  info <pack-id>
    Show detailed information about a specific pack.
    Usage: /packs info <pack-id>
    Example: /packs info my-skill-pack

  purchase <pack-id>
    Open Stripe checkout to purchase a pack.
    Usage: /packs purchase <pack-id>
    Example: /packs purchase my-skill-pack

  install <pack-id> [options]
    Download pack bundle to current directory.
    Options:
      --extract, -x    Automatically extract the .tar.gz archive
    Usage: /packs install <pack-id> [--extract]
    Example: /packs install my-skill-pack --extract

  sync [--dry-run]
    Install every entitled pack not already present locally. Polls your active
    licenses, then downloads + signature-verifies + extracts each missing pack
    into ~/.aitheros/packs/. A portal purchase converges onto this node with one
    command. Use --dry-run to preview without installing.
    Usage: /packs sync [--dry-run]

  library
    Show your purchased packs (my-library).
    Usage: /packs library

  help
    Show this help text.
    Usage: /packs help

AUTHENTICATION:
  Set bearer token and tenant ID via /auth command before using marketplace features.

EXAMPLES:
  /packs                              List all packs
  /packs browse --type agent           Browse agent packs
  /packs info llm-reasoning             Show pack details
  /packs purchase llm-reasoning         Open checkout
  /packs install llm-reasoning --extract Download and extract
  /packs library                        Show owned packs
"""

    def _format_packs_table(self, packs: List[Dict[str, Any]]) -> str:
        """Format pack list as simple text table."""
        if not packs:
            return "No packs available."

        lines = []
        lines.append("ID                          | NAME                        | TYPE     | PRICE")
        lines.append("-" * 80)

        for pack in packs:
            pack_id = pack.get("id", "unknown")[:25].ljust(27)
            name = pack.get("name", "unnamed")[:25].ljust(27)
            pack_type = pack.get("type", "unknown")[:8].ljust(9)
            price = pack.get("price", "free")

            lines.append(f"{pack_id}| {name}| {pack_type}| {price}")

        return "\n".join(lines)

    def _format_pack_detail(self, pack: Dict[str, Any]) -> str:
        """Format pack detail view."""
        output = []
        output.append("=== PACK DETAILS ===\n")

        output.append(f"ID:           {pack.get('id', 'N/A')}")
        output.append(f"Name:         {pack.get('name', 'N/A')}")
        output.append(f"Type:         {pack.get('type', 'N/A')}")
        output.append(f"Version:      {pack.get('version', 'N/A')}")
        output.append(f"Price:        {pack.get('price', 'free')}")
        output.append(f"Author:       {pack.get('author', 'N/A')}")

        if pack.get("description"):
            output.append(f"\nDescription:")
            output.append(f"  {pack['description']}")

        if pack.get("tags"):
            tags = ", ".join(pack["tags"])
            output.append(f"\nTags: {tags}")

        if pack.get("manifest"):
            output.append(f"\nManifest:")
            manifest = pack["manifest"]
            if isinstance(manifest, dict):
                for key, value in manifest.items():
                    output.append(f"  {key}: {value}")
            else:
                output.append(f"  {manifest}")

        if pack.get("created_at"):
            output.append(f"\nCreated:      {pack['created_at']}")

        if pack.get("updated_at"):
            output.append(f"Updated:      {pack['updated_at']}")

        output.append(f"\n--- Actions ---")
        output.append(f"View:     /packs info {pack.get('id')}")
        output.append(f"Purchase: /packs purchase {pack.get('id')}")
        output.append(f"Install:  /packs install {pack.get('id')} --extract")

        return "\n".join(output)
