"""
AitherDrive Sync Engine
========================

Bidirectional file sync between a local directory (the "sync root")
and the AitherOS platform via Strata.

Usage (from CLI):
    adk sync init .          # Initialize current dir as sync root
    adk sync push            # Upload local changes
    adk sync pull            # Download remote changes
    adk sync status          # Show changed/new/deleted files
    adk sync watch           # Auto-push on file change (requires watchdog)
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from adk.client.services.strata import StrataClient
from adk.client.services.data_plane import DataPlaneClient

log = logging.getLogger("adk.sync")

MANIFEST_FILE = ".aither-sync.yaml"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB default

DEFAULT_IGNORE = [
    ".git/", "node_modules/", "__pycache__/", "*.pyc", ".env",
    ".aither-sync.yaml", ".DS_Store", "Thumbs.db", "*.swp", "*.swo",
]


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _hash_file(path: str) -> str:
    """Stream-hash a file without reading it entirely into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _stable_node_id(sync_root: Path) -> str:
    """Generate a stable node ID from hostname + sync root path."""
    fingerprint = f"{socket.gethostname()}-{sync_root.resolve()}"
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    return f"anode-{digest}"


class SyncManifest:
    """Persisted sync state stored as .aither-sync.yaml in the sync root."""

    def __init__(self, sync_root: Path):
        self.path = sync_root / MANIFEST_FILE
        self.version: int = 1
        self.node_id: str = ""
        self.tenant_id: str = ""
        self.workspace_id: str = ""
        self.source_id: str = ""
        self.strata_prefix: str = ""
        self.last_sync_at: str = ""
        self.conflict_strategy: str = "last_writer_wins"
        self.settings_sync: bool = True
        self.ignore: List[str] = list(DEFAULT_IGNORE)
        self.files: Dict[str, Dict[str, Any]] = {}
        self.pending: List[Dict[str, Any]] = []
        self.max_file_size: int = MAX_FILE_SIZE

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.version = data.get("version", 1)
        self.node_id = data.get("node_id", "")
        self.tenant_id = data.get("tenant_id", "")
        self.workspace_id = data.get("workspace_id", "")
        self.source_id = data.get("source_id", "")
        self.strata_prefix = data.get("strata_prefix", "")
        self.last_sync_at = data.get("last_sync_at", "")
        self.conflict_strategy = data.get("conflict_strategy", "last_writer_wins")
        self.settings_sync = data.get("settings_sync", True)
        self.ignore = data.get("ignore", list(DEFAULT_IGNORE))
        self.files = data.get("files", {})
        self.pending = data.get("pending", [])
        self.max_file_size = data.get("max_file_size", MAX_FILE_SIZE)
        return True

    def save(self):
        data = {
            "version": self.version,
            "node_id": self.node_id,
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "source_id": self.source_id,
            "strata_prefix": self.strata_prefix,
            "last_sync_at": self.last_sync_at,
            "conflict_strategy": self.conflict_strategy,
            "settings_sync": self.settings_sync,
            "ignore": self.ignore,
            "files": self.files,
            "pending": self.pending,
            "max_file_size": self.max_file_size,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


class SyncStatus:
    """Result of comparing local files to manifest."""

    def __init__(self):
        self.changed: List[str] = []
        self.new: List[str] = []
        self.deleted: List[str] = []

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.new or self.deleted)

    def summary(self) -> str:
        parts = []
        if self.new:
            parts.append(f"{len(self.new)} new")
        if self.changed:
            parts.append(f"{len(self.changed)} modified")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        return ", ".join(parts) if parts else "up to date"


class SyncManager:
    """Core sync engine — mediates between local filesystem and Strata."""

    def __init__(
        self,
        sync_root: Path,
        strata: StrataClient,
        data_plane: DataPlaneClient,
        tenant_id: str,
        node_id: str = "",
    ):
        self.sync_root = sync_root.resolve()
        self.strata = strata
        self.data_plane = data_plane
        self.tenant_id = tenant_id
        self.manifest = SyncManifest(self.sync_root)
        self.node_id = node_id or _stable_node_id(self.sync_root)
        self._watcher = None
        self._watch_task: Optional[asyncio.Task] = None

    def _is_ignored(self, rel_path: str) -> bool:
        for pattern in self.manifest.ignore:
            if pattern.endswith("/"):
                # Directory pattern — check if any component matches
                if any(fnmatch.fnmatch(part, pattern.rstrip("/"))
                       for part in Path(rel_path).parts):
                    return True
            elif fnmatch.fnmatch(rel_path, pattern):
                return True
            elif fnmatch.fnmatch(Path(rel_path).name, pattern):
                return True
        return False

    def _scan_local(self) -> Dict[str, Dict[str, Any]]:
        """Scan sync root and return {relative_path: {hash, mtime, size}}."""
        files: Dict[str, Dict[str, Any]] = {}
        for root, dirs, filenames in os.walk(self.sync_root):
            # Skip ignored directories in-place
            dirs[:] = [d for d in dirs
                       if not self._is_ignored(
                           os.path.relpath(os.path.join(root, d), self.sync_root) + "/")]
            for name in filenames:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, self.sync_root).replace("\\", "/")
                if self._is_ignored(rel):
                    continue
                try:
                    stat = os.stat(full)
                except OSError:
                    continue
                if stat.st_size > self.manifest.max_file_size:
                    continue
                files[rel] = {
                    "hash": _hash_file(full),
                    "mtime": int(stat.st_mtime),
                    "size": stat.st_size,
                }
        return files

    def _validate_rel_path(self, rel: str) -> bool:
        """Reject path traversal attempts (e.g. ../../etc/passwd)."""
        try:
            resolved = (self.sync_root / rel).resolve()
            return resolved.is_relative_to(self.sync_root)
        except (ValueError, OSError):
            return False

    async def _flush_pending(self):
        """Retry queued items from prior failed push/pull attempts."""
        if not self.manifest.pending:
            return
        remaining: List[Dict[str, Any]] = []
        prefix = self.manifest.strata_prefix
        for item in self.manifest.pending:
            try:
                if item["action"] == "upload":
                    full = self.sync_root / item["path"]
                    if full.exists():
                        data = full.read_bytes()
                        await self.strata.upload_file(
                            path=f"{prefix}/{item['path']}",
                            data=data,
                            metadata={"source": "adk_sync"},
                        )
                        self.manifest.files[item["path"]] = {
                            "hash": _sha256(data),
                            "mtime": int(os.path.getmtime(str(full))),
                            "size": len(data),
                        }
                elif item["action"] == "delete":
                    await self.strata.delete(f"{prefix}/{item['path']}")
                    self.manifest.files.pop(item["path"], None)
            except Exception:
                remaining.append(item)
        self.manifest.pending = remaining

    # -- Public API -----------------------------------------------------------

    async def init(self) -> dict:
        """Initialize sync root: create manifest, register with DataPlane."""
        if self.manifest.load():
            return {"status": "already_initialized", "node_id": self.manifest.node_id}

        self.manifest.node_id = self.node_id
        self.manifest.tenant_id = self.tenant_id
        self.manifest.strata_prefix = f"tenants/{self.tenant_id}/sync/{self.node_id}"

        # Register as DataPlane source
        result = await self.data_plane.register_source(
            name=f"AitherDrive ({platform.node()}:{self.sync_root.name})",
            connector_type="adk_sync",
            connection_config={
                "node_id": self.node_id,
                "hostname": socket.gethostname(),
                "sync_root": str(self.sync_root),
                "strata_prefix": self.manifest.strata_prefix,
            },
        )
        source_id = result.get("id", "")
        if not source_id and "error" not in result:
            source_id = result.get("source_id", "")
        self.manifest.source_id = source_id

        # Scan directory and build initial manifest
        self.manifest.files = self._scan_local()
        self.manifest.save()

        return {
            "status": "initialized",
            "node_id": self.node_id,
            "source_id": source_id,
            "files_scanned": len(self.manifest.files),
        }

    def status(self) -> SyncStatus:
        """Compare local filesystem to manifest. Returns changed/new/deleted."""
        result = SyncStatus()
        current = self._scan_local()
        known = self.manifest.files

        for path, info in current.items():
            if path not in known:
                result.new.append(path)
            elif info["hash"] != known[path].get("hash"):
                result.changed.append(path)

        for path in known:
            if path not in current:
                result.deleted.append(path)

        return result

    async def push(self) -> dict:
        """Upload local changes to Strata."""
        await self._flush_pending()

        st = self.status()
        uploaded = 0
        deleted = 0
        errors: List[str] = []
        prefix = self.manifest.strata_prefix

        for rel in st.new + st.changed:
            if not self._validate_rel_path(rel):
                errors.append(f"{rel}: path traversal rejected")
                continue
            full = self.sync_root / rel
            try:
                data = full.read_bytes()
                await self.strata.upload_file(
                    path=f"{prefix}/{rel}",
                    data=data,
                    metadata={"mtime": os.path.getmtime(str(full)), "source": "adk_sync"},
                )
                self.manifest.files[rel] = {
                    "hash": _sha256(data),
                    "mtime": int(os.path.getmtime(str(full))),
                    "size": len(data),
                }
                uploaded += 1
            except Exception as e:
                errors.append(f"{rel}: {e}")
                log.warning("Push failed for %s: %s", rel, e)
                self.manifest.pending.append({
                    "action": "upload", "path": rel, "timestamp": time.time(),
                })

        for rel in st.deleted:
            try:
                await self.strata.delete(f"{prefix}/{rel}")
                self.manifest.files.pop(rel, None)
                deleted += 1
            except Exception as e:
                errors.append(f"delete {rel}: {e}")
                self.manifest.pending.append({
                    "action": "delete", "path": rel, "timestamp": time.time(),
                })

        self.manifest.last_sync_at = datetime.now(timezone.utc).isoformat()
        self.manifest.save()

        return {"uploaded": uploaded, "deleted": deleted, "errors": errors}

    async def pull(self) -> dict:
        """Download remote changes from Strata."""
        prefix = self.manifest.strata_prefix
        remote_listing = await self.strata.list_dir(prefix)
        if "error" in remote_listing:
            return {"downloaded": 0, "error": remote_listing["error"]}

        remote_files = remote_listing.get("files", remote_listing.get("items", []))
        downloaded = 0
        errors: List[str] = []

        for entry in remote_files:
            remote_path = entry if isinstance(entry, str) else entry.get("path", "")
            if not remote_path:
                continue
            # Strip prefix to get relative path
            if remote_path.startswith(prefix + "/"):
                rel = remote_path[len(prefix) + 1:]
            elif remote_path.startswith(prefix):
                rel = remote_path[len(prefix):]
            else:
                rel = remote_path

            if not rel or self._is_ignored(rel):
                continue

            if not self._validate_rel_path(rel):
                errors.append(f"{rel}: path traversal rejected")
                continue

            # Check if remote is newer than local
            remote_meta = entry if isinstance(entry, dict) else {}
            remote_hash = remote_meta.get("hash", "")
            local_info = self.manifest.files.get(rel, {})

            if remote_hash and remote_hash == local_info.get("hash"):
                continue  # Already in sync

            try:
                data = await self.strata.download_file(f"{prefix}/{rel}")
                if not data:
                    continue

                local_path = self.sync_root / rel

                # Conflict detection: check if local also changed since last sync
                if local_path.exists() and local_info.get("hash"):
                    current_local_hash = _hash_file(str(local_path))
                    manifest_hash = local_info["hash"]
                    remote_file_hash = _sha256(data)
                    if (current_local_hash != manifest_hash
                            and remote_file_hash != current_local_hash):
                        # Both sides changed — conflict
                        if self.manifest.conflict_strategy != "last_writer_wins":
                            conflict_path = local_path.with_suffix(
                                local_path.suffix + ".conflict")
                            conflict_path.write_bytes(local_path.read_bytes())
                            log.warning("Conflict: saved local copy as %s", conflict_path)

                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(data)
                file_hash = _sha256(data)
                self.manifest.files[rel] = {
                    "hash": file_hash,
                    "mtime": int(time.time()),
                    "size": len(data),
                }
                downloaded += 1
            except Exception as e:
                errors.append(f"{rel}: {e}")
                log.warning("Pull failed for %s: %s", rel, e)

        self.manifest.last_sync_at = datetime.now(timezone.utc).isoformat()
        self.manifest.save()

        return {"downloaded": downloaded, "errors": errors}

    async def sync(self) -> dict:
        """Bidirectional sync: push local, then pull remote."""
        push_result = await self.push()
        pull_result = await self.pull()
        return {
            "uploaded": push_result.get("uploaded", 0),
            "downloaded": pull_result.get("downloaded", 0),
            "deleted": push_result.get("deleted", 0),
            "errors": push_result.get("errors", []) + pull_result.get("errors", []),
        }

    async def watch(self) -> bool:
        """Start filesystem watcher. Returns False if watchdog not installed."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            return False

        manager = self

        class _Handler(FileSystemEventHandler):
            def __init__(self):
                self._debounce: Optional[asyncio.TimerHandle] = None
                self._loop = asyncio.get_event_loop()

            def _schedule_push(self):
                if self._debounce:
                    self._debounce.cancel()
                self._debounce = self._loop.call_later(
                    5.0, lambda: asyncio.ensure_future(manager.push()),
                )

            def on_modified(self, event):
                if not event.is_directory:
                    rel = os.path.relpath(event.src_path, str(manager.sync_root))
                    if not manager._is_ignored(rel.replace("\\", "/")):
                        self._schedule_push()

            on_created = on_modified
            on_deleted = on_modified

        observer = Observer()
        observer.schedule(_Handler(), str(self.sync_root), recursive=True)
        observer.start()
        self._watcher = observer
        log.info("File watcher started for %s", self.sync_root)
        return True

    def stop(self):
        """Stop the filesystem watcher."""
        if self._watcher:
            self._watcher.stop()
            self._watcher.join(timeout=5)
            self._watcher = None
            log.info("File watcher stopped")

    async def push_settings(self) -> dict:
        """Upload local ~/.aither/config.yaml to Strata for portal sync."""
        config_path = Path.home() / ".aither" / "config.yaml"
        if not config_path.exists():
            config_path = Path.home() / ".aither" / "config.json"
        if not config_path.exists():
            return {"error": "No local config found"}
        data = config_path.read_bytes()
        prefix = self.manifest.strata_prefix
        return await self.strata.upload_file(
            path=f"{prefix}/_config.yaml",
            data=data,
            metadata={"type": "settings", "source": "adk_sync"},
        )

    async def pull_settings(self) -> dict:
        """Download remote config and merge into local settings (remote wins on conflicts)."""
        prefix = self.manifest.strata_prefix
        data = await self.strata.download_file(f"{prefix}/_config.yaml")
        if not data:
            return {"status": "no_remote_config"}
        config_path = Path.home() / ".aither" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        local_data: dict = {}
        if config_path.exists():
            try:
                local_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                local_data = {}
        try:
            remote_data = yaml.safe_load(data.decode("utf-8")) or {}
        except Exception:
            remote_data = {}
        merged = {**local_data, **remote_data}
        config_path.write_text(
            yaml.safe_dump(merged, default_flow_style=False), encoding="utf-8",
        )
        return {"status": "merged", "bytes": len(data)}

    def add_ignore(self, pattern: str):
        """Add an ignore pattern to the manifest."""
        if pattern not in self.manifest.ignore:
            self.manifest.ignore.append(pattern)
            self.manifest.save()
