"""AitherADK sync — bidirectional sync of adk's own ~/.aither data.

This module provides:
  - Device mTLS identity enrollment and persistence
  - DriveClient for cloud change-feed polling
  - SyncEngine that reuses AitherNode's reconcile algorithm
  - Per-agent fabric for syncing memory/graph/session data

Reuses AitherOS's pure reconcile algorithm (drive_sync_core) without
reimplementing the three-way merge logic.
"""

__all__ = [
    "device_identity",
    "drive_client",
    "adk_sync_engine",
]
