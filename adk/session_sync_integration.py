"""
Integration helper — wiring SessionSyncClient into ConversationStore.

This module provides convenience functions for enabling automatic session
sync (push-on-save) in applications that use ConversationStore.

Typical usage in a shell or agent app:

    from adk.conversations import ConversationStore
    from adk.session_sync_integration import enable_session_sync
    from adk.auth import get_current_credentials

    # During app startup:
    store = ConversationStore()
    creds = await get_current_credentials()
    await enable_session_sync(store, creds)

    # Now every session save will automatically push to the cloud
    # (debounced, with fail-soft error handling)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("adk.session_sync_integration")


@dataclass
class SyncCredentials:
    """Credentials for session sync."""
    access_token: str
    gateway_url: str = "http://localhost:8001"


async def enable_session_sync(
    conversation_store,
    credentials: Optional[SyncCredentials] = None,
    config: Optional[Any] = None,
) -> bool:
    """
    Enable automatic session sync in a ConversationStore.

    This function registers an async hook that pushes sessions to the cloud
    after each local save. The push is debounced and fails soft (logs warnings
    but doesn't block local operation).

    Args:
        conversation_store: ConversationStore instance
        credentials: SyncCredentials with access_token and gateway_url.
                    If None, loads from environment / auth.json
        config: Optional SessionSyncConfig (if None, loads from environment)

    Returns:
        True if sync was enabled, False if disabled or missing credentials
    """
    # Import here to avoid circular dependencies
    from adk.session_sync import SessionSyncClient, SessionSyncConfig

    # Check if sync is enabled globally
    if not os.environ.get("AITHER_SESSION_SYNC", "false").lower() in ("true", "1"):
        log.debug("Session sync disabled via AITHER_SESSION_SYNC")
        return False

    # Load credentials if not provided
    if credentials is None:
        credentials = await _load_sync_credentials()
        if credentials is None:
            log.debug("No credentials available for session sync")
            return False

    # Create the sync client
    sync_client = SessionSyncClient(
        gateway_url=credentials.gateway_url,
        auth_token=credentials.access_token,
        config=config,
    )

    # Create the async hook that will be called on each save
    async def sync_hook(
        session_id: str,
        agent_name: str,
        messages: list[dict[str, Any]],
        created_at: float,
        updated_at: float,
        metadata: dict[str, Any],
    ) -> None:
        """Hook called after each session save."""
        # Check debounce window
        # debounce_sync returns the unix time at which this session becomes
        # eligible to sync. Only skip if that time is still in the FUTURE
        # (a bare ">0" check was always true — sessions never synced).
        next_eligible = sync_client.debounce_sync(session_id)
        if next_eligible > time.time():
            log.debug(
                "Session %s debounced (eligible at %.1f)",
                session_id, next_eligible
            )
            return

        # Push to cloud (fire-and-forget)
        result = await sync_client.sync_session(
            session_id=session_id,
            agent_name=agent_name,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

        if result.get("error"):
            log.warning(
                "Session sync error for %s: %s",
                session_id, result.get("detail", "unknown")
            )
        else:
            log.debug("Session %s synced (synced=%d)", session_id, result.get("synced", 0))

    # Register the hook
    conversation_store.set_sync_hook(sync_hook)
    log.info("Session sync enabled for ConversationStore")
    return True


async def _load_sync_credentials() -> Optional[SyncCredentials]:
    """Load sync credentials from auth.json or environment."""
    import json

    # Try auth.json first
    auth_file = Path.home() / ".aither" / "auth.json"
    if auth_file.exists():
        try:
            auth_data = json.loads(auth_file.read_text(encoding="utf-8"))

            # Try to extract access_token (format varies)
            access_token = None
            if "profiles" in auth_data:
                # New format: profiles -> default -> access_token
                access_token = auth_data.get("profiles", {}).get("default", {}).get("access_token")
            if not access_token:
                # Fallback: flat access_token
                access_token = auth_data.get("access_token")

            if access_token:
                gateway_url = os.environ.get("AITHER_GATEWAY_URL", "http://localhost:8001")
                return SyncCredentials(
                    access_token=access_token,
                    gateway_url=gateway_url,
                )
        except Exception as e:
            log.debug("Failed to load credentials from auth.json: %s", e)

    # Fallback: check environment
    access_token = os.environ.get("AITHER_SYNC_TOKEN")
    if access_token:
        gateway_url = os.environ.get("AITHER_GATEWAY_URL", "http://localhost:8001")
        return SyncCredentials(
            access_token=access_token,
            gateway_url=gateway_url,
        )

    return None


async def sync_all_local_sessions(
    conversation_store,
    credentials: Optional[SyncCredentials] = None,
) -> dict[str, Any]:
    """
    Perform a one-time batch sync of all local sessions to the cloud.

    Useful for initial setup or manual sync.

    Args:
        conversation_store: ConversationStore instance
        credentials: Optional SyncCredentials (loads from env if not provided)

    Returns:
        Summary dict from the sync operation
    """
    from adk.session_sync import SessionSyncClient

    if credentials is None:
        credentials = await _load_sync_credentials()
        if credentials is None:
            return {"error": True, "detail": "No credentials available"}

    sync_client = SessionSyncClient(
        gateway_url=credentials.gateway_url,
        auth_token=credentials.access_token,
    )

    return await sync_client.sync_all_from_disk(conversation_store._dir)
