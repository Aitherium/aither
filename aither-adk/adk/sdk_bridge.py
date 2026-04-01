"""
ADK ↔ AitherSDK Bridge
========================

When aithersdk is installed, ADK uses AitherClient for Genesis calls
instead of raw httpx. This ensures consistent auth, retries, and typing.

The returned client also provides service sub-clients:
    client.context   — LiveContext (port 8098)
    client.a2a       — A2A Gateway (port 8766)
    client.strata    — Strata VFS (port 8136)
    client.expeditions — Expeditions (port 8785)
    client.voice     — AitherVoice (port 8083)
    client.conversations — Conversation management

Usage:
    from adk.sdk_bridge import get_genesis_client

    client = get_genesis_client()
    if client:
        response = await client.chat("hello")
        status = await client.context.status("session-1")
    else:
        # Fall back to raw httpx
        ...
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("adk.sdk_bridge")

_client = None


def get_genesis_client():
    """Get an AitherClient instance, or None if aithersdk not installed."""
    global _client
    if _client is not None:
        return _client
    try:
        from aithersdk import AitherClient
        url = os.environ.get("AITHER_URL",
              os.environ.get("AITHER_ORCHESTRATOR_URL", "http://localhost:8001"))
        _client = AitherClient(url=url)
        logger.debug("Using AitherSDK client for Genesis calls")
        return _client
    except ImportError:
        logger.debug("aithersdk not installed — using raw httpx")
        return None


def sdk_available() -> bool:
    """Check if AitherSDK is importable."""
    try:
        import aithersdk
        return True
    except ImportError:
        return False
