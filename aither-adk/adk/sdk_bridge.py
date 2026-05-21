"""
ADK ↔ AitherClient Bridge
===========================

AitherClient is now built into adk.client (absorbed from aithersdk).
This module provides the same get_genesis_client() API for backward compat.

Usage:
    from adk.sdk_bridge import get_genesis_client

    client = get_genesis_client()
    response = await client.chat("hello")
    status = await client.context.status("session-1")
"""

from __future__ import annotations

import os
from typing import Optional

from adk.client import AitherClient

_client: Optional[AitherClient] = None


def get_genesis_client() -> AitherClient:
    """Get an AitherClient instance (always available — built into ADK)."""
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("AITHER_URL",
          os.environ.get("AITHER_ORCHESTRATOR_URL", "http://localhost:8001"))
    _client = AitherClient(url=url)
    return _client


def sdk_available() -> bool:
    """Always True — client is built into ADK."""
    return True
