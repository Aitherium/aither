"""Client for gateway.aitherium.com — auth, agent registration, discovery, remote inference.

The canonical GatewayClient now lives in adk.client._gateway.
This module re-exports it for backward compatibility.
"""

from adk.client._gateway import GatewayClient  # noqa: F401

Gateway = GatewayClient
