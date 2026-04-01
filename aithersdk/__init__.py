"""
AitherSDK — Python Client for AitherOS
=======================================

The canonical client library for interacting with AitherOS services.
Used by AitherShell, AitherDesktop, AitherConnect, and third-party apps.

Usage:
    from aithersdk import AitherClient

    client = AitherClient()  # connects to localhost:8001
    response = await client.chat("Hello!")
    print(response.text)

    # Service sub-clients
    await client.context.status("session-1")
    await client.strata.write("/path", "content")
    await client.a2a.services()
    await client.expeditions.submit("objective")
    await client.voice.transcribe(audio_bytes)
    await client.conversations.list()
"""

__version__ = "0.2.0"

from aithersdk.client import AitherClient, AitherResponse
from aithersdk.models import ChatRequest, ChatResponse, WillInfo, AgentInfo
from aithersdk.gateway import GatewayClient
from aithersdk.services import (
    ContextClient,
    A2AClient,
    StrataClient,
    ExpeditionClient,
    VoiceClient,
    ConversationClient,
)

__all__ = [
    "AitherClient",
    "AitherResponse",
    "ChatRequest",
    "ChatResponse",
    "WillInfo",
    "AgentInfo",
    "GatewayClient",
    "ContextClient",
    "A2AClient",
    "StrataClient",
    "ExpeditionClient",
    "VoiceClient",
    "ConversationClient",
]
