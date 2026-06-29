"""ChatRelayClient — Python SDK for AitherRelay WebSocket participation.

This module provides a first-class relay client for ADK nodes, enabling them to:
1. Mint relay tokens from AitherID bearer tokens
2. Join channels over WebSocket
3. Post messages and handle @mentions
4. Auto-reconnect with exponential backoff

The server-side WebSocket auth contract is defined in AitherRelay.py (POST
/v1/auth/relay-token → JWT, connect with ?token=JWT).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx
import websockets

logger = logging.getLogger("adk.relay_client")

# ── Relay Protocol Models ─────────────────────────────────────────────────

@dataclass
class RelayMessage:
    """A message received from the relay."""
    msg_id: str
    channel: str
    nick: str
    content: str
    msg_type: str = "message"
    timestamp: float = 0.0
    thread_id: str = ""
    user_id: str = ""
    session_id: str = ""
    node_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> RelayMessage:
        return cls(
            msg_id=data.get("msg_id", ""),
            channel=data.get("channel", ""),
            nick=data.get("nick", ""),
            content=data.get("content", ""),
            msg_type=data.get("msg_type", "message"),
            timestamp=data.get("timestamp", time.time()),
            thread_id=data.get("thread_id", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            node_id=data.get("node_id", ""),
        )


# ── Relay Client ──────────────────────────────────────────────────────────

class ChatRelayClient:
    """WebSocket client for AitherRelay. Handles token minting, connection,
    channel joining, messaging, and @mention handlers with auto-reconnect.

    Usage:
        client = ChatRelayClient(
            base_url="http://localhost:8001",
            aither_bearer="sk-ant-...",
            workspace_id=None,
            user_id="user_123",
            nick="my-agent",
        )
        await client.connect()
        await client.join_channel("#team")
        await client.post_message("#team", "Hello!")

        async def handle_mention(msg):
            return f"You mentioned me in: {msg.content}"
        client.register_mention_handler("bot", handle_mention)
    """

    def __init__(
        self,
        base_url: str,
        aither_bearer: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        nick: str = "adk-node",
    ):
        """Initialize relay client.

        Args:
            base_url: HTTP base URL of the relay server (e.g. "http://localhost:8001")
            aither_bearer: AitherID Bearer token (used to mint relay JWT)
            workspace_id: Workspace ID (for scoped channels; optional)
            user_id: User ID (stored in token claims; defaults to generated UUID)
            nick: Display nick for this client (must match token or join fails)
        """
        self.base_url = base_url.rstrip("/")
        self.aither_bearer = aither_bearer
        self.workspace_id = workspace_id
        self.user_id = user_id or str(uuid.uuid4())
        self.nick = nick

        # WebSocket connection
        self._ws: Any = None
        self._relay_token: Optional[str] = None
        self._relay_token_expires: float = 0.0

        # Channel state
        self._joined_channels: set[str] = set()

        # Mention handlers: agent_name -> async callable
        self._mention_handlers: dict[str, Callable] = {}

        # Reconnect state
        self._should_reconnect = False
        self._backoff_ms = 1000
        self._max_backoff_ms = 30000
        self._reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to the relay and mint a relay token.

        Returns True on success, False if minting failed.
        Automatically retries connection on network failure.
        """
        if not await self._mint_relay_token():
            logger.error("Failed to mint relay token")
            return False

        self._should_reconnect = True
        await self._connect_ws()
        return self._ws is not None

    async def disconnect(self):
        """Disconnect from the relay (no auto-reconnect)."""
        self._should_reconnect = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def list_channels(self) -> list[dict]:
        """Fetch available channels (requires HTTP, no WS needed).

        Returns list of dicts with name, topic, user_count, etc.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/channels",
                    params={"scope": "platform", "nick": self.nick},
                    headers={"Authorization": f"Bearer {self.aither_bearer}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data if isinstance(data, list) else data.get("channels", [])
        except Exception as exc:
            logger.debug("list_channels failed: %s", exc)
        return []

    async def join_channel(self, channel: str):
        """Join a channel. Queues if disconnected; sends on next connection."""
        if not channel.startswith("#"):
            channel = f"#{channel}"

        if self._ws and self._ws.open:
            try:
                await self._ws.send(json.dumps({
                    "type": "join",
                    "channel": channel,
                    "nick": self.nick,
                }))
                self._joined_channels.add(channel)
                logger.debug("Joined channel %s", channel)
            except Exception as exc:
                logger.debug("join_channel send failed: %s", exc)
        else:
            # Queue for after reconnect
            self._joined_channels.add(channel)

    async def part_channel(self, channel: str):
        """Leave a channel."""
        if not channel.startswith("#"):
            channel = f"#{channel}"

        self._joined_channels.discard(channel)
        if self._ws and self._ws.open:
            try:
                await self._ws.send(json.dumps({
                    "type": "part",
                    "channel": channel,
                }))
                logger.debug("Parted channel %s", channel)
            except Exception as exc:
                logger.debug("part_channel send failed: %s", exc)

    async def post_message(self, channel: str, text: str):
        """Post a message to a channel."""
        if not channel.startswith("#"):
            channel = f"#{channel}"

        if not self._ws or not self._ws.open:
            logger.warning("post_message: not connected")
            return

        try:
            await self._ws.send(json.dumps({
                "type": "message",
                "channel": channel,
                "content": text,
            }))
        except Exception as exc:
            logger.debug("post_message send failed: %s", exc)

    def register_mention_handler(
        self,
        agent_name: str,
        handler: Callable[[RelayMessage], Any],
    ):
        """Register an @mention handler.

        Args:
            agent_name: The agent nick to match (e.g. "bot" matches @bot)
            handler: Async callable(msg: RelayMessage) -> str|None
                     Return a string to post back to the channel.
        """
        self._mention_handlers[agent_name.lower()] = handler

    # ── Private / Internal ────────────────────────────────────────────────

    async def _mint_relay_token(self) -> bool:
        """POST /v1/auth/relay-token to get a signed JWT.

        Per the contract, the server validates the Bearer token and returns
        a relay JWT (signed with AITHER_INTERNAL_SECRET, 1-hour expiry).
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/auth/relay-token",
                    json={
                        "workspace_id": self.workspace_id,
                        "tenant_id": None,  # ADK nodes don't have tenant_id
                    },
                    headers={"Authorization": f"Bearer {self.aither_bearer}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._relay_token = data.get("relay_token")
                    self._relay_token_expires = data.get("expires_at", 0)
                    # Use nick from response if provided (server resolves identity)
                    if "nick" in data:
                        self.nick = data["nick"]
                    logger.info(
                        "Minted relay token (expires_at=%s, nick=%s)",
                        self._relay_token_expires, self.nick,
                    )
                    return True
                else:
                    logger.error(
                        "Token mint failed: %d %s",
                        resp.status_code, resp.text[:200],
                    )
        except Exception as exc:
            logger.error("Token mint exception: %s", exc)
        return False

    async def _connect_ws(self):
        """Open WebSocket connection with relay token."""
        if not self._relay_token:
            logger.error("_connect_ws: no relay token")
            return

        ws_url = (
            self.base_url.replace("http://", "ws://")
            .replace("https://", "wss://")
        )
        ws_url = f"{ws_url}/ws/chat?token={self._relay_token}"

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(ws_url, max_size=2**20),
                timeout=10.0,
            )
            logger.info("WebSocket connected to relay")
            self._backoff_ms = 1000  # Reset backoff on success
            # Start the receive loop
            self._reconnect_task = asyncio.create_task(self._receive_loop())
        except asyncio.TimeoutError:
            logger.error("WebSocket connection timeout")
            self._ws = None
            self._schedule_reconnect()
        except Exception as exc:
            logger.error("WebSocket connection failed: %s", exc)
            self._ws = None
            self._schedule_reconnect()

    async def _receive_loop(self):
        """Read messages from the WebSocket until disconnected."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                try:
                    data = json.loads(msg)
                    await self._handle_relay_message(data)
                except json.JSONDecodeError:
                    logger.debug("Invalid JSON from relay: %s", msg[:100])
                except Exception as exc:
                    logger.debug("Message handler error: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("WebSocket receive loop error: %s", exc)
        finally:
            self._ws = None
            if self._should_reconnect:
                self._schedule_reconnect()

    async def _handle_relay_message(self, data: dict):
        """Dispatch a message from the relay."""
        msg_type = data.get("type", "message")

        if msg_type == "history":
            # On join, receive up to 200 historical messages
            messages = [
                RelayMessage.from_dict(m)
                for m in data.get("messages", [])
            ]
            logger.debug("Received %d historical messages", len(messages))

        elif msg_type == "message":
            msg = RelayMessage.from_dict(data)
            logger.debug(
                "Message in %s from %s: %s",
                msg.channel, msg.nick, msg.content[:50],
            )
            # Check for @mentions and fire handlers
            await self._check_mentions(msg)

        elif msg_type == "join":
            logger.debug(
                "%s joined %s (is_agent=%s)",
                data.get("nick"), data.get("channel"), data.get("is_agent"),
            )

        elif msg_type == "part":
            logger.debug(
                "%s left %s", data.get("nick"), data.get("channel"),
            )

        elif msg_type == "error":
            logger.warning("Relay error: %s", data.get("message"))

        elif msg_type == "who_reply":
            logger.debug(
                "Users in %s: %s",
                data.get("channel"),
                [u.get("nick") for u in data.get("users", [])],
            )

    async def _check_mentions(self, msg: RelayMessage):
        """Check if message mentions any registered agents and fire handlers."""
        content_lower = msg.content.lower()
        for agent_nick, handler in list(self._mention_handlers.items()):
            if f"@{agent_nick}" in content_lower:
                try:
                    result = handler(msg)
                    if asyncio.iscoroutine(result):
                        result = await result
                    # Post the response back to the channel
                    if result:
                        await self.post_message(msg.channel, str(result))
                except Exception as exc:
                    logger.error(
                        "Mention handler error for @%s: %s",
                        agent_nick, exc,
                    )

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff."""
        if not self._should_reconnect:
            return
        wait_ms = self._backoff_ms
        self._backoff_ms = min(self._backoff_ms * 2, self._max_backoff_ms)
        logger.info("Scheduling reconnect in %dms", wait_ms)
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.call_later(
                wait_ms / 1000.0,
                lambda: asyncio.create_task(self._reconnect()),
            )

    async def _reconnect(self):
        """Re-mint token and reconnect."""
        if not self._should_reconnect:
            return
        logger.info("Reconnecting to relay...")
        if await self._mint_relay_token():
            await self._connect_ws()
            # Re-join any channels we were in
            for ch in list(self._joined_channels):
                await self.join_channel(ch)
        else:
            logger.warning("Reconnect failed, will retry")
            self._schedule_reconnect()


# ── Singleton ─────────────────────────────────────────────────────────────

_relay_client: Optional[ChatRelayClient] = None


def get_relay_client(
    base_url: Optional[str] = None,
    aither_bearer: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    nick: str = "adk-node",
) -> ChatRelayClient:
    """Get or create the singleton relay client.

    Args:
        base_url: Relay server URL (default: http://localhost:8001)
        aither_bearer: AitherID Bearer token (required)
        workspace_id: Workspace ID (optional)
        user_id: User ID (default: generated UUID)
        nick: Display nick (default: adk-node)

    Returns:
        Singleton ChatRelayClient instance.
    """
    global _relay_client
    if _relay_client is None:
        base_url = base_url or "http://localhost:8001"
        aither_bearer = aither_bearer or ""
        _relay_client = ChatRelayClient(
            base_url=base_url,
            aither_bearer=aither_bearer,
            workspace_id=workspace_id,
            user_id=user_id,
            nick=nick,
        )
    return _relay_client
