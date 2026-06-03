"""
Aither ↔ IRC Bridge — AT Protocol social feed in your IRC client.
==================================================================

Bridges the Aither social platform (AT Protocol) with the ADK's
built-in IRC-compatible ChatRelay. Two-directional:

    AT Protocol → IRC:  New posts appear in #aither-feed
    IRC → AT Protocol:  Messages in #aither-post get published

Architecture::

    AitherSocial service (port 8192)
        │
        │  /timeline  (poll every N seconds)
        │  /post      (publish from IRC)
        ▼
    AitherBridge
        │
        ├── #aither-feed   (read-only: new AT Proto posts show here)
        ├── #aither-post   (write: messages here become Aither posts)
        └── #aither-notifs  (mentions, likes, reposts)
        │
        ▼
    ChatRelay → IRCServer (port 6667) + WebSocket (/ws/chat)

Usage::

    from adk.aither_bridge import AitherBridge

    bridge = AitherBridge(chat_relay, aither_url="http://localhost:8192")
    await bridge.start()

Environment:
    AITHER_SOCIAL_URL       Base URL for AitherSocial service
    AITHER_BRIDGE_AGENT     Agent ID for posting (default: "aeon")
    AITHER_BRIDGE_POLL_SEC  Poll interval in seconds (default: 30)
    AITHER_BRIDGE_ENABLED   Enable/disable bridge (default: "true")
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("adk.aither_bridge")

# ── Configuration ────────────────────────────────────────────────────────

_DEFAULT_AITHER_URL = "http://localhost:8192"
_DEFAULT_AGENT = "aeon"
_DEFAULT_POLL_SEC = 30
_MAX_POST_LEN = 300  # AT Protocol grapheme limit

# IRC channels the bridge manages
FEED_CHANNEL = "#aither-feed"
POST_CHANNEL = "#aither-post"
NOTIF_CHANNEL = "#aither-notifs"

_BRIDGE_NICK = "AitherBot"
_GRAPHEME_APPROX = re.compile(r"[\U00010000-\U0010FFFF]")  # surrogate pairs


def _grapheme_len(text: str) -> int:
    """Approximate grapheme count (AT Protocol counts graphemes, not bytes)."""
    # Each surrogate-pair character counts as 1 grapheme
    count = len(text)
    # Emoji sequences are ~1 grapheme each but len() counts them as 2+
    # This is an approximation; for production use the `grapheme` package
    return count


# ── Data ─────────────────────────────────────────────────────────────────


@dataclass
class BridgeConfig:
    """Bridge configuration loaded from environment."""
    aither_url: str = ""
    agent_id: str = _DEFAULT_AGENT
    poll_interval: int = _DEFAULT_POLL_SEC
    enabled: bool = True
    post_channel: str = POST_CHANNEL
    feed_channel: str = FEED_CHANNEL
    notif_channel: str = NOTIF_CHANNEL

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        return cls(
            aither_url=os.getenv("AITHER_SOCIAL_URL", _DEFAULT_AITHER_URL),
            agent_id=os.getenv("AITHER_BRIDGE_AGENT", _DEFAULT_AGENT),
            poll_interval=int(os.getenv("AITHER_BRIDGE_POLL_SEC", str(_DEFAULT_POLL_SEC))),
            enabled=os.getenv("AITHER_BRIDGE_ENABLED", "true").lower() in ("1", "true", "yes"),
        )


@dataclass
class SeenPost:
    """Track posts we've already relayed to IRC."""
    uri: str
    timestamp: float = 0.0


# ── Bridge ───────────────────────────────────────────────────────────────


class AitherBridge:
    """Bidirectional bridge between AT Protocol (Aither) and IRC ChatRelay.

    Feed direction (AT Proto → IRC):
        Polls /timeline on AitherSocial, formats new posts as IRC messages,
        and injects them into #aither-feed via ChatRelay.post().

    Post direction (IRC → AT Proto):
        Listens for messages in #aither-post via ChatRelay event handler.
        When a user posts there, it calls AitherSocial /post to publish.

    Notification direction (AT Proto → IRC):
        Polls /notifications on AitherSocial and relays mentions/likes
        to #aither-notifs.
    """

    def __init__(
        self,
        chat_relay: Any,  # ChatRelay instance
        config: BridgeConfig | None = None,
    ):
        self._relay = chat_relay
        self._config = config or BridgeConfig.from_env()
        self._http: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._seen_uris: set[str] = set()  # Post URIs already relayed
        self._seen_max = 500  # Trim after this many
        self._last_poll: float = 0.0
        self._stats = {
            "posts_relayed_to_irc": 0,
            "posts_sent_from_irc": 0,
            "errors": 0,
            "last_poll": 0.0,
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self):
        """Start the bridge — create channels, wire handlers, begin polling."""
        if not self._config.enabled:
            logger.info("Aither bridge disabled (AITHER_BRIDGE_ENABLED=false)")
            return

        logger.info(
            "Starting Aither ↔ IRC bridge (url=%s, agent=%s, poll=%ds)",
            self._config.aither_url, self._config.agent_id, self._config.poll_interval,
        )

        # Create bridge channels in the relay
        self._relay.create_channel(
            self._config.feed_channel,
            topic="📡 Live Aither feed — AT Protocol posts appear here",
            mode="public",
            created_by=_BRIDGE_NICK,
        )
        self._relay.create_channel(
            self._config.post_channel,
            topic="✏️ Post to Aither — messages here get published via AT Protocol",
            mode="public",
            created_by=_BRIDGE_NICK,
        )
        self._relay.create_channel(
            self._config.notif_channel,
            topic="🔔 Aither notifications — mentions, likes, reposts",
            mode="public",
            created_by=_BRIDGE_NICK,
        )

        # Register the bot user
        self._relay.join(self._config.feed_channel, _BRIDGE_NICK, is_agent=True)
        self._relay.join(self._config.post_channel, _BRIDGE_NICK, is_agent=True)
        self._relay.join(self._config.notif_channel, _BRIDGE_NICK, is_agent=True)

        # Wire: messages in #aither-post → publish to AT Protocol
        self._relay.on("message", self._on_chat_message)

        # HTTP client for AitherSocial
        self._http = httpx.AsyncClient(
            base_url=self._config.aither_url,
            timeout=httpx.Timeout(15.0),
        )

        # Check connectivity
        healthy = await self._check_health()
        if not healthy:
            logger.warning(
                "AitherSocial not reachable at %s — bridge will retry on poll",
                self._config.aither_url,
            )

        # Start polling loop
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Aither ↔ IRC bridge started")

    async def stop(self):
        """Stop the bridge gracefully."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._http:
            await self._http.aclose()
            self._http = None
        logger.info("Aither ↔ IRC bridge stopped")

    def status(self) -> dict:
        """Return bridge status for health endpoints."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "aither_url": self._config.aither_url,
            "agent_id": self._config.agent_id,
            "poll_interval": self._config.poll_interval,
            "seen_posts": len(self._seen_uris),
            **self._stats,
        }

    # ── Feed Polling (AT Protocol → IRC) ─────────────────────────────

    async def _poll_loop(self):
        """Background loop: poll AitherSocial timeline → relay to IRC."""
        while self._running:
            try:
                await self._poll_timeline()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._stats["errors"] += 1
                logger.debug("Aither poll error: %s", exc)
            await asyncio.sleep(self._config.poll_interval)

    async def _poll_timeline(self):
        """Fetch timeline from AitherSocial and relay new posts to IRC."""
        if not self._http:
            return

        try:
            resp = await self._http.get(
                "/timeline",
                params={"agent_id": self._config.agent_id, "limit": 20},
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            posts = data.get("timeline", data.get("posts", []))

            new_count = 0
            for post in reversed(posts):  # oldest first
                uri = post.get("uri", "")
                if not uri or uri in self._seen_uris:
                    continue

                self._seen_uris.add(uri)
                self._relay_post_to_irc(post)
                new_count += 1

            if new_count:
                self._stats["posts_relayed_to_irc"] += new_count
                logger.debug("Relayed %d new Aither posts to IRC", new_count)

            self._stats["last_poll"] = time.time()
            self._trim_seen()

        except httpx.ConnectError:
            pass  # Service not up yet, silent
        except Exception as exc:
            logger.debug("Timeline poll failed: %s", exc)

    def _relay_post_to_irc(self, post: dict):
        """Format an AT Protocol post and inject into #aither-feed."""
        author = post.get("author", {})
        handle = author.get("handle", author.get("did", "unknown"))
        display = author.get("displayName", handle)

        record = post.get("record", {})
        text = record.get("text", "")
        if not text:
            return

        # Truncate for IRC readability
        if len(text) > 400:
            text = text[:397] + "..."

        # Replace newlines with spaces for IRC single-line
        text = text.replace("\n", " ┃ ")

        # Format: <handle> post content [likes/reposts]
        likes = post.get("likeCount", 0)
        reposts = post.get("repostCount", 0)
        replies = post.get("replyCount", 0)

        stats_parts = []
        if likes:
            stats_parts.append(f"♥{likes}")
        if reposts:
            stats_parts.append(f"🔁{reposts}")
        if replies:
            stats_parts.append(f"💬{replies}")
        stats = f" [{' '.join(stats_parts)}]" if stats_parts else ""

        irc_text = f"@{handle}: {text}{stats}"

        # Post to feed channel as the bridge bot
        self._relay.post(
            self._config.feed_channel,
            _BRIDGE_NICK,
            irc_text,
        )

    # ── IRC → AT Protocol Posting ────────────────────────────────────

    def _on_chat_message(self, data: dict):
        """Handle messages from the ChatRelay — bridge #aither-post to AT Protocol."""
        channel = data.get("channel", "")
        nick = data.get("nick", "")
        content = data.get("content", "")

        # Only bridge messages from the post channel, not from the bot itself
        if channel != self._config.post_channel:
            return
        if nick == _BRIDGE_NICK:
            return
        if not content or content.startswith("/"):
            return  # Skip IRC commands

        # Schedule the async post
        asyncio.ensure_future(self._post_to_aither(nick, content))

    async def _post_to_aither(self, nick: str, content: str):
        """Publish an IRC message to Aither via the AT Protocol service."""
        if not self._http:
            return

        # Enforce grapheme limit
        if _grapheme_len(content) > _MAX_POST_LEN:
            content = content[:_MAX_POST_LEN - 3] + "..."

        # Attribution: prepend IRC nick if it's not the configured agent
        text = f"[via {nick}] {content}" if nick != self._config.agent_id else content

        try:
            resp = await self._http.post("/post", json={
                "agent_id": self._config.agent_id,
                "text": text,
            })

            if resp.status_code == 200:
                result = resp.json()
                uri = result.get("uri", "?")
                self._stats["posts_sent_from_irc"] += 1

                # Confirm in the post channel
                self._relay.post(
                    self._config.post_channel,
                    _BRIDGE_NICK,
                    f"✅ Posted to Aither: {uri}",
                )
                logger.info("IRC→Aither post by %s: %s", nick, text[:80])
            else:
                error = resp.text[:200]
                self._relay.post(
                    self._config.post_channel,
                    _BRIDGE_NICK,
                    f"❌ Post failed ({resp.status_code}): {error}",
                )
                self._stats["errors"] += 1

        except Exception as exc:
            self._relay.post(
                self._config.post_channel,
                _BRIDGE_NICK,
                f"❌ Post failed: {exc}",
            )
            self._stats["errors"] += 1

    # ── Notifications (AT Protocol → IRC) ────────────────────────────

    async def poll_notifications(self):
        """Poll AitherSocial for notifications and relay to IRC.

        Called from the poll loop if the service supports /notifications.
        """
        if not self._http:
            return

        try:
            resp = await self._http.get(
                "/notifications",
                params={"agent_id": self._config.agent_id},
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            notifs = data.get("notifications", [])

            for notif in notifs:
                reason = notif.get("reason", "")
                author = notif.get("author", {})
                handle = author.get("handle", "unknown")

                if reason == "like":
                    self._relay.post(
                        self._config.notif_channel, _BRIDGE_NICK,
                        f"♥ @{handle} liked your post",
                    )
                elif reason == "repost":
                    self._relay.post(
                        self._config.notif_channel, _BRIDGE_NICK,
                        f"🔁 @{handle} reposted your post",
                    )
                elif reason == "follow":
                    self._relay.post(
                        self._config.notif_channel, _BRIDGE_NICK,
                        f"👤 @{handle} followed you",
                    )
                elif reason == "mention":
                    text = notif.get("record", {}).get("text", "")[:200]
                    self._relay.post(
                        self._config.notif_channel, _BRIDGE_NICK,
                        f"📣 @{handle} mentioned you: {text}",
                    )
                elif reason == "reply":
                    text = notif.get("record", {}).get("text", "")[:200]
                    self._relay.post(
                        self._config.notif_channel, _BRIDGE_NICK,
                        f"💬 @{handle} replied: {text}",
                    )

        except httpx.ConnectError:
            pass
        except Exception as exc:
            logger.debug("Notification poll failed: %s", exc)

    # ── Helpers ──────────────────────────────────────────────────────

    async def _check_health(self) -> bool:
        """Check if AitherSocial is reachable."""
        if not self._http:
            return False
        try:
            resp = await self._http.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    def _trim_seen(self):
        """Prevent unbounded growth of the seen-URIs set."""
        if len(self._seen_uris) > self._seen_max:
            # Keep the most recent half
            excess = len(self._seen_uris) - (self._seen_max // 2)
            for _ in range(excess):
                self._seen_uris.pop()


# ── Singleton ────────────────────────────────────────────────────────────

_bridge: AitherBridge | None = None


def get_aither_bridge() -> AitherBridge | None:
    """Return the active bridge instance, if any."""
    return _bridge


async def init_aither_bridge(chat_relay: Any) -> AitherBridge | None:
    """Initialize and start the Aither ↔ IRC bridge.

    Called from server.py after the ChatRelay is initialized.

    Args:
        chat_relay: The ChatRelay instance to bridge with.

    Returns:
        The bridge instance, or None if disabled.
    """
    global _bridge

    config = BridgeConfig.from_env()
    if not config.enabled:
        logger.info("Aither bridge disabled")
        return None

    _bridge = AitherBridge(chat_relay, config)
    await _bridge.start()
    return _bridge
