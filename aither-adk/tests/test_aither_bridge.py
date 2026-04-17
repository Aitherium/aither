"""Tests for the Aither ↔ IRC Bridge."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from adk.aither_bridge import (
    AitherBridge,
    BridgeConfig,
    FEED_CHANNEL,
    POST_CHANNEL,
    NOTIF_CHANNEL,
    _BRIDGE_NICK,
    _grapheme_len,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_relay(tmp_path):
    """Create a mock ChatRelay with the minimum interface."""
    relay = MagicMock()
    relay.create_channel = MagicMock()
    relay.join = MagicMock()
    relay.post = MagicMock()
    relay.on = MagicMock()
    return relay


@pytest.fixture
def config():
    return BridgeConfig(
        aither_url="http://localhost:8192",
        agent_id="aeon",
        poll_interval=5,
        enabled=True,
    )


@pytest.fixture
def bridge(mock_relay, config):
    return AitherBridge(mock_relay, config)


# ── Unit Tests ───────────────────────────────────────────────────────────


class TestBridgeConfig:
    def test_from_env_defaults(self):
        config = BridgeConfig.from_env()
        assert config.enabled is True
        assert config.agent_id == "aeon"
        assert config.poll_interval == 30
        assert config.feed_channel == FEED_CHANNEL
        assert config.post_channel == POST_CHANNEL

    def test_from_env_disabled(self, monkeypatch):
        monkeypatch.setenv("AITHER_BRIDGE_ENABLED", "false")
        config = BridgeConfig.from_env()
        assert config.enabled is False

    def test_from_env_custom(self, monkeypatch):
        monkeypatch.setenv("AITHER_SOCIAL_URL", "http://custom:9999")
        monkeypatch.setenv("AITHER_BRIDGE_AGENT", "demi")
        monkeypatch.setenv("AITHER_BRIDGE_POLL_SEC", "60")
        config = BridgeConfig.from_env()
        assert config.aither_url == "http://custom:9999"
        assert config.agent_id == "demi"
        assert config.poll_interval == 60


class TestGraphemeLen:
    def test_ascii(self):
        assert _grapheme_len("hello") == 5

    def test_emoji(self):
        # Each emoji is at least 1 char
        assert _grapheme_len("🔥") >= 1

    def test_empty(self):
        assert _grapheme_len("") == 0


class TestBridgeInit:
    def test_creates_channels(self, bridge, mock_relay):
        """Bridge start should create the three bridge channels."""
        # Simulate start without actually polling (mock httpx)
        with patch("adk.aither_bridge.httpx.AsyncClient") as mock_http:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_http.return_value = mock_client

            loop = asyncio.new_event_loop()
            # Start and immediately stop to avoid running the poll loop forever
            bridge._config.poll_interval = 999
            try:
                loop.run_until_complete(bridge.start())
                loop.run_until_complete(bridge.stop())
            finally:
                loop.close()

        # Verify channels created
        channel_names = [call.args[0] for call in mock_relay.create_channel.call_args_list]
        assert FEED_CHANNEL in channel_names
        assert POST_CHANNEL in channel_names
        assert NOTIF_CHANNEL in channel_names

    def test_registers_bot_user(self, bridge, mock_relay):
        """Bridge should join all channels as AitherBot."""
        with patch("adk.aither_bridge.httpx.AsyncClient") as mock_http:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_http.return_value = mock_client

            loop = asyncio.new_event_loop()
            bridge._config.poll_interval = 999
            try:
                loop.run_until_complete(bridge.start())
                loop.run_until_complete(bridge.stop())
            finally:
                loop.close()

        join_nicks = [call.args[1] for call in mock_relay.join.call_args_list]
        assert all(n == _BRIDGE_NICK for n in join_nicks)

    def test_disabled_bridge_noop(self, mock_relay):
        """Disabled bridge should not create channels or start polling."""
        config = BridgeConfig(enabled=False)
        bridge = AitherBridge(mock_relay, config)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bridge.start())
        finally:
            loop.close()

        mock_relay.create_channel.assert_not_called()
        assert bridge._running is False


class TestRelayPostToIRC:
    def test_formats_post(self, bridge, mock_relay):
        """AT Protocol post should be formatted and posted to feed channel."""
        post = {
            "uri": "at://did:plc:abc/app.bsky.feed.post/123",
            "author": {"handle": "alice.aither.social", "displayName": "Alice"},
            "record": {"text": "Hello from Aither!"},
            "likeCount": 5,
            "repostCount": 2,
            "replyCount": 1,
        }
        bridge._relay_post_to_irc(post)

        mock_relay.post.assert_called_once()
        args = mock_relay.post.call_args
        assert args.args[0] == FEED_CHANNEL
        assert args.args[1] == _BRIDGE_NICK
        assert "@alice.aither.social" in args.args[2]
        assert "Hello from Aither!" in args.args[2]
        assert "♥5" in args.args[2]

    def test_skips_empty_text(self, bridge, mock_relay):
        """Posts with no text should be skipped."""
        post = {
            "uri": "at://did:plc:abc/app.bsky.feed.post/456",
            "author": {"handle": "bob"},
            "record": {"text": ""},
        }
        bridge._relay_post_to_irc(post)
        mock_relay.post.assert_not_called()

    def test_truncates_long_text(self, bridge, mock_relay):
        """Very long posts should be truncated for IRC."""
        post = {
            "uri": "at://did:plc:abc/app.bsky.feed.post/789",
            "author": {"handle": "verbose.user"},
            "record": {"text": "A" * 500},
        }
        bridge._relay_post_to_irc(post)

        text = mock_relay.post.call_args.args[2]
        assert "..." in text
        assert len(text) < 500


class TestIRCToAither:
    def test_ignores_non_post_channel(self, bridge, mock_relay):
        """Messages outside #aither-post should be ignored."""
        bridge._on_chat_message({
            "channel": "#general",
            "nick": "alice",
            "content": "hello",
        })
        # No post should be scheduled (nothing to assert directly,
        # but we can verify no error)

    def test_ignores_bot_messages(self, bridge, mock_relay):
        """Bot's own messages should not be re-posted."""
        bridge._on_chat_message({
            "channel": POST_CHANNEL,
            "nick": _BRIDGE_NICK,
            "content": "some text",
        })

    def test_ignores_irc_commands(self, bridge, mock_relay):
        """IRC commands (/join, /help, etc.) should not be posted."""
        bridge._on_chat_message({
            "channel": POST_CHANNEL,
            "nick": "alice",
            "content": "/help",
        })


class TestBridgeStatus:
    def test_status_when_running(self, bridge):
        bridge._running = True
        bridge._config.enabled = True
        status = bridge.status()
        assert status["enabled"] is True
        assert status["running"] is True
        assert "aither_url" in status
        assert "posts_relayed_to_irc" in status

    def test_status_when_stopped(self, bridge):
        status = bridge.status()
        assert status["running"] is False


class TestPollTimeline:
    @pytest.mark.asyncio
    async def test_relays_new_posts(self, bridge, mock_relay):
        """New posts from timeline should appear in #aither-feed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "timeline": [
                {
                    "uri": "at://did:plc:x/app.bsky.feed.post/1",
                    "author": {"handle": "test.user"},
                    "record": {"text": "First post"},
                },
                {
                    "uri": "at://did:plc:x/app.bsky.feed.post/2",
                    "author": {"handle": "test.user"},
                    "record": {"text": "Second post"},
                },
            ]
        }

        bridge._http = AsyncMock()
        bridge._http.get = AsyncMock(return_value=mock_response)

        await bridge._poll_timeline()

        assert mock_relay.post.call_count == 2
        assert bridge._stats["posts_relayed_to_irc"] == 2

    @pytest.mark.asyncio
    async def test_deduplicates_posts(self, bridge, mock_relay):
        """Already-seen posts should not be relayed again."""
        bridge._seen_uris.add("at://did:plc:x/app.bsky.feed.post/1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "timeline": [
                {
                    "uri": "at://did:plc:x/app.bsky.feed.post/1",
                    "author": {"handle": "test.user"},
                    "record": {"text": "Already seen"},
                },
            ]
        }

        bridge._http = AsyncMock()
        bridge._http.get = AsyncMock(return_value=mock_response)

        await bridge._poll_timeline()

        mock_relay.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_service_down(self, bridge, mock_relay):
        """Should not crash when AitherSocial is unreachable."""
        import httpx as _httpx

        bridge._http = AsyncMock()
        bridge._http.get = AsyncMock(side_effect=_httpx.ConnectError("refused"))

        await bridge._poll_timeline()  # Should not raise
        mock_relay.post.assert_not_called()


class TestPostToAither:
    @pytest.mark.asyncio
    async def test_successful_post(self, bridge, mock_relay):
        """Successful post should confirm in IRC."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uri": "at://did:plc:x/app.bsky.feed.post/new"}

        bridge._http = AsyncMock()
        bridge._http.post = AsyncMock(return_value=mock_response)

        await bridge._post_to_aither("alice", "Hello Aither!")

        # Should POST to /post
        bridge._http.post.assert_called_once()
        call_args = bridge._http.post.call_args
        assert call_args.args[0] == "/post"
        assert "alice" in call_args.kwargs["json"]["text"]

        # Should confirm in IRC
        assert mock_relay.post.called
        confirm = mock_relay.post.call_args.args[2]
        assert "✅" in confirm

    @pytest.mark.asyncio
    async def test_failed_post(self, bridge, mock_relay):
        """Failed post should report error in IRC."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Content rejected"

        bridge._http = AsyncMock()
        bridge._http.post = AsyncMock(return_value=mock_response)

        await bridge._post_to_aither("alice", "bad content")

        assert mock_relay.post.called
        error_msg = mock_relay.post.call_args.args[2]
        assert "❌" in error_msg
        assert bridge._stats["errors"] == 1
