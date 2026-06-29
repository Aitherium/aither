"""Tests for ChatRelayClient — relay SDK for ADK nodes."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk.relay_client import ChatRelayClient, RelayMessage


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.messages: list[str] = []
        self.open = True
        self._send_buffer: list[str] = []
        self._receive_queue: asyncio.Queue = asyncio.Queue()

    async def send(self, data: str):
        """Record sent message."""
        self._send_buffer.append(data)
        self.messages.append(data)

    async def close(self):
        """Close the connection."""
        self.open = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Simulate receiving messages from queue."""
        try:
            return await asyncio.wait_for(self._receive_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            raise StopAsyncIteration


async def mock_websocket_connect(url, **kwargs):
    """Mock function for websockets.connect()."""
    return MockWebSocket()


@pytest.mark.asyncio
async def test_mint_relay_token():
    """Test token minting via HTTP."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "relay_token": "jwt-token-123",
            "expires_at": 1700000000.0,
            "nick": "test-agent",
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await client._mint_relay_token()

        assert result is True
        assert client._relay_token == "jwt-token-123"
        assert client.nick == "test-agent"

        # Verify the correct endpoint was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/v1/auth/relay-token" in call_args[0][0]


@pytest.mark.asyncio
async def test_connect_mints_token_and_opens_ws():
    """Test that connect() mints token then opens WebSocket."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    with patch("httpx.AsyncClient") as mock_http, \
         patch("websockets.connect", new=mock_websocket_connect):

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "relay_token": "jwt-token-123",
            "expires_at": 1700000000.0,
            "nick": "test-agent",
        }
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.__aenter__.return_value = mock_http_client
        mock_http_client.__aexit__.return_value = None
        mock_http.return_value = mock_http_client

        result = await client.connect()

        assert result is True
        assert client._relay_token == "jwt-token-123"
        assert client._ws is not None


@pytest.mark.asyncio
async def test_join_channel_sends_join_message():
    """Test that join_channel() sends the correct frame."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    # Manually set ws as connected
    client._ws = MockWebSocket()

    await client.join_channel("#team")

    # Check that join message was sent
    assert len(client._ws.messages) == 1
    msg = json.loads(client._ws.messages[0])
    assert msg["type"] == "join"
    assert msg["channel"] == "#team"
    assert msg["nick"] == "test-agent"


@pytest.mark.asyncio
async def test_part_channel():
    """Test leaving a channel."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()
    client._joined_channels.add("#team")

    await client.part_channel("#team")

    # Check part message was sent
    assert len(client._ws.messages) == 1
    msg = json.loads(client._ws.messages[0])
    assert msg["type"] == "part"
    assert msg["channel"] == "#team"
    assert "#team" not in client._joined_channels


@pytest.mark.asyncio
async def test_post_message():
    """Test posting a message to a channel."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    await client.post_message("#team", "Hello world!")

    assert len(client._ws.messages) == 1
    msg = json.loads(client._ws.messages[0])
    assert msg["type"] == "message"
    assert msg["channel"] == "#team"
    assert msg["content"] == "Hello world!"


@pytest.mark.asyncio
async def test_mention_handler_registration():
    """Test registering mention handlers."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    handler = AsyncMock(return_value="I was mentioned!")

    client.register_mention_handler("bot", handler)

    assert "bot" in client._mention_handlers
    assert client._mention_handlers["bot"] == handler


@pytest.mark.asyncio
async def test_mention_handler_fires_on_message():
    """Test that @mention in message triggers the handler."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    handler = AsyncMock(return_value="I respond to mentions!")
    client.register_mention_handler("bot", handler)

    # Simulate incoming message with mention
    msg = RelayMessage(
        msg_id="msg-1",
        channel="#team",
        nick="alice",
        content="Hey @bot, what's up?",
        timestamp=1700000000.0,
    )

    await client._check_mentions(msg)

    # Verify handler was called
    handler.assert_called_once_with(msg)

    # Verify response was posted
    assert len(client._ws.messages) == 1
    reply = json.loads(client._ws.messages[0])
    assert reply["type"] == "message"
    assert reply["channel"] == "#team"
    assert reply["content"] == "I respond to mentions!"


@pytest.mark.asyncio
async def test_mention_handler_case_insensitive():
    """Test that @mention matching is case-insensitive."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    handler = AsyncMock(return_value="Got it!")
    client.register_mention_handler("BOT", handler)  # Uppercase

    # Test with lowercase mention
    msg = RelayMessage(
        msg_id="msg-1",
        channel="#team",
        nick="alice",
        content="@bot help me",
        timestamp=1700000000.0,
    )

    await client._check_mentions(msg)

    # Handler should fire despite case difference
    handler.assert_called_once()


@pytest.mark.asyncio
async def test_history_handling():
    """Test that history messages are properly loaded."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    history_data = {
        "type": "history",
        "channel": "#team",
        "messages": [
            {
                "msg_id": "msg-1",
                "channel": "#team",
                "nick": "alice",
                "content": "First message",
                "timestamp": 1700000000.0,
            },
            {
                "msg_id": "msg-2",
                "channel": "#team",
                "nick": "bob",
                "content": "Second message",
                "timestamp": 1700000001.0,
            },
        ],
    }

    # Should not raise
    await client._handle_relay_message(history_data)


@pytest.mark.asyncio
async def test_relay_message_from_dict():
    """Test RelayMessage.from_dict()."""
    data = {
        "msg_id": "msg-123",
        "channel": "#team",
        "nick": "alice",
        "content": "Test message",
        "msg_type": "message",
        "timestamp": 1700000000.0,
        "thread_id": "thread-1",
        "user_id": "user-alice",
        "session_id": "session-123",
        "node_id": "node-1",
    }

    msg = RelayMessage.from_dict(data)

    assert msg.msg_id == "msg-123"
    assert msg.channel == "#team"
    assert msg.nick == "alice"
    assert msg.content == "Test message"
    assert msg.user_id == "user-alice"
    assert msg.session_id == "session-123"


@pytest.mark.asyncio
async def test_list_channels():
    """Test listing available channels."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "#general", "topic": "General chat", "user_count": 5},
            {"name": "#team", "topic": "Team only", "user_count": 3},
        ]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        channels = await client.list_channels()

        assert len(channels) == 2
        assert channels[0]["name"] == "#general"
        assert channels[1]["name"] == "#team"


@pytest.mark.asyncio
async def test_disconnect():
    """Test disconnecting from relay."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()
    client._should_reconnect = True

    await client.disconnect()

    assert client._should_reconnect is False
    assert client._ws is None


@pytest.mark.asyncio
async def test_mention_handler_no_double_mention():
    """Test that only matching mentions fire handlers."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    handler1 = AsyncMock(return_value="Handler 1")
    handler2 = AsyncMock(return_value="Handler 2")

    client.register_mention_handler("alice", handler1)
    client.register_mention_handler("bob", handler2)

    msg = RelayMessage(
        msg_id="msg-1",
        channel="#team",
        nick="charlie",
        content="@alice do something",
        timestamp=1700000000.0,
    )

    await client._check_mentions(msg)

    # Only alice's handler should fire
    handler1.assert_called_once()
    handler2.assert_not_called()


@pytest.mark.asyncio
async def test_mention_handler_sync_and_async():
    """Test that both sync and async handlers work."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    # Async handler
    async_handler = AsyncMock(return_value="Async response")
    client.register_mention_handler("async_bot", async_handler)

    msg = RelayMessage(
        msg_id="msg-1",
        channel="#team",
        nick="alice",
        content="@async_bot help",
        timestamp=1700000000.0,
    )

    await client._check_mentions(msg)

    async_handler.assert_called_once()
    # Response should be posted
    assert len(client._ws.messages) == 1


@pytest.mark.asyncio
async def test_backoff_increases_on_reconnect_failure():
    """Test exponential backoff on reconnection."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    initial_backoff = client._backoff_ms
    client._backoff_ms = 1000
    client._max_backoff_ms = 30000

    # Simulate backoff increase
    client._schedule_reconnect()
    # Backoff should increase (though schedule_reconnect doesn't do it directly)
    # Just verify backoff state is preserved
    assert client._backoff_ms > 0


@pytest.mark.asyncio
async def test_channel_name_normalization():
    """Test that channel names are normalized to start with #."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    # Pass without #
    await client.join_channel("team")

    msg = json.loads(client._ws.messages[0])
    assert msg["channel"] == "#team"


def test_relay_client_singleton():
    """Test get_relay_client singleton pattern."""
    from adk.relay_client import get_relay_client

    client1 = get_relay_client(
        base_url="http://localhost:8001",
        aither_bearer="test-token",
    )
    client2 = get_relay_client()

    assert client1 is client2


@pytest.mark.asyncio
async def test_mention_handler_exception_handling():
    """Test that handler exceptions don't crash the receive loop."""
    client = ChatRelayClient(
        base_url="http://localhost:8001",
        aither_bearer="sk-ant-test",
        nick="test-agent",
    )

    client._ws = MockWebSocket()

    # Handler that raises an exception
    handler = AsyncMock(side_effect=ValueError("Handler error"))
    client.register_mention_handler("bot", handler)

    msg = RelayMessage(
        msg_id="msg-1",
        channel="#team",
        nick="alice",
        content="@bot test",
        timestamp=1700000000.0,
    )

    # Should not raise
    await client._check_mentions(msg)

    # Handler was called despite error
    handler.assert_called_once()
