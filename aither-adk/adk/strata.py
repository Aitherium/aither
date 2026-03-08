"""Fire-and-forget session ingest to AitherOS Strata.

When an ADK server runs alongside or federated with a full AitherOS (Elysium)
instance, this module sends chat session data to Strata for training harvest,
analytics, and the dark factory learning loop.

Opt-in: only active when AITHER_STRATA_URL is set (or Strata is reachable at
the default localhost:8136). Follows the same privacy-centric pattern as
phonehome.py — never blocks the chat response, queues offline if Strata is
unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger("adk.strata")

# Default Strata URL — only used if AITHER_STRATA_URL is set or localhost is reachable
_DEFAULT_STRATA_URL = "http://localhost:8136"
_QUEUE_MAX_LINES = 2000


class StrataIngest:
    """Fire-and-forget ingest client for AitherOS Strata."""

    def __init__(
        self,
        strata_url: str = "",
        data_dir: str | Path | None = None,
    ):
        self.strata_url = (
            strata_url
            or os.getenv("AITHER_STRATA_URL", "")
        ).rstrip("/")
        self._data_dir = Path(
            data_dir or os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither"))
        )
        self._queue_path = self._data_dir / "strata_queue.jsonl"
        self._enabled: bool | None = None  # lazy — checked on first send

    @property
    def enabled(self) -> bool:
        """Check if Strata ingest is active (URL configured)."""
        if self._enabled is not None:
            return self._enabled
        self._enabled = bool(self.strata_url)
        return self._enabled

    async def ingest_chat(
        self,
        *,
        agent: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        model: str = "",
        tokens_used: int = 0,
        latency_ms: int = 0,
        tool_calls: list[str] | None = None,
    ) -> bool:
        """Send a chat exchange to Strata. Returns True if sent successfully.

        This is fire-and-forget — never raises, never blocks the chat response.
        """
        if not self.enabled:
            return False

        payload = {
            "source": "adk",
            "type": "chat_exchange",
            "timestamp": time.time(),
            "agent": agent,
            "session_id": session_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "model": model,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "tool_calls": tool_calls or [],
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.strata_url}/api/v1/ingest/adk-session",
                    json=payload,
                )
                if resp.status_code < 300:
                    return True
                logger.debug("Strata ingest returned %d", resp.status_code)
        except Exception as exc:
            logger.debug("Strata ingest failed, queuing offline: %s", exc)

        self._queue_offline(payload)
        return False

    async def ingest_session_end(
        self,
        *,
        agent: str,
        session_id: str,
        message_count: int = 0,
        total_tokens: int = 0,
        duration_seconds: float = 0.0,
    ) -> bool:
        """Notify Strata that a session has ended."""
        if not self.enabled:
            return False

        payload = {
            "source": "adk",
            "type": "session_end",
            "timestamp": time.time(),
            "agent": agent,
            "session_id": session_id,
            "message_count": message_count,
            "total_tokens": total_tokens,
            "duration_seconds": duration_seconds,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.strata_url}/api/v1/ingest/adk-session",
                    json=payload,
                )
                return resp.status_code < 300
        except Exception:
            self._queue_offline(payload)
            return False

    def _queue_offline(self, payload: dict):
        """Write to JSONL disk queue for retry when Strata comes back."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._queue_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
            _cap_jsonl(self._queue_path, _QUEUE_MAX_LINES)
        except Exception as exc:
            logger.debug("Failed to queue Strata data: %s", exc)

    async def flush_queue(self) -> int:
        """Try to send queued entries to Strata. Returns count of successfully sent."""
        if not self._queue_path.exists() or not self.enabled:
            return 0

        try:
            lines = self._queue_path.read_text().strip().split("\n")
        except Exception:
            return 0

        if not lines or lines == [""]:
            return 0

        sent = 0
        remaining = []

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for line in lines:
                    try:
                        payload = json.loads(line)
                        resp = await client.post(
                            f"{self.strata_url}/api/v1/ingest/adk-session",
                            json=payload,
                        )
                        if resp.status_code < 300:
                            sent += 1
                        else:
                            remaining.append(line)
                    except Exception:
                        remaining.append(line)
        except Exception:
            remaining = lines

        if remaining:
            self._queue_path.write_text("\n".join(remaining) + "\n")
        else:
            self._queue_path.unlink(missing_ok=True)

        return sent


# Module-level singleton
_instance: StrataIngest | None = None


def get_strata_ingest() -> StrataIngest:
    """Get or create the module-level StrataIngest singleton."""
    global _instance
    if _instance is None:
        _instance = StrataIngest()
    return _instance


def _cap_jsonl(path: Path, max_lines: int):
    """Cap a JSONL file at max_lines, keeping the newest."""
    try:
        lines = path.read_text().strip().split("\n")
        if len(lines) > max_lines:
            path.write_text("\n".join(lines[-max_lines:]) + "\n")
    except Exception:
        pass
