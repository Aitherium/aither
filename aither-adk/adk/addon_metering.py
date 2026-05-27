"""Client-side addon usage metering.

Records usage events locally, batches them to the hub via
``FederationLiteClient.report_addon_usage()``.  When the hub is
unreachable, events queue to a JSONL file (max 10K events, 7-day
retention) and flush on reconnect.

Usage::

    meter = AddonMeter()
    meter.record("knowledge-rag", "rag_query_completed", {"tokens": 150})
    await meter.flush(federation_client)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("adk.addon_metering")

_MAX_QUEUED_EVENTS = 10_000
_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


def _queue_path() -> Path:
    return Path.home() / ".aitheros" / "addons" / "usage_queue.jsonl"


class AddonMeter:
    """Client-side usage metering with offline queuing."""

    def __init__(self) -> None:
        self._pending: List[Dict[str, Any]] = []
        self._load_queue()

    def _load_queue(self) -> None:
        """Load persisted events from JSONL queue."""
        qp = _queue_path()
        if not qp.is_file():
            return
        cutoff = time.time() - _MAX_AGE_SECONDS
        try:
            lines = qp.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                try:
                    evt = json.loads(line)
                    if evt.get("timestamp", 0) >= cutoff:
                        self._pending.append(evt)
                except json.JSONDecodeError:
                    continue
            # Trim to max
            if len(self._pending) > _MAX_QUEUED_EVENTS:
                self._pending = self._pending[-_MAX_QUEUED_EVENTS:]
        except Exception as e:
            log.warning("Failed to load usage queue: %s", e)

    def _save_queue(self) -> None:
        """Persist pending events to JSONL."""
        qp = _queue_path()
        qp.parent.mkdir(parents=True, exist_ok=True)
        # Trim old events
        cutoff = time.time() - _MAX_AGE_SECONDS
        self._pending = [e for e in self._pending if e.get("timestamp", 0) >= cutoff]
        if len(self._pending) > _MAX_QUEUED_EVENTS:
            self._pending = self._pending[-_MAX_QUEUED_EVENTS:]
        with open(qp, "w", encoding="utf-8") as f:
            for evt in self._pending:
                f.write(json.dumps(evt) + "\n")

    def record(
        self,
        addon_id: str,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a usage event locally."""
        evt = {
            "addon_id": addon_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data or {},
        }
        self._pending.append(evt)
        # Auto-persist every 100 events
        if len(self._pending) % 100 == 0:
            self._save_queue()

    async def flush(self, federation_client=None) -> int:
        """Flush pending events to hub. Returns count of events flushed.

        If hub is unreachable, events remain queued for later flush.
        """
        if not self._pending:
            return 0

        if not federation_client:
            self._save_queue()
            return 0

        # Group events by addon
        by_addon: Dict[str, List[Dict[str, Any]]] = {}
        for evt in self._pending:
            aid = evt.get("addon_id", "unknown")
            by_addon.setdefault(aid, []).append(evt)

        flushed = 0
        remaining: List[Dict[str, Any]] = []

        for addon_id, events in by_addon.items():
            try:
                result = await federation_client.report_addon_usage(addon_id, events)
                if not result.get("error"):
                    flushed += len(events)
                    log.info("Flushed %d events for addon %s", len(events), addon_id)
                else:
                    remaining.extend(events)
                    log.warning("Hub rejected events for %s: %s", addon_id, result)
            except Exception as e:
                remaining.extend(events)
                log.warning("Failed to flush events for %s: %s", addon_id, e)

        self._pending = remaining
        self._save_queue()
        return flushed

    @property
    def pending_count(self) -> int:
        return len(self._pending)
