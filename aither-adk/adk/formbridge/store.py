"""CaptureStore — local SQLite store for captured form-field values.

One row per (patient_key, field_path), upserted as the user re-edits fields.
Lives under ~/.aither/formbridge/ (override with AITHER_FORMBRIDGE_DIR).
Auto-purge: records older than AITHER_FORMBRIDGE_TTL_DAYS (default 7) are
deleted opportunistically on every write/read entry point — the store is a
transient working set, not a system of record (the filled PDFs are).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("adk.formbridge.store")

_DEFAULT_TTL_DAYS = 7.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    patient_key   TEXT NOT NULL,
    field_path    TEXT NOT NULL,
    value         TEXT NOT NULL,
    display_name  TEXT NOT NULL DEFAULT '',
    source_origin TEXT NOT NULL DEFAULT '',
    captured_at   REAL NOT NULL,
    PRIMARY KEY (patient_key, field_path)
);
CREATE INDEX IF NOT EXISTS idx_captures_time ON captures (captured_at);
"""


def _store_dir() -> Path:
    override = os.getenv("AITHER_FORMBRIDGE_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".aither" / "formbridge"


def _ttl_seconds() -> float:
    try:
        days = float(os.getenv("AITHER_FORMBRIDGE_TTL_DAYS", str(_DEFAULT_TTL_DAYS)))
    except ValueError:
        days = _DEFAULT_TTL_DAYS
    return max(days, 0.0) * 86400.0


class CaptureStore:
    """Thread-safe SQLite-backed store of captured field values."""

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            d = _store_dir()
            d.mkdir(parents=True, exist_ok=True)
            db_path = d / "captures.db"
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Writes ──

    def ingest_batch(
        self,
        patient_key: str,
        fields: list[dict],
        *,
        display_name: str = "",
        source_origin: str = "",
    ) -> int:
        """Upsert a batch of captured fields for one patient. Returns rows written.

        Each field dict needs ``path`` (the mapping `as:` name or the raw
        selector/name from the capture script) and ``value``. Empty values are
        stored too — clearing a field in the EHR clears it here.
        """
        if not patient_key:
            raise ValueError("patient_key is required (anchor-gated capture)")
        now = time.time()
        rows = [
            (patient_key, str(f["path"]), str(f.get("value", "")), display_name, source_origin, now)
            for f in fields
            if f.get("path")
        ]
        if not rows:
            return 0
        with self._lock, self._connect() as conn:
            self._purge_expired(conn)
            conn.executemany(
                "INSERT INTO captures (patient_key, field_path, value, display_name, source_origin, captured_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(patient_key, field_path) DO UPDATE SET "
                "value=excluded.value, display_name=excluded.display_name, "
                "source_origin=excluded.source_origin, captured_at=excluded.captured_at",
                rows,
            )
        return len(rows)

    # ── Reads ──

    def list_patients(self) -> list[dict]:
        """Summaries only — display name, field count, last capture. No values."""
        with self._lock, self._connect() as conn:
            self._purge_expired(conn)
            cur = conn.execute(
                "SELECT patient_key, MAX(display_name) AS display_name, "
                "COUNT(*) AS field_count, MAX(captured_at) AS last_capture "
                "FROM captures GROUP BY patient_key ORDER BY last_capture DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_record(self, patient_key: str) -> dict[str, str]:
        """Full field→value record for one patient (LOCAL consumers only —
        never serialize this into an MCP tool result or any remote payload)."""
        with self._lock, self._connect() as conn:
            self._purge_expired(conn)
            cur = conn.execute(
                "SELECT field_path, value FROM captures WHERE patient_key = ?",
                (patient_key,),
            )
            return {r["field_path"]: r["value"] for r in cur.fetchall()}

    # ── Purge ──

    def purge(self, patient_key: str | None = None) -> int:
        """Delete one patient's record, or everything when patient_key is None."""
        with self._lock, self._connect() as conn:
            if patient_key:
                cur = conn.execute("DELETE FROM captures WHERE patient_key = ?", (patient_key,))
            else:
                cur = conn.execute("DELETE FROM captures")
            return cur.rowcount

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        ttl = _ttl_seconds()
        if ttl <= 0:
            return
        cutoff = time.time() - ttl
        cur = conn.execute("DELETE FROM captures WHERE captured_at < ?", (cutoff,))
        if cur.rowcount:
            logger.info("formbridge: auto-purged %d expired captures", cur.rowcount)

    def stats(self) -> dict:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(DISTINCT patient_key) AS patients, COUNT(*) AS fields FROM captures"
            )
            row = cur.fetchone()
            return {"patients": row["patients"], "fields": row["fields"], "db_path": self._db_path}


_store: CaptureStore | None = None
_store_lock = threading.Lock()


def get_store() -> CaptureStore:
    """Process-wide CaptureStore singleton (path from env at first use)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = CaptureStore()
        return _store


def reset_store() -> None:
    """Test helper — drop the singleton so the next get_store() re-reads env."""
    global _store
    with _store_lock:
        _store = None
