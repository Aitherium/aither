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

-- Documents: every form we fill / summary we generate for a patient, kept as a
-- per-patient history (a patient accrues many docs over their care). Unlike
-- `captures` (a transient working set with a TTL), documents are durable — they
-- are the system of record — so they are NOT auto-expired. Metadata only; the
-- bytes live in the output dir (filename is the opaque job id, never PHI).
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    patient_key   TEXT NOT NULL,
    form_id       TEXT NOT NULL DEFAULT '',
    pack_id       TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL DEFAULT 'form',     -- form | summary
    filename      TEXT NOT NULL DEFAULT '',
    path          TEXT NOT NULL DEFAULT '',
    encrypted     INTEGER NOT NULL DEFAULT 0,
    password_ref  TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_patient ON documents (patient_key, created_at);
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


def _encrypt_value(value: str) -> str:
    """Encrypt PHI at rest when the office vault is set up (raises if locked —
    the routes gate writes on an unlocked vault). Plaintext when no vault."""
    from adk.formbridge.vault import get_vault

    v = get_vault()
    return v.encrypt(value) if v.initialized else value


def _decrypt_value(value: str) -> str:
    """Decrypt an at-rest value. Unencrypted (pre-vault / dev) values pass
    through unchanged, so enabling the vault never strands existing data."""
    from adk.formbridge.vault import get_vault

    v = get_vault()
    return v.decrypt(value) if v.is_encrypted(value) else value


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
        # Encrypt PHI at rest: the field values and the patient display name.
        # patient_key (the anchor id / PK) and source_origin (the EHR URL) stay
        # plaintext — they are identifiers/routing, not record content.
        enc_display = _encrypt_value(display_name) if display_name else display_name
        rows = [
            (patient_key, str(f["path"]), _encrypt_value(str(f.get("value", ""))),
             enc_display, source_origin, now)
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

    # ── Documents (per-patient history of filled forms / summaries) ──

    def add_document(
        self,
        patient_key: str,
        *,
        doc_id: str,
        form_id: str = "",
        pack_id: str = "",
        kind: str = "form",
        filename: str = "",
        path: str = "",
        encrypted: bool = False,
        password_ref: str = "",
    ) -> str:
        """Record one filled form / generated summary under a patient. doc_id is
        the output job id (also the download handle). Durable — not TTL-purged."""
        if not patient_key or not doc_id:
            raise ValueError("patient_key and doc_id are required")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, patient_key, form_id, pack_id, kind, filename, path, "
                " encrypted, password_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, patient_key, form_id, pack_id, kind, filename, path,
                 1 if encrypted else 0, password_ref, time.time()),
            )
        return doc_id

    def list_documents(self, patient_key: str) -> list[dict]:
        """All documents for one patient, newest first. Metadata only."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT doc_id, form_id, pack_id, kind, filename, path, encrypted, "
                "password_ref, created_at FROM documents WHERE patient_key = ? "
                "ORDER BY created_at DESC",
                (patient_key,),
            )
            out = []
            for r in cur.fetchall():
                d = dict(r)
                d["encrypted"] = bool(d["encrypted"])
                out.append(d)
            return out

    # ── Reads ──

    def list_patients(self) -> list[dict]:
        """Summaries only — display name, field count, doc count, last activity.
        Includes patients that have documents even if their captures expired (the
        documents are the durable record), so a patient never silently vanishes."""
        with self._lock, self._connect() as conn:
            self._purge_expired(conn)
            agg: dict[str, dict] = {}
            for r in conn.execute(
                "SELECT patient_key, MAX(display_name) AS display_name, "
                "COUNT(*) AS field_count, MAX(captured_at) AS last_capture "
                "FROM captures GROUP BY patient_key"
            ).fetchall():
                agg[r["patient_key"]] = {
                    "patient_key": r["patient_key"],
                    "display_name": _decrypt_value(r["display_name"] or ""),
                    "field_count": r["field_count"],
                    "doc_count": 0,
                    "last_capture": r["last_capture"] or 0.0,
                }
            for r in conn.execute(
                "SELECT patient_key, COUNT(*) AS doc_count, MAX(created_at) AS last_doc "
                "FROM documents GROUP BY patient_key"
            ).fetchall():
                d = agg.setdefault(r["patient_key"], {
                    "patient_key": r["patient_key"], "display_name": "",
                    "field_count": 0, "doc_count": 0, "last_capture": 0.0,
                })
                d["doc_count"] = r["doc_count"]
                d["last_capture"] = max(d["last_capture"] or 0.0, r["last_doc"] or 0.0)
            return sorted(agg.values(), key=lambda x: x["last_capture"] or 0.0, reverse=True)

    def get_record(self, patient_key: str) -> dict[str, str]:
        """Full field→value record for one patient (LOCAL consumers only —
        never serialize this into an MCP tool result or any remote payload).
        Excludes internal sentinel fields like '_encounter'."""
        with self._lock, self._connect() as conn:
            self._purge_expired(conn)
            cur = conn.execute(
                "SELECT field_path, value FROM captures WHERE patient_key = ?",
                (patient_key,),
            )
            return {
                r["field_path"]: _decrypt_value(r["value"])
                for r in cur.fetchall()
                if not r["field_path"].startswith("_")
            }

    def delete_fields(self, patient_key: str, field_paths: list[str]) -> int:
        """Delete specific fields for a patient. Returns number of rows deleted."""
        if not patient_key or not field_paths:
            return 0
        with self._lock, self._connect() as conn:
            deleted = 0
            for path in field_paths:
                cur = conn.execute(
                    "DELETE FROM captures WHERE patient_key = ? AND field_path = ?",
                    (patient_key, path),
                )
                deleted += cur.rowcount
            return deleted

    # ── Purge ──

    def purge(self, patient_key: str | None = None) -> int:
        """Delete one patient's captures + document rows (PHI cleanup), or
        everything when patient_key is None. Returns captures deleted. The route
        unlinks the document FILES first (it owns the filesystem)."""
        with self._lock, self._connect() as conn:
            if patient_key:
                cur = conn.execute("DELETE FROM captures WHERE patient_key = ?", (patient_key,))
                conn.execute("DELETE FROM documents WHERE patient_key = ?", (patient_key,))
            else:
                cur = conn.execute("DELETE FROM captures")
                conn.execute("DELETE FROM documents")
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
