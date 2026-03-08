"""Local SQLite memory — conversations, KV store, search."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("adk.memory")


@dataclass
class MemoryEntry:
    """A stored memory entry."""
    key: str
    value: str
    category: str = "general"
    timestamp: float = 0.0
    metadata: dict | None = None


class Memory:
    """Local SQLite-backed memory store for agents.

    Provides:
    - Key-value storage (remember/recall)
    - Conversation history
    - Simple text search
    """

    def __init__(self, db_path: str | Path | None = None, agent_name: str = "default"):
        if db_path is None:
            data_dir = Path.home() / ".aither" / "memory"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / f"{agent_name}.db"

        self._db_path = str(db_path)
        self._agent = agent_name
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    timestamp REAL,
                    metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_kv_category ON kv_store(category);
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def remember(self, key: str, value: str, category: str = "general", metadata: dict | None = None):
        """Store a key-value pair in memory."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, category, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (key, value, category, time.time(), json.dumps(metadata) if metadata else None),
            )

    async def recall(self, key: str) -> str | None:
        """Retrieve a value by key."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search memory entries by substring match."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, category, timestamp, metadata FROM kv_store "
                "WHERE key LIKE ? OR value LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [
            MemoryEntry(
                key=r[0], value=r[1], category=r[2], timestamp=r[3],
                metadata=json.loads(r[4]) if r[4] else None,
            )
            for r in rows
        ]

    async def forget(self, key: str):
        """Remove a key from memory."""
        with self._connect() as conn:
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))

    async def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None):
        """Add a message to conversation history."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, time.time(), json.dumps(metadata) if metadata else None),
            )

    async def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get conversation history for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM conversations "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    async def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))

    async def list_keys(self, category: str | None = None) -> list[str]:
        """List all stored keys, optionally filtered by category."""
        with self._connect() as conn:
            if category:
                rows = conn.execute("SELECT key FROM kv_store WHERE category = ?", (category,)).fetchall()
            else:
                rows = conn.execute("SELECT key FROM kv_store").fetchall()
        return [r[0] for r in rows]

    # ─────────────────────────────────────────────────────────────────────
    # SESSION REPAIR
    # ─────────────────────────────────────────────────────────────────────

    async def repair_session(self, session_id: str) -> dict:
        """Repair a conversation session in the SQLite store.

        Phases:
          1. Remove messages with null/empty content
          2. Fix timestamp monotonicity
          3. Remove orphan tool results
          4. Deduplicate exact-content messages within 1s window
          5. Run SQLite integrity check
        """
        issues = 0
        fixed = 0

        with self._connect() as conn:
            # Phase 1: Remove empty content
            result = conn.execute(
                "DELETE FROM conversations WHERE session_id = ? AND (content IS NULL OR content = '')",
                (session_id,),
            )
            removed = result.rowcount
            if removed:
                issues += removed
                fixed += removed

            # Phase 2: Check timestamp monotonicity
            rows = conn.execute(
                "SELECT id, timestamp FROM conversations WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            last_ts = 0.0
            for row_id, ts in rows:
                if ts is not None and ts < last_ts:
                    issues += 1
                    conn.execute(
                        "UPDATE conversations SET timestamp = ? WHERE id = ?",
                        (last_ts + 0.001, row_id),
                    )
                    fixed += 1
                if ts is not None:
                    last_ts = max(last_ts, ts)

            # Phase 3: Deduplicate (same role + content within 1s)
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM conversations "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            seen: dict[str, float] = {}
            dupes_to_remove = []
            for row_id, role, content, ts in rows:
                key = f"{role}:{content}"
                prev_ts = seen.get(key)
                if prev_ts is not None and ts is not None and abs(ts - prev_ts) < 1.0:
                    dupes_to_remove.append(row_id)
                    issues += 1
                seen[key] = ts or 0.0

            for row_id in dupes_to_remove:
                conn.execute("DELETE FROM conversations WHERE id = ?", (row_id,))
                fixed += 1

            # Phase 4: SQLite integrity check
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            db_ok = integrity and integrity[0] == "ok"
            if not db_ok:
                issues += 1

        return {
            "session_id": session_id,
            "issues_found": issues,
            "issues_fixed": fixed,
            "removed_empty": removed,
            "removed_dupes": len(dupes_to_remove),
            "db_integrity": "ok" if db_ok else "failed",
        }
