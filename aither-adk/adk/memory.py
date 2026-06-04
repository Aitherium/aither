"""Local SQLite memory — conversations, KV store, search."""

from __future__ import annotations

import json
import logging
import os
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
        self._spirit_url = os.environ.get("AITHER_SPIRIT_URL", "http://localhost:8087")
        self._spirit_enabled = os.environ.get("AITHER_SPIRIT_BRIDGE", "true").lower() == "true"
        self._tenant_id = os.environ.get("AITHER_TENANT_ID", "")
        self._workspace_id = os.environ.get("AITHER_WORKSPACE_ID", "")
        # Cloud mode: route memory through gateway when no local Spirit.
        # Paths/URL may come from env (exported by Config.from_env from saved config).
        self._spirit_teach_path = os.environ.get("AITHER_SPIRIT_TEACH_PATH", "/teach")
        self._spirit_recall_path = os.environ.get("AITHER_SPIRIT_RECALL_PATH", "/recall")
        self._spirit_auth_token = ""
        cloud_mode = os.environ.get("AITHER_CLOUD_MODE", "")
        if cloud_mode in ("cloud_first", "cloud_only") and self._spirit_url == "http://localhost:8087":
            gateway = os.environ.get("AITHER_GATEWAY_URL", "https://gateway.aitherium.com")
            self._spirit_url = gateway
            if self._spirit_teach_path == "/teach":
                self._spirit_teach_path = "/v1/memory/teach"
            if self._spirit_recall_path == "/recall":
                self._spirit_recall_path = "/v1/memory/recall"
            # Use ACTA key or bearer token for gateway auth
            self._spirit_auth_token = os.environ.get("AITHER_API_KEY", "")
        self._init_db()

    # Schema version — increment when tables change, add migration in _migrate()
    _SCHEMA_VERSION = 2

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
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_kv_category ON kv_store(category);
            """)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection):
        """Run schema migrations. Each version bump adds an elif branch."""
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        current = row[0] if row else 0

        if current >= self._SCHEMA_VERSION:
            return

        if current < 1:
            # v1: initial schema (tables created above)
            pass

        if current < 2:
            # v2: add agent_name column to conversations for multi-agent filtering
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN agent_name TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Update version
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self._SCHEMA_VERSION,))
        conn.commit()
        if current > 0:
            logger.info("Memory DB migrated %d → %d", current, self._SCHEMA_VERSION)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def _spirit_teach(self, content: str, category: str = "general"):
        """Push memory to Spirit for cross-session persistence (best-effort).

        Scoped by tenant_id + agent_name so each product/workspace gets
        isolated memory. Platform agents (no AITHER_TENANT_ID) use shared scope.
        In cloud mode, routes through the gateway memory proxy.
        """
        if not self._spirit_enabled:
            return
        try:
            import httpx
            payload = {
                "content": content,
                "category": category,
                "source_agent": self._agent,
                "scope": f"agent:{self._agent}",
            }
            if self._tenant_id:
                payload["tenant_id"] = self._tenant_id
            headers = {}
            if self._spirit_auth_token:
                headers["Authorization"] = f"Bearer {self._spirit_auth_token}"
            async with httpx.AsyncClient(timeout=5.0, verify=os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false") as client:
                await client.post(
                    f"{self._spirit_url}{self._spirit_teach_path}",
                    json=payload,
                    headers=headers,
                )
        except Exception as e:
            logger.debug("SpiritBridge teach failed (non-fatal): %s", e)

    async def _spirit_recall(self, query: str, limit: int = 5) -> list[dict]:
        """Pull memories from Spirit (best-effort, returns [] on failure).

        Scoped by tenant_id + agent_id so GARG can't see Chelle's memories.
        In cloud mode, routes through the gateway memory proxy.
        """
        if not self._spirit_enabled:
            return []
        try:
            import httpx
            payload = {
                "query": query,
                "limit": limit,
                "agent_id": self._agent,
            }
            if self._tenant_id:
                payload["tenant_id"] = self._tenant_id
            headers = {}
            if self._spirit_auth_token:
                headers["Authorization"] = f"Bearer {self._spirit_auth_token}"
            async with httpx.AsyncClient(timeout=5.0, verify=os.getenv("AITHER_TLS_VERIFY", "true").lower() != "false") as client:
                resp = await client.post(
                    f"{self._spirit_url}{self._spirit_recall_path}",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", data.get("memories", []))
        except Exception as e:
            logger.debug("SpiritBridge recall failed (non-fatal): %s", e)
        return []

    async def remember(self, key: str, value: str, category: str = "general", metadata: dict | None = None):
        """Store a key-value pair in memory."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, category, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (key, value, category, time.time(), json.dumps(metadata) if metadata else None),
            )
        await self._spirit_teach(f"{key}: {value}", category)

    async def recall(self, key: str) -> str | None:
        """Retrieve a value by key."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        if row:
            return row[0]
        # Spirit fallback -- cross-session recall when local DB misses
        spirit_results = await self._spirit_recall(key, limit=1)
        if spirit_results:
            value = spirit_results[0].get("content", spirit_results[0].get("value", ""))
            if value:
                # Cache locally for next time
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO kv_store (key, value, category, timestamp) VALUES (?, ?, ?, ?)",
                        (key, value, "spirit_cache", time.time()),
                    )
                return value
        return None

    async def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search memory entries by substring match."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, category, timestamp, metadata FROM kv_store "
                "WHERE key LIKE ? OR value LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        results = [
            MemoryEntry(
                key=r[0], value=r[1], category=r[2], timestamp=r[3],
                metadata=json.loads(r[4]) if r[4] else None,
            )
            for r in rows
        ]
        # Fill remaining slots from Spirit if local results are sparse
        if len(results) < limit:
            spirit_results = await self._spirit_recall(query, limit=limit - len(results))
            seen_keys = {r.key for r in results}
            for sr in spirit_results:
                key = sr.get("title", sr.get("id", ""))
                if key and key not in seen_keys:
                    results.append(MemoryEntry(
                        key=key,
                        value=sr.get("content", sr.get("value", "")),
                        category=sr.get("category", "spirit"),
                        timestamp=sr.get("timestamp", 0.0),
                        metadata={"source": "spirit"},
                    ))
                    seen_keys.add(key)
        return results

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

    async def get_entry(self, key: str) -> MemoryEntry | None:
        """Fetch a full stored entry (value + metadata) by key, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT key, value, category, timestamp, metadata FROM kv_store WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return MemoryEntry(
            key=row[0], value=row[1], category=row[2], timestamp=row[3],
            metadata=json.loads(row[4]) if row[4] else None,
        )

    async def recent(self, limit: int = 20, category: str | None = None) -> list[MemoryEntry]:
        """Return the most recently stored entries (newest first)."""
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT key, value, category, timestamp, metadata FROM kv_store "
                    "WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, value, category, timestamp, metadata FROM kv_store "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            MemoryEntry(
                key=r[0], value=r[1], category=r[2], timestamp=r[3],
                metadata=json.loads(r[4]) if r[4] else None,
            )
            for r in rows
        ]

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
