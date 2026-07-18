"""Claude Code subagent runner — scoped headless `claude -p` execution daemon.

This module lets an orchestrator (AitherOS Genesis, another agent, or a human via
`adk claude spawn`) dispatch tightly scoped, headless Claude Code subagents and
collect their results. It is the shared backend behind BOTH frontends:

- HTTP daemon: ``adk claude serve`` (default 127.0.0.1:8360, Bearer-token auth)
- CLI client:  ``adk claude spawn|runs|kill``

Each run gets an isolated workspace (generated ``--settings`` permission
allowlist, ``--strict-mcp-config`` MCP config, appended system prompt, optional
per-run Claude account via ``CLAUDE_CONFIG_DIR``) and produces a stream-json
transcript whose final ``result`` event carries the outcome and token usage.

Security posture (fail-closed, per AitherOS security-review-patterns):
- The daemon REFUSES to serve without a Bearer token; every non-health endpoint
  denies on missing (401) or wrong (403) credentials.
- Scope validation denies by default: empty tool allowlist → rejected; MCP
  servers must be http/sse to an allowlisted gateway host (stdio servers are
  rejected outright — they are arbitrary command execution).
- The task prompt is passed via stdin (never argv — argv leaks to the process
  table) and stored redacted; API responses redact known secret patterns.
- Runs execute in a per-run workdir by default and with ALL settings files
  excluded (--setting-sources ""), so no repo/user hooks or permission grants
  leak into scoped runs; subagent env is stripped of sensitive variables.

Thread safety: watcher threads (_watch/_finalize) and HTTP handlers (submit/
kill) run concurrently; every record load-modify-save cycle MUST hold
``self._lock``. External code must never mutate records directly.

Environment (all optional):
- AITHER_CLAUDE_RUNNER_ROOT: state root (default ~/.aither/claude-runner)
- AITHER_CLAUDE_RUNNER_TOKEN: Bearer token for the daemon
- AITHER_CLAUDE_RUNNER_PORT / _HOST: bind address (default 127.0.0.1:8360)
- AITHER_CLAUDE_RUNNER_MAX_CONCURRENCY: parallel runs (default 2)
- AITHER_CLAUDE_RUNNER_QUEUE_MAX: queued runs before 503 (default 8)
- AITHER_CLAUDE_RUNNER_GATEWAY_HOSTS: comma list of allowed MCP hosts
  (default mcp.aitherium.com,localhost,127.0.0.1,host.docker.internal)
- AITHER_CLAUDE_RUNNER_BIN: claude executable (default "claude")
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets as std_secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from adk.core.logging import get_logger

_log = get_logger("aither_adk.claude_runner")

DEFAULT_PORT = 8360
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_QUEUE_MAX = 8
DEFAULT_TIMEOUT_SEC = 600
MAX_TIMEOUT_SEC = 3600
DEFAULT_GATEWAY_HOSTS = "mcp.aitherium.com,localhost,127.0.0.1,host.docker.internal"
# Exclude ALL settings files by default (fail-closed): no user, project, or
# local settings apply to a scoped run — only the generated --settings file.
# Callers must explicitly opt in (e.g. setting_sources="user") if they want the
# operator's defaults. Empty value verified accepted by claude 2.1.214 live.
DEFAULT_SETTING_SOURCES = ""

# Env vars matching this pattern are STRIPPED from the spawned subagent's
# environment — most importantly the daemon's own Bearer token, which would
# otherwise let a prompt-injected subagent call back into this API.
_SENSITIVE_ENV_RE = re.compile(r"(?i)(token|secret|passw|api[_-]?key|credential)")

# Windows: CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP. NOT DETACHED_PROCESS —
# the claude CLI on Windows is an npm .cmd shim, and cmd.exe HANGS when spawned
# with DETACHED_PROCESS (no console at all). CREATE_NO_WINDOW gives it an
# invisible console instead (proven by isolation test 2026-07-18).
_WIN_FLAGS = 0x08000000 | 0x00000200

# Secret patterns from .claude/rules/secret-safety.md — redacted from stored
# records and API responses. The raw stream file stays on disk under the run
# root (0700 best-effort) for forensics.
_SECRET_RES = [
    re.compile(p)
    for p in (
        r"sk-ant-[A-Za-z0-9_\-]{8,}",
        r"sk-[A-Za-z0-9_\-]{16,}",
        r"ghp_[A-Za-z0-9]{16,}",
        r"ghs_[A-Za-z0-9]{16,}",
        r"AKIA[0-9A-Z]{12,}",
        r"pk_live_[A-Za-z0-9]{8,}",
        r"sk_live_[A-Za-z0-9]{8,}",
        r"xox[bp]-[A-Za-z0-9\-]{8,}",
        r"aither_sk_live_[A-Za-z0-9_\-]{8,}",
        r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S{8,}",
    )
]


class RunnerError(RuntimeError):
    """Raised on runner configuration or lifecycle errors."""


class ScopeError(ValueError):
    """Raised when a submitted scope fails fail-closed validation."""


class QueueFullError(RunnerError):
    """Raised when the pending-run queue is at capacity."""


def clean_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for spawned subagents with sensitive variables stripped.

    The subagent authenticates to Claude via its config home (CLAUDE_CONFIG_DIR
    or the default), never via env vars — so no *TOKEN/*SECRET/*KEY/*PASSW*
    style variable is ever passed through.
    """
    src = dict(os.environ) if base is None else dict(base)
    return {k: v for k, v in src.items() if not _SENSITIVE_ENV_RE.search(k)}


def restrict_file(path: Path) -> None:
    """Best-effort owner-only permissions: chmod 0600 + icacls on Windows."""
    try:
        path.chmod(0o600)
    except OSError:
        pass
    if sys.platform == "win32":
        user = os.environ.get("USERNAME", "")
        if user:
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                capture_output=True,
            )


def redact(text: str | None) -> str | None:
    """Redact known secret patterns from text (for records and API responses)."""
    if not text:
        return text
    out = text
    for rx in _SECRET_RES:
        out = rx.sub("[REDACTED]", out)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_gateway_hosts() -> set[str]:
    raw = os.environ.get("AITHER_CLAUDE_RUNNER_GATEWAY_HOSTS", DEFAULT_GATEWAY_HOSTS)
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


# ---------------------------------------------------------------------------
# Scope — pre-compiled by the caller, validated and enforced here
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunScope:
    """A pre-compiled, validated execution scope for one subagent run."""

    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    mcp_config: Optional[dict[str, Any]] = None
    system_prompt: str = ""
    model: str = ""
    cwd: str = ""
    add_dirs: list[str] = field(default_factory=list)
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    budget_usd: float = 0.0
    account_profile: str = ""
    setting_sources: str = DEFAULT_SETTING_SOURCES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunScope":
        if not isinstance(data, dict):
            raise ScopeError("scope must be an object")
        scope = cls(
            allowed_tools=list(data.get("allowed_tools") or []),
            disallowed_tools=list(data.get("disallowed_tools") or []),
            mcp_config=data.get("mcp_config"),
            system_prompt=str(data.get("system_prompt") or ""),
            model=str(data.get("model") or ""),
            cwd=str(data.get("cwd") or ""),
            add_dirs=list(data.get("add_dirs") or []),
            timeout_sec=int(data.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
            budget_usd=float(data.get("budget_usd") or 0.0),
            account_profile=str(data.get("account_profile") or ""),
            setting_sources=str(data.get("setting_sources") or DEFAULT_SETTING_SOURCES),
        )
        scope.validate()
        return scope

    def validate(self) -> None:
        """Fail-closed validation. Raises ScopeError on ANY doubt."""
        if not self.allowed_tools or not all(
            isinstance(t, str) and t.strip() for t in self.allowed_tools
        ):
            raise ScopeError("scope.allowed_tools must be a non-empty list of tool names")
        for tool in self.allowed_tools + self.disallowed_tools:
            if any(ch in tool for ch in ";|&`$\n\r"):
                raise ScopeError(f"illegal characters in tool name: {tool!r}")
        if self.timeout_sec < 5 or self.timeout_sec > MAX_TIMEOUT_SEC:
            raise ScopeError(
                f"scope.timeout_sec must be 5..{MAX_TIMEOUT_SEC}, got {self.timeout_sec}"
            )
        if self.budget_usd < 0:
            raise ScopeError("scope.budget_usd must be >= 0")
        if self.cwd:
            cwd = Path(self.cwd)
            if not cwd.is_absolute() or not cwd.is_dir():
                raise ScopeError(f"scope.cwd must be an existing absolute directory: {self.cwd}")
        for d in self.add_dirs:
            p = Path(d)
            if not p.is_absolute() or not p.is_dir():
                raise ScopeError(f"scope.add_dirs entries must exist and be absolute: {d}")
        if self.account_profile and not self.account_profile.isidentifier():
            raise ScopeError(f"scope.account_profile must be an identifier: {self.account_profile}")
        self._validate_mcp()

    def _validate_mcp(self) -> None:
        if self.mcp_config is None:
            return
        if not isinstance(self.mcp_config, dict):
            raise ScopeError("scope.mcp_config must be an object")
        servers = self.mcp_config.get("mcpServers")
        if not isinstance(servers, dict):
            raise ScopeError('scope.mcp_config must contain an "mcpServers" object')
        allowed_hosts = _allowed_gateway_hosts()
        for name, server in servers.items():
            if not isinstance(server, dict):
                raise ScopeError(f"mcp server {name!r} must be an object")
            stype = str(server.get("type") or "").lower()
            if stype not in ("http", "sse"):
                # stdio servers execute arbitrary local commands — never allowed
                # from a submitted scope.
                raise ScopeError(f"mcp server {name!r}: only http/sse servers are allowed")
            url = str(server.get("url") or "")
            parsed = urlparse(url)
            if parsed.scheme not in ("https", "http"):
                raise ScopeError(f"mcp server {name!r}: url must be http(s), got {url!r}")
            host = (parsed.hostname or "").lower()
            if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1"):
                raise ScopeError(f"mcp server {name!r}: plain http only allowed to localhost")
            if host not in allowed_hosts:
                raise ScopeError(
                    f"mcp server {name!r}: host {host!r} not in allowed gateway hosts"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": self.allowed_tools,
            "disallowed_tools": self.disallowed_tools,
            "mcp_config": self.mcp_config,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "cwd": self.cwd,
            "add_dirs": self.add_dirs,
            "timeout_sec": self.timeout_sec,
            "budget_usd": self.budget_usd,
            "account_profile": self.account_profile,
            "setting_sources": self.setting_sources,
        }


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunRecord:
    """Durable state of one subagent run (persisted as record.json)."""

    run_id: str
    goal_id: str = ""
    status: str = "queued"  # queued|running|completed|failed|cancelled
    task_redacted: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    pid: int = 0
    exit_code: Optional[int] = None
    result_text: str = ""
    result_subtype: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    num_turns: int = 0
    error: str = ""
    created_at: str = field(default_factory=_now_iso)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "status": self.status,
            "task_redacted": self.task_redacted,
            "scope": self.scope,
            "session_id": self.session_id,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "result_text": self.result_text,
            "result_subtype": self.result_subtype,
            "usage": self.usage,
            "total_cost_usd": self.total_cost_usd,
            "num_turns": self.num_turns,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        rec = cls(run_id=str(data.get("run_id") or ""))
        for key in (
            "goal_id", "status", "task_redacted", "session_id", "result_text",
            "result_subtype", "error", "created_at", "started_at", "completed_at",
        ):
            setattr(rec, key, str(data.get(key) or ""))
        rec.scope = dict(data.get("scope") or {})
        rec.usage = dict(data.get("usage") or {})
        rec.pid = int(data.get("pid") or 0)
        rec.exit_code = data.get("exit_code")
        rec.total_cost_usd = float(data.get("total_cost_usd") or 0.0)
        rec.num_turns = int(data.get("num_turns") or 0)
        return rec


class RunStore:
    """Disk-backed run records under <root>/runs/<run_id>/."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_dir = root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        # run_id is always a server-minted uuid4 — validate anyway so a crafted
        # id can never traverse outside the runs root.
        if not re.fullmatch(r"[0-9a-f\-]{36}", run_id):
            raise RunnerError(f"invalid run id: {run_id!r}")
        return self.runs_dir / run_id

    def save(self, rec: RunRecord) -> None:
        d = self.run_dir(rec.run_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "record.json.tmp"
        tmp.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, d / "record.json")

    def load(self, run_id: str) -> RunRecord:
        path = self.run_dir(run_id) / "record.json"
        if not path.exists():
            raise RunnerError(f"run not found: {run_id}")
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[RunRecord]:
        records = []
        for rec_path in sorted(self.runs_dir.glob("*/record.json")):
            try:
                records.append(RunRecord.from_dict(json.loads(rec_path.read_text("utf-8"))))
            except (json.JSONDecodeError, OSError):
                _log.warning("Skipping malformed run record", extra={"path": str(rec_path)})
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def stream_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "stream.jsonl"


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, text=True,
            )
        except OSError:
            # Cannot determine — assume alive so recovery never falsely
            # orphans a run that is still working.
            return True
        return str(pid) in (out.stdout or "")
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ClaudeRunner:
    """Spawns and tracks scoped headless claude runs."""

    def __init__(
        self,
        root: Optional[Path] = None,
        max_concurrency: Optional[int] = None,
        queue_max: Optional[int] = None,
        claude_bin: Optional[str] = None,
    ) -> None:
        env_root = os.environ.get("AITHER_CLAUDE_RUNNER_ROOT", "")
        self.root = (
            Path(env_root) if env_root else (root or Path.home() / ".aither" / "claude-runner")
        )
        self.store = RunStore(self.root)
        self.max_concurrency = max_concurrency or int(
            os.environ.get("AITHER_CLAUDE_RUNNER_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        self.queue_max = queue_max or int(
            os.environ.get("AITHER_CLAUDE_RUNNER_QUEUE_MAX", DEFAULT_QUEUE_MAX)
        )
        raw_bin = claude_bin or os.environ.get("AITHER_CLAUDE_RUNNER_BIN", "claude")
        # shutil.which honors PATHEXT — required on Windows, where the claude
        # CLI is an npm .cmd shim that bare CreateProcess cannot find.
        self.claude_bin = shutil.which(raw_bin) or raw_bin
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()
        # In-memory active/queued sets so counts() never re-scans the disk
        # (health probes with a large run history were an O(n)-reads vector).
        self._active_ids: set[str] = set()
        self._queued_ids: set[str] = set()
        self._recover_orphans()
        for rec in self.store.list():
            if rec.status == "queued":
                self._queued_ids.add(rec.run_id)
            elif rec.status == "running":
                self._active_ids.add(rec.run_id)

    # -- lifecycle ---------------------------------------------------------

    def _recover_orphans(self) -> None:
        """On startup, mark runs recorded as active-but-dead as failed."""
        for rec in self.store.list():
            if rec.status in ("running", "queued") and not _pid_alive(rec.pid):
                rec.status = "failed"
                rec.error = rec.error or "orphaned: daemon restarted while run was active"
                rec.completed_at = rec.completed_at or _now_iso()
                self.store.save(rec)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {"active": len(self._active_ids), "queued": len(self._queued_ids)}

    def submit(self, task: str, scope: RunScope, goal_id: str = "") -> RunRecord:
        """Validate, persist, and (slot permitting) start a run. Fail-closed."""
        if not task or not task.strip():
            raise ScopeError("task must be a non-empty string")
        scope.validate()
        with self._lock:
            counts = self.counts()
            if counts["active"] + counts["queued"] >= self.max_concurrency + self.queue_max:
                raise QueueFullError(
                    f"runner at capacity ({counts['active']} active, {counts['queued']} queued)"
                )
            rec = RunRecord(
                run_id=str(uuid.uuid4()),
                goal_id=goal_id,
                task_redacted=(redact(task) or "")[:4000],
                scope=scope.to_dict(),
                session_id=str(uuid.uuid4()),
            )
            self.store.save(rec)
            self._queued_ids.add(rec.run_id)
            task_path = self.store.run_dir(rec.run_id) / "task.txt"
            task_path.write_text(task, encoding="utf-8")
            restrict_file(task_path)
            if counts["active"] < self.max_concurrency:
                try:
                    self._start(rec, scope)
                except (OSError, RuntimeError) as e:
                    # Spawn/account failures become an honest failed record, not
                    # a 500 with a phantom queued run left behind. Exception text
                    # can carry credentials (account/store errors) — redact it.
                    self._mark_failed(rec, redact(f"spawn failed: {e}") or "spawn failed")
            return rec

    def _mark_failed(self, rec: RunRecord, error: str) -> None:
        with self._lock:
            rec.status = "failed"
            rec.error = error
            rec.completed_at = _now_iso()
            self.store.save(rec)
            self._queued_ids.discard(rec.run_id)
            self._active_ids.discard(rec.run_id)

    def _pump_queue(self) -> None:
        with self._lock:
            counts = self.counts()
            slots = self.max_concurrency - counts["active"]
            if slots <= 0:
                return
            queued = [r for r in reversed(self.store.list()) if r.status == "queued"]
            for rec in queued[:slots]:
                try:
                    scope = RunScope.from_dict(rec.scope)
                    self._start(rec, scope)
                except (ScopeError, RunnerError, OSError) as e:
                    self._mark_failed(
                        rec, redact(f"failed to start queued run: {e}") or "failed to start"
                    )

    # -- workspace + spawn -------------------------------------------------

    def _build_workspace(self, rec: RunRecord, scope: RunScope) -> tuple[Path, Path, Path]:
        run_dir = self.store.run_dir(rec.run_id)
        workdir = run_dir / "work"
        workdir.mkdir(parents=True, exist_ok=True)

        settings = {
            "permissions": {
                "allow": scope.allowed_tools,
                "deny": scope.disallowed_tools,
            }
        }
        settings_path = run_dir / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

        # Always write an MCP config and pass --strict-mcp-config so the run can
        # NEVER inherit MCP servers from user/project config. Empty by default.
        mcp_path = run_dir / "mcp.json"
        mcp_path.write_text(
            json.dumps(scope.mcp_config or {"mcpServers": {}}, indent=2), encoding="utf-8"
        )
        return run_dir, workdir, settings_path

    def _build_account_env(self, rec: RunRecord, scope: RunScope) -> dict[str, str]:
        # Sensitive vars (the daemon's own Bearer token above all) are stripped
        # so a prompt-injected subagent can never call back into this API.
        env = clean_subprocess_env()
        if not scope.account_profile:
            return env
        try:
            from adk.claude_accounts import ClaudeAccountStore  # lazy: client paths skip it
        except ImportError as e:  # pragma: no cover — packaged together
            raise RunnerError(f"adk.claude_accounts unavailable: {e}") from e

        store = ClaudeAccountStore()
        profile = store.get_profile(scope.account_profile)  # raises if missing
        config_home = self.store.run_dir(rec.run_id) / "claude-home"
        config_home.mkdir(parents=True, exist_ok=True)
        creds_path = config_home / ".credentials.json"
        creds_path.write_text(json.dumps(profile.credentials, indent=2), encoding="utf-8")
        restrict_file(creds_path)
        env["CLAUDE_CONFIG_DIR"] = str(config_home)
        return env

    def _build_argv(self, rec: RunRecord, scope: RunScope, run_dir: Path) -> list[str]:
        argv = [
            self.claude_bin,
            "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--settings", str(run_dir / "settings.json"),
            "--setting-sources", scope.setting_sources,
            "--allowedTools", ",".join(scope.allowed_tools),
            "--mcp-config", str(run_dir / "mcp.json"),
            "--strict-mcp-config",
            "--session-id", rec.session_id,
        ]
        if scope.disallowed_tools:
            argv += ["--disallowedTools", ",".join(scope.disallowed_tools)]
        if scope.system_prompt:
            argv += ["--append-system-prompt", scope.system_prompt]
        if scope.model:
            argv += ["--model", scope.model]
        if scope.budget_usd > 0:
            argv += ["--max-budget-usd", str(scope.budget_usd)]
        for d in scope.add_dirs:
            argv += ["--add-dir", d]
        return argv

    def _start(self, rec: RunRecord, scope: RunScope) -> None:
        run_dir, workdir, _ = self._build_workspace(rec, scope)
        env = self._build_account_env(rec, scope)
        argv = self._build_argv(rec, scope, run_dir)
        task = (run_dir / "task.txt").read_text(encoding="utf-8")

        stream_f = open(self.store.stream_path(rec.run_id), "ab")  # noqa: SIM115
        stderr_f = (run_dir / "stderr.log").open("ab")
        kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": stream_f,
            "stderr": stderr_f,
            "cwd": scope.cwd or str(workdir),
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _WIN_FLAGS
        else:
            kwargs["start_new_session"] = True

        argv_path = run_dir / "argv.txt"
        argv_path.write_text(shlex.join(argv), encoding="utf-8")
        restrict_file(argv_path)
        try:
            proc = subprocess.Popen(argv, **kwargs)
        finally:
            # The child holds its own duplicated handles; release the parent's
            # copies so run files stay deletable on Windows.
            stream_f.close()
            stderr_f.close()
        if proc.stdin is None:  # pragma: no cover — PIPE guarantees a stdin
            raise RunnerError("subprocess stdin unavailable")
        proc.stdin.write(task.encode("utf-8"))
        proc.stdin.close()

        with self._lock:
            rec.status = "running"
            rec.pid = proc.pid
            rec.started_at = _now_iso()
            self.store.save(rec)
            self._queued_ids.discard(rec.run_id)
            self._active_ids.add(rec.run_id)
            self._procs[rec.run_id] = proc

        watcher = threading.Thread(
            target=self._watch, args=(rec.run_id, proc, scope.timeout_sec), daemon=True
        )
        watcher.start()

    def _watch(self, run_id: str, proc: subprocess.Popen, timeout_sec: int) -> None:
        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _log.error("Run process did not die after kill", extra={"run_id": run_id})
        finally:
            with self._lock:
                self._procs.pop(run_id, None)
            try:
                self._finalize(run_id, proc.returncode, timed_out)
            except (RunnerError, OSError, json.JSONDecodeError) as e:
                _log.error("Failed to finalize run", extra={"run_id": run_id, "err": str(e)})
            self._pump_queue()

    def _finalize(self, run_id: str, exit_code: Optional[int], timed_out: bool) -> None:
        # Parse outside the lock (disk I/O), commit the record inside it so a
        # concurrent kill() can never interleave with our load-modify-save.
        result = self._parse_result(run_id)
        with self._lock:
            rec = self.store.load(run_id)
            self._active_ids.discard(run_id)
            self._queued_ids.discard(run_id)
            if rec.status == "cancelled":
                return
            rec.exit_code = exit_code
            rec.completed_at = _now_iso()

            if result:
                rec.result_subtype = str(result.get("subtype") or "")
                rec.result_text = (redact(str(result.get("result") or "")) or "")[:16000]
                rec.usage = dict(result.get("usage") or {})
                rec.total_cost_usd = float(result.get("total_cost_usd") or 0.0)
                rec.num_turns = int(result.get("num_turns") or 0)

            if timed_out:
                rec.status = "failed"
                rec.error = f"timed out after {rec.scope.get('timeout_sec')}s"
            elif exit_code == 0 and result and not result.get("is_error"):
                rec.status = "completed"
            else:
                rec.status = "failed"
                rec.error = rec.error or (
                    f"exit_code={exit_code}, subtype={rec.result_subtype or 'no result event'}"
                )
            self.store.save(rec)

    def _parse_result(self, run_id: str) -> Optional[dict[str, Any]]:
        path = self.store.stream_path(run_id)
        if not path.exists():
            return None
        result = None
        bad_lines = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or '"result"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if obj.get("type") == "result":
                    result = obj
        if bad_lines:
            _log.warning(
                "Unparseable stream-json lines skipped",
                extra={"run_id": run_id, "bad_lines": bad_lines},
            )
        return result

    # -- queries / control -------------------------------------------------

    def get(self, run_id: str) -> RunRecord:
        return self.store.load(run_id)

    def list(self) -> list[RunRecord]:
        return self.store.list()

    def events(self, run_id: str, offset: int = 0) -> tuple[str, int]:
        """Return raw stream-json content from byte offset; (chunk, next_offset)."""
        self.store.load(run_id)  # 404 on unknown id
        path = self.store.stream_path(run_id)
        if not path.exists():
            return "", 0
        size = path.stat().st_size
        offset = max(0, min(offset, size))
        with path.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
        return chunk.decode("utf-8", errors="replace"), offset + len(chunk)

    def kill(self, run_id: str) -> RunRecord:
        with self._lock:
            rec = self.store.load(run_id)
            proc = self._procs.get(run_id)
            if rec.status in ("queued", "running"):
                rec.status = "cancelled"
                rec.completed_at = _now_iso()
                self.store.save(rec)
                self._queued_ids.discard(run_id)
                self._active_ids.discard(run_id)
            else:
                proc = None  # nothing to kill for terminal states
        if proc is not None:
            proc.kill()
        elif rec.status == "cancelled":
            if rec.pid and _pid_alive(rec.pid):
                # Daemon restarted; process survived from a prior life.
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(rec.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    try:
                        os.kill(rec.pid, 9)
                    except OSError:
                        pass
        return self.store.load(run_id)


# ---------------------------------------------------------------------------
# HTTP daemon (FastAPI, Bearer auth, fail-closed)
# ---------------------------------------------------------------------------


def resolve_token(explicit: str = "") -> str:
    """Resolve the daemon Bearer token; generate + persist one if none exists.

    Resolution order: explicit arg → AITHER_CLAUDE_RUNNER_TOKEN →
    AITHER_INTERNAL_SECRET → persisted token file → newly generated (persisted,
    printed once). The daemon NEVER runs tokenless.
    """
    token = (
        explicit
        or os.environ.get("AITHER_CLAUDE_RUNNER_TOKEN", "")
        or os.environ.get("AITHER_INTERNAL_SECRET", "")
    )
    if token:
        return token
    env_root = os.environ.get("AITHER_CLAUDE_RUNNER_ROOT", "")
    root = Path(env_root) if env_root else Path.home() / ".aither" / "claude-runner"
    token_path = root / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    root.mkdir(parents=True, exist_ok=True)
    token = std_secrets.token_urlsafe(32)
    token_path.write_text(token, encoding="utf-8")
    restrict_file(token_path)
    # Never echo the token itself — stdout ends up in logs/CI output.
    print(f"Generated runner token; read it from {token_path}")
    return token


def _record_view(rec: RunRecord) -> dict[str, Any]:
    """API-safe view of a record (already-redacted fields only)."""
    view = rec.to_dict()
    # scope.mcp_config may embed gateway auth headers — never echo them back.
    scope = dict(view.get("scope") or {})
    if scope.get("mcp_config"):
        server_names = sorted((scope["mcp_config"].get("mcpServers") or {}).keys())
        scope["mcp_config"] = {"mcpServers": server_names}
    view["scope"] = scope
    return view


def create_app(runner: ClaudeRunner, token: str):
    """Build the FastAPI app. Token is REQUIRED — no tokenless mode exists."""
    from fastapi import Depends, FastAPI, Header, HTTPException

    if not token:
        raise RunnerError("refusing to build app without a Bearer token (fail-closed)")

    app = FastAPI(title="aither-claude-runner", docs_url=None, redoc_url=None)

    async def _auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        supplied = authorization[len("Bearer "):].strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=403, detail="invalid token")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        counts = runner.counts()
        return {
            "ok": True,
            "service": "aither-claude-runner",
            "max_concurrency": runner.max_concurrency,
            **counts,
        }

    @app.post("/runs", dependencies=[Depends(_auth)])
    async def create_run(body: dict[str, Any]) -> dict[str, Any]:
        task = str(body.get("task") or "")
        goal_id = str(body.get("goal_id") or "")
        try:
            scope = RunScope.from_dict(body.get("scope") or {})
            rec = runner.submit(task, scope, goal_id=goal_id)
        except ScopeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except QueueFullError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        return _record_view(rec)

    @app.get("/runs", dependencies=[Depends(_auth)])
    async def list_runs() -> list[dict[str, Any]]:
        return [_record_view(r) for r in runner.list()]

    @app.get("/runs/{run_id}", dependencies=[Depends(_auth)])
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            return _record_view(runner.get(run_id))
        except RunnerError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/runs/{run_id}/events", dependencies=[Depends(_auth)])
    async def get_events(run_id: str, offset: int = 0) -> dict[str, Any]:
        try:
            chunk, next_offset = runner.events(run_id, offset)
        except RunnerError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"run_id": run_id, "offset": offset, "next_offset": next_offset,
                "chunk": redact(chunk)}

    @app.delete("/runs/{run_id}", dependencies=[Depends(_auth)])
    async def delete_run(run_id: str) -> dict[str, Any]:
        try:
            return _record_view(runner.kill(run_id))
        except RunnerError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    return app


def serve(host: str = "", port: int = 0, token: str = "") -> int:
    """Run the daemon in the foreground (uvicorn)."""
    import uvicorn

    host = host or os.environ.get("AITHER_CLAUDE_RUNNER_HOST", DEFAULT_HOST)
    port = port or int(os.environ.get("AITHER_CLAUDE_RUNNER_PORT", DEFAULT_PORT))
    resolved = resolve_token(token)
    runner = ClaudeRunner()
    app = create_app(runner, resolved)
    _log.info("claude-runner serving", extra={"host": host, "port": port})
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# CLI (adk claude ...)
# ---------------------------------------------------------------------------


def _client_base_and_token(args: Any) -> tuple[str, str]:
    url = getattr(args, "url", "") or os.environ.get(
        "AITHER_CLAUDE_RUNNER_URL", f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    )
    token = resolve_token(getattr(args, "token", "") or "")
    return url.rstrip("/"), token


def cmd_claude(args: Any) -> int:
    """Dispatcher for `adk claude <serve|spawn|runs|kill>`."""
    sub = getattr(args, "claude_command", None)
    try:
        if sub == "serve":
            return serve(
                host=getattr(args, "host", "") or "",
                port=int(getattr(args, "port", 0) or 0),
                token=getattr(args, "token", "") or "",
            )
        if sub == "spawn":
            return _cmd_spawn(args)
        if sub == "runs":
            return _cmd_runs(args)
        if sub == "kill":
            return _cmd_kill(args)
        print("Usage: adk claude [serve|spawn|runs|kill]")
        return 1
    except (RunnerError, ScopeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_spawn(args: Any) -> int:
    import httpx

    task = getattr(args, "task", "") or ""
    task_file = getattr(args, "task_file", "") or ""
    if task_file:
        task = Path(task_file).read_text(encoding="utf-8")
    if not task.strip():
        print("Error: --task or --task-file is required", file=sys.stderr)
        return 1

    scope: dict[str, Any] = {
        "allowed_tools": [t for t in (getattr(args, "allow", "") or "").split(",") if t],
        "disallowed_tools": [t for t in (getattr(args, "deny", "") or "").split(",") if t],
        "system_prompt": getattr(args, "append_system_prompt", "") or "",
        "model": getattr(args, "model", "") or "",
        "cwd": getattr(args, "cwd", "") or "",
        "timeout_sec": int(getattr(args, "timeout", 0) or DEFAULT_TIMEOUT_SEC),
        "budget_usd": float(getattr(args, "budget_usd", 0) or 0),
        "account_profile": getattr(args, "account", "") or "",
    }
    mcp_file = getattr(args, "mcp_config", "") or ""
    if mcp_file:
        scope["mcp_config"] = json.loads(Path(mcp_file).read_text(encoding="utf-8"))

    base, token = _client_base_and_token(args)
    headers = {"Authorization": f"Bearer {token}"}
    body = {"task": task, "goal_id": getattr(args, "goal", "") or "", "scope": scope}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base}/runs", json=body, headers=headers)
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1
        rec = resp.json()
        run_id = rec["run_id"]
        print(f"run_id: {run_id}  (goal: {rec.get('goal_id') or '-'})")

        if getattr(args, "no_wait", False):
            return 0

        deadline = time.monotonic() + scope["timeout_sec"] + 60
        while time.monotonic() < deadline:
            time.sleep(3)
            status = client.get(f"{base}/runs/{run_id}", headers=headers).json()
            if status.get("status") not in ("queued", "running"):
                print(f"status: {status.get('status')}  "
                      f"turns: {status.get('num_turns')}  "
                      f"cost: ${status.get('total_cost_usd', 0):.4f}")
                if status.get("result_text"):
                    print("\n" + status["result_text"])
                if status.get("error"):
                    print(f"error: {status['error']}", file=sys.stderr)
                return 0 if status.get("status") == "completed" else 1
        print("Error: timed out waiting for run", file=sys.stderr)
        return 1


def _cmd_runs(args: Any) -> int:
    import httpx

    base, token = _client_base_and_token(args)
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(f"{base}/runs", headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1
        rows = resp.json()
    if not rows:
        print("No runs.")
        return 0
    print(f"  {'run_id':<38} {'status':<10} {'turns':>5} {'cost':>9}  task")
    for r in rows:
        task = (r.get("task_redacted") or "").replace("\n", " ")[:48]
        print(f"  {r['run_id']:<38} {r['status']:<10} {r.get('num_turns', 0):>5} "
              f"${r.get('total_cost_usd', 0):>8.4f}  {task}")
    return 0


def _cmd_kill(args: Any) -> int:
    import httpx

    run_id = getattr(args, "run_id", "") or ""
    base, token = _client_base_and_token(args)
    with httpx.Client(timeout=15.0) as client:
        resp = client.delete(
            f"{base}/runs/{run_id}", headers={"Authorization": f"Bearer {token}"}
        )
        if resp.status_code != 200:
            print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1
        print(f"status: {resp.json().get('status')}")
    return 0
