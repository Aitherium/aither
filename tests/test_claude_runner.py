"""Tests for adk.claude_runner — scoped headless Claude subagent runner.

No test here ever invokes the real claude CLI: the integration tests use a fake
claude executable (a Python script emitting genuine stream-json shape) via
AITHER_CLAUDE_RUNNER_BIN, and all state roots are redirected with
AITHER_CLAUDE_RUNNER_ROOT so the real ~/.aither is never touched.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from adk.claude_runner import (
    ClaudeRunner,
    QueueFullError,
    RunnerError,
    RunRecord,
    RunScope,
    ScopeError,
    _record_view,
    clean_subprocess_env,
    create_app,
    redact,
    resolve_token,
)


@pytest.fixture
def runner_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "runner-root"
    monkeypatch.setenv("AITHER_CLAUDE_RUNNER_ROOT", str(root))
    return root


@pytest.fixture
def fake_claude(tmp_path: Path) -> str:
    """A fake claude binary that reads stdin and emits stream-json to stdout."""
    script = tmp_path / "fake_claude.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            task = sys.stdin.read()
            print(json.dumps({"type": "system", "subtype": "init", "session_id": "s-1"}))
            print(json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": "fake-done: " + task.strip()[:40],
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "total_cost_usd": 0.0012, "num_turns": 1, "session_id": "s-1",
            }))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    if sys.platform == "win32":
        shim = tmp_path / "fake_claude.cmd"
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        return str(shim)
    shim = tmp_path / "fake_claude.sh"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return str(shim)


def _scope(**overrides) -> RunScope:
    base = {"allowed_tools": ["Read"]}
    base.update(overrides)
    return RunScope.from_dict(base)


# ---------------------------------------------------------------------------
# Scope validation — fail-closed
# ---------------------------------------------------------------------------


class TestScopeValidation:
    def test_empty_allowlist_denied(self):
        with pytest.raises(ScopeError, match="allowed_tools"):
            RunScope.from_dict({"allowed_tools": []})

    def test_missing_scope_denied(self):
        with pytest.raises(ScopeError):
            RunScope.from_dict({})

    def test_illegal_tool_chars_denied(self):
        with pytest.raises(ScopeError, match="illegal"):
            RunScope.from_dict({"allowed_tools": ["Read;rm -rf"]})

    def test_stdio_mcp_server_denied(self):
        mcp = {"mcpServers": {"evil": {"type": "stdio", "command": "pwsh"}}}
        with pytest.raises(ScopeError, match="http/sse"):
            RunScope.from_dict({"allowed_tools": ["Read"], "mcp_config": mcp})

    def test_unlisted_gateway_host_denied(self):
        mcp = {"mcpServers": {"x": {"type": "http", "url": "https://evil.example.com/mcp"}}}
        with pytest.raises(ScopeError, match="allowed gateway hosts"):
            RunScope.from_dict({"allowed_tools": ["Read"], "mcp_config": mcp})

    def test_plain_http_to_remote_denied(self):
        mcp = {"mcpServers": {"x": {"type": "http", "url": "http://mcp.aitherium.com/mcp"}}}
        with pytest.raises(ScopeError, match="localhost"):
            RunScope.from_dict({"allowed_tools": ["Read"], "mcp_config": mcp})

    def test_allowed_gateway_ok(self):
        mcp = {"mcpServers": {"aitheros": {"type": "http", "url": "https://mcp.aitherium.com/mcp"}}}
        scope = RunScope.from_dict({"allowed_tools": ["Read"], "mcp_config": mcp})
        assert scope.mcp_config is not None

    def test_timeout_bounds(self):
        with pytest.raises(ScopeError, match="timeout"):
            RunScope.from_dict({"allowed_tools": ["Read"], "timeout_sec": 999999})

    def test_missing_cwd_denied(self):
        with pytest.raises(ScopeError, match="cwd"):
            RunScope.from_dict({"allowed_tools": ["Read"], "cwd": "Z:/definitely/not/here"})

    def test_bad_account_profile_denied(self):
        with pytest.raises(ScopeError, match="account_profile"):
            RunScope.from_dict({"allowed_tools": ["Read"], "account_profile": "../../etc"})


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-abc123def456ghi789",
            "ghp_ABCDEF1234567890abcd",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234-5678-example0",  # placeholder marker keeps secret_scan quiet
            "aither_sk_live_AbCdEf123456",
            "api_key = supersecretvalue123",
        ],
    )
    def test_secrets_redacted(self, secret: str):
        out = redact(f"before {secret} after")
        assert "[REDACTED]" in out
        assert secret not in out

    def test_plain_text_untouched(self):
        assert redact("just words, nothing secret") == "just words, nothing secret"


# ---------------------------------------------------------------------------
# Store + lifecycle (no HTTP)
# ---------------------------------------------------------------------------


class TestRunnerCore:
    def test_run_dir_traversal_rejected(self, runner_root: Path):
        runner = ClaudeRunner()
        with pytest.raises(RunnerError, match="invalid run id"):
            runner.store.run_dir("../../escape")

    def test_submit_and_complete_with_fake_claude(self, runner_root: Path, fake_claude: str):
        runner = ClaudeRunner(claude_bin=fake_claude)
        rec = runner.submit("say hello to CI", _scope(timeout_sec=60))
        assert rec.status in ("queued", "running")

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            rec = runner.get(rec.run_id)
            if rec.status not in ("queued", "running"):
                break
            time.sleep(0.2)

        assert rec.status == "completed", f"error={rec.error} exit={rec.exit_code}"
        assert rec.result_text.startswith("fake-done: say hello")
        assert rec.usage.get("input_tokens") == 10
        assert rec.total_cost_usd == pytest.approx(0.0012)
        assert rec.num_turns == 1
        # Scoping artifacts actually written and strict flags actually passed.
        run_dir = runner.store.run_dir(rec.run_id)
        argv = (run_dir / "argv.txt").read_text(encoding="utf-8")
        assert "--strict-mcp-config" in argv
        # Fail-closed default: ALL settings files excluded (empty sources arg).
        assert "--setting-sources ''" in argv or '--setting-sources ""' in argv
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        assert settings["permissions"]["allow"] == ["Read"]

    def test_kill_after_completion_does_not_clobber(
        self, runner_root: Path, fake_claude: str
    ):
        runner = ClaudeRunner(claude_bin=fake_claude)
        rec = runner.submit("quick task", _scope(timeout_sec=60))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            rec = runner.get(rec.run_id)
            if rec.status not in ("queued", "running"):
                break
            time.sleep(0.2)
        assert rec.status == "completed"
        after = runner.kill(rec.run_id)
        assert after.status == "completed"  # kill must not rewrite a terminal state

    def test_task_stored_redacted(self, runner_root: Path, fake_claude: str):
        runner = ClaudeRunner(claude_bin=fake_claude)
        rec = runner.submit("use key sk-ant-abc123def456ghi789 now", _scope(timeout_sec=60))
        assert "sk-ant-" not in rec.task_redacted
        assert "[REDACTED]" in rec.task_redacted

    def test_empty_task_denied(self, runner_root: Path):
        runner = ClaudeRunner()
        with pytest.raises(ScopeError, match="task"):
            runner.submit("   ", _scope())

    def test_queue_full_denied(self, runner_root: Path, monkeypatch: pytest.MonkeyPatch):
        runner = ClaudeRunner(max_concurrency=1, queue_max=1)
        # Never actually spawn: pretend starts succeed and stay running.
        monkeypatch.setattr(
            ClaudeRunner,
            "_start",
            lambda self, rec, scope: (
                setattr(rec, "status", "running"),
                setattr(rec, "pid", 999_999_999),
                self.store.save(rec),
            ),
        )
        runner.submit("a", _scope())
        runner.submit("b", _scope())
        with pytest.raises(QueueFullError):
            runner.submit("c", _scope())

    def test_orphan_recovery_marks_failed(self, runner_root: Path):
        runner = ClaudeRunner()
        rec = RunRecord(run_id="11111111-2222-3333-4444-555555555555")
        rec.status = "running"
        rec.pid = 999_999_999  # certainly dead
        runner.store.save(rec)

        recovered = ClaudeRunner()
        rec2 = recovered.get(rec.run_id)
        assert rec2.status == "failed"
        assert "orphaned" in rec2.error


# ---------------------------------------------------------------------------
# HTTP daemon — fail-closed auth
# ---------------------------------------------------------------------------


class TestHttpAuth:
    @pytest.fixture
    def client(self, runner_root: Path):
        from fastapi.testclient import TestClient

        runner = ClaudeRunner()
        app = create_app(runner, token="test-token-123")
        return TestClient(app)

    def test_tokenless_app_refused(self, runner_root: Path):
        with pytest.raises(RunnerError, match="fail-closed"):
            create_app(ClaudeRunner(), token="")

    def test_health_open(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_missing_token_401(self, client):
        assert client.get("/runs").status_code == 401

    def test_wrong_token_403(self, client):
        resp = client.get("/runs", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_right_token_200(self, client):
        resp = client.get("/runs", headers={"Authorization": "Bearer test-token-123"})
        assert resp.status_code == 200

    def test_bad_scope_400(self, client):
        resp = client.post(
            "/runs",
            json={"task": "x", "scope": {"allowed_tools": []}},
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert resp.status_code == 400

    def test_unknown_run_404(self, client):
        resp = client.get(
            "/runs/11111111-2222-3333-4444-555555555555",
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API views + token persistence
# ---------------------------------------------------------------------------


class TestViewsAndToken:
    def test_mcp_auth_headers_never_echoed(self):
        rec = RunRecord(run_id="11111111-2222-3333-4444-555555555555")
        rec.scope = {
            "allowed_tools": ["Read"],
            "mcp_config": {
                "mcpServers": {
                    "aitheros": {
                        "type": "http",
                        "url": "https://mcp.aitherium.com/mcp",
                        "headers": {"Authorization": "Bearer aither_sk_live_SECRET"},
                    }
                }
            },
        }
        view = _record_view(rec)
        assert view["scope"]["mcp_config"] == {"mcpServers": ["aitheros"]}
        assert "aither_sk_live_SECRET" not in json.dumps(view)

    def test_token_generated_and_persisted(
        self, runner_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        monkeypatch.delenv("AITHER_CLAUDE_RUNNER_TOKEN", raising=False)
        monkeypatch.delenv("AITHER_INTERNAL_SECRET", raising=False)
        first = resolve_token()
        assert first and len(first) > 20
        assert (runner_root / "token").exists()
        assert first not in capsys.readouterr().out  # token value never echoed
        assert resolve_token() == first  # stable across calls

    def test_subprocess_env_strips_sensitive_vars(self):
        base = {
            "PATH": "/usr/bin",
            "USERPROFILE": "C:/Users/x",
            "AITHER_CLAUDE_RUNNER_TOKEN": "supersecret",
            "AITHER_INTERNAL_SECRET": "alsosecret",
            "GITHUB_TOKEN": "ghp_abc",
            "MY_API_KEY": "k",
            "DB_PASSWORD": "p",
            "ANTHROPIC_API_KEY": "sk-ant-x",
        }
        env = clean_subprocess_env(base)
        assert env["PATH"] == "/usr/bin"
        assert env["USERPROFILE"] == "C:/Users/x"
        for gone in (
            "AITHER_CLAUDE_RUNNER_TOKEN", "AITHER_INTERNAL_SECRET", "GITHUB_TOKEN",
            "MY_API_KEY", "DB_PASSWORD", "ANTHROPIC_API_KEY",
        ):
            assert gone not in env

    def test_env_token_wins(self, runner_root: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AITHER_CLAUDE_RUNNER_TOKEN", "env-token")
        assert resolve_token() == "env-token"
        assert resolve_token("explicit") == "explicit"
