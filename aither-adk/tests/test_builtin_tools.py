"""Tests for adk/builtin_tools.py — 21 built-in tool functions."""

import sys
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import adk.builtin_tools as bt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_allowed_roots():
    """Reset the module-level allowed roots cache between tests."""
    bt._ALLOWED_ROOTS = None
    yield
    bt._ALLOWED_ROOTS = None


@pytest.fixture(autouse=True)
def reset_secrets_cache():
    """Reset secrets cache between tests."""
    bt._secrets_cache = None
    yield
    bt._secrets_cache = None


@pytest.fixture
def safe_dir(tmp_path, monkeypatch):
    """Set up a temporary directory that passes _is_safe_path."""
    monkeypatch.setattr(bt, "_DEFAULT_ALLOWED_ROOTS", [str(tmp_path)])
    return tmp_path


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class TestPathSafety:
    def test_is_safe_path_within_allowed(self, safe_dir):
        p = safe_dir / "test.txt"
        assert bt._is_safe_path(str(p)) is True

    def test_is_safe_path_outside_allowed(self, safe_dir):
        # A path well outside the safe_dir
        assert bt._is_safe_path("/unlikely/random/path/outside") is False

    def test_get_allowed_roots_includes_env(self, safe_dir, monkeypatch):
        extra = str(safe_dir / "extra")
        monkeypatch.setenv("AITHER_ALLOWED_ROOTS", extra)
        roots = bt._get_allowed_roots()
        assert extra in roots

    def test_get_allowed_roots_empty_env(self, safe_dir, monkeypatch):
        monkeypatch.setenv("AITHER_ALLOWED_ROOTS", "")
        roots = bt._get_allowed_roots()
        assert str(safe_dir) in roots


# ---------------------------------------------------------------------------
# File I/O tools
# ---------------------------------------------------------------------------

class TestFileRead:
    def test_read_existing_file(self, safe_dir):
        f = safe_dir / "hello.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = bt.file_read(str(f))
        assert result == "Hello World"

    def test_read_nonexistent_file(self, safe_dir):
        result = bt.file_read(str(safe_dir / "nope.txt"))
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_read_outside_allowed_roots(self, safe_dir):
        result = bt.file_read("/unlikely/random/path/file.txt")
        data = json.loads(result)
        assert "error" in data
        assert "outside allowed roots" in data["error"].lower()

    def test_read_with_line_range(self, safe_dir):
        f = safe_dir / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")
        result = bt.file_read(str(f), start_line=2, end_line=4)
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert "line1" not in result

    def test_read_large_file_blocked(self, safe_dir):
        f = safe_dir / "big.bin"
        f.write_bytes(b"x" * (10_000_001))
        result = bt.file_read(str(f))
        data = json.loads(result)
        assert "error" in data
        assert "too large" in data["error"].lower()


class TestFileWrite:
    def test_write_new_file(self, safe_dir):
        target = safe_dir / "output.txt"
        result = bt.file_write(str(target), "content here")
        data = json.loads(result)
        assert data["success"] is True
        assert target.read_text(encoding="utf-8") == "content here"

    def test_write_append_mode(self, safe_dir):
        target = safe_dir / "append.txt"
        target.write_text("first", encoding="utf-8")
        bt.file_write(str(target), " second", mode="append")
        assert target.read_text(encoding="utf-8") == "first second"

    def test_write_creates_parent_dirs(self, safe_dir):
        target = safe_dir / "sub" / "dir" / "file.txt"
        result = bt.file_write(str(target), "nested")
        data = json.loads(result)
        assert data["success"] is True
        assert target.exists()

    def test_write_outside_allowed_roots(self, safe_dir):
        result = bt.file_write("/unlikely/random/path/file.txt", "bad")
        data = json.loads(result)
        assert "error" in data


class TestFileEdit:
    def test_edit_replaces_text(self, safe_dir):
        f = safe_dir / "edit.txt"
        f.write_text("Hello World", encoding="utf-8")
        result = bt.file_edit(str(f), "World", "Universe")
        data = json.loads(result)
        assert data["success"] is True
        assert f.read_text(encoding="utf-8") == "Hello Universe"

    def test_edit_nonexistent_file(self, safe_dir):
        result = bt.file_edit(str(safe_dir / "nope.txt"), "a", "b")
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_edit_old_text_not_found(self, safe_dir):
        f = safe_dir / "edit2.txt"
        f.write_text("Hello", encoding="utf-8")
        result = bt.file_edit(str(f), "MISSING", "replacement")
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_edit_ambiguous_multiple_matches(self, safe_dir):
        f = safe_dir / "dup.txt"
        f.write_text("aaa bbb aaa", encoding="utf-8")
        result = bt.file_edit(str(f), "aaa", "ccc")
        data = json.loads(result)
        assert "error" in data
        assert "2 times" in data["error"]

    def test_edit_outside_allowed_roots(self, safe_dir):
        result = bt.file_edit("/unlikely/random/path/file.txt", "a", "b")
        data = json.loads(result)
        assert "error" in data


class TestFileList:
    def test_list_directory(self, safe_dir):
        (safe_dir / "a.txt").touch()
        (safe_dir / "b.py").touch()
        (safe_dir / "subdir").mkdir()
        result = bt.file_list(str(safe_dir))
        data = json.loads(result)
        assert data["count"] >= 3
        names = [e["name"] for e in data["entries"]]
        assert "a.txt" in names
        assert "subdir" in names

    def test_list_with_pattern(self, safe_dir):
        (safe_dir / "a.txt").touch()
        (safe_dir / "b.py").touch()
        result = bt.file_list(str(safe_dir), pattern="*.py")
        data = json.loads(result)
        names = [e["name"] for e in data["entries"]]
        assert "b.py" in names
        assert "a.txt" not in names

    def test_list_nonexistent_dir(self, safe_dir):
        result = bt.file_list(str(safe_dir / "nonexistent"))
        data = json.loads(result)
        assert "error" in data

    def test_list_entries_have_type_and_size(self, safe_dir):
        f = safe_dir / "sized.txt"
        f.write_text("12345", encoding="utf-8")
        result = bt.file_list(str(safe_dir))
        data = json.loads(result)
        entry = next(e for e in data["entries"] if e["name"] == "sized.txt")
        assert entry["type"] == "file"
        assert entry["size"] == 5


class TestFileSearch:
    def test_search_by_name(self, safe_dir):
        (safe_dir / "alpha.py").write_text("hello", encoding="utf-8")
        (safe_dir / "beta.txt").write_text("world", encoding="utf-8")
        result = bt.file_search(str(safe_dir), "*.py")
        data = json.loads(result)
        assert data["count"] == 1
        assert "alpha.py" in data["results"][0]["path"]

    def test_search_with_content_pattern(self, safe_dir):
        (safe_dir / "a.txt").write_text("find me here", encoding="utf-8")
        (safe_dir / "b.txt").write_text("nothing to see", encoding="utf-8")
        result = bt.file_search(str(safe_dir), "*.txt", content_pattern="find me")
        data = json.loads(result)
        assert data["count"] == 1
        assert "a.txt" in data["results"][0]["path"]
        assert data["results"][0]["matches"][0]["line"] == 1

    def test_search_no_matches(self, safe_dir):
        (safe_dir / "a.txt").write_text("nothing", encoding="utf-8")
        result = bt.file_search(str(safe_dir), "*.py")
        data = json.loads(result)
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------

class TestShellExec:
    @patch("adk.builtin_tools.subprocess.run")
    def test_shell_exec_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="hello output",
            stderr="",
        )
        result = bt.shell_exec("echo hello")
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert data["stdout"] == "hello output"

    @patch("adk.builtin_tools.subprocess.run")
    def test_shell_exec_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error occurred",
        )
        result = bt.shell_exec("false")
        data = json.loads(result)
        assert data["exit_code"] == 1
        assert "error" in data["stderr"]

    @patch("adk.builtin_tools.subprocess.run")
    def test_shell_exec_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 60", timeout=5)
        result = bt.shell_exec("sleep 60", timeout=5)
        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"].lower()

    @patch("adk.builtin_tools.subprocess.run")
    def test_shell_exec_truncates_long_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="x" * 60_000,
            stderr="e" * 20_000,
        )
        result = bt.shell_exec("big output")
        data = json.loads(result)
        assert len(data["stdout"]) <= 50_000
        assert len(data["stderr"]) <= 10_000


# ---------------------------------------------------------------------------
# Python execution
# ---------------------------------------------------------------------------

class TestPythonExec:
    def test_python_exec_captures_stdout(self):
        result = bt.python_exec("print('hello world')")
        data = json.loads(result)
        assert "hello world" in data["stdout"]

    def test_python_exec_captures_result_var(self):
        result = bt.python_exec("result = 42")
        data = json.loads(result)
        assert data["result"] == 42

    def test_python_exec_captures_errors(self):
        result = bt.python_exec("raise ValueError('boom')")
        data = json.loads(result)
        assert "ValueError" in data["stderr"]
        assert "boom" in data["stderr"]

    def test_python_exec_no_result_var(self):
        result = bt.python_exec("x = 10")
        data = json.loads(result)
        assert "result" not in data

    def test_python_exec_stderr_capture(self):
        result = bt.python_exec("import sys; sys.stderr.write('warning')")
        data = json.loads(result)
        assert "warning" in data["stderr"]


# ---------------------------------------------------------------------------
# Web tools (async)
# ---------------------------------------------------------------------------

class TestWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_success(self):
        mock_html = (
            '<a class="result__a" href="https://example.com">Example Title</a>'
            '<span class="result__snippet">Example snippet text</a>'
        )
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_search("test query", limit=3)
            data = json.loads(result)
            assert data["query"] == "test query"

    @pytest.mark.asyncio
    async def test_web_search_handles_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_search("test")
            data = json.loads(result)
            assert "error" in data


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_web_fetch_success(self):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Hello World</p></body></html>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_fetch("https://example.com")
            assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_web_fetch_truncates(self):
        mock_resp = MagicMock()
        mock_resp.text = "x" * 50_000
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_fetch("https://example.com", max_chars=100)
            assert len(result) <= 100

    @pytest.mark.asyncio
    async def test_web_fetch_strips_html(self):
        mock_resp = MagicMock()
        mock_resp.text = "<script>alert('xss')</script><p>Clean text</p>"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_fetch("https://example.com")
            assert "alert" not in result
            assert "Clean text" in result

    @pytest.mark.asyncio
    async def test_web_fetch_handles_error(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await bt.web_fetch("https://example.com")
            data = json.loads(result)
            assert "error" in data


# ---------------------------------------------------------------------------
# Secrets store
# ---------------------------------------------------------------------------

class TestSecrets:
    def test_secret_set_and_get(self, safe_dir, monkeypatch):
        secrets_file = safe_dir / "secrets.json"
        monkeypatch.setattr(bt, "_SECRETS_FILE", secrets_file)
        bt._secrets_cache = None

        set_result = bt.secret_set("MY_KEY", "my_value")
        data = json.loads(set_result)
        assert data["success"] is True

        bt._secrets_cache = None  # Force reload from disk
        val = bt.secret_get("MY_KEY")
        assert val == "my_value"

    def test_secret_get_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_ENV_SECRET", "env_value")
        val = bt.secret_get("MY_ENV_SECRET")
        assert val == "env_value"

    def test_secret_get_not_found(self, safe_dir, monkeypatch):
        secrets_file = safe_dir / "secrets.json"
        monkeypatch.setattr(bt, "_SECRETS_FILE", secrets_file)
        bt._secrets_cache = None

        val = bt.secret_get("NONEXISTENT_KEY")
        data = json.loads(val)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_secret_list(self, safe_dir, monkeypatch):
        secrets_file = safe_dir / "secrets.json"
        monkeypatch.setattr(bt, "_SECRETS_FILE", secrets_file)
        bt._secrets_cache = None

        bt.secret_set("KEY_A", "a")
        bt.secret_set("KEY_B", "b")

        result = bt.secret_list()
        data = json.loads(result)
        assert data["count"] == 2
        assert "KEY_A" in data["keys"]
        assert "KEY_B" in data["keys"]

    def test_secret_env_takes_priority(self, safe_dir, monkeypatch):
        secrets_file = safe_dir / "secrets.json"
        monkeypatch.setattr(bt, "_SECRETS_FILE", secrets_file)
        bt._secrets_cache = None

        bt.secret_set("DUALKEY", "file_value")
        monkeypatch.setenv("DUALKEY", "env_value")
        assert bt.secret_get("DUALKEY") == "env_value"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_tool_categories_defined(self):
        assert "file_io" in bt.TOOL_CATEGORIES
        assert "shell" in bt.TOOL_CATEGORIES
        assert "python" in bt.TOOL_CATEGORIES
        assert "web" in bt.TOOL_CATEGORIES
        assert "secrets" in bt.TOOL_CATEGORIES

    def test_file_io_has_five_tools(self):
        assert len(bt.TOOL_CATEGORIES["file_io"]) == 5

    def test_register_builtin_tools_explicit_categories(self):
        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent._tools = MagicMock()

        count = bt.register_builtin_tools(mock_agent, categories=["shell"])
        assert count == 1
        mock_agent._tools.register.assert_called_once_with(bt.shell_exec)

    def test_register_builtin_tools_auto_for_demiurge(self):
        mock_agent = MagicMock()
        mock_agent.name = "demiurge"
        mock_agent._tools = MagicMock()

        count = bt.register_builtin_tools(mock_agent)
        expected_cats = bt.IDENTITY_DEFAULTS["demiurge"]
        expected_count = sum(len(bt.TOOL_CATEGORIES[c]) for c in expected_cats)
        assert count == expected_count

    def test_register_builtin_tools_auto_unknown_identity(self):
        mock_agent = MagicMock()
        mock_agent.name = "unknown_agent"
        mock_agent._tools = MagicMock()

        count = bt.register_builtin_tools(mock_agent)
        # Defaults to ["file_io", "web"] for unknown identities
        expected = len(bt.TOOL_CATEGORIES["file_io"]) + len(bt.TOOL_CATEGORIES["web"])
        assert count == expected

    def test_register_builtin_tools_no_auto(self):
        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent._tools = MagicMock()

        count = bt.register_builtin_tools(mock_agent, auto=False)
        # Should register all categories
        total = sum(len(fns) for fns in bt.TOOL_CATEGORIES.values())
        assert count == total

    def test_register_empty_category(self):
        mock_agent = MagicMock()
        mock_agent.name = "test"
        mock_agent._tools = MagicMock()

        count = bt.register_builtin_tools(mock_agent, categories=["nonexistent_cat"])
        assert count == 0

    def test_tool_functions_return_strings(self, safe_dir):
        """Every tool should return a string."""
        f = safe_dir / "test.txt"
        f.write_text("hello", encoding="utf-8")
        assert isinstance(bt.file_read(str(f)), str)
        assert isinstance(bt.file_write(str(f), "content"), str)
        assert isinstance(bt.file_edit(str(f), "content", "new"), str)
        assert isinstance(bt.file_list(str(safe_dir)), str)
        assert isinstance(bt.file_search(str(safe_dir), "*"), str)
        assert isinstance(bt.python_exec("x=1"), str)

    @patch("adk.builtin_tools.subprocess.run")
    def test_shell_exec_returns_string(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        assert isinstance(bt.shell_exec("echo ok"), str)
