"""Tests for adk.fs_sandbox."""

from __future__ import annotations

import sys

import pytest

from adk.fs_sandbox import (
    CapabilityDenied,
    FSGuard,
    PathEscape,
    UnsafeArgv,
    safe_read,
    safe_run,
    safe_write,
)
from adk.sandbox import Capability


@pytest.fixture
def root(tmp_path):
    return tmp_path / "sb"


def _all_caps():
    return {Capability.FILESYSTEM, Capability.EXEC}


class TestPathConfinement:
    def test_relative_resolves_under_root(self, root):
        g = FSGuard(root, capabilities=_all_caps())
        target = g.resolve("notes/a.txt")
        assert str(target).startswith(str(g.root))

    def test_absolute_rejected(self, root):
        g = FSGuard(root, capabilities=_all_caps())
        with pytest.raises(PathEscape):
            g.resolve("C:/etc/passwd" if sys.platform == "win32" else "/etc/passwd")

    def test_dotdot_escape_rejected(self, root):
        g = FSGuard(root, capabilities=_all_caps())
        with pytest.raises(PathEscape):
            g.resolve("../outside.txt")

    def test_nested_dotdot_inside_allowed(self, root):
        g = FSGuard(root, capabilities=_all_caps())
        target = g.resolve("a/b/../c.txt")
        assert target.name == "c.txt"

    def test_contains_predicate(self, root):
        g = FSGuard(root, capabilities=_all_caps())
        assert g.contains("inside.txt")
        assert not g.contains("../escape.txt")


class TestReadWrite:
    def test_round_trip(self, root):
        g = FSGuard(root, capabilities={Capability.FILESYSTEM})
        g.write("a.txt", "hello")
        assert g.read("a.txt") == "hello"

    def test_creates_parents(self, root):
        g = FSGuard(root, capabilities={Capability.FILESYSTEM})
        g.write("nested/dir/file.txt", "x")
        assert g.read("nested/dir/file.txt") == "x"

    def test_bytes_round_trip(self, root):
        g = FSGuard(root, capabilities={Capability.FILESYSTEM})
        g.write_bytes("b.bin", b"\x00\x01\x02")
        assert g.read_bytes("b.bin") == b"\x00\x01\x02"

    def test_read_denied_without_capability(self, root):
        g = FSGuard(root, capabilities=set())
        with pytest.raises(CapabilityDenied):
            g.read("anything")

    def test_write_denied_without_capability(self, root):
        g = FSGuard(root, capabilities=set())
        with pytest.raises(CapabilityDenied):
            g.write("anything", "x")


class TestRun:
    def test_executes_argv(self, root):
        g = FSGuard(root, capabilities={Capability.EXEC, Capability.FILESYSTEM})
        out = g.run([sys.executable, "-c", "print('ok')"])
        assert out.returncode == 0
        assert "ok" in out.stdout

    def test_requires_exec_capability(self, root):
        g = FSGuard(root, capabilities={Capability.FILESYSTEM})
        with pytest.raises(CapabilityDenied):
            g.run([sys.executable, "-c", "print('x')"])

    def test_empty_argv_rejected(self, root):
        g = FSGuard(root, capabilities={Capability.EXEC})
        with pytest.raises(ValueError):
            g.run([])

    def test_metachars_rejected(self, root):
        g = FSGuard(root, capabilities={Capability.EXEC})
        with pytest.raises(UnsafeArgv):
            g.run([sys.executable, "-c", "print('ok'); print('two')"])

    def test_cwd_is_sandbox_root(self, root):
        g = FSGuard(root, capabilities={Capability.EXEC, Capability.FILESYSTEM})
        g.write("marker.txt", "hi")
        g.write("check.py", "import os\nprint(os.listdir('.'))\n")
        out = g.run([sys.executable, "check.py"])
        assert "marker.txt" in out.stdout


class TestConvenience:
    def test_safe_read_write_run(self, root):
        g = FSGuard(root, capabilities={Capability.FILESYSTEM, Capability.EXEC})
        safe_write(g, "x.txt", "data")
        assert safe_read(g, "x.txt") == "data"
        out = safe_run(g, [sys.executable, "-c", "print('ok')"])
        assert out.returncode == 0
