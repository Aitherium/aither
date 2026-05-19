"""Filesystem-scoped sandbox primitives.

Complements :mod:`adk.sandbox` (which handles tool-level capability +
subprocess isolation) with helpers an agent can use *inside* the
sandbox to do safe file I/O and shell-out — without ever escaping the
allowed root.

::

    from adk.fs_sandbox import FSGuard
    from adk.sandbox import Capability

    guard = FSGuard("/tmp/work", capabilities={Capability.FILESYSTEM, Capability.EXEC})

    guard.write("notes.md", "hello")
    body = guard.read("notes.md")
    out = guard.run(["python", "-c", "print('ok')"], timeout=5)

Everything is path-confined: relative writes resolve under the root,
absolute paths or ``..`` escapes raise :class:`PathEscape`. ``run``
rejects shell metacharacters and never executes with ``shell=True``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from adk.sandbox import Capability

logger = logging.getLogger("adk.fs_sandbox")


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class PathEscape(ValueError):
    """Raised when a requested path resolves outside the sandbox root."""


class CapabilityDenied(PermissionError):
    """Raised when the guard lacks a required capability."""


class UnsafeArgv(ValueError):
    """Raised when subprocess argv contains shell metacharacters."""


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunResult:
    """Outcome of :meth:`FSGuard.run`."""

    returncode: int
    stdout: str
    stderr: str


# ─────────────────────────────────────────────────────────────────────────────
# Metachar policy
# ─────────────────────────────────────────────────────────────────────────────

# Characters that are only dangerous if the caller used ``shell=True``.
# We never do, so we only flag a tight set that survives even in argv-mode
# environments (e.g. when env var expansion or redirect-style strings sneak
# into an argument and the program happens to interpret them).
_SHELL_METACHARS = frozenset(";|&`$<>\n\r")


def _argv_is_safe(argv: Sequence[str]) -> bool:
    return not any(any(c in _SHELL_METACHARS for c in tok) for tok in argv)


# ─────────────────────────────────────────────────────────────────────────────
# FSGuard
# ─────────────────────────────────────────────────────────────────────────────


def _coerce_caps(caps: Iterable[Any] | None) -> frozenset[str]:
    if caps is None:
        return frozenset()
    return frozenset(getattr(c, "value", c) for c in caps)


class FSGuard:
    """Path-confined filesystem + subprocess helper.

    Parameters
    ----------
    root:
        Directory under which all paths must resolve. Created if missing.
    capabilities:
        Iterable of :class:`adk.sandbox.Capability` (or bare strings) that
        the guard is allowed to use. ``FILESYSTEM`` is required for any
        read/write; ``EXEC`` for :meth:`run`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        capabilities: Iterable[Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._caps = _coerce_caps(capabilities)

    # ── Capabilities ────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> frozenset[str]:
        return self._caps

    def has(self, cap: Any) -> bool:
        return getattr(cap, "value", cap) in self._caps

    def _require(self, cap: Any) -> None:
        name = getattr(cap, "value", cap)
        if name not in self._caps:
            raise CapabilityDenied(f"FSGuard lacks capability: {name}")

    # ── Path confinement ────────────────────────────────────────────────────

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against the root, raising on escape."""
        p = Path(path)
        if p.is_absolute():
            raise PathEscape(f"absolute path not allowed: {path}")
        target = (self.root / p).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PathEscape(f"path escapes sandbox root: {path}") from exc
        return target

    def contains(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
        except PathEscape:
            return False
        return True

    # ── Read / write ────────────────────────────────────────────────────────

    def read(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        self._require(Capability.FILESYSTEM)
        target = self.resolve(path)
        return target.read_text(encoding=encoding)

    def read_bytes(self, path: str | Path) -> bytes:
        self._require(Capability.FILESYSTEM)
        return self.resolve(path).read_bytes()

    def write(
        self,
        path: str | Path,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        self._require(Capability.FILESYSTEM)
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return target

    def write_bytes(self, path: str | Path, content: bytes) -> Path:
        self._require(Capability.FILESYSTEM)
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def list(self, path: str | Path = "") -> list[Path]:
        self._require(Capability.FILESYSTEM)
        target = self.resolve(path) if path else self.root
        return sorted(target.iterdir())

    # ── Subprocess ──────────────────────────────────────────────────────────

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = 30.0,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> RunResult:
        """Execute ``argv`` in the sandbox root.

        Always runs with ``shell=False``. Rejects argv containing shell
        metacharacters. cwd defaults to the sandbox root and is confined
        to it.
        """
        self._require(Capability.EXEC)
        if not argv:
            raise ValueError("argv must not be empty")
        if not _argv_is_safe(argv):
            raise UnsafeArgv(f"argv contains shell metachars: {argv!r}")
        work_dir = self.resolve(cwd) if cwd is not None else self.root
        proc_env = dict(os.environ if env is None else env)
        try:
            completed = subprocess.run(  # noqa: S603 - shell=False + metachar guard
                list(argv),
                cwd=str(work_dir),
                env=proc_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"sandbox run exceeded {timeout}s: {argv!r}") from exc
        return RunResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience helpers
# ─────────────────────────────────────────────────────────────────────────────


def safe_read(guard: FSGuard, path: str | Path) -> str:
    return guard.read(path)


def safe_write(guard: FSGuard, path: str | Path, content: str) -> Path:
    return guard.write(path, content)


def safe_run(guard: FSGuard, argv: Sequence[str], **kwargs: Any) -> RunResult:
    return guard.run(argv, **kwargs)


__all__ = [
    "Capability",  # re-export for convenience
    "CapabilityDenied",
    "FSGuard",
    "PathEscape",
    "RunResult",
    "UnsafeArgv",
    "safe_read",
    "safe_run",
    "safe_write",
]
