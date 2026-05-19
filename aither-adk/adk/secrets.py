"""Secret resolution — pluggable stores with capability-checked reads.

Replaces ad-hoc ``os.getenv`` scatter with a tiny, swappable interface.
Every read is gated through :class:`~adk.sandbox.Capability` so untrusted
tools cannot exfiltrate credentials by accident.

Stores
------
* :class:`EnvStore`     — reads ``os.environ`` (optional uppercase prefix)
* :class:`FileStore`    — JSON file, cached, supports hot-reload
* :class:`KeyringStore` — OS keychain via the optional ``keyring`` package
* :class:`ChainStore`   — first-hit wins across an ordered list of stores

Usage
-----
::

    from adk.secrets import handle, resolve, use_store, FileStore

    # Default store reads from the environment.
    api_key = resolve("OPENAI_API_KEY")

    # Swap stores in a scope (tests, dev shells):
    with use_store(FileStore("/etc/aither/secrets.json")):
        token = resolve(handle("github"))

The :class:`SecretHandle` wrapper hides the value in ``repr()`` so secrets
never leak into logs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

logger = logging.getLogger("adk.secrets")


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────


class SecretNotFound(KeyError):
    """Raised when a secret cannot be located in the active store."""


class SecretCapabilityDenied(PermissionError):
    """Raised when the caller lacks the ``secrets`` capability."""


# ─────────────────────────────────────────────────────────────────────────────
# Handles
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SecretHandle:
    """Reference to a secret. Holds only the lookup key, never the value."""

    key: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SecretHandle(key={self.key!r})"


def handle(key: str) -> SecretHandle:
    """Build a :class:`SecretHandle` for ``key``."""
    return SecretHandle(key=key)


# ─────────────────────────────────────────────────────────────────────────────
# Store protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class SecretStore(Protocol):
    """Storage backend protocol. All methods are synchronous."""

    name: str

    def get(self, key: str) -> str:
        ...

    def has(self, key: str) -> bool:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# EnvStore
# ─────────────────────────────────────────────────────────────────────────────


class EnvStore:
    """Read secrets from ``os.environ``.

    If ``prefix`` is set, the lookup key is uppercased and prefixed.
    Example: ``EnvStore(prefix="AITHER_SECRET_").get("openai")`` returns
    ``os.environ["AITHER_SECRET_OPENAI"]``.
    """

    name = "env"

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def _resolved_key(self, key: str) -> str:
        return f"{self.prefix}{key.upper()}" if self.prefix else key

    def has(self, key: str) -> bool:
        return self._resolved_key(key) in os.environ

    def get(self, key: str) -> str:
        try:
            return os.environ[self._resolved_key(key)]
        except KeyError as exc:
            raise SecretNotFound(key) from exc


# ─────────────────────────────────────────────────────────────────────────────
# FileStore
# ─────────────────────────────────────────────────────────────────────────────


class FileStore:
    """Read secrets from a JSON file on disk.

    The file is loaded lazily on first access and cached in memory. Call
    :meth:`reload` to pick up edits.
    """

    name = "file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: dict[str, str] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is not None:
                return self._cache
            try:
                raw = self.path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("secret file must be a JSON object")
                self._cache = {str(k): str(v) for k, v in parsed.items()}
            except FileNotFoundError:
                self._cache = {}
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("secrets.file.parse_failed: %s (%s)", self.path, exc)
                self._cache = {}
            return self._cache

    def has(self, key: str) -> bool:
        return key in self._load()

    def get(self, key: str) -> str:
        data = self._load()
        if key not in data:
            raise SecretNotFound(key)
        return data[key]

    def reload(self) -> None:
        """Drop the cache so the next access re-reads the file."""
        with self._lock:
            self._cache = None


# ─────────────────────────────────────────────────────────────────────────────
# KeyringStore (optional dep)
# ─────────────────────────────────────────────────────────────────────────────


class KeyringStore:
    """Read secrets from the OS keychain via the ``keyring`` package.

    The ``keyring`` import is lazy so this module remains importable even
    when the optional dependency is absent.
    """

    name = "keyring"

    def __init__(self, service: str = "aither") -> None:
        self.service = service
        self._keyring = None

    def _kr(self):
        if self._keyring is None:
            try:
                import keyring  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise SecretNotFound("keyring package not installed") from exc
            self._keyring = keyring
        return self._keyring

    def has(self, key: str) -> bool:
        try:
            return self._kr().get_password(self.service, key) is not None
        except Exception:
            return False

    def get(self, key: str) -> str:
        value = self._kr().get_password(self.service, key)
        if value is None:
            raise SecretNotFound(key)
        return value


# ─────────────────────────────────────────────────────────────────────────────
# ChainStore
# ─────────────────────────────────────────────────────────────────────────────


class ChainStore:
    """Try a list of stores in order; first hit wins."""

    name = "chain"

    def __init__(self, stores: Sequence[SecretStore]) -> None:
        if not stores:
            raise ValueError("ChainStore requires at least one store")
        self.stores: tuple[SecretStore, ...] = tuple(stores)

    def has(self, key: str) -> bool:
        return any(s.has(key) for s in self.stores)

    def get(self, key: str) -> str:
        for s in self.stores:
            if s.has(key):
                return s.get(key)
        raise SecretNotFound(key)


# ─────────────────────────────────────────────────────────────────────────────
# Active-store registry
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_STORE: SecretStore = EnvStore()
_active: contextvars.ContextVar[SecretStore | None] = contextvars.ContextVar(
    "adk_secret_store", default=None
)


def get_store() -> SecretStore:
    """Return the currently-active secret store."""
    return _active.get() or _DEFAULT_STORE


def set_default_store(store: SecretStore) -> None:
    """Replace the process-wide default store."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


@contextmanager
def use_store(store: SecretStore) -> Iterator[SecretStore]:
    """Scope an override store for the current async/sync context."""
    token = _active.set(store)
    try:
        yield store
    finally:
        _active.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# Capability-checked resolve
# ─────────────────────────────────────────────────────────────────────────────


def _check_capability(allowed: Any) -> None:
    """Verify the caller holds the ``secrets`` capability.

    ``allowed`` is the capability set passed by the caller — typically a
    ``set[Capability]`` or ``frozenset[str]`` taken from a Sandbox. When
    ``None`` (no sandbox in scope), enforcement is skipped to keep
    simple scripts ergonomic. Tests can force-deny by passing an empty
    set.
    """
    if allowed is None:
        return
    # Accept both Capability enum members and bare strings.
    flat = {getattr(c, "value", c) for c in allowed}
    if "secrets" not in flat:
        raise SecretCapabilityDenied(
            "tool lacks 'secrets' capability; cannot read protected values"
        )


def resolve(
    key: str | SecretHandle,
    *,
    store: SecretStore | None = None,
    capabilities: Any = None,
) -> str:
    """Resolve a secret by key or :class:`SecretHandle`.

    Parameters
    ----------
    key:
        The lookup key or a :class:`SecretHandle`.
    store:
        Override store for this call. Defaults to :func:`get_store`.
    capabilities:
        The caller's capability set (typically ``sandbox.capabilities``).
        Pass an empty set to force a denial. Pass ``None`` to skip the
        check (default for script-level use).
    """
    _check_capability(capabilities)
    if isinstance(key, SecretHandle):
        key = key.key
    target = store if store is not None else get_store()
    return target.get(key)


__all__ = [
    "ChainStore",
    "EnvStore",
    "FileStore",
    "KeyringStore",
    "SecretCapabilityDenied",
    "SecretHandle",
    "SecretNotFound",
    "SecretStore",
    "get_store",
    "handle",
    "resolve",
    "set_default_store",
    "use_store",
]
