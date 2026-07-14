"""OfficeVault — a recoverable, local-first secrets vault for FormBridge.

Reconciles two goals that look opposed:
  * ZERO-KNOWLEDGE to the operator (we never see plaintext — no key/value off-box);
  * RECOVERABLE by the doctor (never permanently locked — they can query any
    stored password) — like a normal secrets manager.

The reconciliation is the master-key + recovery-code pattern:

    master passphrase ─┐
                       ├─► unwraps ─► MASTER KEY (memory only) ─► decrypts:
    recovery code  ────┘  (backup)                                · the keyring of
                                                                    per-doc passwords
                                                                  · the capture-store key

  - The MASTER KEY (MK) is random and wrapped TWICE: by a key derived from the
    daily passphrase AND by a key derived from a one-time recovery code the
    doctor backs up. Either unlocks MK, so a forgotten passphrase is recoverable.
  - The KEYRING is a queryable map {name -> secret}, each value encrypted with MK
    (e.g. auto-generated per-document PDF passwords). With MK unlocked the doctor
    can list/get any of them — a document is never lost.
  - ``encrypt``/``decrypt`` (used by the capture store) also key off MK.
  - ``vault.json`` holds only ciphertext (wrapped MK + encrypted secrets) — it is
    useless to anyone without the passphrase or the recovery code.

Crypto mirrors ``adk/private_companion.py`` (Fernet + PBKDF2-HMAC-SHA256).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets as _secrets
import threading
from pathlib import Path

logger = logging.getLogger("adk.formbridge.vault")

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:  # graceful degrade — vault stays "unavailable"
    HAS_CRYPTO = False
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore

ENC_PREFIX = "fbenc:v1:"   # marks encrypted capture values; plaintext passes through
_KDF_ITERATIONS = 200_000
_CAPTURE_KEY = "__capture__"   # reserved keyring entry: the capture-store data key


class VaultError(RuntimeError):
    """Vault operation failed (bad passphrase/recovery, locked, crypto missing)."""


def _vault_path() -> Path:
    override = os.getenv("AITHER_FORMBRIDGE_DIR", "").strip()
    base = Path(override) if override else (Path.home() / ".aither" / "formbridge")
    return base / "vault.json"


def _derive(secret: str, salt: bytes, iterations: int = _KDF_ITERATIONS) -> bytes:
    """Derive a Fernet key from a secret (passphrase or recovery code) + salt."""
    if not HAS_CRYPTO:
        raise VaultError("cryptography package required for the FormBridge vault")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
    return base64.urlsafe_b64encode(kdf.derive(secret.encode("utf-8")))


def _new_recovery_code() -> str:
    """A high-entropy, human-transcribable recovery code (~100 bits)."""
    raw = base64.b32encode(_secrets.token_bytes(13)).decode("ascii").rstrip("=")
    return "-".join(raw[i:i + 5] for i in range(0, len(raw), 5))


class OfficeVault:
    """Recoverable shared-key secrets vault. Thread-safe."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else _vault_path()
        self._mk: bytes | None = None        # master key — memory only when unlocked
        self._lock = threading.RLock()

    # ── State ──

    @property
    def available(self) -> bool:
        return HAS_CRYPTO

    @property
    def initialized(self) -> bool:
        return self._path.is_file()

    @property
    def unlocked(self) -> bool:
        return self._mk is not None

    def status(self) -> dict:
        return {
            "available": self.available,
            "initialized": self.initialized,
            "unlocked": self.unlocked,
        }

    # ── Persistence ──

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    # ── Lifecycle ──

    def initialize(self, passphrase: str) -> str:
        """First-time setup. Generates the master key, wraps it by the passphrase
        AND by a fresh recovery code, persists, auto-unlocks. Returns the recovery
        code ONCE — the doctor must back it up (it is never stored in the clear)."""
        if not HAS_CRYPTO:
            raise VaultError("cryptography package required for the FormBridge vault")
        if not passphrase:
            raise VaultError("a passphrase is required")
        with self._lock:
            if self.initialized:
                raise VaultError("vault already initialized — use unlock or rotate")
            mk = Fernet.generate_key()
            recovery_code = _new_recovery_code()
            pass_salt = os.urandom(16)
            rec_salt = os.urandom(16)
            data = {
                "version": 2,
                "kdf_iterations": _KDF_ITERATIONS,
                "pass_salt": base64.b64encode(pass_salt).decode("ascii"),
                "recovery_salt": base64.b64encode(rec_salt).decode("ascii"),
                "mk_by_pass": Fernet(_derive(passphrase, pass_salt)).encrypt(mk).decode("ascii"),
                "mk_by_recovery": Fernet(_derive(recovery_code, rec_salt)).encrypt(mk).decode("ascii"),
                "secrets": {},
            }
            self._save(data)
            self._mk = mk
            return recovery_code

    def unlock(self, passphrase: str) -> None:
        self._unlock_with("mk_by_pass", "pass_salt", passphrase, "incorrect office passphrase")

    def unlock_with_recovery(self, recovery_code: str) -> None:
        self._unlock_with("mk_by_recovery", "recovery_salt", recovery_code.strip(),
                          "incorrect recovery code")

    def _unlock_with(self, wrap_key: str, salt_key: str, secret: str, err: str) -> None:
        if not self.initialized:
            raise VaultError("vault not initialized")
        with self._lock:
            data = self._read()
            salt = base64.b64decode(data[salt_key])
            iters = int(data.get("kdf_iterations", _KDF_ITERATIONS))
            kdf = _derive(secret, salt, iters)
            try:
                self._mk = Fernet(kdf).decrypt(data[wrap_key].encode("ascii"))
            except InvalidToken as e:
                raise VaultError(err) from e

    def lock(self) -> None:
        with self._lock:
            self._mk = None

    def rotate(self, old_passphrase: str, new_passphrase: str) -> None:
        """Change the daily passphrase — re-wrap the master key. Instant; all
        encrypted data and stored secrets stay valid (MK is unchanged)."""
        if not new_passphrase:
            raise VaultError("a new passphrase is required")
        with self._lock:
            self.unlock(old_passphrase)        # validates old
            data = self._read()
            pass_salt = os.urandom(16)
            data["pass_salt"] = base64.b64encode(pass_salt).decode("ascii")
            data["mk_by_pass"] = Fernet(_derive(new_passphrase, pass_salt)).encrypt(self._mk).decode("ascii")
            self._save(data)

    def regenerate_recovery(self) -> str:
        """Issue a NEW recovery code (e.g. the old one was lost). Requires unlock."""
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            data = self._read()
            code = _new_recovery_code()
            rec_salt = os.urandom(16)
            data["recovery_salt"] = base64.b64encode(rec_salt).decode("ascii")
            data["mk_by_recovery"] = Fernet(_derive(code, rec_salt)).encrypt(self._mk).decode("ascii")
            self._save(data)
            return code

    # ── Capture-store encryption (MK-keyed; store.py uses these) ──

    @staticmethod
    def is_encrypted(value: str) -> bool:
        return isinstance(value, str) and value.startswith(ENC_PREFIX)

    def encrypt(self, plaintext: str) -> str:
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            return ENC_PREFIX + Fernet(self._mk).encrypt(str(plaintext).encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        if not self.is_encrypted(value):
            return value
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            return Fernet(self._mk).decrypt(value[len(ENC_PREFIX):].encode("ascii")).decode("utf-8")

    # ── Keyring (recoverable named secrets — per-doc passwords etc.) ──

    def set_secret(self, name: str, value: str) -> None:
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            data = self._read()
            data.setdefault("secrets", {})[name] = Fernet(self._mk).encrypt(value.encode("utf-8")).decode("ascii")
            self._save(data)

    def get_secret(self, name: str) -> str | None:
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            token = self._read().get("secrets", {}).get(name)
            if token is None:
                return None
            return Fernet(self._mk).decrypt(token.encode("ascii")).decode("utf-8")

    def generate_secret(self, name: str, *, length: int = 16) -> str:
        """Idempotent: return the existing secret for *name*, or generate, store,
        and return a new random one. So a given document keeps a stable password."""
        with self._lock:
            existing = self.get_secret(name)
            if existing is not None:
                return existing
            value = _secrets.token_urlsafe(length)
            self.set_secret(name, value)
            return value

    def list_secret_names(self, *, include_reserved: bool = False) -> list[str]:
        names = sorted(self._read().get("secrets", {}).keys()) if self.initialized else []
        return names if include_reserved else [n for n in names if not n.startswith("__")]

    def delete_secret(self, name: str) -> bool:
        with self._lock:
            data = self._read()
            if name in data.get("secrets", {}):
                del data["secrets"][name]
                self._save(data)
                return True
            return False

    def rotate_secret(self, name: str, *, length: int = 16) -> str:
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            value = _secrets.token_urlsafe(length)
            self.set_secret(name, value)
            return value

    def rotate_all_secrets(self, *, length: int = 16) -> dict[str, str]:
        """Mass change: regenerate every non-reserved secret. Returns the new
        {name: value} map so the caller can re-encrypt the affected documents."""
        with self._lock:
            if self._mk is None:
                raise VaultError("vault is locked")
            out: dict[str, str] = {}
            for name in self.list_secret_names():
                out[name] = self.rotate_secret(name, length=length)
            return out


_vault: OfficeVault | None = None
_vault_lock = threading.Lock()


def get_vault() -> OfficeVault:
    """Process-wide OfficeVault singleton."""
    global _vault
    with _vault_lock:
        if _vault is None:
            _vault = OfficeVault()
        return _vault


def reset_vault() -> None:
    """Test helper — drop the singleton so the next get_vault() re-reads env."""
    global _vault
    with _vault_lock:
        _vault = None
