"""Vault lockbox — an on-demand, human-facing front end for the AitherSecrets vault.

The AitherSecrets service (:8111) already stores every fleet credential, but the
only way to read one used to be a hand-built ``curl`` with the master key pasted
on the command line — which then lingers in shell scrollback. This module is the
lockbox on top of it:

  * The vault master key is sealed ONCE into the OS keychain (Windows Credential
    Manager / macOS Keychain / Secret Service) via ``keyring``. It is encrypted at
    rest by the OS, tied to your login account — that login IS the unlock. The key
    never lives in a plaintext ``.env`` you have to source, and is never echoed.
  * An optional PIN adds a second factor before a *value* is revealed in a session.
  * Values are copied to the clipboard by default and only ever printed with an
    explicit ``--show``, so reading a secret doesn't dump it into scrollback.

Talks DIRECTLY to the live vault (list / get / set / rotate) — this is not the
local encrypted keyring that ``adk secret`` syncs into; it's the real vault.

CLI surface (registered in adk/cli.py as ``adk vault``):

    adk vault setup [--from-env] [--pin]     one-time: seal master key into keychain
    adk vault status                          is it set up, reachable, how many secrets
    adk vault ls [--filter TEXT]              list names + metadata (never values)
    adk vault get NAME [--show] [--copy]      reveal one (clipboard by default)
    adk vault search TERM                     fuzzy name search
    adk vault rotate NAME [--length N] [--show]   mint a fresh strong value + store it
    adk vault lock [--forget]                 drop the session PIN unlock (--forget wipes key)
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets as _secrets
import time
from pathlib import Path
from typing import Optional

try:
    from adk._tls import tls_verify
except Exception:  # pragma: no cover - standalone fallback
    def tls_verify():
        return False

# Keychain service/account identifiers (namespaced so we never collide with
# adk's own login token store).
_KR_SERVICE = "aither-vault"
_KR_KEY_ACCOUNT = "master-key"
_KR_URL_ACCOUNT = "url"
_KR_PIN_ACCOUNT = "pin-sha256"

# Where a *session* unlock (not the key) records its expiry. Holds only a
# timestamp — never the key or the PIN.
_SESSION_FILE = Path.home() / ".aither" / "vault-session.json"
_SESSION_TTL_SECONDS = 15 * 60  # a PIN unlock lasts 15 min

# The operator (a machine holding the vault master key) talks to the vault on the
# loopback fleet address; a logged-in end user reaches the same secrets through the
# authenticated public gateway (federation entry), per the AitherMesh onboarding model.
_DEFAULT_URL = "https://127.0.0.1:8111"
_PUBLIC_GATEWAY = "https://gateway.aitherium.com"


# ── keychain plumbing ────────────────────────────────────────────────────────

def _keyring():
    import keyring  # imported lazily so the rest of adk doesn't hard-depend on it
    return keyring


def _get_stored(account: str) -> Optional[str]:
    try:
        return _keyring().get_password(_KR_SERVICE, account)
    except Exception:
        return None


def _set_stored(account: str, value: str) -> None:
    _keyring().set_password(_KR_SERVICE, account, value)


def _del_stored(account: str) -> None:
    try:
        _keyring().delete_password(_KR_SERVICE, account)
    except Exception:
        pass


def vault_url() -> str:
    explicit = os.environ.get("AITHER_SECRETS_URL") or _get_stored(_KR_URL_ACCOUNT)
    if explicit:
        return explicit.rstrip("/")
    # Operator holds the master key → local fleet vault; otherwise a logged-in user
    # reaches the vault through the public gateway (override with AITHER_GATEWAY_URL).
    if _master_key():
        return _DEFAULT_URL
    return os.environ.get("AITHER_GATEWAY_URL", _PUBLIC_GATEWAY).rstrip("/")


def _master_key() -> Optional[str]:
    # env override wins (useful for CI); otherwise the sealed keychain entry.
    return os.environ.get("AITHER_INTERNAL_SECRET") or _get_stored(_KR_KEY_ACCOUNT)


def is_setup() -> bool:
    return bool(_get_stored(_KR_KEY_ACCOUNT))


# ── tier gate: live vault vs local-only ──────────────────────────────────────
# The live AitherSecrets vault is reachable only for (a) the vault operator — a
# machine that holds the master key (this fleet host) — or (b) an end user who is
# logged in AND on a paid tier (subscription). Everyone else stays LOCAL ONLY: an
# encrypted keyring on this machine (~/.aither/secrets.enc), never the live vault.

def _logged_in() -> bool:
    try:
        from adk.shell.auth import AuthStore
        return bool(AuthStore.get_active_token())  # already checks token expiry
    except Exception:
        return False


def _paid_tier() -> bool:
    """True when the resolved license is a paid tier (rank >= STARTER)."""
    try:
        from adk.licensing import get_license_manager, _TIER_ORDER, Tier
        tier = get_license_manager().license.tier
        return _TIER_ORDER.get(tier, 0) >= _TIER_ORDER.get(Tier.STARTER, 1)
    except Exception:
        return False


def remote_mode() -> tuple[bool, str]:
    """Return (use_live_vault, human_reason). Local-only unless operator or
    logged-in+subscribed."""
    if _master_key():
        return True, "operator (vault master key on this machine)"
    if not _logged_in():
        return False, "not logged in — secrets stay local (run: adk login)"
    if not _paid_tier():
        return False, "free tier — secrets stay local; a subscription unlocks the cloud vault"
    return True, "logged in + subscription"


def _use_remote() -> bool:
    return remote_mode()[0]


# ── PIN / session unlock ─────────────────────────────────────────────────────

def _pin_required() -> bool:
    return bool(_get_stored(_KR_PIN_ACCOUNT))


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(("aither-vault:" + pin).encode()).hexdigest()


def _session_valid() -> bool:
    try:
        data = json.loads(_SESSION_FILE.read_text())
        return float(data.get("expires", 0)) > time.time()
    except Exception:
        return False


def _open_session() -> None:
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(json.dumps({"expires": time.time() + _SESSION_TTL_SECONDS}))


def _clear_session() -> None:
    try:
        _SESSION_FILE.unlink()
    except FileNotFoundError:
        pass


def _ensure_unlocked(now: float = 0.0) -> bool:
    """Gate a value reveal behind the PIN, if one is configured. Returns True when
    it is safe to reveal. Prompts for the PIN if needed and caches a short session.
    """
    if not _pin_required():
        return True
    if _session_valid():
        return True
    stored = _get_stored(_KR_PIN_ACCOUNT)
    for _ in range(3):
        entered = getpass.getpass("Vault PIN: ")
        if _hash_pin(entered) == stored:
            _open_session()
            return True
        print("  Incorrect PIN.")
    return False


# ── vault HTTP ───────────────────────────────────────────────────────────────

def _client():
    import httpx
    key = _master_key()
    headers = {"Content-Type": "application/json"}
    if key:
        # Operator path — the master key authorizes system-wide vault access.
        headers["X-API-Key"] = key
    else:
        # End-user path — the adk login bearer scopes access to their tenant.
        from adk.shell.auth import AuthStore
        tok = AuthStore.get_active_token()
        if not tok:
            raise RuntimeError("Not authorized for the live vault (not logged in).")
        headers["Authorization"] = f"Bearer {tok}"
    return httpx.Client(
        base_url=vault_url(), headers=headers, verify=tls_verify(), timeout=15,
    )


# -- local encrypted keyring backend (the free / offline path) ----------------

def _local_all() -> dict:
    from adk.builtin_tools import _load_secrets
    return dict(_load_secrets())


def _local_set(name: str, value: str) -> bool:
    from adk.builtin_tools import _load_secrets, _save_secrets
    data = _load_secrets()
    data[name] = value
    _save_secrets(data)
    return True


# -- backend-routed operations (pick live vault OR local keyring) -------------

def list_secrets() -> list[dict]:
    if not _use_remote():
        return [{"name": k, "type": "local", "store": "keyring"}
                for k in sorted(_local_all(), key=str.lower)]
    with _client() as c:
        r = c.get("/secrets")
        r.raise_for_status()
        data = r.json()
    if isinstance(data, dict):
        data = data.get("secrets", [])
    out = []
    for item in data:
        if isinstance(item, str):
            out.append({"name": item})
        elif isinstance(item, dict) and item.get("name"):
            out.append(item)
    return sorted(out, key=lambda d: d["name"].lower())


def get_secret(name: str) -> Optional[str]:
    if not _use_remote():
        return _local_all().get(name)
    with _client() as c:
        r = c.get(f"/secrets/{name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    return data if isinstance(data, str) else data.get("value")


def set_secret(name: str, value: str, secret_type: str = "generic") -> bool:
    if not _use_remote():
        return _local_set(name, value)
    with _client() as c:
        r = c.post(
            "/secrets",
            json={
                "name": name,
                "value": value,
                "secret_type": secret_type,
                "access_level": "internal",
                "allowed_services": [],
                "expires_in_days": None,
            },
        )
    return r.status_code in (200, 201)


def vault_reachable() -> tuple[bool, str]:
    # In local-only mode there is no live vault to reach — report the local store.
    if not _use_remote():
        return True, f"local keyring ({len(_local_all())} secrets)"
    try:
        import httpx
        with httpx.Client(verify=tls_verify(), timeout=8) as c:
            r = c.get(vault_url() + "/health")
            if r.status_code == 200:
                return True, r.json().get("service", "ok")
            return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)


# ── presentation helpers ─────────────────────────────────────────────────────

def _mask(value: str) -> str:
    if len(value) <= 6:
        return "*" * len(value)
    return value[:1] + "*" * (len(value) - 4) + value[-3:]


def _to_clipboard(value: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(value)
        return True
    except Exception:
        return False


# ── command handlers (called from adk/cli.py) ────────────────────────────────

def cmd_setup(args) -> int:
    from_env = getattr(args, "from_env", False)
    url = getattr(args, "url", None) or _DEFAULT_URL

    if from_env:
        # Pull the master key out of the fleet .env once, then seal it away.
        env_path = getattr(args, "env_file", None) or _find_env()
        key = _read_env_value(env_path, "AITHER_INTERNAL_SECRET") if env_path else None
        if not key:
            print("Could not find AITHER_INTERNAL_SECRET in .env. "
                  "Pass --env-file, or omit --from-env to paste the key.")
            return 1
    else:
        key = getpass.getpass("Vault master key (X-API-Key): ").strip()
        if not key:
            print("No key entered.")
            return 1

    _set_stored(_KR_URL_ACCOUNT, url.rstrip("/"))
    _set_stored(_KR_KEY_ACCOUNT, key)

    # Verify the sealed key actually authenticates against the live vault.
    os.environ.pop("AITHER_INTERNAL_SECRET", None)  # force use of the sealed copy
    try:
        n = len(list_secrets())
    except Exception as exc:
        _del_stored(_KR_KEY_ACCOUNT)
        print(f"Key stored but it did NOT authenticate against {url}: {exc}")
        print("Nothing sealed. Check the key/URL and retry.")
        return 1

    print(f"Vault sealed into the OS keychain — {n} secrets reachable at {url}.")
    print("The master key is now encrypted at rest by your OS login; no .env needed.")

    if getattr(args, "pin", False):
        while True:
            p1 = getpass.getpass("Set a PIN (guards value reveals): ")
            p2 = getpass.getpass("Confirm PIN: ")
            if p1 and p1 == p2:
                _set_stored(_KR_PIN_ACCOUNT, _hash_pin(p1))
                print("PIN set. Value reveals will prompt for it (15-min sessions).")
                break
            print("  PINs didn't match / empty — try again.")
    print("\nTry:  adk vault get OPTIPLEX_7090_ADMIN_PASSWORD --copy")
    return 0


def cmd_status(args) -> int:
    remote, reason = remote_mode()
    ok, detail = vault_reachable()
    if remote:
        print(f"Mode          : LIVE VAULT — {reason}")
        print(f"Lockbox setup : {'yes (key sealed in OS keychain)' if is_setup() else 'using adk login token'}")
        print(f"Vault URL     : {vault_url()}")
        print(f"Vault health  : {'reachable — ' + detail if ok else 'UNREACHABLE — ' + detail}")
    else:
        print(f"Mode          : LOCAL ONLY — {reason}")
        print(f"Store         : {detail}")
    print(f"PIN gate      : {'on' if _pin_required() else 'off'}")
    try:
        print(f"Secrets       : {len(list_secrets())}")
    except Exception as exc:
        print(f"Secrets       : error — {exc}")
    return 0 if ok else 1


def cmd_ls(args) -> int:
    flt = (getattr(args, "filter", None) or "").lower()
    try:
        items = list_secrets()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if flt:
        items = [i for i in items if flt in i["name"].lower()]
    print(f"\n{len(items)} secret(s){' matching ' + repr(flt) if flt else ''}:")
    print("=" * 60)
    for i in items:
        meta = []
        if i.get("type"):
            meta.append(i["type"])
        if i.get("version"):
            meta.append(f"v{i['version']}")
        suffix = f"   ({', '.join(meta)})" if meta else ""
        print(f"  {i['name']}{suffix}")
    return 0


def cmd_get(args) -> int:
    name = args.name
    show = getattr(args, "show", False)
    copy = getattr(args, "copy", False) or not show  # clipboard is the default
    if show and not _ensure_unlocked():
        print("Locked — PIN required to reveal a value.")
        return 1
    try:
        value = get_secret(name)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    if value is None:
        print(f"No secret named '{name}'.  Try:  adk vault search {name}")
        return 1
    copied = _to_clipboard(value) if copy else False
    if show:
        print(value)
    else:
        tail = "copied to clipboard" if copied else "clipboard unavailable — use --show"
        print(f"{name}: {_mask(value)}  ({len(value)} chars — {tail})")
    return 0


def cmd_search(args) -> int:
    term = args.term.lower()
    try:
        items = list_secrets()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    hits = [i for i in items if term in i["name"].lower()]
    if not hits:
        print(f"No secret name contains '{args.term}'.")
        return 1
    print(f"\n{len(hits)} match(es):")
    for i in hits:
        print(f"  {i['name']}")
    return 0


def cmd_rotate(args) -> int:
    name = args.name
    length = getattr(args, "length", 28) or 28
    show = getattr(args, "show", False)
    try:
        existing = get_secret(name)
    except Exception as exc:
        print(f"Error reading '{name}': {exc}")
        return 1
    if existing is None:
        print(f"No secret named '{name}' to rotate.  (Use adk vault set to create.)")
        return 1
    if not _ensure_unlocked():
        print("Locked — PIN required to rotate.")
        return 1
    new_value = _secrets.token_urlsafe(length)[:length]
    try:
        ok = set_secret(name, new_value)
    except Exception as exc:
        print(f"Error writing new value: {exc}")
        return 1
    if not ok:
        print("Vault rejected the write — nothing changed.")
        return 1
    copied = _to_clipboard(new_value)
    print(f"Rotated '{name}'  (old {_mask(existing)} -> new {_mask(new_value)})")
    print(f"  {'new value copied to clipboard' if copied else 'clipboard unavailable'}")
    if show:
        print(f"  new value: {new_value}")
    print("  NOTE: update any service still using the old value.")
    return 0


def cmd_lock(args) -> int:
    _clear_session()
    if getattr(args, "forget", False):
        _del_stored(_KR_KEY_ACCOUNT)
        _del_stored(_KR_PIN_ACCOUNT)
        print("Master key and PIN wiped from the OS keychain. Re-run setup to use the vault again.")
    else:
        print("Session locked. PIN will be required again on the next reveal.")
    return 0


def dispatch(args) -> int:
    sub = getattr(args, "vault_command", None)
    handlers = {
        "setup": cmd_setup, "status": cmd_status, "ls": cmd_ls, "list": cmd_ls,
        "get": cmd_get, "search": cmd_search, "rotate": cmd_rotate, "lock": cmd_lock,
    }
    fn = handlers.get(sub)
    if not fn:
        print("Usage: adk vault {setup|status|ls|get|search|rotate|lock}")
        print("First time?  adk vault setup --from-env")
        return 1
    return fn(args)


# ── .env discovery (only used by `setup --from-env`) ─────────────────────────

def _find_env() -> Optional[str]:
    # Generic, host-agnostic discovery: an explicit override, then a .env in the
    # working directory or the user's ~/.aither config dir. No hardcoded paths.
    for p in (
        os.environ.get("AITHER_ENV_FILE"),
        ".env",
        str(Path.home() / ".aither" / ".env"),
    ):
        if p and Path(p).is_file():
            return p
    return None


def _read_env_value(path: str, key: str) -> Optional[str]:
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line[len(key) + 1:].strip().strip('"').strip("'")
    except Exception:
        return None
    return None
