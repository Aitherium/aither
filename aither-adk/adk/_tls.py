"""Shared TLS verification policy for adk HTTP clients.

The adk is a public SDK that talks to BOTH public-CA portals and self-signed
LOCAL AitherOS deployments. Historically dozens of HTTP calls used
``verify=False``, which silently disables certificate verification and exposes
auth tokens / secrets to man-in-the-middle. This module is the single source of
truth for the ``verify=`` value, defaulting to *verify* and trusting the
AitherNet internal CA when its bundle is installed.

Policy (``tls_verify()``):
  * ``AITHER_TLS_VERIFY`` in {false,0,no,off}  -> ``False`` (disable checks;
    isolated dev box only — never for auth/secret traffic in production).
  * else, if the AitherNet CA bundle is present -> the bundle path, so internal
    self-signed certs are trusted *with* verification.
  * else -> ``True`` (verify against the system trust store).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def _ca_bundle_path() -> str | None:
    """Locate the AitherNet CA bundle installed by ``adk setup`` / mcp_setup.

    Not cached: setup may install the bundle after the first HTTP call. Never
    raises — TLS policy must not crash an HTTP call (e.g. ``Path.home()`` raises
    when no home dir is determinable, such as a stripped test/CI environment).
    """
    try:
        explicit = os.getenv("AITHER_CA_BUNDLE", "").strip()
        if explicit and Path(explicit).is_file():
            return explicit
        home = (
            os.environ.get("AITHER_HOME")
            or os.environ.get("HOME")
            or os.environ.get("USERPROFILE")
        )
        if not home:
            return None
        # AITHER_HOME points at ~/.aither directly; a bare home needs the suffix.
        base = Path(home)
        for candidate in (base / "aithernet-ca-bundle.pem",
                          base / ".aither" / "aithernet-ca-bundle.pem"):
            if candidate.is_file():
                return str(candidate)
    except Exception:
        return None
    return None


def tls_verify() -> Union[bool, str]:
    """Return the ``verify=`` value for an httpx/requests client.

    Defaults to certificate verification. Returns the AitherNet CA bundle path
    when available so self-signed internal certs are trusted *with* verification.
    Only ``AITHER_TLS_VERIFY=false`` (or 0/no/off) disables verification.
    """
    flag = os.getenv("AITHER_TLS_VERIFY", "true").strip().lower()
    if flag in ("false", "0", "no", "off"):
        return False
    bundle = _ca_bundle_path()
    return bundle if bundle else True
