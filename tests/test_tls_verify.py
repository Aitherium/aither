"""TLS verification policy + a ratchet against bare ``verify=False``.

The adk is a public SDK; ``verify=False`` disables certificate verification and
exposes auth tokens / secrets to MITM. All HTTP calls must route their verify
value through ``adk._tls.tls_verify()`` (or the template's local equivalent),
which defaults to verifying and trusts the AitherNet CA bundle when present.
"""

import os
import re
from pathlib import Path

import pytest

from adk._tls import tls_verify

ADK_ROOT = Path(__file__).resolve().parent.parent / "adk"
_BARE_VERIFY_FALSE = re.compile(r"verify\s*=\s*False\b")


def test_no_bare_verify_false_in_adk_package():
    """No source file may hardcode verify=False (use tls_verify())."""
    offenders = []
    for py in ADK_ROOT.rglob("*.py"):
        if py.name == "_tls.py":
            continue  # the helper's docstring documents the antipattern by name
        text = py.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if _BARE_VERIFY_FALSE.search(line):
                offenders.append(f"{py.relative_to(ADK_ROOT.parent)}:{i}")
    assert not offenders, (
        "Bare verify=False found (disables TLS verification — MITM risk). "
        "Use tls_verify() instead:\n  " + "\n  ".join(offenders)
    )


def test_tls_verify_defaults_to_true(monkeypatch, tmp_path):
    monkeypatch.delenv("AITHER_TLS_VERIFY", raising=False)
    monkeypatch.delenv("AITHER_CA_BUNDLE", raising=False)
    monkeypatch.setenv("AITHER_HOME", str(tmp_path))  # no bundle present
    assert tls_verify() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE", "Off"])
def test_tls_verify_can_be_disabled_for_dev(monkeypatch, val):
    monkeypatch.setenv("AITHER_TLS_VERIFY", val)
    assert tls_verify() is False


def test_tls_verify_uses_ca_bundle_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("AITHER_TLS_VERIFY", raising=False)
    monkeypatch.delenv("AITHER_CA_BUNDLE", raising=False)
    monkeypatch.setenv("AITHER_HOME", str(tmp_path))
    bundle = tmp_path / "aithernet-ca-bundle.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    assert tls_verify() == str(bundle)


def test_explicit_ca_bundle_env_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("AITHER_TLS_VERIFY", raising=False)
    explicit = tmp_path / "custom-ca.pem"
    explicit.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("AITHER_CA_BUNDLE", str(explicit))
    assert tls_verify() == str(explicit)
