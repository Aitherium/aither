"""Regression tests for the phone-ready secure chat URL (`adk up` tunnel access).

The bearer must ride in the URL fragment (#k=…) so it is never sent in a header or
logged, and the surface stays fail-closed (a URL without the fragment is 401). The
QR is a Unicode convenience that must degrade — never crash — on a terminal that
can't render its glyphs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adk.cli import _render_phone_access


def test_phone_url_carries_token_in_fragment():
    out = _render_phone_access("https://happy-cat.trycloudflare.com", "TOK3N")
    assert "https://happy-cat.trycloudflare.com/#k=TOK3N" in out
    # trailing slash on the host is normalized, not doubled
    out2 = _render_phone_access("https://happy-cat.trycloudflare.com/", "TOK3N")
    assert "https://happy-cat.trycloudflare.com/#k=TOK3N" in out2


def test_phone_access_explains_fail_closed():
    out = _render_phone_access("https://x.trycloudflare.com", "TKN")
    assert "401" in out and "token is in the link" in out


def test_phone_access_qr_renders_when_lib_present():
    import pytest

    pytest.importorskip("qrcode")
    out = _render_phone_access("https://x.trycloudflare.com", "TKN")
    assert any(glyph in out for glyph in ("█", "▀", "▄")), "expected a rendered QR"


def test_phone_access_never_crashes_and_returns_str():
    # Always a string; the URL is always present even if the QR is dropped.
    out = _render_phone_access("https://x.trycloudflare.com", "TKN")
    assert isinstance(out, str)
    assert "/#k=TKN" in out
