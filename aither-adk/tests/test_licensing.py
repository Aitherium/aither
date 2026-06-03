"""Open-core licensing keystone — fail-closed + genuinely-useful-free tests."""

from __future__ import annotations

import base64
import json

import pytest

import adk.licensing as lic
from adk.licensing import LicenseError, Tier, get_license_manager, reset_license_manager


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts from a pristine, license-free environment."""
    for var in (
        "AITHER_TENANT_SLUG", "AITHER_LICENSE_KEY", "AITHER_LICENSE_FILE",
        "AITHER_LICENSE_ENFORCE", "AITHER_LICENSE_PUBLIC_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_license_manager()
    yield
    reset_license_manager()


# ── default = free, fail-closed ────────────────────────────────────────────

def test_default_is_community_free_tier():
    lm = get_license_manager()
    assert lm.license.tier is Tier.COMMUNITY
    assert lm.license.source == "default"


def test_free_tier_is_genuinely_useful():
    """COMMUNITY must keep the core agent + memory usable — the funnel needs it."""
    lm = get_license_manager()
    assert lm.is_agent_licensed("aither") is True
    assert lm.is_agent_licensed("assistant") is True
    assert lm.max_effort() == 3  # small models, but real work


def test_free_tier_fences_off_scale_capabilities():
    lm = get_license_manager()
    assert lm.can_use_fleet() is False
    assert lm.can_use_channels() is False
    assert lm.can_use_auto_neurons() is False
    assert lm.can_use_cron() is False
    assert lm.can_use_swarm() is False
    assert lm.can_build_custom_agents() is False
    assert lm.is_agent_licensed("hydra") is False  # premium identity


def test_require_raises_on_unlicensed_capability():
    lm = get_license_manager()
    with pytest.raises(LicenseError) as exc:
        lm.require("fleet", friendly="Fleet mode")
    assert "portal.aitherium.com" in str(exc.value)


def test_effort_is_clamped_for_free_tier():
    lm = get_license_manager()
    assert lm.clamp_effort(10) == (3, True)
    assert lm.clamp_effort(2) == (2, False)
    assert lm.clamp_effort(None) == (None, False)


# ── internal dogfood = unrestricted ────────────────────────────────────────

def test_internal_slug_unlocks_everything(monkeypatch):
    monkeypatch.setenv("AITHER_TENANT_SLUG", "aitherium")
    reset_license_manager()
    lm = get_license_manager()
    assert lm.license.tier is Tier.INTERNAL
    assert lm.can_use_fleet() and lm.can_use_swarm() and lm.can_use_channels()
    assert lm.max_effort() == 10
    assert lm.is_agent_licensed("any-agent-name") is True
    lm.require("fleet")  # must not raise


# ── explicit opt-out ───────────────────────────────────────────────────────

def test_enforcement_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AITHER_LICENSE_ENFORCE", "0")
    reset_license_manager()
    lm = get_license_manager()
    lm.require("fleet")  # no raise
    assert lm.max_effort() == 10  # cap lifted when not enforced


# ── unsigned/forged licenses are rejected (fail-closed) ────────────────────

def test_unsigned_license_falls_back_to_free(monkeypatch, tmp_path):
    """A license.json with an invalid signature must NOT grant premium."""
    payload = {
        "tier": "enterprise",
        "tenant_id": "evil",
        "entitlements": {"fleet": True, "max_effort": 10},
    }
    envelope = {
        "payload": base64.b64encode(json.dumps(payload).encode()).decode(),
        "signature": "deadbeef" * 8,  # bogus
    }
    lf = tmp_path / "license.json"
    lf.write_text(json.dumps(envelope), encoding="utf-8")
    monkeypatch.setenv("AITHER_LICENSE_FILE", str(lf))
    reset_license_manager()

    lm = get_license_manager()
    assert lm.license.tier is Tier.COMMUNITY  # forged tier ignored
    assert lm.can_use_fleet() is False


def test_missing_crypto_or_placeholder_key_rejects_signatures():
    """With the placeholder public key, no signature can verify."""
    assert lic._verify_signature(b"anything", "00" * 64) is False
