"""Tests for adk.secrets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adk.sandbox import Capability
from adk.secrets import (
    ChainStore,
    EnvStore,
    FileStore,
    SecretCapabilityDenied,
    SecretHandle,
    SecretNotFound,
    handle,
    resolve,
    use_store,
)


class TestSecretHandle:
    def test_repr_hides_value(self):
        h = handle("openai")
        assert "openai" in repr(h)
        assert isinstance(h, SecretHandle)


class TestEnvStore:
    def test_get_uppercased_prefix(self, monkeypatch):
        monkeypatch.setenv("ADK_TEST_OPENAI", "sk-env")
        s = EnvStore(prefix="ADK_TEST_")
        assert s.has("openai")
        assert s.get("openai") == "sk-env"

    def test_missing_raises(self):
        s = EnvStore(prefix="ADK_NEVER_SET_")
        with pytest.raises(SecretNotFound):
            s.get("nope")

    def test_no_prefix_direct(self, monkeypatch):
        monkeypatch.setenv("ADK_RAW", "raw-value")
        s = EnvStore()
        assert s.get("ADK_RAW") == "raw-value"


class TestFileStore:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text(json.dumps({"db_password": "hunter2"}))
        s = FileStore(p)
        assert s.has("db_password")
        assert s.get("db_password") == "hunter2"

    def test_missing_file_is_empty(self, tmp_path):
        s = FileStore(tmp_path / "absent.json")
        assert not s.has("anything")
        with pytest.raises(SecretNotFound):
            s.get("anything")

    def test_reload_picks_up_edits(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text(json.dumps({"k": "v1"}))
        s = FileStore(p)
        assert s.get("k") == "v1"
        p.write_text(json.dumps({"k": "v2"}))
        s.reload()
        assert s.get("k") == "v2"

    def test_malformed_falls_back_to_empty(self, tmp_path):
        p = tmp_path / "secrets.json"
        p.write_text("not-json{")
        s = FileStore(p)
        assert not s.has("anything")


class TestChainStore:
    def test_first_hit_wins(self, tmp_path, monkeypatch):
        # Use distinct env vars (Windows env vars are case-insensitive).
        env_file = tmp_path / "secrets.json"
        env_file.write_text(json.dumps({"FILE_ONLY": "from-file", "BOTH": "from-file"}))
        monkeypatch.setenv("ENV_ONLY", "from-env")
        monkeypatch.setenv("BOTH", "from-env")
        chain = ChainStore([EnvStore(), FileStore(env_file)])
        assert chain.get("ENV_ONLY") == "from-env"
        assert chain.get("BOTH") == "from-env"
        assert chain.get("FILE_ONLY") == "from-file"

    def test_missing_raises_only_at_end(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({}))
        chain = ChainStore([EnvStore(prefix="ADK_NEVER_"), FileStore(f)])
        with pytest.raises(SecretNotFound):
            chain.get("ghost")

    def test_requires_at_least_one(self):
        with pytest.raises(ValueError):
            ChainStore([])


class TestResolveAndCapability:
    def test_use_store_scopes_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALPHA", "env-value")
        p = tmp_path / "s.json"
        p.write_text(json.dumps({"alpha": "file-value"}))
        with use_store(FileStore(p)):
            assert resolve("alpha") == "file-value"
        assert resolve("ALPHA") == "env-value"

    def test_handle_resolves(self, monkeypatch):
        monkeypatch.setenv("TOKEN", "xyz")
        assert resolve(handle("TOKEN")) == "xyz"

    def test_capability_denied(self, monkeypatch):
        monkeypatch.setenv("DENIED_SECRET", "value")
        with pytest.raises(SecretCapabilityDenied):
            resolve("DENIED_SECRET", capabilities=set())

    def test_capability_allowed_with_enum(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_SECRET", "value")
        out = resolve("ALLOWED_SECRET", capabilities={Capability.SECRETS})
        assert out == "value"

    def test_capability_allowed_with_string(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_STR", "value")
        out = resolve("ALLOWED_STR", capabilities={"secrets"})
        assert out == "value"

    def test_capability_none_skips_check(self, monkeypatch):
        monkeypatch.setenv("SCRIPT_KEY", "value")
        # capabilities=None (default) = unrestricted (script-level use)
        assert resolve("SCRIPT_KEY") == "value"
