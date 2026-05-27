"""Tests for secrets sync client."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk.secrets_sync import SecretsSync, _load_local_secrets, _save_local_secrets


class TestLocalStore:
    def test_save_and_load(self, tmp_path):
        secrets_file = tmp_path / "secrets.json"
        with patch("adk.secrets_sync._SECRETS_FILE", secrets_file), \
             patch("adk.secrets_sync._AITHER_DIR", tmp_path):
            _save_local_secrets({"KEY1": "val1", "KEY2": "val2"})
            loaded = _load_local_secrets()
            assert loaded == {"KEY1": "val1", "KEY2": "val2"}

    def test_load_missing_file(self, tmp_path):
        secrets_file = tmp_path / "nonexistent.json"
        with patch("adk.secrets_sync._SECRETS_FILE", secrets_file):
            assert _load_local_secrets() == {}


class TestSecretsSync:
    def test_init_from_env(self):
        env = {
            "AITHER_API_KEY": "test-key",
            "AITHER_SECRETS_URL": "http://secrets:8111",
            "AITHER_TENANT": "mytenant",
        }
        with patch.dict(os.environ, env, clear=False):
            client = SecretsSync()
            assert client.api_key == "test-key"
            assert client.secrets_url == "http://secrets:8111"
            assert client.tenant == "mytenant"

    def test_headers(self):
        client = SecretsSync(api_key="key123", tenant="t1")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer key123"
        assert headers["X-Tenant-ID"] == "t1"

    @pytest.mark.asyncio
    async def test_pull_from_secrets_service(self, tmp_path):
        secrets_file = tmp_path / "secrets.json"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"API_KEY": "secret123"}

        with patch("adk.secrets_sync._SECRETS_FILE", secrets_file), \
             patch("adk.secrets_sync._AITHER_DIR", tmp_path):
            import httpx
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                client = SecretsSync(
                    api_key="key",
                    secrets_url="http://secrets:8111",
                )
                synced = await client.pull()

                assert "API_KEY" in synced
                assert synced["API_KEY"] == "secret123"

    @pytest.mark.asyncio
    async def test_pull_no_sources(self):
        client = SecretsSync(api_key="", secrets_url="", gateway_url="")
        synced = await client.pull()
        assert synced == {}
