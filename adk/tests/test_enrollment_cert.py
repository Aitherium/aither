"""Tests for device certificate enrollment in rich_enroll."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk import enrollment


@pytest.fixture
def temp_identity_dir():
    """Create a temporary directory for device identity testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_request_device_cert_success(temp_identity_dir):
    """Test successful device cert request."""
    # Mock the httpx.AsyncClient context manager
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "mtls": {
            "certificate": "-----BEGIN CERTIFICATE-----\nCERT\n-----END CERTIFICATE-----",
            "private_key": "FAKE-TEST-PRIVATE-KEY-MATERIAL-NOT-REAL",
            "chain": "-----BEGIN CERTIFICATE-----\nCHAIN\n-----END CERTIFICATE-----",
        }
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        with patch("adk.sync.device_identity.save_enrolled_identity") as mock_save:
            result = await enrollment._request_device_cert(
                "https://identity.example.com",
                "bearer_token_123",
                "test-node",
                "tnt_abc123",
            )

    assert result["success"] is True
    assert "mtls" in result
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_request_device_cert_http_error():
    """Test device cert request with HTTP error."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = await enrollment._request_device_cert(
            "https://identity.example.com",
            "bad_token",
            "test-node",
            "tnt_abc123",
        )

    assert result["success"] is False
    assert "401" in result["error"]


@pytest.mark.asyncio
async def test_request_device_cert_missing_cert_in_response():
    """Test device cert request when response has no cert."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"mtls": {}}  # No certificate

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = await enrollment._request_device_cert(
            "https://identity.example.com",
            "token",
            "test-node",
            "tnt_abc123",
        )

    assert result["success"] is False
    assert "no cert" in result["error"]


@pytest.mark.asyncio
async def test_request_device_cert_missing_token():
    """Test device cert request with missing token (fails closed)."""
    result = await enrollment._request_device_cert(
        "https://identity.example.com",
        "",  # No token
        "test-node",
        "tnt_abc123",
    )
    assert result["success"] is False
    assert "missing" in result["error"]


@pytest.mark.asyncio
async def test_request_device_cert_missing_tenant():
    """Test device cert request with missing tenant (fails closed)."""
    result = await enrollment._request_device_cert(
        "https://identity.example.com",
        "token",
        "test-node",
        "",  # No tenant
    )
    assert result["success"] is False
    assert "missing" in result["error"]


@pytest.mark.asyncio
async def test_rich_enroll_with_cert():
    """Test rich_enroll flow with successful cert enrollment."""
    mock_enroll_response = MagicMock()
    mock_enroll_response.status_code = 200
    mock_enroll_response.json.return_value = {
        "tenant_id": "tnt_abc123",
        "workspace_id": "ws_xyz789",
        "workspace": {"name": "My Workspace"},
        "bearer_token": "bearer_token_123",
    }

    mock_cert_response = MagicMock()
    mock_cert_response.status_code = 200
    mock_cert_response.json.return_value = {
        "mtls": {
            "certificate": "-----BEGIN CERTIFICATE-----\nCERT\n-----END CERTIFICATE-----",
            "private_key": "FAKE-TEST-PRIVATE-KEY-MATERIAL-NOT-REAL",
        }
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        with patch("adk.enrollment._save_workspace"):
            with patch("adk.enrollment._request_device_cert") as mock_cert:
                mock_cert.return_value = {
                    "success": True,
                    "mtls": {
                        "certificate": "CERT",
                        "private_key": "KEY",
                    }
                }

                # Set up mock client for the enroll call
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.post.return_value = mock_enroll_response
                mock_client_class.return_value = mock_client

                result = await enrollment.rich_enroll(
                    "https://identity.example.com",
                    "user_token",
                    "test-node",
                    enable_heartbeat=False,
                )

    assert result["enrolled"] is True
    assert result["cert_enrolled"] is True
    assert result["tenant_id"] == "tnt_abc123"
    mock_cert.assert_called_once()


@pytest.mark.asyncio
async def test_rich_enroll_without_cert_request():
    """Test rich_enroll when cert request fails (non-fatal)."""
    mock_enroll_response = MagicMock()
    mock_enroll_response.status_code = 200
    mock_enroll_response.json.return_value = {
        "tenant_id": "tnt_abc123",
        "workspace_id": "ws_xyz789",
        "workspace": {"name": "My Workspace"},
        "bearer_token": "bearer_token_123",
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        with patch("adk.enrollment._save_workspace"):
            with patch("adk.enrollment._request_device_cert") as mock_cert:
                mock_cert.return_value = {"success": False, "error": "timeout"}

                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client.post.return_value = mock_enroll_response
                mock_client_class.return_value = mock_client

                result = await enrollment.rich_enroll(
                    "https://identity.example.com",
                    "user_token",
                    "test-node",
                    enable_heartbeat=False,
                )

    # Enrollment should still succeed even if cert request fails
    assert result["enrolled"] is True
    assert result["cert_enrolled"] is False
