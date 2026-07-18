"""Test Flow tools port resolution and URL configuration."""

import os
import sys
from unittest.mock import patch

import pytest

# The flow tools live in the INTERNAL aither-platform toolkit (a separate
# package; this test file is stripped from the public sync payload). In an
# environment without that package installed, skip instead of erroring.
pytest.importorskip("aither_platform")


def test_flow_url_default():
    """Test that default FLOW_URL is http://localhost:8164/flow."""
    # Import after patching env to avoid caching
    with patch.dict(os.environ, {}, clear=False):
        # Remove any existing Flow URL env vars
        for key in list(os.environ.keys()):
            if "FLOW" in key.upper():
                del os.environ[key]

        # Force reimport to pick up clean env
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == "http://localhost:8164/flow", (
            f"Expected http://localhost:8164/flow, got {FLOW_URL}"
        )


def test_flow_url_env_override_no_path_appended():
    """Test AITHER_AITHERFLOW_URL is used verbatim with no /flow appended."""
    custom_url = "http://custom-host:9999/my-flow-base"
    with patch.dict(os.environ, {"AITHER_AITHERFLOW_URL": custom_url}):
        # Force reimport
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == custom_url, (
            f"Expected {custom_url}, got {FLOW_URL}. "
            "AITHER_AITHERFLOW_URL should be used verbatim."
        )


def test_flow_url_env_override_trailing_slash_stripped():
    """Test trailing slash is stripped from AITHER_AITHERFLOW_URL."""
    custom_url = "http://custom-host:9999/"
    with patch.dict(os.environ, {"AITHER_AITHERFLOW_URL": custom_url}):
        # Force reimport
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == "http://custom-host:9999", (
            f"Trailing slash not stripped: {FLOW_URL}"
        )


def test_flow_url_port_override():
    """Test AITHER_AITHERFLOW_PORT override still yields /flow suffix."""
    port_override = "9876"
    with patch.dict(os.environ, {"AITHER_AITHERFLOW_PORT": port_override}):
        # Clear other Flow env vars
        for key in list(os.environ.keys()):
            if "AITHER_AITHERFLOW_URL" in key or "AITHER_FLOW_URL" in key:
                del os.environ[key]

        # Force reimport
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == "http://localhost:9876/flow", (
            f"Expected http://localhost:9876/flow, got {FLOW_URL}. "
            "Port override should still append /flow."
        )


def test_flow_url_port_and_host_override():
    """Test AITHER_AITHERFLOW_PORT with custom AITHER_SERVICE_HOST."""
    with patch.dict(
        os.environ,
        {
            "AITHER_AITHERFLOW_PORT": "7777",
            "AITHER_SERVICE_HOST": "custom-host",
        },
    ):
        # Clear other Flow env vars
        for key in list(os.environ.keys()):
            if "AITHER_AITHERFLOW_URL" in key or "AITHER_FLOW_URL" in key:
                del os.environ[key]

        # Force reimport
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == "http://custom-host:7777/flow", (
            f"Expected http://custom-host:7777/flow, got {FLOW_URL}"
        )


def test_flow_url_fallback_aither_flow_url():
    """Test AITHER_FLOW_URL (alternate name) is also supported."""
    custom_url = "https://api.example.com/flow-service"
    with patch.dict(os.environ, {"AITHER_FLOW_URL": custom_url}):
        # Clear other Flow env vars
        for key in list(os.environ.keys()):
            if "AITHER_AITHERFLOW_URL" in key:
                del os.environ[key]

        # Force reimport
        if "aither_platform.tools.flow_tools" in sys.modules:
            del sys.modules["aither_platform.tools.flow_tools"]

        from aither_platform.tools.flow_tools import FLOW_URL

        assert FLOW_URL == custom_url, (
            f"Expected {custom_url}, got {FLOW_URL}. "
            "AITHER_FLOW_URL should be recognized."
        )
