"""ADK test fixtures — ensure tests run in env isolation."""

import os

import pytest

from adk.config import load_saved_config as _real_load_saved_config

# Env vars that ADK classes auto-read from the environment.
# Tests must not inherit these from the developer's shell.
_ISOLATION_VARS = [
    "AITHER_API_KEY",
    "AITHER_MCP_KEY",
    "MCP_SERVICE_TOKEN",
    "AITHER_GATEWAY_URL",
    "AITHER_MCP_URL",
    "AITHERNET_RELAY_URL",
    "AITHER_INFERENCE_URL",
]


def _isolated_load_saved_config(config_path=None):
    """Return empty dict for default config path (no env bleed), passthrough for explicit paths."""
    if config_path is None:
        return {}
    return _real_load_saved_config(config_path)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip AitherOS credentials from the environment for every test.

    Also patches load_saved_config so the default path (~/.aither/config.yaml)
    returns empty dict, preventing credential bleed from the dev machine.
    Tests that pass an explicit path still get the real function.

    The functionality suite runs as the unrestricted INTERNAL tier so that
    feature tests (fleet, channels, cron, swarm, auto-neurons) are not blocked
    by the open-core license gates. The dedicated licensing/moat tests override
    this via their own (module-level, later-running) fixtures to exercise the
    free COMMUNITY tier and fail-closed behavior.
    """
    for var in _ISOLATION_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("adk.config.load_saved_config", _isolated_load_saved_config)

    monkeypatch.setenv("AITHER_TENANT_SLUG", "aitherium")
    try:
        from adk.licensing import reset_license_manager
        reset_license_manager()
    except Exception:
        pass
