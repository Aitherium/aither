"""adk provisioning tools — provision_gateway_key for operator agents.

A first-class adk tool that wraps the platform's ``provision_gateway_key`` MCP tool via the adk
MCP bridge, so an operator agent can mint a scoped gateway key for a deployment WITHOUT the
internal secret ever touching the agent — the mint happens server-side on the platform; the agent
just calls the tool over MCP. Privileged (most agents won't have the scope).

    from adk.provisioning_tools import register_provisioning_tools
    register_provisioning_tools(agent)            # uses AITHER_GATEWAY_URL/KEY from env
"""

import json
import logging
import os

logger = logging.getLogger("adk.provisioning")


def register_provisioning_tools(agent, mcp_url: str = "", api_key: str = "") -> int:
    """Register `provision_gateway_key` on the agent. Returns the number of tools registered.

    mcp_url/api_key default to AITHER_GATEWAY_URL / AITHER_GATEWAY_KEY (the deployment's gateway
    config), so an operator agent on a provisioned deployment can self-provision more keys.
    """
    mcp_url = mcp_url or os.getenv("AITHER_GATEWAY_URL") or "https://mcp.aitherium.com"
    api_key = api_key or os.getenv("AITHER_GATEWAY_KEY", "")

    async def provision_gateway_key(
        tenant: str,
        scopes: str = "chat,agent:ask,agent:read",
        expires_in_days: int = 30,
    ) -> str:
        """Mint a scoped, expiring aither_sk_live_* gateway key for a deployment (operator only).

        Routes through the AitherOS MCP gateway — the internal secret stays on the platform.
        The key is returned ONCE; save it to the deployment's gitignored .env, never commit it.

        tenant: deployment identity the key binds to (e.g. 'daoos-usb-demo').
        scopes: comma-separated (default 'chat,agent:ask,agent:read').
        expires_in_days: 1-365 (default 30).
        """
        try:
            from adk.mcp import MCPBridge
            bridge = MCPBridge(mcp_url=mcp_url, api_key=api_key)
            return await bridge.call_tool(
                "provision_gateway_key",
                {"tenant": tenant, "scopes": scopes, "expires_in_days": int(expires_in_days)},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("provision_gateway_key failed: %s", e)
            return json.dumps({"error": str(e)})

    agent._tools.register(
        provision_gateway_key,
        name="provision_gateway_key",
        description="Mint a scoped aither_sk_live_* gateway key for a deployment (operator only; via MCP).",
    )
    logger.info("Registered provisioning tool on agent %s", getattr(agent, "name", "?"))
    return 1
