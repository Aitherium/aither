"""AitherSDK — Strata virtual filesystem client (port 8136)."""

from typing import Optional

from aithersdk.base import ServiceClient


class StrataClient(ServiceClient):
    """Client for the Strata virtual filesystem service."""

    async def write(
        self,
        path: str,
        content: str,
        tier: str = "warm",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Write content to a Strata path."""
        payload = {"path": path, "content": content, "tier": tier}
        if metadata:
            payload["metadata"] = metadata
        return await self._post("/strata/write", json=payload, timeout=30.0)

    async def read(self, path: str) -> dict:
        """Read content from a Strata path."""
        return await self._get(f"/strata/read?path={path}")

    async def list_artifacts(self, tier: str = "warm", prefix: str = "artifacts") -> dict:
        """List artifacts in a Strata tier."""
        return await self._get(f"/strata/list/{tier}/{prefix}")
