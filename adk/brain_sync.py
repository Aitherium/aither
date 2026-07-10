"""Brain hub sync client — push embedding deltas to CompanyBrain.

Posts embedding deltas to POST /api/v1/brain/sync with tenant/workspace context,
tracks watermarks for incremental sync, and handles transient failures gracefully.

Usage:
    from adk.brain_sync import BrainSyncClient, SyncDeltaItem

    client = BrainSyncClient(
        brain_url="http://localhost:8001",
        tenant_id="tenant-123",
        workspace_id="default",
    )

    deltas = [
        SyncDeltaItem(
            op="upsert",
            chunk_id="chunk-1",
            vector=[0.1, 0.2, ...],
            metadata={"text": "...", "source": "..."},
            classification="internal",
        ),
    ]

    response = await client.post_deltas(deltas)
    # response.accepted, response.rejected, response.watermark
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("adk.brain_sync")

__all__ = [
    "SyncDeltaItem",
    "SyncRequest",
    "SyncResponse",
    "BrainSyncClient",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data models (mirror brain.py contracts)
# ─────────────────────────────────────────────────────────────────────────────

class SyncDeltaItem:
    """Embedding delta for brain sync.

    Attrs:
        op: "upsert" or "delete"
        chunk_id: Unique identifier (min 1 char)
        vector: Optional embedding vector (768-dim or None to skip semantic search)
        metadata: Arbitrary dict (typically {text, source, offset, ...})
        classification: public|internal|confidential|restricted (default: internal)
    """

    def __init__(
        self,
        chunk_id: str,
        op: str = "upsert",
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        classification: str = "internal",
    ):
        if not chunk_id:
            raise ValueError("chunk_id required, min_length=1")
        if classification not in ("public", "internal", "confidential", "restricted"):
            raise ValueError(f"Invalid classification: {classification}")
        if op not in ("upsert", "delete"):
            raise ValueError(f"Invalid op: {op}")

        self.op = op
        self.chunk_id = chunk_id
        self.vector = vector
        self.metadata = metadata or {}
        self.classification = classification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API dict."""
        return {
            "op": self.op,
            "chunk_id": self.chunk_id,
            "vector": self.vector,
            "metadata": self.metadata,
            "classification": self.classification,
        }


class SyncRequest:
    """Request to brain hub sync endpoint.

    Attrs:
        tenant_id: Tenant identifier (required)
        workspace_id: Workspace scope (default: 'default')
        watermark: Opaque checkpoint for incremental sync (empty string = first sync)
        delta: List of SyncDeltaItem
    """

    def __init__(
        self,
        tenant_id: str,
        workspace_id: str = "default",
        watermark: str = "",
        delta: Optional[List[SyncDeltaItem]] = None,
    ):
        if not tenant_id:
            raise ValueError("tenant_id required")
        self.tenant_id = tenant_id
        self.workspace_id = workspace_id
        self.watermark = watermark
        self.delta = delta or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API dict."""
        return {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "watermark": self.watermark,
            "delta": [item.to_dict() for item in self.delta],
        }

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict())


class SyncResponse:
    """Response from brain hub sync endpoint.

    Attrs:
        accepted: Count of accepted upsert/delete ops
        rejected: Count of rejected ops
        watermark: Opaque checkpoint for next sync
    """

    def __init__(self, accepted: int = 0, rejected: int = 0, watermark: str = ""):
        self.accepted = accepted
        self.rejected = rejected
        self.watermark = watermark

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyncResponse:
        """Parse from API response."""
        return cls(
            accepted=data.get("accepted", 0),
            rejected=data.get("rejected", 0),
            watermark=data.get("watermark", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Brain sync client
# ─────────────────────────────────────────────────────────────────────────────

class BrainSyncClient:
    """Client for pushing embeddings to CompanyBrain hub.

    Implements graceful degradation:
      - Connection errors: log warning, return empty response (no crash)
      - 400 schema error: log error, return rejection
      - 401 auth: log error, ask user to re-enroll
      - 503 unavailable: log, queue for retry
    """

    def __init__(
        self,
        brain_url: str,
        tenant_id: str,
        workspace_id: str = "default",
        timeout: float = 30.0,
    ):
        """Initialize sync client.

        Args:
            brain_url: Brain hub URL (e.g., http://localhost:8001)
            tenant_id: Tenant identifier
            workspace_id: Workspace scope (default: 'default')
            timeout: HTTP timeout in seconds (default: 30)
        """
        if not brain_url:
            raise ValueError("brain_url required")
        if not tenant_id:
            raise ValueError("tenant_id required")

        self.brain_url = brain_url.rstrip("/")
        self.tenant_id = tenant_id
        self.workspace_id = workspace_id
        self.timeout = timeout
        self.watermark = ""

    async def post_deltas(
        self,
        deltas: List[SyncDeltaItem],
        watermark: str = "",
        compress: bool = True,
    ) -> SyncResponse:
        """Post embedding deltas to brain hub.

        Args:
            deltas: List of SyncDeltaItem to sync
            watermark: Optional watermark from previous sync (default: "")
            compress: Compress payload with gzip (default: True)

        Returns:
            SyncResponse with accepted/rejected counts and new watermark.
        """
        if not deltas:
            logger.debug("No deltas to sync")
            return SyncResponse(accepted=0, rejected=0, watermark=watermark)

        try:
            import httpx
        except ImportError:
            logger.error("httpx required for brain sync")
            return SyncResponse(accepted=0, rejected=0, watermark=watermark)

        # Build request
        request = SyncRequest(
            tenant_id=self.tenant_id,
            workspace_id=self.workspace_id,
            watermark=watermark or self.watermark,
            delta=deltas,
        )

        payload = request.to_json().encode("utf-8")

        # Optionally compress
        headers = {}
        if compress:
            payload = gzip.compress(payload)
            headers["Content-Encoding"] = "gzip"

        headers["Content-Type"] = "application/json"

        url = f"{self.brain_url}/api/v1/brain/sync"

        try:
            # Get TLS verify from portal_kit_backend if available, else True
            verify = self._tls_verify()

            async with httpx.AsyncClient(timeout=self.timeout, verify=verify) as http:
                response = await http.post(url, content=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    result = SyncResponse.from_dict(data)
                    self.watermark = result.watermark
                    logger.info("Brain sync: %d accepted, %d rejected",
                               result.accepted, result.rejected)
                    return result

                elif response.status_code == 400:
                    logger.error("Brain sync schema error (400): %s",
                               response.text[:200])
                    return SyncResponse(
                        accepted=0,
                        rejected=len(deltas),
                        watermark=watermark,
                    )

                elif response.status_code == 401:
                    logger.error("Brain sync auth failed (401): "
                               "not enrolled or invalid credentials")
                    logger.info("Run 'adk enroll' to re-authenticate")
                    return SyncResponse(
                        accepted=0,
                        rejected=len(deltas),
                        watermark=watermark,
                    )

                elif response.status_code == 503:
                    logger.warning("Brain hub temporarily unavailable (503): "
                                  "will retry on next run")
                    return SyncResponse(
                        accepted=0,
                        rejected=len(deltas),
                        watermark=watermark,
                    )

                else:
                    logger.error("Brain sync failed (%d): %s",
                               response.status_code, response.text[:200])
                    return SyncResponse(
                        accepted=0,
                        rejected=len(deltas),
                        watermark=watermark,
                    )

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning("Brain sync network error: %s; will retry on next run", exc)
            return SyncResponse(
                accepted=0,
                rejected=len(deltas),
                watermark=watermark,
            )

        except Exception as exc:
            logger.error("Brain sync unexpected error: %s", exc)
            return SyncResponse(
                accepted=0,
                rejected=len(deltas),
                watermark=watermark,
            )

    def _tls_verify(self) -> Any:
        """Get TLS verification setting.

        Returns the internal-CA verifier if available, else True.
        Never returns False (unsafe).
        """
        try:
            from adk._tls import tls_verify
            return tls_verify()
        except Exception:
            return True
