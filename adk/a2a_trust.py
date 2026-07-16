"""A2A inbound signature verification and trust enforcement.

Validates incoming A2A requests with X-Signature headers, ensuring they are:
  1. Cryptographically valid (Ed25519 signature matching X-Public-Key)
  2. From a trusted peer (registered in mesh or allowlist)

This module is OPT-IN via AITHER_A2A_REQUIRE_TRUST env var (default: 'false').
When enabled ('true'), unknown keys are rejected with 403 Forbidden.
When 'audit', signatures are verified but failures only log (non-blocking).

Usage:
  from adk.a2a_trust import verify_a2a_request

  # In your A2A endpoint handler:
  verified = await verify_a2a_request(
      request_body=body_bytes,
      x_signature=request.headers.get("X-Signature"),
      x_public_key=request.headers.get("X-Public-Key"),
      x_node_id=request.headers.get("X-Node-ID"),
  )
  if not verified.trusted:
      return JSONResponse({"error": "Untrusted request"}, status_code=403)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("adk.a2a_trust")

_REQUIRE_TRUST_MODE = os.getenv("AITHER_A2A_REQUIRE_TRUST", "false").lower()


@dataclass
class A2ATrustResult:
    """Result of A2A trust verification."""
    verified: bool  # Cryptographic signature valid?
    trusted: bool   # Key is in trusted list or registered?
    node_id: Optional[str]  # X-Node-ID from request
    public_key: Optional[str]  # X-Public-Key from request
    reason: str  # Why verified/trusted or failed


async def verify_a2a_request(
    request_body: bytes,
    x_signature: Optional[str],
    x_public_key: Optional[str],
    x_node_id: Optional[str] = None,
) -> A2ATrustResult:
    """Verify A2A request signature and trust status.

    Args:
        request_body: Raw request body bytes (to be hashed for signature check)
        x_signature: X-Signature header (hex-encoded Ed25519 signature)
        x_public_key: X-Public-Key header (hex-encoded Ed25519 public key)
        x_node_id: X-Node-ID header (optional, for mesh peer lookup)

    Returns:
        A2ATrustResult with verified, trusted, and reason fields.
    """
    # If no signature headers, fail-closed (when enforcement is on)
    if not x_signature or not x_public_key:
        return A2ATrustResult(
            verified=False,
            trusted=False,
            node_id=x_node_id,
            public_key=None,
            reason="Missing X-Signature or X-Public-Key header",
        )

    # Verify signature
    try:
        verified = await _verify_ed25519_signature(request_body, x_public_key, x_signature)
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return A2ATrustResult(
            verified=False,
            trusted=False,
            node_id=x_node_id,
            public_key=x_public_key,
            reason=f"Signature verification error: {e}",
        )

    if not verified:
        return A2ATrustResult(
            verified=False,
            trusted=False,
            node_id=x_node_id,
            public_key=x_public_key,
            reason="Signature does not match public key",
        )

    # Signature is valid; now check if key is trusted
    trusted, reason = await _is_key_trusted(x_public_key, x_node_id)

    return A2ATrustResult(
        verified=True,
        trusted=trusted,
        node_id=x_node_id,
        public_key=x_public_key,
        reason=reason,
    )


async def _verify_ed25519_signature(
    message: bytes,
    public_key_hex: str,
    signature_hex: str,
) -> bool:
    """Verify Ed25519 signature using cryptography library.

    Args:
        message: Message bytes
        public_key_hex: Hex-encoded Ed25519 public key (64 chars = 32 bytes)
        signature_hex: Hex-encoded signature (128 chars = 64 bytes)

    Returns:
        True if signature is valid, False otherwise.

    Raises:
        ValueError: If key/signature format is invalid
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.exceptions import InvalidSignature

        # Decode hex to bytes
        if len(public_key_hex) != 64:
            raise ValueError(
                f"Public key must be 32 bytes (64 hex chars), got {len(public_key_hex)}"
            )
        if len(signature_hex) != 128:
            raise ValueError(
                f"Signature must be 64 bytes (128 hex chars), got {len(signature_hex)}"
            )

        public_key_bytes = bytes.fromhex(public_key_hex)
        signature_bytes = bytes.fromhex(signature_hex)

        # Load public key and verify
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature_bytes, message)
        return True
    except InvalidSignature:
        return False
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
        raise


async def _is_key_trusted(
    public_key_hex: str,
    node_id: Optional[str] = None,
) -> tuple[bool, str]:
    """Check if a public key is in the trusted list.

    Currently checks:
      1. Static AITHER_A2A_TRUSTED_KEYS env var (comma-separated hex keys)
      2. Mesh peer registry (if node_id provided) — TODO: implement mesh lookup
      3. Portal fleet cache (if enabled) — TODO: implement portal lookup

    Args:
        public_key_hex: Hex-encoded Ed25519 public key
        node_id: Optional node ID for mesh lookup

    Returns:
        (trusted: bool, reason: str)
    """
    # Check static trusted keys env var
    trusted_keys_str = os.getenv("AITHER_A2A_TRUSTED_KEYS", "").strip()
    if trusted_keys_str:
        trusted_keys = [k.strip() for k in trusted_keys_str.split(",") if k.strip()]
        if public_key_hex in trusted_keys:
            return True, "Key is in AITHER_A2A_TRUSTED_KEYS"
    else:
        logger.debug("No AITHER_A2A_TRUSTED_KEYS configured")

    # TODO: Check mesh peer registry (AitherNet node a2a_public_key)
    # if node_id:
    #     mesh_key = await aithernet.get_node_a2a_public_key(node_id, tenant_id)
    #     if mesh_key == public_key_hex:
    #         return True, f"Key registered for node {node_id} in mesh"

    # TODO: Check portal fleet cache (if AITHER_A2A_TRUST_FROM_PORTAL=true)
    # if os.getenv("AITHER_A2A_TRUST_FROM_PORTAL", "").lower() in ("true", "1"):
    #     ...

    return False, "Key is not in any trusted source (static, mesh, or portal)"


def should_require_a2a_trust() -> bool:
    """Check if A2A trust enforcement is enabled.

    Modes:
      'false' (default): No verification
      'audit': Verify but only log failures
      'true': Enforce — return 403 for untrusted keys
    """
    return _REQUIRE_TRUST_MODE == "true"


def should_audit_a2a_trust() -> bool:
    """Check if A2A trust should be audited (logged but not enforced)."""
    return _REQUIRE_TRUST_MODE == "audit"
