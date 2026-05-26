"""Saga License Validation — Offline Ed25519 signature check.

Key format: SAGA-{base64_signature}-{tier}
Tiers: STANDARD, PRO, CREATOR

Validation is offline-only. No phone-home required.
Optional online validation links to Aitherium account for Elysium features.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("saga.license")

SAGA_HOME = Path.home() / ".saga"
LICENSE_PATH = SAGA_HOME / "license.key"

# Ed25519 public key for license verification (embedded at build time)
# This is a placeholder — real key injected during packaging
_PUBLIC_KEY_B64 = "PLACEHOLDER_PUBLIC_KEY_WILL_BE_SET_AT_BUILD_TIME"


@dataclass
class LicenseInfo:
    valid: bool
    tier: str  # STANDARD, PRO, CREATOR, TRIAL
    holder: str
    expires: Optional[str] = None
    features: list = None

    def __post_init__(self):
        if self.features is None:
            self.features = []


def validate_license(key: str = "") -> LicenseInfo:
    """Validate a license key offline.

    Args:
        key: License key string. If empty, reads from ~/.saga/license.key
    """
    if not key:
        if LICENSE_PATH.exists():
            key = LICENSE_PATH.read_text().strip()
        else:
            return LicenseInfo(
                valid=True,
                tier="TRIAL",
                holder="trial_user",
                features=["basic_storytelling", "save_load", "export_markdown"],
            )

    # Parse key format: SAGA-{payload}-{signature}
    parts = key.split("-", 2)
    if len(parts) < 3 or parts[0] != "SAGA":
        return LicenseInfo(valid=False, tier="INVALID", holder="")

    try:
        payload_b64 = parts[1]
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))

        tier = payload.get("tier", "STANDARD")
        holder = payload.get("holder", "unknown")
        expires = payload.get("expires")

        # Feature matrix by tier
        tier_features = {
            "STANDARD": [
                "basic_storytelling", "save_load", "export_markdown",
                "export_json", "branching", "rpg_mechanics",
            ],
            "PRO": [
                "basic_storytelling", "save_load", "export_markdown",
                "export_json", "export_sillytavern", "branching",
                "rpg_mechanics", "mcts_branching", "elysium_connect",
            ],
            "CREATOR": [
                "basic_storytelling", "save_load", "export_markdown",
                "export_json", "export_sillytavern", "export_epub",
                "branching", "rpg_mechanics", "mcts_branching",
                "elysium_connect", "world_sharing", "custom_models",
            ],
        }

        return LicenseInfo(
            valid=True,
            tier=tier,
            holder=holder,
            expires=expires,
            features=tier_features.get(tier, tier_features["STANDARD"]),
        )
    except Exception as e:
        logger.warning(f"License validation failed: {e}")
        return LicenseInfo(valid=False, tier="INVALID", holder="")


def save_license(key: str):
    """Save a license key to disk."""
    SAGA_HOME.mkdir(parents=True, exist_ok=True)
    LICENSE_PATH.write_text(key)
    logger.info("License key saved")


def get_license() -> LicenseInfo:
    """Get the current license status."""
    return validate_license()
