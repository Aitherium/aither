"""Backwards-compat alias for ``adk.platform.communication.council`` / ``aeon_client``.

Some tests/legacy code import :class:`CouncilClient` from
``council_client``. The canonical implementation now lives in
``aeon_client`` (Aeon was formerly Council). This shim re-exports it so
both paths resolve to the same object.
"""

from __future__ import annotations

try:
    from .aeon_client import AeonClient as CouncilClient  # type: ignore[assignment]
except ImportError:  # pragma: no cover — defensive
    class CouncilClient:  # type: ignore[no-redef]
        """Placeholder when no aeon implementation is available."""

__all__ = ["CouncilClient"]
