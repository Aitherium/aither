"""FormBridge MCP/agent tools — redacting by construction.

HARD RULE (06-SECURITY-MODEL): tool results NEVER contain captured field
VALUES. An agent (possibly a cloud-hosted one in non-PHI verticals) sees
pseudonymous patient keys, field NAMES, counts, and opaque job ids — nothing
else. The local /formbridge/patients/{key} HTTP endpoint is the only surface
that returns values, and it is loopback-gated.

Plain functions, registered via the `formbridge` category in
adk.builtin_tools.TOOL_CATEGORIES (same pattern as voice/graph tools).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("adk.formbridge.tools")


def formbridge_list_patients() -> dict:
    """List captured patient records (keys, field counts, last capture) — no values."""
    from adk.formbridge.store import get_store

    patients = get_store().list_patients()
    return {
        "patients": [
            {
                "patient_key": p["patient_key"],
                "field_count": p["field_count"],
                "last_capture": p["last_capture"],
            }
            for p in patients
        ],
        "count": len(patients),
    }


def formbridge_fill_form(patient_key: str, form_id: str, pack_id: str = "") -> dict:
    """Fill a PDF form from a captured record. Returns the output path and field NAMES only."""
    from adk.formbridge.mapper import MappingError, resolve
    from adk.formbridge.pdf_fill import fill_pdf
    from adk.formbridge.routes import _licensed, _load_packs
    from adk.formbridge.store import get_store

    if not _licensed():
        return {
            "error": "form-bridge license required",
            "hint": "activate a license or set AITHER_FORMBRIDGE_DEV=1 for dev/demo",
        }
    packs = _load_packs()
    if not packs:
        return {"error": "no formbridge mapping pack installed"}
    if pack_id and pack_id not in packs:
        return {"error": f"pack '{pack_id}' not installed", "available": sorted(packs)}
    if not pack_id:
        if len(packs) > 1:
            return {"error": "multiple packs installed, specify pack_id", "available": sorted(packs)}
        pack_id = next(iter(packs))
    pack = packs[pack_id]

    record = get_store().get_record(patient_key)
    if not record:
        return {"error": f"no captured record for patient_key '{patient_key}'"}
    try:
        resolution = resolve(record, pack, form_id)
        result = fill_pdf(pack, resolution)
    except MappingError as e:
        return {"error": str(e)}

    return {
        "ok": True,
        "job_id": result.job_id,
        "output_path": result.output_path,
        "filled_fields": result.filled,
        "unresolved_fields": result.unresolved,
        "review_required_fields": result.llm_assist_fields,
    }


def formbridge_pack_health() -> dict:
    """Report installed mapping packs, versions, pypdf availability, and store size."""
    from adk.formbridge.pdf_fill import HAS_PYPDF
    from adk.formbridge.routes import _licensed, _load_packs
    from adk.formbridge.store import get_store

    packs = _load_packs()
    stats = get_store().stats()
    return {
        "pypdf": HAS_PYPDF,
        "licensed": _licensed(),
        "packs": {pid: p.version for pid, p in packs.items()},
        "forms": {pid: sorted(p.forms) for pid, p in packs.items()},
        # counts only — no patient keys, no values
        "store": {"patients": stats["patients"], "fields": stats["fields"]},
    }


def formbridge_purge(patient_key: str = "") -> dict:
    """Purge captured data for one patient_key, or ALL captured data when empty."""
    from adk.formbridge.store import get_store

    deleted = get_store().purge(patient_key or None)
    return {"ok": True, "deleted": deleted, "scope": patient_key or "all"}
