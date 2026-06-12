"""FormBridge HTTP routes — loopback-only by construction.

The adk server's bind host is configurable, so these routes do not trust the
bind: every endpoint rejects non-loopback clients (06-SECURITY-MODEL hard
rule — captured values never traverse a network).

Pack discovery: AITHER_FORMBRIDGE_PACK points at a mapping-pack directory;
otherwise ~/.aither/formbridge/packs/* are scanned (first match wins per
pack id). Installed via the normal signed-pack machinery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from adk.formbridge.mapper import MappingError, MappingPack, load_pack, resolve, run_self_check
from adk.formbridge.pdf_fill import HAS_PYPDF, fill_pdf
from adk.formbridge.store import get_store

logger = logging.getLogger("adk.formbridge.routes")

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK:
        raise HTTPException(status_code=403, detail="formbridge is loopback-only")


def _licensed() -> bool:
    if os.getenv("AITHER_FORMBRIDGE_DEV", "").lower() in ("1", "true"):
        return True
    try:
        from adk.licensing import get_license_manager

        return get_license_manager().is_pack_available("form-bridge")
    except Exception as e:  # never crash the server over a license probe
        logger.debug("formbridge license probe failed: %s", e)
        return False


def _pack_dirs() -> list[Path]:
    override = os.getenv("AITHER_FORMBRIDGE_PACK", "").strip()
    if override:
        return [Path(override)]
    base = Path.home() / ".aither" / "formbridge" / "packs"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "mapping.yaml").is_file())


def _load_packs() -> dict[str, MappingPack]:
    packs: dict[str, MappingPack] = {}
    for d in _pack_dirs():
        try:
            p = load_pack(d)
            packs.setdefault(p.pack_id, p)
        except MappingError as e:
            logger.warning("formbridge: skipping invalid pack %s: %s", d, e)
    return packs


class CaptureBatch(BaseModel):
    patient_key: str = Field(min_length=1)
    display_name: str = ""
    source_origin: str = ""
    fields: list[dict] = Field(default_factory=list)
    seen_selectors: list[str] = Field(default_factory=list)


class FillRequest(BaseModel):
    patient_key: str = Field(min_length=1)
    form_id: str = Field(min_length=1)
    pack_id: str = ""  # optional when exactly one pack is installed
    flatten: bool = False


class PurgeRequest(BaseModel):
    patient_key: str | None = None


def _resolve_pack(pack_id: str) -> MappingPack:
    packs = _load_packs()
    if not packs:
        raise HTTPException(status_code=404, detail="no formbridge mapping pack installed")
    if pack_id:
        if pack_id not in packs:
            raise HTTPException(
                status_code=404,
                detail=f"pack '{pack_id}' not installed (have: {sorted(packs)})",
            )
        return packs[pack_id]
    if len(packs) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"multiple packs installed, specify pack_id (have: {sorted(packs)})",
        )
    return next(iter(packs.values()))


def create_formbridge_router() -> APIRouter:
    router = APIRouter(prefix="/formbridge", tags=["formbridge"])

    @router.get("/health")
    async def health(request: Request):
        _require_loopback(request)
        packs = _load_packs()
        return {
            "ok": True,
            "pypdf": HAS_PYPDF,
            "licensed": _licensed(),
            "packs": {pid: p.version for pid, p in packs.items()},
            "store": get_store().stats(),
        }

    @router.get("/forms")
    async def forms(request: Request):
        _require_loopback(request)
        out = []
        for pid, pack in _load_packs().items():
            for fid, f in pack.forms.items():
                out.append({
                    "pack_id": pid,
                    "form_id": fid,
                    "required": f.required,
                    "field_count": len(f.fill),
                })
        return {"forms": out}

    @router.post("/capture")
    async def capture(batch: CaptureBatch, request: Request):
        _require_loopback(request)
        written = get_store().ingest_batch(
            batch.patient_key,
            batch.fields,
            display_name=batch.display_name,
            source_origin=batch.source_origin,
        )
        result: dict = {"ok": True, "written": written}
        if batch.seen_selectors:
            packs = _load_packs()
            if len(packs) == 1:
                result["self_check"] = run_self_check(
                    next(iter(packs.values())), batch.seen_selectors
                )
        return result

    @router.get("/patients")
    async def patients(request: Request):
        _require_loopback(request)
        return {"patients": get_store().list_patients()}

    @router.get("/patients/{patient_key}")
    async def patient_record(patient_key: str, request: Request):
        # Loopback-only like everything else; this is the ONE endpoint that
        # returns values (the local review surface needs them). It exists for
        # local UI use and must never be proxied or forwarded.
        _require_loopback(request)
        record = get_store().get_record(patient_key)
        if not record:
            raise HTTPException(status_code=404, detail="no captured record for that key")
        return {"patient_key": patient_key, "fields": record}

    @router.post("/fill")
    async def fill(req: FillRequest, request: Request):
        _require_loopback(request)
        if not _licensed():
            raise HTTPException(
                status_code=402,
                detail="form-bridge license required (set AITHER_FORMBRIDGE_DEV=1 for dev/demo)",
            )
        pack = _resolve_pack(req.pack_id)
        record = get_store().get_record(req.patient_key)
        if not record:
            raise HTTPException(status_code=404, detail="no captured record for that key")
        try:
            resolution = resolve(record, pack, req.form_id)
            result = fill_pdf(pack, resolution, flatten=req.flatten)
        except MappingError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return {
            "ok": True,
            "job_id": result.job_id,
            "output_path": result.output_path,
            "filled": result.filled,
            "unresolved": result.unresolved,
            "llm_assist_fields": result.llm_assist_fields,
            "unknown_pdf_fields": result.unknown_pdf_fields,
        }

    @router.post("/purge")
    async def purge(req: PurgeRequest, request: Request):
        _require_loopback(request)
        deleted = get_store().purge(req.patient_key)
        return {"ok": True, "deleted": deleted}

    return router
