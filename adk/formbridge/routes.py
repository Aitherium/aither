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
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from adk.formbridge.discovery import generate_source_fields, get_discovery
from adk.formbridge.mapper import MappingError, MappingPack, load_pack, resolve, run_self_check
from adk.formbridge.pdf_fill import HAS_PYPDF, fill_pdf, generate_pdf
from adk.formbridge.store import get_store
from adk.formbridge.vault import VaultError, get_vault

logger = logging.getLogger("adk.formbridge.routes")

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK:
        raise HTTPException(status_code=403, detail="formbridge is loopback-only")


def _require_unlocked() -> None:
    """When an office vault is set up, PHI routes require it to be unlocked.
    No vault configured → no-op (dev/demo + pre-vault deployments work plaintext)."""
    v = get_vault()
    if v.initialized and not v.unlocked:
        raise HTTPException(status_code=423, detail="vault locked — unlock FormBridge with the office passphrase")


def _pdf_password(resource_id: str) -> str | None:
    """Per-document PDF password from the vault keyring (auto-generated, stable,
    recoverable). None when no vault is set up → the PDF is left unencrypted."""
    v = get_vault()
    if v.initialized and v.unlocked:
        return v.generate_secret(f"pdf:{resource_id}")
    return None


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


class ApiCaptureBatch(BaseModel):
    """One API response the interceptor saw (cloud-EHR capture). `body` is the
    parsed JSON for JSON endpoints; `rtf` is the raw text for chart-note RTF."""
    url: str = Field(min_length=1)
    body: object = None
    rtf: str | None = None
    display_name: str = ""
    source_origin: str = ""


class FillRequest(BaseModel):
    patient_key: str = Field(min_length=1)
    form_id: str = Field(min_length=1)
    pack_id: str = ""  # optional when exactly one pack is installed
    flatten: bool = False


class PurgeRequest(BaseModel):
    patient_key: str | None = None


class PassphraseRequest(BaseModel):
    passphrase: str = Field(min_length=1)


class RotateRequest(BaseModel):
    old_passphrase: str = Field(min_length=1)
    new_passphrase: str = Field(min_length=1)


class RecoverRequest(BaseModel):
    recovery_code: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    patient_key: str = Field(min_length=1)
    title: str = ""


class DiscoverRequest(BaseModel):
    origin: str = Field(min_length=1)
    dom: list[dict] = Field(default_factory=list)
    api: list[dict] = Field(default_factory=list)


class GeneratePackRequest(BaseModel):
    origin: str = Field(min_length=1)
    mappings: dict[str, str] = Field(default_factory=dict)


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


_UI_HTML = Path(__file__).parent / "ui.html"


def create_formbridge_router() -> APIRouter:
    router = APIRouter(prefix="/formbridge", tags=["formbridge"])

    @router.get("/app", response_class=HTMLResponse)
    async def app_ui(request: Request):
        # The self-contained control panel (the FormBridge "extension" UI).
        # Served by the product so it ships with the engine/pack, not the
        # AitherConnect bundle. Loopback-only like every other route; its own
        # fetches to /formbridge/* are same-origin (127.0.0.1) so they pass too.
        _require_loopback(request)
        try:
            return HTMLResponse(_UI_HTML.read_text(encoding="utf-8"))
        except OSError as e:
            raise HTTPException(status_code=404, detail="formbridge UI not found") from e

    # ── Office vault (shared-key encryption at rest) ──

    @router.get("/vault/status")
    async def vault_status(request: Request):
        _require_loopback(request)
        return get_vault().status()

    @router.post("/vault/init")
    async def vault_init(req: PassphraseRequest, request: Request):
        _require_loopback(request)
        try:
            recovery_code = get_vault().initialize(req.passphrase)
        except VaultError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        # Returned ONCE. The office must back it up — it's the only recovery if
        # the passphrase is forgotten, and we never store it in the clear.
        return {"ok": True, "recovery_code": recovery_code, **get_vault().status()}

    @router.post("/vault/unlock")
    async def vault_unlock(req: PassphraseRequest, request: Request):
        _require_loopback(request)
        try:
            get_vault().unlock(req.passphrase)
        except VaultError as e:
            raise HTTPException(status_code=401, detail=str(e)) from e
        return {"ok": True, **get_vault().status()}

    @router.post("/vault/lock")
    async def vault_lock(request: Request):
        _require_loopback(request)
        get_vault().lock()
        return {"ok": True, **get_vault().status()}

    @router.post("/vault/rotate")
    async def vault_rotate(req: RotateRequest, request: Request):
        _require_loopback(request)
        try:
            get_vault().rotate(req.old_passphrase, req.new_passphrase)
        except VaultError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"ok": True, **get_vault().status()}

    @router.post("/vault/recover")
    async def vault_recover(req: RecoverRequest, request: Request):
        _require_loopback(request)
        try:
            get_vault().unlock_with_recovery(req.recovery_code)
        except VaultError as e:
            raise HTTPException(status_code=401, detail=str(e)) from e
        return {"ok": True, **get_vault().status()}

    @router.post("/vault/regenerate-recovery")
    async def vault_regen_recovery(request: Request):
        _require_loopback(request)
        try:
            code = get_vault().regenerate_recovery()
        except VaultError as e:
            raise HTTPException(status_code=423, detail=str(e)) from e
        return {"ok": True, "recovery_code": code}

    @router.get("/vault/secrets")
    async def vault_secrets(request: Request):
        _require_loopback(request)
        v = get_vault()
        if v.initialized and not v.unlocked:
            raise HTTPException(status_code=423, detail="vault locked")
        return {"names": v.list_secret_names()}

    @router.get("/vault/secret/{name}")
    async def vault_secret(name: str, request: Request):
        # Query a stored password (so a document is never permanently locked).
        _require_loopback(request)
        v = get_vault()
        if not v.initialized:
            raise HTTPException(status_code=404, detail="no vault")
        if not v.unlocked:
            raise HTTPException(status_code=423, detail="vault locked")
        try:
            value = v.get_secret(name)
        except VaultError as e:
            raise HTTPException(status_code=423, detail=str(e)) from e
        if value is None:
            raise HTTPException(status_code=404, detail="no such secret")
        return {"name": name, "value": value}

    @router.get("/health")
    async def health(request: Request):
        _require_loopback(request)
        packs = _load_packs()
        return {
            "ok": True,
            "pypdf": HAS_PYPDF,
            "licensed": _licensed(),
            "packs": {pid: p.version for pid, p in packs.items()},
            "ehr_url": next((p.ehr_url for p in packs.values() if p.ehr_url), ""),
            "vault": get_vault().status(),
            "store": get_store().stats(),
        }

    @router.get("/capture-config")
    async def capture_config(request: Request):
        # The browser-side capture config for installed pack(s): origin, anchor,
        # denylist, tabbed-editor map. The AitherConnect "Enable capture" button
        # fetches this to set itself up in one click (no console seed paste).
        _require_loopback(request)
        return {
            "packs": [
                {
                    "pack_id": pid,
                    "origin": p.origin,
                    "anchor": p.anchor,
                    "ignore": p.ignore,
                    "tabbed_editors": p.tabbed_editors,
                }
                for pid, p in _load_packs().items()
            ]
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

    def _translate_fields(batch: CaptureBatch) -> list[dict]:
        """Apply the pack's source.fields capture→as map to incoming paths.

        The browser capture script sends raw selector paths ("#subjective",
        "input[name=dob]"); mapping packs declare the selector → logical-name
        correspondence in source.fields. Unmapped paths pass through verbatim
        so logical-name producers (tabbed-editor entries, tests, curl) work
        unchanged.
        """
        packs = _load_packs()
        selector_map: dict[str, str] = {}
        anchor_logicals: set[str] = set()
        candidates = [
            p for p in packs.values()
            if not batch.source_origin or p.origin == batch.source_origin.rstrip("/")
        ] or list(packs.values())
        for p in candidates:
            for f in p.capture_fields:
                cap, logical = f.get("capture"), f.get("as")
                if cap and logical:
                    selector_map.setdefault(cap, logical)
            # If a pack maps its ANCHOR element to a logical field (e.g. the
            # patient-name heading → "iw.name"), that value never arrives as an
            # edited <input> — it's a static text element the capture script reads
            # only for patient identity (display_name). Materialize it from the
            # batch's display_name so required name fields actually fill.
            anchor_sel = (p.anchor or {}).get("selector")
            if anchor_sel and anchor_sel in selector_map:
                anchor_logicals.add(selector_map[anchor_sel])
        translated = [
            {**f, "path": selector_map.get(str(f.get("path", "")), f.get("path"))}
            for f in batch.fields
        ]
        if batch.display_name:
            present = {str(f.get("path", "")) for f in translated}
            for logical in anchor_logicals:
                if logical not in present:
                    translated.append({"path": logical, "value": batch.display_name})
        return translated

    @router.post("/capture")
    async def capture(batch: CaptureBatch, request: Request):
        _require_loopback(request)
        _require_unlocked()
        written = get_store().ingest_batch(
            batch.patient_key,
            _translate_fields(batch),
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

    @router.post("/capture-api")
    async def capture_api(batch: ApiCaptureBatch, request: Request):
        # Cloud-EHR capture: the MAIN-world interceptor forwards an API response;
        # the matching pack's source.api map turns it into the patient record.
        _require_loopback(request)
        _require_unlocked()
        from adk.formbridge.api_capture import (
            encounter_id_from_url,
            encounter_scoped_fields,
            extract_api_fields,
            patient_key_from_url,
        )

        api_pack = next((p for p in _load_packs().values() if p.api_fields), None)
        if not api_pack:
            raise HTTPException(status_code=404, detail="no API-capture pack installed")
        key = patient_key_from_url(api_pack, batch.url)
        if not key:
            return {"ok": True, "written": 0, "note": "no patient context in URL"}
        fields = extract_api_fields(
            api_pack, batch.url, json_body=batch.body, rtf_text=batch.rtf,
        )
        if not fields:
            return {"ok": True, "written": 0}

        # Handle encounter scoping: if this response is from an encounter-specific
        # endpoint and the encounter id differs from the stored one, clear all
        # encounter-scoped fields first to prevent mixing data from two encounters.
        store = get_store()
        encounter_id = encounter_id_from_url(batch.url)
        if encounter_id:
            record = store.get_record(key)
            stored_encounter = record.get("_encounter", "")
            if stored_encounter and stored_encounter != encounter_id:
                # Switching encounters — delete the old encounter-scoped fields
                scoped_names = encounter_scoped_fields(api_pack)
                store.delete_fields(key, list(scoped_names))
            # Store the current encounter sentinel
            fields.append({"path": "_encounter", "value": encounter_id})

        written = store.ingest_batch(
            key, fields, display_name=batch.display_name, source_origin=batch.source_origin,
        )
        return {
            "ok": True,
            "patient_key": key,
            "written": written,
            "encounter_id": encounter_id or None,
            "fields": [f["path"] for f in fields],
        }

    @router.get("/patients")
    async def patients(request: Request):
        _require_loopback(request)
        _require_unlocked()
        return {"patients": get_store().list_patients()}

    @router.get("/patients/{patient_key}/documents")
    async def patient_documents(patient_key: str, request: Request):
        # Per-patient document history (filled forms + generated summaries). Each
        # entry's doc_id is the /download/{job_id} handle. Loopback-only.
        _require_loopback(request)
        _require_unlocked()
        return {"patient_key": patient_key, "documents": get_store().list_documents(patient_key)}

    @router.get("/patients/{patient_key}")
    async def patient_record(patient_key: str, request: Request):
        # Loopback-only like everything else; this is the ONE endpoint that
        # returns values (the local review surface needs them). It exists for
        # local UI use and must never be proxied or forwarded.
        _require_loopback(request)
        _require_unlocked()
        record = get_store().get_record(patient_key)
        if not record:
            raise HTTPException(status_code=404, detail="no captured record for that key")
        return {"patient_key": patient_key, "fields": record}

    @router.post("/fill")
    async def fill(req: FillRequest, request: Request):
        _require_loopback(request)
        _require_unlocked()
        if not _licensed():
            raise HTTPException(
                status_code=402,
                detail="form-bridge license required (set AITHER_FORMBRIDGE_DEV=1 for dev/demo)",
            )
        pack = _resolve_pack(req.pack_id)
        record = get_store().get_record(req.patient_key)
        if not record:
            raise HTTPException(status_code=404, detail="no captured record for that key")
        ref = f"{req.patient_key}:{req.form_id}"
        pw = _pdf_password(ref)
        try:
            resolution = resolve(record, pack, req.form_id)
            result = fill_pdf(pack, resolution, flatten=req.flatten, password=pw)
        except MappingError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        pw_ref = f"pdf:{ref}" if pw else ""
        get_store().add_document(
            req.patient_key, doc_id=result.job_id, form_id=req.form_id,
            pack_id=pack.pack_id, kind="form",
            filename=os.path.basename(result.output_path), path=result.output_path,
            encrypted=result.encrypted, password_ref=pw_ref,
        )
        return {
            "ok": True,
            "job_id": result.job_id,
            "output_path": result.output_path,
            "filled": result.filled,
            "unresolved": result.unresolved,
            "unresolved_reasons": resolution.unresolved_reasons,
            "llm_assist_fields": result.llm_assist_fields,
            "unknown_pdf_fields": result.unknown_pdf_fields,
            "encrypted": result.encrypted,
            # handle to query the per-doc password (never the password itself)
            "pdf_password_ref": pw_ref or None,
        }

    @router.post("/generate")
    async def generate(req: GenerateRequest, request: Request):
        # Auto-generate a PDF from the captured record — no customer template
        # needed (the same record fills their real AcroForm once we have it).
        _require_loopback(request)
        _require_unlocked()
        if not _licensed():
            raise HTTPException(
                status_code=402,
                detail="form-bridge license required (set AITHER_FORMBRIDGE_DEV=1 for dev/demo)",
            )
        record = get_store().get_record(req.patient_key)
        if not record:
            raise HTTPException(status_code=404, detail="no captured record for that key")
        ref = f"{req.patient_key}:generated"
        pw = _pdf_password(ref)
        try:
            result = generate_pdf(
                req.patient_key, record,
                title=req.title or "Captured Patient Record", password=pw,
            )
        except MappingError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        pw_ref = f"pdf:{ref}" if pw else ""
        get_store().add_document(
            req.patient_key, doc_id=result.job_id, form_id="", pack_id="",
            kind="summary", filename=os.path.basename(result.output_path),
            path=result.output_path, encrypted=result.encrypted, password_ref=pw_ref,
        )
        return {
            "ok": True,
            "job_id": result.job_id,
            "output_path": result.output_path,
            "filled": result.filled,
            "encrypted": result.encrypted,
            "pdf_password_ref": pw_ref or None,
        }

    @router.post("/purge")
    async def purge(req: PurgeRequest, request: Request):
        _require_loopback(request)
        # PHI cleanup: unlink the filled PDFs for this patient (the store deletes
        # the metadata rows). The store owns the DB; the route owns the files.
        store = get_store()
        docs = store.list_documents(req.patient_key) if req.patient_key else []
        deleted = store.purge(req.patient_key)
        for d in docs:
            try:
                p = d.get("path")
                if p and os.path.isfile(p):
                    os.remove(p)
            except OSError:
                logger.warning("formbridge: could not unlink purged document %s", d.get("filename"))
        return {"ok": True, "deleted": deleted}

    # ── Discovery (build a customer pack's selectors from the live EHR) ──

    @router.post("/discover")
    async def discover(req: DiscoverRequest, request: Request):
        _require_loopback(request)
        counts = get_discovery().ingest(req.origin, dom=req.dom, api=req.api)
        return {"ok": True, **counts}

    @router.get("/discovered")
    async def discovered(request: Request, origin: str = ""):
        _require_loopback(request)
        return get_discovery().get(origin or None)

    @router.post("/discover/generate")
    async def discover_generate(req: GeneratePackRequest, request: Request):
        _require_loopback(request)
        return {"ok": True, "yaml": generate_source_fields(req.origin, req.mappings)}

    @router.post("/discover/clear")
    async def discover_clear(request: Request, origin: str = ""):
        _require_loopback(request)
        get_discovery().clear(origin or None)
        return {"ok": True}

    @router.get("/download/{job_id}")
    async def download(job_id: str, request: Request):
        # Serve a generated/filled PDF back to the browser (download). Loopback
        # only; job_id is the opaque output id, validated + matched within the
        # output dir (no arbitrary-path access).
        _require_loopback(request)
        if not re.fullmatch(r"[A-Za-z0-9]{6,40}", job_id):
            raise HTTPException(status_code=400, detail="invalid job id")
        from adk.formbridge.pdf_fill import output_dir

        matches = sorted(output_dir().glob(f"*{job_id}*.pdf"))
        if not matches:
            raise HTTPException(status_code=404, detail="no such output")
        return FileResponse(
            str(matches[0]), media_type="application/pdf", filename=matches[0].name,
        )

    return router
