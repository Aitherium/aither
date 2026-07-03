"""PDF AcroForm filling via pypdf (optional `pdf` extra).

Output filenames are opaque job ids — NEVER patient names (06-SECURITY-MODEL:
no PHI in paths; paths cross the MCP/tool boundary as results).
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from adk.formbridge.mapper import CHECKBOX_ON, FillResolution, MappingError, MappingPack

logger = logging.getLogger("adk.formbridge.pdf")

try:  # graceful when the extra isn't installed — mirror adk's voice/graphs pattern
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, NumberObject

    HAS_PYPDF = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYPDF = False

try:  # overlay mode (flat scans) additionally needs fpdf2
    from fpdf import FPDF

    HAS_FPDF = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_FPDF = False


def _require_pypdf() -> None:
    if not HAS_PYPDF:
        raise MappingError(
            "pypdf is not installed — install the pdf extra: pip install 'aither-adk[pdf]'"
        )


def output_dir() -> Path:
    d = os.getenv("AITHER_FORMBRIDGE_OUTPUT", "").strip()
    candidates = [Path(d)] if d else []
    # Documents may be a redirected known folder that doesn't exist at its
    # literal path (OneDrive/roaming profiles) — fall back under ~/.aither.
    candidates += [
        Path.home() / "Documents" / "FormBridge",
        Path.home() / ".aither" / "formbridge" / "output",
    ]
    last_err: Exception | None = None
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as e:
            last_err = e
    raise MappingError(f"No writable FormBridge output directory: {last_err}")


def list_template_fields(template: Path | str) -> list[str]:
    """Enumerate AcroForm field names in a template PDF."""
    _require_pypdf()
    reader = PdfReader(str(template))
    fields = reader.get_fields() or {}
    return sorted(fields.keys())


@dataclass
class FillResult:
    job_id: str
    output_path: str
    filled: list[str]
    unresolved: list[str]
    llm_assist_fields: list[str]
    unknown_pdf_fields: list[str]  # mapping referenced fields the template lacks
    encrypted: bool = False        # output PDF is password-protected (AES-256)


def _finalize(writer: "PdfWriter", out: Path, password: str | None) -> bool:
    """Optionally AES-256 password-protect the PDF, then write it. Returns True
    if encrypted. The password comes from the office vault (per-document)."""
    enc = False
    if password:
        try:
            writer.encrypt(password, algorithm="AES-256")
        except TypeError:  # older pypdf without the algorithm kwarg
            writer.encrypt(password)
        enc = True
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        writer.write(fh)
    return enc


def _fill_overlay(
    pack: MappingPack,
    resolution: FillResolution,
    template: Path,
    out_dir: Path | None,
    password: str | None = None,
) -> FillResult:
    """Overlay mode — draw resolved values at x/y onto a flat (scanned) form.

    Coordinates are PDF POINTS measured from the TOP-LEFT of each page (how
    you'd measure from a screenshot). Checkbox values draw an X; /Off draws
    nothing. Optional per-entry: page (0-based, default 0), size (pt, default 10).
    """
    if not HAS_FPDF:
        raise MappingError(
            "fpdf2 is not installed — overlay forms need the pdf extra: pip install 'aither-adk[pdf]'"
        )
    form = pack.form(resolution.form_id)
    coords = {e["pdf_field"]: e for e in form.fill}

    reader = PdfReader(str(template))
    page_sizes = [
        (float(p.mediabox.width), float(p.mediabox.height)) for p in reader.pages
    ]

    overlay = FPDF(unit="pt", format=page_sizes[0])
    overlay.set_auto_page_break(False)
    overlay.set_font("Helvetica", size=10)
    for w, h in page_sizes:
        overlay.add_page(format=(w, h))
    for name, value in resolution.values.items():
        entry = coords.get(name) or {}
        page_idx = int(entry.get("page", 0))
        if page_idx >= len(page_sizes):
            continue
        text = "X" if value == CHECKBOX_ON else ("" if value == "/Off" else value)
        if not text:
            continue
        overlay.page = page_idx + 1  # fpdf pages are 1-based
        overlay.set_font_size(float(entry.get("size", 10)))
        overlay.set_xy(float(entry["x"]), float(entry["y"]))
        overlay.cell(text=text)

    overlay_reader = PdfReader(BytesIO(overlay.output()))
    writer = PdfWriter()
    writer.append(reader)
    # merge on writer-attached pages (merging reader pages is deprecated in pypdf)
    for idx, page in enumerate(writer.pages):
        if idx < len(overlay_reader.pages):
            page.merge_page(overlay_reader.pages[idx])

    job_id = uuid.uuid4().hex[:12]
    out = (out_dir or output_dir()) / f"{resolution.form_id}-{job_id}.pdf"
    enc = _finalize(writer, out, password)

    logger.info(
        "formbridge: overlay-filled %s -> %s (%d fields, %d unresolved%s)",
        resolution.form_id, out.name, len(resolution.values), len(resolution.unresolved),
        ", encrypted" if enc else "",
    )
    return FillResult(
        job_id=job_id,
        output_path=str(out),
        filled=sorted(resolution.values.keys()),
        unresolved=resolution.unresolved,
        llm_assist_fields=resolution.llm_assist_fields,
        unknown_pdf_fields=[],
        encrypted=enc,
    )


def fill_pdf(
    pack: MappingPack,
    resolution: FillResolution,
    *,
    flatten: bool = False,
    out_dir: Path | None = None,
    password: str | None = None,
) -> FillResult:
    """Fill the form's template with resolved values; write to the output dir.

    AcroForm mode fills named fields (checkboxes via the /Yes-/Off convention
    set by the checkbox transform). Overlay mode (mapping `mode: overlay`)
    draws text at x/y for flat scanned forms with no fields.
    """
    _require_pypdf()
    template = pack.template_path(resolution.form_id)
    if not template.is_file():
        raise MappingError(f"Template not found: {template}")

    if pack.form(resolution.form_id).mode == "overlay":
        return _fill_overlay(pack, resolution, template, out_dir, password)

    reader = PdfReader(str(template))
    writer = PdfWriter()
    writer.append(reader)

    template_fields = set((reader.get_fields() or {}).keys())
    unknown = sorted(f for f in resolution.values if f not in template_fields)
    if unknown:
        logger.warning(
            "formbridge: mapping references PDF fields missing from template %s: %s",
            template.name, unknown,
        )

    text_values: dict[str, str] = {}
    checkbox_values: dict[str, str] = {}
    for name, value in resolution.values.items():
        if name not in template_fields:
            continue
        if value in (CHECKBOX_ON, "/Off"):
            checkbox_values[name] = value
        else:
            text_values[name] = value

    for page in writer.pages:
        if text_values:
            writer.update_page_form_field_values(page, text_values)
        if checkbox_values:
            # Set checkbox /V + /AS directly on the widget annotations:
            # update_page_form_field_values requires an /AP appearance dict
            # (KeyError on minimal forms), while direct state-setting works on
            # any AcroForm — viewers draw the check glyph from /AS.
            annots = page.get("/Annots") or []
            for annot in annots:
                obj = annot.get_object()
                name = obj.get("/T")
                if name and str(name) in checkbox_values:
                    state = NameObject(checkbox_values[str(name)])
                    obj[NameObject("/V")] = state
                    obj[NameObject("/AS")] = state

    # NeedAppearances so viewers regenerate field appearance streams —
    # without it many viewers show filled fields as blank until clicked.
    try:
        writer.set_need_appearances_writer(True)
    except AttributeError:  # older pypdf
        pass

    if flatten:
        # Make fields read-only rather than stripping the AcroForm: keeps the
        # visual result identical across viewers while preventing edits.
        for page in writer.pages:
            annots = page.get("/Annots")
            if not annots:
                continue
            for annot in annots:
                obj = annot.get_object()
                ff = int(obj.get("/Ff", 0))
                obj[NameObject("/Ff")] = NumberObject(ff | 1)  # bit 1 = ReadOnly

    job_id = uuid.uuid4().hex[:12]
    out = (out_dir or output_dir()) / f"{resolution.form_id}-{job_id}.pdf"
    enc = _finalize(writer, out, password)

    logger.info(
        "formbridge: filled %s -> %s (%d fields, %d unresolved%s)",
        resolution.form_id, out.name, len(resolution.values), len(resolution.unresolved),
        ", encrypted" if enc else "",
    )
    return FillResult(
        job_id=job_id,
        output_path=str(out),
        filled=sorted(resolution.values.keys()),
        unresolved=resolution.unresolved,
        llm_assist_fields=resolution.llm_assist_fields,
        unknown_pdf_fields=unknown,
        encrypted=enc,
    )


def _humanize_label(key: str) -> str:
    """Readable label for the generated-record PDF, so an UNMAPPED capture shows
    as "Dob" not "input[name=dob]" and a logical name "provider.name_address" shows
    as "Provider Name Address". Falls back to the raw key if nothing's left."""
    s = str(key)
    m = re.search(r"\[name=([^\]]+)\]", s)        # input[name=dob] -> dob
    if m:
        s = m.group(1)
    elif s[:1] in "#.":                            # #patient-id / .field -> strip marker
        s = s[1:]
    s = re.sub(r"[._\-]+", " ", s).strip()         # dots/underscores/hyphens -> spaces
    return s.title() if s else str(key)


def generate_pdf(
    patient_key: str,
    record: dict[str, str],
    *,
    title: str = "Captured Patient Record",
    out_dir: Path | None = None,
    password: str | None = None,
) -> FillResult:
    """Render a captured record into a clean PDF via fpdf2 — NO AcroForm template
    required. For offices whose real form we don't have yet; the SAME record fills
    their real PDF once we do. Optionally AES-256 password-protected from the vault."""
    if not HAS_FPDF:
        raise MappingError(
            "fpdf2 is not installed — generate needs the pdf extra: pip install 'aither-adk[pdf]'"
        )
    pdf = FPDF(unit="pt", format="letter")
    pdf.set_auto_page_break(True, margin=54)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 24, text=title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 14, text=f"Patient: {patient_key}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_text_color(20, 20, 20)
    for key in sorted(record):
        val = record[key]
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(170, 16, text=_humanize_label(key))
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 16, text=(str(val) if val else "—"), new_x="LMARGIN", new_y="NEXT")
    raw = bytes(pdf.output())

    job_id = uuid.uuid4().hex[:12]
    out = (out_dir or output_dir()) / f"generated-{job_id}.pdf"
    enc = False
    if password:
        _require_pypdf()
        writer = PdfWriter()
        writer.append(PdfReader(BytesIO(raw)))
        enc = _finalize(writer, out, password)
    else:
        out.write_bytes(raw)
    logger.info(
        "formbridge: generated PDF for %s -> %s (%d fields%s)",
        patient_key, out.name, len(record), ", encrypted" if enc else "",
    )
    return FillResult(
        job_id=job_id, output_path=str(out),
        filled=sorted(record.keys()), unresolved=[], llm_assist_fields=[],
        unknown_pdf_fields=[], encrypted=enc,
    )
