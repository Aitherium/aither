"""PDF AcroForm filling via pypdf (optional `pdf` extra).

Output filenames are opaque job ids — NEVER patient names (06-SECURITY-MODEL:
no PHI in paths; paths cross the MCP/tool boundary as results).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from adk.formbridge.mapper import CHECKBOX_ON, FillResolution, MappingError, MappingPack

logger = logging.getLogger("adk.formbridge.pdf")

try:  # graceful when the extra isn't installed — mirror adk's voice/graphs pattern
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject, NumberObject

    HAS_PYPDF = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_PYPDF = False


def _require_pypdf() -> None:
    if not HAS_PYPDF:
        raise MappingError(
            "pypdf is not installed — install the pdf extra: pip install 'aither-adk[pdf]'"
        )


def output_dir() -> Path:
    d = os.getenv("AITHER_FORMBRIDGE_OUTPUT", "").strip()
    path = Path(d) if d else Path.home() / "Documents" / "FormBridge"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def fill_pdf(
    pack: MappingPack,
    resolution: FillResolution,
    *,
    flatten: bool = False,
    out_dir: Path | None = None,
) -> FillResult:
    """Fill the form's template with resolved values; write to the output dir.

    Checkbox values use the /Yes-/Off AcroForm convention (set by the
    checkbox transform); everything else is written as text.
    """
    _require_pypdf()
    template = pack.template_path(resolution.form_id)
    if not template.is_file():
        raise MappingError(f"Template not found: {template}")

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
            # update_page_form_field_values handles checkboxes when handed
            # NameObjects; build them per page.
            writer.update_page_form_field_values(
                page, {k: NameObject(v) for k, v in checkbox_values.items()}
            )

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
    with open(out, "wb") as fh:
        writer.write(fh)

    logger.info(
        "formbridge: filled %s -> %s (%d fields, %d unresolved)",
        resolution.form_id, out.name, len(resolution.values), len(resolution.unresolved),
    )
    return FillResult(
        job_id=job_id,
        output_path=str(out),
        filled=sorted(resolution.values.keys()),
        unresolved=resolution.unresolved,
        llm_assist_fields=resolution.llm_assist_fields,
        unknown_pdf_fields=unknown,
    )
