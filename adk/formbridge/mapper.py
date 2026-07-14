"""Mapping engine — mapping.yaml schema v1 → resolved PDF field values.

Deterministic by design: a captured record plus a mapping produces exactly one
fill dict. Anything the mapping cannot resolve lands in ``unresolved`` (it is
flagged for the human, never guessed), and missing ``required`` source fields
hard-fail the whole form.

Schema v1 (authority: .PRODUCTS/.FORMBRIDGE/05-API-DESIGN.md):

    pack: demo-clinic
    version: 1.0.0
    source:
      origin: "https://app.example"
      anchor: { selector: "...", key_selector: "..." }
      self_check: [ { screen: "soap", expect: ["#subjective", ...] } ]
      ignore: ["input[type=password]", "[name*=ssn]"]
      fields:
        - { capture: "#subjective", as: "soap.subjective" }
        - { capture: "input[name=dob]", as: "patient.dob", transform: "date:MM/DD/YYYY" }
    forms:
      - id: new-patient-intake
        template: templates/new-patient-intake.pdf
        fill:
          - { pdf_field: "PatientName", from: "patient.name" }
          - { pdf_field: "Subjective", from: "soap.subjective", llm_assist: true, max_len: 800 }
          - { pdf_field: "Carrier_BCBS", from: "insurance.carrier", transform: "checkbox:equals(BCBS)" }
        required: ["patient.name", "patient.dob"]

Transforms v1: date:<fmt> · name_split:first|last · checkbox:equals(v)|contains(v)
· concat(a,b,...) · truncate(n) · phone:###-###-####
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger("adk.formbridge.mapper")

#: pypdf checkbox "on" value — standard AcroForm convention.
CHECKBOX_ON = "/Yes"
CHECKBOX_OFF = "/Off"


class MappingError(ValueError):
    """Raised for invalid mapping packs or hard-fail fill conditions."""


@dataclass
class FormDef:
    id: str
    template: str
    fill: list[dict]
    required: list[str] = field(default_factory=list)
    mode: str = "acroform"  # acroform = fill named fields; overlay = draw text at x/y (flat scans)


@dataclass
class MappingPack:
    pack_id: str
    version: str
    origin: str
    anchor: dict
    capture_fields: list[dict]
    ignore: list[str]
    self_check: list[dict]
    forms: dict[str, FormDef]
    root: Path  # pack directory — templates resolve relative to this
    #: Tabbed single-editor capture config (the ChiroTouch Chart Notes shape).
    #: Extension-side capture hint; ignored by the deterministic mapper.
    tabbed_editors: list[dict] = field(default_factory=list)
    #: The EHR's entry URL (the doctor's EHR login page; the demo mock in dev).
    #: Surfaced so the UI can offer an "Open EHR" link.
    ehr_url: str = ""
    #: API-response capture (cloud EHRs like ChiroTouch): list of
    #: {endpoint, fields:[{path, as}]} OR {endpoint, kind: rtf, as}. The engine
    #: reads the app's OWN JSON/RTF responses instead of scraping the DOM.
    api_fields: list[dict] = field(default_factory=list)
    #: Static per-office config merged into every record at fill time (the office's
    #: identity on the form: provider name/address, BWC number, phone/fax). Set once.
    office: dict = field(default_factory=dict)
    #: For API packs: the URL path segment whose following id keys the record
    #: (e.g. "patients" → .../patients/{patientId}/...).
    url_id_segment: str = ""

    def form(self, form_id: str) -> FormDef:
        f = self.forms.get(form_id)
        if not f:
            raise MappingError(
                f"Unknown form '{form_id}' — available: {sorted(self.forms)}"
            )
        return f

    def template_path(self, form_id: str) -> Path:
        p = (self.root / self.form(form_id).template).resolve()
        # Templates must live inside the pack — a mapping is data, never an
        # escape hatch to arbitrary filesystem paths.
        if not str(p).startswith(str(self.root.resolve())):
            raise MappingError(f"Template path escapes pack root: {p}")
        return p


def load_pack(pack_dir: Path | str) -> MappingPack:
    """Load + validate a mapping pack directory (must contain mapping.yaml)."""
    root = Path(pack_dir)
    mapping_file = root / "mapping.yaml"
    if not mapping_file.is_file():
        raise MappingError(f"No mapping.yaml in {root}")
    raw = yaml.safe_load(mapping_file.read_text(encoding="utf-8")) or {}

    src = raw.get("source") or {}
    anchor = src.get("anchor") or {}
    api_fields = list(src.get("api") or [])
    # Two capture modes: DOM (needs an anchor.selector) OR API (reads the app's own
    # responses — keyed by a URL id segment, no DOM selector). Require one of them.
    if not anchor.get("selector") and not api_fields:
        raise MappingError(
            "pack needs either source.anchor.selector (DOM capture) or source.api (API capture)"
        )
    # origin pins the DOM-capture host; API packs may use `base` instead.
    origin = (src.get("origin") or src.get("base") or "").rstrip("/")
    if not origin and not api_fields:
        raise MappingError("source.origin is required for DOM-capture packs")

    forms: dict[str, FormDef] = {}
    for f in raw.get("forms") or []:
        fid = f.get("id")
        if not fid or not f.get("template"):
            raise MappingError(f"Form entry needs id + template: {f}")
        mode = str(f.get("mode", "acroform"))
        if mode not in ("acroform", "overlay"):
            raise MappingError(f"Form {fid}: mode must be acroform|overlay, got {mode!r}")
        entries = f.get("fill") or []
        for e in entries:
            if not e.get("pdf_field") or not e.get("from"):
                raise MappingError(f"fill entry needs pdf_field + from (form {fid}): {e}")
            if mode == "overlay" and not (
                isinstance(e.get("x"), (int, float)) and isinstance(e.get("y"), (int, float))
            ):
                raise MappingError(
                    f"overlay fill entry needs numeric x + y (form {fid}): {e}"
                )
        forms[fid] = FormDef(
            id=fid,
            template=f["template"],
            fill=entries,
            required=list(f.get("required") or []),
            mode=mode,
        )
    if not forms:
        raise MappingError("Mapping pack defines no forms")

    return MappingPack(
        pack_id=raw.get("pack", root.name),
        version=str(raw.get("version", "0.0.0")),
        origin=origin,
        anchor=anchor,
        capture_fields=list(src.get("fields") or []),
        ignore=list(src.get("ignore") or []),
        self_check=list(src.get("self_check") or []),
        forms=forms,
        root=root,
        tabbed_editors=list(src.get("tabbed_editors") or []),
        ehr_url=(src.get("ehr_url") or "").strip(),
        api_fields=api_fields,
        office=dict(raw.get("office") or {}),
        url_id_segment=str(anchor.get("url_id_segment") or "").strip(),
    )


# ── Transforms ───────────────────────────────────────────────────────────────

_DATE_TOKENS = [
    ("YYYY", "%Y"), ("MM", "%m"), ("DD", "%d"), ("YY", "%y"),
]

_DATE_INPUT_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d",
    "%b %d, %Y", "%B %d, %Y", "%m/%d/%y",
]


def _parse_date(value: str) -> datetime:
    v = value.strip()
    for fmt in _DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    raise MappingError(f"Unparseable date value: {value!r}")


def _t_date(value: str, fmt_spec: str) -> str:
    out_fmt = fmt_spec
    for token, strf in _DATE_TOKENS:
        out_fmt = out_fmt.replace(token, strf)
    return _parse_date(value).strftime(out_fmt)


def _t_name_split(value: str, part: str) -> str:
    v = value.strip()
    if not v:
        return ""
    if "," in v:  # "Last, First"
        last, _, first = v.partition(",")
        first, last = first.strip(), last.strip()
    else:
        bits = v.split()
        first = bits[0]
        last = " ".join(bits[1:]) if len(bits) > 1 else ""
    if part == "first":
        return first
    if part == "last":
        return last
    raise MappingError(f"name_split part must be first|last, got {part!r}")


def _t_checkbox(value: str, mode: str, arg: str) -> str:
    v = value.strip().lower()
    a = arg.strip().lower()
    if mode == "equals":
        return CHECKBOX_ON if v == a else CHECKBOX_OFF
    if mode == "contains":
        return CHECKBOX_ON if a in v else CHECKBOX_OFF
    raise MappingError(f"checkbox mode must be equals|contains, got {mode!r}")


def _t_phone(value: str, pattern: str) -> str:
    digits = re.sub(r"\D", "", value)
    needed = pattern.count("#")
    if len(digits) == needed + 1 and digits.startswith("1"):
        digits = digits[1:]  # tolerate a leading country code
    if len(digits) != needed:
        raise MappingError(f"Phone {value!r} has {len(digits)} digits, pattern needs {needed}")
    out = []
    it = iter(digits)
    for ch in pattern:
        out.append(next(it) if ch == "#" else ch)
    return "".join(out)


_TRANSFORM_RE = re.compile(
    r"^(?P<name>[a-z_]+)(?::(?P<colon_arg>[^()]*))?(?:\((?P<paren_arg>.*)\))?$"
)


def apply_transform(value: str, spec: str, record: dict[str, str] | None = None) -> str:
    """Apply one transform spec to a value. Raises MappingError on bad specs."""
    m = _TRANSFORM_RE.match(spec.strip())
    if not m:
        raise MappingError(f"Unparseable transform: {spec!r}")
    name = m.group("name")
    colon_arg = m.group("colon_arg")
    paren_arg = m.group("paren_arg")

    if name == "date":
        if not colon_arg:
            raise MappingError("date transform needs a format, e.g. date:MM/DD/YYYY")
        return _t_date(value, colon_arg)
    if name == "name_split":
        return _t_name_split(value, colon_arg or "")
    if name == "checkbox":
        # checkbox:equals(BCBS) — the mode rides the colon segment up to '('.
        if colon_arg is None or paren_arg is None:
            raise MappingError("checkbox transform: checkbox:equals(v) or checkbox:contains(v)")
        return _t_checkbox(value, colon_arg, paren_arg)
    if name == "truncate":
        if not paren_arg or not paren_arg.isdigit():
            raise MappingError("truncate transform needs a length: truncate(80)")
        return value[: int(paren_arg)]
    if name == "concat":
        if paren_arg is None:
            raise MappingError("concat transform needs field names: concat(a,b)")
        rec = record or {}
        parts = [rec.get(p.strip(), "") for p in paren_arg.split(",")]
        return " ".join(p for p in parts if p)
    if name == "phone":
        if not colon_arg:
            raise MappingError("phone transform needs a pattern: phone:###-###-####")
        return _t_phone(value, colon_arg)
    raise MappingError(f"Unknown transform: {name!r}")


# ── Resolution ───────────────────────────────────────────────────────────────

@dataclass
class FillResolution:
    form_id: str
    values: dict[str, str]          # pdf_field -> final value
    unresolved: list[str]           # pdf_field names with no/failed source
    llm_assist_fields: list[str]    # pdf_field names flagged llm_assist (review!)
    #: pdf_field -> human-readable reason it's unresolved (no value vs failed
    #: transform). Surfaced to the operator so "unresolved" isn't a silent mystery.
    unresolved_reasons: dict[str, str] = field(default_factory=dict)


def _as_list(raw: str) -> list | None:
    """An API array-capture value is stored as a JSON array string. Return the
    list if `raw` is one, else None (a plain scalar)."""
    if raw[:1] == "[":
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else None
        except (ValueError, TypeError):
            return None
    return None


def _resolve_from(entry: dict, record: dict[str, str]) -> str:
    """Resolve a fill entry's source value. Handles @today, concat[] (join several
    fields), index (one element of an array-captured field), and a plain field
    (an array value joins with ', '; a scalar passes through)."""
    src = entry.get("from", "")
    if src == "@today":
        return datetime.now().strftime("%Y-%m-%d")
    concat = entry.get("concat")
    if concat:
        parts = []
        for k in concat:
            v = record.get(k, "")
            lst = _as_list(v)
            parts.append(str(lst[0]) if lst else v)
        return " ".join(p.strip() for p in parts if p and p.strip())
    raw = record.get(src, "")
    idx = entry.get("index")
    lst = _as_list(raw)
    if idx is not None:
        if lst is not None:
            return str(lst[idx]).strip() if 0 <= idx < len(lst) else ""
        return raw if idx == 0 else ""          # scalar: index 0 is the value
    if lst is not None:
        return ", ".join(str(x) for x in lst if str(x).strip())
    return raw


def resolve(record: dict[str, str], pack: MappingPack, form_id: str) -> FillResolution:
    """Resolve a captured record against one form. Deterministic; never guesses.

    Raises MappingError listing missing fields when any `required` source
    field is absent/empty (hard fail — 06-SECURITY-MODEL rule: a partial
    regulated form must never look complete).
    """
    form = pack.form(form_id)

    # Static office config is just more record fields (captured data overrides it).
    record = {**pack.office, **record}

    missing = [r for r in form.required if not record.get(r, "").strip()]
    if missing:
        raise MappingError(
            f"Required source fields missing for form '{form_id}': {sorted(missing)}"
        )

    values: dict[str, str] = {}
    unresolved: list[str] = []
    unresolved_reasons: dict[str, str] = {}
    llm_fields: list[str] = []

    for entry in form.fill:
        pdf_field = entry["pdf_field"]
        source = entry["from"]
        raw = _resolve_from(entry, record)
        if entry.get("llm_assist"):
            llm_fields.append(pdf_field)

        transform = entry.get("transform")
        if not raw.strip():
            unresolved.append(pdf_field)
            unresolved_reasons[pdf_field] = f"no captured value for '{source}'"
            continue
        try:
            value = apply_transform(raw, transform, record) if transform else raw
        except MappingError as e:
            logger.warning("formbridge: transform failed for %s: %s", pdf_field, e)
            unresolved.append(pdf_field)
            unresolved_reasons[pdf_field] = str(e)
            continue

        max_len = entry.get("max_len")
        if isinstance(max_len, int) and max_len > 0:
            value = value[:max_len]
        values[pdf_field] = value

    return FillResolution(
        form_id=form_id,
        values=values,
        unresolved=sorted(unresolved),
        llm_assist_fields=sorted(llm_fields),
        unresolved_reasons=unresolved_reasons,
    )


def run_self_check(pack: MappingPack, seen_selectors: list[str]) -> dict:
    """Evaluate self_check probes against selectors observed on the page.

    The capture script reports which expected selectors were present; missing
    ones indicate the source UI drifted (mapping pack needs an update).
    """
    seen = set(seen_selectors)
    failures = []
    for probe in pack.self_check:
        expected = probe.get("expect") or []
        absent = [s for s in expected if s not in seen]
        if absent:
            failures.append({"screen": probe.get("screen", "?"), "missing": absent})
    return {"ok": not failures, "failures": failures, "pack_version": pack.version}
