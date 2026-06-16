"""API-response capture — read a cloud EHR's OWN responses, not its DOM.

For cloud apps (ChiroTouch, …) the data the office types is served back to the
SPA as JSON (+ chart notes as RTF). FormBridge's MAIN-world interceptor sees
those responses in the doctor's authenticated session; this module turns one
response into logical fields per the pack's ``source.api`` map — no DOM
selectors, no credentials. Pure functions (no I/O), so they unit-test cleanly.

See AitherOS .PRODUCTS/.FORMBRIDGE/CHIROTOUCH-INTEGRATION-MAP.md for the schema.
"""
from __future__ import annotations

import json
import re
from typing import Any

from adk.formbridge.mapper import MappingPack

_SOAP_HDR = re.compile(r"\b(subjective|objective|assessment|plan)\b\s*:?", re.I)


def endpoint_matches(pattern: str, url: str) -> bool:
    """A pack endpoint pattern uses ``*`` to skip any run of path characters
    (one or more segments, never crossing into the query string), matched
    anywhere in the URL — e.g. 'core-api/*/encounters/*/diagnoses' matches
    '.../core-api/api/v1/tenants/abc/encounters/123/diagnoses'."""
    if not pattern:
        return False
    rx = re.escape(pattern).replace(r"\*", r"[^?]*")
    return re.search(rx, url) is not None


def json_path(node: Any, path: str) -> list:
    """Resolve a dotted path with ``[]`` iteration markers → a FLAT list of leaf
    values. 'data[].code' over {data:[{code:'A'},{code:'B'}]} -> ['A','B'].
    '[].procedure' iterates a top-level list. 'data.x' is a plain descend."""
    out: list = []

    def descend(cur: Any, segs: list[str]) -> None:
        if cur is None:
            return
        if not segs:
            if not isinstance(cur, (dict, list)):
                out.append(cur)
            return
        seg, rest = segs[0], segs[1:]
        iterate = seg.endswith("[]")
        key = seg[:-2] if iterate else seg
        if key:
            if not isinstance(cur, dict):
                return
            cur = cur.get(key)
        if iterate:
            if isinstance(cur, list):
                for item in cur:
                    descend(item, rest)
        else:
            descend(cur, rest)

    descend(node, path.split("."))
    return out


def _drop_rtf_groups(s: str, keywords: tuple[str, ...]) -> str:
    """Remove whole `{\\fonttbl …}` / `{\\colortbl …}` etc. destination groups so
    their contents (font names, "Arial;") don't leak into the text. A group is
    dropped only when ITS OWN leading control word is a header keyword — nested
    keywords don't count (the outer `{\\rtf1 …}` must survive and be descended)."""
    # {  optional whitespace  optional \*  \<word>
    lead = re.compile(r"\{\s*(?:\\\*\s*)?\\([a-zA-Z]+)")
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == "{":
            depth, j = 0, i
            while j < n:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            m = lead.match(s, i)
            if m and m.group(1) in keywords:
                i = j + 1                     # drop the whole destination group
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def rtf_to_text(rtf: str) -> str:
    """Best-effort RTF → plain text (enough for a dictated chart note). Strips
    control words/symbols, hex/unicode escapes, and groups; \\par/\\line → newline.
    Not a full RTF parser — complex docs (tables, embedded objects) degrade."""
    if not rtf:
        return ""
    s = _drop_rtf_groups(rtf, ("fonttbl", "colortbl", "stylesheet", "info", "generator"))
    s = re.sub(r"\\'[0-9a-fA-F]{2}", "", s)        # \'hh hex escapes
    s = re.sub(r"\\u-?\d+\??", "", s)              # \uNNNN unicode escapes
    s = re.sub(r"\\(par|line)\b", "\n", s)         # paragraph/line breaks
    s = re.sub(r"\\\*", "", s)                       # \* (ignorable group marker)
    s = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", s)        # control words (\b, \fs24, …)
    s = re.sub(r"[{}]", "", s)                       # group braces
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def split_soap(text: str) -> dict[str, str]:
    """Split a SOAP narrative into subjective/objective/assessment/plan by header.
    Best-effort — returns only the sections it can find."""
    out: dict[str, str] = {}
    matches = list(_SOAP_HDR.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip(" :\n\t")
        if section:
            out[key] = section
    return out


def patient_key_from_url(pack: MappingPack, url: str) -> str:
    """The record key = the id segment after pack.url_id_segment in the URL
    (e.g. url_id_segment='patients' → .../patients/{key}/...)."""
    seg = pack.url_id_segment
    if not seg:
        return ""
    m = re.search(r"/" + re.escape(seg) + r"/([^/?]+)", url)
    return m.group(1) if m else ""


def encounter_id_from_url(url: str) -> str:
    """Extract the encounter id from a URL if present.
    E.g. .../encounters/123/diagnoses → '123'."""
    m = re.search(r"/encounters/([^/?]+)", url)
    return m.group(1) if m else ""


def encounter_scoped_fields(pack: MappingPack) -> set[str]:
    """Return the set of logical 'as' names that come from encounter-scoped
    API endpoints (where the endpoint pattern contains 'encounters/')."""
    scoped: set[str] = set()
    for ep in pack.api_fields:
        endpoint = ep.get("endpoint", "")
        if "encounters/" not in endpoint:
            continue
        # RTF endpoints may not have a 'fields' key — check the 'as' field
        if ep.get("kind") == "rtf":
            logical = ep.get("as")
            if logical:
                scoped.add(logical)
        # JSON endpoints have fields
        for f in ep.get("fields", []):
            logical = f.get("as")
            if logical:
                scoped.add(logical)
    return scoped


def extract_api_fields(
    pack: MappingPack, url: str, *, json_body: Any = None, rtf_text: str | None = None
) -> list[dict]:
    """One captured API response → [{path: logical_name, value: str}]. Multi-value
    paths (arrays) are stored as a JSON array string so an indexed fill can pick
    one element (e.g. the 5 ICD codes on the MEDCO-14)."""
    out: list[dict] = []
    for ep in pack.api_fields:
        if not endpoint_matches(ep.get("endpoint", ""), url):
            continue
        if ep.get("kind") == "rtf":
            text = rtf_to_text(rtf_text or "")
            if not text:
                continue
            base = ep.get("as", "soap.narrative")
            out.append({"path": base, "value": text})
            if base.startswith("soap"):
                for k, v in split_soap(text).items():
                    out.append({"path": f"soap.{k}", "value": v})
            continue
        for f in ep.get("fields", []):
            path, logical = f.get("path"), f.get("as")
            if not path or not logical:
                continue
            vals = [str(v).strip() for v in json_path(json_body, path) if v is not None and str(v).strip()]
            if not vals:
                continue
            out.append({"path": logical, "value": vals[0] if len(vals) == 1 else json.dumps(vals)})
    return out
