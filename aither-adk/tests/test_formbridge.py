"""FormBridge engine tests — store, mapper transforms, PDF round-trip, routes, gates.

Uses the demo-clinic pack fixture pattern but builds its own minimal pack in
tmp_path so the suite has zero dependency on the AitherOS Library checkout.
"""

from __future__ import annotations

import pytest

pypdf = pytest.importorskip("pypdf", reason="formbridge tests need the pdf extra")

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from adk.formbridge import store as store_mod
from adk.formbridge.mapper import (
    MappingError,
    apply_transform,
    load_pack,
    resolve,
    run_self_check,
)
from adk.formbridge.pdf_fill import fill_pdf, list_template_fields
from adk.formbridge.routes import create_formbridge_router
from adk.formbridge.store import CaptureStore

MAPPING_YAML = """
pack: test-clinic
version: 1.0.0
source:
  origin: "http://localhost:8910"
  anchor: { selector: "#patient-name", key_selector: "#patient-id" }
  self_check:
    - screen: soap
      expect: ["#subjective", "#objective"]
  ignore: ["[name*=ssn]"]
  fields:
    - { capture: "#patient-name", as: "patient.name" }
forms:
  - id: intake
    template: templates/intake.pdf
    required: ["patient.name", "patient.dob"]
    fill:
      - { pdf_field: "PatientName",  from: "patient.name" }
      - { pdf_field: "PatientFirst", from: "patient.name", transform: "name_split:first" }
      - { pdf_field: "PatientLast",  from: "patient.name", transform: "name_split:last" }
      - { pdf_field: "DOB",          from: "patient.dob",  transform: "date:MM/DD/YYYY" }
      - { pdf_field: "Phone",        from: "patient.phone", transform: "phone:###-###-####" }
      - { pdf_field: "Carrier_BCBS", from: "insurance.carrier", transform: "checkbox:equals(BCBS)" }
      - { pdf_field: "Subjective",   from: "soap.subjective", llm_assist: true, max_len: 20 }
"""

RECORD = {
    "patient.name": "Pat Example",
    "patient.dob": "1980-04-12",
    "patient.phone": "(555) 010-1234",
    "insurance.carrier": "BCBS",
    "soap.subjective": "Lower back pain radiating to the left leg for two weeks",
}


def _make_template(path: Path, field_names: list[str]) -> None:
    """Minimal AcroForm PDF with text fields (checkbox for Carrier_*)."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    annots = ArrayObject()
    y = 700
    for name in field_names:
        f = DictionaryObject()
        is_checkbox = name.startswith("Carrier_")
        f.update({
            NameObject("/FT"): NameObject("/Btn" if is_checkbox else "/Tx"),
            NameObject("/T"): TextStringObject(name),
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/Rect"): ArrayObject(
                [NumberObject(60), NumberObject(y), NumberObject(260), NumberObject(y + 18)]
            ),
            NameObject("/F"): NumberObject(4),
        })
        if is_checkbox:
            f[NameObject("/V")] = NameObject("/Off")
            f[NameObject("/AS")] = NameObject("/Off")
        else:
            f[NameObject("/V")] = TextStringObject("")
            f[NameObject("/DA")] = TextStringObject("/Helv 10 Tf 0 g")
        annots.append(writer._add_object(f))
        y -= 30
    page[NameObject("/Annots")] = annots
    acroform = DictionaryObject()
    acroform.update({
        NameObject("/Fields"): annots,
        NameObject("/NeedAppearances"): BooleanObject(True),
    })
    writer._root_object[NameObject("/AcroForm")] = writer._add_object(acroform)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        writer.write(fh)


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test-clinic"
    d.mkdir()
    (d / "mapping.yaml").write_text(MAPPING_YAML, encoding="utf-8")
    _make_template(
        d / "templates" / "intake.pdf",
        ["PatientName", "PatientFirst", "PatientLast", "DOB", "Phone", "Carrier_BCBS", "Subjective"],
    )
    return d


@pytest.fixture
def fb_store(tmp_path: Path, monkeypatch) -> CaptureStore:
    monkeypatch.setenv("AITHER_FORMBRIDGE_DIR", str(tmp_path / "fbstore"))
    store_mod.reset_store()
    yield store_mod.get_store()
    store_mod.reset_store()


# ── Store ────────────────────────────────────────────────────────────────────

class TestCaptureStore:
    def test_ingest_and_get(self, fb_store):
        n = fb_store.ingest_batch(
            "DEMO-1",
            [{"path": "patient.name", "value": "Pat Example"},
             {"path": "patient.dob", "value": "1980-04-12"}],
            display_name="Pat Example",
        )
        assert n == 2
        rec = fb_store.get_record("DEMO-1")
        assert rec["patient.name"] == "Pat Example"

    def test_upsert_overwrites(self, fb_store):
        fb_store.ingest_batch("DEMO-1", [{"path": "patient.name", "value": "Old"}])
        fb_store.ingest_batch("DEMO-1", [{"path": "patient.name", "value": "New"}])
        assert fb_store.get_record("DEMO-1")["patient.name"] == "New"

    def test_list_patients_no_values(self, fb_store):
        fb_store.ingest_batch("DEMO-1", [{"path": "patient.name", "value": "Secret Value"}])
        out = fb_store.list_patients()
        assert out[0]["patient_key"] == "DEMO-1"
        assert "Secret Value" not in str(out)

    def test_purge_one_and_all(self, fb_store):
        fb_store.ingest_batch("A", [{"path": "x", "value": "1"}])
        fb_store.ingest_batch("B", [{"path": "x", "value": "2"}])
        assert fb_store.purge("A") == 1
        assert fb_store.get_record("A") == {}
        assert fb_store.purge() == 1  # B remains, purge-all removes it
        assert fb_store.list_patients() == []

    def test_anchor_required(self, fb_store):
        with pytest.raises(ValueError):
            fb_store.ingest_batch("", [{"path": "x", "value": "1"}])

    def test_ttl_expiry(self, fb_store, monkeypatch):
        fb_store.ingest_batch("A", [{"path": "x", "value": "1"}])
        monkeypatch.setenv("AITHER_FORMBRIDGE_TTL_DAYS", "0.0000001")  # ~8ms
        import time

        time.sleep(0.05)
        assert fb_store.get_record("A") == {}


# ── Transforms ───────────────────────────────────────────────────────────────

class TestTransforms:
    def test_date(self):
        assert apply_transform("1980-04-12", "date:MM/DD/YYYY") == "04/12/1980"
        assert apply_transform("04/12/1980", "date:YYYY-MM-DD") == "1980-04-12"

    def test_date_bad_value(self):
        with pytest.raises(MappingError):
            apply_transform("not a date", "date:MM/DD/YYYY")

    def test_name_split(self):
        assert apply_transform("Pat Example", "name_split:first") == "Pat"
        assert apply_transform("Pat Example", "name_split:last") == "Example"
        assert apply_transform("Example, Pat", "name_split:first") == "Pat"
        assert apply_transform("Example, Pat", "name_split:last") == "Example"

    def test_checkbox(self):
        assert apply_transform("BCBS", "checkbox:equals(BCBS)") == "/Yes"
        assert apply_transform("Aetna", "checkbox:equals(BCBS)") == "/Off"
        assert apply_transform("Blue Cross BCBS plan", "checkbox:contains(bcbs)") == "/Yes"

    def test_phone(self):
        assert apply_transform("(555) 010-1234", "phone:###-###-####") == "555-010-1234"
        assert apply_transform("1 555 010 1234", "phone:###-###-####") == "555-010-1234"
        with pytest.raises(MappingError):
            apply_transform("12345", "phone:###-###-####")

    def test_truncate_and_concat(self):
        assert apply_transform("abcdef", "truncate(3)") == "abc"
        rec = {"a": "Hello", "b": "World"}
        assert apply_transform("", "concat(a,b)", rec) == "Hello World"

    def test_unknown_transform(self):
        with pytest.raises(MappingError):
            apply_transform("x", "frobnicate(1)")


# ── Mapping pack + resolution ────────────────────────────────────────────────

class TestMapping:
    def test_load_pack(self, pack_dir):
        pack = load_pack(pack_dir)
        assert pack.pack_id == "test-clinic"
        assert "intake" in pack.forms
        assert pack.anchor["selector"] == "#patient-name"

    def test_anchor_mandatory(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "mapping.yaml").write_text(
            "pack: bad\nsource: {origin: 'http://x'}\nforms: [{id: f, template: t.pdf, fill: []}]",
            encoding="utf-8",
        )
        with pytest.raises(MappingError, match="anchor"):
            load_pack(d)

    def test_resolve_full_record(self, pack_dir):
        pack = load_pack(pack_dir)
        res = resolve(RECORD, pack, "intake")
        assert res.values["PatientName"] == "Pat Example"
        assert res.values["PatientFirst"] == "Pat"
        assert res.values["DOB"] == "04/12/1980"
        assert res.values["Phone"] == "555-010-1234"
        assert res.values["Carrier_BCBS"] == "/Yes"
        assert res.values["Subjective"] == RECORD["soap.subjective"][:20]  # max_len
        assert res.llm_assist_fields == ["Subjective"]
        assert res.unresolved == []

    def test_required_hard_fail_lists_fields(self, pack_dir):
        pack = load_pack(pack_dir)
        partial = {k: v for k, v in RECORD.items() if k != "patient.dob"}
        with pytest.raises(MappingError, match="patient.dob"):
            resolve(partial, pack, "intake")

    def test_unresolved_flagged_never_guessed(self, pack_dir):
        pack = load_pack(pack_dir)
        rec = dict(RECORD)
        rec["patient.phone"] = ""
        res = resolve(rec, pack, "intake")
        assert "Phone" in res.unresolved
        assert "Phone" not in res.values

    def test_self_check(self, pack_dir):
        pack = load_pack(pack_dir)
        ok = run_self_check(pack, ["#subjective", "#objective"])
        assert ok["ok"] is True
        drift = run_self_check(pack, ["#subjective"])
        assert drift["ok"] is False
        assert drift["failures"][0]["missing"] == ["#objective"]

    def test_template_path_cannot_escape_pack(self, pack_dir):
        pack = load_pack(pack_dir)
        pack.forms["intake"].template = "../../escape.pdf"
        with pytest.raises(MappingError, match="escapes"):
            pack.template_path("intake")


# ── PDF round-trip ───────────────────────────────────────────────────────────

class TestPdfFill:
    def test_round_trip(self, pack_dir, tmp_path):
        pack = load_pack(pack_dir)
        res = resolve(RECORD, pack, "intake")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = fill_pdf(pack, res, out_dir=out_dir)

        reader = PdfReader(result.output_path)
        fields = reader.get_fields()
        assert fields["PatientName"].get("/V") == "Pat Example"
        assert fields["DOB"].get("/V") == "04/12/1980"
        assert str(fields["Carrier_BCBS"].get("/V")) == "/Yes"
        assert result.unknown_pdf_fields == []
        # Opaque output name — no patient identity in the path
        assert "Pat" not in Path(result.output_path).name

    def test_list_template_fields(self, pack_dir):
        pack = load_pack(pack_dir)
        names = list_template_fields(pack.template_path("intake"))
        assert "PatientName" in names and "Carrier_BCBS" in names


# ── Overlay mode (flat scanned forms, no AcroForm fields) ───────────────────

OVERLAY_MAPPING = """
pack: overlay-clinic
version: 1.0.0
source:
  origin: "http://localhost:8910"
  anchor: { selector: "#patient-name" }
  fields:
    - { capture: "#patient-name", as: "patient.name" }
forms:
  - id: flat-letter
    template: templates/flat.pdf
    mode: overlay
    required: ["patient.name"]
    fill:
      - { pdf_field: "Name",    from: "patient.name",      x: 150, y: 96,  size: 11 }
      - { pdf_field: "DOB",     from: "patient.dob",       x: 150, y: 130, transform: "date:MM/DD/YYYY" }
      - { pdf_field: "BCBSBox", from: "insurance.carrier", x: 72,  y: 200, transform: "checkbox:equals(BCBS)" }
"""


@pytest.fixture
def overlay_pack_dir(tmp_path: Path) -> Path:
    fpdf2 = pytest.importorskip("fpdf", reason="overlay tests need fpdf2 from the pdf extra")
    d = tmp_path / "overlay-clinic"
    d.mkdir()
    (d / "mapping.yaml").write_text(OVERLAY_MAPPING, encoding="utf-8")
    # flat one-page PDF, NO AcroForm — like a scanned office form
    pdf = fpdf2.FPDF(unit="pt", format=(612, 792))
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.set_xy(72, 96)
    pdf.cell(text="Patient Name: ____________")
    (d / "templates").mkdir()
    (d / "templates" / "flat.pdf").write_bytes(bytes(pdf.output()))
    return d


class TestOverlayFill:
    def test_overlay_round_trip(self, overlay_pack_dir, tmp_path):
        pack = load_pack(overlay_pack_dir)
        res = resolve(RECORD, pack, "flat-letter")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = fill_pdf(pack, res, out_dir=out_dir)
        text = PdfReader(result.output_path).pages[0].extract_text()
        assert "Pat Example" in text
        assert "04/12/1980" in text
        assert "X" in text  # BCBS checkbox drawn as X
        assert "Pat" not in Path(result.output_path).name  # opaque filename rule

    def test_overlay_requires_coords(self, tmp_path):
        d = tmp_path / "bad-overlay"
        d.mkdir()
        (d / "mapping.yaml").write_text(
            OVERLAY_MAPPING.replace("x: 150, y: 96,  size: 11", "size: 11"),
            encoding="utf-8",
        )
        with pytest.raises(MappingError, match="needs numeric x"):
            load_pack(d)

    def test_overlay_off_checkbox_draws_nothing(self, overlay_pack_dir, tmp_path):
        pack = load_pack(overlay_pack_dir)
        rec = dict(RECORD)
        rec["insurance.carrier"] = "Aetna"  # checkbox:equals(BCBS) -> /Off
        res = resolve(rec, pack, "flat-letter")
        out_dir = tmp_path / "out2"
        out_dir.mkdir()
        result = fill_pdf(pack, res, out_dir=out_dir)
        text = PdfReader(result.output_path).pages[0].extract_text()
        assert "Pat Example" in text
        # the only X would be the checkbox; with /Off nothing is drawn at 72,200
        assert "X" not in text


# ── HTTP routes ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(pack_dir, fb_store, monkeypatch, tmp_path):
    monkeypatch.setenv("AITHER_FORMBRIDGE_PACK", str(pack_dir))
    monkeypatch.setenv("AITHER_FORMBRIDGE_DEV", "1")
    monkeypatch.setenv("AITHER_FORMBRIDGE_OUTPUT", str(tmp_path / "out"))
    app = FastAPI()
    app.include_router(create_formbridge_router())
    return TestClient(app)


class TestRoutes:
    def test_health(self, client):
        r = client.get("/formbridge/health")
        assert r.status_code == 200
        body = r.json()
        assert body["pypdf"] is True
        assert "test-clinic" in body["packs"]

    def test_capture_then_fill(self, client):
        batch = {
            "patient_key": "DEMO-1",
            "display_name": "Pat Example",
            "fields": [{"path": k, "value": v} for k, v in RECORD.items()],
        }
        r = client.post("/formbridge/capture", json=batch)
        assert r.status_code == 200 and r.json()["written"] == len(RECORD)

        r = client.post("/formbridge/fill", json={"patient_key": "DEMO-1", "form_id": "intake"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert Path(body["output_path"]).is_file()
        assert "PatientName" in body["filled"]

    def test_fill_missing_required_is_422(self, client):
        client.post("/formbridge/capture", json={
            "patient_key": "DEMO-2",
            "fields": [{"path": "patient.name", "value": "Only Name"}],
        })
        r = client.post("/formbridge/fill", json={"patient_key": "DEMO-2", "form_id": "intake"})
        assert r.status_code == 422
        assert "patient.dob" in r.json()["detail"]

    def test_fill_unknown_patient_404(self, client):
        r = client.post("/formbridge/fill", json={"patient_key": "NOPE", "form_id": "intake"})
        assert r.status_code == 404

    def test_license_gate(self, client, monkeypatch):
        monkeypatch.delenv("AITHER_FORMBRIDGE_DEV")
        client.post("/formbridge/capture", json={
            "patient_key": "DEMO-3",
            "fields": [{"path": k, "value": v} for k, v in RECORD.items()],
        })
        r = client.post("/formbridge/fill", json={"patient_key": "DEMO-3", "form_id": "intake"})
        # conftest pins AITHER_TENANT_SLUG=aitherium (INTERNAL tier → all packs
        # available), so explicitly drop to community for this test.
        if r.status_code == 200:
            monkeypatch.delenv("AITHER_TENANT_SLUG", raising=False)
            from adk.licensing import reset_license_manager

            reset_license_manager()
            r = client.post("/formbridge/fill", json={"patient_key": "DEMO-3", "form_id": "intake"})
            reset_license_manager()
        assert r.status_code == 402

    def test_purge(self, client):
        client.post("/formbridge/capture", json={
            "patient_key": "DEMO-9", "fields": [{"path": "x", "value": "1"}],
        })
        r = client.post("/formbridge/purge", json={"patient_key": "DEMO-9"})
        assert r.json()["deleted"] == 1

    def test_capture_translates_selector_paths(self, client):
        """The browser sends SELECTOR paths; ingest must translate them via the
        pack's source.fields capture→as map so the mapper finds logical names.
        (Regression: this gap let unit tests pass while the real browser→engine
        path produced empty resolutions.)"""
        batch = {
            "patient_key": "DEMO-SEL",
            "source_origin": "http://localhost:8910",
            "fields": [
                {"path": "#patient-name", "value": "Pat Example"},   # mapped selector
                {"path": "patient.dob", "value": "1980-04-12"},      # already-logical passes through
                {"path": "#unmapped-widget", "value": "kept verbatim"},
            ],
        }
        r = client.post("/formbridge/capture", json=batch)
        assert r.status_code == 200 and r.json()["written"] == 3
        rec = client.get("/formbridge/patients/DEMO-SEL").json()["fields"]
        assert rec["patient.name"] == "Pat Example"        # translated
        assert rec["patient.dob"] == "1980-04-12"          # passthrough
        assert rec["#unmapped-widget"] == "kept verbatim"  # verbatim
        assert "#patient-name" not in rec


# ── Tool redaction (the 06 hard rule) ───────────────────────────────────────

class TestToolRedaction:
    def test_tool_results_never_contain_values(self, pack_dir, fb_store, monkeypatch, tmp_path):
        monkeypatch.setenv("AITHER_FORMBRIDGE_PACK", str(pack_dir))
        monkeypatch.setenv("AITHER_FORMBRIDGE_DEV", "1")
        monkeypatch.setenv("AITHER_FORMBRIDGE_OUTPUT", str(tmp_path / "out"))
        from adk.formbridge.tools import (
            formbridge_fill_form,
            formbridge_list_patients,
            formbridge_pack_health,
        )

        fb_store.ingest_batch(
            "DEMO-1",
            [{"path": k, "value": v} for k, v in RECORD.items()],
            display_name="Pat Example",
        )
        sentinel = RECORD["soap.subjective"]
        for result in (
            formbridge_list_patients(),
            formbridge_fill_form("DEMO-1", "intake"),
            formbridge_pack_health(),
        ):
            assert sentinel not in str(result)
            assert "(555) 010-1234" not in str(result)
