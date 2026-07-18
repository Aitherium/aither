"""Encounter-scoped field isolation tests.

When a doctor navigates between encounters in ChiroTouch, the store should
never mix clinical data from different encounters. This test suite verifies:
1. Encounter switching clears old encounter-scoped fields (diagnoses, charges)
2. Patient-scoped fields (cases, demographics) persist across encounters
3. The encounter sentinel is tracked internally but not exposed to callers
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf", reason="formbridge tests need the pdf extra")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from adk.formbridge import store as store_mod
from adk.formbridge.api_capture import (
    encounter_id_from_url,
    encounter_scoped_fields,
)
from adk.formbridge.mapper import load_pack
from adk.formbridge.routes import create_formbridge_router
from adk.formbridge.store import CaptureStore


@pytest.fixture
def fb_store(tmp_path: Path, monkeypatch) -> CaptureStore:
    monkeypatch.setenv("AITHER_FORMBRIDGE_DIR", str(tmp_path / "fbstore"))
    store_mod.reset_store()
    yield store_mod.get_store()
    store_mod.reset_store()


@pytest.fixture(autouse=True)
def _reset_vault_singleton():
    """Each test gets a fresh vault + discovery singleton."""
    from adk.formbridge import vault as vault_mod

    vault_mod.reset_vault()
    yield
    vault_mod.reset_vault()


@pytest.fixture
def chirotouch_pack():
    """Load the real chirotouch-live pack for integration testing."""
    pack_dir = Path(__file__).resolve().parents[2] / "AitherOS/Library/packs/form-bridge/chirotouch-live"
    pack = load_pack(str(pack_dir))
    # Set office config for C-9 fill tests
    pack.office.update({
        "office.provider.name_address": "Demo Clinic, 1 Main St, Columbus OH",
        "office.provider.bwc_number": "BWC123456",
        "office.from": "Demo Clinic",
        "office.phone": "5135550100",
        "office.fax": "5135550101",
    })
    return pack


@pytest.fixture
def client(fb_store, monkeypatch, tmp_path):
    """TestClient with chirotouch-live pack and formbridge router."""
    pack_dir = Path(__file__).resolve().parents[2] / "AitherOS/Library/packs/form-bridge/chirotouch-live"
    monkeypatch.setenv("AITHER_FORMBRIDGE_PACK", str(pack_dir))
    monkeypatch.setenv("AITHER_FORMBRIDGE_DEV", "1")
    monkeypatch.setenv("AITHER_FORMBRIDGE_OUTPUT", str(tmp_path / "out"))
    app = FastAPI()
    app.include_router(create_formbridge_router())
    return TestClient(app)


# ── Encounter ID extraction ───────────────────────────────────────────────


class TestEncounterIdExtraction:
    def test_encounter_id_from_encounters_endpoint(self):
        url = "https://x/core-api/api/v1/tenants/a/encounters/999/diagnoses"
        assert encounter_id_from_url(url) == "999"

    def test_encounter_id_with_query_string(self):
        url = "https://x/core-api/api/v1/tenants/a/encounters/888?filter=active"
        assert encounter_id_from_url(url) == "888"

    def test_no_encounter_id_in_patient_endpoint(self):
        url = "https://x/billing-api/api/v1/tenants/a/patients/777/cases"
        assert encounter_id_from_url(url) == ""

    def test_no_encounter_id_in_empty_url(self):
        assert encounter_id_from_url("") == ""


# ── Encounter-scoped field identification ─────────────────────────────────


class TestEncounterScopedFields:
    def test_identifies_encounter_scoped_fields(self, chirotouch_pack):
        scoped = encounter_scoped_fields(chirotouch_pack)
        # The chirotouch pack has encounter-scoped endpoints for diagnoses, charges, and SOAP
        assert "diagnosis.code" in scoped
        assert "diagnosis.description" in scoped
        assert "service.cpt" in scoped
        assert "service.description" in scoped
        assert "soap.narrative" in scoped

    def test_excludes_patient_scoped_fields(self, chirotouch_pack):
        scoped = encounter_scoped_fields(chirotouch_pack)
        # These come from patient-scoped "cases" endpoint (no /encounters/id/)
        assert "claim.number" not in scoped
        assert "injury.date" not in scoped
        assert "provider.first" not in scoped

    def test_excludes_encounter_metadata_but_includes_encounter_data(self, chirotouch_pack):
        scoped = encounter_scoped_fields(chirotouch_pack)
        # The /encounters/* endpoint carries patient demographics (iw.first, iw.last)
        # which are encounter-metadata, not encounter-specific clinical data.
        # For safety, we include them in scoped since they're encounter-specific
        # (different from patient-scoped case data).
        assert "iw.first" in scoped
        assert "iw.last" in scoped
        assert "service.date" in scoped


# ── Store encounter field deletion ────────────────────────────────────────


class TestStoreDeleteFields:
    def test_delete_fields_removes_specified_paths(self, fb_store):
        # Ingest some fields
        fb_store.ingest_batch(
            "P1",
            [
                {"path": "diagnosis.code", "value": "M54.5"},
                {"path": "claim.number", "value": "24-001"},
                {"path": "service.cpt", "value": "98941"},
            ],
        )
        # Verify all 3 are present
        rec = fb_store.get_record("P1")
        assert len(rec) == 3

        # Delete the encounter-scoped ones
        deleted = fb_store.delete_fields("P1", ["diagnosis.code", "service.cpt"])
        assert deleted == 2

        # Verify claim.number persists
        rec = fb_store.get_record("P1")
        assert rec == {"claim.number": "24-001"}

    def test_delete_fields_with_empty_list(self, fb_store):
        fb_store.ingest_batch("P2", [{"path": "x", "value": "1"}])
        deleted = fb_store.delete_fields("P2", [])
        assert deleted == 0
        assert fb_store.get_record("P2") == {"x": "1"}

    def test_delete_fields_for_nonexistent_patient(self, fb_store):
        deleted = fb_store.delete_fields("NONE", ["x"])
        assert deleted == 0

    def test_delete_fields_nonexistent_path(self, fb_store):
        fb_store.ingest_batch("P3", [{"path": "x", "value": "1"}])
        deleted = fb_store.delete_fields("P3", ["y"])
        assert deleted == 0
        assert fb_store.get_record("P3") == {"x": "1"}


# ── Integration: API capture with encounter switching ─────────────────────


class TestApiCaptureEncounterSwitching:
    """The core scenario: doctor navigates E1 → E2, store never mixes data."""

    def test_capture_two_encounters_switches_cleanly(self, client, fb_store):
        """Capture diagnoses for E1, then E2 for the same patient.
        The record should only hold E2's diagnoses, never a mix."""
        patient_id = "100"  # Must match URL segment after /patients/

        # ── E1: First encounter, diagnose M54.5 ──
        r1 = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/core-api/api/v1/tenants/acme/patients/{patient_id}/encounters/111/diagnoses",
                "body": {"data": [{"code": "M54.5", "shortDescription": "Low back pain"}]},
                "source_origin": "https://practice.chirotouch.com",
                "display_name": "Pat Example",
            },
        )
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["encounter_id"] == "111"
        assert "diagnosis.code" in body1["fields"]

        # Check what's stored: E1's diagnosis
        rec1 = fb_store.get_record(patient_id)
        assert rec1.get("diagnosis.code") == "M54.5"  # single value stored as string, not array
        # Sentinel is stored but not returned by get_record

        # ── E2: Same patient, different encounter, diagnose M99.01 ──
        r2 = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/core-api/api/v1/tenants/acme/patients/{patient_id}/encounters/222/diagnoses",
                "body": {"data": [{"code": "M99.01", "shortDescription": "Subluxation"}]},
                "source_origin": "https://practice.chirotouch.com",
                "display_name": "Pat Example",
            },
        )
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["encounter_id"] == "222"

        # Check what's stored: ONLY E2's diagnosis, E1's is gone
        rec2 = fb_store.get_record(patient_id)
        assert rec2.get("diagnosis.code") == "M99.01"  # E2 only, E1 cleared
        # Sentinel is updated but not returned by get_record

    def test_patient_scoped_fields_survive_encounter_switch(self, client, fb_store):
        """Patient-scoped data (cases, demographics) must NOT be cleared
        when an encounter switches."""
        patient_id = "200"

        # ── E1: Capture encounter demographic ──
        r1 = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/encounters/300",
                "body": {
                    "data": [
                        {
                            "patient": {"firstName": "Pat", "lastName": "Example", "dateOfBirth": "1980-04-12"},
                            "serviceDate": "2024-05-01",
                            "currentIllnessDate": "2024-05-01",
                        }
                    ]
                },
                "source_origin": "https://practice.chirotouch.com",
                "display_name": "Pat Example",
            },
        )
        assert r1.status_code == 200

        # ── E1: Also capture the case (claim) info — patient-scoped endpoint ──
        r1_case = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/cases",
                "body": {
                    "data": [
                        {
                            "claimNumber": "24-998877",
                            "accidentDate": "2024-03-15",
                            "orderingProviderFirstName": "Jane",
                            "orderingProviderLastName": "Dee",
                            "orderingProviderCredentials": "DC",
                        }
                    ]
                },
                "source_origin": "https://practice.chirotouch.com",
                "display_name": "Pat Example",
            },
        )
        assert r1_case.status_code == 200

        rec1 = fb_store.get_record(patient_id)
        assert rec1.get("claim.number") == "24-998877"
        assert rec1.get("iw.first") == "Pat"

        # ── E2: Same patient, new encounter ──
        r2 = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/encounters/400",
                "body": {
                    "data": [
                        {
                            "patient": {"firstName": "Pat", "lastName": "Example", "dateOfBirth": "1980-04-12"},
                            "serviceDate": "2024-06-01",  # different date
                            "currentIllnessDate": "2024-06-01",
                        }
                    ]
                },
                "source_origin": "https://practice.chirotouch.com",
                "display_name": "Pat Example",
            },
        )
        assert r2.status_code == 200

        # The claim data must STILL be present (patient-scoped, not cleared)
        rec2 = fb_store.get_record(patient_id)
        assert rec2.get("claim.number") == "24-998877"  # PRESERVED
        assert rec2.get("iw.first") == "Pat"  # PRESERVED
        assert rec2.get("service.date") == "2024-06-01"  # UPDATED (new encounter)
        # Sentinel is updated internally but not exposed via get_record

    def test_non_encounter_responses_ignore_sentinel(self, client, fb_store):
        """A patient-scoped endpoint (no /encounters/) should not trigger
        encounter-scoped field deletion. The sentinel is only checked for
        encounter-specific responses."""
        patient_id = "PT-300"

        # Ingest some case data (no encounter context)
        r = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/cases",
                "body": {
                    "data": [
                        {
                            "claimNumber": "24-001",
                            "accidentDate": "2024-03-15",
                        }
                    ]
                },
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("encounter_id") is None  # No encounter in this URL

        # Ingest again — should not trigger clearing (no _encounter sentinel)
        r2 = client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/cases",
                "body": {
                    "data": [
                        {
                            "claimNumber": "24-002",
                            "accidentDate": "2024-04-15",
                        }
                    ]
                },
            },
        )
        assert r2.status_code == 200

        rec = fb_store.get_record(patient_id)
        # The most recent upsert wins (normal behavior, not encounter-driven)
        assert rec.get("claim.number") == "24-002"

    def test_internal_sentinel_never_in_get_record(self, fb_store):
        """The _encounter sentinel is an internal implementation detail.
        get_record() must never return it to consumers."""
        fb_store.ingest_batch(
            "P_SENT",
            [
                {"path": "_encounter", "value": "999"},
                {"path": "claim.number", "value": "24-001"},
            ],
        )
        rec = fb_store.get_record("P_SENT")
        # Only the real field is returned
        assert rec == {"claim.number": "24-001"}
        assert "_encounter" not in rec

    def test_sentinel_persists_internally_across_reads(self, fb_store):
        """While not exposed to callers, the sentinel persists in the DB
        so the store can detect encounter switches."""
        fb_store.ingest_batch(
            "P_PERSIST",
            [{"path": "_encounter", "value": "E1"}],
        )
        # Read 1 — get_record filters it
        rec1 = fb_store.get_record("P_PERSIST")
        assert "_encounter" not in rec1

        # But the store can still detect it for the next encounter switch
        rec1_raw = fb_store._connect().execute(
            "SELECT field_path, value FROM captures WHERE patient_key = ? AND field_path = '_encounter'",
            ("P_PERSIST",),
        ).fetchone()
        assert rec1_raw["value"] == "E1"


# ── Full integration: C-9 fill avoids cross-encounter data ────────────────


class TestC9FillEncounterIsolation:
    """Verify that a filled C-9 never carries diagnoses from the wrong encounter."""

    def test_c9_fill_reflects_correct_encounter_diagnoses(self, client, chirotouch_pack, tmp_path):
        """Ingest two encounters' diagnoses in sequence. Only the latest should fill."""
        patient_id = "C9"

        # ── E1: Capture encounter + diagnoses + case ──
        client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/encounters/500",
                "body": {
                    "data": [{
                        "patient": {"firstName": "Pat", "lastName": "Example", "dateOfBirth": "1980-04-12"},
                        "serviceDate": "2024-05-01",
                    }]
                },
            },
        )
        client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/core-api/api/v1/tenants/acme/patients/{patient_id}/encounters/500/diagnoses",
                "body": {"data": [{"code": "M54.5", "shortDescription": "Low back pain"}]},
            },
        )
        client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/cases",
                "body": {
                    "data": [{
                        "claimNumber": "24-999",
                        "accidentDate": "2024-03-15",
                        "orderingProviderFirstName": "Jane",
                        "orderingProviderLastName": "Doe",
                        "orderingProviderCredentials": "DC",
                    }]
                },
            },
        )

        # ── E2: Switch encounter, new diagnoses ──
        client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/billing-api/api/v1/tenants/acme/patients/{patient_id}/encounters/600",
                "body": {
                    "data": [{
                        "patient": {"firstName": "Pat", "lastName": "Example", "dateOfBirth": "1980-04-12"},
                        "serviceDate": "2024-06-01",
                    }]
                },
            },
        )
        client.post(
            "/formbridge/capture-api",
            json={
                "url": f"https://practice.chirotouch.com/core-api/api/v1/tenants/acme/patients/{patient_id}/encounters/600/diagnoses",
                "body": {"data": [{"code": "M99.01", "shortDescription": "Subluxation"}]},
            },
        )

        # Fill the C-9 for E2
        from adk.formbridge.mapper import resolve
        from adk.formbridge.pdf_fill import fill_pdf

        rec = client.get(f"/formbridge/patients/{patient_id}").json()["fields"]
        resolution = resolve(rec, chirotouch_pack, "C-9")

        # The diagnosis should reflect E2 (M99.01), never E1 (M54.5)
        assert "Subluxation" in resolution.values.get("Treating diagnosis for this request to include body part/levels", "")
        assert "Low back pain" not in str(resolution.values)
