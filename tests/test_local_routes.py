"""Contract tests for adk.local_routes — the /api/local/* router behind the
`local` UI pack (Tasks queue, local image generation, awmail-backed mail).

Everything here is pinned WITHOUT network or fleet: the queue wrappers are
monkeypatched to return their documented JSON strings, the image discovery is
replaced with fixture lanes, and the mail half is tested on both branches
(awmail absent, awmail misconfigured, awmail working with a fake Mailer).
The point is the ROUTE contract — shapes the UI depends on — not the
underlying stores, which have their own tests.
"""

from __future__ import annotations

import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adk.local_routes import router as local_router


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(local_router)
    return TestClient(app)


# ── awrun queue (Tasks tab) ───────────────────────────────────────────────


def test_awrun_list_wraps_queue_list(client, monkeypatch):
    import adk.builtin_tools as bt

    monkeypatch.setattr(
        bt, "queue_list",
        lambda *a, **k: json.dumps([{"id": "r-1", "status": "queued", "task": "t"}]),
    )
    r = client.get("/api/local/awrun")
    assert r.status_code == 200
    assert r.json() == {"runs": [{"id": "r-1", "status": "queued", "task": "t"}]}


def test_awrun_list_passes_degradation_through(client, monkeypatch):
    import adk.builtin_tools as bt

    monkeypatch.setattr(
        bt, "queue_list",
        lambda *a, **k: json.dumps({"error": "awrun not available",
                                    "fix": "pip install awdk[queue]"}),
    )
    r = client.get("/api/local/awrun")
    assert r.status_code == 200
    assert r.json()["error"] == "awrun not available"


def test_awrun_submit_passes_task_agent(client, monkeypatch):
    import adk.builtin_tools as bt

    captured = {}

    def fake_submit(kind, priority=0, paths=None, task="", agent="", adk_args=None,
                    workflow="", ref="", inputs=None, service_name="", target="",
                    spec=None):
        captured.update(kind=kind, task=task, agent=agent, priority=priority)
        return json.dumps({"id": "r-x", "status": "queued"})

    monkeypatch.setattr(bt, "queue_submit", fake_submit)
    r = client.post("/api/local/awrun",
                    json={"kind": "agent", "task": "Do the thing", "agent": "aither",
                          "priority": 1})
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert captured == {"kind": "agent", "task": "Do the thing",
                        "agent": "aither", "priority": 1}


def test_awrun_submit_refuses_comet_deploy(client):
    """The money-spending kind must be refused on an HTTP route whose identity
    is the daemon's bearer, never a per-caller session (fail closed)."""
    r = client.post("/api/local/awrun",
                    json={"kind": "comet-deploy", "service_name": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["error"]
    assert "comet-deploy" in body["error"]


def test_awrun_status_and_cancel(client, monkeypatch):
    import adk.builtin_tools as bt

    monkeypatch.setattr(bt, "queue_status",
                        lambda run_id: json.dumps({"id": run_id, "status": "running"}))
    monkeypatch.setattr(bt, "queue_cancel",
                        lambda run_id: json.dumps({"id": run_id, "status": "cancelled"}))
    assert client.get("/api/local/awrun/r-1").json()["status"] == "running"
    assert client.post("/api/local/awrun/r-1/cancel").json()["status"] == "cancelled"


# ── local image generation (Visual tab) ───────────────────────────────────


def _lane(lane_id="test", up=True):
    from adk.images import Lane

    return Lane(id=lane_id, label="Test Lane", port=0, kind="openai",
                up=up, status=200 if up else 0, note="fixture")


def test_image_backends_shape(client, monkeypatch):
    import adk.images as img

    async def fake_discover():
        return [_lane("comfyui", True), _lane("sana", False)]

    monkeypatch.setattr(img, "discover", fake_discover)
    r = client.get("/api/local/images/backends")
    assert r.status_code == 200
    body = r.json()
    assert body["usable"] == ["comfyui"]
    assert [b["id"] for b in body["backends"]] == ["comfyui", "sana"]
    assert body["backends"][0]["up"] is True


def test_image_generate_bad_size_400(client):
    r = client.post("/api/local/images/generations",
                    json={"prompt": "x", "size": "not-a-size"})
    assert r.status_code == 400
    assert "size must look like" in r.json()["detail"]


def test_image_generate_no_backend_503(client, monkeypatch):
    import adk.images as img

    async def fake_discover():
        return []

    async def fake_generate(req):
        raise img.ImageError("No local image backend is able to generate. Tried: 8188, 8202, 7860")

    monkeypatch.setattr(img, "discover", fake_discover)
    monkeypatch.setattr(img, "generate", fake_generate)
    r = client.post("/api/local/images/generations", json={"prompt": "x"})
    assert r.status_code == 503
    assert "No local image backend" in r.json()["detail"]


# ── mail (Mail tab) ───────────────────────────────────────────────────────


def test_mail_status_when_awmail_absent(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "awmail", None)
    monkeypatch.setitem(sys.modules, "awmail.client", None)
    r = client.get("/api/local/mail/status")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "pip install awmail" in body["fix"]


def test_mail_status_when_unconfigured(client, monkeypatch):
    awmail = pytest.importorskip("awmail")

    def raise_missing():
        raise RuntimeError("AWMAIL_FROM, AWMAIL_PASSWORD, AWMAIL_ALLOW are required")

    monkeypatch.setattr(awmail.client.Mailer, "from_env", staticmethod(raise_missing))
    monkeypatch.delenv("AWMAIL_FROM", raising=False)
    monkeypatch.delenv("AWMAIL_PASSWORD", raising=False)
    monkeypatch.delenv("AWMAIL_ALLOW", raising=False)
    r = client.get("/api/local/mail/status")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "AWMAIL_FROM" in body["fix"]


def test_mail_send_no_recipient(client, monkeypatch):
    """With mail CONFIGURED, a missing recipient is the domain answer. With
    mail UNCONFIGURED the config message wins instead (fix the config first —
    that is the more useful answer and the order the route pins)."""
    awmail = pytest.importorskip("awmail")

    class FakeMailer:
        def send(self, to, subject="", body=""):
            raise AssertionError("send must not be reached without a recipient")

    monkeypatch.setattr(awmail.client.Mailer, "from_env", staticmethod(lambda: FakeMailer()))
    r = client.post("/api/local/mail/send", json={"to": "", "subject": "s", "body": "b"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "recipient" in body["message"]


def test_mail_send_unconfigured_beats_recipient_check(client, monkeypatch):
    """When awmail is not configured, that answer must win over the recipient
    check — an unconfigured mailer would otherwise read as a form bug."""
    awmail = pytest.importorskip("awmail")

    def raise_missing():
        raise RuntimeError("AWMAIL_FROM, AWMAIL_PASSWORD, AWMAIL_ALLOW are required")

    monkeypatch.setattr(awmail.client.Mailer, "from_env", staticmethod(raise_missing))
    r = client.post("/api/local/mail/send", json={"to": "", "subject": "s", "body": "b"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "AWMAIL_FROM" in body["fix"]


def test_mail_send_maps_accepted_and_refused(client, monkeypatch):
    awmail = pytest.importorskip("awmail")
    from awmail.message import ACCEPTED, REFUSED, SendResult

    class FakeMailer:
        last = None

        def send(self, to, subject="", body=""):
            FakeMailer.last = (to, subject, body)
            return SendResult(status=ACCEPTED, detail="", accepted=[to])

    monkeypatch.setattr(awmail.client.Mailer, "from_env", staticmethod(lambda: FakeMailer()))
    r = client.post("/api/local/mail/send",
                    json={"to": "you@example.com", "subject": "hi", "body": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["accepted"] is True
    assert FakeMailer.last == ("you@example.com", "hi", "hello")

    class RefusingMailer:
        def send(self, to, subject="", body=""):
            return SendResult(status=REFUSED, detail="allowlist blocks you@example.com",
                              rejected={"you@example.com": "not allowed"})

    monkeypatch.setattr(awmail.client.Mailer, "from_env", staticmethod(lambda: RefusingMailer()))
    r = client.post("/api/local/mail/send",
                    json={"to": "you@example.com", "subject": "hi", "body": "hello"})
    body = r.json()
    assert body["ok"] is False
    assert "not allowed" in body["message"]
