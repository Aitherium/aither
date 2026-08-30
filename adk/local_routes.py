"""Local-first routes for the Aitherium Local AI UI pack.

The `local` UI pack (adk/webui/packs/local/) is served from THIS daemon, so
everything privileged must live behind the same auth plane. These routes reuse
the existing awrun queue wrappers and the adk.images discovery/generation
pipeline -- the same modules the :9001 harness daemon mounts -- so the UI has
one origin and one bearer to talk to, and "the queue" in the Tasks tab is the
real awrun queue, not a mock.

Why this router exists instead of pointing the UI at the harness daemon:
the harness daemon (:9001) authenticates ONE shared bearer and does not serve
the UI packs. The Local AI page is served by aither-serve; calling a second
daemon cross-origin would need a second token and CORS. One origin, one auth,
one queue.

Degradation is inherited from the wrappers on purpose: queue_submit returns
{"error": "awrun not available", "fix": "pip install awdk[queue]"} when the
store is missing, and adk.images answers "no local backend is able to
generate. Tried: ..." -- both are written to be shown to a person and are
passed through verbatim rather than re-wrapped.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/local", tags=["local"])


# ── awrun queue (Tasks tab) ────────────────────────────────────────────────
# Same wrappers the harness daemon uses (adk.harnesses.daemon /awrun/*): they
# carry the awrun[queue]-missing degradation and the trust-plane gate for
# comet-deploy. Responses are {"error": ...} 200-bodies for domain answers.


@router.get("/awrun")
def local_awrun_list(kind: str = "", include_closed: bool = False) -> dict[str, Any]:
    from adk.builtin_tools import queue_list

    parsed = json.loads(queue_list(kind=kind, include_closed=include_closed))
    # The queue wrappers answer {"error": ..., "fix": ...} in a 200 body when
    # the store is missing. Surface that at the TOP level, not nested under
    # "runs" — a UI must show "queue unavailable", never an empty queue that
    # reads as "nothing queued" (the silent-failure shape this router exists
    # to avoid).
    if isinstance(parsed, dict) and parsed.get("error"):
        return parsed
    return {"runs": parsed}


@router.post("/awrun")
def local_awrun_submit(body: dict[str, Any]) -> dict[str, Any]:
    from adk.builtin_tools import queue_submit

    # kind="comet-deploy" spends real money and is trust-plane gated against
    # the caller's OWN session bearer -- see the harness daemon's identical
    # refusal for the full reasoning. This route authenticates the daemon's
    # bearer, not a per-caller session, so that kind is refused here too.
    kind = str(body.get("kind") or "agent")
    if kind == "comet-deploy":
        return {
            "error": "comet-deploy is not available through this route — "
                     "use `awrun submit --kind comet-deploy` on the machine "
                     "holding the session.",
        }
    return json.loads(queue_submit(
        kind,
        priority=int(body.get("priority") or 0),
        paths=body.get("paths"),
        task=str(body.get("task") or ""),
        agent=str(body.get("agent") or ""),
        adk_args=body.get("adk_args"),
        workflow=str(body.get("workflow") or ""),
        ref=str(body.get("ref") or ""),
        inputs=body.get("inputs"),
        service_name=str(body.get("service_name") or ""),
        target=str(body.get("target") or ""),
        spec=body.get("spec"),
    ))


@router.get("/awrun/{run_id}")
def local_awrun_status(run_id: str) -> dict[str, Any]:
    from adk.builtin_tools import queue_status

    return json.loads(queue_status(run_id))


@router.post("/awrun/{run_id}/cancel")
def local_awrun_cancel(run_id: str) -> dict[str, Any]:
    from adk.builtin_tools import queue_cancel

    return json.loads(queue_cancel(run_id))


# ── local image generation (Visual tab) ───────────────────────────────────
# Same discovery/generation pipeline as the harness daemon's /v1/images/*.
# Discovers ComfyUI / Sana / SD.Next already on loopback; starts nothing.


@router.get("/images/backends")
async def local_image_backends() -> dict[str, Any]:
    from adk import images as _img

    lanes = await _img.discover()
    return {
        "backends": [ln.as_dict() for ln in lanes],
        "usable": [ln.id for ln in lanes if ln.up],
    }


@router.post("/images/generations")
async def local_image_generate(body: dict[str, Any]) -> dict[str, Any]:
    from adk import images as _img

    size = str(body.get("size") or "768x768")
    try:
        w_s, h_s = size.lower().split("x", 1)
        width, height = int(w_s), int(h_s)
    except (ValueError, AttributeError):
        raise HTTPException(400, f"size must look like 768x768, got {size!r}")

    req = _img.ImageRequest(
        prompt=str(body.get("prompt") or ""),
        negative=str(body.get("negative_prompt") or ""),
        width=width, height=height,
        steps=int(body.get("steps") or 20),
        cfg=float(body.get("cfg") or 6.0),
        seed=body.get("seed"),
        model=str(body.get("model") or ""),
        backend=str(body.get("backend") or ""),
    )
    try:
        out = await _img.generate(req)
    except _img.ImageError as e:
        # 503, not 500: every ImageError means "no local backend can do this
        # right now", a service-availability answer whose message is written
        # to be shown to a person.
        raise HTTPException(503, str(e))

    return {
        "created": 0,
        "data": [{"b64_json": b} for b in out["images_b64"]],
        "backend": out["backend"],
        "model": out["model"],
    }


# ── mail (Mail tab) ───────────────────────────────────────────────────────
# Backed by the awmail package (bridge or smtp transport). awmail is an
# OPTIONAL dependency: when it is not installed, or AWMAIL_* is not
# configured, status answers cleanly that mail is unavailable and send
# returns a 200-body naming the fix, so the UI can show a person-readable
# answer instead of a 500.
#
# awmail's own rails are inherited untouched: Mailer.from_env() refuses to
# construct without AWMAIL_FROM / AWMAIL_PASSWORD / AWMAIL_ALLOW (an unset
# allowlist must never quietly permit the whole internet), and SendResult.ok
# is True ONLY for ACCEPTED -- UNKNOWN ("handed to the relay, could not tell")
# deliberately reads as not-ok. Both verdicts are passed through verbatim.


def _mail_status_dict() -> dict[str, Any]:
    """awmail availability: (mailer, {status-dict}). mailer is None when the
    package is missing, an Exception when it is misconfigured."""
    try:
        from awmail.client import Mailer
    except ImportError:
        return None, {"available": False, "transports": [],
                      "message": "Mail unavailable — awmail is not installed.",
                      "fix": "pip install awmail"}
    try:
        mailer = Mailer.from_env()
    except Exception as exc:  # noqa: BLE001 - from_env raises for missing vars
        return None, {"available": False, "transports": [],
                      "message": f"Mail unavailable — {exc}", "fix": "set AWMAIL_* (AWMAIL_FROM, AWMAIL_PASSWORD, AWMAIL_ALLOW)"}
    import os
    transport = (os.environ.get("AWMAIL_TRANSPORT") or "bridge").strip().lower()
    return mailer, {"available": True, "transports": [transport],
                    "message": "", "transport": transport}


@router.get("/mail/status")
def local_mail_status() -> dict[str, Any]:
    _, status = _mail_status_dict()
    return status


@router.post("/mail/send")
def local_mail_send(body: dict[str, Any]) -> dict[str, Any]:
    mailer, status = _mail_status_dict()
    if mailer is None:
        return {"ok": False, "message": status["message"], "fix": status.get("fix", "")}

    to = str(body.get("to") or "").strip()
    subject = str(body.get("subject") or "").strip()
    text = str(body.get("body") or "").strip()
    if not to:
        return {"ok": False, "message": "No recipient address.", "fix": "enter a To address"}

    try:
        result = mailer.send(to=to, subject=subject or "(no subject)", body=text)
    except Exception as exc:  # noqa: BLE001 - a transport refusal is a domain answer
        return {"ok": False, "message": f"Send refused: {exc}",
                "fix": "check the allowlist and transport config"}

    # SendResult: status is ACCEPTED | REFUSED | UNKNOWN. ok == ACCEPTED only.
    from awmail.message import ACCEPTED

    s = getattr(result, "status", "")
    detail = getattr(result, "detail", "")
    accepted = list(getattr(result, "accepted", []) or [])
    rejected = getattr(result, "rejected", {}) or {}
    ok = s == ACCEPTED
    if ok:
        message = "Sent." + (f" ({', '.join(accepted)})" if accepted else "")
    elif rejected:
        message = "Not sent — refused: " + "; ".join(
            f"{addr}: {why}" for addr, why in sorted(rejected.items()))
    else:
        message = "Not sent — " + (detail or s or "unknown outcome")
    return {"ok": ok, "message": message,
            "accepted": bool(accepted), "delivered": ok,
            "transport": status.get("transport", ""),
            "fix": "" if ok else "check awmail config"}
