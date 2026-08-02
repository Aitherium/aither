"""The AitherShell harness daemon — one API for every front-end.

This is what makes "AitherShell on my desktop" and "AitherShell on
aitherium.com" the same thing rather than two lookalikes: both are clients of
this daemon. The desktop CLI talks to it over loopback; the browser talks to it
through AitherTunnel. A session started in one is visible and attachable in the
other, because there is exactly one manager behind this API.

Security posture (fail-closed, per .claude/rules/security-review-patterns.md)
----------------------------------------------------------------------------
This daemon spawns coding agents with filesystem access. It is therefore
treated as a privileged surface:

- It REFUSES to start without a bearer token. There is no "no auth in dev"
  mode — that is how an unauthenticated port ends up exposed through a tunnel.
- Every route except ``/health`` denies on missing (401) or wrong (403)
  credentials, and the comparison is constant-time.
- CORS is an explicit ALLOWLIST. ``*`` is rejected outright when credentials
  are involved, and the allowlist is printed at startup so an over-broad one is
  visible rather than discovered later.
- ``cwd`` is validated against an allowlist of roots when
  ``AITHER_HARNESS_ALLOWED_ROOTS`` is set, so a tunnel-exposed daemon cannot be
  pointed at ``C:\\`` by a caller. Unset means "this host is trusted", which is
  the desktop default and is stated in ``/health`` rather than assumed.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, Optional

from adk.harnesses.manager import ManagerError, SessionManager, default_manager
from adk.harnesses.models import ProfileError, list_profiles
from adk.harnesses.session import SessionConfig

DEFAULT_HOST = os.environ.get("AITHER_HARNESS_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("AITHER_HARNESS_PORT", "8362"))
TOKEN_PATH = Path.home() / ".aither" / "harness_token"

#: Browser origins allowed to call this daemon. AitherShell-in-the-browser is
#: served from these; anything else is refused.
DEFAULT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://aitherium.com",
    "https://www.aitherium.com",
    "https://portal.aitherium.com",
    "https://veil.aitherium.com",
    "https://tunnel.aitherium.com",
)


def allowed_origins() -> list[str]:
    raw = os.environ.get("AITHER_HARNESS_ALLOWED_ORIGINS", "")
    if not raw.strip():
        return list(DEFAULT_ORIGINS)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError(
            "AITHER_HARNESS_ALLOWED_ORIGINS='*' is refused: this daemon spawns "
            "coding agents with filesystem access and uses bearer credentials. "
            "List the origins explicitly."
        )
    return origins


def allowed_roots() -> list[Path]:
    raw = os.environ.get("AITHER_HARNESS_ALLOWED_ROOTS", "")
    return [Path(p.strip()).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]


def resolve_token(explicit: str = "") -> str:
    """Resolve the daemon bearer token, minting and persisting one if needed.

    Order: explicit > env > on-disk. A generated token is written with
    owner-only permissions so it is not world-readable on a shared box.
    """
    if explicit:
        return explicit.strip()
    env_token = os.environ.get("AITHER_HARNESS_TOKEN", "").strip()
    if env_token:
        return env_token
    if TOKEN_PATH.exists():
        existing = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    token = secrets.token_urlsafe(32)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token, encoding="utf-8")
    try:
        os.chmod(TOKEN_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:
        sys.stderr.write(f"[harness] could not restrict {TOKEN_PATH}: {exc}\n")
    return token


def validate_cwd(cwd: str) -> str:
    """Confirm ``cwd`` is inside an allowed root. Empty allowlist = host trusted."""
    if not cwd:
        return cwd
    roots = allowed_roots()
    if not roots:
        return cwd
    target = Path(cwd).expanduser().resolve()
    for root in roots:
        try:
            target.relative_to(root)
            return str(target)
        except ValueError:
            continue
    raise ManagerError(
        f"cwd {target} is outside the allowed roots for this daemon "
        f"({os.pathsep.join(str(r) for r in roots)})"
    )


try:  # pragma: no cover - import guard
    from pydantic import BaseModel, Field
except ImportError:  # The daemon needs fastapi+pydantic; the rest of the
    # package must stay importable without them so the CLI can still run
    # sessions in-process on a box with no web stack.
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


if BaseModel is not None:

    class CreateSession(BaseModel):  # type: ignore[misc]
        """Request body for POST /sessions.

        Defined at MODULE level on purpose. This file uses
        ``from __future__ import annotations``, so every endpoint annotation is
        a string that FastAPI resolves against the endpoint's MODULE globals —
        a model defined inside ``create_app`` is invisible there, and FastAPI
        silently degrades it to a query parameter (a 422 that reads like a
        client bug rather than a wiring bug).
        """

        harness: str = "claude"
        cwd: str = ""
        model_profile: str = ""
        model: str = ""
        permission_mode: str = ""
        resume_session_id: str = ""
        system_prompt_append: str = ""
        add_dirs: list[str] = Field(default_factory=list)
        allowed_tools: list[str] = Field(default_factory=list)
        mcp_config: str = ""
        target: str = ""
        extra_args: list[str] = Field(default_factory=list)
        title: str = ""
        owner: str = ""

    class SendInput(BaseModel):  # type: ignore[misc]
        text: str

    class ResizeInput(BaseModel):  # type: ignore[misc]
        rows: int = 30
        cols: int = 100

    class ProvisionSandbox(BaseModel):  # type: ignore[misc]
        workspace_slug: str
        repos: list[str] = Field(default_factory=list)


def create_app(manager: Optional[SessionManager] = None, token: str = ""):
    """Build the FastAPI app. Raises if no token can be resolved (fail-closed)."""
    if BaseModel is None:
        raise RuntimeError("fastapi and pydantic are required: pip install fastapi uvicorn")
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    mgr = manager or default_manager()
    bearer = resolve_token(token)
    if not bearer:
        raise RuntimeError("harness daemon refuses to start without a bearer token")
    origins = allowed_origins()

    app = FastAPI(title="AitherShell Harness Daemon", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def auth(authorization: str = Header(default="")) -> None:
        if not authorization:
            raise HTTPException(status_code=401, detail="missing bearer token")
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() != "bearer" or not value:
            raise HTTPException(status_code=401, detail="malformed Authorization header")
        if not hmac.compare_digest(value.strip(), bearer):
            raise HTTPException(status_code=403, detail="invalid token")

    class CreateSession(BaseModel):
        harness: str = "claude"
        cwd: str = ""
        model_profile: str = ""
        model: str = ""
        permission_mode: str = ""
        resume_session_id: str = ""
        system_prompt_append: str = ""
        add_dirs: list[str] = Field(default_factory=list)
        allowed_tools: list[str] = Field(default_factory=list)
        mcp_config: str = ""
        target: str = ""
        extra_args: list[str] = Field(default_factory=list)
        title: str = ""
        owner: str = ""

    class SendInput(BaseModel):
        text: str

    # ── unauthenticated liveness ────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, Any]:
        roots = allowed_roots()
        return {
            "ok": True,
            "service": "aithershell-harness",
            "sessions": len(mgr.list_sessions()),
            "harnesses_installed": mgr.available_harness_ids(),
            "cors_origins": origins,
            # Stated explicitly so "this host is trusted" is a visible posture,
            # never an unnoticed default on a tunnel-exposed daemon.
            "cwd_restricted": bool(roots),
            "allowed_roots": [str(r) for r in roots],
        }

    # ── discovery ───────────────────────────────────────────────────────────

    @app.get("/harnesses", dependencies=[Depends(auth)])
    def harnesses(versions: bool = Query(default=False)) -> dict[str, Any]:
        return {"harnesses": mgr.harnesses(with_version=versions)}

    @app.get("/profiles", dependencies=[Depends(auth)])
    def profiles() -> dict[str, Any]:
        try:
            raw = list_profiles()
        except ProfileError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "profiles": [
                {
                    "id": name,
                    "description": p.get("description", ""),
                    "transport": p.get("transport", ""),
                    "context_window": p.get("context_window"),
                    "model": p.get("model", ""),
                }
                for name, p in raw.items()
            ]
        }

    # ── sessions ────────────────────────────────────────────────────────────

    @app.post("/sessions", dependencies=[Depends(auth)])
    def create_session(body: CreateSession) -> dict[str, Any]:
        try:
            cwd = validate_cwd(body.cwd)
            config = SessionConfig(**{**body.model_dump(), "cwd": cwd})
            session = mgr.create(config)
        except ManagerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return session.info()

    @app.get("/sessions", dependencies=[Depends(auth)])
    def list_sessions(owner: str = Query(default="")) -> dict[str, Any]:
        return {"sessions": mgr.list_sessions(owner=owner)}

    @app.get("/sessions/{session_id}", dependencies=[Depends(auth)])
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return mgr.get_session(session_id).info()
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/input", dependencies=[Depends(auth)])
    def send_input(session_id: str, body: SendInput) -> dict[str, Any]:
        try:
            session = mgr.get_session(session_id)
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        accepted = session.send(body.text)
        if not accepted:
            raise HTTPException(
                status_code=409, detail=f"session {session.state} cannot accept input"
            )
        return {"ok": True, "turn": session.turn, "seq": session.last_seq}

    def _browse_roots() -> Any:
        from adk.harnesses.fs import browse_roots

        return browse_roots([s.get("cwd", "") for s in mgr.list_sessions()])

    @app.get("/fs/list", dependencies=[Depends(auth)])
    def fs_list(path: str = Query(default="")) -> dict[str, Any]:
        from adk.harnesses.fs import FsDeniedError, list_dir

        try:
            return list_dir(path, _browse_roots())
        except FsDeniedError as exc:
            # 403 with the reason, never an empty listing. An empty listing
            # reads as "that folder is empty" and hides a containment refusal.
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/fs/read", dependencies=[Depends(auth)])
    def fs_read(path: str = Query(...)) -> dict[str, Any]:
        from adk.harnesses.fs import FsDeniedError, read_file

        try:
            return read_file(path, _browse_roots())
        except FsDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/workforce", dependencies=[Depends(auth)])
    def workforce() -> dict[str, Any]:
        """Aitherium Workforce roster, proxied from Genesis.

        Reported with an explicit ``available`` flag rather than an empty list:
        ``workforce.py`` itself notes the runtime is "frequently not" running,
        and an empty roster would read as "you have no agents" instead of
        "the Workforce service did not answer".
        """
        from adk.harnesses.agents import fetch_workforce

        roster, reason = fetch_workforce()
        return {"agents": roster, "available": bool(roster), "reason": reason}

    @app.post("/sandboxes", dependencies=[Depends(auth)])
    def provision_sandbox(body: ProvisionSandbox) -> dict[str, Any]:
        """Provision a dev workspace container and report how to get inside it.

        Returns the descriptor plus the ``container`` a ``sandbox`` session
        should target, so the caller does not have to know the naming
        convention. An empty ``container`` is reported honestly rather than
        guessed — a wrong name fails later as "no such container", which reads
        as a broken terminal instead of an uninterpretable provisioning result.
        """
        from adk.harnesses import sandbox as sandbox_mod

        try:
            descriptor = sandbox_mod.provision(body.workspace_slug, body.repos or None)
        except sandbox_mod.SandboxError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        container, reason = sandbox_mod.container_name(descriptor)
        # `reason` is returned, not swallowed: a provisioned workspace we cannot
        # exec into must say WHY, or the UI shows a sandbox that silently
        # refuses to open a terminal.
        return {"workspace": descriptor, "container": container, "container_reason": reason}

    @app.get("/sandboxes", dependencies=[Depends(auth)])
    def list_sandboxes() -> dict[str, Any]:
        from adk.harnesses import sandbox as sandbox_mod

        try:
            return {"sandboxes": sandbox_mod.mine()}
        except sandbox_mod.SandboxError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.delete("/sandboxes/{workspace_id}", dependencies=[Depends(auth)])
    def teardown_sandbox(workspace_id: str) -> dict[str, Any]:
        from adk.harnesses import sandbox as sandbox_mod

        try:
            return sandbox_mod.teardown(workspace_id)
        except sandbox_mod.SandboxError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/agents", dependencies=[Depends(auth)])
    def agents() -> dict[str, Any]:
        from adk.harnesses.agents import AGENT_ROSTER

        return {"agents": AGENT_ROSTER}

    @app.post("/sessions/{session_id}/resize", dependencies=[Depends(auth)])
    def resize(session_id: str, body: ResizeInput) -> dict[str, Any]:
        try:
            return {"resized": mgr.resize(session_id, body.rows, body.cols)}
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/interrupt", dependencies=[Depends(auth)])
    def interrupt(session_id: str) -> dict[str, Any]:
        try:
            return {"interrupted": mgr.interrupt(session_id)}
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id}", dependencies=[Depends(auth)])
    def stop_session(session_id: str) -> dict[str, Any]:
        try:
            return mgr.stop(session_id)
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/events", dependencies=[Depends(auth)])
    def events(session_id: str, since: int = Query(default=0)) -> dict[str, Any]:
        try:
            session = mgr.get_session(session_id)
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        evs = session.events_since(since)
        return {"events": evs, "last_seq": session.last_seq, "state": session.state}

    @app.get("/sessions/{session_id}/stream", dependencies=[Depends(auth)])
    async def stream(session_id: str, since: int = Query(default=0)):
        """Server-sent events. Resumable via ``?since=`` after a reconnect."""
        try:
            session = mgr.get_session(session_id)
        except ManagerError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def gen():
            cursor = since
            idle_ticks = 0
            while True:
                batch = session.events_since(cursor)
                if batch:
                    idle_ticks = 0
                    for event in batch:
                        cursor = event["seq"]
                        yield f"event: {event['kind']}\ndata: {json.dumps(event)}\n\n"
                        if event["kind"] == "session.exited":
                            return
                else:
                    idle_ticks += 1
                    # A comment frame keeps proxies (and Cloudflare, on the
                    # tunnel path) from closing an idle stream. Without it a
                    # long model turn looks like a dead connection.
                    if idle_ticks % 20 == 0:
                        yield ": keepalive\n\n"
                await asyncio.sleep(0.15)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def serve(host: str = "", port: int = 0, token: str = "") -> int:
    """Run the daemon. Prints the token location, never the token itself."""
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("uvicorn is required: pip install uvicorn fastapi\n")
        return 2

    bind_host = host or DEFAULT_HOST
    bind_port = port or DEFAULT_PORT
    app = create_app(token=token)
    sys.stderr.write(
        f"AitherShell harness daemon on http://{bind_host}:{bind_port}\n"
        f"  token: {TOKEN_PATH} (or $AITHER_HARNESS_TOKEN)\n"
        f"  cors : {', '.join(allowed_origins())}\n"
    )
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="warning")
    return 0
