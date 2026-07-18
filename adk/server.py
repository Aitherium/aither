"""FastAPI server wrapping an AitherAgent — OpenAI-compatible + Genesis-compatible.

Supports two modes:
- Single agent: `aither-serve --identity aither`
- Fleet mode:   `aither-serve --fleet fleet.yaml` or `aither-serve --agents aither,lyra,demiurge`
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from adk import __version__
from adk.agent import AitherAgent, AgentResponse
from adk.config import Config
from adk.identity import list_identities, load_identity
from adk.llm import LLMRouter, Message
from adk.metrics import get_metrics
from adk.trace import TraceMiddleware, get_trace_id, new_trace

logger = logging.getLogger("adk.server")

_WEBUI_CACHE: str | None = None


def _load_webui() -> str | None:
    """Load the packaged admin-console SPA (adk/webui/index.html).

    Shipped as package data in the wheel; read once and cached. Returns None if
    the asset is missing so callers can fall back to the minimal chat page.
    """
    global _WEBUI_CACHE
    if _WEBUI_CACHE is not None:
        return _WEBUI_CACHE or None
    try:
        from importlib.resources import files
        html = (files("adk") / "webui" / "index.html").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        # Dev fallback: read from the source tree next to this module.
        try:
            html = (os.path.join(os.path.dirname(__file__), "webui", "index.html"))
            with open(html, encoding="utf-8") as fh:
                html = fh.read()
        except OSError:
            _WEBUI_CACHE = ""
            return None
    _WEBUI_CACHE = html
    return html


# ─── Swappable agent UI packs ────────────────────────────────────────────────
# The page an agent serves at "/" is a swappable UI PACK, so you can drop in,
# test, and deploy different chat frontends (the full console, a minimal chat,
# AitherAeon, a company room, a custom brand) without touching the agent.
#
# Selection (first that is set wins): $AITHER_AGENT_UI env, else "console".
# Resolution order for a pack name:
#   1. drop-in dir: $AITHER_UI_PACKS_DIR/<name>/index.html  (default ~/.aither/ui-packs)
#   2. packaged built-in: adk/webui/packs/<name>/index.html
#   3. special built-ins: "console" -> the full admin SPA (adk/webui/index.html),
#      "minimal" -> the tiny streaming chat page (_CHAT_HTML)
# Anything unresolvable falls back to console -> minimal, so "/" is NEVER blank.

_BUILTIN_UI_PACKS = ("console", "minimal")


# ── Central-console admin proxy allowlist (module-level so it is unit-testable) ─
#
# The console proxies each mesh agent's OWN /admin API for owner-authed
# management: OBSERVE routes + reversible controls + CONFIGURATION (switch
# backend local<->cloud, set the model, set provider API keys, edit config).
# The ONE thing never proxied is arbitrary code execution — POST /admin/cli/exec
# and GET /admin/cli/commands are hard-denied (that, not config management, was
# the real RCE surface). Everything here is gated by the owner server bearer and
# forwarded to a single discovered agent (no fan-out).
#
# SECRET BOUNDARY: API-key VALUES flow browser -> proxy -> the agent's own
# admin-save (which persists to its vault). The console never RENDERS a stored
# key — reads return the agent's masked view (admin_api._mask_*). The owner
# supplies key values in the UI; the assistant never handles them.
_MESH_ADMIN_ALLOW = {
    "GET": {
        "config", "meta", "routes",
        "llm/status",
        "packs", "catalog/packs",
        "sessions",
        "logs/tail",
        "mcp/servers",
        "graph/stats",
    },
    "POST": {
        # Reversible controls
        "packs/enable", "packs/disable", "packs/reload",
        # Push+enable a bundled pack to a remote agent (no SSH). Only bundled
        # packs are accepted by the target; no arbitrary uploads.
        "packs/apply",
        # LLM configuration — switch backend (local<->cloud), set provider API
        # key, test the connection. These are owner config actions, not RCE.
        "llm/switch", "llm/keys", "llm/test",
        # MCP server management (add via prepare/confirm)
        "mcp/servers/prepare", "mcp/servers/confirm",
    },
    "PATCH": {
        "config",  # edit allowlisted config fields
    },
    "DELETE": {
        "sessions",       # terminate a session (prefix match: sessions/{id})
        "mcp/servers",    # remove an MCP server (prefix match: mcp/servers/{id})
    },
}
# Prefix allowances for parameterized routes (path starts with these).
_MESH_ADMIN_GET_PREFIXES = ("packs/", "sessions/", "mcp/servers/")
# Routes that are NEVER proxied regardless of method — arbitrary code execution.
_MESH_ADMIN_DENY = ("cli/exec", "cli/commands", "cli")


def _mesh_admin_allowed(method: str, sub: str) -> bool:
    """True only for explicitly-allowed (method, admin-subpath) pairs. Fail-closed:
    anything not named here is refused, and the cli/* exec surface is hard-denied
    regardless of method. Prefix rules cover /{id}-style routes."""
    sub = (sub or "").strip("/")
    if not sub or ".." in sub:
        return False
    # Hard deny: arbitrary code execution is never reachable through the console.
    for bad in _MESH_ADMIN_DENY:
        if sub == bad or sub.startswith(bad + "/"):
            return False
    m = method.upper()
    if sub in _MESH_ADMIN_ALLOW.get(m, set()):
        return True
    if m == "GET":
        for pfx in _MESH_ADMIN_GET_PREFIXES:
            if sub.startswith(pfx) and sub != pfx:
                return True
    if m == "PATCH" and (sub.startswith("packs/") and sub.endswith("/settings")):
        return True  # per-pack settings edit
    if m == "DELETE" and sub.startswith("sessions/") and sub != "sessions/":
        return True
    if m == "DELETE" and sub.startswith("mcp/servers/") and sub != "mcp/servers/":
        return True
    return False


def _ui_packs_dir() -> str:
    """Drop-in directory for custom UI packs (one folder per pack)."""
    explicit = os.getenv("AITHER_UI_PACKS_DIR", "").strip()
    if explicit:
        return explicit
    home = os.environ.get("AITHER_HOME") or os.environ.get("HOME") \
        or os.environ.get("USERPROFILE") or "."
    return os.path.join(home, ".aither", "ui-packs")


def resolve_ui_pack_name() -> str:
    """The selected UI pack name: $AITHER_AGENT_UI, else the persisted
    ``agent_ui`` from ~/.aither/config.json (set by `adk ui set`), else 'llamacpp'."""
    env = os.getenv("AITHER_AGENT_UI", "").strip()
    if env:
        return env
    try:
        from adk.config import load_saved_config
        saved = load_saved_config().get("agent_ui")
        if saved:
            return str(saved).strip()
    except Exception:
        pass
    return "llamacpp"


def list_ui_packs() -> dict[str, str]:
    """Map of available pack name -> source ('builtin' | 'packaged' | drop-in path)."""
    packs: dict[str, str] = {name: "builtin" for name in _BUILTIN_UI_PACKS}
    # packaged built-ins under adk/webui/packs/*
    try:
        pkg = os.path.join(os.path.dirname(__file__), "webui", "packs")
        if os.path.isdir(pkg):
            for name in sorted(os.listdir(pkg)):
                if os.path.isfile(os.path.join(pkg, name, "index.html")):
                    packs.setdefault(name, "packaged")
    except OSError:
        pass
    # drop-in packs (override packaged/builtin of the same name)
    try:
        dropin = _ui_packs_dir()
        if os.path.isdir(dropin):
            for name in sorted(os.listdir(dropin)):
                idx = os.path.join(dropin, name, "index.html")
                if os.path.isfile(idx):
                    packs[name] = idx
    except OSError:
        pass
    return packs


def load_ui_pack(name: str | None = None) -> str | None:
    """Load the selected UI pack's HTML, with a fail-soft fallback chain so "/"
    is never blank. Not cached (a dev swapping packs sees the change on reload)."""
    name = (name or resolve_ui_pack_name()).strip() or "console"
    # 1. drop-in dir wins
    try:
        idx = os.path.join(_ui_packs_dir(), name, "index.html")
        if os.path.isfile(idx):
            with open(idx, encoding="utf-8") as fh:
                return fh.read()
    except OSError:
        pass
    # 2. packaged built-in under adk/webui/packs/<name>/
    try:
        idx = os.path.join(os.path.dirname(__file__), "webui", "packs", name, "index.html")
        if os.path.isfile(idx):
            with open(idx, encoding="utf-8") as fh:
                return fh.read()
    except OSError:
        pass
    # 3. special built-ins
    if name == "minimal":
        return _CHAT_HTML
    if name == "console":
        return _load_webui() or _CHAT_HTML
    # 4. unknown pack -> console -> minimal (never blank)
    logger.warning("UI pack %r not found; falling back to console", name)
    return _load_webui() or _CHAT_HTML


_PACK_SDK_CACHE: str | None = None


def _load_pack_sdk() -> str | None:
    """Load the pack-UI bridge SDK (adk/webui/pack_sdk.js), packaged like the console."""
    global _PACK_SDK_CACHE
    if _PACK_SDK_CACHE is not None:
        return _PACK_SDK_CACHE or None
    try:
        from importlib.resources import files
        js = (files("adk") / "webui" / "pack_sdk.js").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        try:
            path = os.path.join(os.path.dirname(__file__), "webui", "pack_sdk.js")
            with open(path, encoding="utf-8") as fh:
                js = fh.read()
        except OSError:
            _PACK_SDK_CACHE = ""
            return None
    _PACK_SDK_CACHE = js
    return js


# A tiny, self-contained streaming chat page the agent serves at "/" so a person
# who ran `adk up` has somewhere to talk to it with live feedback (a "thinking…"
# indicator + token-by-token streaming) instead of a 30-second blank wait. It
# calls the agent's own gated /chat/stream; the callback bearer is read from the
# URL fragment (#k=…), which the browser never sends to the server or logs, so
# the inference endpoint stays authenticated (not an open, abusable proxy).
_CHAT_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aither Agent</title>
<style>
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:#0b0d10;color:#e6e8eb;display:flex;flex-direction:column;height:100vh}
@media(prefers-color-scheme:light){body{background:#f6f7f9;color:#12141a}}
header{padding:12px 16px;border-bottom:1px solid #2a2f37;font-weight:600;display:flex;
 gap:8px;align-items:center}
@media(prefers-color-scheme:light){header{border-color:#e2e5ea}}
.dot{width:8px;height:8px;border-radius:50%;background:#3ba55d}
#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:760px;padding:10px 13px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:3px}
.assistant{align-self:flex-start;background:#1b2027;border-bottom-left-radius:3px}
@media(prefers-color-scheme:light){.assistant{background:#eceef2}}
.thinking{opacity:.7;font-style:italic}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid #2a2f37}
@media(prefers-color-scheme:light){form{border-color:#e2e5ea}}
#in{flex:1;padding:11px 13px;border-radius:10px;border:1px solid #2a2f37;background:#12151a;
 color:inherit;font:inherit;resize:none}
@media(prefers-color-scheme:light){#in{background:#fff;border-color:#cfd4dc}}
button{padding:0 18px;border:0;border-radius:10px;background:#2563eb;color:#fff;font:inherit;
 font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:default}
</style></head><body>
<header><span class="dot"></span><span id="name">Aither Agent</span></header>
<div id="log"></div>
<form id="f"><textarea id="in" rows="1" placeholder="Message your agent…" autofocus></textarea>
<button id="send">Send</button></form>
<script>
var log=document.getElementById('log'),input=document.getElementById('in'),
 form=document.getElementById('f'),btn=document.getElementById('send');
var key=(location.hash.match(/[#&]k=([^&]+)/)||[])[1]||'';
var sid=localStorage.getItem('adk_sid')||('web-'+Math.random().toString(36).slice(2));
localStorage.setItem('adk_sid',sid);
function hdrs(h){h=h||{};if(key)h['Authorization']='Bearer '+decodeURIComponent(key);return h;}
function bubble(role,text){var d=document.createElement('div');d.className='msg '+role;
 d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
fetch('/health').then(function(r){return r.json()}).then(function(j){
 if(j&&j.agent)document.getElementById('name').textContent=j.agent;}).catch(function(){});
function thinking(el){var i=0;el.classList.add('thinking');
 var t=setInterval(function(){i=(i+1)%4;el.textContent='thinking'+Array(i+1).join('.');},400);
 return t;}
async function send(text){
 bubble('user',text);
 var el=bubble('assistant','');var tk=thinking(el);var got=false,acc='';
 btn.disabled=true;
 try{
  var res=await fetch('/chat/stream',{method:'POST',headers:hdrs({'Content-Type':'application/json'}),
   body:JSON.stringify({message:text,session_id:sid})});
  if(!res.ok){clearInterval(tk);el.classList.remove('thinking');
   el.textContent='⚠ '+res.status+(res.status===401?' — reopen this page from `adk up`':'');btn.disabled=false;return;}
  var reader=res.body.getReader(),dec=new TextDecoder(),buf='';
  while(true){var c=await reader.read();if(c.done)break;buf+=dec.decode(c.value,{stream:true});
   var idx;while((idx=buf.indexOf('\n\n'))>=0){var frame=buf.slice(0,idx);buf=buf.slice(idx+2);
    var line=frame.split('\n').filter(function(l){return l.indexOf('data:')===0})[0];if(!line)continue;
    var d;try{d=JSON.parse(line.slice(5).trim())}catch(e){continue}
    if(d.type==='token'){if(!got){got=true;clearInterval(tk);el.classList.remove('thinking');el.textContent=''}
     acc+=(d.t||'');el.textContent=acc;log.scrollTop=log.scrollHeight}
    else if(d.type==='answer'){if(!got){clearInterval(tk);el.classList.remove('thinking');got=true}
     acc=d.answer||acc;el.textContent=acc;log.scrollTop=log.scrollHeight}
    else if(d.type==='error'){clearInterval(tk);el.classList.remove('thinking');el.textContent='⚠ '+(d.error||'error')}}}
 }catch(e){clearInterval(tk);el.classList.remove('thinking');el.textContent='⚠ '+e.message}
 clearInterval(tk);el.classList.remove('thinking');if(!acc&&!el.textContent)el.textContent='(no response)';
 btn.disabled=false;input.focus();
}
form.addEventListener('submit',function(e){e.preventDefault();var t=input.value.trim();
 if(!t)return;input.value='';send(t)});
input.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();
 form.requestSubmit()}});
</script></body></html>'''


def create_app(
    agent: AitherAgent | None = None,
    identity: str = "aither",
    config: Config | None = None,
    fleet_path: str | None = None,
    fleet_agents: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app wrapping an AitherAgent or a fleet of agents.

    Returns a fully configured app with both OpenAI-compatible and Genesis-compatible endpoints.
    """
    config = config or Config.from_env()

    is_fleet = bool(fleet_path or fleet_agents)

    # ─── Lifespan (replaces deprecated @app.on_event("startup")) ───

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Run startup tasks, yield for request handling, then cleanup."""
        # Configure structured logging
        from adk.chronicle import configure_logging
        configure_logging(
            level=os.getenv("AITHER_LOG_LEVEL", "INFO"),
            json_output=config.json_logging,
        )

        # Sovereign/offline mode: skip ALL network registration (gateway, secrets,
        # AitherNet, relays, Elysium, mesh, fleet heartbeat, telemetry flushes).
        # The server still mounts every LOCAL route (incl. FormBridge) + the local
        # MCP/A2A servers — it just never phones home. This is the PHI/sovereign
        # posture AND the FormBridge demo runtime, so there is ONE server, not a
        # separate demo shim. Enable with AITHER_OFFLINE=1.
        offline = os.getenv("AITHER_OFFLINE", "").lower() in ("1", "true", "yes")
        if offline:
            logger.info("AITHER_OFFLINE=1 — sovereign mode: local only, no network registration")

        if not offline:
            await _connect_gateway_mcp()
            await _register_with_gateway()
            await _sync_secrets()
            await _join_aithernet()
            await _rich_enroll_identity()
            await _init_chat_relay()
            await _init_mail_relay()
            await _init_relay_client()
        await _init_mcp_server()
        await _init_a2a_server()
        await _connect_service_bridge()
        if not offline:
            await _flush_strata_queue()
            await _flush_chronicle_queue()
            await _start_watch_reporter()
            await _flush_pulse_queue()
        # Eagerly detect LLM backend so /health shows the right provider
        try:
            a = await get_agent()
            await a.llm.get_provider()
        except (ImportError, RuntimeError, OSError):
            pass

        # ── Settings sync: portal profile is the source of truth ──
        # Pull the user's profile settings and apply them over the local cache,
        # then keep the client on _state so admin-console mutations can push
        # updates back up. Fail-soft: never blocks boot.
        if not offline:
            try:
                from adk.sync.settings import build_client
                _settings_sync = build_client()
                if _settings_sync is not None:
                    _state["settings_sync"] = _settings_sync
                    try:
                        a = await get_agent()
                    except (ImportError, RuntimeError, OSError, ConnectionError):
                        a = None
                    result = await _settings_sync.pull_and_apply(a)
                    logger.info("settings sync: pulled portal profile — %s", result)
            except Exception as exc:  # noqa: BLE001 — sync is advisory
                logger.warning("settings sync init failed (using local config): %s", exc)

        if not offline:
            # ── Elysium auto-reconnect + mesh hosting ──
            await _reconnect_elysium()
            await _start_mesh_hosting()

            # ── Fleet endpoint registration + continuous heartbeat ──
            await _register_fleet_endpoint()
            _heartbeat_task = asyncio.create_task(_fleet_heartbeat_loop())
            _state["heartbeat_task"] = _heartbeat_task

        yield
        # ── Shutdown cleanup ──
        hb_task = _state.get("heartbeat_task")
        if hb_task and not hb_task.done():
            hb_task.cancel()
            try:
                await hb_task
            except (asyncio.CancelledError, Exception):
                pass
        await _deregister_fleet_endpoint()
        _elysium_relay = _state.get("elysium_relay")
        if _elysium_relay:
            await _elysium_relay.stop_heartbeat()
            await _elysium_relay.disconnect_relay_hub()
        bridge = _state.get("aither_bridge")
        if bridge:
            await bridge.stop()
        chat = _state.get("chat_relay")
        if chat:
            await chat.stop_irc_server()
        relay_client = _state.get("relay_client")
        if relay_client:
            try:
                await relay_client.disconnect()
            except (OSError, RuntimeError) as exc:
                logger.warning(f"relay_client disconnect on shutdown failed: {exc}")

    app = FastAPI(
        title=f"AitherADK — {'Fleet' if is_fleet else identity}",
        version=__version__,
        docs_url="/docs",
        lifespan=_lifespan,
    )

    _cors_origins = os.getenv("AITHER_CORS_ORIGINS", "").split(",")
    _cors_origins = [o.strip() for o in _cors_origins if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins or ["http://localhost:3000", "http://localhost:8080"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        # Athena: no wildcard — only the headers our clients actually send. A
        # permissive list widens what a whitelisted-origin page can do cross-site.
        allow_headers=["Authorization", "Content-Type", "Accept",
                       "X-Caller-Type", "X-Request-ID", "X-Tenant-ID", "X-Workspace-ID"],
    )

    # Trace ID middleware — generates/propagates X-Request-ID on every request
    app.add_middleware(TraceMiddleware)

    # State shared across endpoints
    _state: dict[str, Any] = {
        "agent": agent,
        "identity": identity,
        "config": config,
        "fleet": None,
        "is_fleet": is_fleet,
        "service_bridge": None,
    }

    # ─── Auth middleware (optional, enabled via AITHER_API_KEY or --api-key) ───

    _server_api_key = os.getenv("AITHER_SERVER_API_KEY", "")
    # "/" and "/chat" serve only the static chat page (no data); the page then
    # authenticates to the gated /chat/stream with the bearer from its URL fragment.
    # Similarly, "/aeon" serves the group-chat UI; "/aeon/stream" is bearer-gated.
    _skip_auth_paths = {"/", "/chat", "/aeon", "/ui", "/health", "/docs", "/openapi.json",
                        "/metrics", "/demo", "/redoc"}

    def _is_pack_ui_asset(path: str) -> bool:
        """Pack-UI static assets are unauthenticated like the console shell at "/".

        The console mounts pack UIs in sandboxed iframes with NO token — that is
        the security model: the bearer never enters pack code — so the iframe
        cannot send an Authorization header for its own HTML/JS/CSS. The asset
        route only serves files from an enabled pack's declared assets dir, and
        every privileged action still goes through the bearer-gated
        /admin/packs/*/tools/*/invoke bridge. Deliberately narrow match: the
        pack-UI prefix and the bridge SDK, nothing else under /packs/.
        """
        return path == "/packs/_sdk.js" or (path.startswith("/packs/") and "/ui/" in path)

    # Valid caller types for header validation (prevents spoofing)
    _valid_caller_types = {"PLATFORM", "PUBLIC", "DEMO", "TENANT", "ANONYMOUS"}

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        """Bearer token auth + caller-type header validation.

        Validates X-Caller-Type header to prevent header-spoofing attacks.
        External requests cannot claim PLATFORM caller type.
        """
        if request.url.path in _skip_auth_paths or _is_pack_ui_asset(request.url.path):
            return await call_next(request)

        # Validate X-Caller-Type if present (prevent spoofing)
        caller_type = request.headers.get("x-caller-type", "")
        if caller_type:
            if caller_type not in _valid_caller_types:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid X-Caller-Type: {caller_type}"},
                )
            # External requests cannot claim PLATFORM — that's internal-only
            if caller_type == "PLATFORM" and _server_api_key:
                auth_header = request.headers.get("authorization", "")
                platform_tok = auth_header[7:] if auth_header.startswith("Bearer ") else ""
                # hmac.compare_digest: constant-time, no early-exit timing leak over the tunnel
                if not platform_tok or not hmac.compare_digest(platform_tok, _server_api_key):
                    return JSONResponse(
                        status_code=403,
                        content={"error": "PLATFORM caller type requires valid API key"},
                    )

        if not _server_api_key:
            return await call_next(request)

        # Genuine local access is trusted — no token needed when you're on the box.
        # A request whose immediate peer is loopback AND which carries no proxy /
        # forwarding headers can only have originated on THIS machine. Cloudflare
        # (and any reverse proxy) hits loopback too, but always adds
        # X-Forwarded-For / Cf-Connecting-Ip / Forwarded, so the PUBLIC tunnel stays
        # bearer-gated. This is what makes "open my own agent" seamless in the mesh
        # without pasting a token into the URL.
        client_host = request.client.host if request.client else ""
        forwarded = (
            request.headers.get("x-forwarded-for")
            or request.headers.get("cf-connecting-ip")
            or request.headers.get("forwarded")
        )
        if client_host in ("127.0.0.1", "::1") and not forwarded:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"error": "Missing or invalid Authorization header"})
        token = auth_header[7:]
        # hmac.compare_digest: constant-time comparison — the bearer is reachable over
        # the public trycloudflare tunnel, so a byte-by-byte `!=` would leak the token
        # position-by-position to a timing attacker.
        if not hmac.compare_digest(token, _server_api_key):
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
        return await call_next(request)

    async def _init_fleet():
        """Initialize fleet mode (lazy, on first request)."""
        if _state["fleet"] is not None:
            return _state["fleet"]
        from adk.fleet import load_fleet
        fleet = load_fleet(
            path=fleet_path,
            agent_names=fleet_agents,
            config=config,
        )
        _state["fleet"] = fleet
        return fleet

    async def get_agent(name: str | None = None) -> AitherAgent:
        """Get agent by name. In fleet mode, routes to the right agent."""
        if is_fleet:
            fleet = await _init_fleet()
            if name and name in fleet.registry:
                return fleet.registry.get(name)
            # Default to orchestrator
            orch = fleet.get_orchestrator()
            if orch:
                return orch
            # Fallback to first agent
            if fleet.agents:
                return fleet.agents[0]

        if _state["agent"] is None:
            # Load agent spec with customization overrides
            from adk.pack_discovery import load_agent_spec
            from pathlib import Path

            identity = _state["identity"]
            agent_spec = {}

            # Try to load agent spec from installed pack
            pack_dir = Path.home() / ".aither" / "agents" / identity
            if pack_dir.exists():
                agent_yaml = pack_dir / "agent.yaml"
                if agent_yaml.exists():
                    agent_spec = load_agent_spec(agent_yaml) or {}

            # Build AitherAgent with optional system_prompt override
            kwargs = {
                "name": identity,
                "identity": identity,
                "config": _state["config"],
            }
            if agent_spec.get("system_prompt"):
                kwargs["system_prompt"] = agent_spec["system_prompt"]

            _state["agent"] = AitherAgent(**kwargs)
        agent = _state["agent"]

        # If a different agent is requested in single mode, create it
        if name and name != agent.name:
            # Load agent spec for the requested agent
            from adk.pack_discovery import load_agent_spec
            from pathlib import Path

            agent_spec = {}
            pack_dir = Path.home() / ".aither" / "agents" / name
            if pack_dir.exists():
                agent_yaml = pack_dir / "agent.yaml"
                if agent_yaml.exists():
                    agent_spec = load_agent_spec(agent_yaml) or {}

            kwargs = {
                "name": name,
                "identity": name,
                "config": _state["config"],
            }
            if agent_spec.get("system_prompt"):
                kwargs["system_prompt"] = agent_spec["system_prompt"]

            return AitherAgent(**kwargs)

        return agent

    # ─── Metrics (Prometheus) ───

    @app.get("/metrics")
    async def metrics_endpoint(request: Request):
        """Prometheus-compatible metrics export. Requires auth token or localhost."""
        # Allow localhost and container-internal access without auth
        client_host = request.client.host if request.client else ""
        is_local = client_host in ("127.0.0.1", "::1", "localhost", "")
        if not is_local:
            auth = request.headers.get("authorization", "")
            metrics_token = os.getenv("AITHER_METRICS_TOKEN", "")
            if metrics_token and auth != f"Bearer {metrics_token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return PlainTextResponse(get_metrics().export(), media_type="text/plain; version=0.0.4")

    # ─── Health ───

    @app.get("/health")
    async def health():
        try:
            a = await get_agent()
            provider = a.llm.provider_name or "detecting..."
            agent_name = a.name
        except ConnectionError:
            provider = "none"
            agent_name = _state["identity"]

        result = {
            "status": "healthy",
            "agent": agent_name,
            "llm_backend": provider,
            "version": __version__,
            "gateway_connected": _state.get("gateway_connected", False),
            "gateway_mcp_connected": _state.get("gateway_mcp_connected", False),
        }

        if is_fleet and _state["fleet"]:
            fleet = _state["fleet"]
            result["fleet"] = {
                "name": fleet.name,
                "agents": fleet.registry.agent_names,
                "orchestrator": fleet.orchestrator_name,
            }

        return result

    # ─── Built-in streaming chat page (so a human has somewhere to talk) ───

    @app.get("/", response_class=HTMLResponse)
    async def console_page():
        # The page `adk up` opens is the SELECTED UI pack ($AITHER_AGENT_UI,
        # default "console" = the full admin SPA). Swap it with `adk ui set
        # <pack>` or drop a folder in ~/.aither/ui-packs/. Never blank — falls
        # back console -> minimal.
        return HTMLResponse(load_ui_pack())

    @app.get("/ui", response_class=HTMLResponse)
    async def console_page_alias():
        return HTMLResponse(load_ui_pack())

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page_minimal():
        # Back-compat: the original lightweight streaming chat page (the
        # "minimal" pack), regardless of the selected pack.
        return HTMLResponse(load_ui_pack("minimal"))

    @app.get("/aeon", response_class=HTMLResponse)
    async def aeon_page():
        # Aeon group-chat UI pack — multi-agent discussion.
        return HTMLResponse(load_ui_pack("aeon"))

    # ─── Pack UI assets + bridge SDK (sandboxed-iframe plugin system) ───
    # A pack may declare a `ui:` block in its .toolpack.yaml; the console then
    # mounts its page as <iframe sandbox="allow-scripts allow-forms"> (opaque
    # origin, NO token). These static routes are auth-skipped like "/" itself;
    # everything privileged goes through the bearer-gated invoke bridge.

    _PACK_ASSET_TYPES = {
        ".html": "text/html; charset=utf-8", ".js": "text/javascript",
        ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
        ".woff": "font/woff", ".woff2": "font/woff2", ".txt": "text/plain",
        ".map": "application/json",
    }

    @app.get("/packs/_sdk.js")
    async def pack_sdk_js():
        js = _load_pack_sdk()
        if js is None:
            return JSONResponse(status_code=404, content={"error": "sdk_asset_missing"})
        return PlainTextResponse(js, media_type="text/javascript",
                                 headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/packs/{pack_id}/ui/{asset_path:path}")
    async def pack_ui_asset(pack_id: str, asset_path: str):
        """Serve a static file from an ENABLED pack's declared ui assets dir.

        Fail-closed everywhere: unknown pack → 404; pack not enabled → 403;
        no ui block → 404; any path that resolves outside the assets dir
        (traversal, absolute, drive-qualified, symlink escape) → 403.
        """
        from adk.config import load_saved_config as _lsc
        from adk.pack_scope import valid_pack_id

        if not valid_pack_id(pack_id):
            return JSONResponse(status_code=400, content={"error": "invalid_pack_id"})
        try:
            from adk.tool_pack_loader import get_tool_pack_loader
            manifest = get_tool_pack_loader().discover().get(pack_id)
        except (ImportError, RuntimeError):
            manifest = None
        if manifest is None:
            return JSONResponse(status_code=404, content={"error": "pack_not_found"})
        if pack_id not in (_lsc().get("required_packs") or []):
            return JSONResponse(status_code=403, content={"error": "pack_not_enabled"})
        assets_dir = manifest.ui_assets_dir
        if assets_dir is None or not assets_dir.is_dir():
            return JSONResponse(status_code=404, content={"error": "pack_has_no_ui"})

        rel = (asset_path or "index.html").replace("\\", "/")
        if rel.endswith("/") or rel == "":
            rel += "index.html"
        # Reject traversal/absolute forms before touching the filesystem…
        if rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
            return JSONResponse(status_code=403, content={"error": "invalid_asset_path"})
        try:
            target = (assets_dir / rel).resolve()
            # …and verify the RESOLVED path (catches symlink escapes).
            if not target.is_relative_to(assets_dir.resolve()):
                return JSONResponse(status_code=403, content={"error": "invalid_asset_path"})
        except (OSError, ValueError):
            return JSONResponse(status_code=403, content={"error": "invalid_asset_path"})
        if not target.is_file():
            return JSONResponse(status_code=404, content={"error": "asset_not_found"})
        if target.stat().st_size > 5_000_000:
            return JSONResponse(status_code=413, content={"error": "asset_too_large"})
        media = _PACK_ASSET_TYPES.get(target.suffix.lower(), "application/octet-stream")
        from fastapi.responses import FileResponse
        return FileResponse(
            target, media_type=media,
            headers={
                "X-Content-Type-Options": "nosniff",
                # Only the console (same origin) may frame pack pages.
                "Content-Security-Policy": "frame-ancestors 'self'",
                "Cache-Control": "no-cache",
            })

    # ─── Admin/settings console API (all under /admin/*, bearer-gated) ───
    # Registered as a function (not an APIRouter) so the handlers can close over
    # get_agent + _state and operate on the LIVE agent — backend swap, pack
    # reload, and MCP registration all take effect without a restart. /admin is
    # intentionally absent from _skip_auth_paths so _auth_middleware gates it.
    try:
        from adk.admin_api import register_admin_routes
        register_admin_routes(app, get_agent=get_agent, state=_state)
    except ImportError as exc:
        logger.warning("admin console API unavailable: %s", exc)

    # ISO Factory bridge (/admin/factory/*) — shells to the monorepo factory CLI
    # across the wheel boundary; no-op where the factory isn't present.
    try:
        from adk.admin_factory import register_admin_factory_routes
        register_admin_factory_routes(app, state=_state)
    except ImportError as exc:
        logger.warning("factory bridge unavailable: %s", exc)

    # ─── No-backend handler ───

    @app.get("/demo")
    async def demo_redirect():
        """Redirect to demo.aitherium.com when no local backend is available."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse("https://demo.aitherium.com")

    @app.exception_handler(ConnectionError)
    async def _no_backend_handler(request: Request, exc: ConnectionError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "no_backend",
                "message": "No LLM backend available. Set AITHER_API_KEY to use the gateway, or install Ollama locally.",
                "demo": "https://demo.aitherium.com",
                "gateway": "https://gateway.aitherium.com",
                "docs": "https://github.com/Aitherium/aither-adk/blob/main/docs/GETTING_STARTED.md",
            },
        )

    # ─── Fleet endpoints ───

    @app.get("/agents")
    async def list_agents_endpoint():
        """List all agents in the fleet (or the single agent)."""
        if is_fleet:
            fleet = await _init_fleet()
            return {
                "fleet": fleet.name,
                "orchestrator": fleet.orchestrator_name,
                "agents": fleet.registry.list(),
            }
        a = await get_agent()
        return {
            "fleet": None,
            "orchestrator": a.name,
            "agents": [{
                "name": a.name,
                "identity": a._identity.name,
                "description": a._identity.description,
                "skills": a._identity.skills,
                "tools": [t.name for t in a._tools.list_tools()],
                "status": "running",
            }],
        }

    @app.post("/agents/{agent_name}/chat")
    async def agent_chat(agent_name: str, request: Request):
        """Chat with a specific agent in the fleet."""
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id")
        request_id = get_trace_id()

        a = await get_agent(agent_name)
        start = time.time()
        resp = await a.chat(message, session_id=session_id)
        latency_ms = (time.time() - start) * 1000

        # Record metrics (safe — latency_ms may be MagicMock in tests)
        try:
            _metrics = get_metrics()
            _metrics.record_request(latency_ms=latency_ms, status_code=200)
            _metrics.record_llm_call(
                model=str(resp.model or ""), latency_ms=float(resp.latency_ms or 0),
                tokens=int(resp.tokens_used or 0),
            )
        except (TypeError, ValueError):
            pass

        # Fire-and-forget Strata ingest
        asyncio.ensure_future(_strata_ingest(
            agent=a.name, session_id=resp.session_id,
            user_message=message, assistant_response=resp.content,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
        ))

        # Fire-and-forget Chronicle log
        asyncio.ensure_future(_chronicle_log_chat(
            agent=a.name, session_id=resp.session_id,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, request_id=request_id,
        ))

        return {
            "response": resp.content,
            "agent": a.name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
            "artifacts": resp.artifacts,
            "request_id": request_id,
        }

    @app.get("/agents/{agent_name}/sessions")
    async def agent_sessions(agent_name: str):
        """List conversation sessions for an agent."""
        from adk.conversations import get_conversation_store
        store = get_conversation_store()
        sessions = await store.list_sessions(agent_name=agent_name)
        return {"agent": agent_name, "sessions": sessions}

    @app.post("/agent/packs/reload")
    async def reload_agent_packs():
        """Hot-reload agent packs without restarting the process.

        Rediscovers installed packs and rebuilds the ToolRegistry to pick up
        any new skills or tools that were installed. Called when pack.applied
        Flux events arrive, or manually via the endpoint.
        """
        try:
            # Get the current agent
            agent = await get_agent()

            original_tool_count = len(agent._tools.list_tools())
            logger.info("Pack reload: re-registering discovered packs for agent %s", agent.name)

            # Actually rebuild: re-run pack discovery + tool-pack registration so
            # newly-installed packs' tools land in agent._tools. _load_discovered_packs
            # scans the discovery dirs and registers licensed packs into the live
            # registry; fall back to register_tool_packs directly if unavailable.
            reloaded = False
            if hasattr(agent, "_load_discovered_packs"):
                agent._load_discovered_packs()
                reloaded = True
            else:
                try:
                    from adk.builtin_tools import register_tool_packs
                    register_tool_packs(agent)
                    reloaded = True
                except ImportError:
                    logger.warning("register_tool_packs unavailable — reload is a no-op")

            new_tool_count = len(agent._tools.list_tools())

            return {
                "status": "reloaded" if reloaded else "noop",
                "agent": agent.name,
                "tools_before": original_tool_count,
                "tools_after": new_tool_count,
                "tools_added": max(0, new_tool_count - original_tool_count),
                "message": "Packs reloaded" if reloaded else "No reload mechanism available",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Pack reload failed: %s", e)
            return {
                "status": "failed",
                "error": str(e),
                "message": "Pack reload encountered an error",
            }

    @app.post("/forge/dispatch")
    async def forge_dispatch(request: Request):
        """Dispatch a task via AgentForge."""
        from adk.forge import ForgeSpec, get_forge
        body = await request.json()
        spec = ForgeSpec(
            agent_type=body.get("agent", body.get("agent_type", "auto")),
            task=body.get("task", body.get("message", "")),
            timeout=body.get("timeout", 120.0),
            effort=body.get("effort", 5),
            context=body.get("context", ""),
        )
        forge = get_forge()
        result = await forge.dispatch(spec)
        return {
            "content": result.content,
            "agent": result.agent,
            "tokens_used": result.tokens_used,
            "tool_calls": result.tool_calls,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }

    @app.post("/v1/forge/dispatch")
    async def v1_forge_dispatch(request: Request):
        """Accept forge dispatch from sovereign AitherOS nodes (canonical path)."""
        return await forge_dispatch(request)

    # ─── Genesis-compatible chat ───

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id")
        agent_name = body.get("agent")
        request_id = get_trace_id()

        # Inference controls (null=auto pattern — only pass if explicitly set)
        chat_kwargs: dict[str, Any] = {}
        if body.get("effort") is not None:
            chat_kwargs["effort"] = int(body["effort"])
        if body.get("temperature") is not None:
            chat_kwargs["temperature"] = float(body["temperature"])
        if body.get("top_p") is not None:
            chat_kwargs["top_p"] = float(body["top_p"])
        if body.get("repetition_penalty") is not None:
            chat_kwargs["repetition_penalty"] = float(body["repetition_penalty"])
        if body.get("max_tokens") is not None:
            chat_kwargs["max_tokens"] = int(body["max_tokens"])
        if body.get("model") is not None:
            chat_kwargs["model"] = body["model"]
        if body.get("tool_choice") is not None:
            chat_kwargs["tool_choice"] = body["tool_choice"]

        a = await get_agent(agent_name)
        start = time.time()
        resp = await a.chat(message, session_id=session_id, **chat_kwargs)
        latency_ms = (time.time() - start) * 1000

        # Record metrics (safe — latency_ms may be MagicMock in tests)
        try:
            _metrics = get_metrics()
            _metrics.record_request(latency_ms=latency_ms, status_code=200)
            _metrics.record_llm_call(
                model=str(resp.model or ""), latency_ms=float(resp.latency_ms or 0),
                tokens=int(resp.tokens_used or 0),
            )
        except (TypeError, ValueError):
            pass

        # Fire-and-forget Strata ingest (training loop)
        asyncio.ensure_future(_strata_ingest(
            agent=a.name, session_id=resp.session_id,
            user_message=message, assistant_response=resp.content,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
        ))

        # Fire-and-forget Chronicle log
        asyncio.ensure_future(_chronicle_log_chat(
            agent=a.name, session_id=resp.session_id,
            model=resp.model, tokens_used=resp.tokens_used,
            latency_ms=resp.latency_ms, request_id=request_id,
        ))

        return {
            "response": resp.content,
            "agent": a.name,
            "model": resp.model,
            "tokens_used": resp.tokens_used,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "session_id": resp.session_id,
            "tool_calls": resp.tool_calls_made,
            "artifacts": resp.artifacts,
            "request_id": request_id,
            "finish_reason": resp.finish_reason,
            "effort_level": resp.effort_level,
            "cache_status": resp.cache_status,
        }

    # ─── AitherOS-typed SSE streaming ───

    @app.post("/stream")
    async def stream_chat(request: Request):
        """SSE stream using AitherOS event protocol.

        Emits typed events: session_start, thinking, tool_call, tool_result,
        token, answer, complete — matching the Genesis/MicroScheduler protocol
        so shell-core's useAitherStream works identically against ADK and Genesis.
        """
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id") or f"adk-{uuid.uuid4().hex[:8]}"
        agent_name = body.get("agent")
        reasoning = body.get("reasoning", False)
        mcp_endpoints = body.get("mcp_endpoints")

        return StreamingResponse(
            _aitheros_stream(get_agent, message, session_id, agent_name, reasoning, mcp_endpoints),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/chat/stream")
    async def stream_chat_genesis_compat(request: Request):
        """Genesis-compatible SSE stream — alias for /stream.

        AitherShell sends POST /chat/stream with {message, persona, ...}.
        Maps the Genesis body shape to the ADK handler so `adk shell`
        and any AitherShell pointing at an ADK server works out of the box.
        """
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        session_id = body.get("session_id") or f"adk-{uuid.uuid4().hex[:8]}"
        agent_name = body.get("agent") or body.get("persona")
        reasoning = body.get("reasoning", False)
        mcp_endpoints = body.get("mcp_endpoints")

        # Optional per-turn generation params (honored by the OpenAI-compatible
        # providers). Only forward what the caller actually set + what is valid,
        # so a UI's settings drawer really takes effect instead of being decorative.
        gen_params: dict[str, Any] = {}
        try:
            if body.get("temperature") is not None:
                gen_params["temperature"] = float(body["temperature"])
            if body.get("max_tokens") is not None:
                gen_params["max_tokens"] = int(body["max_tokens"])
        except (TypeError, ValueError):
            gen_params = {}

        return StreamingResponse(
            _aitheros_stream(get_agent, message, session_id, agent_name, reasoning,
                             mcp_endpoints, gen_params=gen_params or None),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/aeon/stream")
    async def stream_aeon_group_chat(request: Request):
        """Aeon group-chat SSE stream — multi-agent discussion.

        Body: {message, preset?, agents?, rounds?, session_id?, temperature?, max_tokens?}
        Response: SSE with agent_message events per participant + synthesis.
        """
        body = await request.json()
        message = body.get("message", "")
        preset = body.get("preset", "balanced")
        agents = body.get("agents")
        rounds = body.get("rounds", 1)
        session_id = body.get("session_id") or f"aeon-{uuid.uuid4().hex[:8]}"

        # NOTE: no per-turn temperature/max_tokens here — AeonSession.chat() runs a
        # multi-agent round and does not thread per-call generation params, so
        # accepting them would be a silent no-op. Omitted deliberately.

        return StreamingResponse(
            _aeon_stream(message, preset, agents, rounds, session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ─── Artifact endpoints ───

    @app.get("/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str):
        """Get artifact metadata by ID."""
        from adk.artifacts import get_registry
        art = get_registry().get_by_id(artifact_id)
        if not art:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return art.to_dict()

    @app.get("/sessions/{session_id}/artifacts")
    async def get_session_artifacts(session_id: str):
        """List artifacts produced in a session."""
        from adk.artifacts import get_registry
        arts = get_registry().get(session_id)
        return {"session_id": session_id, "artifacts": [a.to_dict() for a in arts]}

    @app.post("/sessions/{session_id}/confirm")
    async def confirm_tools(session_id: str, request: Request):
        """Resume a turn paused for tool approval (human-in-the-loop).

        Body: ``{decisions: [{tool_use_id|tool, result: "allow"|"deny", deny_message?}]}``.
        Records the decisions then re-runs the paused turn, streaming the continuation as
        the same SSE trace (``session_start`` → … → ``complete``). 409 if nothing paused."""
        from adk.approval import get_approval_store
        body = await request.json()
        decisions = body.get("decisions", []) or []
        store = get_approval_store()
        paused = store.get(session_id)
        if not paused:
            return JSONResponse(
                {"error": "no_paused_turn",
                 "message": "no turn is awaiting approval for this session"},
                status_code=409,
            )
        store.record_decisions(session_id, decisions)
        message = paused.get("user_message", "")
        agent_name = paused.get("agent")
        return StreamingResponse(
            _aitheros_stream(get_agent, message, session_id, agent_name, False),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ─── OpenAI-compatible endpoints ───

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages_raw = body.get("messages", [])
        model = body.get("model")
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 4096)
        stream = body.get("stream", False)

        a = await get_agent()

        # Convert to Message objects
        messages = [Message(role=m["role"], content=m.get("content", "")) for m in messages_raw]

        if stream:
            # Extract last user message for agent.chat_stream()
            last_user_msg = ""
            history_for_stream = []
            for m in messages_raw:
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                if m.get("role") in ("user", "assistant"):
                    history_for_stream.append({"role": m["role"], "content": m.get("content", "")})
            # Remove last user message from history (chat_stream takes it separately)
            if history_for_stream and history_for_stream[-1]["role"] == "user":
                history_for_stream = history_for_stream[:-1]

            return StreamingResponse(
                _stream_agent_response(a, last_user_msg, history_for_stream, model),
                media_type="text/event-stream",
            )

        resp = await a.llm.chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.content},
                    "finish_reason": resp.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.tokens_used,
            },
        }

    @app.get("/v1/models")
    async def list_models_endpoint():
        a = await get_agent()
        try:
            models = await a.llm.list_models()
        except (RuntimeError, OSError, ConnectionError):
            models = []

        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
                for m in models
            ],
        }

    @app.get("/v1/identities")
    async def list_identities_endpoint():
        """List available agent identities."""
        return {"identities": list_identities()}

    # ─── Strata ingest helper (fire-and-forget) ───

    async def _strata_ingest(**kwargs):
        """Send chat data to Strata for training/analytics. Never blocks or raises."""
        try:
            from adk.strata import get_strata_ingest
            strata = get_strata_ingest()
            await strata.ingest_chat(**kwargs)
        except (ImportError, RuntimeError, OSError):
            pass  # Truly fire-and-forget

    # ─── MCP server (every node is also an MCP server) ───

    async def _init_mcp_server():
        """Initialize the MCP server so this node SERVES tools, not just consumes them."""
        try:
            from adk.mcp_server import MCPServer

            a = _state.get("agent")
            if a is None and not is_fleet:
                a = AitherAgent(
                    name=_state["identity"],
                    identity=_state["identity"],
                    config=_state["config"],
                )
                _state["agent"] = a

            # Build a merged registry: agent tools + fleet tools
            if is_fleet and _state.get("fleet"):
                from adk.tools import ToolRegistry
                merged = ToolRegistry()
                for fleet_agent in _state["fleet"].agents:
                    for td in fleet_agent._tools.list_tools():
                        # Prefix with agent name to avoid collisions
                        prefixed_name = f"{fleet_agent.name}__{td.name}"
                        merged._tools[prefixed_name] = td._replace(name=prefixed_name) if hasattr(td, '_replace') else td
                        # Also keep original name from first agent that has it
                        if td.name not in merged._tools:
                            merged._tools[td.name] = td
                mcp = MCPServer(tool_registry=merged, server_name=_state["fleet"].name)
            elif a:
                mcp = MCPServer(tool_registry=a._tools, server_name=a.name)
            else:
                mcp = MCPServer(server_name=_state["identity"])

            mcp.mount(app)
            _state["mcp_server"] = mcp

            # Wire relay → MCP server so inbound mesh tool calls are handled locally
            relay_obj = _state.get("relay")
            if relay_obj:
                relay_obj.set_local_mcp_server(mcp)

            logger.info("MCP server initialized (%d tools)", len(mcp.registry.list_tools()))
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("MCP server init failed (non-fatal): %s", exc)

    # ─── A2A protocol server (Google A2A v0.3.0) ───

    async def _init_a2a_server():
        """Initialize the A2A protocol server for cross-agent interop."""
        try:
            from adk.a2a import A2AServer

            a = _state.get("agent")
            base_url = f"http://localhost:{config.server_port}"

            a2a = A2AServer(
                agent=a,
                base_url=base_url,
                server_name=a.name if a else _state.get("identity", "adk-agent"),
            )
            a2a.mount(app)
            _state["a2a_server"] = a2a
            logger.info("A2A server initialized (protocol v0.3.0)")
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("A2A server init failed (non-fatal): %s", exc)

    async def _connect_service_bridge():
        """Connect ServiceBridge to discover AitherOS services (non-fatal)."""
        try:
            from adk.services import ServiceBridge
            bridge = ServiceBridge()
            status = await bridge.connect()
            _state["service_bridge"] = bridge

            if status.mode != "standalone":
                # Register MCP tools on fleet agents or single agent
                if is_fleet and _state["fleet"]:
                    for a in _state["fleet"].agents:
                        await bridge.register_on_agent(a)
                elif _state["agent"]:
                    await bridge.register_on_agent(_state["agent"])
            else:
                # Visible warning when AitherOS is not detected
                import sys
                agent = _state.get("agent")
                builtin_count = len(agent._tools.list_tools()) if agent else 0
                print(
                    "\n\033[33m\u26a0  STANDALONE MODE \u2014 AitherOS not detected\033[0m\n"
                    "   AitherNode (localhost:8080) and Genesis (localhost:8001) "
                    "are unreachable.\n"
                    f"   Only {builtin_count} built-in tools available "
                    f"(vs 449+ with AitherOS).\n"
                    "   Start AitherOS or set AITHER_NODE_URL to connect.\n",
                    file=sys.stderr,
                )
                # Start background reconnect so we auto-upgrade when
                # AitherOS services come online
                await bridge.start_background_reconnect()

            logger.info("ServiceBridge mode: %s (tools: %d)",
                        status.mode, status.tools_count)
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("ServiceBridge startup failed (non-fatal): %s", exc)

    async def _flush_strata_queue():
        """Flush any queued Strata entries from previous sessions."""
        try:
            from adk.strata import get_strata_ingest
            strata = get_strata_ingest()
            flushed = await strata.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Strata entries", flushed)
        except (ImportError, RuntimeError, OSError):
            pass

    async def _chronicle_log_chat(**kwargs):
        """Send chat event to Chronicle. Never blocks or raises."""
        try:
            import inspect
            from adk.chronicle import get_chronicle
            chronicle = get_chronicle()
            # Drop kwargs log_llm_call doesn't accept (e.g. session_id) — callers
            # pass a richer set; signature drift must not crash this fire-and-forget.
            accepted = set(inspect.signature(chronicle.log_llm_call).parameters)
            await chronicle.log_llm_call(**{k: v for k, v in kwargs.items() if k in accepted})
        except Exception:
            pass  # Truly fire-and-forget

    async def _flush_chronicle_queue():
        """Flush any queued Chronicle entries from previous sessions."""
        try:
            from adk.chronicle import get_chronicle
            chronicle = get_chronicle()
            flushed = await chronicle.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Chronicle entries", flushed)
        except (ImportError, RuntimeError, OSError):
            pass

    async def _start_watch_reporter():
        """Start the background Watch health reporter."""
        try:
            from adk.watch import get_watch_reporter
            reporter = get_watch_reporter()

            # Register a collector that reports fleet/agent state
            def _collect_health():
                data = {"version": __version__}
                try:
                    if is_fleet and _state["fleet"]:
                        fleet = _state["fleet"]
                        data["agents"] = fleet.registry.agent_names
                        data["agent_count"] = len(fleet.agents)
                    elif _state["agent"]:
                        data["agents"] = [_state["agent"].name]
                        data["agent_count"] = 1
                except (KeyError, AttributeError):
                    pass
                return data

            reporter.register_collector(_collect_health)
            await reporter.start()
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("Watch reporter startup failed (non-fatal): %s", exc)

    async def _flush_pulse_queue():
        """Flush any queued Pulse pain signals from previous sessions."""
        try:
            from adk.pulse import get_pulse
            pulse = get_pulse()
            flushed = await pulse.flush_queue()
            if flushed:
                logger.info("Flushed %d queued Pulse pain signals", flushed)
        except (ImportError, RuntimeError, OSError):
            pass

    async def _register_fleet_endpoint():
        """Register invoke_url with portal fleet so sovereign can dispatch to us."""
        from adk.config import load_saved_config
        saved = load_saved_config()
        api_key = saved.get("api_key", "") or config.aither_api_key
        if not api_key:
            logger.debug("Fleet endpoint registration skipped (no API key)")
            return
        # Determine reach mode and invoke_url (tunnel vs mesh overlay)
        reach_mode = "tunnel"
        invoke_url = os.getenv("AITHER_INVOKE_URL", "")
        if not invoke_url:
            # Check for mesh mode (overlay IP registration)
            mesh_overlay_ip = os.getenv("AITHER_MESH_OVERLAY_IP", "").strip()
            if mesh_overlay_ip:
                reach_mode = "mesh"
                invoke_url = f"http://{mesh_overlay_ip}:{config.server_port}"
            else:
                invoke_url = f"http://localhost:{config.server_port}"
        tenant_id = saved.get("tenant_id", "") or os.getenv("AITHER_TENANT_ID", "")
        agent_name = _state.get("identity", identity)
        try:
            portal_url = os.getenv("AITHER_PORTAL_URL", "https://portal.aitherium.com")
            import httpx
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(
                    f"{portal_url}/v1/agents/upsert",
                    json={
                        "name": agent_name,
                        "scope": {"visibility": "workspace", "tenant_id": tenant_id},
                        "invoke_url": invoke_url,
                        "reach": reach_mode,
                        "status": "online",
                        # Advertise what inference this agent runs, so discovery
                        # (`adk agents ls`) shows "optiplex → bonsai (llamacpp)".
                        "model": getattr(config, "model", "") or "",
                        "provider_hint": getattr(config, "llm_backend", "") or "",
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            _state["fleet_registered"] = True
            logger.info("Registered fleet endpoint: %s -> %s", agent_name, invoke_url)
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            _state["fleet_registered"] = False
            logger.warning("Fleet endpoint registration failed (non-fatal): %s", exc)

        # Register with new fleet API (separate try/except for resilience)
        workspace_id = saved.get("workspace_id", "") or os.getenv("AITHER_WORKSPACE_ID", "")
        try:
            portal_url = os.getenv("AITHER_PORTAL_URL", "https://portal.aitherium.com")
            import httpx
            billing_email = saved.get("billing_email", "")
            async with httpx.AsyncClient(timeout=15) as client:
                fleet_resp = await client.post(
                    f"{portal_url}/api/fleet/endpoints/register",
                    json={
                        "name": agent_name,
                        "url": invoke_url,
                        "reach": reach_mode,
                        "agent_type": "adk-agent",
                        "capabilities": ["chat", "tools"],
                        "billing_email": billing_email,
                        # Inference backend advertised for mesh discovery.
                        "model": getattr(config, "model", "") or "",
                        "provider_hint": getattr(config, "llm_backend", "") or "",
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-Tenant-ID": tenant_id,
                        "X-Workspace-ID": workspace_id,
                    },
                )
                if fleet_resp.status_code == 200:
                    data = fleet_resp.json()
                    _state["fleet_endpoint_id"] = data.get("endpoint_id", "")
                    _state["fleet_api_key"] = data.get("api_key", "")
                    logger.info("Registered fleet API endpoint: %s (id=%s)", agent_name, _state.get("fleet_endpoint_id"))
                else:
                    logger.debug("Fleet API registration returned %d", fleet_resp.status_code)
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("Fleet API registration failed (non-fatal): %s", exc)

        # Register with AitherFleet service (central roster for cycle dispatch)
        try:
            fleet_svc_url = os.getenv("AITHER_FLEET_URL", "http://localhost:8162")
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"{fleet_svc_url}/fleet/agents",
                    json={
                        "name": agent_name,
                        "visibility": "tenant",
                        "status": "active",
                        "card": {
                            "agent_type": "adk-agent",
                            "capabilities": ["chat", "tools", "forge"],
                            "invoke_url": invoke_url,
                            "reach": reach_mode,
                        },
                    },
                    headers={
                        "X-Tenant-ID": tenant_id,
                        "X-Workspace-ID": workspace_id,
                    },
                )
            logger.info("Registered with AitherFleet service: %s", agent_name)
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("AitherFleet registration failed (non-fatal): %s", exc)

    async def _deregister_fleet_endpoint():
        """Mark agent as offline in portal fleet on shutdown."""
        if not _state.get("fleet_registered"):
            return
        from adk.config import load_saved_config
        saved = load_saved_config()
        api_key = saved.get("api_key", "") or config.aither_api_key
        if not api_key:
            return
        agent_name = _state.get("identity", identity)
        tenant_id = saved.get("tenant_id", "") or os.getenv("AITHER_TENANT_ID", "")
        try:
            portal_url = os.getenv("AITHER_PORTAL_URL", "https://portal.aitherium.com")
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"{portal_url}/v1/agents/upsert",
                    json={
                        "name": agent_name,
                        "scope": {"visibility": "workspace", "tenant_id": tenant_id},
                        "status": "offline",
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            logger.info("Deregistered fleet endpoint: %s", agent_name)
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("Fleet endpoint deregistration failed (non-fatal): %s", exc)

        # Deregister from fleet API (separate try/except for resilience)
        fleet_endpoint_id = _state.get("fleet_endpoint_id")
        if fleet_endpoint_id:
            try:
                portal_url = os.getenv("AITHER_PORTAL_URL", "https://portal.aitherium.com")
                import httpx
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.delete(
                        f"{portal_url}/api/fleet/endpoints/{fleet_endpoint_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                logger.info("Deregistered fleet API endpoint: %s (id=%s)", agent_name, fleet_endpoint_id)
            except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
                logger.debug("Fleet API deregistration failed (non-fatal): %s", exc)

    async def _fleet_heartbeat_loop():
        """Continuously heartbeat to portal every 60s using FederationLiteClient.

        Reports: status, CPU/memory, active agents, inference backend, tokens processed.
        On failure, queues locally and retries next cycle.
        """
        from adk.config import load_saved_config
        saved = load_saved_config()
        api_key = saved.get("api_key", "") or config.aither_api_key
        if not api_key:
            logger.debug("Fleet heartbeat loop skipped (no API key)")
            return

        tenant_id = saved.get("tenant_id", "") or os.environ.get("AITHER_TENANT_ID", "")
        portal_url = os.environ.get("AITHER_PORTAL_URL", "https://portal.aitherium.com")
        agent_name = _state.get("identity", identity)
        instance_id = os.environ.get("AITHER_INSTANCE_ID", "")
        invoke_url = os.environ.get("AITHER_INVOKE_URL", f"http://localhost:{config.server_port}")

        try:
            from adk.federation_lite import FederationLiteClient
            fed_client = FederationLiteClient(
                hub_url=portal_url,
                api_key=api_key,
                node_id=instance_id or agent_name,
            )
        except ImportError:
            fed_client = None

        consecutive_failures = 0

        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break

            try:
                # Collect metrics
                metrics_data = {}
                try:
                    m = get_metrics()
                    metrics_data = {
                        "tokens_processed": getattr(m, "tokens_processed", 0),
                        "requests_total": getattr(m, "requests_total", 0),
                        "uptime_seconds": getattr(m, "uptime_seconds", 0),
                    }
                except (RuntimeError, AttributeError):
                    pass

                # System resource metrics
                try:
                    import psutil
                    metrics_data["cpu_percent"] = psutil.cpu_percent(interval=0)
                    mem = psutil.virtual_memory()
                    metrics_data["memory_percent"] = mem.percent
                except ImportError:
                    pass

                # Inference backend info
                metrics_data["inference_mode"] = config.cloud_mode or config.llm_backend
                metrics_data["invoke_url"] = invoke_url

                # Agent list
                agents_list = [{
                    "name": agent_name,
                    "invoke_url": invoke_url,
                    "status": "online",
                    "tenant_id": tenant_id,
                }]

                if fed_client:
                    result = await fed_client.heartbeat(
                        status="online",
                        metrics=metrics_data,
                        agents=agents_list,
                    )
                    if result.get("error"):
                        consecutive_failures += 1
                        logger.debug("Fleet heartbeat failed (%d): %s", consecutive_failures, result)
                    else:
                        consecutive_failures = 0
                        logger.debug("Fleet heartbeat OK")
                else:
                    # Fallback: direct HTTP heartbeat
                    import httpx
                    async with httpx.AsyncClient(timeout=15) as client:
                        await client.post(
                            f"{portal_url}/v1/agents/upsert",
                            json={
                                "name": agent_name,
                                "scope": {"visibility": "workspace", "tenant_id": tenant_id},
                                "invoke_url": invoke_url,
                                "status": "online",
                                "metrics": metrics_data,
                            },
                            headers={"Authorization": f"Bearer {api_key}"},
                        )
                    consecutive_failures = 0
                    logger.debug("Fleet heartbeat OK (direct)")

            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ConnectionError) as exc:
                consecutive_failures += 1
                if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                    logger.warning("Fleet heartbeat error (%d): %s", consecutive_failures, exc)

    async def _sync_secrets():
        """Pull secrets from platform vault into local ADK store on startup."""
        try:
            from adk.sync.secrets import sync_secrets
            synced = await sync_secrets()
            if synced:
                logger.info("Synced %d secrets from vault", len(synced))
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("Secrets sync failed (non-fatal): %s", exc)

    async def _connect_gateway_mcp():
        """Connect MCP client to gateway for platform tool access.

        Self-hosted agents use this to access platform tools (code search, memory,
        secrets, etc.) without running the full AitherOS stack locally.

        Authentication hierarchy:
          1. AITHER_API_KEY (device-flow token, ACTA key, or Identity bearer)
          2. Local AitherSecrets self-mint (fallback via fleet_enroll)

        Fail-soft: no token or connection failure logs warning, agent runs offline.
        """
        if not config.aither_api_key:
            logger.debug("Gateway MCP: no API key configured (running offline)")
            return

        try:
            from adk.client._gateway_mcp import create_gateway_mcp_client

            gateway_url = config.gateway_url or os.getenv("AITHER_GATEWAY_URL", "")
            if not gateway_url:
                gateway_url = "https://mcp.aitherium.com"

            # Create and test connection (never crashes startup)
            mcp_client = await create_gateway_mcp_client(
                gateway_url=gateway_url,
                api_key=config.aither_api_key,
            )

            if mcp_client:
                _state["gateway_mcp_client"] = mcp_client
                _state["gateway_mcp_connected"] = True
                logger.info("Gateway MCP client connected: %s", gateway_url)

                # Register gateway tools in agent (best-effort, async)
                try:
                    a = await get_agent()
                    if a:
                        tools = await mcp_client.list_tools()
                        for tool_spec in tools:
                            # Create a closure for this tool
                            tool_name = tool_spec["name"]

                            async def _gateway_tool_call(
                                tn=tool_name, **kwargs
                            ) -> str:
                                result = await mcp_client.call_tool(tn, kwargs)
                                if result.get("success"):
                                    return result.get("text", "")
                                else:
                                    return f"Error: {result.get('message', 'unknown')}"

                            _gateway_tool_call.__name__ = tool_name
                            _gateway_tool_call.__doc__ = tool_spec.get("description", "")

                            a._tools.register(
                                _gateway_tool_call,
                                name=tool_name,
                                description=tool_spec.get("description", ""),
                            )
                        logger.info(
                            "Registered %d gateway tools with agent %s",
                            len(tools), a.name,
                        )
                except Exception as exc:  # noqa: BLE001 — tool registration is advisory
                    logger.debug("Gateway tool registration failed (non-fatal): %s", exc)
            else:
                _state["gateway_mcp_connected"] = False
                logger.debug("Gateway MCP connection failed (running offline)")
        except ImportError:
            logger.debug("Gateway MCP unavailable (continuing offline)")
        except Exception as exc:  # noqa: BLE001 — must never crash startup
            _state["gateway_mcp_connected"] = False
            logger.warning("Gateway MCP init failed (continuing offline): %s", exc)

    async def _register_with_gateway():
        if not config.gateway_url or not config.aither_api_key:
            logger.debug("Gateway auto-registration skipped (not configured)")
            return
        if not config.register_agent:
            logger.debug("Gateway auto-registration skipped (AITHER_REGISTER_AGENT not set)")
            _state["gateway_connected"] = False
            return
        try:
            from adk.client import GatewayClient
            gw = GatewayClient(gateway_url=config.gateway_url, api_key=config.aither_api_key)
            ident = load_identity(identity)

            # Get owner email for registration (required by gateway contract)
            owner_email = os.getenv("AITHER_OWNER_EMAIL", "").strip()
            if not owner_email:
                # Try to extract from saved config
                from adk.config import load_saved_config
                saved = load_saved_config()
                owner_email = saved.get("billing_email", "") or saved.get("email", "")

            if not owner_email:
                logger.warning(
                    "Gateway registration skipped: AITHER_OWNER_EMAIL not set "
                    "(required by gateway contract)"
                )
                _state["gateway_connected"] = False
                return

            result = await gw.register_agent(
                name=ident.name,
                owner_email=owner_email,
                description=ident.description,
                framework="adk",
            )
            _state["gateway_connected"] = True
            logger.info("Registered with gateway %s: %s", config.gateway_url, result)
        except Exception as exc:  # noqa: BLE001 — registration is ALWAYS best-effort;
            # a bad signature / gateway hiccup must never take down the agent server.
            _state["gateway_connected"] = False
            logger.warning("Gateway registration failed (non-fatal): %s", exc)

    async def _join_aithernet():
        """Auto-join the AitherNet mesh relay if API key is configured."""
        if not config.aither_api_key:
            return
        try:
            from adk.relay import get_relay
            agent_names = []
            if is_fleet and _state.get("fleet"):
                agent_names = [a.name for a in _state["fleet"].agents]
            elif _state.get("agent"):
                agent_names = [_state["agent"].name]

            relay = get_relay(
                api_key=config.aither_api_key,
                gateway_url=config.gateway_url or "",
                node_name=os.getenv("AITHER_NODE_NAME", ""),
                agents=agent_names,
                capabilities=_detect_node_capabilities(),
                port=config.server_port,
            )
            result = await relay.register()
            if result.get("ok") is not False:
                _state["relay"] = relay
                await relay.start_heartbeat(interval=60)
                logger.info(
                    "Joined AitherNet mesh as %s (node_id=%s, agents=%s)",
                    relay.node_name, relay.node_id[:12], agent_names,
                )
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("AitherNet join failed (non-fatal): %s", exc)

    async def _rich_enroll_identity():
        """Register with AitherIdentity's rich node spine (separate from the AitherNet
        mesh relay joined by _join_aithernet, above — that's presence/messaging; this
        is identity + capability-token issuance). On success, mints a tenant-scoped
        bearer_token and self-services a real avk_... gateway key via AitherSecrets,
        replacing the node's reliance on the user's own access token for follow-up
        calls. Non-fatal: enrollment.rich_enroll() never raises.
        """
        if not config.aither_api_key:
            return
        try:
            import platform
            from adk import enrollment
            from adk import fleet_enroll

            idp_url = os.getenv("AITHER_IDP_URL", os.getenv("AITHER_IDP_BASE_URL", "https://idp.aitherium.com"))
            relay = _state.get("relay")
            node_id = relay.node_id if relay else os.getenv("AITHER_NODE_NAME", "") or platform.node()

            result = await enrollment.rich_enroll(
                idp_url, config.aither_api_key, node_id, enable_heartbeat=True,
            )
            if not result.get("enrolled"):
                logger.debug("Identity enrollment skipped/failed (non-fatal): %s", result.get("error"))
                return

            logger.info(
                "Identity-enrolled with AitherNet as %s (tenant=%s)",
                node_id[:12], result.get("tenant_id", ""),
            )

            bearer_token = result.get("bearer_token", "")
            if bearer_token:
                await fleet_enroll._self_mint_gateway_key(bearer_token, node_id)
        except (ImportError, RuntimeError, OSError, ConnectionError, httpx.HTTPError) as exc:
            logger.debug("Identity enrollment failed (non-fatal): %s", exc)

    def _detect_node_capabilities() -> list[str]:
        """Detect what this node can do."""
        caps = ["chat", "tools", "mcp", "a2a", "irc", "smtp"]
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                caps.append("inference")
                caps.append("gpu")
        except (FileNotFoundError, OSError):
            pass
        if is_fleet:
            caps.append("fleet")
        return caps

    # ─── Elysium reconnect + mesh hosting ───

    async def _reconnect_elysium():
        """Re-join desktop mesh on startup if previously connected via `adk connect --elysium`."""
        from adk.config import load_saved_config  # noqa: F811
        try:
            saved = load_saved_config()
            elysium_url = saved.get("elysium_url", "")
            if not elysium_url:
                return

            node_token = saved.get("node_token", "")
            mesh_url = saved.get("mesh_url", "")

            # Set env vars for LLM router dual-mode
            core_llm = saved.get("core_llm_url", "")
            if core_llm:
                os.environ.setdefault("AITHER_CORE_LLM_URL", core_llm)
            if node_token:
                os.environ.setdefault("AITHER_NODE_TOKEN", node_token)

            # Re-join mesh
            if mesh_url:
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            f"{mesh_url}/heartbeat",
                            json={"node_id": saved.get("node_id", ""), "status": "online"},
                            headers={
                                "Authorization": f"Bearer {node_token}" if node_token else "",
                                "Content-Type": "application/json",
                            },
                        )
                    logger.info("Reconnected to desktop mesh at %s", mesh_url)
                except (httpx.HTTPError, OSError) as e:
                    logger.debug("Desktop mesh reconnect failed (non-fatal): %s", e)

        except (OSError, ValueError) as e:
            logger.debug("Elysium reconnect skipped: %s", e)

    async def _start_mesh_hosting():
        """Start mesh hosting if --mesh flag or config mesh_enabled is set."""
        from adk.config import load_saved_config  # noqa: F811
        mesh_enabled = os.getenv("AITHER_MESH_ENABLED", "").lower() in ("true", "1", "yes")
        if not mesh_enabled:
            try:
                saved = load_saved_config()
                mesh_enabled = saved.get("mesh_enabled", False)
            except (OSError, ValueError):
                pass

        if not mesh_enabled:
            return

        try:
            from adk.relay import AitherNetRelay  # noqa: F401

            saved = load_saved_config()
            base_host = saved.get("elysium_base_host", "")
            node_token = saved.get("node_token", "")

            # Create relay pointed at desktop (not cloud gateway)
            relay_kwargs = {
                "node_name": os.getenv("AITHER_NODE_NAME", ""),
                "capabilities": _detect_node_capabilities(),
                "port": config.server_port,
            }
            if base_host:
                relay_kwargs["gateway_url"] = f"{base_host}:8001"
            if node_token:
                relay_kwargs["api_key"] = node_token

            relay = AitherNetRelay(**relay_kwargs)
            result = await relay.register()

            if result.get("ok") is not False:
                # Wire MCP server for inbound tool calls
                mcp = _state.get("mcp_server")
                if mcp:
                    relay.set_local_mcp_server(mcp)

                await relay.start_heartbeat(interval=60)
                await relay.connect_relay_hub()
                _state["elysium_relay"] = relay
                logger.info(
                    "Mesh hosting active: node=%s, capabilities=%s",
                    relay.node_id[:12], relay.capabilities,
                )
        except (ImportError, OSError, ConnectionError, ValueError) as e:
            logger.debug("Mesh hosting startup failed (non-fatal): %s", e)

    # ─── Chat relay startup + endpoints ───

    async def _init_chat_relay():
        """Initialize the chat relay and wire federation handlers."""
        try:
            from adk.chat import get_chat_relay
            relay_obj = _state.get("relay")
            node_id = relay_obj.node_id if relay_obj else ""
            chat = get_chat_relay(node_id=node_id)
            _state["chat_relay"] = chat

            # Register agents as chat participants
            if is_fleet and _state.get("fleet"):
                for a in _state["fleet"].agents:
                    chat.register_agent(a.name)
            elif _state.get("agent"):
                chat.register_agent(_state["agent"].name)

            # Wire federation: relay mesh "chat" messages → local chat
            if relay_obj:
                relay_obj.on("chat", chat.handle_federated_message)
                relay_obj.on("mail", lambda data: _handle_mesh_mail(data))

            # Start raw IRC protocol server (opt-in via AITHER_IRC_PORT)
            irc_port_env = os.getenv("AITHER_IRC_PORT", "")
            if irc_port_env:
                try:
                    irc_port = int(irc_port_env)
                    await chat.start_irc_server(port=irc_port)
                    logger.info("IRC server listening on port %d", irc_port)
                except (RuntimeError, OSError) as irc_exc:
                    logger.debug("IRC server startup failed (non-fatal): %s", irc_exc)

                # Start Aither bridge only when IRC is enabled
                try:
                    from adk.aither_bridge import init_aither_bridge
                    bridge = await init_aither_bridge(chat)
                    if bridge:
                        _state["aither_bridge"] = bridge
                        logger.info("Aither bridge active")
                except (ImportError, RuntimeError, OSError) as bridge_exc:
                    logger.debug("Aither bridge startup failed (non-fatal): %s", bridge_exc)

            logger.info("Chat relay initialized (channels=%d)", len(chat._channels))
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("Chat relay init failed (non-fatal): %s", exc)

    async def _init_mail_relay():
        """Initialize the mail relay."""
        try:
            from adk.smtp import get_mail_relay
            relay_obj = _state.get("relay")
            node_id = relay_obj.node_id if relay_obj else ""
            mail = get_mail_relay(node_id=node_id)
            _state["mail_relay"] = mail

            # Auto-provision mailboxes for agents
            if is_fleet and _state.get("fleet"):
                for a in _state["fleet"].agents:
                    mail.provision_mailbox(a.name)
            elif _state.get("agent"):
                mail.provision_mailbox(_state["agent"].name)

            # Start inbound SMTP listener (non-fatal)
            smtp_port = int(os.getenv("AITHER_SMTP_PORT", "2525"))
            try:
                started = await mail.start_inbound_server(port=smtp_port)
                if started:
                    logger.info("Inbound SMTP server started on port %d", smtp_port)
            except (RuntimeError, OSError) as smtp_exc:
                logger.debug("Inbound SMTP server startup failed (non-fatal): %s", smtp_exc)

            logger.info("Mail relay initialized (configured=%s)", mail.is_configured)
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("Mail relay init failed (non-fatal): %s", exc)

    async def _init_relay_client():
        """Initialize ChatRelayClient so this node can be a first-class relay
        participant. Requires RELAY_URL and AITHER_BEARER env vars.
        Registers agents as mention handlers that dispatch to agent.chat().
        """
        relay_url = os.getenv("RELAY_URL", "")
        aither_bearer = os.getenv("AITHER_BEARER", "")
        if not relay_url or not aither_bearer:
            logger.debug(
                "Relay client skipped (RELAY_URL or AITHER_BEARER not set)"
            )
            return

        try:
            from adk.relay_client import get_relay_client

            workspace_id = os.getenv("RELAY_WORKSPACE_ID", "")
            client = get_relay_client(
                base_url=relay_url,
                aither_bearer=aither_bearer,
                workspace_id=workspace_id or None,
                user_id=_state.get("identity", identity),
                nick=_state.get("identity", identity),
            )

            # Register mention handlers to dispatch to agents
            if is_fleet and _state.get("fleet"):
                for a in _state["fleet"].agents:

                    def _make_handler(agent):
                        async def handle(msg):
                            try:
                                # Strip @mentions for natural input
                                txt = msg.content
                                for nick in (ag.name for ag in _state["fleet"].agents):
                                    txt = txt.replace(f"@{nick}", "").strip()
                                resp = await agent.chat(
                                    txt,
                                    session_id=f"relay:{msg.channel}:{msg.nick}",
                                )
                                return resp.content
                            except Exception as exc:
                                logger.error(
                                    "Mention handler error for @%s: %s",
                                    agent.name, exc,
                                )
                                return None

                        return handle

                    client.register_mention_handler(a.name, _make_handler(a))
            elif _state.get("agent"):
                a = _state["agent"]

                async def _default_mention_handler(msg):
                    try:
                        # Strip @nick from message
                        txt = msg.content.replace(f"@{a.name}", "").strip()
                        resp = await a.chat(
                            txt,
                            session_id=f"relay:{msg.channel}:{msg.nick}",
                        )
                        return resp.content
                    except Exception as exc:
                        logger.error(
                            "Mention handler error for @%s: %s",
                            a.name, exc,
                        )
                        return None

                client.register_mention_handler(a.name, _default_mention_handler)

            # Connect and store on app state
            if await client.connect():
                _state["relay_client"] = client
                logger.info("Relay client connected as %s", client.nick)
            else:
                logger.warning("Relay client connection failed (non-fatal)")
        except (ImportError, RuntimeError, OSError, ConnectionError) as exc:
            logger.debug("Relay client init failed (non-fatal): %s", exc)

    def _handle_mesh_mail(data: dict):
        """Handle incoming mail from the mesh relay."""
        try:
            mail = _state.get("mail_relay")
            if mail:
                mail.receive_mesh_mail(data)
        except (RuntimeError, OSError) as exc:
            logger.debug("Mesh mail handler error: %s", exc)

    # ── Chat WebSocket ──

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        """WebSocket endpoint for real-time chat (IRC-compatible)."""
        chat = _state.get("chat_relay")
        if not chat:
            await websocket.close(code=4000, reason="Chat relay not initialized")
            return

        await websocket.accept()
        nick = f"user_{uuid.uuid4().hex[:6]}"

        try:
            # Wait for join message with nick
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            if init_data.get("type") == "join":
                nick = init_data.get("nick", nick)
                channel = init_data.get("channel", "#general")
            else:
                channel = "#general"

            chat.connect_ws(nick, websocket)
            chat.join(channel, nick)

            # Send channel history
            history = chat.history(channel, limit=50)
            await websocket.send_json({"type": "history", "channel": channel, "messages": history})

            # Message loop
            while True:
                data = await websocket.receive_json()
                await chat.handle_ws_message(nick, data)

        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            pass
        except (RuntimeError, OSError, ConnectionError) as exc:
            logger.debug("WebSocket chat error: %s", exc)
        finally:
            chat.disconnect_ws(nick)
            # Part all channels
            user = chat._users.get(nick)
            if user:
                for ch in list(user.channels):
                    chat.part(ch, nick)

    # ── Chat REST endpoints ──

    @app.get("/chat/channels")
    async def chat_channels():
        """List available chat channels."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"channels": []}
        return {"channels": chat.list_channels()}

    @app.get("/chat/channels/{channel}/history")
    async def chat_channel_history(channel: str, limit: int = 50):
        """Get message history for a channel."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"messages": []}
        ch = f"#{channel}" if not channel.startswith("#") else channel
        return {"channel": ch, "messages": chat.history(ch, limit=limit)}

    @app.get("/chat/channels/{channel}/users")
    async def chat_channel_users(channel: str):
        """List users in a channel."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"users": []}
        ch = f"#{channel}" if not channel.startswith("#") else channel
        return {"channel": ch, "users": chat.who(ch)}

    @app.post("/chat/channels/{channel}/message")
    async def chat_post_message(channel: str, request: Request):
        """Post a message to a channel (REST alternative to WebSocket)."""
        chat = _state.get("chat_relay")
        if not chat:
            return JSONResponse({"error": "Chat relay not initialized"}, status_code=503)
        body = await request.json()
        nick = body.get("nick", body.get("from", "api"))
        content = body.get("content", body.get("message", ""))
        if not content:
            return JSONResponse({"error": "content is required"}, status_code=400)

        ch = f"#{channel}" if not channel.startswith("#") else channel
        msg = chat.post(ch, nick, content)

        # Federate to mesh
        if msg and _state.get("relay"):
            asyncio.ensure_future(chat.federate_message(msg))

        return {"ok": bool(msg), "msg_id": msg.msg_id if msg else None}

    @app.get("/chat/users")
    async def chat_online_users():
        """List all online users across channels."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"users": []}
        return {"users": chat.online_users()}

    @app.get("/chat/status")
    async def chat_status():
        """Chat relay status."""
        chat = _state.get("chat_relay")
        if not chat:
            return {"active": False, "message": "Chat relay not initialized"}
        return {**chat.status(), "active": True}

    @app.get("/bridge/status")
    async def bridge_status():
        """Aither ↔ IRC bridge status."""
        bridge = _state.get("aither_bridge")
        if not bridge:
            return {"active": False, "message": "Aither bridge not running"}
        return {**bridge.status(), "active": True}

    # ── Mail REST endpoints ──

    @app.post("/mail/send")
    async def mail_send(request: Request):
        """Send an email (queued for delivery)."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        result = await mail.send(
            to=body.get("to", ""),
            subject=body.get("subject", ""),
            body=body.get("body", ""),
            html=body.get("html", ""),
            from_addr=body.get("from", ""),
            agent=body.get("agent", ""),
            attachments=body.get("attachments"),
        )
        return result

    @app.get("/mail/inbox")
    async def mail_inbox(agent: str = "", limit: int = 50):
        """Get received emails."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.inbox(agent=agent, limit=limit)}

    @app.get("/mail/sent")
    async def mail_sent(agent: str = "", limit: int = 50):
        """Get sent/queued emails."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.sent(agent=agent, limit=limit)}

    @app.get("/mail/email/{email_id}")
    async def mail_get_email(email_id: str):
        """Get email by ID."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "not_found"}, status_code=404)
        email_obj = mail.get_email(email_id)
        if not email_obj:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return email_obj

    @app.post("/mail/config")
    async def mail_configure(request: Request):
        """Configure SMTP settings."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        mail.configure(**body)
        return {"ok": True, "config": mail.get_config()}

    @app.get("/mail/config")
    async def mail_get_config():
        """Get SMTP configuration (password redacted)."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"configured": False}
        return mail.get_config()

    @app.get("/mail/providers")
    async def mail_providers():
        """List available SMTP provider presets."""
        from adk.smtp import PROVIDER_PRESETS
        return {"providers": PROVIDER_PRESETS}

    @app.post("/mail/mailbox/provision")
    async def mail_provision_mailbox(request: Request):
        """Provision a mailbox for a user or agent."""
        mail = _state.get("mail_relay")
        if not mail:
            return JSONResponse({"error": "Mail relay not initialized"}, status_code=503)
        body = await request.json()
        return mail.provision_mailbox(
            username=body.get("username", ""),
            email_address=body.get("email_address", ""),
            display_name=body.get("display_name", ""),
            domain=body.get("domain", ""),
        )

    @app.get("/mail/mailboxes")
    async def mail_list_mailboxes():
        """List all provisioned mailboxes."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"mailboxes": []}
        return {"mailboxes": mail.list_mailboxes()}

    @app.get("/mail/mailbox/{username}/inbox")
    async def mail_user_inbox(username: str, limit: int = 50):
        """Get inbox for a specific user/agent."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"emails": []}
        return {"emails": mail.inbox(agent=username, limit=limit)}

    @app.get("/mail/status")
    async def mail_status():
        """Mail relay status."""
        mail = _state.get("mail_relay")
        if not mail:
            return {"active": False, "message": "Mail relay not initialized"}
        return {**mail.status(), "active": True}

    # ─── Mesh relay endpoints ───

    @app.get("/mesh/status")
    async def mesh_status():
        """AitherNet mesh relay status."""
        relay = _state.get("relay")
        if not relay:
            return {"joined": False, "message": "Set AITHER_API_KEY to join AitherNet"}
        return relay.status()

    @app.get("/mesh/nodes")
    async def mesh_nodes(capability: str = "", limit: int = 50):
        """Discover other nodes on the mesh."""
        relay = _state.get("relay")
        if not relay:
            return {"nodes": [], "message": "Not connected to AitherNet"}
        nodes = await relay.discover(capability=capability, limit=limit)
        return {"nodes": [n.__dict__ for n in nodes], "total": len(nodes)}

    @app.post("/mesh/relay")
    async def mesh_relay(request: Request):
        """Relay a message to another node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        to_node = body.get("to_node", "")
        msg_type = body.get("msg_type", "chat")
        payload = body.get("payload", {})
        if not to_node:
            return JSONResponse({"error": "to_node is required"}, status_code=400)
        ok = await relay.send(to_node, msg_type, payload)
        return {"ok": ok, "relayed_to": to_node, "msg_type": msg_type}

    @app.get("/mesh/messages")
    async def mesh_messages():
        """Poll for relay messages addressed to this node."""
        relay = _state.get("relay")
        if not relay:
            return {"messages": []}
        messages = await relay.poll_messages()
        return {"messages": [m.__dict__ for m in messages], "count": len(messages)}

    @app.get("/mesh/agents")
    async def mesh_agents():
        """Discover every agent in the mesh (name + invoke_url + inference backend +
        skills). Auth-gated: the server queries the owner registry with ITS token and
        returns the list to the authenticated caller — the browser never sees the
        owner token (it auths to this endpoint with the server bearer instead)."""
        from adk.mesh_discovery import discover_agents
        agents, warnings = await discover_agents()
        return {"agents": [a.to_dict() for a in agents], "warnings": warnings,
                "total": len(agents)}

    def _is_safe_proxy_url(url: str) -> bool:
        """Guard the mesh proxy targets. invoke_url values come from owner-authed
        discovery (registry / local agents.json / a2a-fleet), NOT from caller input,
        so this is defense-in-depth against a poisoned registry entry — not the
        primary control. We deliberately ALLOW private/LAN IPs because real mesh
        agents live there (e.g. the OptiPlex on 192.168.x.x); we only reject
        non-HTTP(S) schemes and the cloud-metadata address."""
        try:
            from urllib.parse import urlparse
            u = urlparse(url)
            if u.scheme not in ("http", "https"):
                return False
            host = (u.hostname or "").lower()
            if not host:
                return False
            # Block cloud instance-metadata endpoints (link-local 169.254.169.254).
            if host in ("169.254.169.254", "metadata", "metadata.google.internal"):
                return False
            if host.startswith("169.254."):
                return False
            return True
        except Exception:
            return False

    async def _proxy_agent_stream(ref, message: str):
        """SSE generator that forwards a chat to a mesh agent (protocol-aware) and
        re-emits frames in the pack format (token/answer/error), so the web packs can
        talk to a REMOTE agent through this server without holding its credentials."""
        try:
            if ref.chat_protocol == "openai":
                from adk.llm.base import Message
                from adk.llm.openai_compat import OpenAIProvider
                base = ref.invoke_url.rstrip("/")
                if not base.endswith("/v1"):
                    base += "/v1"
                prov = OpenAIProvider(base_url=base, default_model=ref.model or "")
                got = False
                async for chunk in prov.chat_stream([Message(role="user", content=message)],
                                                    max_tokens=1024):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        got = True
                        yield f"data: {json.dumps({'type': 'token', 't': text})}\n\n"
                if not got:
                    resp = await prov.chat([Message(role="user", content=message)], max_tokens=1024)
                    yield f"data: {json.dumps({'type': 'answer', 'answer': getattr(resp, 'content', '') or ''})}\n\n"
            else:
                from adk.shell.genesis_client import GenesisClient
                client = GenesisClient(base_url=ref.invoke_url)
                async for chunk in client.chat_stream(message):
                    yield f"data: {json.dumps({'type': 'token', 't': chunk})}\n\n"
        except Exception as exc:  # noqa: BLE001 - surface as an SSE error frame, never 500 mid-stream
            # Log the full exception server-side; emit a generic frame so remote
            # agent internals / URLs / secrets never leak to the browser.
            logger.warning("mesh chat proxy to %s failed: %s", getattr(ref, "name", "?"), exc)
            yield f"data: {json.dumps({'type': 'error', 'error': 'upstream agent error'})}\n\n"

    @app.post("/mesh/agents/{name}/chat/stream")
    async def mesh_agent_chat(name: str, request: Request):
        """Proxy a streaming chat to the named mesh agent — it replies on its OWN
        inference. Auth-gated (server bearer); the remote agent's URL/creds stay
        server-side."""
        from adk.mesh_discovery import resolve_agent
        ref = await resolve_agent(name)
        if ref is None or not ref.invoke_url:
            return JSONResponse({"error": f"agent '{name}' not found or has no invoke_url"},
                                status_code=404)
        if not _is_safe_proxy_url(ref.invoke_url):
            logger.warning("mesh chat proxy: refusing unsafe invoke_url for %s", name)
            return JSONResponse({"error": "agent has an unsafe invoke_url"}, status_code=502)
        body = await request.json()
        message = body.get("message", body.get("content", ""))
        return StreamingResponse(
            _proxy_agent_stream(ref, message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Central console: aggregate each agent's own /admin API through an ALLOWLIST
    # (module-level _mesh_admin_allowed). See its definition for the rationale —
    # cli/exec + credential writes are never reachable through this proxy.
    @app.api_route("/mesh/agents/{name}/admin/{path:path}",
                   methods=["GET", "POST", "PATCH", "DELETE"])
    async def mesh_agent_admin(name: str, path: str, request: Request):
        """Owner-authed proxy to a mesh agent's OWN /admin API, restricted to an
        allowlist (observe + narrow safe controls). Auth-gated by the server
        bearer; the remote agent's URL stays server-side; secret values are never
        surfaced (remote admin masks them). cli/exec + credential writes are not
        reachable through here."""
        import httpx
        from adk.mesh_discovery import resolve_agent

        if not _mesh_admin_allowed(request.method, path):
            logger.warning("mesh admin proxy: refusing %s /admin/%s for %s",
                           request.method, path, name)
            return JSONResponse(
                {"error": "admin route not permitted through the mesh console",
                 "method": request.method, "path": path},
                status_code=403,
            )

        ref = await resolve_agent(name)
        if ref is None or not ref.invoke_url:
            return JSONResponse({"error": f"agent '{name}' not found or has no invoke_url"},
                                status_code=404)
        if not _is_safe_proxy_url(ref.invoke_url):
            logger.warning("mesh admin proxy: refusing unsafe invoke_url for %s", name)
            return JSONResponse({"error": "agent has an unsafe invoke_url"}, status_code=502)

        target = ref.invoke_url.rstrip("/") + "/admin/" + path.strip("/")
        # Forward the owner's bearer to the remote agent (mesh-shared server key);
        # a mismatch surfaces as the remote's own 401, never a silent success.
        fwd_headers = {}
        auth = request.headers.get("authorization")
        if auth:
            fwd_headers["Authorization"] = auth
        body_bytes = await request.body()
        if request.headers.get("content-type"):
            fwd_headers["Content-Type"] = request.headers["content-type"]

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.request(
                    request.method, target,
                    params=dict(request.query_params),
                    content=body_bytes or None,
                    headers=fwd_headers,
                )
        except httpx.RequestError as exc:
            logger.warning("mesh admin proxy to %s failed: %s", name, exc)
            return JSONResponse({"error": "upstream agent unreachable"}, status_code=502)

        media = resp.headers.get("content-type", "application/json")
        return Response(content=resp.content, status_code=resp.status_code,
                        media_type=media)

    @app.post("/mesh/tools/call")
    async def mesh_tool_call(request: Request):
        """Call an MCP tool on a remote mesh node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        node_id = body.get("node_id", "")
        tool_name = body.get("name", body.get("tool", ""))
        arguments = body.get("arguments", {})
        if not node_id or not tool_name:
            return JSONResponse({"error": "node_id and name are required"}, status_code=400)
        result = await relay.call_remote_tool(node_id, tool_name, arguments)
        return result

    @app.get("/mesh/tools")
    async def mesh_discover_tools(node_id: str = ""):
        """Discover MCP tools on mesh nodes."""
        relay = _state.get("relay")
        if not relay:
            return {"tools": [], "message": "Not connected to AitherNet"}
        if node_id:
            tools = await relay.list_remote_tools(node_id)
            return {"node_id": node_id, "tools": tools}
        all_tools = await relay.discover_mesh_tools()
        return {"tools": all_tools, "total": len(all_tools)}

    @app.post("/mesh/agent/call")
    async def mesh_agent_call(request: Request):
        """Call an agent on a remote mesh node."""
        relay = _state.get("relay")
        if not relay:
            return JSONResponse({"error": "Not connected to AitherNet"}, status_code=503)
        body = await request.json()
        agent_name = body.get("agent", "")
        message = body.get("message", body.get("content", ""))
        target_node = body.get("node_id", "")
        if not agent_name or not message:
            return JSONResponse({"error": "agent and message are required"}, status_code=400)
        result = await relay.call_remote_agent(agent_name, message, target_node=target_node)
        return result

    # ─── MCP server status ───

    @app.get("/mcp/status")
    async def mcp_status():
        """MCP server status."""
        mcp = _state.get("mcp_server")
        if not mcp:
            return {"active": False, "message": "MCP server not initialized"}
        return {**mcp.status(), "active": True}

    # ─── Aeon — Multi-Agent Group Chat ───

    _state["aeon_sessions"] = {}

    @app.post("/aeon/chat")
    async def aeon_chat(request: Request):
        """Multi-agent group chat. Creates or reuses an AeonSession."""
        from adk.aeon import AeonSession, AEON_PRESETS

        body = await request.json()
        message = body.get("message", "")
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        session_id = body.get("session_id")
        preset = body.get("preset", "balanced")
        participants = body.get("participants")
        rounds = body.get("rounds", 1)
        synthesize = body.get("synthesize", True)

        # Reuse existing session or create new
        sessions = _state["aeon_sessions"]
        if session_id and session_id in sessions:
            session = sessions[session_id]
        else:
            session = AeonSession(
                participants=participants,
                preset=preset,
                rounds=rounds,
                synthesize=synthesize,
                config=config,
            )
            sessions[session.session_id] = session

        response = await session.chat(message)
        return {
            "session_id": response.session_id,
            "messages": [m.to_dict() for m in response.messages],
            "synthesis": response.synthesis.to_dict() if response.synthesis else None,
            "total_tokens": response.total_tokens,
            "total_latency_ms": response.total_latency_ms,
            "round_number": response.round_number,
            "participants": session.participants,
        }

    @app.get("/aeon/presets")
    async def aeon_presets():
        """List available group chat presets."""
        from adk.aeon import AEON_PRESETS
        return {"presets": AEON_PRESETS}

    @app.get("/aeon/sessions/{session_id}")
    async def aeon_session_detail(session_id: str):
        """Get history and stats for an Aeon session."""
        sessions = _state["aeon_sessions"]
        if session_id not in sessions:
            return JSONResponse({"error": "session not found"}, status_code=404)
        session = sessions[session_id]
        return {
            "session_id": session.session_id,
            "participants": session.participants,
            "history": [m.to_dict() for m in session.history],
            "rounds": session._round_counter,
            "total_messages": len(session.history),
        }

    # ─── Slash-command manifest for AitherShell auto-discovery ───

    @app.get("/slash-commands")
    async def slash_commands():
        """Return structured manifest of all ADK CLI commands.

        AitherShell queries this on startup and auto-registers each command
        as a /name slash command with tab-completion for arguments.
        """
        from adk.cli import build_command_manifest
        manifest = build_command_manifest()
        return {
            "commands": manifest,
            "version": __version__,
            "total": len(manifest),
        }

    @app.post("/cli/execute")
    async def cli_execute(request: Request):
        """Execute an ADK CLI command from AitherShell.

        Body: {"command": "train", "args": ["status"]}
        or:   {"command": "backend", "args": ["list"]}

        Returns the command's stdout/stderr as text.
        This lets AitherShell run any CLI command without shelling out.
        """
        import asyncio
        import subprocess

        body = await request.json()
        command = body.get("command", "")
        args = body.get("args", [])

        if not command:
            return JSONResponse({"error": "command is required"}, status_code=400)

        # Security: only allow known ADK commands, not arbitrary shell execution
        from adk.cli import build_command_manifest
        valid_commands = {c["name"] for c in build_command_manifest()}
        if command not in valid_commands:
            return JSONResponse(
                {"error": f"Unknown command: {command}", "valid": sorted(valid_commands)},
                status_code=400,
            )

        cmd = ["python", "-m", "adk.cli", command] + [str(a) for a in args]
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=60,
            )
            return {
                "command": command,
                "args": args,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return JSONResponse({"error": "Command timed out (60s)"}, status_code=504)
        except FileNotFoundError:
            return JSONResponse({"error": "Python not found"}, status_code=500)

    # ── FormBridge (local form automation; routes are loopback-guarded) ──
    if os.getenv("AITHER_FORMBRIDGE_ROUTES", "1").lower() not in ("0", "false"):
        try:
            from adk.formbridge.routes import create_formbridge_router

            app.include_router(create_formbridge_router())
            logger.info("FormBridge routes mounted (/formbridge/*)")
        except ImportError as e:
            logger.debug("FormBridge routes not available: %s", e)

    return app


async def _strata_ingest_bg(**kwargs):
    """Module-level fire-and-forget Strata ingest for _aitheros_stream.

    _aitheros_stream is module-level and cannot see create_app's _strata_ingest
    closure — referencing it raised NameError and silently truncated the SSE
    stream after the answer. Never blocks or raises.
    """
    try:
        from adk.strata import get_strata_ingest
        strata = get_strata_ingest()
        await strata.ingest_chat(**kwargs)
    except Exception:
        pass


async def _aitheros_stream(get_agent_fn, message: str, session_id: str, agent_name: str | None, reasoning: bool, mcp_endpoints: list | None = None, gen_params: dict | None = None):
    """SSE generator emitting AitherOS-typed events.

    Uses the app-level get_agent() for shared memory/tools/fleet support.
    Emits heartbeat during sync tool execution to prevent frontend timeout.

    Protocol:
      event: session_start  -> {session_id, agent, model}
      event: heartbeat      -> {} (every 2s during tool execution)
      event: tool_call      -> {tools: [{name, args}]}
      event: tool_result    -> {results: [{tool, success, output}]}
      event: token          -> {t: "chunk", n: count}
      event: answer         -> {answer: "full response"}
      event: complete       -> {duration_ms, tokens_used}
    """
    start = time.time()
    try:
        agent = await get_agent_fn(agent_name)
        # Attach the customer's self-hosted MCP "hands" relayed by the platform in
        # the /stream body (brain/body/HANDS). Best-effort; never blocks the turn.
        if mcp_endpoints:
            try:
                from adk.mcp_endpoint_tools import register_mcp_endpoint_tools
                register_mcp_endpoint_tools(agent, mcp_endpoints)
            except Exception:
                pass
        model_name = getattr(agent.llm, "provider_name", "unknown")

        # session_start
        yield f"event: session_start\ndata: {json.dumps({'type': 'session_start', 'session_id': session_id, 'agent': agent.name, 'model': model_name})}\n\n"

        _gp = gen_params or {}
        # If agent has tools, use sync chat with background heartbeat
        if agent._tools.list_tools():
            # Run chat in a task with heartbeat to keep SSE alive
            chat_task = asyncio.ensure_future(agent.chat(message, session_id=session_id, **_gp))
            while not chat_task.done():
                yield f"event: heartbeat\ndata: {json.dumps({'type': 'heartbeat'})}\n\n"
                await asyncio.sleep(2)
            resp = chat_task.result()

            # HITL: the turn paused for tool approval — surface the pending tools and stop.
            # The customer allows/denies via POST /sessions/{id}/confirm, which resumes.
            if getattr(resp, "requires_action", False):
                pending = getattr(resp, "pending", []) or []
                yield f"event: requires_action\ndata: {json.dumps({'type': 'requires_action', 'pending': pending, 'session_id': session_id})}\n\n"
                duration_ms = int((time.time() - start) * 1000)
                yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'content': resp.content, 'pending': pending, 'requires_action': True, 'duration_ms': duration_ms, 'session_id': session_id})}\n\n"
                return

            # Emit tool calls if any. AgentResponse.tool_calls_made is a list[str]
            # of tool names (a clean name = success; a "name[denied|blocked|
            # circuit_break]" suffix marks a non-execution). Tolerate dict entries
            # too in case a caller enriches them.
            if resp.tool_calls_made:
                for tc in resp.tool_calls_made:
                    if isinstance(tc, dict):
                        tname = tc.get("name", "?")
                        targs = tc.get("args", {})
                        toutput = tc.get("output", "")
                    else:
                        tname = str(tc)
                        targs = {}
                        toutput = ""
                    success = "[" not in tname
                    yield f"event: tool_call\ndata: {json.dumps({'type': 'tool_call', 'tools': [{'name': tname, 'args': targs}]})}\n\n"
                    yield f"event: tool_result\ndata: {json.dumps({'type': 'tool_result', 'results': [{'tool': tname, 'success': success, 'output': str(toutput)[:500]}]})}\n\n"

            yield f"event: answer\ndata: {json.dumps({'type': 'answer', 'answer': resp.content})}\n\n"

            duration_ms = int((time.time() - start) * 1000)
            yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms, 'tokens_used': resp.tokens_used, 'model': resp.model, 'session_id': session_id})}\n\n"

            # Fire-and-forget Strata + Chronicle
            asyncio.ensure_future(_strata_ingest_bg(
                agent=agent.name, session_id=session_id,
                user_message=message, assistant_response=resp.content,
                model=resp.model, tokens_used=resp.tokens_used,
                latency_ms=resp.latency_ms, tool_calls=resp.tool_calls_made,
            ))
            return

        # Streaming path — no tools
        full_content = ""
        token_count = 0
        async for chunk in agent.chat_stream(message, session_id=session_id, **_gp):
            if chunk:
                full_content += chunk
                token_count += 1
                yield f"event: token\ndata: {json.dumps({'type': 'token', 't': chunk, 'n': token_count})}\n\n"

        yield f"event: answer\ndata: {json.dumps({'type': 'answer', 'answer': full_content})}\n\n"

        duration_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms, 'tokens_used': token_count, 'model': model_name, 'session_id': session_id})}\n\n"

        asyncio.ensure_future(_strata_ingest_bg(
            agent=agent.name, session_id=session_id,
            user_message=message, assistant_response=full_content,
            model=model_name, tokens_used=token_count,
            latency_ms=duration_ms, tool_calls=[],
        ))

    except Exception as exc:
        # Broad on purpose: an unhandled exception inside an SSE generator silently
        # truncates the stream (client sees session_start + heartbeats then EOF, with
        # no error/complete). Surface it as a typed error + terminal complete so the
        # caller always gets a clean end, and log the full traceback for diagnosis.
        logger.exception("AitherOS stream error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        duration_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms})}\n\n"


async def _aeon_stream(message: str, preset: str, agents: list[str] | None,
                       rounds: int, session_id: str):
    """SSE generator for Aeon group-chat — multi-agent discussion with synthesis.

    Protocol:
      event: session_start  -> {type, session_id, orchestrator, participants, model}
      event: agent_message  -> {type, agent, content, tokens_used, latency_ms, round_number}
      event: synthesis      -> {type, agent, content, tokens_used, latency_ms} (if enabled)
      event: error          -> {type, error}
      event: complete       -> {type, total_tokens, total_latency_ms, session_id}
    """
    start = time.time()
    try:
        from adk.aeon import AeonSession

        # Create the session with the given preset/agents/rounds
        session = AeonSession(
            participants=agents,
            preset=preset,
            rounds=rounds,
            synthesize=True,
        )
        model_name = getattr(session._shared_llm, "provider_name", "unknown")

        # Emit session_start
        yield f"event: session_start\ndata: {json.dumps({'type': 'session_start', 'session_id': session_id, 'orchestrator': session.orchestrator, 'participants': session.participants, 'model': model_name})}\n\n"

        # Run the group chat
        response = await session.chat(message)

        # Emit each agent message (excluding synthesis)
        for msg in response.messages:
            yield f"event: agent_message\ndata: {json.dumps({'type': 'agent_message', 'agent': msg.agent, 'content': msg.content, 'tokens_used': msg.tokens_used, 'latency_ms': msg.latency_ms, 'round_number': msg.round_number})}\n\n"

        # Emit synthesis if present
        if response.synthesis:
            yield f"event: synthesis\ndata: {json.dumps({'type': 'synthesis', 'agent': response.synthesis.agent, 'content': response.synthesis.content, 'tokens_used': response.synthesis.tokens_used, 'latency_ms': response.synthesis.latency_ms})}\n\n"

        # Emit complete
        total_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'total_tokens': response.total_tokens, 'total_latency_ms': response.total_latency_ms, 'session_id': session_id})}\n\n"

    except Exception as exc:
        logger.exception("Aeon stream error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
        duration_ms = int((time.time() - start) * 1000)
        yield f"event: complete\ndata: {json.dumps({'type': 'complete', 'duration_ms': duration_ms})}\n\n"


async def _stream_agent_response(agent, message: str, history: list[dict], model: str | None):
    """SSE stream generator using agent.chat_stream() — full pipeline.

    Routes through the agent's tool loop, safety, context manager, memory,
    and events — NOT a raw LLM stream bypass.

    If the agent has tools and the LLM requests a tool call, chat_stream()
    falls back to sync chat() and yields the full response as one chunk.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        async for text_chunk in agent.chat_stream(
            message, history=history or None, model=model,
        ):
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model or getattr(agent.llm, "provider_name", ""),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as exc:  # noqa: BLE001 — this is the terminal SSE error
        # boundary; any uncaught exception here (observed live: httpx.HTTPStatusError
        # from a backend rejecting a tool-calling request, e.g. an Ollama model that
        # doesn't support the tools schema) must surface as a clear error chunk to
        # the client, not crash the generator and leave the HTTP response hanging
        # with zero bytes sent ("(no response)" client-side, indistinguishable from
        # a real hang). A narrower (RuntimeError, OSError, ConnectionError) catch
        # here previously let exactly this class of error through uncaught.
        logger.error("Streaming error: %s", exc)
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model or "",
            "choices": [{"index": 0, "delta": {"content": f"Error: {exc}"}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Final stop chunk
    stop_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_data)}\n\n"
    yield "data: [DONE]\n\n"


def _resolve_portal_kit_backend() -> bool:
    """Add portal-kit-backend to sys.path if running in the monorepo.

    Search order:
    1. Already importable (pip install / Docker mount) — no action needed
    2. PORTAL_KIT_BACKEND_PATH env var
    3. Monorepo sibling: AitherOS/apps/packages/portal-kit-backend/
    4. Relative from ADK: ../../AitherOS/apps/packages/portal-kit-backend/

    Returns True if portal_kit_backend is importable after this call.
    """
    import importlib
    import sys
    from pathlib import Path

    # Already available?
    try:
        importlib.import_module("portal_kit_backend")
        return True
    except ImportError:
        pass

    candidates = []

    # Env override
    env_path = os.getenv("PORTAL_KIT_BACKEND_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Monorepo: ADK is at <root>/aither-adk/, backend at <root>/AitherOS/apps/packages/
    adk_root = Path(__file__).resolve().parent.parent  # aither-adk/
    monorepo_root = adk_root.parent  # project root
    candidates.append(monorepo_root / "AitherOS" / "apps" / "packages")
    # Also try if portal-kit-backend parent is directly at packages/
    candidates.append(monorepo_root / "AitherOS" / "apps" / "packages" / "portal-kit-backend")

    for cand in candidates:
        # The package dir must contain portal_kit_backend/__init__.py or be the
        # parent that contains portal_kit_backend/ as a subdirectory.
        pkg_init = cand / "portal_kit_backend" / "__init__.py"
        parent_init = cand / "__init__.py"
        if pkg_init.exists():
            # cand is the parent we need on sys.path
            path_str = str(cand)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        elif parent_init.exists() and cand.name == "portal-kit-backend":
            # portal-kit-backend IS the package (uses portal_kit_backend as import name)
            # Add its parent so `import portal_kit_backend` works
            path_str = str(cand.parent)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            importlib.import_module("portal_kit_backend")
            return True
        except ImportError:
            continue

    return False


def _load_capability_domains_config() -> dict:
    """Load capability_domains.yaml from AitherOS config.

    Returns parsed dict or {} if not found.
    """
    from pathlib import Path

    candidates = [
        Path(os.getenv("CAPABILITY_DOMAINS_PATH", "")),
        Path(__file__).resolve().parent.parent.parent / "AitherOS" / "config" / "capability_domains.yaml",
        Path("/app/AitherOS/config/capability_domains.yaml"),
    ]
    for p in candidates:
        if p.exists():
            try:
                import yaml
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
    return {}


def _mount_workspace_routers(app: FastAPI, port: int) -> None:
    """Mount portal-kit-backend routers for workspace mode.

    Makes the ADK server function like a full WorkspaceRuntime with
    calendar, mail, social, executive, workspace intelligence, documents,
    onboarding, file sync, contacts, and config endpoints.

    Resolves portal-kit-backend from the monorepo if not pip-installed,
    initialises the SQLite workspace store, and wires portal registration
    + Proton auto-connect into the server lifespan.
    """
    _ws_log = logging.getLogger("adk.workspace")
    mounted = 0

    # ── Resolve portal-kit-backend package ──
    if not _resolve_portal_kit_backend():
        _ws_log.error(
            "portal-kit-backend not found. Install it or set PORTAL_KIT_BACKEND_PATH. "
            "Workspace routers will not be available."
        )
        return

    # ── Set up workspace data directory ──
    from pathlib import Path
    data_dir = Path(os.getenv("AITHER_DATA_DIR", os.path.expanduser("~/.aither")))
    store_path = data_dir / "workspace" / "aitherchat.db"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AITHER_CHAT_STORE_PATH", str(store_path))

    # ── Discover brain pack + agent.yaml via pack_discovery ──
    from adk.pack_discovery import discover_agent_yaml, discover_brain_pack, discover_pack_dir
    import yaml

    agent_yaml_path = discover_agent_yaml()
    brain_pack_path = discover_brain_pack()
    pack_dir = discover_pack_dir()

    # Set env vars so portal-kit-backend picks them up
    if brain_pack_path:
        os.environ.setdefault("AGENT_BRAIN_PACK", str(brain_pack_path))
        _ws_log.info("Brain pack: %s", brain_pack_path)
    if pack_dir:
        _ws_log.info("Pack directory: %s", pack_dir)

    # Read agent spec
    enabled_domains: list[str] = []
    agent_spec: dict = {}
    if agent_yaml_path and agent_yaml_path.exists():
        try:
            agent_spec = yaml.safe_load(agent_yaml_path.read_text(encoding="utf-8")) or {}
            enabled_domains = agent_spec.get("enabled_domains", [])
            _ws_log.info("Agent spec: %s (domains=%s)", agent_yaml_path, enabled_domains or "all")
        except Exception:
            pass

    # ── Load structured domain definitions for API ──
    domain_config = _load_capability_domains_config()
    domain_defs = domain_config.get("capability_domains", {})

    def _domain_ok(domain: str) -> bool:
        return not enabled_domains or domain in enabled_domains

    # ── Router mount helper ──
    def _try_mount(import_fn, label: str, domain: str = ""):
        nonlocal mounted
        if domain and not _domain_ok(domain):
            _ws_log.debug("Skipped %s (domain '%s' not enabled)", label, domain)
            return
        try:
            router = import_fn()
            app.include_router(router)
            mounted += 1
            _ws_log.info("Workspace router: %s", label)
        except ImportError as e:
            _ws_log.debug("Workspace router %s not available: %s", label, e)
        except Exception as e:
            _ws_log.warning("Failed to mount %s: %s", label, e)

    app_id = os.getenv("APP_ID", "aither-local")

    # ── Mount portal-kit-backend routers (domain-gated) ──

    # Calendar + mail
    def _cal():
        from portal_kit_backend.routers.calendar import create_calendar_router
        return create_calendar_router(app_id=app_id)
    _try_mount(_cal, "calendar (/api/calendar/*)", "calendar_mail")

    def _mail():
        from portal_kit_backend.routers.workspace_mail import create_workspace_mail_router
        return create_workspace_mail_router(app_id=app_id)
    _try_mount(_mail, "mail (/api/mail/*)", "calendar_mail")

    # People + directory
    def _dir():
        from portal_kit_backend.routers.workspace_directory import create_workspace_directory_router
        return create_workspace_directory_router()
    _try_mount(_dir, "directory (/api/directory/*)", "people")

    def _contacts():
        from portal_kit_backend.routers.contacts import create_contacts_router
        return create_contacts_router()
    _try_mount(_contacts, "contacts (/api/contacts/*)", "people")

    # Proton suite / file sync
    def _fs():
        from portal_kit_backend.routers.file_sync import router
        return router
    _try_mount(_fs, "file-sync (/api/file-sync/*)", "proton_suite")

    # Social + marketing
    def _soc():
        from portal_kit_backend.routers.social import create_social_router
        return create_social_router()
    _try_mount(_soc, "social (/api/social/*)", "social_marketing")

    # Executive assistant
    def _exec():
        from portal_kit_backend.routers.executive_briefing import create_executive_briefing_router
        return create_executive_briefing_router(app_id=app_id)
    _try_mount(_exec, "executive (/api/executive/*)", "executive_assistant")

    # Workspace intelligence
    def _wi():
        from portal_kit_backend.routers.workspace_intelligence import create_workspace_intelligence_router
        return create_workspace_intelligence_router(app_id=app_id)
    _try_mount(_wi, "workspace-intelligence (/api/workspace-intelligence/*)", "workspace_intelligence")

    # Documents
    def _docs():
        from portal_kit_backend.routers.documents import create_documents_router
        return create_documents_router()
    _try_mount(_docs, "documents (/api/documents/*)", "documents")

    # Onboarding (no domain gate — always available)
    def _onb():
        from portal_kit_backend.routers.onboarding import create_onboarding_router
        return create_onboarding_router(app_id=app_id)
    _try_mount(_onb, "onboarding (/api/onboarding/*)")

    # ── Workspace config endpoints (portal iframe + domain info) ──

    @app.get("/api/config/embed")
    async def config_embed():
        return {
            "app_id": app_id,
            "name": agent_spec.get("name", "Aither"),
            "embed": True,
            "url": f"http://localhost:{port}",
            "embed_url": f"http://localhost:{port}/?embedded=true",
        }

    @app.get("/api/config/tabs")
    async def config_tabs():
        portal = agent_spec.get("portal", {})
        return {"tabs": portal.get("capabilities", []), "app_id": app_id}

    @app.get("/api/config/domains")
    async def config_domains():
        """Return structured domain definitions with enabled state."""
        domains_out = []
        for did, ddef in domain_defs.items():
            domains_out.append({
                "id": did,
                "label": ddef.get("label", did),
                "description": ddef.get("description", ""),
                "panels": ddef.get("panels", []),
                "routers": ddef.get("routers", []),
                "tools": ddef.get("tools", []),
                "enabled": _domain_ok(did),
            })
        return {
            "domains": domains_out,
            "enabled_domains": enabled_domains,
            "app_id": app_id,
        }

    # ── Wire lifespan: store init, portal registration, Proton auto-connect ──

    _orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _workspace_lifespan(a):
        async with _orig_lifespan(a):
            # Initialise SQLite workspace store
            try:
                from portal_kit_backend.aither_store import _ensure_init
                await _ensure_init()
                _ws_log.info("Workspace store initialised at %s", store_path)
            except Exception as e:
                _ws_log.warning("Workspace store init failed: %s", e)

            # Register with portal
            try:
                from adk.registration import start_registration
                await start_registration(
                    agent_spec=agent_spec,
                    server_url=f"http://localhost:{port}",
                )
            except Exception as e:
                _ws_log.debug("Portal registration skipped: %s", e)

            # Auto-connect Proton if detected
            try:
                from adk.proton_setup import detect_proton_bridge, auto_connect_mail
                detection = detect_proton_bridge()
                if detection["bridge_running"]:
                    _ws_log.info("Proton Bridge detected — attempting auto-connect")
                    result = await auto_connect_mail(
                        api_base=f"http://localhost:{port}",
                    )
                    if result.get("ok"):
                        _ws_log.info("Proton mail connected: %s", result.get("email"))
                    else:
                        _ws_log.info(
                            "Proton auto-connect: %s (run --setup or configure secrets)",
                            result.get("reason", "unknown"),
                        )
            except Exception:
                pass

            yield

            # Cleanup
            try:
                from adk.registration import stop_registration
                await stop_registration()
            except Exception:
                pass

    app.router.lifespan_context = _workspace_lifespan

    # ── Serve workspace frontend (catch-all, MUST be last) ──
    # Search order: custom frontend-dist, WorkspaceRuntime frontend, bundled fallback
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    _frontend_candidates = [
        Path(os.getenv("AITHER_FRONTEND_DIR", "")),                    # explicit override
        Path.cwd() / "frontend-dist",                                  # local build output
        Path.cwd() / "frontend",                                       # local dev frontend
        Path(__file__).resolve().parent / "workspace-frontend",        # bundled with ADK
    ]
    for fdir in _frontend_candidates:
        if fdir.exists() and (fdir / "index.html").exists():
            app.mount("/", StaticFiles(directory=str(fdir), html=True), name="frontend")
            _ws_log.info("Workspace frontend: %s", fdir)
            break
    else:
        _ws_log.warning("No workspace frontend found — API-only mode")

    _ws_log.info("Workspace mode: %d portal-kit routers mounted, store=%s", mounted, store_path)


def main():
    """CLI entry point: aither-serve"""
    parser = argparse.ArgumentParser(description="AitherADK Agent Server")
    parser.add_argument("--identity", "-i", default="aither", help="Agent identity to load (single-agent mode)")
    parser.add_argument("--port", "-p", type=int, default=None, help="Port (default: 8080)")
    parser.add_argument("--host", default=None, help="Host (default: 0.0.0.0)")
    parser.add_argument("--backend", "-b", help="LLM backend: ollama, openai, anthropic")
    parser.add_argument("--model", "-m", help="Model name override")
    parser.add_argument("--fleet", "-f", default=None, help="Fleet YAML config file for multi-agent mode")
    parser.add_argument("--agents", "-a", default=None, help="Comma-separated agent identities for fleet mode (e.g. aither,lyra,demiurge)")
    parser.add_argument("--invoke-url", default=None, help="Publicly reachable URL for fleet dispatch (e.g. http://192.168.1.50:8900)")
    parser.add_argument("--workspace", action="store_true", help="Workspace mode: mount portal-kit-backend routers, register with portal")
    parser.add_argument("--setup", action="store_true", help="Run first-time setup wizard (Proton auto-detect, etc.)")
    args = parser.parse_args()

    config = Config.from_env()
    if args.backend:
        config.llm_backend = args.backend
    if args.model:
        config.model = args.model

    port = args.port or config.server_port
    host = args.host or config.server_host
    # Write the resolved port/host back so lifespan helpers (_join_aithernet,
    # invoke_url, gateway/fleet registration) all see the actual bound port —
    # not the config default. Without this, --port diverges from config.server_port
    # (and _join_aithernet referenced an undefined `port`, crashing startup).
    config.server_port = port
    config.server_host = host

    if args.invoke_url:
        os.environ["AITHER_INVOKE_URL"] = args.invoke_url

    # Determine mode
    fleet_path = args.fleet
    fleet_agents = args.agents.split(",") if args.agents else None
    is_fleet = bool(fleet_path or fleet_agents)

    # Workspace mode flag (env var or CLI)
    workspace_mode = args.workspace or os.getenv("AITHER_WORKSPACE_MODE", "").lower() in ("true", "1")
    if workspace_mode:
        os.environ["AITHER_WORKSPACE_MODE"] = "true"

    # First-run setup wizard (--setup)
    if args.setup:
        from adk.proton_setup import print_detection_summary
        print("\n  AitherADK Workspace Setup\n")
        print_detection_summary()
        print("  Setup complete. Start the workspace server with:")
        print(f"    aither serve --workspace --port {port}")
        return

    app = create_app(
        identity=args.identity,
        config=config,
        fleet_path=fleet_path,
        fleet_agents=fleet_agents,
    )

    # ── Workspace mode: mount portal-kit-backend routers ──
    if workspace_mode:
        _mount_workspace_routers(app, port)

    import uvicorn

    if config.gateway_url and config.aither_api_key:
        gateway_line = f"  Gateway: {config.gateway_url} (will register on startup)"
    else:
        gateway_line = (
            "  Gateway: not configured — set AITHER_API_KEY to connect\n"
            "  Demo:    https://demo.aitherium.com"
        )

    if is_fleet:
        agents_str = fleet_agents if fleet_agents else f"from {fleet_path}"
        print(f"Starting AitherADK fleet server — agents: {agents_str}, port: {port}")
        print(f"  Fleet:  GET  http://localhost:{port}/agents")
        print(f"  Chat:   POST http://localhost:{port}/agents/<name>/chat")
        print(f"  Forge:  POST http://localhost:{port}/forge/dispatch")
    elif workspace_mode:
        print(f"Starting AitherADK workspace server — identity: {args.identity}, port: {port}")
        print(f"  Portal: http://localhost:{port} (workspace UI)")
        print(f"  Embed:  http://localhost:{port}/?embedded=true")
    else:
        print(f"Starting AitherADK server — identity: {args.identity}, port: {port}")

    print(f"  Chat:   POST http://localhost:{port}/chat")
    print(f"  OpenAI: POST http://localhost:{port}/v1/chat/completions")
    print(f"  WS:     WS   ws://localhost:{port}/ws/chat")
    irc_port_env = os.getenv("AITHER_IRC_PORT", "")
    if irc_port_env:
        print(f"  IRC:    TCP  localhost:{irc_port_env} (mIRC, WeeChat, HexChat, irssi)")
    print(f"  MCP:    POST http://localhost:{port}/mcp (JSON-RPC 2.0)")
    print(f"  A2A:    POST http://localhost:{port}/a2a (Google A2A v0.3.0)")
    print(f"  Card:   GET  http://localhost:{port}/.well-known/agent.json")
    print(f"  Mail:   POST http://localhost:{port}/mail/send")
    print(f"  Health: GET  http://localhost:{port}/health")
    print(f"  Docs:   GET  http://localhost:{port}/docs")
    print(gateway_line)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main_workspace():
    """CLI entry point: adk-workspace — shortcut for `adk-serve --workspace`."""
    import sys
    if "--workspace" not in sys.argv:
        sys.argv.insert(1, "--workspace")
    main()


if __name__ == "__main__":
    main()
