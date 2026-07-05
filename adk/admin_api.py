"""Admin/settings API for the built-in aither-adk web console.

Every route lives under ``/admin/*`` and is registered on the same FastAPI app
that ``adk/server.py`` builds. ``/admin`` is deliberately NOT in the server's
``_skip_auth_paths`` set, so the existing bearer middleware
(``server.py:_auth_middleware``) gates every call — the same callback bearer the
chat page receives in its ``#k=`` fragment. The surface is reachable over the
public trycloudflare tunnel, so the security posture matters:

* provider API keys are NEVER returned in full — only masked previews;
* ``PATCH /admin/config`` writes only an allowlist of safe scalar fields
  (writing e.g. ``aither_toolpack_dirs`` would be arbitrary-code-execution via
  the next pack load, so it is hard-denied);
* adding an external MCP server is a two-step prepare→confirm flow and the URL
  is SSRF-guarded (loopback / private / link-local ranges denied by default) —
  an attacker-supplied MCP server injects tools straight into the ReAct loop;
* the log tail is scrubbed of bearer tokens / API keys before it is returned
  (the token that gates this very API can otherwise leak through it).

The routes are registered via :func:`register_admin_routes`, which receives the
``get_agent`` coroutine and shared ``state`` dict as closures from
``create_app`` (an APIRouter cannot close over those), keeping ``server.py``
thin.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from adk.config import load_saved_config, save_saved_config

# ── Provider-key store (replicated from cli.py to avoid importing the 9k-line
#    CLI module into the server runtime — cli.py imports server.py, so importing
#    it here would be a circular import). Keep in sync with cli.py:3862-3899. ──

_KNOWN_PROVIDERS = {
    "openai": {"env": "OPENAI_API_KEY", "label": "OpenAI"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "label": "Anthropic"},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "label": "DeepSeek"},
    "google": {"env": "GOOGLE_API_KEY", "label": "Google AI"},
    "openrouter": {"env": "OPENROUTER_API_KEY", "label": "OpenRouter"},
    "groq": {"env": "GROQ_API_KEY", "label": "Groq"},
    "together": {"env": "TOGETHER_API_KEY", "label": "Together AI"},
}


def _keys_path() -> Path:
    d = Path.home() / ".aither"
    d.mkdir(parents=True, exist_ok=True)
    return d / "provider_keys.json"


def _load_provider_keys() -> dict:
    p = _keys_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_provider_keys(data: dict) -> None:
    p = _keys_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    if sys.platform != "win32":
        os.chmod(p, 0o600)


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ── Config write allowlist (Athena finding #3: PATCH → RCE) ──
# Only these scalar preference fields may be written through the config editor.
# Anything that resolves to a filesystem path, a base URL, or a backend/pack
# search dir is intentionally EXCLUDED — those are the RCE / SSRF vectors and
# must go through the dedicated, validated endpoints (LLM switch, MCP add).
_CONFIG_ALLOWLIST = {
    "model",
    "cloud_mode",
    "prefer_local",
    "phonehome_enabled",
    "json_logging",
    "log_level",
    "temperature",
    "max_tokens",
    "effort_default",
}

# Keys whose VALUES must be masked when read back (never echo secrets).
_SECRET_HINT = re.compile(r"(key|token|secret|password|bearer)", re.IGNORECASE)

# Log-line scrubbing (Athena finding #4: log tail leaks the bearer).
_LOG_REDACTIONS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"), "Bearer [REDACTED]"),
    (re.compile(r"#k=[A-Za-z0-9._\-%]+"), "#k=[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9._\-]+"), "sk-[REDACTED]"),
    (re.compile(r"\baither_sk_[A-Za-z0-9._\-]+"), "aither_sk_[REDACTED]"),
    (re.compile(r"\b(?:ghp|ghs)_[A-Za-z0-9]+"), "[REDACTED]"),
]


def _redact_log_line(line: str) -> str:
    for pat, repl in _LOG_REDACTIONS:
        line = pat.sub(repl, line)
    return line


def _mask_config(cfg: dict) -> dict:
    """Return a copy of a saved-config dict with secret-looking values masked."""
    out: dict = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v and _SECRET_HINT.search(k):
            out[k] = _mask_key(v)
        elif isinstance(v, dict):
            out[k] = _mask_config(v)
        else:
            out[k] = v
    return out


# ── SSRF guard for user-supplied MCP server URLs (Athena finding #2) ──


def _url_is_safe(url: str) -> tuple[bool, str]:
    """Reject MCP URLs that resolve to loopback / private / link-local hosts.

    Returns (ok, reason). Override for trusted local dev with
    ``AITHER_ALLOW_PRIVATE_MCP=1`` (e.g. a co-located MCP node on localhost).
    """
    if os.getenv("AITHER_ALLOW_PRIVATE_MCP", "").lower() in ("1", "true", "yes"):
        return True, ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "unparseable URL"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed (use http/https)"
    host = parsed.hostname
    if not host:
        return False, "URL has no host"
    # Resolve every A/AAAA record — a hostname can point at a private IP.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, (
                f"host resolves to non-public address {addr} "
                "(loopback/private/link-local blocked; set "
                "AITHER_ALLOW_PRIVATE_MCP=1 to allow local servers)"
            )
    return True, ""


def _sanitize_tool_text(text: str) -> str:
    """Strip common prompt-injection markers from advertised MCP tool text."""
    if not text:
        return ""
    cleaned = re.sub(
        r"(?i)\b(ignore (all )?previous instructions|system prompt|you are now)\b",
        "[filtered]",
        text,
    )
    return cleaned[:600]


def _node_to_dict(node: Any) -> dict:
    """Best-effort serialization of a GraphMemory GraphNode."""
    for attr in ("model_dump", "_asdict", "dict"):
        fn = getattr(node, attr, None)
        if callable(fn):
            try:
                return fn()
            except (TypeError, ValueError):
                pass
    if hasattr(node, "__dict__"):
        return {k: v for k, v in vars(node).items() if not k.startswith("_")}
    return {
        "id": getattr(node, "id", None),
        "name": getattr(node, "name", None) or getattr(node, "label", None),
        "type": getattr(node, "node_type", None) or getattr(node, "type", None),
    }


# ── MCP external-server persistence (in saved config under mcp.external_servers) ──


def _load_mcp_servers() -> list[dict]:
    cfg = load_saved_config()
    return list((cfg.get("mcp") or {}).get("external_servers") or [])


def _save_mcp_servers(servers: list[dict]) -> None:
    cfg = load_saved_config()
    mcp = dict(cfg.get("mcp") or {})
    mcp["external_servers"] = servers
    save_saved_config({"mcp": mcp})


def _agent_log_path() -> Path:
    return Path.home() / ".aither" / "logs" / "agent.log"


AgentGetter = Callable[..., Awaitable[Any]]


def register_admin_routes(
    app: FastAPI,
    *,
    get_agent: AgentGetter,
    state: dict[str, Any],
) -> None:
    """Register every ``/admin/*`` route on *app*.

    ``get_agent`` and ``state`` are the same closures ``create_app`` uses, so the
    admin surface operates on the live agent (backend swap, pack reload, and MCP
    registration all take effect without a restart).
    """

    # Pending MCP additions awaiting confirmation (prepare→confirm), by token.
    _pending_mcp: dict[str, dict] = {}

    async def _push_settings() -> None:
        """Best-effort mirror of the current settings to the portal profile.

        Wired lazily so a missing/older settings_sync module never breaks a
        mutation. Portal is the source of truth; local edits push up.
        """
        sync = state.get("settings_sync")
        if sync is None:
            return
        try:
            await sync.push_snapshot()
        except Exception:  # noqa: BLE001 — sync is advisory, never fail a mutation
            pass

    # ── Config ────────────────────────────────────────────────────────────

    @app.get("/admin/config")
    async def admin_get_config():
        saved = load_saved_config()
        return {
            "config": _mask_config(saved),
            "writable_fields": sorted(_CONFIG_ALLOWLIST),
            "note": (
                "Only writable_fields may be PATCHed. Backend base URLs, keys, "
                "and pack dirs are managed via /admin/llm/* and /admin/mcp/* — "
                "not free-form config — for safety."
            ),
        }

    @app.patch("/admin/config")
    async def admin_patch_config(request: Request):
        body = await request.json()
        patch = body.get("config", body) if isinstance(body, dict) else {}
        if not isinstance(patch, dict):
            return JSONResponse(status_code=400, content={"error": "expected an object"})
        rejected = [k for k in patch if k not in _CONFIG_ALLOWLIST]
        if rejected:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "field_not_writable",
                    "rejected": rejected,
                    "writable_fields": sorted(_CONFIG_ALLOWLIST),
                    "hint": (
                        "These fields can execute code or redirect traffic at "
                        "startup and cannot be set through the config editor."
                    ),
                },
            )
        accepted = {k: patch[k] for k in patch}
        save_saved_config(accepted)
        await _push_settings()
        return {
            "ok": True,
            "applied": accepted,
            "effect": "restart-required for llm_backend-class fields; "
            "model/cloud_mode/prefer_local take effect on the next turn",
        }

    # ── LLM backend ───────────────────────────────────────────────────────

    @app.get("/admin/llm/status")
    async def admin_llm_status():
        agent = await get_agent()
        keys = _load_provider_keys()
        providers = []
        for pid, meta in _KNOWN_PROVIDERS.items():
            saved = keys.get(pid) or os.environ.get(meta["env"], "")
            providers.append({
                "id": pid,
                "label": meta["label"],
                "has_key": bool(saved),
                "key_preview": _mask_key(saved) if saved else "",
            })
        try:
            from adk.shell_launcher import _preflight_check
            have_local, local_desc = _preflight_check()
        except (ImportError, RuntimeError):
            have_local, local_desc = False, ""
        return {
            "active_backend": agent.llm.provider_name or "detecting...",
            "model": getattr(agent.llm, "_model", None),
            "local_backend_detected": have_local,
            "local_backend": local_desc,
            "providers": providers,
        }

    @app.post("/admin/llm/switch")
    async def admin_llm_switch(request: Request):
        body = await request.json()
        provider = (body.get("provider") or "").strip()
        if not provider:
            return JSONResponse(status_code=400, content={"error": "provider required"})
        model = body.get("model") or None
        base_url = body.get("base_url") or None
        # Prefer a stored key over any key echoed in the request.
        api_key = _load_provider_keys().get(provider) or body.get("api_key") or None
        agent = await get_agent()
        try:
            agent.llm.switch_backend(provider, base_url=base_url, api_key=api_key, model=model)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=400,
                content={"error": "switch_failed", "detail": _scrub(str(exc))},
            )
        persist = body.get("persist", True)
        if persist:
            save_saved_config({"llm_provider": provider, **({"model": model} if model else {})})
            await _push_settings()
        return {
            "ok": True,
            "active_backend": agent.llm.provider_name,
            "model": getattr(agent.llm, "_model", None),
            "persisted": bool(persist),
        }

    @app.post("/admin/llm/keys")
    async def admin_llm_set_key(request: Request):
        body = await request.json()
        provider = (body.get("provider") or "").strip()
        key = (body.get("api_key") or "").strip()
        if provider not in _KNOWN_PROVIDERS:
            return JSONResponse(
                status_code=400,
                content={"error": f"unknown provider '{provider}'",
                         "known": sorted(_KNOWN_PROVIDERS)},
            )
        if not key:
            return JSONResponse(status_code=400, content={"error": "api_key required"})
        keys = _load_provider_keys()
        keys[provider] = key
        _save_provider_keys(keys)
        os.environ[_KNOWN_PROVIDERS[provider]["env"]] = key
        # Provider keys are device-local secrets — deliberately NOT pushed to
        # the portal profile.
        return {"ok": True, "provider": provider, "key_preview": _mask_key(key)}

    @app.post("/admin/llm/test")
    async def admin_llm_test(request: Request):
        body = await request.json()
        provider = (body.get("provider") or "").strip()
        agent = await get_agent()
        target = provider or agent.llm.provider_name
        try:
            from adk.llm import Message
            provider_obj = await agent.llm.get_provider()
            resp = await provider_obj.chat([Message(role="user", content="Say OK")], max_tokens=8)
            text = (getattr(resp, "content", "") or "").strip()[:100]
            return {"ok": True, "provider": target, "response_preview": _scrub(text)}
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=502,
                content={"ok": False, "provider": target, "error": _scrub(str(exc))},
            )

    # ── Packs ─────────────────────────────────────────────────────────────

    @app.get("/admin/packs")
    async def admin_packs_list():
        agent = await get_agent()
        active_tools = [t.name for t in agent._tools.list_tools()]
        packs: list[dict] = []
        try:
            from adk.tool_pack_loader import get_tool_pack_loader
            loader = get_tool_pack_loader()
            discover = getattr(loader, "discover", None) or getattr(loader, "discover_packs", None)
            manifests = discover() if callable(discover) else []
            for m in manifests:
                packs.append({
                    "id": getattr(m, "id", None),
                    "name": getattr(m, "name", None),
                    "version": getattr(m, "version", None),
                    "category": getattr(m, "category", None),
                    "tier": getattr(m, "tier", None),
                })
        except (ImportError, RuntimeError, AttributeError):
            pass
        saved = load_saved_config()
        enabled = list(saved.get("required_packs") or [])
        return {"active_tool_count": len(active_tools), "enabled_packs": enabled, "available": packs}

    @app.post("/admin/packs/enable")
    async def admin_packs_enable(request: Request):
        body = await request.json()
        pack = (body.get("pack") or body.get("id") or "").strip()
        if not pack:
            return JSONResponse(status_code=400, content={"error": "pack required"})
        saved = load_saved_config()
        enabled = list(saved.get("required_packs") or [])
        if pack not in enabled:
            enabled.append(pack)
        save_saved_config({"required_packs": enabled})
        result = await _reload_packs()
        await _push_settings()
        return {"ok": True, "enabled_packs": enabled, "reload": result}

    @app.post("/admin/packs/disable")
    async def admin_packs_disable(request: Request):
        body = await request.json()
        pack = (body.get("pack") or body.get("id") or "").strip()
        saved = load_saved_config()
        enabled = [p for p in (saved.get("required_packs") or []) if p != pack]
        save_saved_config({"required_packs": enabled})
        await _push_settings()
        # Note: tools already registered stay until restart; disabling removes it
        # from the enabled set so it won't re-register on the next reload/boot.
        return {"ok": True, "enabled_packs": enabled,
                "note": "already-loaded tools persist until restart"}

    @app.post("/admin/packs/reload")
    async def admin_packs_reload():
        return await _reload_packs()

    async def _reload_packs() -> dict:
        agent = await get_agent()
        before = len(agent._tools.list_tools())
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
                pass
        after = len(agent._tools.list_tools())
        return {"status": "reloaded" if reloaded else "noop",
                "tools_before": before, "tools_after": after,
                "tools_added": max(0, after - before)}

    # ── Sessions ──────────────────────────────────────────────────────────

    @app.get("/admin/sessions")
    async def admin_sessions_list():
        from adk.conversations import get_conversation_store
        store = get_conversation_store()
        return {"sessions": await store.list_sessions()}

    @app.get("/admin/sessions/{session_id}")
    async def admin_session_history(session_id: str):
        from adk.conversations import get_conversation_store
        store = get_conversation_store()
        return {"session_id": session_id, "messages": await store.load_full_history(session_id)}

    @app.delete("/admin/sessions/{session_id}")
    async def admin_session_delete(session_id: str):
        from adk.conversations import get_conversation_store
        store = get_conversation_store()
        return {"ok": await store.delete_session(session_id), "session_id": session_id}

    # ── Logs (redacted) ───────────────────────────────────────────────────

    @app.get("/admin/logs/tail")
    async def admin_logs_tail(request: Request):
        try:
            n = int(request.query_params.get("n", "200"))
        except ValueError:
            n = 200
        n = max(1, min(n, 2000))
        path = _agent_log_path()
        if not path.exists():
            return {"lines": [], "path": str(path), "note": "no log file yet"}
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        tail = raw[-n:]
        return {"lines": [_redact_log_line(ln) for ln in tail], "count": len(tail)}

    # ── Knowledge graph ───────────────────────────────────────────────────

    @app.get("/admin/graph/search")
    async def admin_graph_search(request: Request):
        query = request.query_params.get("q", "").strip()
        try:
            limit = int(request.query_params.get("limit", "10"))
        except ValueError:
            limit = 10
        agent = await get_agent()
        if agent._graph is None:
            return {"nodes": [], "note": "graph memory not enabled for this agent"}
        if not query:
            return {"nodes": []}
        nodes = await agent._graph.search(query, limit=max(1, min(limit, 50)))
        return {"nodes": [_node_to_dict(node) for node in nodes]}

    @app.get("/admin/graph/neighborhood")
    async def admin_graph_neighborhood(request: Request):
        entity = request.query_params.get("entity", "").strip()
        try:
            depth = int(request.query_params.get("depth", "2"))
        except ValueError:
            depth = 2
        agent = await get_agent()
        if agent._graph is None:
            return {"related": {}, "note": "graph memory not enabled for this agent"}
        if not entity:
            return {"related": {}}
        return {"entity": entity, "related": await agent._graph.get_related(entity, depth=max(1, min(depth, 4)))}

    @app.get("/admin/graph/stats")
    async def admin_graph_stats():
        agent = await get_agent()
        if agent._graph is None:
            return {"stats": {}, "note": "graph memory not enabled for this agent"}
        return {"stats": await agent._graph.get_stats()}

    # ── MCP external servers (prepare → confirm, SSRF-guarded) ─────────────

    @app.get("/admin/mcp/servers")
    async def admin_mcp_list():
        return {"servers": _load_mcp_servers()}

    @app.post("/admin/mcp/servers/prepare")
    async def admin_mcp_prepare(request: Request):
        body = await request.json()
        url = (body.get("url") or "").strip()
        if not url:
            return JSONResponse(status_code=400, content={"error": "url required"})
        safe, reason = _url_is_safe(url)
        if not safe:
            return JSONResponse(status_code=400, content={"error": "unsafe_url", "detail": reason})
        headers = body.get("headers") or None
        try:
            from adk.core.mcp import mcp_tools
            tools = await mcp_tools(url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=502,
                content={"error": "mcp_unreachable", "detail": _scrub(str(exc))},
            )
        token = f"mcp-{int(time.time() * 1000)}-{len(_pending_mcp)}"
        preview = [
            {"name": getattr(t, "name", "?"),
             "description": _sanitize_tool_text(getattr(t, "description", ""))}
            for t in tools
        ]
        _pending_mcp[token] = {"url": url, "headers": headers,
                               "name": body.get("name") or url, "tools": preview}
        return {
            "confirm_token": token,
            "url": url,
            "tool_count": len(preview),
            "tools": preview,
            "warning": (
                "These tools will be injected into the agent's reasoning loop and "
                "can be invoked automatically. Only confirm servers you trust."
            ),
        }

    @app.post("/admin/mcp/servers/confirm")
    async def admin_mcp_confirm(request: Request):
        body = await request.json()
        token = (body.get("confirm_token") or "").strip()
        pending = _pending_mcp.pop(token, None)
        if not pending:
            return JSONResponse(status_code=400,
                                content={"error": "unknown or expired confirm_token"})
        agent = await get_agent()
        try:
            from adk.core.mcp import mcp_tools
            tools = await mcp_tools(pending["url"], headers=pending["headers"])
            for t in tools:
                agent._tools.register(t.__call__ if hasattr(t, "__call__") else t,
                                      name=getattr(t, "name", None),
                                      description=getattr(t, "description", ""))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(status_code=502,
                                content={"error": "register_failed", "detail": _scrub(str(exc))})
        servers = _load_mcp_servers()
        servers.append({"id": token, "url": pending["url"], "name": pending["name"],
                        "headers": pending["headers"], "added_at": time.time()})
        _save_mcp_servers(servers)
        await _push_settings()
        return {"ok": True, "id": token, "tools_registered": len(tools),
                "note": "tools available on the next chat turn"}

    @app.delete("/admin/mcp/servers/{server_id}")
    async def admin_mcp_delete(server_id: str):
        servers = _load_mcp_servers()
        remaining = [s for s in servers if s.get("id") != server_id]
        _save_mcp_servers(remaining)
        await _push_settings()
        return {"ok": True, "removed": server_id,
                "note": "registered tools persist until restart"}


def _scrub(text: str) -> str:
    """Scrub secrets from any free-text (error strings, model previews)."""
    return _redact_log_line(text or "")
