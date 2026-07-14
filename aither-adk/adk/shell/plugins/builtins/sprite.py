"""
Sprite Plugin for AitherShell
=============================

Your AI companion creature (AitherSprite) — check on it, care for it, talk
to it, watch it evolve. State lives server-side (Genesis /api/v1/aither-sprite);
this plugin is just a window into the terrarium.

Usage:
    /sprite                       — Show your creature (ASCII render + needs)
    /sprite hatch [NAME]          — Hatch a new sprite
    /sprite feed|play|clean|rest  — Care actions
    /sprite talk <message>        — Talk to it (mood-tuned reply)
    /sprite history [N]           — Recent care/evolution events
    /sprite revive                — Wake a dormant sprite
    /sprite art                   — Generated-art status (media-forge renders)

Aliases: /pet
"""

import os
from typing import Any, Dict, List, Optional

from adk._tls import tls_verify
from adk.shell.plugins import SlashCommand

try:
    from adk.shell.auth import AuthStore
except ImportError:
    AuthStore = None  # type: ignore


def _genesis_url() -> str:
    return os.environ.get("AITHER_GENESIS_URL", "http://localhost:8100")


def _api_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if AuthStore:
        token = AuthStore.get_active_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        profile = AuthStore.get_active_profile() if hasattr(AuthStore, "get_active_profile") else None
        if profile and profile.get("tenant_id"):
            headers["X-Tenant-ID"] = profile["tenant_id"]
    return headers


async def _get(path: str, params: dict = None) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=40, verify=tls_verify()) as c:
        resp = await c.get(f"{_genesis_url()}{path}", params=params or {}, headers=_api_headers())
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, body: dict = None) -> dict:
    import httpx
    async with httpx.AsyncClient(timeout=40, verify=tls_verify()) as c:
        resp = await c.post(f"{_genesis_url()}{path}", json=body or {}, headers=_api_headers())
        resp.raise_for_status()
        return resp.json()


BASE = "/api/v1/aither-sprite"

# Mood → (eyes, mouth) for the ASCII creature. Same mood labels as the engine.
_FACES: Dict[str, tuple] = {
    "joyful": ("^", "▽"), "content": ("•", "‿"), "curious": ("o", "o"),
    "sleepy": ("-", "~"), "grumpy": ("¬", "⌒"), "sad": (";", "⌢"),
    "dormant": ("_", "_"),
}

_STAGE_BODY: Dict[str, List[str]] = {
    "egg": [
        "    .-\"\"-.   ",
        "   /  {c}  \\  ",
        "   \\      /  ",
        "    '-..-'   ",
    ],
    "baby": [
        "    .--.    ",
        "   ( {e}{e} )   ",
        "   (  {m}  )   ",
        "    `--'    ",
    ],
    "child": [
        "   /\\ .--. /\\  ",
        "  (  ( {e}{e} )  ) ",
        "   \\ (  {m}  ) /  ",
        "     `----'    ",
    ],
    "teen": [
        "   /\\ .---. /\\  ",
        "  (  ( {e} {e} )  ) ",
        "   \\ (  {m}   ) /  ",
        "     `-----'  ~ ",
    ],
    "adult": [
        "  /\\  .----.  /\\ ",
        " (  )( {e}  {e} )(  )",
        "  \\/ (  {m}    ) \\/ ",
        "      `------'   ",
    ],
    "elder": [
        "  /\\  .----.  /\\ ",
        " (  )( {e}  {e} )(  )",
        "  \\/ ( ={m}=   ) \\/ ",
        "   *  `------' *  ",
    ],
}


def _render_creature(status: Dict[str, Any]) -> str:
    stage = status.get("stage", "baby")
    mood = status.get("mood_label", "content")
    eyes, mouth = _FACES.get(mood, _FACES["content"])
    body = _STAGE_BODY.get(stage, _STAGE_BODY["baby"])
    crack = "*" if stage == "egg" else ""
    return "\n".join(
        line.replace("{e}", eyes).replace("{m}", mouth).replace("{c}", crack or " ")
        for line in body
    )


def _bar(value: float, width: int = 12) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _render_status(s: Dict[str, Any]) -> str:
    needs = s.get("needs", {})
    lines = [
        "",
        _render_creature(s),
        "",
        f"  {s.get('name', 'Sprite')} — {s.get('stage')} ({s.get('form')}), "
        f"{s.get('age_days', 0):.1f} days old — feeling {s.get('mood_label')}",
        "",
        f"  🍎 Food   {_bar(needs.get('hunger', 0))} {round(needs.get('hunger', 0) * 100):3d}%",
        f"  ⚡ Energy {_bar(needs.get('energy', 0))} {round(needs.get('energy', 0) * 100):3d}%",
        f"  🫧 Clean  {_bar(needs.get('hygiene', 0))} {round(needs.get('hygiene', 0) * 100):3d}%",
        f"  💙 Bond   {_bar(needs.get('bond', 0))} {round(needs.get('bond', 0) * 100):3d}%",
    ]
    if s.get("dormant"):
        lines.append("\n  💤 Dormant — `/sprite revive` to wake it (bond penalty applies)")
    return "\n".join(lines)


class SpritePlugin(SlashCommand):
    name: str = "sprite"
    aliases: List[str] = ["pet"]
    description: str = "AitherSprite — your companion creature (care, talk, evolve)"
    category: str = "fun"

    async def run(self, args: List[str], ctx: Dict[str, Any]) -> Optional[str]:
        import httpx

        sub = (args[0].lower() if args else "status")
        rest = args[1:]
        try:
            if sub in ("status", "show", "me"):
                return _render_status(await _get(f"{BASE}/me/status"))
            if sub == "hatch":
                name = " ".join(rest).strip() or "Sprite"
                s = await _post(f"{BASE}/hatch", {"name": name})
                return f"🐣 {s.get('name')} hatched!\n" + _render_status(s)
            if sub in ("feed", "play", "clean", "rest"):
                s = await _post(f"{BASE}/me/care", {"action": sub})
                return _render_status(s)
            if sub == "talk":
                message = " ".join(rest).strip()
                if not message:
                    return "Say something: /sprite talk <message>"
                data = await _post(f"{BASE}/me/talk", {"message": message})
                reply = data.get("reply", "…")
                tail = _render_status(data["sprite"]) if data.get("sprite") else ""
                return f'  💬 "{reply}"\n{tail}'
            if sub == "history":
                limit = int(rest[0]) if rest and rest[0].isdigit() else 15
                data = await _get(f"{BASE}/me/history", params={"limit": limit})
                events = data.get("events", [])
                if not events:
                    return "  (no history yet)"
                import datetime as _dt
                lines = ["  Recent events:"]
                for e in events:
                    ts = _dt.datetime.fromtimestamp(e["timestamp"]).strftime("%m-%d %H:%M")
                    detail = f" {e['detail']}" if e.get("detail") else ""
                    lines.append(f"   {ts}  {e['kind']}{detail}")
                return "\n".join(lines)
            if sub == "revive":
                s = await _post(f"{BASE}/me/revive")
                return "💫 Revived!\n" + _render_status(s)
            if sub == "art":
                data = await _get(f"{BASE}/me/appearance")
                assets = data.get("assets", {})
                ready = sum(1 for a in assets.values() if a.get("status") == "ready")
                lines = [f"  🎨 Art: {ready}/{len(assets)} renders ready "
                         f"(stage {data.get('stage')}/{data.get('form')})"]
                for key, a in sorted(assets.items()):
                    mark = {"ready": "✓", "failed": "✗"}.get(a.get("status"), "…")
                    lines.append(f"   [{mark}] {key}")
                return "\n".join(lines)
            return self.get_help()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return "No sprite yet — hatch one: /sprite hatch <name>"
            if e.response.status_code == 409:
                try:
                    return f"⚠ {e.response.json().get('detail', 'conflict')}"
                except ValueError:
                    return "⚠ conflict"
            return f"Sprite service error: {e.response.status_code}"
        except httpx.HTTPError as e:
            return f"Cannot reach the sprite service: {e}"
