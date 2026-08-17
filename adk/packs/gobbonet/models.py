"""GobboNet's model picker, on macOS and Linux.

GobboNet is a Windows application — not because its UI is, but because four of
its endpoints are served by `fileserver.ps1` and `launch.bat`:

    GET  /models-list.json    what you can switch to
    GET  /active-model.json   what is loaded now
    POST /swap-model          switch to another GGUF
    GET  /swap-status         poll until the new one is up

The UI itself is a page and some JS with no Windows in it. So the app does not
need porting, emulating, or rewriting — it needs those four answers from
something that runs everywhere. That is this module, and with it `adk gobbonet`
runs GobboNet unmodified on machines where it currently cannot run at all.

THE CONTRACT IS THEIRS, READ FROM THEIR SOURCE. Field names come from
`js/02-model.js` and `js/01-config.js`, not from a guess: the picker reads
`m.file`, `m.name`, `m.id`, `m.family` and `m.thinkingFormat`, selects on
`m.active`, and polls `st.phase` until it is `ready` or `error`. A response that
is merely well-formed JSON would render an empty dropdown and look like "no
models installed" rather than like a broken integration — which is exactly the
failure mode this pack keeps running into.

SWAPPING IS NOT INSTANT AND MUST NOT PRETEND TO BE. Their poller waits up to
three minutes and tolerates the server being unreachable mid-swap. So the phase
goes `loading` while llama-server restarts and only becomes `ready` once a real
request succeeds — never when the process merely started. A model that is
loading answers the socket long before it can answer a prompt.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Where adk keeps GGUFs. Shared with llamacpp_setup so a model installed by
#: `adk gobbonet --setup-model` shows up in the picker without extra wiring.
try:  # pragma: no cover - import shape differs when run as a script
    from adk.llamacpp_setup import DEFAULT_PORT as LLAMA_PORT
    from adk.llamacpp_setup import MODELS_DIR
except Exception:  # noqa: BLE001 - adk internals absent in a bare checkout
    MODELS_DIR = Path.home() / ".aither" / "models"
    LLAMA_PORT = 8200


#: Family + thinking format inferred from the filename, because that is all a
#: GGUF on disk tells you. Their UI uses `thinkingFormat` to strip reasoning
#: tags; getting it wrong leaks `<think>` blocks into the chat, which reads as
#: a broken model rather than as a metadata error.
_FAMILIES = [
    (re.compile(r"deepseek", re.I), "deepseek", "deepseek"),
    (re.compile(r"qwen|qwq", re.I), "qwen", "deepseek"),
    (re.compile(r"gemma", re.I), "gemma", "gemma"),
    (re.compile(r"llama|nemotron", re.I), "llama", "none"),
    (re.compile(r"mistral|mixtral", re.I), "mistral", "none"),
    (re.compile(r"phi", re.I), "phi", "none"),
    (re.compile(r"gpt-oss|harmony", re.I), "harmony", "harmony"),
]


def _classify(filename: str) -> tuple[str, str]:
    for pattern, family, thinking in _FAMILIES:
        if pattern.search(filename):
            return family, thinking
    return "custom", "none"


def _pretty(filename: str) -> str:
    """A human label from a GGUF filename.

    `Bonsai-27B-Q1_0.gguf` -> `Bonsai 27B Q1_0`. Deliberately mechanical: an
    invented marketing name would not match what the user downloaded.
    """
    stem = re.sub(r"\.gguf$", "", filename, flags=re.I)
    stem = re.sub(r"-0000\d-of-0000\d$", "", stem)  # sharded weights
    return stem.replace("_", " ").replace("-", " ").strip()


@dataclass
class SwapState:
    """What `/swap-status` reports. `phase` is the field their poller reads."""

    phase: str = "idle"  # idle | loading | ready | error
    message: str = ""
    file: str = ""
    started: float = 0.0

    def as_json(self) -> dict:
        out = {"phase": self.phase, "file": self.file}
        if self.message:
            out["message"] = self.message
        if self.phase == "loading" and self.started:
            out["elapsed"] = round(time.time() - self.started, 1)
        return out


@dataclass
class ModelManager:
    """Serves GobboNet's four model endpoints from a local llama.cpp server."""

    models_dir: Path = field(default_factory=lambda: Path(MODELS_DIR))
    port: int = LLAMA_PORT
    #: Injected in tests. Real swapping shells llama-server; the endpoint
    #: contract is what these tests are about, not process management.
    _spawn: Optional[object] = None

    def __post_init__(self) -> None:
        self._state = SwapState()
        self._lock = threading.Lock()
        self._active: str = ""

    # ── discovery ────────────────────────────────────────────────────────
    def available(self) -> list[str]:
        """GGUF filenames on disk, first shard only for sharded weights.

        Listing every shard would put `…00002-of-00003` in the picker as if it
        were a separate model, and selecting it would load a fragment.
        """
        if not self.models_dir.is_dir():
            return []
        seen: set[str] = set()
        out: list[str] = []
        for p in sorted(self.models_dir.glob("*.gguf")):
            m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
            if m:
                if m.group(1) != "00001":
                    continue
                key = p.name[: m.start()]
                if key in seen:
                    continue
                seen.add(key)
            out.append(p.name)
        return out

    def active_file(self) -> str:
        """The loaded model, asked of the SERVER rather than remembered.

        A cached answer goes stale the moment llama-server is restarted by
        anything other than us — and then the picker shows the wrong model
        selected, which is worse than showing none.
        """
        if self._active:
            return self._active
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/v1/models", timeout=3
            ) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return ""
        ids = [m.get("id", "") for m in (data.get("data") or [])]
        for name in self.available():
            if any(name in i or Path(name).stem in i for i in ids):
                return name
        return ids[0] if ids else ""

    # ── the four endpoints ───────────────────────────────────────────────
    def models_list(self) -> dict:
        """`GET /models-list.json` — every key their <option> builder reads."""
        active = self.active_file()
        models = []
        for name in self.available():
            family, thinking = _classify(name)
            models.append({
                "file": name,          # option VALUE — unambiguous
                "name": _pretty(name),
                "id": Path(name).stem.lower(),
                "family": family,
                "thinkingFormat": thinking,
                "active": name == active,
            })
        return {"models": models}

    def active_model(self) -> dict:
        """`GET /active-model.json` — id, name, ggufFile, thinkingFormat."""
        name = self.active_file()
        if not name:
            # Reported honestly rather than as a fabricated default: the UI
            # shows "Unknown", which is true, instead of naming a model that
            # is not loaded.
            return {"id": "custom", "name": "No model loaded", "ggufFile": "",
                    "thinkingFormat": "none"}
        family, thinking = _classify(name)
        return {
            "id": Path(name).stem.lower(),
            "name": _pretty(name),
            "ggufFile": name,
            "family": family,
            "thinkingFormat": thinking,
        }

    def swap(self, filename: str) -> tuple[bool, str]:
        """`POST /swap-model` — begin a swap. Returns (accepted, message).

        Returns immediately and does the work on a thread, because their UI
        expects a fast response and then polls. Blocking here would trip the
        fetch long before the model finished loading.
        """
        if not filename:
            return False, "no file given"
        if filename not in self.available():
            # Named explicitly. "Swap failed: HTTP 400" sends someone looking
            # at the network; naming the file sends them to the models folder.
            return False, f"{filename} is not in {self.models_dir}"

        with self._lock:
            if self._state.phase == "loading":
                return False, f"already loading {self._state.file}"
            self._state = SwapState(phase="loading", file=filename,
                                    started=time.time())

        threading.Thread(target=self._do_swap, args=(filename,),
                         daemon=True, name="gobbonet-swap").start()
        return True, "swapping"

    def swap_status(self) -> dict:
        """`GET /swap-status` — polled every ~1.5s for up to three minutes."""
        with self._lock:
            return self._state.as_json()

    # ── the work ─────────────────────────────────────────────────────────
    def _do_swap(self, filename: str) -> None:
        try:
            spawn = self._spawn or self._spawn_llama
            spawn(self.models_dir / filename, self.port)
            # READY means it answered a real request, not that the process
            # started. A loading model accepts the socket long before it can
            # answer a prompt, and reporting ready there makes the first
            # message hang with no explanation.
            if self._wait_until_answering(self.port, deadline=170):
                with self._lock:
                    self._active = filename
                    self._state = SwapState(phase="ready", file=filename)
            else:
                # Cite the port failure if there was one. "Did not answer in
                # time" sends the user to look at the model; "could not free
                # port 8200" sends them to the actual cause.
                why = " — ".join(x for x in (
                    getattr(self, "_stop_error", ""),
                    getattr(self, "_probe_error", ""),
                ) if x)
                with self._lock:
                    self._state = SwapState(
                        phase="error", file=filename,
                        message=(f"the model did not answer in time — {why}" if why
                                 else "the model did not answer in time"))
        except Exception as e:  # noqa: BLE001 - must reach the UI, not a log
            with self._lock:
                self._state = SwapState(phase="error", file=filename,
                                        message=f"{type(e).__name__}: {e}")

    def _spawn_llama(self, model_path: Path, port: int) -> None:
        """Restart llama-server on the chosen model."""
        from adk import llamacpp_setup as lc

        binary = getattr(lc, "LLAMACPP_DIR", Path.home() / ".aither" / "llamacpp")
        exe = next((p for p in Path(binary).rglob("llama-server*") if p.is_file()), None)
        if exe is None:
            raise RuntimeError(
                "llama-server is not installed — run `adk gobbonet --setup-model`")

        import subprocess

        self._stop_existing(port)
        cmd = [str(exe), "-m", str(model_path), "--port", str(port), "--host", "127.0.0.1"]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _stop_existing(self, port: int) -> None:
        """Free the port. A failure here is REMEMBERED, not swallowed.

        If the old server survives, the new one exits immediately with "address
        in use" and the swap times out three minutes later saying the model did
        not answer — which sends the user to look at the model. Keeping the
        reason means the timeout message can name the actual cause instead.
        """
        import subprocess
        import sys

        cmd = (["taskkill", "/F", "/IM", "llama-server.exe"] if sys.platform == "win32"
               else ["pkill", "-f", f"llama-server.*--port {port}"])
        try:
            subprocess.run(cmd, capture_output=True, timeout=20)
            self._stop_error = ""
        except (OSError, subprocess.SubprocessError) as e:
            # Not fatal on its own: "nothing to kill" is the common case and is
            # not an error at all. Recorded so a later timeout can cite it.
            self._stop_error = f"could not free port {port} ({type(e).__name__}: {e})"
        time.sleep(1.0)

    def _wait_until_answering(self, port: int, deadline: float) -> bool:
        """True once /v1/models returns a NAMED model.

        Connection failures are the EXPECTED state here — the server is still
        mapping weights — so they are caught narrowly and RECORDED rather than
        discarded. Recording matters: if the deadline expires, the last error is
        the only evidence of why. "Connection refused" for three minutes means
        the process died on startup; a read timeout means it is alive and slow.
        Those need opposite responses, and without this they look identical.
        """
        end = time.time() + deadline
        while time.time() < end:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=3
                ) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data.get("data"):
                    self._probe_error = ""
                    return True
                self._probe_error = "server answered with an empty model list"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                self._probe_error = f"{type(e).__name__}: {e}"
            time.sleep(1.5)
        return False
