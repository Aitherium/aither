"""`adk bonsai-local` must be REACHABLE by an adk agent.

THE GAP THIS CLOSES (found 2026-07-31 while chasing "make it dead simple to install/use
Bonsai-27B with awdk agents seamlessly"):

`adk bonsai-local` starts llama.cpp on **:8090**. Every backend preset pointed at **:8201**,
which is AitherVLLMSwap — a FLEET service that does not exist on anyone else's machine. So
the one-command install worked perfectly and then no `--backend` could talk to it:

    $ adk bonsai-local          # -> healthy server on :8090
    $ adk --backend bonsai      # -> dials :8201, finds nothing

Nothing failed loudly. The container was up, `/health` returned 200, and the backend simply
could not connect — which reads as "the local model is broken", not "the two halves were
never introduced". `cmd_bonsai_local`'s own docstring asserted :8090 was "the ladder's
`local` tier", so the code claimed a wiring that did not exist, and that claim is what made
it look already-done.

The port lived as FOUR separate literals (docstring, `--port` default, docker publish, help
text) and the preset's URL was a fifth, different number. These tests assert the pairing
rather than any one of the copies.
"""

import re
from pathlib import Path

import pytest

CLI = Path(__file__).resolve().parents[1] / "cli.py"


@pytest.fixture(scope="module")
def src() -> str:
    return CLI.read_text(encoding="utf-8")


def _presets(src: str) -> str:
    m = re.search(r"_BACKEND_PRESETS: dict\[str, dict\] = \{(.*?)\n\}", src, re.S)
    assert m, "could not locate _BACKEND_PRESETS — this test is stale, not passing"
    return m.group(1)


def test_a_backend_preset_targets_the_port_bonsai_local_serves(src):
    """The whole point: after `adk bonsai-local`, some `--backend` must reach it."""
    port = re.search(r"BONSAI_LOCAL_PORT = (\d+)", src)
    assert port, "BONSAI_LOCAL_PORT is gone — the port has been un-unified"
    body = _presets(src)
    assert "BONSAI_LOCAL_PORT" in body, (
        "no backend preset references BONSAI_LOCAL_PORT. `adk bonsai-local` would start a "
        "server no agent can dial — the exact gap this file exists to prevent."
    )


def test_the_preset_is_named_after_the_command_that_starts_it(src):
    """Discoverability is the feature. `adk bonsai-local` -> `--backend bonsai-local`."""
    assert '"bonsai-local"' in _presets(src)


def test_the_cli_default_port_is_not_a_separate_literal(src):
    """A second copy of the number is how the two halves drifted apart in the first place."""
    assert 'default=BONSAI_LOCAL_PORT' in src, (
        "the --port default is a bare literal again; it must read the shared constant"
    )
    assert 'os.environ.get("AITHER_BONSAI_PORT", str(BONSAI_LOCAL_PORT))' in src


def test_local_and_bonsai_presets_still_point_at_the_fleet_server(src):
    """Do NOT silently repoint the fleet presets.

    `local` and `bonsai` mean AitherVLLMSwap on :8201 for existing fleet users. Fixing the
    laptop path by hijacking those would trade one broken audience for another — the new
    preset is additive on purpose.
    """
    body = _presets(src)
    assert body.count("localhost:8201/v1") == 2, (
        "the `local`/`bonsai` presets no longer both target :8201 — repointing them breaks "
        "fleet users instead of serving laptop users"
    )


def test_the_docstring_no_longer_claims_8090_is_the_local_tier(src):
    """The false claim is what made this look wired. It must not come back."""
    m = re.search(r"def cmd_bonsai_local\(args\).*?\"\"\"(.*?)\"\"\"", src, re.S)
    assert m, "cmd_bonsai_local is gone — this test is stale, not passing"
    doc = m.group(1)
    assert "ladder's `local` tier" not in doc, (
        "the docstring again claims :8090 is the `local` tier; `local` targets :8201"
    )
