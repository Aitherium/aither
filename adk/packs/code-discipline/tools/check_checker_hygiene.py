#!/usr/bin/env python3
"""Assert the checker contract against the checkers themselves.

WHY THIS EXISTS (2026-08-08)
---------------------------------------------------------------------------
The AitherOS coding discipline (``.claude/rules/tech-debt-ledger.md``) says a
mechanically-detectable defect must become a CHECK, not a ledger row, and that
every check must:

  1. have a ``--self-test`` that proves it can still fail,
  2. be wired somewhere unattended (CI or a routine),
  3. exit non-zero (2), never 0, when it cannot run,
  4. name a debt id in any allowlist entry, printed on every run.

Nothing asserted that the checkers obey their own contract. Docs drifted from
reality (``tech-debt-ledger.md`` said "22 checkers"; the gate tree holds ~140).
This check is the process codified: it reads the rules, the CI workflow and the
routine gate list and asserts the contract against the tools they name.

Invariants
---------------------------------------------------------------------------
  HYG001 (gate)   every checker that is DOCUMENTED (named in the quality-gate
                  rules) or WIRED (named in the debt-invariants workflow or the
                  debt-gate-probe routine) declares a ``--self-test``. A gate
                  that cannot prove it can fail is not a gate.
  HYG002 (gate)   every checker path named in the quality-gate rules exists on
                  disk. A documented gate that is not a file is a broken ref.
  HYG003 (gate)   every checker wired in CI or a routine exists on disk. A
                  wired gate that is not a file runs nothing.
  HYG004 (report) the backlog a gate must never open with: wired-but-
                  undocumented checkers (an invisible gate nobody can
                  question), total ``check_*.py``, how many lack ``--self-test``
                  tree-wide, and how many are wired nowhere. Printed by
                  ``--all``; never a gate. A check that floods gets switched
                  off, which is how this repo's per-file-ignores came to exist.
  HYG005 (gate)   the discipline's duplicated artifacts stay IDENTICAL across
                  their delivery homes — the portable twin in awskills,
                  the bundled awdk pack, and the published pack template.
                  A drift between copies is the "duplicated source of truth"
                  hazard code-like-david rule 11 calls active: editing one
                  copy silently orphans the others. Skipped for a layout that
                  does not carry a given home.

A probe that cannot read its sources exits 2, never 0 — silence is not a pass.
Use ``--self-test`` to prove it can still fail (it points the reference set at
a nonexistent file and asserts the gate goes red).

Portable twin lives at ``awskills/tools/check_checker_hygiene.py`` — same
logic, layout-detected root (AitherOS monorepo, generic ``dev/tools`` install,
or bare ``tools/``), so an external repo that adopted the discipline can run it.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Gate tools that are NOT named check_*.py but ARE gates. Used so a bare
# mention in the rules or a run line is still recognised as a gate.
NON_CHECK_GATES = {
    "security_lint.py",
    "rotate_internal_secret.py",
    "repair_ps1_encoding.py",
    "check_actions_allowlist.py",
    "debt_ledger.py",
    "triage_debt_ledger.py",
    "next_debt_id.py",
}

# Any `<something>.py` token in the rules/run text.
TOKEN_RE = re.compile(r"([A-Za-z_][\w.-]*\.py)\b")
# `python <path>/<name>.py` on a workflow run line (path may be relative or
# absolute, possibly with subdirs like dev/tools/pool/).
WF_RUN_RE = re.compile(r"\b(?:python|python3)\s+([\w./-]+\.py)")
# A quoted tool name inside the debt-gate-probe routine or a routine yaml.
ROUTINE_STR_RE = re.compile(
    r'"((?:check_|compare_|security_lint|rotate_internal|repair_ps1)[\w-]*\.py)'
)

# ── Repo layout: the ONLY monorepo-specific data, and it does not live here ──
#
# This module is a PORTABLE TWIN: byte-identical copies ship in awskills and
# the awdk code-discipline pack, and the parity rule below asserts that
# identity. Those two facts used to be in direct conflict with a third — the
# published-tree path scan
# This module is a PORTABLE TWIN: byte-identical copies ship in aither-skills and
# the aither-adk code-discipline pack, and HYG005 asserts that identity. Those two
# facts used to be in direct conflict with a third — the published-tree path scan
# forbids monorepo-internal paths in anything that ships, and this file hardcoded
# nine of them (the PARITY registry, the checker-root candidates, the host-gate
# runner, the debt probe, the self-test fixtures).
#
# The result was a twin that could be neither excluded from publishing
# (SYNC_EXCLUDES is skills-only) nor deleted (PARITY requires the home), so the
# skills mirror simply stopped publishing on 2026-08-13 and stayed stopped.
#
# Resolution: the paths are DATA, in an optional file beside this one. The
# monorepo ships it; a public reader does not have it and gets portable defaults.
# Both copies stay byte-identical AND carry no path a stranger cannot follow.
#
# A rule whose data is absent reports NOT APPLICABLE — never silently passes.
# "No parity registry" and "parity holds" are different facts, and collapsing
# them would make this gate vacuous in exactly the home it ships to.
LAYOUT_FILE = Path(__file__).resolve().parent / "checker_hygiene_layout.json"

_PORTABLE_DEFAULTS: dict = {
    "checker_roots": ["dev/tools", "tools"],
    "parity": [],
    "host_gate_runner": None,
    "routines_dir": None,
    "debt_probe": None,
}


def _load_layout() -> dict:
    """Repo layout from the sibling data file, else portable defaults."""
    layout = dict(_PORTABLE_DEFAULTS)
    try:
        if LAYOUT_FILE.is_file():
            data = json.loads(LAYOUT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                layout.update({k: v for k, v in data.items() if k in layout})
    except (OSError, ValueError) as exc:
        # A malformed layout must be LOUD. Falling back to defaults would silently
        # downgrade every monorepo-only rule to "not applicable" and report green.
        raise SystemExit(f"DEAD: {LAYOUT_FILE} is unreadable/malformed: {exc}")
    return layout


LAYOUT = _load_layout()

#: Artifacts intentionally duplicated across delivery homes. The parity rule asserts each
#: Artifacts intentionally duplicated across delivery homes. HYG005 asserts each
#: copy stays byte-identical to its canonical, so editing one home cannot silently
#: orphan the others. Empty outside the monorepo — see LAYOUT_FILE above.
PARITY = [tuple(p) for p in LAYOUT["parity"] if isinstance(p, (list, tuple)) and len(p) == 2]


def _hyg5(root: Path) -> list[str]:
    """Copy-parity violations across the discipline's delivery homes.

    A pair is asserted only when THIS layout carries both the canonical and the
    copy's home: a generic adopted repo has neither awskills nor the adk
    pack, so skipping there is correct, not a pass-by-omission.
    """
    violations: list[str] = []
    for canon_rel, copy_rel in PARITY:
        canon = root / canon_rel
        copy = root / copy_rel
        if not canon.is_file():
            continue
        if not copy.parent.exists():
            continue
        if not copy.is_file():
            violations.append(f"{copy_rel} missing (canonical {canon_rel} exists)")
            continue
        if canon.read_bytes() != copy.read_bytes():
            violations.append(f"{copy_rel} differs from canonical {canon_rel}")
    return violations


def _safe_print(text: str) -> None:
    """Never let the console codec stop a verdict (Windows cp1252 + a
    non-ASCII finding would otherwise raise UnicodeEncodeError mid-report,
    truncating the list — the exact class check_skills_publishable.py guards)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc, "replace"))


def find_root(start: Path) -> Path:
    """Repo root via git; fall back to walking up for a marker."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        # git missing / not a repo / too slow — fall through to the marker
        # walk. Not silent: a root we could not resolve from git is diagnostic
        # noise, never a verdict (the sources themselves gate on exit 2).
        print(f"[check_checker_hygiene] git rev-parse unavailable ({exc}); "
              f"walking up for a marker", file=sys.stderr)
    cur = start.resolve()
    for p in (cur, *cur.parents):
        if (p / ".git").exists() or (p / ".claude").exists():
            return p
    return cur


def candidate_trees(root: Path) -> list[Path]:
    """Every plausible checker tree, in priority order, from LAYOUT.

    Resolution spans ALL of them because a checker can live at repo-root
    `dev/tools` rather than under the primary tree. The list is data (see
    LAYOUT_FILE) so this module names no repo's private layout.
    """
    """Every plausible checker tree: AitherOS monorepo, generic dev/tools, bare
    tools/. Resolution spans ALL of them — check_actions_allowlist.py lives at
    repo-root dev/tools. The list is data (see LAYOUT_FILE)."""
    out: list[Path] = []
    for rel in LAYOUT["checker_roots"]:
        cand = root / rel
        if cand.is_dir():
            out.append(cand)
    return out


def detect_gate_tree(root: Path) -> Path | None:
    """The primary checker tree (first that exists)."""
    trees = candidate_trees(root)
    return trees[0] if trees else None


def _is_gate(name: str) -> bool:
    return name.startswith("check_") or name in NON_CHECK_GATES


def documented_gates(rules: Path) -> set[str]:
    """Basenames named in the quality-gate rules file."""
    text = rules.read_text(encoding="utf-8", errors="replace")
    names = set()
    for m in TOKEN_RE.finditer(text):
        name = m.group(1)
        if _is_gate(name):
            names.add(name)
    return names


def wired_in_workflow(workflow: Path) -> set[str]:
    """Basenames of checkers invoked on a workflow run line."""
    text = workflow.read_text(encoding="utf-8", errors="replace")
    names = set()
    for m in WF_RUN_RE.finditer(text):
        base = Path(m.group(1)).name
        if _is_gate(base):
            names.add(base)
    return names


def wired_in_routines(routines_dir: Path, probe: Path) -> set[str]:
    """Basenames of checkers named in the gate routine or any routine yaml."""
    names: set[str] = set()
    for source in (probe,):
        if source.is_file():
            text = source.read_text(encoding="utf-8", errors="replace")
            for m in ROUTINE_STR_RE.finditer(text):
                names.add(Path(m.group(1)).name)
    if routines_dir.is_dir():
        for yml in sorted(routines_dir.glob("*.yaml")):
            text = yml.read_text(encoding="utf-8", errors="replace")
            for m in ROUTINE_STR_RE.finditer(text):
                names.add(Path(m.group(1)).name)
    return names


def resolve(name: str, trees: list[Path]) -> Path | None:
    """A file under any gate tree matching the basename (top-level or nested)."""
    for tree in trees:
        direct = tree / name
        if direct.is_file():
            return direct
        for hit in tree.rglob(name):
            if hit.is_file():
                return hit
    return None



# Handlers that mean "I could not read this input".
# Deliberately narrow: only the "I could not ANALYSE this source" family. The first
# version also listed OSError/ValueError/FileNotFoundError and produced a flood (100+
# hits) — because a helper returning a default for a missing OPTIONAL file is correct,
# not a defect. A rule that floods gets switched off, which is how this repo's
# per-file-ignores came to exist, so the rule is scoped to the shape that actually
# caused all three measured incidents: a PARSE failure reported as no findings.
_CANNOT_READ_EXC = {
    "SyntaxError", "UnicodeDecodeError", "IndentationError", "TokenError",
}
# Return values that mean "nothing to report" — i.e. CLEAN.
def _is_clean_return(node: "ast.AST") -> bool:
    """True if this return says 'no findings' rather than 'I could not judge'."""
    v = getattr(node, "value", None)
    if v is None:
        return True                                   # bare `return`
    if isinstance(v, ast.Constant) and v.value in (None, 0, True):
        return True
    if (isinstance(v, (ast.List, ast.Dict, ast.Set))
            and not getattr(v, "elts", None)
            and not getattr(v, "keys", None)):
        return True                                   # [] / {} / set()
    if isinstance(v, ast.Tuple):                      # ([], []) — the awgit shape
        return all(
            isinstance(e, (ast.List, ast.Dict, ast.Set))
            and not getattr(e, "elts", None) and not getattr(e, "keys", None)
            for e in v.elts
        ) and bool(v.elts)
    return False



# HYG004 baseline — checkers KNOWN to swallow a parse failure and report clean, pinned
# 2026-08-09. Keyed by FILE, not file:line: a line-keyed pin drifts on every unrelated
# edit above it and would false-positive its way into being muted. The gate fires when a
# NEW checker joins this list; the list itself must SHRINK, never grow.
#
# Two were fixed the day this rule was written, which is why they are absent:
# check_undefined_names' main analysis path (11 shipped service modules were invisible to
# it because a BOM made ast.parse raise) and check_async_blocking (same BOM blindness, so
# PQ010's blocking-call gate was clean on those same files). The rest are real and open.
_HYG004_BASELINE = {
    "check_capability_surfaced.py",
    "check_llm_facade_conformance.py",
    "check_local_call_signatures.py",
    "check_mcp_apps_contract.py",
    "check_nexus_client_contract.py",
    "check_python_quality.py",
    "check_service_registry_resolves.py",
    "check_undefined_names.py",
}

# HYG008 — a checker that talks to the container engine as `docker` ONLY.
#
# This fleet runs rootful podman in WSL2. A checker that shells `docker ps` does not
# report a smaller truth here, it reports NOTHING: every call fails, the tool exits 2,
# and the invariant it guards goes unwatched for as long as nobody reads the exit code.
# `check_container_import_errors.py` was found in exactly that state on 2026-08-13,
# minutes after a new rule was added to it — the rule could never have run.
#
# The fix is the ladder both check_lb_upstream_liveness.py and (now)
# check_container_import_errors.py use: PROBE for the engine, docker first so a box
# mid-cutover still works, podman-in-WSL second, and say which one answered.
#
# Pinned, not zero: 36 of the 64 engine-touching checkers were docker-only when this
# rule was written, and a gate that opens red gets bypassed rather than satisfied —
# which is exactly how this repo's per-file-ignores came to exist. The count must
# SHRINK. A checker newly hardcoding docker fails immediately; remediating one lowers
# the pin in the same commit.
# 36 -> 32 on 2026-08-15. One of those four is check_wal_retention_bounded.py,
# which was docker-only and therefore had NEVER judged the WAL invariant it was
# written for — every run returned NOT VERIFIED. It is exactly the state this
# rule's own comment describes, found by asking the fleet rather than the tool.
_HYG008_PIN = 32

_ENGINE_CALL_RE = re.compile(r'["\']docker["\']|\bdocker\s+(?:ps|inspect|exec|logs)\b')
_ENGINE_LADDER_RE = re.compile(r"_engine_prefix|def engine\(|podman")


def _hyg8(trees: list[Path]) -> list[str]:
    """HYG008 — engine-touching checkers with no podman path. Returns offenders."""
    out: list[str] = []
    for tree in trees:
        for f in sorted(tree.glob("check_*.py")):
            try:
                src = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not _ENGINE_CALL_RE.search(src):
                continue
            if _ENGINE_LADDER_RE.search(src):
                continue
            out.append(f.name)
    return sorted(set(out))

#: EMPTY as of 2026-08-11, and it must stay that way. All eight entries were
#: discharged rather than re-pinned: three checkers now record unparseable files and
#: exit 2, one records None as an explicit "cannot judge" (check_python_quality, which
#: OWNS syntax via ruff and must not go DEAD over a file it is already reporting), and
#: the rest were FALSE POSITIVES this rule no longer produces -- `literal_eval` on an
#: already-parsed node, and a `return None` from a function whose signature DECLARES
#: the sentinel. The root cause under all of it was a UTF-8 BOM on eleven service
#: files, which made them unparseable to every ast-based gate while they ran fine.
#: A NEW entry here is a regression, not a backlog: fix the checker instead.
_HYG004_BASELINE: set[str] = set()

def _hyg4(trees: list[Path]) -> list[str]:
    """HYG004 — a checker that swallows an unreadable input and reports CLEAN.

    THE PATTERN, measured three separate times on 2026-08-09 and each time it made a
    working gate blind rather than noisy:
      * check_undefined_names read files as plain utf-8, so a leading BOM raised
        SyntaxError and `except SyntaxError: return [], []` printed
        "OK: no unresolvable names" for 11 shipped service modules it never parsed —
        including the mesh fabric, which was hiding a guaranteed NameError;
      * sqlite_store_integrity scanned one tree while the service's databases lived in
        two others, so it ran clean over three unopenable stores (7158 rows recovered);
      * awgit capture returned [] for an unparseable file, and an empty node set diffs
        as DELETION — so a conflicted file was recorded as "every function deleted".
    Three instances is a class, not a coincidence, and the rule these tools already
    state is "exit non-zero when it CANNOT run — a probe that cannot emit a verdict is
    DEAD, never passing". `empty` and `unknown` are different answers.

    Flags an `except` handler catching a cannot-read exception whose body returns a
    clean/empty value with NO other action — no raise, no recording, no non-zero exit.
    A handler that logs AND records, or re-raises, or returns a sentinel, is fine.
    """
    out: list[str] = []
    for tree in trees:
        for f in sorted(tree.rglob("check_*.py")):
            try:
                mod = ast.parse(f.read_text(encoding="utf-8-sig", errors="replace"))
            except SyntaxError:
                # This checker cannot parse a checker — that is itself an unknown, so
                # say so rather than passing over it (the very rule being enforced).
                out.append(f"{f.name}: could not be parsed by HYG004")
                continue
            for h in (n for n in ast.walk(mod) if isinstance(n, ast.ExceptHandler)):
                names = set()
                t = h.type
                for node in ast.walk(t) if t is not None else []:
                    if isinstance(node, ast.Name):
                        names.add(node.id)
                    elif isinstance(node, ast.Attribute):
                        names.add(node.attr)
                if not (names & _CANNOT_READ_EXC):
                    continue
                # Any escalation in the handler makes it honest.
                escalates = any(
                    isinstance(n, ast.Raise) for n in ast.walk(h)
                ) or any(
                    isinstance(n, ast.Call) and (
                        getattr(getattr(n, "func", None), "id", "") in ("exit", "SystemExit")
                        or getattr(getattr(n, "func", None), "attr", "") in (
                            "exit", "append", "add", "warning", "error")
                    )
                    for n in ast.walk(h)
                )
                if escalates:
                    continue
                for n in h.body:
                    if isinstance(n, ast.Return) and _is_clean_return(n):
                        out.append(f"{f.name}:{n.lineno}: "
                                   f"except {sorted(names & _CANNOT_READ_EXC)} returns "
                                   f"CLEAN — unreadable input reported as no findings")
    return out



_MARKER_NAME = "canonical-deploy-root"


def _marker_parse_violations(tools_dir: Path) -> list[str]:
    """HYG006 — a tool that reads the deploy-root marker WHOLE instead of line-wise.

    The marker carries explanatory comment lines before the path. Reading it with
    `read_text().strip()` therefore yields a string that is not a directory, the
    is_dir()/exists() test fails, and the tool falls back to the WORKING TREE —
    silently operating on the wrong root. Nothing errors; the tool just answers
    about a different tree than the fleet deploys from.

    Measured 2026-08-10 in provision_quadlet_env.py: it resolved credentials from
    D:'s .env (49 names) instead of the deploy root's (53) and refused to
    provision MinIO for want of a password sitting in the file it was not
    reading. check_image_store_integrity.py had the identical bug.

    LIMITATION, stated because a checker that oversells itself is worse than a
    narrow one: this is a FILE-LEVEL heuristic. It asks whether the file contains
    a comment-skipping idiom at all, so a tool that parses the marker correctly in
    one place and carelessly in another passes. It reliably catches the real shape
    — a tool that never handles comments anywhere, which is how BOTH live
    instances looked — and deliberately does not attempt dataflow it cannot do
    accurately.

    BOTH correct idioms are accepted — `startswith("#")` skipping and
    `split("#", 1)[0]`. An earlier version of this detector knew only the first
    and produced a FALSE POSITIVE on check_baked_code_staleness.py, which was
    correct all along; a rule that cries wolf gets switched off.
    """
    out: list[str] = []
    for f in sorted(tools_dir.glob("*.py")):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _MARKER_NAME not in src:
            continue
        linewise = ('startswith("#")' in src or "startswith('#')" in src
                    or 'split("#", 1)[0]' in src or 'split("#",1)[0]' in src)
        if not linewise:
            out.append(f"{f.name}: reads the {_MARKER_NAME} marker without "
                       f"skipping its comment lines — falls back to the working "
                       f"tree and silently uses the wrong root")
    return out


def _collect(root: Path):
    """Read every source once and return the verdict buckets.

    Returns (hyg1, hyg2, hyg3, hyg5, report) where hyg1/2/3/5 are violation
    lists and report is a dict of non-gating stats.
    """
    trees = candidate_trees(root)
    tree = trees[0] if trees else None
    rules = root / ".claude/rules/quality-gate.md"
    workflow = root / ".github/workflows/debt-invariants.yml"
    routines_dir = root / LAYOUT["routines_dir"] if LAYOUT["routines_dir"] else None
    probe = root / LAYOUT["debt_probe"] if LAYOUT["debt_probe"] else None

    if tree is None or not rules.is_file() or not workflow.is_file():
        raise RuntimeError(
            "gate tree=%s rules=%s workflow=%s — a source is missing" % (
                tree, rules.is_file(), workflow.is_file()
            )
        )

    doc = documented_gates(rules)
    wired = wired_in_workflow(workflow) | wired_in_routines(routines_dir, probe)

    hyg1: list[str] = []
    hyg2 = sorted(n for n in doc if resolve(n, trees) is None)
    hyg3 = sorted(n for n in wired if resolve(n, trees) is None)
    hyg9 = _untracked_live(root, doc | wired, resolve, trees)
    # HYG002 asks TWO questions, because disk presence is not the bar. The second was
    # asymmetric until 2026-08-11: HYG003 already refused an untracked WIRED tool
    # ("absent on a CI checkout") while HYG002 accepted an untracked DOCUMENTED one.
    # A rule pointing at a file that exists on exactly one machine documents nothing.
    hyg2_raw = [n for n in doc if resolve(n, trees) is None]
    for _n in sorted(documented_tools(rules, trees)):
        _p = resolve(_n, trees)
        if _p is not None and not _git_tracked(root, _p):
            hyg2_raw.append(f"{_n} (on disk but UNTRACKED — absent on a CI checkout)")
    hyg2 = sorted(set(hyg2_raw))
    # Disk presence is not enough: CI runs a fresh checkout, so an UNTRACKED file
    # is absent there while the gate goes green locally.
    hyg3_raw = []
    for _name in wired:
        _path = resolve(_name, trees)
        if _path is None:
            hyg3_raw.append(_name)
        elif not _git_tracked(root, _path):
            hyg3_raw.append(f"{_name} (on disk but UNTRACKED — absent on a CI checkout)")
    hyg3 = sorted(set(hyg3_raw))

    for name in sorted(doc | wired):
        found = resolve(name, trees)
        if found is None:
            continue
        src = found.read_text(encoding="utf-8", errors="replace")
        if "--self-test" not in src:
            hyg1.append(name)

    all_checkers = sorted(
        p.name for p in tree.rglob("check_*.py") if p.is_file()
    )
    no_selftest_tree = []
    for n in all_checkers:
        f = resolve(n, trees)
        if f is None:
            continue
        if "--self-test" not in f.read_text(encoding="utf-8", errors="replace"):
            no_selftest_tree.append(n)

    hyg4 = _hyg4(trees)
    hyg5 = _hyg5(root)

    report = {
        "total_checkers": len(all_checkers),
        "no_selftest_tree": no_selftest_tree,
        "wired_nowhere": sorted(
            n for n in all_checkers if n not in doc and n not in wired
        ),
        "wired_but_undocumented": sorted(n for n in wired if n not in doc),
        "documented_count": len(doc),
        "wired_count": len(wired),
    }
    hyg6 = _marker_parse_violations(root / 'AitherOS' / 'dev' / 'tools')
    hyg7 = _hyg7(root, trees)
    hyg8 = _hyg8(trees)
    return hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, hyg7, hyg8, hyg9, report


def _hyg7(root: Path, trees: list[Path]) -> list[str]:
    """HYG007 — a deploy-playbook phase gate must name a checker that EXISTS.

    Why this is a gate and not a convention
    ---------------------------------------
    AitherOS/config/deploy/phases/*.yaml drive orchestrate_fleet.py, and each phase declares
    `gate.tool`. If that tool does not exist on disk the phase does not fail loudly -- the
    orchestrator finds nothing to run, and the phase reads as SATISFIED. A gate that names a
    missing tool therefore ALWAYS PASSES, which is strictly worse than having no gate: the
    playbook reports a verified deployment it never verified.

    Added 2026-08-11, when the phases were authored by agents. The rule "never invent a
    checker" was given to them as an INSTRUCTION, and an instruction is not an assertion --
    the same reasoning that turned the python-quality rules into check_python_quality.py
    after one of seven was found actually enforced. A later rename of any check_*.py would
    also silently hollow out every phase that gated on it.

    Skipped cleanly when the phases tree is absent (an adopted repo has none), which is a
    correct skip rather than a pass-by-omission.
    """
    violations: list[str] = []
    phases = root / "AitherOS" / "config" / "deploy" / "phases"
    if not phases.is_dir():
        return violations
    try:
        import yaml  # noqa: PLC0415 -- optional dep; absence must not crash the whole gate
    except ImportError:
        # Cannot parse => cannot judge. Surfaced as a violation rather than silence, because
        # "I could not look" must never render as "nothing is wrong" (HYG004's whole lesson).
        return ["(pyyaml missing — phase gate tools NOT VERIFIED)"]

    for f in sorted(phases.glob("*.yaml")):
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception:  # noqa: BLE001 -- an unparseable phase is itself a finding
            violations.append(f"{f.name}: unparseable YAML — gate NOT VERIFIED")
            continue
        # `gate:` appears in BOTH shapes across the authored phases: a mapping with an
        # explicit `tool:`, and a free-text string describing the command. A first version
        # of this rule assumed the mapping and crashed on the string form -- caught by its
        # own negative control, which is the only reason it is not silently skipping those
        # files today. Handle both; anything else is a finding, not an exception.
        gate = doc.get("gate")
        tool = None
        if isinstance(gate, dict):
            # orchestrate_fleet.py reads gate["tool"] (line ~250). Phases authored later
            # declare the checker under gate["name"] instead, so the orchestrator resolves
            # None and RUNS NOTHING -- the phase passes without ever being verified.
            # Measured 2026-08-11: 7 of 11 phases, INCLUDING 08-inference-engine and
            # 09-embeddings-boot, whose gate is the DGX memory headroom check that exists
            # specifically to stop a model service starting into a starved pool. A grep for
            # the checker NAME finds it in all of them and looks correct; only asking the
            # key the orchestrator actually reads exposes it. Assert `tool`, and report a
            # `name`-only gate as the silent-pass defect it is.
            tool = gate.get("tool")
            if not tool and gate.get("name"):
                violations.append(
                    f"{f.name}: gate declares '{gate.get('name')}' under `name`, but "
                    f"orchestrate_fleet.py reads `gate.tool` — this gate NEVER RUNS and "
                    f"the phase always passes"
                )
                continue
        elif isinstance(gate, str):
            m = re.search(r"(check_[A-Za-z0-9_]+\.py)", gate)
            tool = m.group(1) if m else None
            if tool is None:
                continue  # a prose gate naming no checker: an inline command, allowed
        if not tool:
            # A phase with no gate cannot fail. That is the defect this family exists for.
            violations.append(f"{f.name}: phase declares NO gate.tool — it can never fail")
            continue
        if not str(tool).endswith(".py"):
            continue  # an inline command, deliberately allowed; nothing to resolve
        if resolve(str(tool), trees) is None:
            violations.append(
                f"{f.name}: gate.tool '{tool}' does not exist on disk — this phase "
                f"ALWAYS PASSES"
            )
    return violations
    hyg7 = _duplicate_gate_ids(root / '.claude' / 'rules' / 'quality-gate.md')
    return hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, hyg7, report


def run(root: Path, show_all: bool) -> int:
    try:
        hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, hyg7, hyg8, hyg9, report = _collect(root)
        hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, hyg7, report = _collect(root)
    except RuntimeError as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        return 2

    findings = False
    for code, label, items in (
        ("HYG001", "documented-or-wired checkers with no --self-test", hyg1),
        ("HYG002", "documented checkers missing on disk", hyg2),
        ("HYG003", "wired checkers missing on disk", hyg3),
        ("HYG004", "checkers that report CLEAN on input they could not read (NEW)",
         [x for x in hyg4 if x.split(":")[0] not in _HYG004_BASELINE]),
        ("HYG005", "discipline copies drifted across delivery homes", hyg5),
        ("HYG006", "tools parsing the deploy-root marker whole (wrong root)", hyg6),
        ("HYG007", "deploy-playbook phase gates naming a checker that does not exist "
                   "(such a phase ALWAYS PASSES)", hyg7),
        ("HYG008", f"docker-only checkers — inert on this podman fleet "
                   f"(pin {_HYG008_PIN}, must shrink)",
         hyg8 if len(hyg8) > _HYG008_PIN else []),
    ):
        if items:
            findings = True
            _safe_print(f"{code}: {label}:")
            for n in items:
                _safe_print(f"  - {n}")

    if len(hyg8) <= _HYG008_PIN:
        _safe_print(
            f"HYG008 baseline: {len(hyg8)} docker-only checker(s) of the engine-touching "
            f"set — pin {_HYG008_PIN}, must shrink, never grow"
        )
        if len(hyg8) < _HYG008_PIN:
            _safe_print(f"  ratchet: lower _HYG008_PIN to {len(hyg8)} in this commit")

    _safe_print(
        f"HYG009 baseline: {len(hyg9)} documented-or-wired checker(s) git is NOT tracking "
        f"— pin {HYG009_PIN}, must shrink, never grow. A wired gate that is not committed "
        f"resolves to nothing in every clone and every CI run:"
    )
    for _n in hyg9:
        _safe_print(f"  [untracked] {_n}")
    if len(hyg9) > HYG009_PIN:
        findings = True
        _safe_print(
            f"HYG009: {len(hyg9)} untracked live checker(s) exceeds the pin of "
            f"{HYG009_PIN} — fix is a commit, not an allowlist entry"
        )
    elif len(hyg9) < HYG009_PIN:
        findings = True
        _safe_print(
            f"HYG009: pin is stale — {len(hyg9)} untracked, pin says {HYG009_PIN}. "
            f"Lower HYG009_PIN in the same commit that fixed them, or the win is given back."
        )

    known4 = [x for x in hyg4 if x.split(":")[0] in _HYG004_BASELINE]
    if known4:
        _safe_print(
            f"HYG004 baseline: {len(known4)} known parse-swallowing site(s) in "
            f"{len(_HYG004_BASELINE)} file(s) — must shrink, never grow:"
        )
        for n in known4:
            _safe_print(f"  [known] {n}")

    if show_all or not findings:
        r = report
        _safe_print(f"tree: {r['total_checkers']} check_*.py | "
                    f"{len(r['no_selftest_tree'])} without --self-test | "
                    f"{len(r['wired_nowhere'])} wired nowhere | "
                    f"{len(r['wired_but_undocumented'])} wired-but-undocumented")
        if show_all:
            for n in r["wired_but_undocumented"]:
                _safe_print(f"  [report] wired but not documented: {n}")
            for n in r["no_selftest_tree"]:
                _safe_print(f"  [report] no --self-test (tree-wide): {n}")
            for n in r["wired_nowhere"]:
                _safe_print(f"  [report] wired nowhere (not in rules/CI/routines): {n}")

    if findings:
        n_viol = len(hyg1) + len(hyg2) + len(hyg3) + len(hyg5)
        _safe_print(f"HYGIENE: FAIL ({n_viol} violation(s))")
        return 1
    _safe_print("HYGIENE: ok")
    return 0


def self_test() -> int:
    """Prove the gate can still fail: feed it a rules file naming a checker
    that does not exist and assert HYG002 fires and the exit code is 1."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".claude/rules").mkdir(parents=True)
        (root / "dev/tools").mkdir(parents=True)
        (root / ".github/workflows").mkdir(parents=True)
        (root / ".claude/rules/quality-gate.md").write_text(
            "run python dev/tools/check_nonexistent_thing_xyz.py\n",
            encoding="utf-8",
        )
        (root / ".github/workflows/debt-invariants.yml").write_text(
            "name: debt\non: {pull_request: {branches: [develop]}}\n"
            "jobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: python dev/tools/check_other_missing_abc.py\n",
            encoding="utf-8",
        )
        # A real checker with no self-test, so HYG001 can also fire.
        (root / "dev/tools/check_real_but_no_selftest.py").write_text(
            '"""a gate without a self-test"""\nimport sys\nsys.exit(0)\n',
            encoding="utf-8",
        )
        (root / ".claude/rules/quality-gate.md").write_text(
            "run python dev/tools/check_nonexistent_thing_xyz.py\n"
            "run python dev/tools/check_real_but_no_selftest.py\n",
            encoding="utf-8",
        )
        # A drifted copy across delivery homes, so the parity rule can also fire. The pair
        # is taken from PARITY (data) rather than written as literals: this module
        # ships publicly and must name no repo's private layout. With no parity
        # registry — the portable copy — there is nothing for it to assert, and
        # the fixture is skipped rather than faked.
        # A drifted copy across delivery homes, so the parity rule can also fire.
        # The pair comes from PARITY (data), never literals: this module ships
        # publicly and must name no repo's private layout. With no parity registry
        # — the portable copy — there is nothing for it to assert, so the fixture
        # is skipped rather than faked.
        if PARITY:
            canon_rel, copy_rel = PARITY[0]
            for rel, body in ((canon_rel, "canonical\n"), (copy_rel, "DRIFTED\n")):
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body, encoding="utf-8")
        # HYG004: a swallowing handler must fire; a subscript-recording one must not.
        (root / "dev/tools/check_swallows_parse.py").write_text(
            '"""x"""\nimport ast\n\n\ndef scan(p):\n'
            "    try:\n        return ast.parse(p)\n"
            "    except SyntaxError:\n        return None\n\n\n"
            'def self_test():\n    return 0\n',
            encoding="utf-8",
        )
        (root / "dev/tools/check_records_parse.py").write_text(
            '"""x"""\nimport ast\nBAD = {}\n\n\ndef scan(p):\n'
            "    try:\n        return ast.parse(p)\n"
            "    except SyntaxError as exc:\n"
            "        BAD[p] = type(exc).__name__\n        return None\n\n\n"
            'def self_test():\n    return 0\n',
            encoding="utf-8",
        )
        hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, _hyg7, _hyg8, hyg9, _ = _collect(root)
        if not any("check_swallows_parse" in x for x in hyg4):
            print("SELF-TEST FAIL: HYG004 did not fire on a swallowing handler",
                  file=sys.stderr)
            return 1
        if any("check_records_parse" in x for x in hyg4):
            print("SELF-TEST FAIL: HYG004 fired on a handler that RECORDS the failure",
                  file=sys.stderr)
            return 1
        hyg1, hyg2, hyg3, hyg4, hyg5, hyg6, _hyg7, _ = _collect(root)
        # HYG002's SECOND half specifically: a documented tool that is present on disk
        # but not tracked. Asserted separately from "hyg2 is non-empty", because the
        # missing-file half alone satisfies that and would let the untracked half rot
        # silently -- which is the failure mode this whole gate exists to name.
        if not any("UNTRACKED" in x for x in hyg2):
            print(
                "SELF-TEST FAIL: HYG002 did not flag a documented-but-untracked tool; "
                f"got {hyg2!r}",
                file=sys.stderr,
            )
            return 1
        if PARITY and not hyg5:
            print("SELF-TEST FAIL: expected the parity rule to fire", file=sys.stderr)
            return 1
        if not PARITY:
            print("  parity rule: NOT APPLICABLE — no parity registry in this copy")
        if not hyg1 or not hyg2 or not hyg3:
            print(
                "SELF-TEST FAIL: expected HYG001/HYG002/HYG003/HYG005 to "
                f"fire, got {hyg1!r} {hyg2!r} {hyg3!r} {hyg5!r}",
                file=sys.stderr,
            )
            return 1
        code = run(root, show_all=False)
        if code != 1:
            print(f"SELF-TEST FAIL: expected exit 1, got {code}", file=sys.stderr)
            return 1
    _safe_print("SELF-TEST: ok - the gate can still fail")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="prove the gate can still fail")
    ap.add_argument("--all", action="store_true",
                    help="print the full tree backlog (report, never a gate)")
    ap.add_argument("--root", default=None,
                    help="repo root (default: git rev-parse from cwd)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    root = Path(args.root).resolve() if args.root else find_root(Path.cwd())
    return run(root, show_all=args.all)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a probe that cannot judge is never a pass
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        sys.exit(2)
