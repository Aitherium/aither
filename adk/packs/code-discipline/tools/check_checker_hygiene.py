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
                  their delivery homes — the portable twin in aither-skills,
                  the bundled aither-adk pack, and the published pack template.
                  A drift between copies is the "duplicated source of truth"
                  hazard code-like-david rule 11 calls active: editing one
                  copy silently orphans the others. Skipped for a layout that
                  does not carry a given home.

A probe that cannot read its sources exits 2, never 0 — silence is not a pass.
Use ``--self-test`` to prove it can still fail (it points the reference set at
a nonexistent file and asserts the gate goes red).

Portable twin lives at ``aither-skills/tools/check_checker_hygiene.py`` — same
logic, layout-detected root (AitherOS monorepo, generic ``dev/tools`` install,
or bare ``tools/``), so an external repo that adopted the discipline can run it.
"""
from __future__ import annotations

import argparse
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

# Artifacts intentionally duplicated across delivery homes: the native tree,
# the portable twin in aither-skills, the bundled aither-adk pack and the
# published pack template. HYG005 asserts each copy stays byte-identical to its
# canonical, so editing one home cannot silently orphan the others.
PARITY = [
    ("AitherOS/dev/tools/check_checker_hygiene.py",
     "aither-skills/tools/check_checker_hygiene.py"),
    ("AitherOS/dev/tools/check_checker_hygiene.py",
     "aither-adk/adk/packs/code-discipline/tools/check_checker_hygiene.py"),
    ("AitherOS/dev/tools/check_checker_hygiene.py",
     "aither-skills/packs/code-discipline/tools/check_checker_hygiene.py"),
    ("aither-skills/tools/next_debt_id.py",
     "aither-adk/adk/packs/code-discipline/tools/next_debt_id.py"),
    ("aither-skills/tools/next_debt_id.py",
     "aither-skills/packs/code-discipline/tools/next_debt_id.py"),
    ("aither-skills/skills/code-discipline.md",
     "aither-adk/adk/packs/code-discipline/skills/code-discipline.md"),
    ("aither-skills/skills/code-discipline.md",
     "aither-skills/packs/code-discipline/skills/code-discipline.md"),
    ("aither-skills/skills/debt-ledger.md",
     "aither-adk/adk/packs/code-discipline/skills/debt-ledger.md"),
    ("aither-skills/skills/debt-ledger.md",
     "aither-skills/packs/code-discipline/skills/debt-ledger.md"),
    ("aither-skills/skills/prove-it-live.md",
     "aither-adk/adk/packs/code-discipline/skills/prove-it-live.md"),
    ("aither-skills/skills/prove-it-live.md",
     "aither-skills/packs/code-discipline/skills/prove-it-live.md"),
    ("aither-adk/adk/packs/code-discipline/agent.yaml",
     "aither-skills/packs/code-discipline/agent.yaml"),
    ("aither-adk/adk/packs/code-discipline/brain_pack.yaml",
     "aither-skills/packs/code-discipline/brain_pack.yaml"),
]


def _hyg5(root: Path) -> list[str]:
    """Copy-parity violations across the discipline's delivery homes.

    A pair is asserted only when THIS layout carries both the canonical and the
    copy's home: a generic adopted repo has neither aither-skills nor the adk
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
    """Every plausible checker tree: AitherOS monorepo, generic dev/tools, bare
    tools/. Resolution spans ALL of them — check_actions_allowlist.py lives at
    repo-root dev/tools, not under AitherOS/dev/tools."""
    out: list[Path] = []
    for cand in (root / "AitherOS/dev/tools", root / "dev/tools", root / "tools"):
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


def _collect(root: Path):
    """Read every source once and return the verdict buckets.

    Returns (hyg1, hyg2, hyg3, hyg5, report) where hyg1/2/3/5 are violation
    lists and report is a dict of non-gating stats.
    """
    trees = candidate_trees(root)
    tree = trees[0] if trees else None
    rules = root / ".claude/rules/quality-gate.md"
    workflow = root / ".github/workflows/debt-invariants.yml"
    routines_dir = root / "AitherOS/config/routines"
    probe = root / "AitherOS/lib/routines/debt_gate_probe.py"

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
    return hyg1, hyg2, hyg3, hyg5, report


def run(root: Path, show_all: bool) -> int:
    try:
        hyg1, hyg2, hyg3, hyg5, report = _collect(root)
    except RuntimeError as exc:
        print(f"CANNOT RUN: {exc}", file=sys.stderr)
        return 2

    findings = False
    for code, label, items in (
        ("HYG001", "documented-or-wired checkers with no --self-test", hyg1),
        ("HYG002", "documented checkers missing on disk", hyg2),
        ("HYG003", "wired checkers missing on disk", hyg3),
        ("HYG005", "discipline copies drifted across delivery homes", hyg5),
    ):
        if items:
            findings = True
            _safe_print(f"{code}: {label}:")
            for n in items:
                _safe_print(f"  - {n}")

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
        # A drifted copy across delivery homes, so HYG005 can also fire.
        (root / "AitherOS/dev/tools").mkdir(parents=True)
        (root / "aither-skills/tools").mkdir(parents=True)
        (root / "AitherOS/dev/tools/check_checker_hygiene.py").write_text(
            "canonical\n", encoding="utf-8",
        )
        (root / "aither-skills/tools/check_checker_hygiene.py").write_text(
            "DRIFTED\n", encoding="utf-8",
        )
        hyg1, hyg2, hyg3, hyg5, _ = _collect(root)
        if not hyg1 or not hyg2 or not hyg3 or not hyg5:
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
