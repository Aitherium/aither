---
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, PowerShell
description: Install the AitherOS coding discipline — debt-invariant checks, stop hooks, self-skeptic, live proof — onto any coding agent, including aither-adk agents. A defect becomes a CHECK (not a row); every check is self-tested and wired; the stop hook makes skipping a stated act.
argument-hint: [install | add-check | audit | explain]
---

# Code Discipline — the AitherOS gate stack you can install

The doctrine underneath AitherOS's `debt-ledger` and `prove-it-live` skills is
only half the battle. The other half is that a **defect found becomes a check,
not a row**, and that something **enforces** the turn: a stop hook that blocks
ending a code change without a ledger row, a check, or a stated
"no new debt". This skill installs the whole stack — doctrine, decision table,
check contract, hooks, and the runnable assertion of the process itself — onto
whatever coding agent you have, including aither-adk.

> Pair with `code-like-david` (the operating doctrine) and `debt-ledger` +
> `prove-it-live` (the two hooks). This skill is the *enforcement layer*: it
> makes the doctrine load-bearing instead of vibes.

---

## The one-line doctrine

**A mechanically-detectable defect becomes a CHECK, not a ledger row; a row is
for the genuinely one-off; the stop hook makes skipping that a *stated* act.**

The checkers in the repo's gate tree (`dev/tools/check_*.py` under the platform
source root, enumerated in the repo's `.claude/rules/quality-gate.md`) each
exist because a specific defect class cost a real session. Every gate in the
platform's `.github/workflows/debt-invariants.yml` and its `config/routines/*.yaml`
is one of those lessons turned into code.

## The decision table — what earns what

| you found | discharge |
|---|---|
| a defect a static check could detect | **add/extend a checker + a test** (no row) |
| a defect only a live probe can see | **add it to a routine that pages** on failure (no row) |
| genuinely one-off / needs a human / legal | **a row** in the debt ledger |
| you fixed it | **a commit** (not debt) |
| nothing | **state it**: "checked, no new debt: \<one line why\>" — silence is not an answer |

## The check contract — every check must

1. **have a `--self-test` that proves it can still fail.** A checker nobody has
   watched fail is not a gate. The self-test feeds the checker synthetic input
   that should trip its detection and asserts it goes red.
2. **exit non-zero (2), never 0, when it CANNOT run.** A probe that cannot emit
   a verdict is DEAD, not passing. Silence is not a pass.
3. **be wired somewhere unattended** — the platform's `debt-invariants.yml` CI
   workflow if it needs no fleet, a routine if it does. A gate that depends on
   being remembered is documentation, not enforcement.
4. **name a debt id in any allowlist entry**, and print the allowlist on every
   run so it cannot quietly grow.

The bundled `tools/check_checker_hygiene.py` asserts rules 1–3 against the
checkers themselves (see `audit` below) — the process codified.

## The stop-hook contract

The bundled `hooks/stop_debt_ledger.py` is a Stop hook: it runs when the agent
tries to end a turn and can refuse.

| turn | decision |
|---|---|
| wrote a file, ledger untouched, no check/test added | **block** |
| wrote a file, also wrote the ledger | allow |
| wrote a file, also added/extended a check or a test | allow (a check is MORE than the ledger asked for) |
| wrote a file, said "no new debt: \<why\>" | allow |
| wrote nothing | allow |
| already continuing from a block | allow |

The escape hatches are load-bearing: a gate with no legitimate way through gets
disabled within a week. What they buy is not enforcement of the outcome — it is
that skipping the check becomes a **stated** act instead of silence.

The AitherOS-native hook (`stop-tech-debt-ledger.ps1`) already implements the
"a check or test was added" discharge. The portable Python version here is the
copy-anywhere twin.

## Self-skeptic + live proof

- A feeling is not a check. Require a check that can FAIL — a live round-trip, a
  positive assertion that data actually flows, terminal output shown.
- Watch for the **silent no-op**: a fail-closed path that always returns empty
  passes every "returns nothing" assertion while being completely inert.
- Every delivery: state at least one concrete weakness, limitation, or untested
  assumption — then fix it or flag it. "Looks right to me" is not a check.

---

## What this skill does when invoked

### `install` — put the discipline on this agent

1. **Inspect read-only first.** Look for an existing rules directory, a debt
   ledger, and a hooks setup. Overwrite nothing — merge and append.
2. **Claude Code (native):** copy the portable bundle and wire the Stop hooks:
   ```bash
   mkdir -p .claude/hooks .claude/tools
   cp hooks/hook_common.py hooks/stop_debt_ledger.py hooks/stop_live_proof.py .claude/hooks/
   cp tools/check_checker_hygiene.py tools/next_debt_id.py .claude/tools/
   python3 hooks/test_hooks.py          # prove the hooks work before wiring them
   ```
   Then add the `Stop` entries from `hooks/README.md` to `.claude/settings.json`
   (merge — do not replace the file) and restart the agent. Seed the ledger if
   none exists: a one-table `TECH_DEBT.md` at the repo root with the row shape
   from the `debt-ledger` skill.
3. **aither-adk:** the discipline ships as the `code-discipline` pack
   (`adk install pack:code-discipline`), or merge the doctrine into an existing
   pack's system prompt (`adk pack customize <pack> --system-prompt "<doctrine>"`).
   ADK agents have no end-of-turn hook, so enforcement there is in-prompt plus
   the bundled runnable tools — the pack's system prompt tells the agent to end
   every code turn with a ledger row, a check, or a stated "checked, no new
   debt". See `packs/code-discipline/` in this pack.
4. **Any other harness:** copy the bundle into the agent's working tree, wire
   whatever end-of-turn hook it exposes using the transport in `hooks/README.md`,
   and install the doctrine into its rules/context file.
5. **Report** exactly what was written where and what was skipped.

### `add-check` — THE codified procedure

Given a defect (or a finding that "the feature silently returns nothing"):

1. **Run the decision table.** Static-detectable → check. Live-probe-only →
   routine. One-off / needs-human → row. Fixed → commit. Nothing → state it.
2. **If a check is the answer:** read the repo's `.claude/rules/quality-gate.md`
   first (the 1a–1z ladder names the exact commands). Prefer extending an
   existing multi-invariant checker over adding a new file.
3. **Scaffold it** in the repo's gate tree (`dev/tools/check_*.py`) with the
   contract: a `--self-test` that feeds synthetic input and asserts the check
   goes red, `exit(2)` when it cannot run, and an argparse `--self-test` flag.
4. **Wire it** into `.github/workflows/debt-invariants.yml` (static) or a
   `config/routines/*.yaml` (fleet), and add the quality-gate rule that documents
   when it runs. A wired gate nobody documented is an invisible gate.
5. **Verify it can fail before trusting it passes:** `python <checker> --self-test`
   must go red when you mutate its detection, then green on the real tree.
6. **Discharge the stop hook:** the new check IS the discharge. If you chose a
   row instead, run `tools/next_debt_id.py` immediately before appending (reading
   max+1 from the file is a race — ids collide when two agents read the same
   ledger seconds apart).

### `audit` — is the discipline holding?

```bash
python .claude/tools/check_checker_hygiene.py --all   # contract vs the checkers
python .claude/tools/next_debt_id.py --audit          # duplicate ids
```

(The pack-root `tools/` paths above are the source; after `install` the runnable
copies live at `.claude/tools/`.)

`check_checker_hygiene.py` reports HYG001–HYG003 as gates and the long-tail
backlog (wired-but-undocumented, wired-nowhere) as `--all` reporting only — a
gate that opens red fleet-wide gets bypassed rather than satisfied.

### `explain`

Walk the doctrine, the decision table and the hook contract with their
"why" — and the honest limits below.

---

## The bundled tools

| path | job |
|---|---|
| `tools/check_checker_hygiene.py` | asserts the check contract against the checkers themselves (HYG001 self-test, HYG002 documented refs exist, HYG003 wired refs exist, HYG005 the discipline's duplicated copies across delivery homes stay identical); stdlib-only, layout-detected, portable |
| `tools/next_debt_id.py` | collision-safe debt ids (reserves in a sidecar; never read max+1 from the file) |
| `hooks/hook_common.py` | the agent-agnostic hook transport (JSON in on stdin, block/allow out) |
| `hooks/stop_debt_ledger.py` | the debt Stop hook (code changed, ledger untouched → block) |
| `hooks/stop_live_proof.py` | the live-proof Stop hook (done/fixed claimed, nothing exercised → block) |
| `hooks/test_hooks.py` | 21 mutation-verified cases for both hooks |
| `hooks/README.md` | the install/wiring block and per-platform notes |

## Honest limits

- **The stop hook is a blunt instrument.** It detects "code changed, ledger
  didn't", which fires on genuinely debt-free changes too. That is why the
  stated-exemption path exists, and why the honest answer is sometimes just to
  say so and move on.
- **Evidence detection is a heuristic.** It reads command lines, so it is
  satisfied by running something irrelevant. It raises the floor; it proves
  nothing by itself.
- **No hook reads your diff.** The hooks see which tools ran and what the agent
  wrote — not whether the fix is correct. Correctness is the self-skeptic's job.
- **A gate that floods gets switched off.** The audit's backlog columns are
  reporting for exactly this reason; the gates are the three that should already
  hold.
- **ADK enforcement is in-prompt.** aither-adk agents have no end-of-turn hook,
  so the discipline there is a system prompt that must be obeyed and runnable
  tools the agent is told to run — a human reading the agent's closing message
  is the backstop.
