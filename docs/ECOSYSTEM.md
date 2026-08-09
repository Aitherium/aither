# The AitherOS ecosystem — one installer, five payloads

Aither ADK is the **single entry point** for the AitherOS client ecosystem. It is
not five separate setups — it is one installer that provisions the five payload
repos, each of which solves one job:

| Repo | What it is | Installed by |
|---|---|---|
| **[aither-adk](https://github.com/aitherium/aither-adk)** | The installer/orchestrator + agent SDK (this repo) | `pip install aither-adk` |
| **[aitherzero](https://github.com/aitherium/aitherzero)** | Ops automation — a PowerShell library of battle-tested automation scripts | `adk setup-all --with-stack <profile>` |
| **[aitherkvcache](https://github.com/aitherium/aitherkvcache)** | Near-optimal KV-cache quantization (sub-byte compression, 2.7× of the information-theoretic optimum) | `adk setup-all --only aitherkvcache` |
| **[aither-skills](https://github.com/aitherium/aither-skills)** | The agent skill catalog — reusable skills your agents invoke (`/awgit-setup`, …) | `adk setup-all --only aither-skills` |
| **[awgit](https://github.com/aitherium/awgit)** | Semantic version control on top of git — every commit becomes an edit-op on stable node ids, with verified-identity attribution and differential sync | `adk setup-all --only awgit` |

## One command

```bash
adk setup-all                    # the whole ecosystem
adk setup-all --only awgit,aitherkvcache,aither-skills   # just the three dev tools
adk setup-all --dry-run          # see the plan first
adk setup-all --only <p> --skip <q>   # select / exclude
```

Each step calls the payload's **real** installer (pip / git clone / the AitherZero
stack installer) over subprocess — it never reimplements one, so it never drifts
from the canonical setup. Every step is best-effort: a failure is reported in the
summary and does not abort the rest (unless `--strict`).

## What each payload does for an agent

- **aither-adk** — the agent framework itself: define agents as Python classes,
  serve them, wire inference (local or cloud), grow into fleets.
- **aitherzero** — the ops brain: automated infrastructure, deployment and
  recovery scripts your agents (or you) run. Heavy; opt-in via `--with-stack`.
- **aitherkvcache** — makes model serving cheaper: near-optimal KV-cache
  compression so long-context inference fits more into the same VRAM.
- **aither-skills** — the skill catalog: drop-in capabilities (`/awgit-setup`,
  `/website-as-code`, …) that any agent can invoke. Cloned into
  `~/.aither/skills`.
- **awgit** — the git layer with a world model: post-commit capture turns every
  commit into a semantic edit-op (which functions changed, who under a verified
  GitHub identity, a deterministic attribution handle per op). Node-level merges,
  leases, content-addressed bodies, differential sync to peer nodes.

## How they compose

An agent running on aither-adk, with aither-skills installed, can be taught
awgit's workflow (the `/awgit-setup` and `/awgit-claude-code` skills) so its
commits become attribution records. AitherZero automates the box around it. On a
serving node, aitherkvcache stretches the VRAM. The five repos are five layers of
one system — the installer makes them one command.
