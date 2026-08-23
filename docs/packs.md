# Agent packs

Each pack is a standalone download. Take one, run its installer, and adk finds
it — you do not need the rest of the framework to try a single pack.

```bash
tar xzf <pack>-<version>.tar.gz
python <pack>/install.py
```

That copies the pack to `~/.aither/packs/<name>/`, a location adk discovers with
no configuration, then **verifies** the pack is discoverable rather than assuming
it. If adk is not installed yet the installer says so and still places the files,
so the order does not matter:

```bash
pip install aither-adk
```

Every artifact ships a `.sha256` next to it. Verify before you trust it:

```bash
sha256sum -c <pack>-<version>.sha256
```


Built from `v3.7.3` (adk 3.7.3).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.7.3` | [aither-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/aither-3.7.3.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.7.3` | [analyst-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/analyst-3.7.3.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.7.3` | [bead-space-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/bead-space-3.7.3.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.7.3` | [claude-code-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/claude-code-3.7.3.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.7.3` | [gobbonet-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/gobbonet-3.7.3.tar.gz) | 40.1 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.7.3` | [hermes-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/hermes-3.7.3.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.7.3` | [iris-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/iris-3.7.3.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.7.3` | [openclaw-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/openclaw-3.7.3.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.7.3` | [persona-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/persona-3.7.3.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.7.3` — skills  
  `sha256:549d8f5e41105ead…`
- **analyst** `3.7.3` — agent config, skills  
  `sha256:ba494c67f6486734…`
- **bead-space** `3.7.3` — brain pack only  
  `sha256:2baa9eed5747fb76…`
- **claude-code** `3.7.3` — agent config, skills  
  `sha256:2caa3b0b3c002106…`
- **gobbonet** `3.7.3` — agent config, Python  
  `sha256:2625e6c4a13cf581…`
- **hermes** `3.7.3` — agent config, skills  
  `sha256:e520b9f7019bfa3e…`
- **iris** `3.7.3` — skills  
  `sha256:ee258fe13edc0935…`
- **openclaw** `3.7.3` — agent config, skills  
  `sha256:262a6756405b7d48…`
- **persona** `3.7.3` — brain pack only  
  `sha256:6c577bfad07229e7…`
