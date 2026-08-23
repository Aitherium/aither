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
  `sha256:6f672c933c9d3bef…`
- **analyst** `3.7.3` — agent config, skills  
  `sha256:2e8b371b60c1c0ed…`
- **bead-space** `3.7.3` — brain pack only  
  `sha256:bb10eebe85cb5162…`
- **claude-code** `3.7.3` — agent config, skills  
  `sha256:3c25904635ec181b…`
- **gobbonet** `3.7.3` — agent config, Python  
  `sha256:8e7c47bf32678902…`
- **hermes** `3.7.3` — agent config, skills  
  `sha256:f17edd857b20f578…`
- **iris** `3.7.3` — skills  
  `sha256:0ca7f460a2832036…`
- **openclaw** `3.7.3` — agent config, skills  
  `sha256:fd5c55d2f0fc590e…`
- **persona** `3.7.3` — brain pack only  
  `sha256:ec21b452464ed00a…`
