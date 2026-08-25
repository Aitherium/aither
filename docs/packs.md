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


Built from `v3.8.2` (adk 3.8.2).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.8.2` | [aither-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/aither-3.8.2.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.8.2` | [analyst-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/analyst-3.8.2.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.8.2` | [bead-space-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/bead-space-3.8.2.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.2` | [claude-code-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/claude-code-3.8.2.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.2` | [gobbonet-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/gobbonet-3.8.2.tar.gz) | 45.6 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.2` | [hermes-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/hermes-3.8.2.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.2` | [iris-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/iris-3.8.2.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.2` | [openclaw-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/openclaw-3.8.2.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.2` | [persona-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/persona-3.8.2.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.2` — skills  
  `sha256:4e5366932c4e693a…`
- **analyst** `3.8.2` — agent config, skills  
  `sha256:00e86bf26be6b150…`
- **bead-space** `3.8.2` — brain pack only  
  `sha256:978d3cfaa393cd09…`
- **claude-code** `3.8.2` — agent config, skills  
  `sha256:66d2a0992f519886…`
- **gobbonet** `3.8.2` — agent config, Python  
  `sha256:18182ae98efa6fdf…`
- **hermes** `3.8.2` — agent config, skills  
  `sha256:0229185086f83897…`
- **iris** `3.8.2` — skills  
  `sha256:bd0bb9bf4f1d433d…`
- **openclaw** `3.8.2` — agent config, skills  
  `sha256:086bc63ab461f363…`
- **persona** `3.8.2` — brain pack only  
  `sha256:3a9d2af6a5d5f143…`
