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


Built from `v3.7.0` (adk 3.7.0).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.7.0` | [aither-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/aither-3.7.0.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.7.0` | [analyst-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/analyst-3.7.0.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.7.0` | [bead-space-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/bead-space-3.7.0.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.7.0` | [claude-code-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/claude-code-3.7.0.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.7.0` | [gobbonet-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/gobbonet-3.7.0.tar.gz) | 35.9 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.7.0` | [hermes-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/hermes-3.7.0.tar.gz) | 4.8 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.7.0` | [iris-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/iris-3.7.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.7.0` | [openclaw-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/openclaw-3.7.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.7.0` | [persona-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/persona-3.7.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.7.0` — skills  
  `sha256:d27aba57ec1b6cd9…`
- **analyst** `3.7.0` — agent config, skills  
  `sha256:f3c8112fa41dc693…`
- **bead-space** `3.7.0` — brain pack only  
  `sha256:806906bc2863186b…`
- **claude-code** `3.7.0` — agent config, skills  
  `sha256:a4576101a56dbaed…`
- **gobbonet** `3.7.0` — agent config, Python  
  `sha256:c776bc2939217120…`
- **hermes** `3.7.0` — agent config, skills  
  `sha256:7df1419b55cc8b4d…`
- **iris** `3.7.0` — skills  
  `sha256:0f518d97190d7dc0…`
- **openclaw** `3.7.0` — agent config, skills  
  `sha256:ab65b119e7494277…`
- **persona** `3.7.0` — brain pack only  
  `sha256:4564a9268a6cf1c0…`
