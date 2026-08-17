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


Built from `v3.3.0` (adk 3.3.0).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.3.0` | [aither-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/aither-3.3.0.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.3.0` | [analyst-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/analyst-3.3.0.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.3.0` | [bead-space-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/bead-space-3.3.0.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.3.0` | [claude-code-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/claude-code-3.3.0.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.3.0` | [gobbonet-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/gobbonet-3.3.0.tar.gz) | 23.4 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.3.0` | [hermes-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/hermes-3.3.0.tar.gz) | 4.8 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.3.0` | [iris-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/iris-3.3.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.3.0` | [openclaw-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/openclaw-3.3.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.3.0` | [persona-3.3.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.3.0/persona-3.3.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.3.0` — skills  
  `sha256:97e0b173e29f2f9a…`
- **analyst** `3.3.0` — agent config, skills  
  `sha256:3cb1f56d04f677ef…`
- **bead-space** `3.3.0` — brain pack only  
  `sha256:138be2afaa248083…`
- **claude-code** `3.3.0` — agent config, skills  
  `sha256:15f939082e0789bb…`
- **gobbonet** `3.3.0` — agent config, Python  
  `sha256:c52121b18888bcae…`
- **hermes** `3.3.0` — agent config, skills  
  `sha256:c562125a9306dcba…`
- **iris** `3.3.0` — skills  
  `sha256:ac6b8dff58d62fd5…`
- **openclaw** `3.3.0` — agent config, skills  
  `sha256:e205b6df30f03f23…`
- **persona** `3.3.0` — brain pack only  
  `sha256:d82159e2677ae029…`
