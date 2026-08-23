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
  `sha256:9c47415a2c7f0203…`
- **analyst** `3.7.0` — agent config, skills  
  `sha256:9f582c1ce7f379e7…`
- **bead-space** `3.7.0` — brain pack only  
  `sha256:e561e6568f3a636c…`
- **claude-code** `3.7.0` — agent config, skills  
  `sha256:5528c0161fbb6060…`
- **gobbonet** `3.7.0` — agent config, Python  
  `sha256:cd22606b84f514f2…`
- **hermes** `3.7.0` — agent config, skills  
  `sha256:67886c72f70d1382…`
- **iris** `3.7.0` — skills  
  `sha256:1ca3cac778165f80…`
- **openclaw** `3.7.0` — agent config, skills  
  `sha256:bcca32e758be087d…`
- **persona** `3.7.0` — brain pack only  
  `sha256:5f342072cbdd275f…`
