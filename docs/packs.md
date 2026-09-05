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


Built from `v3.8.12` (adk 3.8.12).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.8.12` | [aither-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/aither-3.8.12.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.8.12` | [analyst-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/analyst-3.8.12.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.8.12` | [bead-space-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/bead-space-3.8.12.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.12` | [claude-code-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/claude-code-3.8.12.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.12` | [gobbonet-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/gobbonet-3.8.12.tar.gz) | 45.7 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.12` | [hermes-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/hermes-3.8.12.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.12` | [iris-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/iris-3.8.12.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.12` | [openclaw-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/openclaw-3.8.12.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.12` | [persona-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/persona-3.8.12.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.12` — skills  
  `sha256:49eec80f8b9c20bd…`
- **analyst** `3.8.12` — agent config, skills  
  `sha256:50fffe7b339bd711…`
- **bead-space** `3.8.12` — brain pack only  
  `sha256:f6f1d57cc6064bc7…`
- **claude-code** `3.8.12` — agent config, skills  
  `sha256:2cdd64a8b1667b6a…`
- **gobbonet** `3.8.12` — agent config, Python  
  `sha256:30432d7e3f237ee7…`
- **hermes** `3.8.12` — agent config, skills  
  `sha256:ae064f0057ba0e2c…`
- **iris** `3.8.12` — skills  
  `sha256:a776ac419d8247c8…`
- **openclaw** `3.8.12` — agent config, skills  
  `sha256:28b3fc1f001442cf…`
- **persona** `3.8.12` — brain pack only  
  `sha256:32b2bfea9404b0c7…`
