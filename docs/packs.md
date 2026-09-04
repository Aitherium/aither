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


Built from `v3.8.11` (adk 3.8.11).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.8.11` | [aither-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/aither-3.8.11.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.8.11` | [analyst-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/analyst-3.8.11.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.8.11` | [bead-space-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/bead-space-3.8.11.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.11` | [claude-code-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/claude-code-3.8.11.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.11` | [gobbonet-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/gobbonet-3.8.11.tar.gz) | 45.7 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.11` | [hermes-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/hermes-3.8.11.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.11` | [iris-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/iris-3.8.11.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.11` | [openclaw-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/openclaw-3.8.11.tar.gz) | 5.2 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.11` | [persona-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/persona-3.8.11.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.11` — skills  
  `sha256:9b95e4583c11344a…`
- **analyst** `3.8.11` — agent config, skills  
  `sha256:5df2491be8dab0d3…`
- **bead-space** `3.8.11` — brain pack only  
  `sha256:9fbfb8424fc71976…`
- **claude-code** `3.8.11` — agent config, skills  
  `sha256:5405d28386220ba6…`
- **gobbonet** `3.8.11` — agent config, Python  
  `sha256:fff5f56bbeda4802…`
- **hermes** `3.8.11` — agent config, skills  
  `sha256:05d9602670d69f4f…`
- **iris** `3.8.11` — skills  
  `sha256:1e723ac6b75bb966…`
- **openclaw** `3.8.11` — agent config, skills  
  `sha256:c5d8c501dd00f786…`
- **persona** `3.8.11` — brain pack only  
  `sha256:0ec362b580cddd36…`
