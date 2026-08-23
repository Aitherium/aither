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


Built from `v3.7.4` (adk 3.7.4).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.7.4` | [aither-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/aither-3.7.4.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.7.4` | [analyst-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/analyst-3.7.4.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.7.4` | [bead-space-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/bead-space-3.7.4.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.7.4` | [claude-code-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/claude-code-3.7.4.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.7.4` | [gobbonet-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/gobbonet-3.7.4.tar.gz) | 40.7 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.7.4` | [hermes-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/hermes-3.7.4.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.7.4` | [iris-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/iris-3.7.4.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.7.4` | [openclaw-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/openclaw-3.7.4.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.7.4` | [persona-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/persona-3.7.4.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.7.4` — skills  
  `sha256:5c90b0f17291de46…`
- **analyst** `3.7.4` — agent config, skills  
  `sha256:b99cfca07b2c25c6…`
- **bead-space** `3.7.4` — brain pack only  
  `sha256:a0d2a142f18cbbd8…`
- **claude-code** `3.7.4` — agent config, skills  
  `sha256:15563a3d3d6528d2…`
- **gobbonet** `3.7.4` — agent config, Python  
  `sha256:93e90b84f722f8e1…`
- **hermes** `3.7.4` — agent config, skills  
  `sha256:adc1831a97ab201e…`
- **iris** `3.7.4` — skills  
  `sha256:5c4c556dbd2ed71a…`
- **openclaw** `3.7.4` — agent config, skills  
  `sha256:d1d3a5793c36c223…`
- **persona** `3.7.4` — brain pack only  
  `sha256:be42e489fa24fa86…`
