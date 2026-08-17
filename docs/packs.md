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


Built from `v3.4.0` (adk 3.4.0).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.4.0` | [aither-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/aither-3.4.0.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.4.0` | [analyst-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/analyst-3.4.0.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.4.0` | [bead-space-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/bead-space-3.4.0.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.4.0` | [claude-code-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/claude-code-3.4.0.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.4.0` | [gobbonet-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/gobbonet-3.4.0.tar.gz) | 29.1 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.4.0` | [hermes-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/hermes-3.4.0.tar.gz) | 4.8 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.4.0` | [iris-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/iris-3.4.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.4.0` | [openclaw-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/openclaw-3.4.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.4.0` | [persona-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/persona-3.4.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.4.0` — skills  
  `sha256:cc2b4362493400bd…`
- **analyst** `3.4.0` — agent config, skills  
  `sha256:afd35661a916004a…`
- **bead-space** `3.4.0` — brain pack only  
  `sha256:7d3d48e85f3632d4…`
- **claude-code** `3.4.0` — agent config, skills  
  `sha256:c44fa91566b1019c…`
- **gobbonet** `3.4.0` — agent config, Python  
  `sha256:bff9cbdb3da0c040…`
- **hermes** `3.4.0` — agent config, skills  
  `sha256:ec95647158f7455f…`
- **iris** `3.4.0` — skills  
  `sha256:9ec1b4bbb68da0a3…`
- **openclaw** `3.4.0` — agent config, skills  
  `sha256:c720b3e84fd8a008…`
- **persona** `3.4.0` — brain pack only  
  `sha256:f8d7f723768c6f42…`
