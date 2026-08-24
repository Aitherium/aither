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


Built from `v3.8.0` (adk 3.8.0).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.8.0` | [aither-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/aither-3.8.0.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.8.0` | [analyst-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/analyst-3.8.0.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.8.0` | [bead-space-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/bead-space-3.8.0.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.0` | [claude-code-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/claude-code-3.8.0.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.0` | [gobbonet-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/gobbonet-3.8.0.tar.gz) | 45.5 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.0` | [hermes-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/hermes-3.8.0.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.0` | [iris-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/iris-3.8.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.0` | [openclaw-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/openclaw-3.8.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.0` | [persona-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/persona-3.8.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.0` — skills  
  `sha256:169d1bc2a88ad06e…`
- **analyst** `3.8.0` — agent config, skills  
  `sha256:761984da03ab6d0a…`
- **bead-space** `3.8.0` — brain pack only  
  `sha256:ebec5a78b3f78add…`
- **claude-code** `3.8.0` — agent config, skills  
  `sha256:b45dceb506780356…`
- **gobbonet** `3.8.0` — agent config, Python  
  `sha256:23a2eb878b7e7c8b…`
- **hermes** `3.8.0` — agent config, skills  
  `sha256:8e149dfc7cf944c3…`
- **iris** `3.8.0` — skills  
  `sha256:8205ce49da6aa2b2…`
- **openclaw** `3.8.0` — agent config, skills  
  `sha256:e20e6abb48652a56…`
- **persona** `3.8.0` — brain pack only  
  `sha256:6a73d9a90db6e7ec…`
