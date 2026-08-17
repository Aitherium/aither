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


Built from `v3.4.1` (adk 3.4.1).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.4.1` | [aither-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/aither-3.4.1.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.4.1` | [analyst-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/analyst-3.4.1.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.4.1` | [bead-space-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/bead-space-3.4.1.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.4.1` | [claude-code-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/claude-code-3.4.1.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.4.1` | [gobbonet-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/gobbonet-3.4.1.tar.gz) | 29.1 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.4.1` | [hermes-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/hermes-3.4.1.tar.gz) | 4.8 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.4.1` | [iris-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/iris-3.4.1.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.4.1` | [openclaw-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/openclaw-3.4.1.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.4.1` | [persona-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/persona-3.4.1.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.4.1` — skills  
  `sha256:1a1af9cb2ebcb452…`
- **analyst** `3.4.1` — agent config, skills  
  `sha256:677e90721263e94a…`
- **bead-space** `3.4.1` — brain pack only  
  `sha256:4566a09d0e716369…`
- **claude-code** `3.4.1` — agent config, skills  
  `sha256:4b64c05389e1a0fe…`
- **gobbonet** `3.4.1` — agent config, Python  
  `sha256:97b9511e6bfc96f4…`
- **hermes** `3.4.1` — agent config, skills  
  `sha256:40f4e04def31d086…`
- **iris** `3.4.1` — skills  
  `sha256:512ac3c2f6259dfd…`
- **openclaw** `3.4.1` — agent config, skills  
  `sha256:2b4cb4755509272c…`
- **persona** `3.4.1` — brain pack only  
  `sha256:45979375c6b6b683…`
