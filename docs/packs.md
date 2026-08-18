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


Built from `v3.5.0` (adk 3.5.0).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.5.0` | [aither-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/aither-3.5.0.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.5.0` | [analyst-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/analyst-3.5.0.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.5.0` | [bead-space-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/bead-space-3.5.0.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.5.0` | [claude-code-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/claude-code-3.5.0.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.5.0` | [gobbonet-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/gobbonet-3.5.0.tar.gz) | 29.1 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.5.0` | [hermes-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/hermes-3.5.0.tar.gz) | 4.8 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.5.0` | [iris-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/iris-3.5.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.5.0` | [openclaw-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/openclaw-3.5.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.5.0` | [persona-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/persona-3.5.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.5.0` — skills  
  `sha256:e2e1fe39d7ff8de5…`
- **analyst** `3.5.0` — agent config, skills  
  `sha256:d79987179a718431…`
- **bead-space** `3.5.0` — brain pack only  
  `sha256:93a71b90c0c8e3eb…`
- **claude-code** `3.5.0` — agent config, skills  
  `sha256:3fb4c2ad25699af4…`
- **gobbonet** `3.5.0` — agent config, Python  
  `sha256:c3dd3b9a2d341d67…`
- **hermes** `3.5.0` — agent config, skills  
  `sha256:146a926a0c49c99a…`
- **iris** `3.5.0` — skills  
  `sha256:b2a7cedc177fc16c…`
- **openclaw** `3.5.0` — agent config, skills  
  `sha256:c65abf00733060b0…`
- **persona** `3.5.0` — brain pack only  
  `sha256:0138e69c915170b1…`
