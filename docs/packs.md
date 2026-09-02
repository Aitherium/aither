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
| **[BeadSpace](packs/bead-space.md)** | `3.8.11` | [bead-space-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/bead-space-3.8.11.tar.gz) | 1.6 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.11` | [claude-code-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/claude-code-3.8.11.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.11` | [gobbonet-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/gobbonet-3.8.11.tar.gz) | 45.7 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.11` | [hermes-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/hermes-3.8.11.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.11` | [iris-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/iris-3.8.11.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.11` | [openclaw-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/openclaw-3.8.11.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.11` | [persona-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/persona-3.8.11.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.11` — skills  
  `sha256:eb9834a5a169a8c4…`
- **analyst** `3.8.11` — agent config, skills  
  `sha256:ed9cbb06c12cdaf3…`
- **bead-space** `3.8.11` — brain pack only  
  `sha256:7802ab900a138153…`
- **claude-code** `3.8.11` — agent config, skills  
  `sha256:ddb1ebff65026597…`
- **gobbonet** `3.8.11` — agent config, Python  
  `sha256:9dc6bb807526c05f…`
- **hermes** `3.8.11` — agent config, skills  
  `sha256:ec06a36b0f7a785b…`
- **iris** `3.8.11` — skills  
  `sha256:6a0815e4d71f7d74…`
- **openclaw** `3.8.11` — agent config, skills  
  `sha256:eafe6b41750ba65d…`
- **persona** `3.8.11` — brain pack only  
  `sha256:d1ce843cc6fd61ab…`
