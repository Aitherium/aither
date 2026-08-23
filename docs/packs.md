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
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.7.0` | [hermes-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/hermes-3.7.0.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.7.0` | [iris-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/iris-3.7.0.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.7.0` | [openclaw-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/openclaw-3.7.0.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.7.0` | [persona-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/persona-3.7.0.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.7.0` — skills  
  `sha256:de9d4ea0a3813894…`
- **analyst** `3.7.0` — agent config, skills  
  `sha256:4ef754eb970cedd9…`
- **bead-space** `3.7.0` — brain pack only  
  `sha256:8fe19b230e3b6436…`
- **claude-code** `3.7.0` — agent config, skills  
  `sha256:de0c138f4fea8937…`
- **gobbonet** `3.7.0` — agent config, Python  
  `sha256:45ae21c2010d330e…`
- **hermes** `3.7.0` — agent config, skills  
  `sha256:216dbee84809f774…`
- **iris** `3.7.0` — skills  
  `sha256:f70671e414eaf448…`
- **openclaw** `3.7.0` — agent config, skills  
  `sha256:16ed494170744da7…`
- **persona** `3.7.0` — brain pack only  
  `sha256:f0439bf1d636c27a…`
