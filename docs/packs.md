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


Built from `v3.8.10` (adk 3.8.10).

| Pack | Version | Download | Size | What it is |
|---|---|---|---|---|
| **[Aither System Orchestrator](packs/aither.md)** | `3.8.10` | [aither-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/aither-3.8.10.tar.gz) | 4.4 KB | Aither — System Overseer & Orchestrator Brain Pack |
| **[Analyst Studio](packs/analyst.md)** | `3.8.10` | [analyst-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/analyst-3.8.10.tar.gz) | 5.3 KB | Analyst — Data & Structured-ML Agent Brain Pack |
| **[BeadSpace](packs/bead-space.md)** | `3.8.10` | [bead-space-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/bead-space-3.8.10.tar.gz) | 1.5 KB | BeadSpace — an aither-adk agent pack for bead-space |
| **[Claude Code Studio](packs/claude-code.md)** | `3.8.10` | [claude-code-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/claude-code-3.8.10.tar.gz) | 5.0 KB | Claude Code — Software Development Agent Brain Pack |
| **[GobboPack](packs/gobbonet.md)** | `3.8.10` | [gobbonet-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/gobbonet-3.8.10.tar.gz) | 45.5 KB | GobboNet Companion — an agent harness for a local-first chat client |
| **[Hermes Architecture Studio](packs/hermes.md)** | `3.8.10` | [hermes-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/hermes-3.8.10.tar.gz) | 4.9 KB | Hermes — Architecture & Reasoning Agent Brain Pack |
| **[Iris Visual Artisan](packs/iris.md)** | `3.8.10` | [iris-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/iris-3.8.10.tar.gz) | 8.2 KB | Iris — Visual Artisan Brain Pack |
| **[OpenClaw Research Studio](packs/openclaw.md)** | `3.8.10` | [openclaw-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/openclaw-3.8.10.tar.gz) | 5.1 KB | OpenClaw — Web Research Agent Brain Pack |
| **[Persona](packs/persona.md)** | `3.8.10` | [persona-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/persona-3.8.10.tar.gz) | 1.5 KB | Persona — an aither-adk agent pack for persona |

## Contents

- **aither** `3.8.10` — skills  
  `sha256:fe14833782a80716…`
- **analyst** `3.8.10` — agent config, skills  
  `sha256:349fb593fa6a5aa7…`
- **bead-space** `3.8.10` — brain pack only  
  `sha256:967af2b98726f1af…`
- **claude-code** `3.8.10` — agent config, skills  
  `sha256:02d7f0b136c394a9…`
- **gobbonet** `3.8.10` — agent config, Python  
  `sha256:c52d97ba82a84353…`
- **hermes** `3.8.10` — agent config, skills  
  `sha256:884544f969872376…`
- **iris** `3.8.10` — skills  
  `sha256:95b078e3194e490e…`
- **openclaw** `3.8.10` — agent config, skills  
  `sha256:bf87f40cbeb8c9cf…`
- **persona** `3.8.10` — brain pack only  
  `sha256:811b6acf48eb8860…`
