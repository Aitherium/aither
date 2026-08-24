# Aither System Orchestrator

`aither` · version `3.8.0` · 4.4 KB

**[Download aither-3.8.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/aither-3.8.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/aither-3.8.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.0/aither-3.8.0.tar.gz
tar xzf aither-3.8.0.tar.gz
python aither/install.py
```

Installs to `~/.aither/packs/aither/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

The default brain pack for the Aither orchestrator agent. Provides core
capabilities for system coordination, synthesis, delegation, and memory-based
decision-making. Bundles GraphRAG memory for persistent knowledge retention.

## Skills

- `coordination`
- `memory-recall`

## Contents

```
brain_pack.yaml
skills/coordination.md
skills/memory-recall.md
```

---

sha256 `33b6577ec98ff5f48ac859057b768a3a3d273d11982b2d198a73929fb9729d7a`  
Built from `v3.8.0` (adk 3.8.0). [All packs](../packs.md)
