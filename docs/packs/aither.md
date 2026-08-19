# Aither System Orchestrator

`aither` · version `3.5.0` · 4.4 KB

**[Download aither-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/aither-3.5.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/aither-3.5.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/aither-3.5.0.tar.gz
tar xzf aither-3.5.0.tar.gz
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

sha256 `bb72f1f2c90f54ad23755afccb1fc11f1c8b0a9d065a69acd2d9bf386cd7a926`  
Built from `v3.5.0` (adk 3.5.0). [All packs](../packs.md)
