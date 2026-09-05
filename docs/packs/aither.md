# Aither System Orchestrator

`aither` · version `3.8.12` · 4.4 KB

**[Download aither-3.8.12.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/aither-3.8.12.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/aither-3.8.12.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.12/aither-3.8.12.tar.gz
tar xzf aither-3.8.12.tar.gz
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

sha256 `4580033124804d884fa26f2362f160c2c428e74efe31475ef5dda0bd9cf578c5`  
Built from `v3.8.12` (adk 3.8.12). [All packs](../packs.md)
