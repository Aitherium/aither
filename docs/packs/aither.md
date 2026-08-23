# Aither System Orchestrator

`aither` · version `3.7.4` · 4.4 KB

**[Download aither-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/aither-3.7.4.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/aither-3.7.4.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/aither-3.7.4.tar.gz
tar xzf aither-3.7.4.tar.gz
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

sha256 `2f03ade272a5c348362721962600a5f446ed34a940078ab2e95b39edae7a3dda`  
Built from `v3.7.4` (adk 3.7.4). [All packs](../packs.md)
