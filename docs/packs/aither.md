# Aither System Orchestrator

`aither` · version `3.8.2` · 4.4 KB

**[Download aither-3.8.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/aither-3.8.2.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/aither-3.8.2.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.2/aither-3.8.2.tar.gz
tar xzf aither-3.8.2.tar.gz
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

sha256 `325aa9e243d92b4c248ea70b9cce8f3a3b24cdb1d7e906723e3a939c780ab86c`  
Built from `v3.8.2` (adk 3.8.2). [All packs](../packs.md)
