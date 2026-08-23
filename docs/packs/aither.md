# Aither System Orchestrator

`aither` · version `3.7.3` · 4.4 KB

**[Download aither-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/aither-3.7.3.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/aither-3.7.3.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/aither-3.7.3.tar.gz
tar xzf aither-3.7.3.tar.gz
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

sha256 `0ab0c0fa8fdc4a03c243714f80bc9ebc9c166bb773bd2706628373de1c275b43`  
Built from `v3.7.3` (adk 3.7.3). [All packs](../packs.md)
