# Aither System Orchestrator

`aither` · version `3.4.1` · 4.4 KB

**[Download aither-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/aither-3.4.1.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/aither-3.4.1.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/aither-3.4.1.tar.gz
tar xzf aither-3.4.1.tar.gz
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

sha256 `1a1af9cb2ebcb452d3a4b3cfddc6f673e92eef15bbffd074471bb3e42bdcf755`  
Built from `v3.4.1` (adk 3.4.1). [All packs](../packs.md)
