# Aither System Orchestrator

`aither` · version `3.8.11` · 4.4 KB

**[Download aither-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/aither-3.8.11.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/aither-3.8.11.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/aither-3.8.11.tar.gz
tar xzf aither-3.8.11.tar.gz
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

sha256 `435e5299facfe53a42c23142416dd2c9f6d75c6f62a028eb2b8bd5197f64a86f`  
Built from `v3.8.11` (adk 3.8.11). [All packs](../packs.md)
