# OpenClaw Research Studio

`openclaw` · version `3.7.0` · 5.1 KB

**[Download openclaw-3.7.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/openclaw-3.7.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/openclaw-3.7.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.0/openclaw-3.7.0.tar.gz
tar xzf openclaw-3.7.0.tar.gz
python openclaw/install.py
```

Installs to `~/.aither/packs/openclaw/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

A web research agent that searches the open web, reads primary sources,
cross-checks claims against its knowledge graph, and writes cited reports.
Sign-in-free; runs on the operator's own LLM key.

## Skills

- `source-verification`
- `web-research`

## Contents

```
agent.yaml
brain_pack.yaml
skills/source-verification.md
skills/web-research.md
```

---

sha256 `81b796b3178378a2700b17a6c3bf9cbed5fd567c16d0e2c6afa8b76416950a00`  
Built from `v3.7.0` (adk 3.7.0). [All packs](../packs.md)
