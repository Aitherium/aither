# OpenClaw Research Studio

`openclaw` · version `3.7.3` · 5.2 KB

**[Download openclaw-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/openclaw-3.7.3.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/openclaw-3.7.3.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/openclaw-3.7.3.tar.gz
tar xzf openclaw-3.7.3.tar.gz
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

sha256 `7bfd4ff55b39c5c1e9b2b71d4a3babc417cfd80b38ec25ef4629c7c1c7a87255`  
Built from `v3.7.3` (adk 3.7.3). [All packs](../packs.md)
