# OpenClaw Research Studio

`openclaw` · version `3.4.1` · 5.1 KB

**[Download openclaw-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/openclaw-3.4.1.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/openclaw-3.4.1.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/openclaw-3.4.1.tar.gz
tar xzf openclaw-3.4.1.tar.gz
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

sha256 `2b4cb4755509272c8b7a8ee5c59f8096ebe624499b84c4524b084d3f88308204`  
Built from `v3.4.1` (adk 3.4.1). [All packs](../packs.md)
