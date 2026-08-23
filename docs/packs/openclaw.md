# OpenClaw Research Studio

`openclaw` · version `3.7.4` · 5.1 KB

**[Download openclaw-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/openclaw-3.7.4.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/openclaw-3.7.4.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/openclaw-3.7.4.tar.gz
tar xzf openclaw-3.7.4.tar.gz
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

sha256 `e3f6ea86b6cdc2a1f71ae659ec8a2726087a32179d54de5823cd3957d41ba9ce`  
Built from `v3.7.4` (adk 3.7.4). [All packs](../packs.md)
