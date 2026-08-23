# OpenClaw Research Studio

`openclaw` · version `3.7.3` · 5.1 KB

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

sha256 `fd5c55d2f0fc590ea02418cd6e1f56225650517438bc78f685d5fa3ca438c0ae`  
Built from `v3.7.3` (adk 3.7.3). [All packs](../packs.md)
