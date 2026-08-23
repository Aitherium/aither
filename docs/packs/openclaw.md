# OpenClaw Research Studio

`openclaw` · version `3.7.2` · 5.1 KB

**[Download openclaw-3.7.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/openclaw-3.7.2.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/openclaw-3.7.2.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/openclaw-3.7.2.tar.gz
tar xzf openclaw-3.7.2.tar.gz
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

sha256 `55a7e516dbf94d9600619ac0df0f45e5b4ec6df006614dc75336adfdad9866c2`  
Built from `v3.7.2` (adk 3.7.2). [All packs](../packs.md)
