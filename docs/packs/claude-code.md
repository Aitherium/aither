# Claude Code Studio

`claude-code` · version `3.5.0` · 5.0 KB

**[Download claude-code-3.5.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/claude-code-3.5.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/claude-code-3.5.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.5.0/claude-code-3.5.0.tar.gz
tar xzf claude-code-3.5.0.tar.gz
python claude-code/install.py
```

Installs to `~/.aither/packs/claude-code/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

A coding-focused agent for feature development, debugging, testing,
refactoring, and code review. Works with any programming language and
integrates with version control.

## Skills

- `debugging`
- `feature-development`

## Contents

```
agent.yaml
brain_pack.yaml
skills/debugging.md
skills/feature-development.md
```

---

sha256 `cd4a14925174ea508e4580abf974336ba2c2f9010d783c2572973601dabab2ed`  
Built from `v3.5.0` (adk 3.5.0). [All packs](../packs.md)
