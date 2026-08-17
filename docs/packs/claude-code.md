# Claude Code Studio

`claude-code` · version `3.4.0` · 5.0 KB

**[Download claude-code-3.4.0.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/claude-code-3.4.0.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/claude-code-3.4.0.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.4.0/claude-code-3.4.0.tar.gz
tar xzf claude-code-3.4.0.tar.gz
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

sha256 `6bfd99925ed9035f5422438d3baf1e5d1684910b8a8b23e007a3e23200f076fa`  
Built from `v3.4.0` (adk 3.4.0). [All packs](../packs.md)
