# Claude Code Studio

`claude-code` · version `3.4.1` · 5.0 KB

**[Download claude-code-3.4.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/claude-code-3.4.1.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/claude-code-3.4.1.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.4.1/claude-code-3.4.1.tar.gz
tar xzf claude-code-3.4.1.tar.gz
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

sha256 `4b64c05389e1a0fe2929272bb06b764b5774bc2f71fb0ae707118f76eb441ba0`  
Built from `v3.4.1` (adk 3.4.1). [All packs](../packs.md)
