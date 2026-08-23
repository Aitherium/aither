# Claude Code Studio

`claude-code` · version `3.7.4` · 5.0 KB

**[Download claude-code-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/claude-code-3.7.4.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/claude-code-3.7.4.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/claude-code-3.7.4.tar.gz
tar xzf claude-code-3.7.4.tar.gz
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

sha256 `e7006febb6cc930d84942811d0778092b1b0a9a7ebc80d6cf134ea03654c2bc1`  
Built from `v3.7.4` (adk 3.7.4). [All packs](../packs.md)
