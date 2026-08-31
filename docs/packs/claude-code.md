# Claude Code Studio

`claude-code` · version `3.8.10` · 5.0 KB

**[Download claude-code-3.8.10.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/claude-code-3.8.10.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/claude-code-3.8.10.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.10/claude-code-3.8.10.tar.gz
tar xzf claude-code-3.8.10.tar.gz
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

sha256 `eafd1dd2ad6c8b1e6714dd6f6595f763feb5c57178fc5df2bde6054018fcc779`  
Built from `v3.8.10` (adk 3.8.10). [All packs](../packs.md)
