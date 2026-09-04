# Claude Code Studio

`claude-code` · version `3.8.11` · 5.0 KB

**[Download claude-code-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/claude-code-3.8.11.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/claude-code-3.8.11.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/claude-code-3.8.11.tar.gz
tar xzf claude-code-3.8.11.tar.gz
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

sha256 `07cabb6332112f71911101a92d15c471c00c9e6f74e1b4c799e7625b789eb27f`  
Built from `v3.8.11` (adk 3.8.11). [All packs](../packs.md)
