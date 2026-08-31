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

sha256 `02d7f0b136c394a986a1b5a77b80b8226a31b7a8c6bb1f7d516cd231a311a585`  
Built from `v3.8.10` (adk 3.8.10). [All packs](../packs.md)
