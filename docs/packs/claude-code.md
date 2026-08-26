# Claude Code Studio

`claude-code` · version `3.8.3` · 5.0 KB

**[Download claude-code-3.8.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.3/claude-code-3.8.3.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.3/claude-code-3.8.3.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.3/claude-code-3.8.3.tar.gz
tar xzf claude-code-3.8.3.tar.gz
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

sha256 `6c1dc5bd76a17a9b2f2e249abbc2fdc2f27f11da78b07ae3baaa12605f37518b`  
Built from `v3.8.3` (adk 3.8.3). [All packs](../packs.md)
