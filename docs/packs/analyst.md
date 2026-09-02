# Analyst Studio

`analyst` · version `3.8.11` · 5.3 KB

**[Download analyst-3.8.11.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/analyst-3.8.11.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/analyst-3.8.11.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.8.11/analyst-3.8.11.tar.gz
tar xzf analyst-3.8.11.tar.gz
python analyst/install.py
```

Installs to `~/.aither/packs/analyst/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

A data-analysis agent that classifies, regresses, and forecasts over
structured data using zero-shot foundation models (TabFM + TimesFM), and
reasons about the results. Adapts to new labeled data in-context (support
set) instead of gradient training.

## Skills

- `anomaly-detection`
- `structured-inference`

## Contents

```
agent.yaml
brain_pack.yaml
skills/anomaly-detection.md
skills/structured-inference.md
```

---

sha256 `139b554fd5025a5bde8af6c44fcc37fd7936035e1654f7c3cbfbe3852009154d`  
Built from `v3.8.11` (adk 3.8.11). [All packs](../packs.md)
