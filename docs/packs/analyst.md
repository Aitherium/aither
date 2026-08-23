# Analyst Studio

`analyst` · version `3.7.4` · 5.3 KB

**[Download analyst-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/analyst-3.7.4.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/analyst-3.7.4.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/analyst-3.7.4.tar.gz
tar xzf analyst-3.7.4.tar.gz
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

sha256 `e59c6e447d354cb856ebd20c8dbf4b543f29d0a1bbd4409b950c60b8549336c4`  
Built from `v3.7.4` (adk 3.7.4). [All packs](../packs.md)
