# Analyst Studio

`analyst` · version `3.7.2` · 5.3 KB

**[Download analyst-3.7.2.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/analyst-3.7.2.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/analyst-3.7.2.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.2/analyst-3.7.2.tar.gz
tar xzf analyst-3.7.2.tar.gz
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

sha256 `fb714e8a72e1e8cf22e6e08d257b7b60f8087f6dc1b99f71d69f19c57f758977`  
Built from `v3.7.2` (adk 3.7.2). [All packs](../packs.md)
