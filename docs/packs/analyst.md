# Analyst Studio

`analyst` · version `3.7.1` · 5.3 KB

**[Download analyst-3.7.1.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.1/analyst-3.7.1.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.1/analyst-3.7.1.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.1/analyst-3.7.1.tar.gz
tar xzf analyst-3.7.1.tar.gz
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

sha256 `7f72966c0c6171fb3f8a2895b42028657575044b203a8b502380dda212f18f47`  
Built from `v3.7.1` (adk 3.7.1). [All packs](../packs.md)
