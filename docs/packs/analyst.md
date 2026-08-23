# Analyst Studio

`analyst` · version `3.7.3` · 5.3 KB

**[Download analyst-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/analyst-3.7.3.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/analyst-3.7.3.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/analyst-3.7.3.tar.gz
tar xzf analyst-3.7.3.tar.gz
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

sha256 `dccbe338b83ac31e0280e24f4a8ef3afe6de67b4c439e20614d7aec41fd8bb54`  
Built from `v3.7.3` (adk 3.7.3). [All packs](../packs.md)
