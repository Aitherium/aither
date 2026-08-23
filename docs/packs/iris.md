# Iris Visual Artisan

`iris` · version `3.7.4` · 8.2 KB

**[Download iris-3.7.4.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/iris-3.7.4.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/iris-3.7.4.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.4/iris-3.7.4.tar.gz
tar xzf iris-3.7.4.tar.gz
python iris/install.py
```

Installs to `~/.aither/packs/iris/`, which adk discovers with no
configuration. The installer verifies the pack is discoverable rather than
assuming it. adk itself:

```bash
pip install aither-adk
```

## About

The image-generation counterpart to the aither orchestrator pack. Iris stands
up her OWN image-gen backend (via the imagegen_* bootstrap toolpack), then
generates through it — ComfyUI for control, SANA for speed. She knows the one
law that governs character work: txt2img + IPAdapter CANNOT pin a character.

## Skills

- `character-consistency`
- `deploy-image-gen`
- `image-generation`

## Contents

```
brain_pack.yaml
skills/character-consistency.md
skills/deploy-image-gen.md
skills/image-generation.md
```

---

sha256 `c5d3fff9012e17a05d89bf06f4eafe3b05f3116c4d803678b29383f0df2bbe10`  
Built from `v3.7.4` (adk 3.7.4). [All packs](../packs.md)
