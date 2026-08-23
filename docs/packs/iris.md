# Iris Visual Artisan

`iris` · version `3.7.3` · 8.2 KB

**[Download iris-3.7.3.tar.gz](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/iris-3.7.3.tar.gz)** · [checksum](https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/iris-3.7.3.sha256)

```bash
curl -LO https://github.com/Aitherium/aither-adk/releases/download/v3.7.3/iris-3.7.3.tar.gz
tar xzf iris-3.7.3.tar.gz
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

sha256 `0ca7f460a2832036e7ea5cf030b27cc023700849878d4e0f65fd8b623b729ef2`  
Built from `v3.7.3` (adk 3.7.3). [All packs](../packs.md)
