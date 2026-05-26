"""Saga Setup Wizard — Multi-backend LLM provisioning.

Detects hardware, offers three inference backends:
  1. llama.cpp — Best for standalone (bundled in .exe, no external deps)
  2. Ollama — Easiest setup (manages models, nice CLI)
  3. vLLM — Best throughput (GPU batching, paged attention, tensor parallelism)
  4. Cloud API — No GPU needed (Anthropic, OpenAI, DeepSeek)
  5. Existing endpoint — Point at any OpenAI-compatible URL
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("saga.setup")

SAGA_HOME = Path.home() / ".saga"
CONFIG_PATH = SAGA_HOME / "config.json"

MODEL_TIERS = [
    {"name": "gemma4:4b", "vram_gb": 3, "label": "Minimum (any GPU)", "context": "32K"},
    {"name": "nemotron-orchestrator-8b", "vram_gb": 6, "label": "Recommended (ADK default)", "context": "64K"},
    {"name": "mistral-nemo:12b", "vram_gb": 8, "label": "Best local (128K context)", "context": "128K"},
]

# Backend detection functions

def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    if not shutil.which("ollama"):
        return False
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_vllm() -> dict:
    """Check for running vLLM instances on standard ports."""
    import urllib.request
    for port in [8120, 8200, 8201, 8000]:
        try:
            url = f"http://localhost:{port}/v1/models"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m.get("id", "") for m in data.get("data", [])]
                return {"running": True, "port": port, "models": models}
        except Exception:
            continue
    return {"running": False}


def check_llamacpp() -> dict:
    """Check if llama.cpp server is installed or running."""
    aither_home = Path.home() / ".aither"
    config_path = aither_home / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            if cfg.get("setup_backend") == "llamacpp":
                return {
                    "installed": True,
                    "port": cfg.get("llamacpp", {}).get("port", 8200),
                    "url": cfg.get("inference_url", ""),
                }
        except Exception:
            pass
    return {"installed": False}


def detect_hardware() -> dict:
    """Detect GPU and system capabilities."""
    try:
        from adk.llamacpp_setup import detect_accel
        accel = detect_accel()
        return {
            "gpu": accel.kind,
            "gpu_name": accel.name,
            "vram_gb": accel.vram_gb,
            "ram_gb": accel.ram_gb,
            "os": accel.os_family,
            "arch": accel.arch,
        }
    except ImportError:
        return {"gpu": "unknown", "vram_gb": 0, "ram_gb": 0}


def get_available_models() -> list[str]:
    """List models already pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


def recommend_model(vram_gb: float = 0) -> dict:
    """Recommend a model tier based on available VRAM."""
    for tier in reversed(MODEL_TIERS):
        if vram_gb >= tier["vram_gb"]:
            return tier
    return MODEL_TIERS[0]


def pull_model(model_name: str) -> bool:
    """Pull a model via Ollama."""
    print(f"Pulling {model_name}... this may take a few minutes.")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=600,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to pull model: {e}")
        return False


def save_config(config: dict):
    SAGA_HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def is_setup_complete() -> bool:
    config = load_config()
    return config.get("setup_complete", False)


def run_setup():
    """Interactive first-run setup with multi-backend support."""
    print("\n=== Saga Setup Wizard ===\n")

    # Detect hardware
    hw = detect_hardware()
    if hw.get("gpu_name"):
        print(f"Hardware: {hw.get('gpu_name', 'Unknown')} ({hw.get('vram_gb', 0):.1f} GB VRAM, "
              f"{hw.get('ram_gb', 0):.1f} GB RAM)")
    else:
        print(f"Hardware: CPU only ({hw.get('ram_gb', 0):.1f} GB RAM)")

    # Detect available backends
    ollama_ok = check_ollama()
    vllm_info = check_vllm()
    llamacpp_info = check_llamacpp()

    print("\nAvailable inference backends:\n")

    backends = []

    # vLLM (already running)
    if vllm_info.get("running"):
        models = ", ".join(vllm_info.get("models", [])[:3]) or "auto"
        backends.append({
            "id": "vllm",
            "label": f"vLLM (running on port {vllm_info['port']}, models: {models})",
            "note": "Best throughput — GPU batching, paged attention",
        })

    # llama.cpp (installed or installable)
    if llamacpp_info.get("installed"):
        backends.append({
            "id": "llamacpp_existing",
            "label": f"llama.cpp (installed, port {llamacpp_info.get('port', 8200)})",
            "note": "Fast local inference — bundleable into .exe",
        })
    else:
        backends.append({
            "id": "llamacpp_install",
            "label": "llama.cpp (install now — auto-downloads binary + GGUF model)",
            "note": "Recommended for standalone — no Docker, embeds in .exe, CUDA/Vulkan/Metal/CPU",
        })

    # Ollama
    if ollama_ok:
        available = get_available_models()
        backends.append({
            "id": "ollama",
            "label": f"Ollama (running, {len(available)} models available)",
            "note": "Easiest — manages models via 'ollama pull'",
        })
    else:
        backends.append({
            "id": "ollama_install",
            "label": "Ollama (not installed — get from ollama.com/download)",
            "note": "Easy model management, one-click install",
        })

    # Cloud API
    has_cloud_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
    )
    if has_cloud_key:
        providers = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers.append("Anthropic")
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("OpenAI")
        if os.environ.get("DEEPSEEK_API_KEY"):
            providers.append("DeepSeek")
        backends.append({
            "id": "cloud",
            "label": f"Cloud API ({', '.join(providers)} keys found)",
            "note": "No GPU needed — uses cloud models",
        })
    else:
        backends.append({
            "id": "cloud",
            "label": "Cloud API (set ANTHROPIC_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY)",
            "note": "No GPU needed — pay per token",
        })

    # Custom endpoint
    backends.append({
        "id": "custom",
        "label": "Custom OpenAI-compatible endpoint (enter URL)",
        "note": "Point at LM Studio, text-gen-webui, or any /v1/chat/completions server",
    })

    for i, b in enumerate(backends):
        print(f"  [{i+1}] {b['label']}")
        print(f"      {b['note']}")

    choice = input(f"\nSelect backend [1-{len(backends)}]: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(backends)):
        # Default: first available working backend
        choice = "1"
    selected = backends[int(choice) - 1]

    config = {"setup_complete": True, "backend": selected["id"]}

    # --- Handle each backend ---

    if selected["id"] == "vllm":
        config["provider"] = "vllm"
        config["inference_url"] = f"http://localhost:{vllm_info['port']}/v1"
        config["model"] = vllm_info.get("models", [""])[0] if vllm_info.get("models") else ""
        print(f"\nUsing vLLM at port {vllm_info['port']}")

    elif selected["id"] == "llamacpp_existing":
        config["provider"] = "llamacpp"
        config["inference_url"] = llamacpp_info.get("url", f"http://localhost:{llamacpp_info.get('port', 8200)}/v1")
        config["model"] = "aither-orchestrator"
        print(f"\nUsing existing llama.cpp at port {llamacpp_info.get('port', 8200)}")

    elif selected["id"] == "llamacpp_install":
        print("\nInstalling llama.cpp + Nemotron-Orchestrator-8B...")
        try:
            from adk.llamacpp_setup import install, detect_accel
            result = install()
            if result.success:
                config["provider"] = "llamacpp"
                config["inference_url"] = f"http://localhost:{result.port}/v1"
                config["model"] = "aither-orchestrator"
                config["llamacpp_quant"] = result.quant
                print(f"\nllama.cpp installed! Endpoint: http://localhost:{result.port}/v1")
            else:
                print(f"\nInstallation failed: {result.error}")
                print("Falling back to Ollama...")
                config["provider"] = "ollama"
                config["model"] = "gemma4:4b"
        except ImportError:
            print("adk.llamacpp_setup not available. Install aither-adk first.")
            config["provider"] = "ollama"
            config["model"] = "gemma4:4b"

    elif selected["id"] == "ollama":
        config["provider"] = "ollama"
        available = get_available_models()

        print("\nModel tiers:")
        for i, tier in enumerate(MODEL_TIERS):
            marker = "*" if tier["name"] in available else " "
            print(f"  {marker} [{i+1}] {tier['label']}: {tier['name']} "
                  f"({tier['vram_gb']}GB VRAM, {tier['context']} context)")

        vram = hw.get("vram_gb", 8)
        recommended = recommend_model(vram)
        print(f"\nRecommended for your GPU: {recommended['name']}")

        model_choice = input(f"Select model [1-{len(MODEL_TIERS)}] or Enter for recommended: ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(MODEL_TIERS):
            selected_model = MODEL_TIERS[int(model_choice) - 1]
        else:
            selected_model = recommended

        if selected_model["name"] not in available:
            pull_model(selected_model["name"])

        config["model"] = selected_model["name"]
        config["model_tier"] = selected_model["label"]

    elif selected["id"] == "ollama_install":
        print("\nOllama is not installed.")
        print("Download from: https://ollama.com/download")
        print("After installing, run: saga setup")
        config["provider"] = "ollama"
        config["model"] = "gemma4:4b"
        config["setup_complete"] = False

    elif selected["id"] == "cloud":
        config["provider"] = "cloud"
        if os.environ.get("ANTHROPIC_API_KEY"):
            config["cloud_provider"] = "anthropic"
            config["model"] = "claude-sonnet-4-6"
        elif os.environ.get("OPENAI_API_KEY"):
            config["cloud_provider"] = "openai"
            config["model"] = "gpt-4o-mini"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            config["cloud_provider"] = "deepseek"
            config["model"] = "deepseek-chat"
        else:
            print("\nNo API keys found in environment.")
            key_type = input("Provider (anthropic/openai/deepseek): ").strip().lower()
            api_key = input("API key: ").strip()
            if key_type and api_key:
                config["cloud_provider"] = key_type
                config["cloud_api_key"] = api_key
                models = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini", "deepseek": "deepseek-chat"}
                config["model"] = models.get(key_type, "")
            else:
                config["setup_complete"] = False

    elif selected["id"] == "custom":
        url = input("\nEndpoint URL (e.g. http://localhost:1234/v1): ").strip()
        model = input("Model name (or press Enter for auto-detect): ").strip()
        config["provider"] = "custom"
        config["inference_url"] = url
        config["model"] = model

    save_config(config)

    if config.get("setup_complete"):
        print(f"\nSetup complete!")
        print(f"  Backend: {config.get('provider', 'unknown')}")
        print(f"  Model:   {config.get('model', 'auto')}")
        if config.get("inference_url"):
            print(f"  URL:     {config['inference_url']}")
        print("\nRun 'saga' to start.\n")
    else:
        print("\nSetup incomplete. Run 'saga setup' again after installing prerequisites.\n")


if __name__ == "__main__":
    run_setup()
