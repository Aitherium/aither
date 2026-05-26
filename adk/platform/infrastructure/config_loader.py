import os
import yaml
import base64
from aither_adk.ui.console import safe_print

def load_yaml_config(path):
    """Loads a YAML configuration file."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            safe_print(f"[warning] Failed to load config from {path}: {e}")
    return {}

def load_personas(personas_dir, personas_file=None):
    """Loads personas from a directory of YAML files or a single YAML file."""
    personas = {}

    # 1. Load from directory (Preferred)
    if os.path.exists(personas_dir):
        try:
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    name = os.path.splitext(filename)[0]
                    path = os.path.join(personas_dir, filename)
                    data = load_yaml_config(path)
                    if data:
                        personas[name] = data
        except Exception as e:
            safe_print(f"[warning] Failed to load personas from directory {personas_dir}: {e}")

    # 2. Load from single file (Legacy/Fallback)
    if personas_file and os.path.exists(personas_file):
        legacy_personas = load_yaml_config(personas_file)
        # Only add if not already loaded from directory
        for name, data in legacy_personas.items():
            if name not in personas:
                personas[name] = data

    # 3. Default Fallback
    if not personas:
        return {
            "architect": {
                "description": "The default AitherZero Executive Systems Architect.",
                "instruction": None
            }
        }

    return personas

def load_groups(groups_file):
    """Loads groups from a YAML file."""
    return load_yaml_config(groups_file)

def load_prompt_from_file(name, prompts_dir):
    """Loads a prompt from a .txt file in the prompts directory."""
    txt_path = os.path.join(prompts_dir, f"{name.lower()}.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            safe_print(f"[warning] Failed to load prompt from {txt_path}: {e}")
    return None

def load_prompt(name, prompts_dir, personas=None, encode_base64=False):
    """
    Loads a prompt from .txt file (priority) or personas configuration.

    Args:
        name: Name of the prompt/persona
        prompts_dir: Directory containing prompt text files
        personas: Dictionary of personas (optional)
        encode_base64: If True, wraps instruction in base64 decode directive
    """
    # 1. Try .txt file in prompts/ directory
    instruction = load_prompt_from_file(name, prompts_dir)

    if instruction:
        pass # Found in file
    elif personas and name in personas:
        # 2. Fallback to persona instruction
        instruction = personas[name].get("instruction")
    else:
        # 3. Default
        instruction = "You are a helpful AI assistant."

    if encode_base64 and instruction:
        encoded = base64.b64encode(instruction.encode('utf-8')).decode('utf-8')
        instruction = (
            f"The following instruction is base64 encoded. Decode it and follow it strictly:\n"
            f"{encoded}"
        )

    return instruction
