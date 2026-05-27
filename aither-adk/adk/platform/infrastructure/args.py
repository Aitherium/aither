import logging
import os
import sys

from aither_adk.ui.console import safe_print

logger = logging.getLogger(__name__)

# Safety level mapping for unified naming across CLI, UI, and Council
SAFETY_LEVEL_MAP = {
    # CLI/legacy names -> AitherSafety canonical names
    "off": "unrestricted",
    "low": "unrestricted",
    "unsafe": "unrestricted",
    "medium": "casual",
    "casual": "casual",
    "high": "professional",
    "professional": "professional",
    "safe": "professional",
}


def setup_extra_args(parser):
    """
    Adds common extra arguments to the argparse parser.
    """
    parser.add_argument("--safety", type=str, default=None,
                        help="Safety level: professional (high), casual (medium), unrestricted (off/low).")
    parser.add_argument("--env", action="append", help="Set environment variables (KEY=VALUE).")
    parser.add_argument("--persistent", action="store_true", help="Run in persistent mode (server).")
    parser.add_argument("--port", type=int, default=8001, help="Port for persistent mode.")


def _sync_safety_with_service(level: str) -> bool:
    """
    Sync safety level with AitherSafety service and AitherCouncil.

    This ensures CLI agents have the same safety settings as the UI.

    Args:
        level: Canonical safety level (professional, casual, unrestricted)

    Returns:
        bool: True if sync was successful
    """
    synced = False

    # 1. Set via direct module (always works, local state)
    try:
        from aither_adk.ai.safety_mode import set_safety_level
        set_safety_level(level)
        synced = True
    except ImportError:
        pass

    # 2. Sync with AitherSafety HTTP service (if running)
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.post(
                "http://localhost:8105/level",
                json={"level": level}
            )
            if resp.status_code == 200:
                synced = True
    except Exception as exc:
        logger.debug(f"AitherSafety service sync failed (service not running): {exc}")

    # 3. Sync with AitherCouncil (if running) for UI parity
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.post(f"http://localhost:8765/tools/safety/level/{level}")
            if resp.status_code == 200:
                synced = True
    except Exception as exc:
        logger.debug(f"AitherCouncil sync failed (council not running): {exc}")

    return synced


def handle_extra_args(args, current_models):
    """
    Handles the extra arguments and returns configuration updates.

    Args:
        args: The parsed arguments.
        current_models: The current list of available models.

    Returns:
        tuple: (updated_models, skip_auth, config_overrides)
            - updated_models: List of models (potentially modified)
            - skip_auth: Boolean indicating if auth should be skipped
            - config_overrides: Dictionary of configuration overrides
    """
    config_overrides = {}
    skip_auth = False
    models = current_models.copy()

    # Handle Env Vars
    if args.env:
        for env_var in args.env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                os.environ[key] = value
                safe_print(f"[dim]Set env: {key}={value}[/]")

    # Handle Safety Level from CLI - now syncs with AitherSafety service
    if args.safety:
        mode = args.safety.lower()
        # Map to canonical name
        canonical_level = SAFETY_LEVEL_MAP.get(mode, "professional")

        # Also keep legacy uppercase for backwards compatibility
        legacy_level = {"professional": "HIGH", "casual": "MEDIUM", "unrestricted": "LOW"}.get(canonical_level, "HIGH")
        config_overrides['safety_level'] = legacy_level

        # Sync with AitherSafety service for CLI/UI parity
        synced = _sync_safety_with_service(canonical_level)
        emoji = {"professional": "", "casual": "", "unrestricted": ""}.get(canonical_level, "")
        sync_status = "[green](synced)[/]" if synced else "[yellow](local only)[/]"

        if canonical_level in ["professional", "casual", "unrestricted"]:
            safe_print(f"[bold cyan]{emoji} Safety: {canonical_level.title()} {sync_status}[/]")
        else:
            safe_print(f"[yellow]Warning: Invalid safety level '{args.safety}'. Using default.[/]")

    # Handle Model Selection from CLI
    if args.model:
        # If it doesn't look like a Gemini model, assume it's local
        if "gemini" not in args.model.lower():
             local_model_name = args.model
             config_overrides['use_local_models'] = True
             config_overrides['local_model_name'] = local_model_name

             safe_print(f"[bold yellow]CLI Override: Using Local Model {local_model_name}[/]")
             # Ensure local model is in the list
             if local_model_name not in models:
                 models.insert(0, local_model_name)
             skip_auth = True
        else:
             config_overrides['use_local_models'] = False

    # Handle implicit model flags (e.g. --mistral-nemo)
    # Check sys.argv for flags that are not handled by argparse
    if not args.model:
        known_flags = ["--model", "--select-model", "--prompt", "-p", "--safety", "--env", "--help", "-h"]
        for arg in sys.argv[1:]:
            if arg.startswith("--") and arg not in known_flags:
                # Ignore flags with values if we can't determine them easily,
                # but for now assume any unknown flag is a model request
                if "=" in arg: continue

                model_name = arg[2:]
                config_overrides['use_local_models'] = True
                config_overrides['local_model_name'] = model_name

                safe_print(f"[bold yellow]CLI Override: Using Local Model {model_name} (from flag)[/]")
                if model_name not in models:
                    models.insert(0, model_name)
                skip_auth = True
                break

    return models, skip_auth, config_overrides
