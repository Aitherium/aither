"""
AitherOS ADK Path Resolution

Centralizes all writable path resolution for the ADK.
Inside Docker containers, the ADK source tree (/app/aither_adk) is read-only.
This module resolves writable data paths using AITHER_DATA_DIR or /app/data
as the writable root, falling back to the source-relative Saga/ directory
for local (non-Docker) development.

Usage:
    from aither_adk.paths import get_saga_data_dir, get_saga_subdir

    # Returns writable path like /app/data/Saga/memory
    memory_dir = get_saga_subdir("memory")
"""

import os
import logging

log = logging.getLogger(__name__)

# Detect whether we're running inside Docker
_IN_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("AITHER_IN_DOCKER", "").lower() in ("1", "true")

# Source-relative Saga dir (read-only in Docker, writable locally)
_SOURCE_SAGA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saga")


def get_data_root() -> str:
    """
    Returns the writable data root directory.
    
    Priority:
    1. AITHER_DATA_DIR env var (explicit override)
    2. /app/data (Docker default - must be a volume mount)
    3. Source-relative (local development fallback)
    """
    explicit = os.environ.get("AITHER_DATA_DIR")
    if explicit:
        return explicit
    
    if _IN_DOCKER:
        return "/app/data"
    
    # Local dev: use the source-relative parent (aither_adk/)
    return os.path.dirname(os.path.abspath(__file__))


def get_saga_data_dir() -> str:
    """Returns the writable Saga data directory root."""
    return os.path.join(get_data_root(), "Saga")


def get_saga_subdir(*parts: str, create: bool = False) -> str:
    """
    Returns a writable Saga subdirectory path.
    
    Args:
        *parts: Path components under Saga/ (e.g. "memory", "anchors")
        create: If True, create the directory if it doesn't exist.
    
    Returns:
        Absolute path to the writable directory.
    
    Examples:
        get_saga_subdir("memory")                    -> /app/data/Saga/memory
        get_saga_subdir("output", "comics")           -> /app/data/Saga/output/comics
        get_saga_subdir("config", "personas")         -> <read from source, write to data>
    """
    path = os.path.join(get_saga_data_dir(), *parts)
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            log.warning(f"Cannot create directory {path}: {e}")
    return path


def get_saga_config_dir(*parts: str) -> str:
    """
    Returns the Saga config directory (read-only source tree).
    Config files like personas/*.yaml are shipped with the image and should
    be read from the source tree, not the writable data dir.
    
    Falls back to writable data dir if source path doesn't exist
    (e.g. when configs are mounted as volumes).
    """
    source_path = os.path.join(_SOURCE_SAGA_DIR, "config", *parts)
    if os.path.exists(source_path):
        return source_path
    # Fallback: check writable data dir (for volume-mounted configs)
    return os.path.join(get_saga_data_dir(), "config", *parts)
