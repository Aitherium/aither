import re
import sys
import logging
import warnings
import socket

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\-_]|[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# =============================================================================
# THINKING CONTENT FILTER
# =============================================================================
# Some models (like aither-orchestrator, qwen, deepseek-r1) output internal 
# reasoning in <think>...</think> tags. By default we SHOW these so users
# see that the model is working (not hanging). Can be toggled off if desired.

# Global toggle for showing thinking content
_SHOW_THINKING = True

def set_show_thinking(show: bool):
    """Enable/disable display of <think> content in model responses."""
    global _SHOW_THINKING
    _SHOW_THINKING = show

def get_show_thinking() -> bool:
    """Check if thinking content should be shown."""
    return _SHOW_THINKING

def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.
    
    Args:
        text: Raw model output that may contain thinking tags
        
    Returns:
        Text with thinking blocks removed (unless show_thinking is True)
    """
    if not text:
        return text
    
    if _SHOW_THINKING:
        return text
    
    # Pattern to match <think>...</think> including newlines
    # Handle both lowercase and mixed case
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def extract_thinking(text: str) -> tuple[str, str]:
    """
    Extract thinking content separately from main response.
    
    Args:
        text: Raw model output
        
    Returns:
        Tuple of (thinking_content, main_response)
    """
    if not text:
        return "", ""
    
    # Extract thinking blocks
    pattern = r'<think>(.*?)</think>'
    thinking_matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    thinking = "\n".join(thinking_matches).strip()
    
    # Remove thinking blocks from main content
    main = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    return thinking, main.strip()

def configure_logging():
    # Configure logging to suppress library warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    # Suppress Google SDK specific warnings
    warnings.filterwarnings("ignore", message=".*non-text parts.*")
    warnings.filterwarnings("ignore", message=".*concatenated text.*")
    warnings.filterwarnings("ignore", message=".*returning concatenated.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="google")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
    
    # Suppress resource warnings (unclosed sockets, etc.)
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Suppress specific loggers
    logging.getLogger("google.genai").setLevel(logging.ERROR)
    logging.getLogger("google.adk").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress "Event from an unknown agent" which comes from root or adk
    class EventFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Filter out common noise
            suppress_patterns = [
                "Event from an unknown agent",
                "non-text parts",
                "concatenated text",
                "App name mismatch detected",
            ]
            return not any(p in msg for p in suppress_patterns)

    # Apply to ALL loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).addFilter(EventFilter())
    logging.getLogger().addFilter(EventFilter())

def get_local_ip():
    try:
        # Connect to an external server (doesn't actually send data) to get the interface IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_fqdn():
    try:
        return socket.getfqdn()
    except Exception:
        return "localhost"
