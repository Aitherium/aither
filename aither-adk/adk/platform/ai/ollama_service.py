import os
import sys
import subprocess
import time
import requests
from aither_adk.ui.console import safe_print

# Get Ollama URL from services.yaml (SINGLE SOURCE OF TRUTH)
def _get_ollama_url():
    # Try environment variable first (simplest for Docker)
    ollama_url_env = os.environ.get("OLLAMA_URL")
    if ollama_url_env:
        return ollama_url_env
    
    # Try multiple import paths
    import_attempts = []
    
    # Attempt 1: Path relative to aither_adk package
    try:
        _adk_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        _lib_path = os.path.join(_adk_root, "..", "lib", "core")
        if os.path.exists(_lib_path) and _lib_path not in sys.path:
            sys.path.insert(0, _lib_path)
        from AitherPorts import ollama_url
        return ollama_url()
    except ImportError as e:
        import_attempts.append(f"AitherPorts direct: {e}")
    
    # Attempt 2: lib.core path (common Docker layout)
    try:
        from lib.core.AitherPorts import ollama_url
        return ollama_url()
    except ImportError as e:
        import_attempts.append(f"lib.core.AitherPorts: {e}")
    
    # Attempt 3: Just AitherPorts if PYTHONPATH includes lib/core
    try:
        import AitherPorts
        return AitherPorts.ollama_url()
    except ImportError as e:
        import_attempts.append(f"AitherPorts module: {e}")
    
    # Final fallback: Docker default Ollama URL
    # In Docker environment, Ollama typically runs on host
    return os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")

class OllamaService:
    """
    Manages the local Ollama service.
    """

    BASE_URL = _get_ollama_url()

    @staticmethod
    def is_running() -> bool:
        """Checks if the Ollama service is running and reachable."""
        try:
            response = requests.get(f"{OllamaService.BASE_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def start():
        """Attempts to start the Ollama service."""
        safe_print("[bold yellow]Starting Ollama service...[/]")
        try:
            # Start in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            # Wait for it to become available
            max_retries = 10
            for i in range(max_retries):
                if OllamaService.is_running():
                    safe_print("[bold green]Ollama service started successfully.[/]")
                    return True
                time.sleep(1)
                safe_print(f"[dim]Waiting for Ollama... ({i+1}/{max_retries})[/]")

            safe_print("[bold red]Failed to start Ollama service (timeout).[/]")
            return False
        except FileNotFoundError:
            safe_print("[bold red]Ollama executable not found. Please install Ollama.[/]")
            return False
        except Exception as e:
            safe_print(f"[bold red]Error starting Ollama: {e}[/]")
            return False

    @staticmethod
    def stop():
        """Attempts to stop the Ollama service (Linux/macOS only)."""
        # This is tricky as 'ollama serve' might be a system service or just a process.
        # We'll try pkill for now as a simple solution for dev environments.
        try:
            subprocess.run(["pkill", "ollama"], check=False)
            safe_print("[bold green]Ollama service stopped.[/]")
        except Exception as e:
            safe_print(f"[bold red]Error stopping Ollama: {e}[/]")

    @staticmethod
    def ensure_running():
        """Ensures Ollama is running, starting it if necessary."""
        if OllamaService.is_running():
            # safe_print("[dim]Ollama service is already running.[/]")
            return True

        safe_print("[yellow]Ollama service is not running.[/]")
        return OllamaService.start()

    @staticmethod
    def get_status() -> str:
        """Returns a status string for the service."""
        if OllamaService.is_running():
            return "[green]Running[/]"
        return "[red]Stopped[/]"

    @staticmethod
    def ensure_model(model_name: str):
        """Ensures the specified model is available, pulling it if necessary."""
        if not model_name:
            return
            
        # Strip ollama/ prefix if present
        clean_name = model_name.replace("ollama/", "")
        
        try:
            # Check if model exists
            response = requests.get(f"{OllamaService.BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m.get("name", "").split(":")[0] for m in response.json().get("models", [])]
                full_names = [m.get("name", "") for m in response.json().get("models", [])]
                
                # Check exact match or name without tag
                if clean_name in full_names or clean_name in models:
                    return True
                
                # Also check if clean_name has a tag (e.g. mistral-nemo:latest)
                if ":" not in clean_name:
                     if any(m.startswith(clean_name + ":") for m in full_names):
                         return True

            safe_print(f"[bold yellow]Model '{clean_name}' not found locally. Pulling...[/]")
            
            # Pull model
            # We use subprocess to show progress bar if possible, or just wait
            try:
                process = subprocess.Popen(
                    ["ollama", "pull", clean_name],
                    stdout=None, # Let it print to stdout
                    stderr=None
                )
                process.wait()
                
                if process.returncode == 0:
                    safe_print(f"[bold green]Successfully pulled '{clean_name}'.[/]")
                    return True
                else:
                    safe_print(f"[bold red]Failed to pull '{clean_name}'. Return code: {process.returncode}[/]")
                    return False
            except FileNotFoundError:
                # Fallback to API pull if CLI not found
                safe_print("[yellow]Ollama CLI not found. Using API to pull (this may take a while without progress)...[/]")
                response = requests.post(f"{OllamaService.BASE_URL}/api/pull", json={"name": clean_name, "stream": False}, timeout=3600)
                if response.status_code == 200:
                    safe_print(f"[bold green]Successfully pulled '{clean_name}' via API.[/]")
                    return True
                else:
                    safe_print(f"[bold red]Failed to pull '{clean_name}' via API: {response.text}[/]")
                    return False
                    
        except Exception as e:
            safe_print(f"[bold red]Error checking/pulling model: {e}[/]")
            return False
