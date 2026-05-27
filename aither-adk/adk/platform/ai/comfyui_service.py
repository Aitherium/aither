import os
import requests
from aither_adk.ui.console import safe_print

class ComfyUIService:
    """
    Manages the connection to the ComfyUI service (local or Cloudflare gateway).
    """

    @staticmethod
    def get_base_url():
        url = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        return url

    @staticmethod
    def check_connection() -> tuple[bool, str]:
        """Checks connectivity and returns (is_reachable, error_message)."""
        url = ComfyUIService.get_base_url().rstrip("/")
        error_messages = []

        # Bypass proxies for local connections
        proxies = {"http": None, "https": None}

        # Try /system/stats first (standard ComfyUI API)
        try:
            response = requests.get(f"{url}/system/stats", timeout=5, proxies=proxies)
            if response.status_code == 200:
                return True, ""
            else:
                error_messages.append(f"/system/stats: {response.status_code}")
        except requests.RequestException as e:
            error_messages.append(f"/system/stats: {type(e).__name__}")

        # Fallback: Try root URL
        try:
            response = requests.get(url, timeout=5, proxies=proxies)
            # Accept 200 OK or 401/403 (auth required means it's reachable)
            if response.status_code in [200, 401, 403]:
                return True, ""
            else:
                error_messages.append(f"Root: {response.status_code}")
        except requests.RequestException as e:
            error_messages.append(f"Root: {type(e).__name__}")

        return False, " | ".join(error_messages)

    @staticmethod
    def is_reachable() -> bool:
        """Checks if the ComfyUI service is reachable."""
        reachable, _ = ComfyUIService.check_connection()
        return reachable

    @staticmethod
    def get_status() -> str:
        """Returns a status string for the service."""
        url = ComfyUIService.get_base_url()
        if "trycloudflare.com" in url:
            service_type = "Cloudflare Gateway"
        elif "127.0.0.1" in url or "localhost" in url:
            service_type = "Local"
        else:
            service_type = "Remote"

        reachable, error_msg = ComfyUIService.check_connection()

        if reachable:
            return f"[green]Connected ({service_type})[/]"
        else:
            return f"[red]Disconnected ({service_type})[/] - [dim]{url}[/] ([dim]{error_msg}[/])"

    @staticmethod
    def ensure_reachable():
        """Checks connectivity and prints status."""
        status = ComfyUIService.get_status()
        safe_print(f"ComfyUI Status: {status}")
        return "Connected" in status
