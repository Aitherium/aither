"""Privacy-centric, opt-in telemetry for AitherADK agents."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import httpx

from adk import __version__

logger = logging.getLogger("adk.phonehome")


class Phonehome:
    """Opt-in, privacy-centric telemetry client.

    What it sends:
    - Agent name (hashed by default)
    - Backend type (not URL)
    - Model family (not exact model)
    - Tool count
    - Uptime
    - OS type + Python version
    - ADK version

    What it NEVER sends:
    - Prompts or responses
    - API keys or secrets
    - User data or PII
    - Specific model names (only family: "llama", "gpt", "claude")
    """

    def __init__(
        self,
        gateway_url: str = "https://gateway.aitherium.com",
        enabled: bool = False,
        agent_name: str = "",
        data_dir: str | Path | None = None,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.enabled = enabled
        self.agent_name = agent_name
        self._start_time = time.time()
        self._data_dir = Path(data_dir or Path.home() / ".aither")
        self._queue_path = self._data_dir / "phonehome_queue.jsonl"

    def _anonymize_name(self, name: str) -> str:
        """Hash agent name for privacy."""
        return hashlib.sha256(name.encode()).hexdigest()[:12]

    def _model_family(self, model: str) -> str:
        """Extract model family without specific version."""
        model = model.lower()
        if "llama" in model:
            return "llama"
        if "gpt" in model:
            return "gpt"
        if "claude" in model:
            return "claude"
        if "deepseek" in model:
            return "deepseek"
        if "mistral" in model or "mixtral" in model:
            return "mistral"
        if "gemma" in model:
            return "gemma"
        if "phi" in model:
            return "phi"
        if "qwen" in model:
            return "qwen"
        return "other"

    def build_payload(
        self,
        backend_type: str = "",
        model: str = "",
        tool_count: int = 0,
    ) -> dict:
        """Build the telemetry payload (inspectable for transparency)."""
        return {
            "agent_hash": self._anonymize_name(self.agent_name),
            "backend_type": backend_type,
            "model_family": self._model_family(model),
            "tool_count": tool_count,
            "uptime_seconds": int(time.time() - self._start_time),
            "os_type": platform.system(),
            "python_version": platform.python_version(),
            "adk_version": __version__,
            "timestamp": time.time(),
        }

    async def send(
        self,
        backend_type: str = "",
        model: str = "",
        tool_count: int = 0,
    ) -> bool:
        """Send telemetry if enabled. Returns True if sent successfully."""
        if not self.enabled:
            return False

        payload = self.build_payload(backend_type, model, tool_count)

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.gateway_url}/v1/telemetry",
                    json=payload,
                )
                if resp.status_code < 300:
                    return True
        except Exception as e:
            logger.debug(f"Phonehome failed, queuing offline: {e}")

        # Queue for later
        self._queue_offline(payload)
        return False

    def _queue_offline(self, payload: dict):
        """Write to JSONL disk queue for retry later."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._queue_path, "a") as f:
                f.write(json.dumps(payload) + "\n")

            # Cap at 1000 entries
            _cap_jsonl(self._queue_path, 1000)
        except Exception as e:
            logger.debug(f"Failed to queue phonehome data: {e}")

    async def flush_queue(self) -> int:
        """Try to send queued telemetry entries. Returns count of successfully sent."""
        if not self._queue_path.exists():
            return 0

        lines = self._queue_path.read_text().strip().split("\n")
        if not lines or lines == [""]:
            return 0

        sent = 0
        remaining = []

        async with httpx.AsyncClient(timeout=5.0) as client:
            for line in lines:
                try:
                    payload = json.loads(line)
                    resp = await client.post(
                        f"{self.gateway_url}/v1/telemetry",
                        json=payload,
                    )
                    if resp.status_code < 300:
                        sent += 1
                    else:
                        remaining.append(line)
                except Exception:
                    remaining.append(line)

        if remaining:
            self._queue_path.write_text("\n".join(remaining) + "\n")
        else:
            self._queue_path.unlink(missing_ok=True)

        return sent


def _cap_jsonl(path: Path, max_lines: int):
    """Cap a JSONL file at max_lines, keeping the newest."""
    try:
        lines = path.read_text().strip().split("\n")
        if len(lines) > max_lines:
            path.write_text("\n".join(lines[-max_lines:]) + "\n")
    except Exception:
        pass
