"""Voice client -- STT, TTS, and emotion analysis via AitherVoice service.

Async HTTP client for speech-to-text, text-to-speech, and vocal emotion
detection. Connects to a local AitherVoice service (port 8083) by default,
with convenience functions for quick one-shot use.

Usage:
    from adk.voice import get_voice_client, hear, say, feel

    client = get_voice_client()

    # Transcribe audio
    result = await client.transcribe("recording.wav")
    print(result.text)

    # Synthesize speech
    result = await client.synthesize("Hello world", voice="nova")
    with open("output.wav", "wb") as f:
        f.write(result.audio_data)

    # Detect emotion
    result = await client.analyze_emotion("recording.wav")
    print(f"{result.emotion} ({result.intensity:.1%})")

    # Convenience functions
    text = await hear("recording.wav")
    audio = await say("Hello world")
    mood = await feel("recording.wav")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("adk.voice")

_DEFAULT_SERVICE_URL = "http://localhost:8083"
_DEFAULT_VOICE = "nova"
_TIMEOUT_SECONDS = 30.0

_AVAILABLE_VOICES = [
    {"id": "alloy", "name": "Alloy", "description": "Neutral and balanced"},
    {"id": "echo", "name": "Echo", "description": "Warm and resonant"},
    {"id": "fable", "name": "Fable", "description": "Expressive storyteller"},
    {"id": "nova", "name": "Nova", "description": "Clear and friendly"},
    {"id": "onyx", "name": "Onyx", "description": "Deep and authoritative"},
    {"id": "shimmer", "name": "Shimmer", "description": "Bright and energetic"},
]


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    success: bool
    text: str = ""
    language: str = ""
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""
    success: bool
    audio_path: str = ""
    audio_data: bytes = b""
    error: str = ""


@dataclass
class EmotionResult:
    """Result from vocal emotion analysis."""
    success: bool
    emotion: str = ""
    intensity: float = 0.0
    sensation: str = ""
    error: str = ""


class VoiceClient:
    """Async client for the AitherVoice STT/TTS/emotion service.

    Args:
        service_url: Base URL for the voice service.
            Defaults to ``http://localhost:8083`` or the
            ``AITHER_VOICE_URL`` environment variable.
    """

    def __init__(self, service_url: str = ""):
        self._url = (
            service_url
            or os.getenv("AITHER_VOICE_URL", "")
            or _DEFAULT_SERVICE_URL
        ).rstrip("/")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (WAV, MP3, OGG, etc.).
            language: Optional language hint (e.g. "en", "es", "ja").

        Returns:
            TranscriptionResult with transcribed text.
        """
        path = Path(audio_path)
        if not path.exists():
            return TranscriptionResult(
                success=False, error=f"Audio file not found: {audio_path}"
            )

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                files = {"file": (path.name, path.read_bytes())}
                data = {}
                if language:
                    data["language"] = language

                resp = await client.post(
                    f"{self._url}/api/v1/transcribe",
                    files=files,
                    data=data,
                )

                if resp.status_code == 200:
                    body = resp.json()
                    return TranscriptionResult(
                        success=True,
                        text=body.get("text", ""),
                        language=body.get("language", ""),
                        duration_seconds=body.get("duration_seconds", 0.0),
                    )

                return TranscriptionResult(
                    success=False,
                    error=f"Service returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.ConnectError:
            return TranscriptionResult(
                success=False, error=f"Voice service unavailable at {self._url}"
            )
        except Exception as exc:
            return TranscriptionResult(success=False, error=str(exc))

    async def synthesize(
        self,
        text: str,
        voice: str = _DEFAULT_VOICE,
        output_path: str | None = None,
    ) -> SynthesisResult:
        """Synthesize text to speech audio.

        Args:
            text: Text to convert to speech.
            voice: Voice ID (alloy, echo, fable, nova, onyx, shimmer).
            output_path: Optional path to write the audio file.

        Returns:
            SynthesisResult with audio data and optional file path.
        """
        if not text.strip():
            return SynthesisResult(success=False, error="Text must not be empty")

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{self._url}/api/v1/synthesize",
                    json={"text": text, "voice": voice},
                )

                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")

                    if "audio" in content_type or "octet-stream" in content_type:
                        audio_data = resp.content
                    else:
                        # JSON response with base64 or path
                        body = resp.json()
                        import base64
                        audio_b64 = body.get("audio", "")
                        audio_data = base64.b64decode(audio_b64) if audio_b64 else b""

                    saved_path = ""
                    if output_path and audio_data:
                        out = Path(output_path)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(audio_data)
                        saved_path = str(out)

                    return SynthesisResult(
                        success=True,
                        audio_path=saved_path,
                        audio_data=audio_data,
                    )

                return SynthesisResult(
                    success=False,
                    error=f"Service returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.ConnectError:
            return SynthesisResult(
                success=False, error=f"Voice service unavailable at {self._url}"
            )
        except Exception as exc:
            return SynthesisResult(success=False, error=str(exc))

    async def analyze_emotion(self, audio_path: str) -> EmotionResult:
        """Analyze vocal emotion from an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            EmotionResult with detected emotion and intensity.
        """
        path = Path(audio_path)
        if not path.exists():
            return EmotionResult(
                success=False, error=f"Audio file not found: {audio_path}"
            )

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                files = {"file": (path.name, path.read_bytes())}
                resp = await client.post(
                    f"{self._url}/api/v1/emotion",
                    files=files,
                )

                if resp.status_code == 200:
                    body = resp.json()
                    return EmotionResult(
                        success=True,
                        emotion=body.get("emotion", "neutral"),
                        intensity=body.get("intensity", 0.0),
                        sensation=body.get("sensation", ""),
                    )

                return EmotionResult(
                    success=False,
                    error=f"Service returned {resp.status_code}: {resp.text[:200]}",
                )
        except httpx.ConnectError:
            return EmotionResult(
                success=False, error=f"Voice service unavailable at {self._url}"
            )
        except Exception as exc:
            return EmotionResult(success=False, error=str(exc))

    async def status(self) -> dict:
        """Check voice service health status.

        Returns:
            Dict with service status info, or error details.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "error", "code": resp.status_code}
        except httpx.ConnectError:
            return {"status": "unavailable", "url": self._url}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def available_voices(self) -> list[dict]:
        """Return the list of available voice options.

        Returns:
            List of dicts with id, name, and description for each voice.
        """
        return list(_AVAILABLE_VOICES)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

async def hear(path: str) -> str:
    """Transcribe audio to text. Returns empty string on failure."""
    client = get_voice_client()
    result = await client.transcribe(path)
    return result.text if result.success else ""


async def say(text: str, voice: str = _DEFAULT_VOICE) -> bytes:
    """Synthesize text to audio bytes. Returns empty bytes on failure."""
    client = get_voice_client()
    result = await client.synthesize(text, voice=voice)
    return result.audio_data if result.success else b""


async def feel(path: str) -> str:
    """Detect emotion from audio. Returns emotion string or empty on failure."""
    client = get_voice_client()
    result = await client.analyze_emotion(path)
    return result.emotion if result.success else ""


# ─────────────────────────────────────────────────────────────────────────────
# Module singleton
# ─────────────────────────────────────────────────────────────────────────────

_instance: VoiceClient | None = None


def get_voice_client(service_url: str | None = None) -> VoiceClient:
    """Get or create the module-level VoiceClient singleton.

    Args:
        service_url: Optional service URL override.

    Returns:
        The global VoiceClient instance.
    """
    global _instance
    if _instance is None:
        _instance = VoiceClient(service_url=service_url or "")
    return _instance
