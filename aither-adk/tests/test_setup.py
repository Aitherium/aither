"""Tests for adk.setup — agent self-setup module."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adk.setup import (
    AgentSetup,
    GPUInfo,
    SetupReport,
    SystemInfo,
    VLLMInfo,
    _detect_gpu,
    _detect_ram,
    _recommended_models,
    _select_profile,
    auto_setup,
)


# ---------------------------------------------------------------------------
# Profile selection tests
# ---------------------------------------------------------------------------

class TestProfileSelection:
    def test_no_gpu_returns_cpu_only(self):
        assert _select_profile(GPUInfo(), 16.0) == "cpu_only"

    def test_nvidia_low_vram(self):
        assert _select_profile(GPUInfo(vendor="nvidia", vram_mb=8000), 16.0) == "nvidia_low"

    def test_nvidia_mid_vram(self):
        assert _select_profile(GPUInfo(vendor="nvidia", vram_mb=16000), 32.0) == "nvidia_mid"

    def test_nvidia_high_vram(self):
        assert _select_profile(GPUInfo(vendor="nvidia", vram_mb=24000), 64.0) == "nvidia_high"

    def test_nvidia_ultra_vram(self):
        assert _select_profile(GPUInfo(vendor="nvidia", vram_mb=80000), 128.0) == "nvidia_ultra"

    def test_amd_gpu(self):
        assert _select_profile(GPUInfo(vendor="amd", vram_mb=16000), 32.0) == "amd"

    def test_apple_silicon(self):
        assert _select_profile(GPUInfo(vendor="apple", vram_mb=32000), 32.0) == "apple_silicon"

    def test_nvidia_minimal(self):
        assert _select_profile(GPUInfo(vendor="nvidia", vram_mb=4000), 8.0) == "minimal"


class TestRecommendedModels:
    def test_cpu_only_gets_small_models(self):
        models = _recommended_models("cpu_only")
        assert "llama3.2:1b" in models
        assert "nomic-embed-text" in models

    def test_nvidia_high_gets_bigger_models(self):
        models = _recommended_models("nvidia_high")
        assert "llama3.1:8b" in models
        assert "deepseek-r1:14b" in models

    def test_unknown_profile_defaults(self):
        models = _recommended_models("nonexistent")
        assert len(models) > 0


# ---------------------------------------------------------------------------
# GPU detection tests (mocked)
# ---------------------------------------------------------------------------

class TestGPUDetection:
    @pytest.mark.asyncio
    async def test_nvidia_detected(self):
        with patch("adk.setup._run") as mock_run:
            # nvidia-smi query
            mock_run.side_effect = [
                (0, "NVIDIA RTX 4090, 24564, 560.35\n", ""),
                (0, "CUDA Version: 12.4", ""),
            ]
            gpu = await _detect_gpu()
            assert gpu.vendor == "nvidia"
            assert gpu.vram_mb == 24564
            assert gpu.cuda_version == "12.4"
            assert gpu.name == "NVIDIA RTX 4090"

    @pytest.mark.asyncio
    async def test_no_gpu(self):
        with patch("adk.setup._run") as mock_run:
            mock_run.return_value = (-1, "", "Command not found")
            gpu = await _detect_gpu()
            assert gpu.vendor == "none"

    @pytest.mark.asyncio
    async def test_multi_gpu_vram_total(self):
        with patch("adk.setup._run") as mock_run:
            mock_run.side_effect = [
                (0, "RTX 4090, 24564, 560.35\nRTX 4090, 24564, 560.35\n", ""),
                (0, "CUDA Version: 12.4", ""),
            ]
            gpu = await _detect_gpu()
            assert gpu.count == 2
            assert gpu.vram_mb == 49128  # 2x24564


# ---------------------------------------------------------------------------
# RAM detection tests
# ---------------------------------------------------------------------------

class TestRAMDetection:
    @pytest.mark.asyncio
    async def test_ram_detected(self):
        with patch("adk.setup._run") as mock_run:
            mock_run.return_value = (0, "TotalPhysicalMemory=34359738368\n", "")
            ram = await _detect_ram()
            assert ram > 30  # ~32 GB


# ---------------------------------------------------------------------------
# AgentSetup tests
# ---------------------------------------------------------------------------

class TestAgentSetup:
    @pytest.mark.asyncio
    async def test_detect_hardware(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch("adk.setup._detect_gpu") as mock_gpu, \
             patch("adk.setup._detect_ram") as mock_ram, \
             patch("adk.setup._check_ollama") as mock_ollama, \
             patch("adk.setup._check_vllm") as mock_vllm, \
             patch("adk.setup._check_docker") as mock_docker:
            mock_gpu.return_value = GPUInfo(vendor="nvidia", vram_mb=24000, name="RTX 4090")
            mock_ram.return_value = 64.0
            mock_ollama.return_value = (True, True, ["llama3.2:3b"])
            mock_vllm.return_value = VLLMInfo(running=False)
            mock_docker.return_value = True

            info = await setup.detect_hardware()
            assert info.gpu.vendor == "nvidia"
            assert info.ram_gb == 64.0
            assert info.ollama_running is True
            assert info.docker_installed is True
            assert info.profile == "nvidia_high"
            assert info.active_backend == "ollama"

    @pytest.mark.asyncio
    async def test_detect_hardware_vllm_takes_priority(self, tmp_path):
        """When both vLLM and Ollama are running, vLLM should be active backend."""
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch("adk.setup._detect_gpu") as mock_gpu, \
             patch("adk.setup._detect_ram") as mock_ram, \
             patch("adk.setup._check_ollama") as mock_ollama, \
             patch("adk.setup._check_vllm") as mock_vllm, \
             patch("adk.setup._check_docker") as mock_docker:
            mock_gpu.return_value = GPUInfo(vendor="nvidia", vram_mb=24000)
            mock_ram.return_value = 64.0
            mock_ollama.return_value = (True, True, ["llama3.2:3b"])
            mock_vllm.return_value = VLLMInfo(running=True, ports=[8200], models=["llama3.1"])
            mock_docker.return_value = True

            info = await setup.detect_hardware()
            assert info.active_backend == "vllm"

    @pytest.mark.asyncio
    async def test_ensure_ollama_already_running(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(ollama_installed=True, ollama_running=True)
        result = await setup.ensure_ollama()
        assert result is True

    @pytest.mark.asyncio
    async def test_pull_models_skips_existing(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(
            ollama_models=["llama3.2:3b", "nomic-embed-text"],
            profile="nvidia_mid",
        )
        with patch("adk.setup._run") as mock_run:
            # Only pull models not in existing list
            mock_run.return_value = (0, "success", "")
            pulled = await setup.pull_models(["llama3.2:3b", "nomic-embed-text"])
            # Both already exist, no pulls needed
            assert "llama3.2:3b" in pulled
            assert "nomic-embed-text" in pulled
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_pull_models_pulls_missing(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(ollama_models=["llama3.2:3b"], profile="nvidia_mid")
        with patch("adk.setup._run") as mock_run:
            mock_run.return_value = (0, "success", "")
            pulled = await setup.pull_models(["llama3.2:3b", "deepseek-r1:8b"])
            assert "llama3.2:3b" in pulled
            assert "deepseek-r1:8b" in pulled

    @pytest.mark.asyncio
    async def test_ensure_vllm_no_docker(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(docker_installed=False, gpu=GPUInfo(vendor="nvidia"))
        result = await setup.ensure_vllm()
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_vllm_no_nvidia(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(docker_installed=True, gpu=GPUInfo(vendor="amd"))
        result = await setup.ensure_vllm()
        assert result is False

    @pytest.mark.asyncio
    async def test_full_setup(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect, \
             patch.object(setup, "pull_models") as mock_pull:
            mock_detect.return_value = SystemInfo(
                gpu=GPUInfo(vendor="nvidia", vram_mb=24000),
                ram_gb=64.0,
                ollama_running=True,
                ollama_models=["llama3.1:8b"],
                vllm=VLLMInfo(running=False),
                profile="nvidia_high",
            )
            setup._system = mock_detect.return_value
            mock_pull.return_value = ["llama3.1:8b", "nomic-embed-text"]

            report = await setup.full_setup()
            assert report.ready is True
            assert report.backend == "ollama"
            assert report.ollama_ready is True
            assert report.profile == "nvidia_high"

    @pytest.mark.asyncio
    async def test_full_setup_saves_report(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect, \
             patch.object(setup, "pull_models") as mock_pull:
            mock_detect.return_value = SystemInfo(
                ollama_running=True,
                vllm=VLLMInfo(running=False),
                profile="cpu_only",
            )
            setup._system = mock_detect.return_value
            mock_pull.return_value = ["llama3.2:1b"]

            await setup.full_setup()
            report_path = tmp_path / "setup_report.json"
            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert data["ready"] is True

    @pytest.mark.asyncio
    async def test_health_check(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
            mock_client.get.return_value = mock_resp

            results = await setup.health_check()
            assert results["ollama"] is True


# ---------------------------------------------------------------------------
# Auto-setup convenience
# ---------------------------------------------------------------------------

class TestVLLMSafety:
    """Tests that ensure Ollama never starts when vLLM holds the GPU."""

    @pytest.mark.asyncio
    async def test_ensure_ollama_blocked_when_vllm_running(self, tmp_path):
        """CRITICAL: Ollama must NOT start when vLLM is using GPU."""
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(
            ollama_installed=True,
            ollama_running=False,
            vllm=VLLMInfo(running=True, ports=[8200], models=["llama3.1"]),
            gpu=GPUInfo(vendor="nvidia", vram_mb=24000),
        )
        result = await setup.ensure_ollama()
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_ollama_allowed_when_vllm_not_running(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(
            ollama_installed=True,
            ollama_running=True,
            vllm=VLLMInfo(running=False),
        )
        result = await setup.ensure_ollama()
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_ollama_force_overrides_vllm_check(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        setup._system = SystemInfo(
            ollama_installed=True,
            ollama_running=True,
            vllm=VLLMInfo(running=True, ports=[8200]),
        )
        result = await setup.ensure_ollama(force=True)
        assert result is True  # ollama_running=True, force=True

    @pytest.mark.asyncio
    async def test_full_setup_uses_vllm_when_running(self, tmp_path):
        """full_setup should detect vLLM and use it instead of Ollama."""
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect:
            mock_detect.return_value = SystemInfo(
                gpu=GPUInfo(vendor="nvidia", vram_mb=24000),
                ram_gb=64.0,
                ollama_installed=True,
                ollama_running=False,
                vllm=VLLMInfo(running=True, ports=[8200, 8201], models=["meta-llama/Llama-3.1-8B-Instruct"]),
                profile="nvidia_high",
            )
            setup._system = mock_detect.return_value

            report = await setup.full_setup()
            assert report.ready is True
            assert report.backend == "vllm"
            assert report.vllm_ready is True
            assert report.ollama_ready is False  # must NOT have started Ollama
            assert "meta-llama/Llama-3.1-8B-Instruct" in report.models_available

    @pytest.mark.asyncio
    async def test_full_setup_uses_ollama_when_no_vllm(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect, \
             patch.object(setup, "ensure_ollama") as mock_ollama, \
             patch.object(setup, "pull_models") as mock_pull:
            mock_detect.return_value = SystemInfo(
                gpu=GPUInfo(vendor="nvidia", vram_mb=24000),
                ollama_running=False,
                vllm=VLLMInfo(running=False),
                profile="nvidia_high",
            )
            setup._system = mock_detect.return_value
            mock_ollama.return_value = True
            mock_pull.return_value = ["llama3.1:8b"]

            report = await setup.full_setup()
            assert report.ready is True
            assert report.backend == "ollama"
            mock_ollama.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_setup_detects_cloud_api_keys(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect, \
             patch.object(setup, "ensure_ollama") as mock_ollama, \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            mock_detect.return_value = SystemInfo(
                vllm=VLLMInfo(running=False),
                profile="cpu_only",
            )
            setup._system = mock_detect.return_value
            mock_ollama.return_value = False  # no local backend

            report = await setup.full_setup()
            assert report.ready is True
            assert report.backend == "cloud"

    @pytest.mark.asyncio
    async def test_full_setup_not_ready_when_nothing_works(self, tmp_path):
        setup = AgentSetup(data_dir=str(tmp_path))
        with patch.object(setup, "detect_hardware") as mock_detect, \
             patch.object(setup, "ensure_ollama") as mock_ollama:
            mock_detect.return_value = SystemInfo(
                vllm=VLLMInfo(running=False),
                profile="cpu_only",
            )
            setup._system = mock_detect.return_value
            mock_ollama.return_value = False

            report = await setup.full_setup()
            assert report.ready is False
            assert report.backend == ""


class TestAutoSetup:
    @pytest.mark.asyncio
    async def test_auto_setup_returns_report(self):
        with patch("adk.setup.AgentSetup.full_setup") as mock_full:
            mock_full.return_value = SetupReport(ready=True, profile="nvidia_mid", backend="ollama")
            report = await auto_setup()
            assert report.ready is True
