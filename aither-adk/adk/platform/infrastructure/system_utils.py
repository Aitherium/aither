import os
import logging
import subprocess
import time
from aither_adk.infrastructure.profiling import profile

logger = logging.getLogger(__name__)

# Determine Project Root
# .../AitherOS/agents/common/system_utils.py -> .../AitherZero
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

_cache = {}

def cached(ttl=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, frozenset(kwargs.items()))
            now = time.time()
            if key in _cache:
                val, timestamp = _cache[key]
                if now - timestamp < ttl:
                    return val
            val = func(*args, **kwargs)
            _cache[key] = (val, now)
            return val
        return wrapper
    return decorator

@cached(ttl=5)
@profile
def get_system_load():
    try:
        # Try using psutil for cross-platform support
        try:
            import psutil
            # CPU Load (percentage)
            # Use cpu_percent for a more responsive "Load" metric (0-100%)
            # interval=None compares to last call (non-blocking)
            cpu_usage = psutil.cpu_percent(interval=None)
            load_avg = f"{cpu_usage:.1f}%"

            # Memory Usage
            mem = psutil.virtual_memory()
            mem_usage = f"{mem.percent:.0f}%"

            return f"Load: {load_avg} | RAM: {mem_usage}"
        except ImportError:
            pass

        # Fallback to Linux /proc for environments without psutil
        if os.path.exists("/proc/loadavg"):
            with open("/proc/loadavg", "r") as f:
                load_avg = f.read().split()[0]
        else:
            load_avg = "?"

        # Memory Usage
        mem_usage = "?"
        if os.path.exists("/proc/meminfo"):
            mem_total = 0
            mem_available = 0
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_total = int(line.split()[1])
                    elif "MemAvailable" in line:
                        mem_available = int(line.split()[1])
            if mem_total > 0:
                used_percent = ((mem_total - mem_available) / mem_total) * 100
                mem_usage = f"{used_percent:.0f}%"

        return f"Load: {load_avg} | RAM: {mem_usage}"
    except Exception as e:
        # print(f"Error getting system load: {e}")
        return "Sys: ?"

@cached(ttl=10)
@profile
def get_git_status():
    try:
        # Branch
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        # Changes
        changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        count = len(changes.splitlines()) if changes else 0

        status = branch
        if count > 0:
            status += f"* ({count})"
        return status
    except Exception as e:
        # print(f"Error getting git status: {e}")
        return "Git: ?"

def get_detailed_system_specs():
    """
    Returns a dictionary of detailed system specifications (CPU, RAM, GPU).
    """
    specs = {}
    try:
        import psutil
        import platform

        # CPU
        cpu_count = psutil.cpu_count(logical=False) or "?"
        cpu_threads = psutil.cpu_count(logical=True) or "?"

        cpu_model = platform.processor()
        # On Linux, platform.processor() is often just the arch (e.g. x86_64)
        # Try to get better name from /proc/cpuinfo if on Linux
        if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
            except Exception as exc:
                logger.debug(f"CPU info read failed: {exc}")

        specs["CPU"] = f"{cpu_model} ({cpu_count} Cores / {cpu_threads} Threads)"

        # RAM
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        specs["RAM"] = f"{total_gb:.1f} GB Total"

        # GPU (Try nvidia-smi)
        try:
            # Check if nvidia-smi is in path
            import shutil
            if shutil.which("nvidia-smi"):
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpus = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            name = parts[0].strip()
                            mem = parts[1].strip() if len(parts) > 1 else "?"
                            gpus.append(f"{name} ({mem})")

                    if gpus:
                        specs["GPU"] = ", ".join(gpus)
        except Exception as exc:
            logger.debug(f"nvidia-smi GPU query failed: {exc}")

    except ImportError:
        specs["System Info"] = "psutil not installed"
    except Exception as e:
        specs["System Info Error"] = str(e)

    return specs
