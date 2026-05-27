import time
import functools
import inspect
from collections import defaultdict
from rich.table import Table
from rich.console import Console

console = Console()

class Profiler:
    _stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "min_time": float('inf'), "max_time": 0.0})
    _enabled = True

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def record(cls, name, duration):
        if not cls._enabled:
            return
        stats = cls._stats[name]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)

    @classmethod
    def get_stats(cls):
        return dict(cls._stats)

    @classmethod
    def reset(cls):
        cls._stats.clear()

    @classmethod
    def print_stats(cls):
        if not cls._stats:
            console.print("[yellow]No profiling data collected.[/]")
            return

        table = Table(title="Performance Profile")
        table.add_column("Function", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Total (s)", justify="right")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")

        # Sort by total time descending
        sorted_stats = sorted(cls._stats.items(), key=lambda x: x[1]["total_time"], reverse=True)

        for name, data in sorted_stats:
            avg_ms = (data["total_time"] / data["count"]) * 1000
            min_ms = data["min_time"] * 1000
            max_ms = data["max_time"] * 1000
            table.add_row(
                name,
                str(data["count"]),
                f"{data['total_time']:.4f}",
                f"{avg_ms:.2f}",
                f"{min_ms:.2f}",
                f"{max_ms:.2f}"
            )

        console.print(table)

def profile(func=None, name=None):
    """
    Decorator to profile a function.
    Usage:
        @profile
        def my_func(): ...

        @profile(name="custom_name")
        def my_func(): ...
    """
    def decorator(f):
        func_name = name or f.__qualname__

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    return await f(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start_time
                    Profiler.record(func_name, duration)
            return async_wrapper
        else:
            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    return f(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start_time
                    Profiler.record(func_name, duration)
            return sync_wrapper

    if func is None:
        return decorator
    return decorator(func)
