"""OpenTelemetry exporter — optional OTLP shipping of ADK traces.

The published :mod:`adk.trace` module owns request-id propagation across
HTTP middleware, Chronicle, Strata, and downstream tools. This module
**adds** a thin layer that, when the ``opentelemetry-sdk`` package is
installed, also publishes those traces over OTLP (gRPC) so they show up
in Grafana Tempo, Jaeger, or any OTel-compatible backend.

Design goals
------------
* Zero hard dep on ``opentelemetry`` — import lazily.
* Re-use the existing :func:`adk.trace.get_trace_id` so the OTel trace
  context and the ADK request-id stay aligned in logs.
* Provide a tiny :class:`OTelTracer` whose :meth:`span` context manager
  records attributes, exceptions, and durations.

Wire-up
-------
::

    from adk.otel import try_build_otel_tracer

    tracer = try_build_otel_tracer(endpoint="http://localhost:4317")
    if tracer:
        with tracer.span("agent.run", agent="atlas"):
            await agent.run(task)

When the optional dep is missing or the call returns ``None``, callers
should fall back to logging or to a no-op span.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Protocol, runtime_checkable

from adk.trace import get_trace_id

logger = logging.getLogger("adk.otel")


# ─────────────────────────────────────────────────────────────────────────────
# Tracer protocol + simple in-process impls
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """Lightweight span object yielded by tracer context managers."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    span_id: str = ""


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracer interface implemented by NoOp/Logging/OTel tracers."""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]: ...


class NoOpTracer:
    """Tracer that does nothing — the safe default."""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        yield Span(name=name, attributes=dict(attrs))


class LoggingTracer:
    """Tracer that logs span open/close at INFO level."""

    def __init__(self, logger_name: str = "adk.otel.log") -> None:
        self._log = logging.getLogger(logger_name)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        rid = get_trace_id()
        if rid:
            attrs = {"aither.request_id": rid, **attrs}
        self._log.info("span.open name=%s attrs=%s", name, attrs)
        s = Span(name=name, attributes=dict(attrs), trace_id=rid)
        try:
            yield s
        except Exception as exc:
            self._log.warning("span.error name=%s exc=%s", name, exc)
            raise
        finally:
            self._log.info("span.close name=%s attrs=%s", name, s.attributes)


# ─────────────────────────────────────────────────────────────────────────────
# Active-tracer registry (module-level, swappable)
# ─────────────────────────────────────────────────────────────────────────────


_tracer: Tracer | None = None


def set_tracer(tracer: Tracer | None) -> None:
    """Replace the process-wide active tracer (``None`` resets)."""
    global _tracer
    _tracer = tracer


def get_tracer() -> Tracer:
    """Return the active tracer, building one from env on first call.

    Resolution order:

    1. Explicit :func:`set_tracer` value, if set.
    2. ``AITHER_OTLP_ENDPOINT`` env → :class:`OTelTracer` (falls back to
       :class:`LoggingTracer` if the SDK is missing).
    3. ``AITHER_TRACE=log`` → :class:`LoggingTracer`.
    4. Default → :class:`NoOpTracer`.
    """
    global _tracer
    if _tracer is not None:
        return _tracer
    endpoint = os.environ.get("AITHER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if endpoint:
        service = os.environ.get("OTEL_SERVICE_NAME", "adk")
        built = try_build_otel_tracer(endpoint=endpoint, service_name=service)
        _tracer = built if built is not None else LoggingTracer()
        return _tracer
    if os.environ.get("AITHER_TRACE", "").lower() == "log":
        _tracer = LoggingTracer()
        return _tracer
    _tracer = NoOpTracer()
    return _tracer


class OTelNotInstalled(ImportError):
    """Raised when ``opentelemetry-sdk`` is not available."""


_AttrScalar = (str, bool, int, float)


def _coerce(value: Any) -> Any:
    """Coerce ``value`` into something OTel ``set_attribute`` accepts."""
    if isinstance(value, _AttrScalar):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first_type = type(value[0])
        if first_type not in _AttrScalar:
            return repr(value)
        if all(type(v) is first_type for v in value):
            return list(value)
        return repr(value)
    return repr(value)


def _clean(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _coerce(v) for k, v in attrs.items()}


def _import_otel(*, require_otlp: bool = False) -> dict[str, Any]:
    """Lazy import — raises :class:`OTelNotInstalled` on failure.

    The OTLP exporter is only required when ``require_otlp=True`` (i.e.
    when the caller actually wants to ship spans over the wire). Local
    in-process usage (e.g. tests with an InMemorySpanExporter) only
    needs the SDK core.
    """
    try:
        from opentelemetry import trace as otel_trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
        from opentelemetry.trace import Status, StatusCode  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise OTelNotInstalled(
            "opentelemetry-sdk is not installed; "
            "pip install opentelemetry-sdk"
        ) from exc
    otlp_exporter = None
    if require_otlp:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
            otlp_exporter = OTLPSpanExporter
        except ImportError as exc:  # pragma: no cover - depends on env
            raise OTelNotInstalled(
                "opentelemetry OTLP exporter is not installed; "
                "pip install opentelemetry-exporter-otlp"
            ) from exc
    return {
        "trace": otel_trace,
        "OTLPSpanExporter": otlp_exporter,
        "Resource": Resource,
        "TracerProvider": TracerProvider,
        "BatchSpanProcessor": BatchSpanProcessor,
        "Status": Status,
        "StatusCode": StatusCode,
    }


class OTelTracer:
    """Thin tracer that opens OTel spans aligned to ADK request IDs."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        service_name: str = "adk",
        provider: Any = None,
        processors: Iterable[Any] | None = None,
    ) -> None:
        mods = _import_otel(require_otlp=bool(endpoint))
        self._mods = mods
        if provider is None:
            resource = mods["Resource"].create({"service.name": service_name})
            provider = mods["TracerProvider"](resource=resource)
            if endpoint:
                exporter = mods["OTLPSpanExporter"](endpoint=endpoint)
                provider.add_span_processor(mods["BatchSpanProcessor"](exporter))
            if processors:
                for proc in processors:
                    provider.add_span_processor(proc)
        self._provider = provider
        self._tracer = mods["trace"].get_tracer(service_name, tracer_provider=provider)

    @property
    def provider(self) -> Any:
        return self._provider

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Open an OTel span and yield a mutable :class:`Span`.

        Mutate ``span.attributes`` inside the block to attach more
        attributes; on exit they are copied onto the OTel span. The
        current ADK trace ID (if any) is attached as
        ``aither.request_id``.
        """
        request_id = get_trace_id()
        clean = _clean(attrs)
        if request_id:
            clean.setdefault("aither.request_id", request_id)
        ctx_mgr = self._tracer.start_as_current_span(name)
        otel_span = ctx_mgr.__enter__()
        for k, v in clean.items():
            otel_span.set_attribute(k, v)
        ctx = otel_span.get_span_context()
        adk_span = Span(
            name=name,
            attributes=dict(clean),
            trace_id=format(ctx.trace_id, "032x") if ctx.trace_id else "",
            span_id=format(ctx.span_id, "016x") if ctx.span_id else "",
        )
        try:
            yield adk_span
        except Exception as exc:
            otel_span.record_exception(exc)
            otel_span.set_status(self._mods["Status"](self._mods["StatusCode"].ERROR, str(exc)))
            raise
        finally:
            for k, v in adk_span.attributes.items():
                otel_span.set_attribute(k, _coerce(v))
            ctx_mgr.__exit__(None, None, None)

    def shutdown(self) -> None:
        """Flush + shutdown the underlying provider."""
        try:
            self._provider.shutdown()
        except Exception:  # pragma: no cover - best effort
            logger.debug("otel.shutdown failed", exc_info=True)


def try_build_otel_tracer(
    *,
    endpoint: str | None = None,
    service_name: str = "adk",
) -> OTelTracer | None:
    """Return an :class:`OTelTracer` if the SDK is importable, else ``None``."""
    try:
        return OTelTracer(endpoint=endpoint, service_name=service_name)
    except OTelNotInstalled:
        return None


def build_from_env() -> OTelTracer | None:
    """Build a tracer using ``AITHER_OTLP_ENDPOINT`` / ``OTEL_SERVICE_NAME``."""
    endpoint = os.environ.get("AITHER_OTLP_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if not endpoint:
        return None
    service = os.environ.get("OTEL_SERVICE_NAME", "adk")
    return try_build_otel_tracer(endpoint=endpoint, service_name=service)


__all__ = [
    "LoggingTracer",
    "NoOpTracer",
    "OTelNotInstalled",
    "OTelTracer",
    "Span",
    "Tracer",
    "build_from_env",
    "get_tracer",
    "set_tracer",
    "try_build_otel_tracer",
]
