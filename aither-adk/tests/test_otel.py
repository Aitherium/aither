"""Tests for adk.otel — skipped when opentelemetry-sdk is absent."""

from __future__ import annotations

import pytest

# Skip the whole module unless the SDK is importable.
pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore
    InMemorySpanExporter,
)

from adk import otel as otel_mod
from adk.otel import OTelTracer, _coerce, try_build_otel_tracer
from adk.trace import new_trace


@pytest.fixture
def in_memory_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider=provider)
    yield tracer, exporter
    tracer.shutdown()


class TestCoerce:
    def test_scalars_pass_through(self):
        assert _coerce("a") == "a"
        assert _coerce(1) == 1
        assert _coerce(1.5) == 1.5
        assert _coerce(True) is True

    def test_homogeneous_list_passes(self):
        assert _coerce([1, 2, 3]) == [1, 2, 3]
        assert _coerce(["a", "b"]) == ["a", "b"]

    def test_mixed_list_reprs(self):
        out = _coerce([1, "x"])
        assert isinstance(out, str)
        assert "1" in out and "x" in out

    def test_empty_list(self):
        assert _coerce([]) == []

    def test_complex_object_reprs(self):
        out = _coerce({"a": 1})
        assert isinstance(out, str)
        assert "a" in out


class TestSpan:
    def test_records_attributes(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.span("test.op", agent="atlas", count=3):
            pass
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["agent"] == "atlas"
        assert attrs["count"] == 3

    def test_inline_attribute_mutation(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.span("op") as s:
            s.attributes["result"] = "ok"
        spans = exporter.get_finished_spans()
        assert dict(spans[0].attributes)["result"] == "ok"

    def test_exception_recorded(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with pytest.raises(RuntimeError):
            with tracer.span("boom"):
                raise RuntimeError("kaboom")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        events = spans[0].events
        assert any(e.name == "exception" for e in events)
        assert spans[0].status.status_code.name == "ERROR"

    def test_request_id_attached(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        new_trace("req-xyz")
        with tracer.span("op"):
            pass
        attrs = dict(exporter.get_finished_spans()[0].attributes)
        assert attrs["aither.request_id"] == "req-xyz"

    def test_multiple_spans(self, in_memory_tracer):
        tracer, exporter = in_memory_tracer
        with tracer.span("a"):
            pass
        with tracer.span("b"):
            pass
        assert {s.name for s in exporter.get_finished_spans()} == {"a", "b"}


class TestBuilders:
    def test_try_build_returns_tracer(self):
        t = try_build_otel_tracer()
        assert t is not None
        t.shutdown()

    def test_try_build_returns_none_when_sdk_missing(self, monkeypatch):
        def _raise(*a, **kw):
            raise otel_mod.OTelNotInstalled("forced")

        monkeypatch.setattr(otel_mod, "_import_otel", _raise)
        assert try_build_otel_tracer() is None

    def test_build_from_env_requires_endpoint(self, monkeypatch):
        monkeypatch.delenv("AITHER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert otel_mod.build_from_env() is None
