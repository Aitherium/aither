"""Tests for the simple Tracer abstractions in adk.otel."""

from __future__ import annotations

import logging

import pytest

from adk import otel as otel_mod
from adk.otel import (
    LoggingTracer,
    NoOpTracer,
    Span,
    get_tracer,
    set_tracer,
)


@pytest.fixture(autouse=True)
def _reset_tracer():
    set_tracer(None)
    yield
    set_tracer(None)


class TestSpan:
    def test_default_construction(self):
        s = Span(name="x")
        assert s.name == "x"
        assert s.attributes == {}
        assert s.trace_id == ""
        assert s.span_id == ""


class TestNoOpTracer:
    def test_yields_span_object(self):
        t = NoOpTracer()
        with t.span("op", a=1) as s:
            assert isinstance(s, Span)
            assert s.name == "op"
            assert s.attributes == {"a": 1}

    def test_attribute_mutation_no_throw(self):
        t = NoOpTracer()
        with t.span("op") as s:
            s.attributes["k"] = "v"


class TestLoggingTracer:
    def test_logs_open_and_close(self, caplog):
        t = LoggingTracer("adk.otel.test")
        with caplog.at_level(logging.INFO, logger="adk.otel.test"):
            with t.span("op", agent="atlas") as s:
                s.attributes["result"] = "ok"
        messages = [r.getMessage() for r in caplog.records]
        assert any("span.open" in m and "op" in m for m in messages)
        assert any("span.close" in m and "result" in m for m in messages)

    def test_logs_exception(self, caplog):
        t = LoggingTracer("adk.otel.test")
        with caplog.at_level(logging.WARNING, logger="adk.otel.test"):
            with pytest.raises(RuntimeError):
                with t.span("boom"):
                    raise RuntimeError("kaboom")
        assert any("span.error" in r.getMessage() for r in caplog.records)


class TestGetTracerEnv:
    def test_default_is_noop(self, monkeypatch):
        monkeypatch.delenv("AITHER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("AITHER_TRACE", raising=False)
        assert isinstance(get_tracer(), NoOpTracer)

    def test_aither_trace_log(self, monkeypatch):
        monkeypatch.delenv("AITHER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("AITHER_TRACE", "log")
        assert isinstance(get_tracer(), LoggingTracer)

    def test_otlp_endpoint_falls_back_when_sdk_missing(self, monkeypatch):
        monkeypatch.setenv("AITHER_OTLP_ENDPOINT", "http://localhost:4317")

        def _none(*a, **kw):
            return None

        monkeypatch.setattr(otel_mod, "try_build_otel_tracer", _none)
        assert isinstance(get_tracer(), LoggingTracer)

    def test_explicit_override_wins(self):
        custom = NoOpTracer()
        set_tracer(custom)
        assert get_tracer() is custom
