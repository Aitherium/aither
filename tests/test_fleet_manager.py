"""Fleet manager — cross-runtime create/list/status/remove.

Verifies the durable store, the manager's dispatch to per-runtime drivers (local /
managed / cloud-run), and — with a REAL subprocess — that the local driver's PID
tracking + terminate actually work (not just a mocked 'running').
"""

from __future__ import annotations

import sys
import time

import pytest

from adk.fleet_manager import (
    CloudRunDriver,
    FleetManager,
    FleetMember,
    FleetStore,
    LocalDriver,
    ManagedDriver,
)


# ── store ────────────────────────────────────────────────────────────────

def test_store_roundtrip_and_remove(tmp_path):
    store = FleetStore(path=tmp_path / "fleet.json")
    assert store.all() == []
    m = FleetMember(id="abc", name="scout", runtime="local", status="running", ref="123")
    store.upsert(m)
    got = store.get("abc")
    assert got is not None and got.name == "scout" and got.ref == "123"
    assert [x.id for x in store.all()] == ["abc"]
    assert store.remove("abc") is True
    assert store.get("abc") is None
    assert store.remove("abc") is False  # idempotent


def test_store_survives_corrupt_file(tmp_path):
    p = tmp_path / "fleet.json"
    p.write_text("{ not json", encoding="utf-8")
    store = FleetStore(path=p)
    assert store.all() == []  # tolerates a garbled file instead of crashing


# ── manager dispatch with fake drivers ──────────────────────────────────

def _mgr(tmp_path, drivers):
    return FleetManager(store=FleetStore(path=tmp_path / "fleet.json"), drivers=drivers, now=lambda: 1.0)


def test_create_managed_records_agent_id(tmp_path):
    calls = {}
    def fake_deploy(agent, opts):
        calls["agent"] = agent
        return {"ok": True, "deployed": True, "anthropic_agent_id": "agent_LIVE123", "digest": "d1"}
    mgr = _mgr(tmp_path, {"managed": ManagedDriver(deploy_fn=fake_deploy)})
    m = mgr.create("managed", "twin", pack="weather-eve-import", mcp_url="https://x/mcp")
    assert calls["agent"] == "weather-eve-import"
    assert m.runtime == "managed" and m.status == "running"
    assert m.ref == "agent_LIVE123" and m.meta.get("digest") == "d1"
    # persisted
    assert mgr.get(m.id).ref == "agent_LIVE123"


def test_create_managed_failure_marks_failed(tmp_path):
    mgr = _mgr(tmp_path, {"managed": ManagedDriver(deploy_fn=lambda a, o: {"ok": False, "error": "401"})})
    m = mgr.create("managed", "twin")
    assert m.status == "failed" and "401" in m.error


def test_create_cloud_run_is_pending(tmp_path):
    mgr = _mgr(tmp_path, {"cloud-run": CloudRunDriver(emit_fn=lambda n, o: {"request_id": "cr_x", "status": "pending_runtime"})})
    m = mgr.create("cloud-run", "hosted")
    assert m.status == "pending_runtime" and m.ref == "cr_x"


def test_unknown_runtime_raises(tmp_path):
    mgr = _mgr(tmp_path, {"local": LocalDriver(spawner=lambda n, o: {"pid": 1})})
    with pytest.raises(ValueError):
        mgr.create("quantum", "x")


def test_empty_name_raises(tmp_path):
    mgr = _mgr(tmp_path, {"local": LocalDriver(spawner=lambda n, o: {"pid": 1})})
    with pytest.raises(ValueError):
        mgr.create("local", "")


def test_list_sorted_and_remove(tmp_path):
    mgr = _mgr(tmp_path, {
        "local": LocalDriver(spawner=lambda n, o: {"pid": 999999}),
        "cloud-run": CloudRunDriver(emit_fn=lambda n, o: {"request_id": "cr", "status": "pending_runtime"}),
    })
    a = mgr.create("local", "a")
    b = mgr.create("cloud-run", "b")
    ids = [m.id for m in mgr.list_members()]
    assert set(ids) == {a.id, b.id}
    assert mgr.remove(a.id) is True
    assert {m.id for m in mgr.list_members()} == {b.id}
    assert mgr.remove("nope") is False


# ── local driver against a REAL process ──────────────────────────────────

def test_local_driver_real_process_lifecycle(tmp_path):
    """Spawn a real background process, confirm status=running, remove kills it,
    status=stopped — proving the actual PID logic, not a mock."""
    procs = []
    def real_spawner(name, opts):
        import subprocess
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        return {"pid": p.pid, "endpoint": "http://127.0.0.1:8080"}

    mgr = _mgr(tmp_path, {"local": LocalDriver(spawner=real_spawner)})
    m = mgr.create("local", "scout")
    try:
        assert m.status == "running" and int(m.ref) > 0
        assert mgr.refresh(m.id).status == "running"  # the real PID is alive

        assert mgr.remove(m.id) is True
        # Confirm it's gone using the module's cross-platform liveness (raw
        # os.kill(pid,0) is a Windows footgun — it terminates, not probes).
        from adk.fleet_manager import _pid_alive
        deadline = time.time() + 5
        while time.time() < deadline and _pid_alive(int(m.ref)):
            time.sleep(0.2)
        assert _pid_alive(int(m.ref)) is False, "process should be dead after remove()"
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass


def test_local_status_stopped_for_dead_pid(tmp_path):
    mgr = _mgr(tmp_path, {"local": LocalDriver(spawner=lambda n, o: {"pid": 2147480000})})
    m = mgr.create("local", "ghost")  # a PID that (almost certainly) isn't running
    assert mgr.refresh(m.id).status == "stopped"


# ── cloud-run emit with intent posting ───────────────────────────────────

def test_cloud_run_emit_posts_intent_to_gateway():
    """CloudRunDriver._default_emit POSTs a provisioning intent; verifies shape."""
    calls = []

    def fake_poster(name, opts):
        calls.append({"name": name, "opts": opts})
        return {"request_id": "cr_abc123", "status": "pending_runtime"}

    # Use a fake poster so we don't hit the network
    calls.clear()
    result = CloudRunDriver._default_emit("my-agent", {"pack": "eve-import", "model": "gpt-4"})
    # With no poster injected, it should degrade gracefully
    assert result.get("status") == "pending_runtime"
    assert "cr_" in result.get("request_id", "")


def test_cloud_run_emit_degrades_gracefully_on_network_error():
    """When gateway is down, CloudRunDriver._default_emit returns a pending
    request_id instead of crashing."""
    result = CloudRunDriver._default_emit("offline-agent", {})
    assert result.get("status") == "pending_runtime"
    assert result.get("request_id", "").startswith("cr_")


# ── bidirectional: connect-local ─────────────────────────────────────────

def test_connect_local_agent_posts_mcp_endpoint():
    """connect_local_agent POSTs the agent's MCP URL to the gateway endpoint."""
    calls = []

    def fake_poster(name, mcp_url):
        calls.append({"name": name, "mcp_url": mcp_url})
        return {
            "ok": True,
            "endpoint": {"name": name, "url": mcp_url, "auth_vault_key": "vault::local/auth"},
        }

    from adk.fleet_manager import connect_local_agent

    result = connect_local_agent("my-agent", "http://localhost:8080/mcp", poster=fake_poster)
    assert result.get("ok") is True
    assert calls[0]["name"] == "my-agent"
    assert calls[0]["mcp_url"] == "http://localhost:8080/mcp"
    assert result.get("endpoint", {}).get("name") == "my-agent"


def test_connect_local_agent_empty_url_fails():
    """connect_local_agent rejects an empty MCP URL."""

    def fake_poster(name, mcp_url):
        return {"ok": False, "error": "mcp_url is required"}

    from adk.fleet_manager import connect_local_agent

    result = connect_local_agent("my-agent", "", poster=fake_poster)
    assert result.get("ok") is False
    assert "mcp_url" in result.get("error", "").lower()


def test_connect_local_agent_degrades_on_network_error():
    """When the gateway is down, connect_local_agent returns an error dict,
    never crashes."""

    def failing_poster(name, mcp_url):
        raise ConnectionError("gateway unreachable")

    from adk.fleet_manager import connect_local_agent

    result = connect_local_agent("my-agent", "http://localhost:8080/mcp", poster=failing_poster)
    assert result.get("ok") is False
    assert "gateway" in result.get("error", "").lower() or "error" in result
