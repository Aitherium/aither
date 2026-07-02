"""AitherSwarm — self-contained multi-agent context-routing swarm.

Dozens of aither-adk agents (local processes OR remote runner jobs) work a
SHARED goal in bite-size chunks, each dumping its partial context into a shared
"context superhighway" (the tenant's Qdrant dataplane, via
:class:`~adk.graph_memory.GraphMemory`). A reasoning **synthesizer** then
assembles every agent's findings into ONE answer — so the local orchestrator is
never overloaded holding all N chunks at once.

Three roles, one shared dataplane (``AITHER_FLEET_QDRANT_URL``):

  plan       decompose a goal into N bite-size tasks → write the TASK LEDGER
  worker     claim an unclaimed task (deconflict) → LLM → dump finding → sync
  synthesize pull every finding for the goal → reasoning model → final answer

Deconfliction is a best-effort CLAIM on the shared dataplane: a worker records
``{task → agent}`` in a claims collection and re-reads to confirm it won the
task before working it (Qdrant is last-write-wins, so a tiny jitter + re-read
resolves a single winner; a rigorously atomic claim would need a lock service —
see :func:`_claim_task`). Every finding is a 768-d GraphMemory node, so it rides
the same node+edge sync + ``graph.synced`` notification the rest of the SDK uses.

CLI (what a runner job or a local shell invokes)::

    python -m adk.swarm plan       --goal "..." --tasks 6
    python -m adk.swarm worker     --goal "..." --agent worker-3
    python -m adk.swarm synthesize --goal "..." --effort 8

All three key off the SAME ``--goal`` (its sha1 is the goal_id) + the same
``AITHER_FLEET_QDRANT_URL`` + ``AITHER_TENANT_ID``, so they converge with zero
shared state beyond the dataplane.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
import uuid

from adk.graph_memory import GraphMemory

logger = logging.getLogger("adk.swarm")

_TASKS_COLLECTION = "swarm_tasks"
_CLAIMS_COLLECTION = "swarm_claims"
# A claim older than this (seconds) is considered abandoned (a crashed worker) and
# may be re-claimed. Keeps a dead runner from parking a task forever.
_CLAIM_TTL_SECS = 600


def goal_id_of(goal: str) -> str:
    """Deterministic id for a goal — the join key across plan/worker/synthesize."""
    return hashlib.sha1(goal.strip().encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Shared-dataplane primitives (direct Qdrant — fast, no full graph pull)
# ─────────────────────────────────────────────────────────────────────────────

def _qdrant() -> tuple[str, dict]:
    """(base_url, headers) for the shared Qdrant, or ('', {}) when unset."""
    url = os.environ.get("AITHER_FLEET_QDRANT_URL", "").strip().rstrip("/")
    key = os.environ.get("AITHER_FLEET_QDRANT_API_KEY", "").strip()
    return url, ({"api-key": key} if key else {})


def _tenant() -> str:
    return os.environ.get("AITHER_TENANT_ID", "").strip()


def _verify():
    """SDK-canonical TLS verify (trusts the AitherNet CA bundle when installed)."""
    try:
        from adk._tls import tls_verify
        return tls_verify()
    except Exception:  # noqa: BLE001
        return True


async def _ensure_collection(client, url: str, hdr: dict, name: str) -> None:
    """Idempotently create a scroll-only collection (size-1 Dot dummy vector —
    tasks/claims carry no semantic vector, and Dot tolerates the zero vector
    that Cosine would reject)."""
    r = await client.get(f"{url}/collections/{name}", headers=hdr)
    if r.status_code != 200:
        await client.put(
            f"{url}/collections/{name}",
            json={"vectors": {"size": 1, "distance": "Dot"}}, headers=hdr,
        )


def _point_id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts)))


# ─────────────────────────────────────────────────────────────────────────────
# AitherFlux context superhighway (the shared context plane for synthesis)
# ─────────────────────────────────────────────────────────────────────────────
# When AITHER_FLUX_URL is set, findings are ALSO dumped into the live AitherFlux
# context plane (`/context/chunks/store`, keyed by session=goal_id) so the
# synthesizer assembles from the fleet's purpose-built, deduped context store —
# owner's chosen primary path. GraphMemory/Qdrant remains the DURABLE dataplane;
# Flux is the ephemeral assembly plane. No Flux URL → pure self-contained Qdrant.

def _flux_url() -> str:
    return os.environ.get("AITHER_FLUX_URL", "").strip().rstrip("/")


async def flux_store_finding(goal_id: str, agent_id: str, task_index: int,
                             subtask: str, content: str) -> str | None:
    """Dump a finding into the AitherFlux context plane (best-effort)."""
    url = _flux_url()
    if not url:
        return None
    import httpx
    body = {
        "content": f"[subtask {task_index}] {subtask}\n{content}",
        "source": f"swarm:{agent_id}:task-{task_index}",
        "session_id": goal_id, "relevance": 0.9, "priority": 5,
        "group_id": goal_id,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, verify=_verify()) as c:
            r = await c.post(f"{url}/context/chunks/store", json=body)
            if r.status_code == 200:
                return r.json().get("chunk_id")
    except Exception as exc:  # noqa: BLE001 — Qdrant remains the durable path
        logger.debug("flux store failed (non-fatal): %s", exc)
    return None


async def flux_gather_findings(goal_id: str) -> list[dict]:
    """Assemble every finding for a goal from the AitherFlux context plane."""
    url = _flux_url()
    if not url:
        return []
    import httpx
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=20.0, verify=_verify()) as c:
            r = await c.get(f"{url}/context/sessions/{goal_id}/chunks")
            if r.status_code != 200:
                return []
            refs = r.json().get("chunks", []) or []
            for ref in refs:
                cid = ref.get("chunk_id")
                if not cid:
                    continue
                cr = await c.get(f"{url}/context/chunks/{cid}")
                if cr.status_code == 200:
                    d = cr.json()
                    out.append({"content": d.get("content", ""),
                                "source": d.get("source", "")})
    except Exception as exc:  # noqa: BLE001
        logger.debug("flux gather failed (non-fatal): %s", exc)
    return out


async def write_task_ledger(goal: str, tasks: list[str]) -> int:
    """Write the decomposed tasks as the shared TASK LEDGER (best-effort)."""
    url, hdr = _qdrant()
    if not url:
        logger.warning("no AITHER_FLEET_QDRANT_URL — task ledger is local-only")
        return 0
    gid, tid = goal_id_of(goal), _tenant()
    points = []
    for i, prompt in enumerate(tasks):
        points.append({
            "id": _point_id(tid, gid, "task", str(i)), "vector": [0.0],
            "payload": {
                "tenant_id": tid, "goal_id": gid, "index": i,
                "goal": goal, "prompt": prompt, "status": "open",
            },
        })
    import httpx
    async with httpx.AsyncClient(timeout=20.0, verify=_verify()) as c:
        await _ensure_collection(c, url, hdr, _TASKS_COLLECTION)
        r = await c.put(
            f"{url}/collections/{_TASKS_COLLECTION}/points?wait=true",
            json={"points": points}, headers=hdr,
        )
        return len(points) if r.status_code in (200, 201) else 0


async def read_task_ledger(goal: str) -> list[dict]:
    """Scroll every task for a goal from the shared ledger (ordered by index)."""
    url, hdr = _qdrant()
    if not url:
        return []
    gid, tid = goal_id_of(goal), _tenant()
    import httpx
    async with httpx.AsyncClient(timeout=15.0, verify=_verify()) as c:
        r = await c.post(
            f"{url}/collections/{_TASKS_COLLECTION}/points/scroll",
            json={"limit": 1000, "with_payload": True, "filter": {"must": [
                {"key": "tenant_id", "match": {"value": tid}},
                {"key": "goal_id", "match": {"value": gid}},
            ]}}, headers=hdr,
        )
        if r.status_code != 200:
            return []
        pts = r.json().get("result", {}).get("points", []) or []
    tasks = [p.get("payload", {}) for p in pts]
    return sorted(tasks, key=lambda t: t.get("index", 0))


async def _claim_task(goal_id: str, index: int, agent_id: str) -> bool:
    """Best-effort deconfliction: claim task ``index`` for ``agent_id``.

    Returns True if THIS agent holds the claim. Qdrant has no atomic
    compare-and-set, so this is: read → (open or stale?) → write my claim →
    re-read → I win iff the persisted owner is me. Concurrent claimers race on
    the write, but the re-read collapses to ONE winner; a small jitter shrinks
    the race window. Good enough for swarm deconfliction (worst case: two agents
    briefly double-work one task — the synthesizer dedups). A hard guarantee
    would need a real lock service.
    """
    url, hdr = _qdrant()
    if not url:
        return True  # no dataplane → single-process, no contention
    tid = _tenant()
    pid = _point_id(tid, goal_id, "claim", str(index))
    now = time.time()
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as c:
        await _ensure_collection(c, url, hdr, _CLAIMS_COLLECTION)
        # 1. read current claim
        r = await c.post(
            f"{url}/collections/{_CLAIMS_COLLECTION}/points",
            json={"ids": [pid], "with_payload": True}, headers=hdr,
        )
        existing = None
        if r.status_code == 200:
            got = r.json().get("result", []) or []
            if got:
                existing = got[0].get("payload", {})
        if existing:
            owner = existing.get("agent_id")
            ts = float(existing.get("ts", 0) or 0)
            if owner and owner != agent_id and (now - ts) < _CLAIM_TTL_SECS:
                return False  # actively held by someone else
        # 2. write my claim
        await c.put(
            f"{url}/collections/{_CLAIMS_COLLECTION}/points?wait=true",
            json={"points": [{
                "id": pid, "vector": [0.0],
                "payload": {"tenant_id": tid, "goal_id": goal_id,
                            "index": index, "agent_id": agent_id, "ts": now},
            }]}, headers=hdr,
        )
        # 3. re-read to confirm I won the race
        await asyncio.sleep(0.15)
        r = await c.post(
            f"{url}/collections/{_CLAIMS_COLLECTION}/points",
            json={"ids": [pid], "with_payload": True}, headers=hdr,
        )
        if r.status_code == 200:
            got = r.json().get("result", []) or []
            if got and got[0].get("payload", {}).get("agent_id") == agent_id:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper (offload to DeepSeek / Anthropic / gateway per env)
# ─────────────────────────────────────────────────────────────────────────────

async def _llm(system: str, user: str, effort: int, backend: str = "") -> str:
    """One-shot chat through the SDK router. ``backend`` (deepseek/anthropic/
    gateway/…) forces a provider; empty = auto-detect (env keys). Keeps the
    swarm offloading to DeepSeek/Anthropic 'as much as possible' via env."""
    from adk.llm import LLMRouter, Message
    try:
        from adk.config import Config
        cfg = Config.from_env()
    except Exception:  # noqa: BLE001 — env-only is fine in CI
        cfg = None
    # Explicit gateway target (MicroScheduler / mcp.aitherium.com) takes priority
    # so every LLM call routes THROUGH the scheduler (never bypass it) and works
    # in CI without a host API key — LLMRouter's gateway auto-detect otherwise
    # needs a truthy key. An explicit --backend still wins over this.
    gw = os.environ.get("AITHER_GATEWAY_URL", "").strip().rstrip("/")
    if not backend and gw:
        base = gw if gw.endswith("/v1") else gw + "/v1"
        router = LLMRouter(
            provider="openai", base_url=base,
            api_key=os.environ.get("AITHER_API_KEY", "") or "internal",
            model=os.environ.get("AITHER_SWARM_MODEL", "aither-orchestrator"),
            config=cfg,
        )
    else:
        router = LLMRouter(provider=(backend or None), config=cfg)
    resp = await router.chat(
        [Message(role="system", content=system),
         Message(role="user", content=user)],
        effort=effort,
    )
    return (resp.content or "").strip()


def _gm(agent_id: str) -> GraphMemory:
    """A GraphMemory bound to this agent — auto-syncs findings (768-d) to the
    shared Qdrant dataplane, and pulls the whole tenant graph on rehydrate."""
    return GraphMemory(
        agent_name=agent_id,
        tenant_id=_tenant() or None,
        workspace_id=os.environ.get("AITHER_WORKSPACE_ID") or None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Roles
# ─────────────────────────────────────────────────────────────────────────────

_PLAN_SYS = (
    "You are a swarm planner. Decompose the user's goal into independent, "
    "bite-size subtasks that different agents can research IN PARALLEL with no "
    "dependencies between them. Each subtask must be self-contained and "
    "answerable on its own. Respond ONLY with a JSON array of subtask strings, "
    "no prose, no markdown fences."
)


async def plan(goal: str, n_tasks: int = 6, effort: int = 6, backend: str = "") -> list[str]:
    """Decompose ``goal`` into ~``n_tasks`` subtasks and write the task ledger."""
    out = await _llm(
        _PLAN_SYS,
        f"Goal: {goal}\n\nProduce about {n_tasks} parallel subtasks as a JSON array.",
        effort=effort, backend=backend,
    )
    tasks = _parse_task_list(out, goal, n_tasks)
    written = await write_task_ledger(goal, tasks)
    logger.info("planned %d tasks, wrote %d to ledger (goal_id=%s)",
                len(tasks), written, goal_id_of(goal))
    return tasks


def _parse_task_list(text: str, goal: str, n_tasks: int) -> list[str]:
    """Extract a JSON array of task strings, tolerating fences/prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s
        s = s.lstrip("json").strip("`").strip()
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(s[start:end + 1])
            tasks = [str(x).strip() for x in arr if str(x).strip()]
            if tasks:
                return tasks[:max(n_tasks * 2, n_tasks)]
        except Exception:  # noqa: BLE001
            pass
    # Fallback: one line per task
    lines = [ln.strip("-*# ").strip() for ln in text.splitlines() if ln.strip()]
    return lines[:n_tasks] or [f"Research this aspect of: {goal}"]


_WORKER_SYS = (
    "You are one agent in a swarm working a shared goal. You are given ONE "
    "bite-size subtask. Research and answer ONLY that subtask, concisely and "
    "concretely. Your answer becomes shared context for a synthesizer agent, so "
    "state findings as clear facts. Do not restate the whole goal."
)


async def worker(goal: str, agent_id: str, effort: int = 5, backend: str = "",
                 max_tasks: int = 3) -> list[dict]:
    """Claim unclaimed tasks and work them, dumping each finding to the dataplane.

    Loops up to ``max_tasks``: read ledger → claim the next open, unclaimed task
    → LLM → write a 768-d finding node (auto-syncs) → repeat until the ledger is
    dry. Returns the findings this worker produced.
    """
    gid = goal_id_of(goal)
    gm = _gm(agent_id)
    findings: list[dict] = []
    done_indices: set[int] = set()
    for _ in range(max_tasks):
        tasks = await read_task_ledger(goal)
        if not tasks:
            logger.warning("empty ledger for goal_id=%s — run `plan` first", gid)
            break
        claimed = None
        for t in tasks:
            idx = int(t.get("index", 0))
            if idx in done_indices:
                continue
            if await _claim_task(gid, idx, agent_id):
                claimed = t
                break
        if claimed is None:
            logger.info("%s: no unclaimed tasks left — stopping", agent_id)
            break
        idx = int(claimed.get("index", 0))
        done_indices.add(idx)
        prompt = claimed.get("prompt", "")
        logger.info("%s claimed task %d: %s", agent_id, idx, prompt[:80])
        answer = await _llm(_WORKER_SYS, f"Goal: {goal}\n\nSubtask: {prompt}",
                            effort=effort, backend=backend)
        node = await gm.add_node(
            label=f"finding:{gid}:{idx}",
            node_type="finding",
            content=answer,
            tags=[gid, agent_id, f"task-{idx}"],
            importance=0.8,
            metadata={"goal_id": gid, "task_index": idx, "agent_id": agent_id,
                      "subtask": prompt},
        )
        findings.append({"index": idx, "node_id": node.id, "subtask": prompt,
                         "answer": answer})
        # Also dump into the AitherFlux context superhighway (assembly plane) if
        # configured — no-op otherwise. Durable copy already went to GraphMemory.
        await flux_store_finding(gid, agent_id, idx, prompt, answer)
    # Push everything this worker produced to the shared dataplane + notify swarm.
    result = await gm.fleet_push_all_nodes()
    await gm.drain_sync()
    logger.info("%s produced %d findings, sync=%s", agent_id, len(findings), result)
    return findings


_SYNTH_SYS = (
    "You are the swarm SYNTHESIZER. Multiple agents each researched one subtask "
    "of a shared goal; their findings are provided as shared context. Assemble "
    "them into ONE coherent, complete answer to the original goal. Resolve "
    "overlaps, note contradictions explicitly, and do not invent facts beyond "
    "the provided findings. Produce the final deliverable only."
)


async def synthesize(goal: str, effort: int = 8, backend: str = "") -> dict:
    """Assemble every agent's findings for the goal into the final answer.

    Pulls the whole tenant graph from the shared dataplane (so it sees findings
    written by OTHER agents/runners), gathers this goal's finding nodes, and runs
    a high-effort reasoning model over them. Returns
    ``{answer, findings_used, agents}``.
    """
    gid = goal_id_of(goal)
    gm = _gm("synthesizer")
    # Durable path: rehydrate the tenant graph from the Qdrant dataplane.
    pulled = await gm.fleet_pull()
    logger.info("synthesizer pulled %d nodes from dataplane", pulled)
    qdrant_findings = await _gather_findings(gm, gid)
    # Assembly path (owner's primary): the AitherFlux context superhighway.
    flux_findings = await flux_gather_findings(gid)
    logger.info("synthesizer findings: qdrant=%d flux=%d",
                len(qdrant_findings), len(flux_findings))

    if flux_findings:
        source_plane = "aitherflux"
        context = "\n\n".join(
            f"### {f['source']}\n{f['content']}" for f in flux_findings
        )
        findings_used = len(flux_findings)
        agents = sorted({
            f["source"].split(":")[1] for f in flux_findings
            if f.get("source", "").count(":") >= 1
        })
    elif qdrant_findings:
        source_plane = "qdrant"
        context = "\n\n".join(
            f"### Subtask {f['task_index']} (agent {f['agent_id']}): {f['subtask']}\n{f['content']}"
            for f in qdrant_findings
        )
        findings_used = len(qdrant_findings)
        agents = sorted({f["agent_id"] for f in qdrant_findings})
    else:
        return {"answer": "", "findings_used": 0, "agents": [],
                "error": "no findings on either the Flux or Qdrant plane"}
    answer = await _llm(
        _SYNTH_SYS, f"Original goal: {goal}\n\nAgent findings:\n{context}",
        effort=effort, backend=backend,
    )
    # Record the synthesized answer back to the graph (durable + shareable).
    await gm.add_node(
        label=f"synthesis:{gid}", node_type="synthesis", content=answer,
        tags=[gid, "synthesis"], importance=1.0,
        metadata={"goal_id": gid, "agents": agents, "goal": goal},
    )
    await gm.fleet_push_all_nodes()
    await gm.drain_sync()
    return {"answer": answer, "findings_used": findings_used,
            "agents": agents, "source_plane": source_plane}


async def _gather_findings(gm: GraphMemory, goal_id: str) -> list[dict]:
    """Read every finding node for a goal from the (now rehydrated) local graph."""
    out: list[dict] = []
    # GraphMemory has no public 'list by type+tag', so read the SQLite directly —
    # it's the local mirror of the shared dataplane after fleet_pull.
    import sqlite3
    try:
        conn = sqlite3.connect(gm._db_path)  # noqa: SLF001 — same-module intimacy
        rows = conn.execute(
            "SELECT content, metadata FROM nodes WHERE node_type = 'finding'"
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return out
    for content, md_json in rows:
        try:
            md = json.loads(md_json or "{}")
        except Exception:  # noqa: BLE001
            md = {}
        if md.get("goal_id") != goal_id:
            continue
        out.append({
            "content": content or "",
            "task_index": md.get("task_index", 0),
            "agent_id": md.get("agent_id", "?"),
            "subtask": md.get("subtask", ""),
        })
    return sorted(out, key=lambda f: f["task_index"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("AITHER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser("adk.swarm", description="AitherSwarm roles")
    sub = p.add_subparsers(dest="role", required=True)

    pp = sub.add_parser("plan", help="decompose a goal → task ledger")
    pp.add_argument("--goal", required=True)
    pp.add_argument("--tasks", type=int, default=6)
    pp.add_argument("--effort", type=int, default=6)
    pp.add_argument("--backend", default="")

    pw = sub.add_parser("worker", help="claim + work tasks, dump findings")
    pw.add_argument("--goal", required=True)
    pw.add_argument("--agent", required=True)
    pw.add_argument("--effort", type=int, default=5)
    pw.add_argument("--backend", default="")
    pw.add_argument("--max-tasks", type=int, default=3)

    ps = sub.add_parser("synthesize", help="assemble findings → final answer")
    ps.add_argument("--goal", required=True)
    ps.add_argument("--effort", type=int, default=8)
    ps.add_argument("--backend", default="")
    ps.add_argument("--out", default="")

    args = p.parse_args(argv)

    if args.role == "plan":
        tasks = asyncio.run(plan(args.goal, args.tasks, args.effort, args.backend))
        print(json.dumps({"goal_id": goal_id_of(args.goal), "tasks": tasks}, indent=2))
        return 0
    if args.role == "worker":
        findings = asyncio.run(
            worker(args.goal, args.agent, args.effort, args.backend, args.max_tasks))
        print(json.dumps({"agent": args.agent, "findings": findings}, indent=2))
        return 0
    if args.role == "synthesize":
        result = asyncio.run(synthesize(args.goal, args.effort, args.backend))
        blob = json.dumps(result, indent=2)
        print(blob)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(blob)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
