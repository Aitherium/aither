"""Graph-RAG memory: authority/activation scoring + governed ingest.

Self-contained building blocks for the self-maintaining memory layer:

* :mod:`adk.graph_rag.activation_scoring` — authority + spreading-activation
  + supersession-cascade scoring over the typed memory graph.
* :mod:`adk.graph_rag.governance` — append-only mutation ledger, conflict
  detection, reversible tombstones, and stable node ids.

Both are stdlib-only and are wired into :class:`adk.graph_memory.GraphMemory`
only when ``AITHER_UNIFIED_MEMORY`` is enabled.
"""
