"""Deterministic ODS model resolver.

This module is a THIN WRAPPER over `adk.ods._upstream_select`, which is the ODS
`scripts/select-model.py` vendored byte-for-byte under Apache-2.0. All selection
semantics — memory envelopes, family/profile routing, runtime-profile matching,
the Spark-aarch64 and unified-memory coder-next substitutions, the size ceiling —
live upstream. This file only adapts the result into ADK's typed API.

WHY A WRAPPER AND NOT A REIMPLEMENTATION
----------------------------------------
An earlier version of this file re-derived the selection algorithm in Python
from a prose description of upstream's behaviour. Differential-tested against
the upstream script across 20 hardware envelopes, that re-derivation returned a
different model in 16 of 20 cases — including one identical model for 8GB, 12GB,
24GB, 48GB and 96GB NVIDIA envelopes, because its memory-capacity computation
was wrong. Model selection gates every node that joins the compute market, so
upstream code is carried verbatim and this wrapper stays thin.

`adk/ods/tests/test_differential.py` re-runs that comparison as a real test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import _upstream_select as ups
from .data import load_catalog
from .model_types import ModelRecord, OdsError, OdsRecommendation

logger = logging.getLogger("adk.ods.resolver")

# Upstream emits a confidence *label*; ADK's API exposes a float. Keep the
# mapping explicit rather than inventing a score.
_CONFIDENCE_TO_FLOAT = {"high": 0.90, "medium": 0.75, "low": 0.60}


def _default_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "model-library.json"


def _to_record(model: dict[str, Any]) -> ModelRecord:
    """Adapt an upstream normalised model dict into the typed ModelRecord.

    Upstream's `normalize_model()` guarantees every key read here, so a missing
    key is a genuine contract break and should raise rather than be defaulted.
    """
    return ModelRecord(
        id=model["id"],
        name=model["name"],
        family=model["family"],
        gguf_file=model["gguf_file"],
        gguf_url=model["gguf_url"],
        gguf_sha256=model["gguf_sha256"],
        size_mb=model["size_mb"],
        vram_required_gb=model["vram_required_gb"],
        context_length=model["context_length"],
        quantization=model["quantization"],
        specialty=model["specialty"],
        llm_model_name=model["llm_model_name"],
        install_recommendation=model["install_recommendation"],
        runtime_profiles=model.get("runtime_profiles") or [],
        app_compatibility=model.get("app_compatibility") or [],
    )


class OdsResolver:
    """Deterministic, offline model selection from the vendored ODS catalog."""

    def __init__(self, catalog_path: str | None = None) -> None:
        self.catalog_path = catalog_path
        self._catalog: dict[str, Any] | None = None
        self._models: list[dict[str, Any]] | None = None
        # Fail fast on a missing/corrupt catalog rather than at first resolve().
        _ = self.catalog

    @property
    def catalog(self) -> dict[str, Any]:
        """Raw catalog dict (schema-validated). Raises OdsError if unusable."""
        if self._catalog is None:
            self._catalog = load_catalog(self.catalog_path)
        return self._catalog

    @property
    def models(self) -> list[dict[str, Any]]:
        """Upstream-normalised model list used for selection."""
        if self._models is None:
            path = Path(self.catalog_path) if self.catalog_path else _default_catalog_path()
            try:
                self._models = ups.load_catalog(path)
            except (OSError, ValueError) as e:
                raise OdsError(f"Failed to load catalog for selection: {e}") from e
            if not self._models:
                raise OdsError(
                    f"Model catalog at {path} contains no usable models — refusing to "
                    "return an empty pick (fail closed)."
                )
        return self._models

    def resolve(
        self,
        backend: str,
        memory_type: str | None,
        vram_mb: int,
        ram_gb: int,
        profile: str,
        tier: str | None = None,
        host_arch: str = "x86_64",
        max_size_mb: float | None = None,
        installable_only: bool = False,
    ) -> OdsRecommendation:
        """Resolve a model for the given hardware envelope.

        Mirrors `select-model.py main()` exactly, then adapts to the typed API.

        Raises:
            OdsError: catalog unusable, or no candidate survives the constraints.
                      Never returns an empty/None pick.
        """
        catalog = self.models
        tier_value = tier if tier is not None else "1"
        ceiling = float(max_size_mb) if max_size_mb else 0.0
        memory_type_value = memory_type if memory_type is not None else "discrete"

        resolved_profile = ups.effective_profile(
            ups.normalize_profile(profile), backend, tier_value
        )
        capacity_gb, memory_label = ups.usable_memory_gb(
            backend, memory_type_value, vram_mb, ram_gb
        )
        confidence_label = (
            "high"
            if ups.normalize_key(backend) not in {"unknown", "none"} and capacity_gb > 0
            else "medium"
        )

        ranked = ups.rank_models(
            catalog,
            capacity_gb,
            resolved_profile,
            installable_only,
            backend,
            memory_type_value,
            vram_mb,
            ram_gb,
            host_arch,
            ceiling,
        )
        if not ranked:
            raise OdsError(
                "No model in the ODS catalog satisfies the constraints "
                f"(backend={backend}, capacity={capacity_gb:.1f}GB, "
                f"profile={resolved_profile}, installable_only={installable_only}, "
                f"max_size_mb={ceiling or 'unbounded'}). Failing closed."
            )

        arch_selected, arch_policy_tag = ups.arch_policy_model(
            catalog,
            tier_value,
            resolved_profile,
            host_arch,
            memory_type_value,
            installable_only,
            ranked[0],
        )
        if arch_selected:
            selected = arch_selected
            alternatives = [selected] + [
                m
                for m in ranked
                if m["id"] != selected["id"] and not ups.is_spark_aarch64_excluded_model(m)
            ][:2]
            policy = f"{ups.POLICY}+{arch_policy_tag}"
            reason = ups.arch_policy_reason(
                selected, capacity_gb, memory_label, arch_policy_tag
            )
        else:
            selected = ranked[0]
            alternatives = ranked[:3]
            policy = ups.POLICY
            reason = ups.recommendation_reason(
                selected, capacity_gb, memory_label, backend, confidence_label
            )

        logger.debug(
            "ODS resolve: backend=%s capacity=%.1fGB profile=%s -> %s (policy=%s)",
            backend, capacity_gb, resolved_profile, selected["id"], policy,
        )

        return OdsRecommendation(
            policy=policy,
            source="ods",
            confidence=_CONFIDENCE_TO_FLOAT.get(confidence_label, 0.75),
            profile=resolved_profile,
            host_arch=ups.normalize_host_arch(host_arch),
            memory_capacity_gb=round(capacity_gb, 1),
            memory_label=memory_label,
            selected=_to_record(selected),
            reason=reason,
            alternatives=[_to_record(m) for m in alternatives],
        )
