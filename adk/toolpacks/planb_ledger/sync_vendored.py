"""Plan B Ledger — copy the PURE modules to every consumer that needs them.

There is exactly one source of truth for the ledger merge rules
(``engine.py``) and the paper-sheet layout (``sheet_render.py``). Faces that
cannot import this package get a byte-identical vendored copy, produced HERE
and asserted by ``test_planb_vendored_in_sync``.

Why a generator plus an assertion, rather than trusting a one-time copy: an
artifact with no build step and no drift check is how a mirror silently rots
(the AitherConnect bonsai-worker case). Copies made by hand diverge; copies
made by a script that nothing re-runs diverge more quietly.

Run: python adk/toolpacks/planb_ledger/sync_vendored.py [--check]
``--check`` exits 1 on drift and writes nothing — that is the CI shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent
REPO = PACK.parents[3]

PURE = ["engine.py", "sheet_render.py"]

# consumer dir -> filename prefix (backend keeps a flat module namespace)
TARGETS = {
    REPO / "AitherOS" / "apps" / "packages" / "portal-kit-backend": "planb_",
}

HEADER = (
    "# GENERATED — do not edit. Source of truth:\n"
    "#   aither-adk/adk/toolpacks/planb_ledger/{name}\n"
    "# Regenerate: python adk/toolpacks/planb_ledger/sync_vendored.py\n"
    "# Drift is asserted by test_planb_vendored_in_sync.\n"
)


def rendered(name: str) -> str:
    body = (PACK / name).read_text(encoding="utf-8")
    # The vendored engine is a FLAT module (`planb_engine`). Consumers import it
    # either as a package member or flat, exactly like the rest of the backend,
    # so emit the same dual-import the house uses.
    body = body.replace(
        "from . import engine as ledger",
        "try:  # package import (deployed) / flat import (standalone app image)\n"
        "    from portal_kit_backend import planb_engine as ledger\n"
        "except ImportError:  # pragma: no cover\n"
        "    import planb_engine as ledger",
    )
    return HEADER.format(name=name) + body


def sync(check: bool = False) -> int:
    drift = 0
    for target_dir, prefix in TARGETS.items():
        if not target_dir.is_dir():
            print(f"SKIP (absent): {target_dir}")
            continue
        for name in PURE:
            dest = target_dir / f"{prefix}{name}"
            want = rendered(name)
            have = dest.read_text(encoding="utf-8") if dest.exists() else None
            if have == want:
                continue
            drift += 1
            if check:
                print(f"DRIFT: {dest} differs from canonical {name}")
            else:
                dest.write_text(want, encoding="utf-8")
                print(f"wrote {dest}")
    if check and drift:
        print(f"\n{drift} vendored file(s) out of sync — run sync_vendored.py")
        return 1
    if not check:
        print(f"in sync ({len(PURE)} pure module(s) x {len(TARGETS)} consumer(s))")
    return 0


if __name__ == "__main__":
    sys.exit(sync(check="--check" in sys.argv))
