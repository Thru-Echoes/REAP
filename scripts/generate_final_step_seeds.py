"""Draw REAP's final-step and projection-head seeds once, from fresh randomness.

Seeded steps use fixed seeds drawn at random once, not constants such as 0 or
42. This script draws ten final-step seeds (the first seeds REAP's main final
map and its k-means; all ten are the repeats that measure how labels move with
the final seed) and one seed for the projection head, from operating-system
entropy, disjoint from every seed set in the given seed manifest. It records
the entropy, so the draw can be re-derived, and refuses to overwrite an
existing file: a draw is made once and never redrawn.

The seed file is research material and lives in the research repository; the
reference-plus-probes driver reads it with ``--final-step-seeds``.

Re-derive: ``get_seeds(int(entropy), 11, exclude)`` reproduces the eleven
seeds bit for bit.

Usage::

    python scripts/generate_final_step_seeds.py --output <seed file> \\
        --manifest manuscript/seeds/seed_manifest.json

Side effects: writes ``--output`` once.
"""

from __future__ import annotations

import argparse
import datetime
import json
import secrets
import sys
from pathlib import Path

import numpy as np

N_FINAL_STEP = 10
SEED_HIGH_EXCLUSIVE = 2**31


def get_seeds(entropy: int, n: int, exclude: set[int]) -> list[int]:
    """``n`` distinct seeds in [0, 2**31) not in ``exclude``, from one generator."""
    rng = np.random.default_rng(entropy)
    seeds: list[int] = []
    while len(seeds) < n:
        s = int(rng.integers(0, SEED_HIGH_EXCLUSIVE))
        if s not in exclude and s not in seeds:
            seeds.append(s)
    return seeds


def get_excluded(manifest: Path) -> set[int]:
    """Every seed of every set in a seed manifest. Side effects: reads the file."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {int(s) for v in data["sets"].values() for s in v["seeds"]}


def main(argv: list[str] | None = None) -> int:
    """Draw and write the seeds; return 0. Refuses (returns 1) if the file exists."""
    p = argparse.ArgumentParser(description="Draw REAP's final-step and head seeds once.")
    p.add_argument("--output", required=True, help="Seed file to write (must not exist).")
    p.add_argument("--manifest", required=True, help="Seed manifest whose sets are excluded.")
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        print(f"{out} already exists: the seeds are drawn once and never redrawn",
              file=sys.stderr)
        return 1
    exclude = get_excluded(Path(args.manifest))
    entropy = secrets.randbits(128)
    seeds = get_seeds(entropy, N_FINAL_STEP + 1, exclude)
    record = {
        "schema_version": 1,
        "description": (
            "Seeds drawn once from operating-system entropy for REAP's final step and "
            "projection head. final_step_seeds[0] seeds the main final map and its k-means; "
            "all ten are the final-step repeats. head_seed seeds the projection head."
        ),
        "final_step_seeds": seeds[:N_FINAL_STEP],
        "head_seed": seeds[N_FINAL_STEP],
        "provenance": {
            "generated_by": "REAP scripts/generate_final_step_seeds.py",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rng": "numpy.random.default_rng (PCG64)",
            "entropy": str(entropy),
            "exclude": f"every seed set of {Path(args.manifest).name} ({len(exclude)} seeds)",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
