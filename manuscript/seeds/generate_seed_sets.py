"""Deterministically generate REAP's Set B and Set C seed lists.

Runs ONCE to produce `seed_manifest.json`. Set A is adopted verbatim from
`green-narrative/hye_in/data/random_seeds_v2.json` (master_seed=20260120)
for continuity with prior published work. Sets B and C are drawn from the
same 32-bit positive-integer range [0, 2**31 - 1] using a documented new
master seed so they are exchangeable samples with Set A.

Why this design:
  * Set A ties REAP's manuscript results to the seeds already used in the
    Korean forest policy (green-narrative) project, giving continuity.
  * Sets B and C provide independent resamples of the seed-generating
    distribution, letting us quantify between-seed-set reliability of the
    REAP consensus (evt_004).
  * Drawing B and C without replacement and checking zero overlap with A
    ensures the three sets are disjoint.

Reproducibility:
  * master_seed_BC = 20260413 (the date this generator was first run).
  * Numpy default_rng PCG64 — deterministic across platforms for a given
    seed.
  * Output file `seed_manifest.json` is the committed artifact. The script
    should not be re-run; any re-run MUST produce byte-identical output.

No side effects other than writing `seed_manifest.json` next to this script.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SET_A_SOURCE_REPO = "green-narrative"
SET_A_SOURCE_PATH = "hye_in/data/random_seeds_v2.json"
SET_A_MASTER_SEED = 20260120

MASTER_SEED_BC = 20260413
SEED_RANGE_LOW = 0
SEED_RANGE_HIGH = 2**31 - 1
N_SEEDS_PER_SET = 30


def get_set_a() -> list[int]:
    """Return the 30 seeds from green-narrative/hye_in/data/random_seeds_v2.json.

    Hardcoded here to keep REAP's manifest self-contained; the canonical
    copy still lives in green-narrative.
    """
    return [
        1884790947, 1206441759, 589233666, 949766833, 376381057,
        1744035759, 269928035, 1759443864, 1868472089, 1452958465,
        1698401433, 923594877, 874926060, 800993601, 406060346,
        1611942755, 1920765303, 509457931, 1247518203, 1122313674,
        260655260, 864930855, 1679695910, 682141862, 445714205,
        668290747, 2013064847, 463123507, 1226639985, 94670597,
    ]


def get_sets_b_and_c(
    master_seed: int = MASTER_SEED_BC,
    n_per_set: int = N_SEEDS_PER_SET,
    low: int = SEED_RANGE_LOW,
    high: int = SEED_RANGE_HIGH,
    exclude: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Draw 2 * n_per_set unique integers from [low, high], split into B and C.

    Parameters
    ----------
    master_seed : deterministic seed for the PCG64 generator.
    n_per_set : seeds per returned set.
    low, high : inclusive / exclusive bounds matching numpy.integers semantics.
    exclude : seeds that must not appear in B or C (e.g., Set A).

    Returns
    -------
    (set_b, set_c) — two lists of length n_per_set, disjoint with each other
    and with `exclude`.
    """
    exclude_set = set(exclude or [])
    rng = np.random.default_rng(master_seed)
    drawn: list[int] = []
    while len(drawn) < 2 * n_per_set:
        # Draw more than needed to absorb rejections from `exclude`.
        candidates = rng.integers(low=low, high=high, size=4 * n_per_set).tolist()
        for c in candidates:
            if c in exclude_set or c in drawn:
                continue
            drawn.append(int(c))
            if len(drawn) == 2 * n_per_set:
                break
    return drawn[:n_per_set], drawn[n_per_set:]


def build_manifest() -> dict:
    """Assemble the full three-set manifest with full provenance."""
    set_a = get_set_a()
    set_b, set_c = get_sets_b_and_c(exclude=set_a)
    assert len(set(set_a) & set(set_b)) == 0, "Set A and Set B overlap"
    assert len(set(set_a) & set(set_c)) == 0, "Set A and Set C overlap"
    assert len(set(set_b) & set(set_c)) == 0, "Set B and Set C overlap"
    assert len(set_a) == N_SEEDS_PER_SET
    assert len(set_b) == N_SEEDS_PER_SET
    assert len(set_c) == N_SEEDS_PER_SET
    return {
        "schema_version": 1,
        "description": (
            "REAP three-seed-set design. Set A = legacy seeds from the "
            "green-narrative Korean forest policy project. Sets B and C = "
            "one-time deterministic draws from the same range, used to "
            "quantify between-seed-set reliability of the REAP consensus."
        ),
        "seed_range": {"low": SEED_RANGE_LOW, "high_exclusive": SEED_RANGE_HIGH + 1},
        "n_seeds_per_set": N_SEEDS_PER_SET,
        "sets": {
            "A": {
                "seeds": set_a,
                "provenance": {
                    "source_repo": SET_A_SOURCE_REPO,
                    "source_path": SET_A_SOURCE_PATH,
                    "master_seed": SET_A_MASTER_SEED,
                    "generated_at": "2026-01-20T09:23:45.337944",
                },
            },
            "B": {
                "seeds": set_b,
                "provenance": {
                    "master_seed": MASTER_SEED_BC,
                    "generated_by": "manuscript/seeds/generate_seed_sets.py",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "rng": "numpy.random.default_rng (PCG64)",
                },
            },
            "C": {
                "seeds": set_c,
                "provenance": {
                    "master_seed": MASTER_SEED_BC,
                    "generated_by": "manuscript/seeds/generate_seed_sets.py",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "rng": "numpy.random.default_rng (PCG64)",
                },
            },
        },
    }


def main() -> None:
    """Generate seed_manifest.json next to this script (one-shot)."""
    manifest = build_manifest()
    out_path = Path(__file__).resolve().parent / "seed_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path}")
    print(f"Set A (first 5): {manifest['sets']['A']['seeds'][:5]}")
    print(f"Set B (first 5): {manifest['sets']['B']['seeds'][:5]}")
    print(f"Set C (first 5): {manifest['sets']['C']['seeds'][:5]}")


if __name__ == "__main__":
    main()
