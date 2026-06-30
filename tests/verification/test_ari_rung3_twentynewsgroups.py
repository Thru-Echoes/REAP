"""Rung 3 - 20-Newsgroups cross-check ARI verification (verification ladder).

Loads the committed REAP 20-Newsgroups reference snapshot's per-seed and
consensus labels for Sets A/B/C, computes seed-to-seed (s2s) and
seed-to-consensus (s2c) ARI statistics two ways:

    1. via ``reap.evaluation.compute_seed_stability`` (production)
    2. via a from-scratch path built atop ``_reference_ari.reference_ari``

and asserts the two paths agree within 1e-9 (Rung 3 tolerance per the plan)
for all six returned statistics (mean / std / median, s2s / s2c). Also pins
input-array integrity via SHA256: any silent regeneration of the snapshot
will trip the rung and the ladder restarts from Rung 0.

For each set, also reads the published numbers from
``results/twenty_newsgroups_reference/combined_set_{A,B,C}/all_methods.csv``
and verifies that the recomputed numbers match the published numbers within
1e-9. This closes the loop: the numbers in the manuscript-feeding CSVs are
themselves produced by ``compute_seed_stability`` and therefore inherit
Rung 0-3 correctness once this rung is GREEN.

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), Task 4.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from _reference_ari import reference_ari

from reap.evaluation import compute_seed_stability

TOL = 1e-9

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
RESULTS_ROOT = REPO_ROOT / "results" / "twenty_newsgroups_reference"

# Pinned SHA256 hashes from inspection on 2026-05-20. Any change indicates
# silent regeneration of the input arrays and invalidates the rung.
EXPECTED_HASHES: dict[str, str] = {
    "reap/set_A/seed_labels.npy": (
        "675f98ca260cf967ec01ec5aff75d3b0d5ec35faac11e6bc9661708808f9d5eb"
    ),
    "reap/set_A/consensus_labels.npy": (
        "27c61f70e3cf8b1c432baba93d45d6ac42b133e3dffc0761588d3534dcd71562"
    ),
    "reap/set_B/seed_labels.npy": (
        "0f9218fe035093a11e6d9ea4b943ad00c7913180653bd5fecb594d798d283609"
    ),
    "reap/set_B/consensus_labels.npy": (
        "212075918a313251762968e79dda0642914b58011fdf95fda76015f8f0a0e6d9"
    ),
    "reap/set_C/seed_labels.npy": (
        "5db2640256e835a08b64ea46e57a85deb5e0133b5293b4b9c11773e80ec8ef6c"
    ),
    "reap/set_C/consensus_labels.npy": (
        "135cfc98c546bb2052694a498358ae83ba671d09dde46e2d062e9efe9ce29541"
    ),
}


def _sha256_of(path: Path) -> str:
    """Return the hex SHA256 of `path`'s contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pinned(rel: str) -> np.ndarray:
    """Load an .npy file under RESULTS_ROOT after verifying its pinned SHA256."""
    path = RESULTS_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"Rung 3 input missing: {path}")
    actual = _sha256_of(path)
    expected = EXPECTED_HASHES[rel]
    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch for {rel}: got {actual!r}, expected {expected!r}. "
            "An input array was silently regenerated; Rung 3 cannot proceed."
        )
    return np.load(path)


def _reference_seed_stability(
    seed_labels: np.ndarray, consensus_labels: np.ndarray
) -> dict[str, float]:
    """Compute the same six s2s/s2c statistics as `compute_seed_stability`
    using only `reference_ari` and numpy. No reap import, no sklearn.

    Returns the exact same key set as production: s2s_ari_mean/std/median,
    s2c_ari_mean/std/median (population std, ddof=0).
    """
    n_seeds = seed_labels.shape[0]
    # s2s: upper-triangle k=1 pairwise ARIs.
    s2s_values: list[float] = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            s2s_values.append(reference_ari(seed_labels[i], seed_labels[j]))
    s2s = np.asarray(s2s_values, dtype=np.float64)
    # s2c: each seed vs the consensus.
    s2c = np.asarray(
        [reference_ari(seed_labels[i], consensus_labels) for i in range(n_seeds)],
        dtype=np.float64,
    )
    return {
        "s2s_ari_mean": float(s2s.mean()),
        "s2s_ari_std": float(s2s.std()),
        "s2s_ari_median": float(np.median(s2s)),
        "s2c_ari_mean": float(s2c.mean()),
        "s2c_ari_std": float(s2c.std()),
        "s2c_ari_median": float(np.median(s2c)),
    }


def _read_published_row(set_letter: str) -> dict[str, float]:
    """Pull REAP's published s2s/s2c numbers for one set from the combined CSV."""
    csv_path = RESULTS_ROOT / f"combined_set_{set_letter}" / "all_methods.csv"
    with open(csv_path) as fp:
        for row in csv.DictReader(fp):
            if row.get("method") == "reap":
                return {
                    "s2s_ari_mean": float(row["s2s_ari_mean"]),
                    "s2s_ari_std": float(row["s2s_ari_std"]),
                    "s2c_ari_mean": float(row["s2c_ari_mean"]),
                    "s2c_ari_std": float(row["s2c_ari_std"]),
                }
    raise AssertionError(f"REAP row not found in {csv_path}")


# ---------------------------------------------------------------------------
# Provenance: SHA256 pinning over the load-bearing inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", list(EXPECTED_HASHES.keys()))
def test_rung3_input_hashes_pinned(rel):
    """Every load-bearing input array's SHA256 matches the pinned value."""
    actual = _sha256_of(RESULTS_ROOT / rel)
    expected = EXPECTED_HASHES[rel]
    assert actual == expected, (
        f"{rel}: SHA256 drift detected: got {actual!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Cross-check: compute_seed_stability ↔ reference_ari path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_letter", ["A", "B", "C"])
def test_rung3_compute_seed_stability_matches_reference(set_letter):
    """Production ↔ from-scratch reference: every statistic within 1e-9."""
    seed_labels = _load_pinned(f"reap/set_{set_letter}/seed_labels.npy")
    consensus_labels = _load_pinned(f"reap/set_{set_letter}/consensus_labels.npy")

    # Shape pre-flight (don't trust silently)
    assert seed_labels.ndim == 2, seed_labels.shape
    assert consensus_labels.ndim == 1, consensus_labels.shape
    assert seed_labels.shape[1] == consensus_labels.shape[0], (
        seed_labels.shape, consensus_labels.shape,
    )

    # compute_seed_stability expects a list of (n_samples,) arrays.
    production = compute_seed_stability(list(seed_labels), consensus_labels)
    reference = _reference_seed_stability(seed_labels, consensus_labels)

    for key in production:
        prod = production[key]
        ref = reference[key]
        assert abs(prod - ref) < TOL, (
            f"set {set_letter}: {key} disagrees beyond {TOL}: "
            f"production={prod!r} vs reference={ref!r} (delta={prod - ref!r})"
        )
        assert -1.0 <= prod <= 1.0 or key.endswith("_std"), (
            f"set {set_letter}: {key}={prod!r} outside [-1, 1] (and is not a std)"
        )


# ---------------------------------------------------------------------------
# Pairwise-matrix sanity: diagonal == 1.0 and matrix is symmetric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_letter", ["A", "B", "C"])
def test_rung3_pairwise_matrix_invariants(set_letter):
    """The full 30×30 pairwise ARI matrix has 1.0 on the diagonal and is symmetric."""
    from reap.evaluation import compute_pairwise_ari

    seed_labels = _load_pinned(f"reap/set_{set_letter}/seed_labels.npy")
    matrix = compute_pairwise_ari(list(seed_labels))

    assert matrix.shape == (30, 30)
    assert np.array_equal(matrix, matrix.T), "pairwise ARI matrix is not symmetric"
    assert np.all(np.diag(matrix) == 1.0), "diagonal is not all 1.0"
    assert ((matrix >= -1.0) & (matrix <= 1.0)).all(), "matrix has out-of-range entries"


# ---------------------------------------------------------------------------
# CSV-published values vs recomputed (closes the loop into the manuscript pipeline)
# ---------------------------------------------------------------------------


# Looser tolerance for CSV comparison: the combined_set_*/all_methods.csv
# values are written to 6 decimal places, so the maximum representable
# precision is ~5e-7. Using 1e-6 keeps the test honest: it confirms the
# published manuscript-feeding numbers match the recomputed values to within
# the published precision, no tighter.
CSV_TOL = 1e-6


@pytest.mark.parametrize("set_letter", ["A", "B", "C"])
def test_rung3_published_csv_matches_recomputed(set_letter):
    """Recomputed s2s/s2c (mean and std) match the published CSV row to within
    the CSV's own 6-decimal precision (1e-6 tolerance).

    The strong claim (recomputed equals the reference-impl exactly within 1e-9)
    is enforced by ``test_rung3_compute_seed_stability_matches_reference``.
    This test confirms the downstream artifact has not drifted from what the
    pipeline currently produces.
    """
    seed_labels = _load_pinned(f"reap/set_{set_letter}/seed_labels.npy")
    consensus_labels = _load_pinned(f"reap/set_{set_letter}/consensus_labels.npy")

    recomputed = compute_seed_stability(list(seed_labels), consensus_labels)
    published = _read_published_row(set_letter)

    for key in ("s2s_ari_mean", "s2s_ari_std", "s2c_ari_mean", "s2c_ari_std"):
        delta = recomputed[key] - published[key]
        assert abs(delta) < CSV_TOL, (
            f"set {set_letter}: {key} published vs recomputed beyond {CSV_TOL}: "
            f"published={published[key]!r} vs recomputed={recomputed[key]!r} (delta={delta!r})"
        )
