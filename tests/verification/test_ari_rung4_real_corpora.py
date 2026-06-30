"""Rung 4 - AI-art + Korean forest cross-check ARI verification.

Final rung of the s2s/s2c ARI track of the verification ladder. Applies the
same protocol as Rung 3 (20-Newsgroups) to the two production corpora:

    - ai_art       : 1,742 chunks, no expert ground-truth labels in pipeline
                     (ext_ari column is null); 30 seeds per set; K varies
                     per seed via find_best_k.
    - korean_forest: 905 chunks, 5 expert categories as ground truth;
                     30 seeds per set; K varies per seed.

For every (corpus, set) tuple:
    - SHA256 pin on `reap/set_{S}/seed_labels.npy` and `consensus_labels.npy`
    - Production ``compute_seed_stability`` matches the from-scratch reference
      path (built atop ``_reference_ari`` + numpy mean/std/median) within 1e-9
      on all six returned statistics
    - The 30×30 pairwise ARI matrix has diagonal 1.0, is symmetric, and lies
      in [-1, 1]
    - The published numbers in
      ``results/{corpus}/combined_set_{S}/all_methods.csv`` (REAP row) match
      the recomputed numbers within 1e-6 (the CSV's 6-decimal precision)

This closes Rungs 0-4 for the s2s and s2c ARI metrics on the three corpora
that feed the manuscript. Anything that quotes those exact REAP s2s/s2c ARI
numbers becomes Rungs-0-through-4 verified.

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), Task 5.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from _reference_ari import reference_ari

from reap.evaluation import compute_pairwise_ari, compute_seed_stability

TOL = 1e-9
CSV_TOL = 1e-6

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
RESULTS_ROOT = REPO_ROOT / "results"

# SHA256 pins computed on 2026-05-20. A mismatch trips the rung.
EXPECTED_HASHES: dict[str, str] = {
    # ai_art
    "ai_art/reap/set_A/seed_labels.npy": (
        "aaf5b4eced943fce37ab1affc65a08dc7bf70d6ee2d44d52b0ce8ca100e71868"
    ),
    "ai_art/reap/set_A/consensus_labels.npy": (
        "d790bd237b325350e3d64383bd77e90d743356ac70c0d7bdedce4c11a87d01db"
    ),
    "ai_art/reap/set_B/seed_labels.npy": (
        "76ebb25f238572d8647e8d6d70033fa98e549b5b1c1087d7ffd1a206bdce8f7c"
    ),
    "ai_art/reap/set_B/consensus_labels.npy": (
        "6e040b94df48ef140b26e2518834ec7842cac38b9deeae4e31e67e930d8c8df6"
    ),
    "ai_art/reap/set_C/seed_labels.npy": (
        "5c8020349620a430363523d77184e32465d3d27ae26d2121ce113bafc099fddd"
    ),
    "ai_art/reap/set_C/consensus_labels.npy": (
        "112e185e1e097c391733368c81103521d2e9a60164fe32e3ff4d53b66ea17ee9"
    ),
    # korean_forest
    "korean_forest/reap/set_A/seed_labels.npy": (
        "fb14d090ed6302ced57d74c49158bf33d0d4e51fb02da797ffef9d5dfd30883a"
    ),
    "korean_forest/reap/set_A/consensus_labels.npy": (
        "66aad4d4fa991fd489503d4e342cd3e8174bf1201d4388e510e4e96d91a29426"
    ),
    "korean_forest/reap/set_B/seed_labels.npy": (
        "e75b14e2b9ae5bd3c14af8972533d4c795cccdae4b5c6540252b19a0e1b1bc0f"
    ),
    "korean_forest/reap/set_B/consensus_labels.npy": (
        "80110ec1ecb93908cfa98a555449f583c858face4c662639cc02879d79248fd7"
    ),
    "korean_forest/reap/set_C/seed_labels.npy": (
        "123e784910063eca710f83d42d200e1cfa62ad3b807775df92a0a958b9664f01"
    ),
    "korean_forest/reap/set_C/consensus_labels.npy": (
        "ebf5652baca0d6541530bbdf6a91148098c0fb5b76b675d0e93e1ee3d58ffa02"
    ),
}

CORPORA = ("ai_art", "korean_forest")
SETS = ("A", "B", "C")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pinned(rel: str) -> np.ndarray:
    path = RESULTS_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"Rung 4 input missing: {path}")
    actual = _sha256_of(path)
    expected = EXPECTED_HASHES[rel]
    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch for {rel}: got {actual!r}, expected {expected!r}. "
            "Input array was silently regenerated; Rung 4 cannot proceed."
        )
    return np.load(path)


def _reference_seed_stability(
    seed_labels: np.ndarray, consensus_labels: np.ndarray
) -> dict[str, float]:
    """Same six statistics as ``compute_seed_stability`` via ``reference_ari``."""
    n_seeds = seed_labels.shape[0]
    s2s_values: list[float] = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            s2s_values.append(reference_ari(seed_labels[i], seed_labels[j]))
    s2s = np.asarray(s2s_values, dtype=np.float64)
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


def _read_published_row(corpus: str, set_letter: str) -> dict[str, float]:
    """Pull REAP's published s2s/s2c numbers from the combined CSV."""
    csv_path = RESULTS_ROOT / corpus / f"combined_set_{set_letter}" / "all_methods.csv"
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
# Provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", list(EXPECTED_HASHES.keys()))
def test_rung4_input_hashes_pinned(rel):
    """Every Rung-4 input array's SHA256 matches the pinned value."""
    actual = _sha256_of(RESULTS_ROOT / rel)
    expected = EXPECTED_HASHES[rel]
    assert actual == expected, (
        f"{rel}: SHA256 drift detected: got {actual!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Code-vs-reference (the load-bearing claim)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_rung4_compute_seed_stability_matches_reference(corpus, set_letter):
    """Production ↔ from-scratch reference: every statistic within 1e-9."""
    seed_labels = _load_pinned(f"{corpus}/reap/set_{set_letter}/seed_labels.npy")
    consensus_labels = _load_pinned(
        f"{corpus}/reap/set_{set_letter}/consensus_labels.npy"
    )

    assert seed_labels.ndim == 2
    assert consensus_labels.ndim == 1
    assert seed_labels.shape[1] == consensus_labels.shape[0]

    production = compute_seed_stability(list(seed_labels), consensus_labels)
    reference = _reference_seed_stability(seed_labels, consensus_labels)

    for key in production:
        prod = production[key]
        ref = reference[key]
        assert abs(prod - ref) < TOL, (
            f"{corpus} set {set_letter}: {key} disagrees beyond {TOL}: "
            f"production={prod!r} vs reference={ref!r} (delta={prod - ref!r})"
        )


# ---------------------------------------------------------------------------
# Pairwise-matrix invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_rung4_pairwise_matrix_invariants(corpus, set_letter):
    """30×30 pairwise ARI: diagonal 1.0, symmetric, bounded in [-1, 1]."""
    seed_labels = _load_pinned(f"{corpus}/reap/set_{set_letter}/seed_labels.npy")
    matrix = compute_pairwise_ari(list(seed_labels))

    assert matrix.shape == (30, 30)
    assert np.array_equal(matrix, matrix.T), (
        f"{corpus} set {set_letter}: pairwise ARI matrix is not symmetric"
    )
    assert np.all(np.diag(matrix) == 1.0)
    assert ((matrix >= -1.0) & (matrix <= 1.0)).all()


# ---------------------------------------------------------------------------
# Closes the loop: published CSV vs recomputed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_rung4_published_csv_matches_recomputed(corpus, set_letter):
    """Recomputed s2s/s2c match the published CSV row within CSV-precision (1e-6)."""
    seed_labels = _load_pinned(f"{corpus}/reap/set_{set_letter}/seed_labels.npy")
    consensus_labels = _load_pinned(
        f"{corpus}/reap/set_{set_letter}/consensus_labels.npy"
    )

    recomputed = compute_seed_stability(list(seed_labels), consensus_labels)
    published = _read_published_row(corpus, set_letter)

    for key in ("s2s_ari_mean", "s2s_ari_std", "s2c_ari_mean", "s2c_ari_std"):
        delta = recomputed[key] - published[key]
        assert abs(delta) < CSV_TOL, (
            f"{corpus} set {set_letter}: {key} published vs recomputed beyond {CSV_TOL}: "
            f"published={published[key]!r} vs recomputed={recomputed[key]!r} (delta={delta!r})"
        )
