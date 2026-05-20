"""Rung 3 - 20-Newsgroups cross-check silhouette verification.

Loads REAP's 20-Newsgroups reference snapshot for Sets A/B/C: the consensus
embedding (`consensus_embedding.npy`) and the consensus labels
(`consensus_labels.npy`). Computes silhouette two ways:

    1. ``reap.evaluation.compute_silhouette(consensus_emb, consensus_labels)``
       (production wrapper over `sklearn.metrics.silhouette_score`)
    2. ``_reference_silhouette(consensus_emb, consensus_labels)``
       (from-scratch Rung-0 verified Rousseeuw implementation)

and asserts agreement within 1e-9. The recomputed value is also checked
against REAP's published `silhouette` column in
``results/twenty_newsgroups_reference/combined_set_{A,B,C}/all_methods.csv``
within 1e-6 (the CSV's printed precision). SHA256 pins on the inputs.

This closes the silhouette ladder for 20NG. Per the silhouette ladder log,
Rung 3 by itself does not yet *settle* the circularity question — that
needs the per-seed vs consensus comparison (the tri-view), which Rung 4
exercises on real corpora.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from _reference_silhouette import reference_silhouette

from reap.evaluation import compute_silhouette

TOL = 1e-9
CSV_TOL = 1e-6
REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
RESULTS_ROOT = REPO_ROOT / "results" / "twenty_newsgroups_reference"

EXPECTED_HASHES: dict[str, str] = {
    "reap/set_A/consensus_embedding.npy": (
        "f2f04bd5b76d1283a85f29885a553a72cb8d4a5b5c267c9ab58d2229b7aa1401"
    ),
    "reap/set_A/consensus_labels.npy": (
        "27c61f70e3cf8b1c432baba93d45d6ac42b133e3dffc0761588d3534dcd71562"
    ),
    "reap/set_B/consensus_embedding.npy": (
        "2369b9f31922deb3053cee7f23e92332aea7d3fca5091294ecea43d66e055a41"
    ),
    "reap/set_B/consensus_labels.npy": (
        "212075918a313251762968e79dda0642914b58011fdf95fda76015f8f0a0e6d9"
    ),
    "reap/set_C/consensus_embedding.npy": (
        "7860fcf5b10e1c881fc08d6a70358e93c781482862996ddfcce5c14f4e737dde"
    ),
    "reap/set_C/consensus_labels.npy": (
        "135cfc98c546bb2052694a498358ae83ba671d09dde46e2d062e9efe9ce29541"
    ),
}


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pinned(rel: str) -> np.ndarray:
    path = RESULTS_ROOT / rel
    actual = _sha256_of(path)
    expected = EXPECTED_HASHES[rel]
    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch for {rel}: got {actual!r}, expected {expected!r}"
        )
    return np.load(path)


def _read_published_silhouette(set_letter: str) -> dict[str, float]:
    csv_path = RESULTS_ROOT / f"combined_set_{set_letter}" / "all_methods.csv"
    with open(csv_path) as fp:
        for row in csv.DictReader(fp):
            if row.get("method") == "reap":
                return {
                    "silhouette": float(row["silhouette"]),
                    "consensus_silhouette": float(row["consensus_silhouette"]),
                }
    raise AssertionError(f"REAP row not found in {csv_path}")


# ---------------------------------------------------------------------------
# Provenance pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", list(EXPECTED_HASHES.keys()))
def test_rung3_input_hashes_pinned(rel):
    """Every silhouette input file's SHA256 matches the pinned value."""
    actual = _sha256_of(RESULTS_ROOT / rel)
    expected = EXPECTED_HASHES[rel]
    assert actual == expected, (
        f"{rel}: SHA256 drift detected: got {actual!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Code-vs-reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_letter", ["A", "B", "C"])
def test_rung3_compute_silhouette_matches_reference(set_letter):
    """Production silhouette ↔ from-scratch Rung-0-verified silhouette within 1e-9."""
    emb = _load_pinned(f"reap/set_{set_letter}/consensus_embedding.npy")
    labels = _load_pinned(f"reap/set_{set_letter}/consensus_labels.npy")

    prod = compute_silhouette(emb, labels)
    ref = reference_silhouette(emb, labels)

    assert abs(prod - ref) < TOL, (
        f"set {set_letter}: production={prod!r} vs reference={ref!r}, delta={prod - ref!r}"
    )
    assert -1.0 <= prod <= 1.0


# ---------------------------------------------------------------------------
# Published-CSV-vs-recomputed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("set_letter", ["A", "B", "C"])
def test_rung3_published_csv_matches_recomputed(set_letter):
    """Recomputed silhouette matches the published `silhouette` and
    `consensus_silhouette` columns in the all_methods.csv REAP row,
    within 1e-6 (CSV's printed precision)."""
    emb = _load_pinned(f"reap/set_{set_letter}/consensus_embedding.npy")
    labels = _load_pinned(f"reap/set_{set_letter}/consensus_labels.npy")

    recomputed = compute_silhouette(emb, labels)
    published = _read_published_silhouette(set_letter)

    # Both `silhouette` and `consensus_silhouette` are the same number for
    # methods that report a consensus (REAP, Procrustes, etc.), per the
    # benchmarks harness. Verify both match.
    assert abs(recomputed - published["silhouette"]) < CSV_TOL, (
        f"set {set_letter}: silhouette publshed={published['silhouette']!r} "
        f"vs recomputed={recomputed!r}, delta={recomputed - published['silhouette']!r}"
    )
    assert abs(recomputed - published["consensus_silhouette"]) < CSV_TOL, (
        f"set {set_letter}: consensus_silhouette published={published['consensus_silhouette']!r} "
        f"vs recomputed={recomputed!r}, "
        f"delta={recomputed - published['consensus_silhouette']!r}"
    )
