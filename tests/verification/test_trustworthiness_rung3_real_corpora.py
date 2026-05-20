"""Rung 3 + Rung 4 - real-corpora trustworthiness cross-check.

Combined into one file because the protocol is identical across all three
production corpora (20-Newsgroups, AI-art, Korean forest). The plan
nominally separates Rung 3 (20NG) and Rung 4 (real corpora); for
trustworthiness we collapse them because the loading mechanism is the
same (datasets via `reap.datasets`, consensus embedding via the
`results/.../consensus_embedding.npy` file). Each (corpus, set) is
verified independently.

For each (corpus, set):
    - SHA256-pin the consensus_embedding.npy (the load-bearing low-d input).
    - Load X_high via `reap.datasets.load_<corpus>()` (only filter-retained
      rows participate in trustworthiness; the consensus is computed on the
      retained subset).
    - Recompute trustworthiness with `compute_trustworthiness` and with
      `_reference_trustworthiness`, both using the production metric
      ("cosine") and the production n_neighbors (15, per
      `benchmarks.py:380`).
    - Production ↔ CSV-published value within 1e-6.
    - Production ↔ from-scratch reference within 1e-1 (cosine tie-breaking
      tolerance from Rung 1).

Manuscript implication of this rung: the published `trustworthiness`
numbers in `results/<corpus>/combined_set_*/all_methods.csv` are
cosine-metric values computed under sklearn's argpartition-tie-breaking
convention. Document this in the methods section so a future reader knows
not to compare these numbers against a Euclidean-metric alternative
without re-computing.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from _reference_trustworthiness import reference_trustworthiness

from reap.evaluation import compute_trustworthiness

TOL_PROD_VS_CSV = 1e-6
TOL_PROD_VS_REF_COSINE = 1e-1  # Cosine tie-breaking tolerance (pinned by Rung 1)

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
RESULTS_ROOT = REPO_ROOT / "results"

EXPECTED_CONSENSUS_EMB_HASHES: dict[str, str] = {
    "twenty_newsgroups_reference/reap/set_A/consensus_embedding.npy": (
        "f2f04bd5b76d1283a85f29885a553a72cb8d4a5b5c267c9ab58d2229b7aa1401"
    ),
    "twenty_newsgroups_reference/reap/set_B/consensus_embedding.npy": (
        "2369b9f31922deb3053cee7f23e92332aea7d3fca5091294ecea43d66e055a41"
    ),
    "twenty_newsgroups_reference/reap/set_C/consensus_embedding.npy": (
        "7860fcf5b10e1c881fc08d6a70358e93c781482862996ddfcce5c14f4e737dde"
    ),
    "ai_art/reap/set_A/consensus_embedding.npy": (
        "bc7c53fa7891b09dfd3adfb16a1c219c1ef2a5176d265494c3a0256dc264cdbe"
    ),
    "ai_art/reap/set_B/consensus_embedding.npy": (
        "23f93c2276520570d7132afc7c111fa6cb1e764549c828af796f2f9f55992d48"
    ),
    "ai_art/reap/set_C/consensus_embedding.npy": (
        "634d23c33946f281c246f2155bcb0e88fc6a5bfec8e1332111def2b9bf4940b8"
    ),
    "korean_forest/reap/set_A/consensus_embedding.npy": (
        "6eaeb17dc6e8af5c6c90c73b331f005e11f1c2f384e994848b0a597b0ed7c513"
    ),
    "korean_forest/reap/set_B/consensus_embedding.npy": (
        "30f5ba5d7711588f00816db0d525a2891f75f9cd0d76a562530efde0f31402d4"
    ),
    "korean_forest/reap/set_C/consensus_embedding.npy": (
        "5153349717d52aef23d530981a5d31929d03058cb0c8e9b5c22a519c2867076a"
    ),
}

CORPUS_LOADERS = {
    "twenty_newsgroups_reference": "load_twenty_newsgroups_reference",
    "ai_art": "load_ai_art",
    "korean_forest": "load_korean_forest",
}

CORPORA = tuple(CORPUS_LOADERS)
SETS = ("A", "B", "C")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_consensus_emb(rel: str) -> np.ndarray:
    path = RESULTS_ROOT / rel
    actual = _sha256_of(path)
    expected = EXPECTED_CONSENSUS_EMB_HASHES[rel]
    if actual != expected:
        raise AssertionError(
            f"SHA256 mismatch for {rel}: got {actual!r}, expected {expected!r}"
        )
    return np.load(path)


def _load_x_high(corpus: str) -> np.ndarray:
    """Load the original sentence embeddings via the production loader."""
    import reap.datasets as ds

    loader = getattr(ds, CORPUS_LOADERS[corpus])
    snapshot = loader()
    return np.asarray(snapshot.embeddings, dtype=np.float64)


def _read_published_trustworthiness(corpus: str, set_letter: str) -> dict[str, float]:
    csv_path = RESULTS_ROOT / corpus / f"combined_set_{set_letter}" / "all_methods.csv"
    with open(csv_path) as fp:
        for row in csv.DictReader(fp):
            if row.get("method") == "reap":
                return {
                    "trustworthiness": float(row["trustworthiness"]),
                    "consensus_trustworthiness": float(row["consensus_trustworthiness"]),
                }
    raise AssertionError(f"REAP row not found in {csv_path}")


# ---------------------------------------------------------------------------
# Provenance pins (consensus embeddings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", list(EXPECTED_CONSENSUS_EMB_HASHES.keys()))
def test_rung3_consensus_emb_hashes_pinned(rel):
    actual = _sha256_of(RESULTS_ROOT / rel)
    expected = EXPECTED_CONSENSUS_EMB_HASHES[rel]
    assert actual == expected


# ---------------------------------------------------------------------------
# Production ↔ published CSV (the manuscript-feeding gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_rung3_production_matches_published_csv(corpus, set_letter):
    """Production `compute_trustworthiness` matches the published
    `consensus_trustworthiness` CSV column for REAP within 1e-6."""
    consensus_emb = _load_consensus_emb(
        f"{corpus}/reap/set_{set_letter}/consensus_embedding.npy"
    )
    x_high = _load_x_high(corpus)

    # Production pipeline uses n_neighbors = min(15, n_samples - 1).
    n_nn = min(15, consensus_emb.shape[0] - 1)

    # Production metric default for trustworthiness in the benchmark is
    # `input_metric = "cosine"` (benchmarks.py:258).
    prod = compute_trustworthiness(
        x_high, consensus_emb, n_neighbors=n_nn, metric="cosine"
    )
    published = _read_published_trustworthiness(corpus, set_letter)

    delta = prod - published["consensus_trustworthiness"]
    assert abs(delta) < TOL_PROD_VS_CSV, (
        f"{corpus} set {set_letter}: production={prod!r} vs "
        f"published_consensus_trustworthiness={published['consensus_trustworthiness']!r} "
        f"(delta={delta!r})"
    )

    # `trustworthiness` column = mean of per-seed trustworthiness for
    # methods that don't have a consensus (e.g. single_seed). For methods
    # with a consensus, both columns equal each other; we DO NOT assume
    # `prod == trustworthiness` here because that column is the per-seed
    # mean, which we haven't recomputed at this rung.


# ---------------------------------------------------------------------------
# Production ↔ from-scratch reference (cosine, loose due to tie-breaking)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_rung3_production_matches_reference_cosine_loose(corpus, set_letter):
    """Production ↔ from-scratch reference within 1e-1 (cosine tie-breaking)."""
    consensus_emb = _load_consensus_emb(
        f"{corpus}/reap/set_{set_letter}/consensus_embedding.npy"
    )
    x_high = _load_x_high(corpus)
    n_nn = min(15, consensus_emb.shape[0] - 1)

    prod = compute_trustworthiness(
        x_high, consensus_emb, n_neighbors=n_nn, metric="cosine"
    )
    ref = reference_trustworthiness(
        x_high, consensus_emb, n_neighbors=n_nn, metric="cosine"
    )

    delta = prod - ref
    assert abs(delta) < TOL_PROD_VS_REF_COSINE, (
        f"{corpus} set {set_letter}: cosine prod={prod!r} ref={ref!r} delta={delta!r}"
    )
