"""REAP baselines — external comparator implementations.

This subpackage hosts the third-party-tool baselines that the methods
paper compares REAP against. Each baseline is implemented as a pure
function that consumes the *same* pre-computed embeddings REAP receives,
so the cross-method comparison is fair (matched preprocessing, matched
encoder, matched seed list).

Currently exposed:

* ``run_bertopic_baseline`` — BERTopic's default pipeline
  (UMAP → HDBSCAN → c-TF-IDF). Pre-registered as baseline #5 in
  ``manuscript/evaluation_protocol.md`` §3.

Each baseline returns a frozen :class:`BaselineResult` carrying the
low-dimensional embedding, the per-document cluster labels, and the
per-cluster centroids in the embedding space — the same trio that
internal REAP methods produce (embedding from ``get_consensus_embedding``,
labels from KMeans, centroids derived from labels).
"""

from __future__ import annotations

from reap.baselines.bertopic_baseline import BaselineResult, run_bertopic_baseline
from reap.baselines.parametric_umap_baseline import (
    ParametricUMAPResult,
    run_parametric_umap_baseline,
)

__all__ = [
    "BaselineResult",
    "ParametricUMAPResult",
    "run_bertopic_baseline",
    "run_parametric_umap_baseline",
]
