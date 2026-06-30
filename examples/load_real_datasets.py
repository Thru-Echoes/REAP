#!/usr/bin/env python
"""Example: load the real paper datasets and run a small REAP consensus.

Demonstrates the `reap.datasets` API on the two currently-locked
corpora — AI-art discourse (1,736 × 1,024 e5-large-v2) and Korean
forest policy (905 × 384 multilingual MiniLM).

Prerequisites
-------------
Both snapshots must already be materialized in the REAP cache
(``~/.cache/reap/datasets/``). To build them once from the sibling
research projects::

    python scripts/build_datasets.py --all \\
        --ai-art-source ~/Documents/Berkeley/Research/When-Algorithms-Meet-Artists \\
        --korean-forest-source ~/Documents/Berkeley/Research/green-narrative/hye_in

Usage
-----
    python examples/load_real_datasets.py

Side effects
------------
- Reads from the REAP cache.
- Runs `run_consensus_pipeline` (CPU-bound; ~10–30 s per dataset at the
  small seed count used below).
- Writes nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import numpy as np

from reap import run_consensus_pipeline
from reap.datasets import (
    DatasetSnapshot,
    load_ai_art,
    load_korean_forest,
)
from reap.evaluation import compute_silhouette, compute_trustworthiness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def summarize_snapshot(snap: DatasetSnapshot) -> None:
    """Print a human-readable summary of a snapshot's provenance."""
    m = snap.metadata
    notes = json.loads(m.notes) if m.notes else {}
    print(f"\n=== {m.name} @ {m.version} ===")
    print(f"  n_samples           : {m.n_samples}")
    print(f"  embedding_dim       : {m.embedding_dim}")
    print(f"  embedding_model     : {m.embedding_model}")
    print(f"  preprocessing       : {m.preprocessing_version}")
    print(f"  sha256              : {m.sha256[:16]}…")
    print(f"  license             : {m.license}")
    print(f"  citation            : {m.citation!r}")
    if "citation_status" in notes:
        print(f"  citation_status     : {notes['citation_status']}")
    if snap.labels is None:
        print("  labels              : None (no per-chunk expert labels)")
    else:
        n_classes = len(set(snap.labels.tolist()))
        print(f"  labels              : {snap.labels.shape}, {n_classes} classes")
    print(f"  texts               : {len(snap.texts) if snap.texts else 0} strings")


def run_small_consensus(
    snap: DatasetSnapshot,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    """Run REAP consensus at the given parameters and print quick metrics.

    Uses a small seed count (default 5) so the demo completes in <1 min
    per dataset on laptop CPU. Not a paper-grade run — for that, see
    `examples/run_benchmark.py`.
    """
    logger.info(
        "running REAP consensus on %s (seeds=%d, d=%d, nn=%d, md=%g)",
        snap.metadata.name, len(seeds), n_components, n_neighbors, min_dist,
    )
    embedding, _meta = run_consensus_pipeline(
        X=snap.embeddings,
        seeds=seeds,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    assert np.isfinite(embedding).all(), "consensus embedding contains non-finite"

    n_clusters = (
        len(set(snap.labels.tolist())) if snap.labels is not None else 8
    )
    from sklearn.cluster import KMeans

    km_labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit_predict(
        embedding
    )
    trust = compute_trustworthiness(snap.embeddings, embedding, n_neighbors=15)
    sil = compute_silhouette(embedding, km_labels)

    print(f"  consensus shape     : {embedding.shape}")
    print(f"  trustworthiness@15  : {trust:.3f}")
    print(f"  silhouette (K={n_clusters:>2}) : {sil:.3f}")

    if snap.labels is not None:
        from sklearn.metrics import adjusted_rand_score

        ari = adjusted_rand_score(snap.labels, km_labels)
        print(f"  ARI vs expert labels: {ari:.3f}")


def _demo(
    label: str,
    loader: Callable[[], DatasetSnapshot],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> None:
    try:
        snap = loader()
    except FileNotFoundError as exc:
        logger.warning("skipping %s: %s", label, exc)
        return
    summarize_snapshot(snap)
    run_small_consensus(
        snap,
        seeds=[0, 1, 2, 3, 4],
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def main() -> int:
    # AI-art params follow the known-good config from the WAMA project
    # (config B in figures/config_comparison/). Korean forest params
    # follow Hye In's validated config (d=18, nn=19, md=0.005). Both
    # are reduced-seed demos, not paper-grade runs.
    _demo(
        "ai_art",
        load_ai_art,
        n_components=5,
        n_neighbors=53,
        min_dist=0.01,
    )
    _demo(
        "korean_forest",
        load_korean_forest,
        n_components=18,
        n_neighbors=19,
        min_dist=0.005,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
