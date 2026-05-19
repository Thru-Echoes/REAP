"""Common-space silhouette evaluation — defuses the silhouette-circularity reviewer concern.

A reviewer evaluating REAP's silhouette headline can point out that
silhouette is computed in each method's *own* embedding space, and that
REAP's distance-matrix consensus minimises the exact distance structure
silhouette rewards. To answer that, we evaluate every method's
``consensus_labels.npy`` against TWO shared reference spaces:

1. **Raw input embeddings** — the original pre-UMAP encoder output (e5
   for ai_art, MiniLM for korean_forest / twenty_newsgroups_reference).
   This is a metric-fair common ground.
2. **Shared canonical UMAP** — a single UMAP fit per dataset with a
   fixed seed (=42, distinct from the A/B/C manifest), shared across all
   methods. This is what a reader naively visualises.

The script reads existing committed consensus_labels.npy files; no new
benchmark runs needed. Output: one CSV per dataset at
``results/<dataset>/common_space_silhouette.csv`` with columns:

  ``method, set_name, sil_raw_input, sil_shared_umap, db_raw_input,
   db_shared_umap, ch_raw_input, ch_shared_umap, n_clusters``

For methods that lack consensus_labels.npy (currently ``single_seed``)
the row is emitted with empty cells.

Usage
-----
After all method runs have completed for a dataset::

    python scripts/analysis/common_space_silhouette.py --dataset korean_forest --dataset ai_art

Side effects
------------
- Reads ``~/.cache/reap/datasets/`` (cached dataset snapshots).
- Writes ``results/<dataset>/common_space_silhouette.csv``.
- Caches the canonical shared UMAP under
  ``results/<dataset>/_canonical_umap_seed42.npy`` for reuse across runs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from reap.datasets import (  # noqa: E402
    DatasetSnapshot,
    load_ai_art,
    load_korean_forest,
    load_twenty_newsgroups_reference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset registry — must match scripts/run_paper_benchmark.py config so
# the shared UMAP uses the same hyperparameters the consensus methods used.
# ---------------------------------------------------------------------------


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "korean_forest": {
        "loader": load_korean_forest,
        "n_components": 18,
        "n_neighbors": 19,
        "min_dist": 0.005,
    },
    "ai_art": {
        "loader": load_ai_art,
        "n_components": 5,
        "n_neighbors": 53,
        "min_dist": 0.01,
    },
    "twenty_newsgroups_reference": {
        "loader": load_twenty_newsgroups_reference,
        "n_components": 5,
        "n_neighbors": 15,
        "min_dist": 0.1,
    },
}

CANONICAL_SEED: int = 42
METHODS_WITH_CONSENSUS: list[str] = [
    "best_of_n",
    "naive_average",
    "procrustes",
    "reap",
    "bertopic",
    "parametric_umap",
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def get_canonical_umap(
    embeddings: np.ndarray,
    dataset_root: Path,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
) -> np.ndarray:
    """Build a single shared UMAP per dataset with a fixed canonical seed.

    Cached at ``<dataset_root>/_canonical_umap_seed42.npy`` so re-runs of
    this script don't repeat the fit.
    """
    cache_path = dataset_root / "_canonical_umap_seed42.npy"
    if cache_path.exists():
        cached: np.ndarray = np.load(cache_path)
        if cached.shape == (embeddings.shape[0], n_components):
            logger.info("Loaded canonical UMAP from cache: %s", cache_path)
            return cached

    from umap import UMAP

    logger.info(
        "Fitting canonical UMAP (seed=%d) on %s; n=%d d_in=%d d_out=%d",
        CANONICAL_SEED, dataset_root.name,
        embeddings.shape[0], embeddings.shape[1], n_components,
    )
    model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=CANONICAL_SEED,
    )
    fitted = np.asarray(model.fit_transform(embeddings))
    np.save(cache_path, fitted)
    return fitted


def evaluate_cell(
    labels: np.ndarray,
    raw_embeddings: np.ndarray,
    shared_umap: np.ndarray,
) -> dict[str, float | int | str]:
    """Compute silhouette / DB / CH for a single (method, set) cell in both spaces.

    Returns a row dict ready for writerow().
    """
    finite_labels_mask = labels >= 0
    n_clusters = len(set(int(c) for c in labels[finite_labels_mask].tolist()))
    if n_clusters < 2:
        return {
            "sil_raw_input": "",
            "sil_shared_umap": "",
            "db_raw_input": "",
            "db_shared_umap": "",
            "ch_raw_input": "",
            "ch_shared_umap": "",
            "n_clusters": n_clusters,
        }

    labels_for_metric = labels[finite_labels_mask]
    raw_for_metric = raw_embeddings[finite_labels_mask]
    umap_for_metric = shared_umap[finite_labels_mask]

    return {
        "sil_raw_input": _fmt(silhouette_score(raw_for_metric, labels_for_metric)),
        "sil_shared_umap": _fmt(silhouette_score(umap_for_metric, labels_for_metric)),
        "db_raw_input": _fmt(davies_bouldin_score(raw_for_metric, labels_for_metric)),
        "db_shared_umap": _fmt(davies_bouldin_score(umap_for_metric, labels_for_metric)),
        "ch_raw_input": _fmt(calinski_harabasz_score(raw_for_metric, labels_for_metric)),
        "ch_shared_umap": _fmt(calinski_harabasz_score(umap_for_metric, labels_for_metric)),
        "n_clusters": n_clusters,
    }


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def run_one_dataset(dataset_name: str, output_dir: Path) -> None:
    """Compute common-space silhouette/DB/CH for all method × set cells of one dataset.

    Side effects:
      - Loads dataset snapshot from cache.
      - Builds (or reuses cached) canonical shared UMAP fit.
      - Writes ``<output_dir>/<dataset>/common_space_silhouette.csv``.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    dataset_root = output_dir / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    snap: DatasetSnapshot = cfg["loader"]()
    raw = np.asarray(snap.embeddings, dtype=np.float32)
    shared_umap = get_canonical_umap(
        raw, dataset_root,
        n_components=cfg["n_components"],
        n_neighbors=cfg["n_neighbors"],
        min_dist=cfg["min_dist"],
    )

    fieldnames = [
        "method", "set_name", "n_clusters",
        "sil_raw_input", "sil_shared_umap",
        "db_raw_input", "db_shared_umap",
        "ch_raw_input", "ch_shared_umap",
    ]
    rows: list[dict[str, Any]] = []
    for method in METHODS_WITH_CONSENSUS:
        for set_name in ["A", "B", "C"]:
            labels_path = dataset_root / method / f"set_{set_name}" / "consensus_labels.npy"
            if not labels_path.exists():
                logger.info(
                    "skipping %s/%s/set_%s (no consensus_labels.npy)",
                    dataset_name, method, set_name,
                )
                continue
            labels = np.load(labels_path)
            metric_row = evaluate_cell(labels, raw, shared_umap)
            rows.append({
                "method": method,
                "set_name": set_name,
                **metric_row,
            })
            logger.info(
                "%s / %s / set_%s : sil_raw=%s  sil_shared=%s  k=%s",
                dataset_name, method, set_name,
                metric_row["sil_raw_input"], metric_row["sil_shared_umap"],
                metric_row["n_clusters"],
            )

    out_csv = dataset_root / "common_space_silhouette.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), out_csv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Common-space silhouette/DB/CH for every method × set.",
    )
    parser.add_argument(
        "--dataset", action="append", choices=sorted(DATASET_CONFIGS), required=True,
        help="Dataset(s) to evaluate. Pass multiple times.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Result root directory (default 'results/').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    for dataset_name in args.dataset:
        run_one_dataset(dataset_name, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
