"""Parametric UMAP bridge-worlds comparator on 20NG.

Direct head-to-head with REAP's bridge experiment
(scripts/experiments/run_20ng_oos_bridge.py): same reference + OOS data, same
filter, same evaluation tables — only the projection mechanism changes.

Method-of-record: train Sainburg 2021 Parametric UMAP on the 20NG reference
corpus, use its native Keras-encoder ``.transform()`` to project the OOS
documents into the same low-d space, then apply the SAME conformal filter
(``mahalanobis_no_correction``, α=0.01) that REAP uses. Cluster assignment is
KMeans-on-reference at the same ``k`` REAP selected (k=8 ground-truth), so the
two methods are compared on equal footing — same labels-on-reference, same
filter calibration, only the encoder differs.

Why this matters
----------------
Parametric UMAP is the only comparator in our 7-method set that natively
supports out-of-sample projection (the others can't project new data at all).
REAP's projection-head claim must therefore be benchmarked head-to-head
against Parametric UMAP. The expected outcome the manuscript is testing:
REAP's projection-head + filter is at least competitive with PUMAP's native
encoder + filter, and ideally REAP's filter rejects OOS more reliably because
REAP's reference space is more reproducible.

Usage
-----
::

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_20ng_oos_bridge_pumap.py --set A

Side effects
------------
- Fits a fresh Parametric UMAP model on the 800-doc reference (~1-2 min).
- Loads reference + OOS DatasetSnapshots from the REAP cache.
- Writes three CSVs under
  ``results/twenty_newsgroups_reference/oos_bridge/pumap_set_<X>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Reuse helpers from the REAP version (CSV writers, calibration wrapper, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_20ng_oos_bridge import (  # type: ignore[import-not-found]
    apply_and_assign,
    calibrate_filter,
    write_outputs,
)

from reap.datasets import (  # noqa: E402
    OOS_20NG_CLASSES,
    REFERENCE_20NG_CLASSES,
    load_twenty_newsgroups_oos,
    load_twenty_newsgroups_reference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> None:
    """Seed numpy, Python, TF for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf  # type: ignore[import-untyped]

    tf.random.set_seed(seed)


def fit_pumap_and_kmeans(
    ref_X: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """Fit Parametric UMAP on the reference and KMeans on its embedding.

    Returns a dict with:
      - ``model`` — fitted ParametricUMAP (has ``.transform`` for OOS)
      - ``ref_embedding`` — (n_ref, n_components) reference points in PUMAP space
      - ``ref_labels`` — (n_ref,) KMeans labels at the requested ``k``
      - ``ref_centroids`` — dict[cluster_id, centroid_array]
    """
    from sklearn.cluster import KMeans
    from umap.parametric_umap import ParametricUMAP  # type: ignore[import-untyped]

    _seed_all(seed)
    model = ParametricUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    ref_emb: np.ndarray = np.asarray(model.fit_transform(ref_X), dtype=np.float64)

    # n_init="auto" yields 10 for default init in sklearn>=1.4 — equivalent to
    # explicit 10 while matching the stricter stub signature.
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    ref_labels = np.asarray(km.fit_predict(ref_emb), dtype=np.int64)

    centroids: dict[int, np.ndarray] = {}
    for c in sorted({int(x) for x in ref_labels.tolist()}):
        mask = ref_labels == c
        if mask.sum() > 0:
            centroids[c] = ref_emb[mask].mean(axis=0)

    return {
        "model": model,
        "ref_embedding": ref_emb,
        "ref_labels": ref_labels,
        "ref_centroids": centroids,
    }


def project_oos_via_pumap(model: Any, oos_X: np.ndarray) -> np.ndarray:
    """Use the fitted ParametricUMAP encoder to transform OOS docs."""
    projected: np.ndarray = np.asarray(model.transform(oos_X), dtype=np.float64)
    return projected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parametric UMAP bridge-worlds comparator on 20NG.",
    )
    parser.add_argument(
        "--set", default="A", choices=["A", "B", "C"],
        help="Seed-set label used only for output-dir naming (PUMAP itself is single-seed).",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Result root directory (default 'results/').",
    )
    parser.add_argument(
        "--k", type=int, default=8,
        help=(
            "Number of clusters for KMeans on the reference PUMAP embedding "
            "(default 8 = ground-truth K for 20NG-reference). Matches REAP's "
            "consensus-K selection for fair comparison."
        ),
    )
    parser.add_argument(
        "--n-components", type=int, default=5,
        help="UMAP output dim (default 5, matches run_paper_benchmark config).",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15,
        help="UMAP n_neighbors (default 15, matches run_paper_benchmark config).",
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.1,
        help="UMAP min_dist (default 0.1, matches run_paper_benchmark config).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Random seed for ParametricUMAP fit + KMeans + TF. "
            "Defaults to first seed of the requested set in seed_manifest.json."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve seed from the manifest if not explicitly provided.
    if args.seed is None:
        import json
        manifest_path = (
            Path(__file__).resolve().parent.parent.parent
            / "manuscript" / "seeds" / "seed_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        seed = int(manifest["sets"][args.set]["seeds"][0])
        logger.info(
            "Using first seed of Set %s from manifest: %d",
            args.set, seed,
        )
    else:
        seed = args.seed

    results_root = Path(args.output_dir)
    out_dir = (
        results_root / "twenty_newsgroups_reference" / "oos_bridge"
        / f"pumap_set_{args.set}"
    )

    logger.info("Loading reference + OOS data...")
    ref_snap = load_twenty_newsgroups_reference()
    oos_snap = load_twenty_newsgroups_oos()
    ref_X = np.asarray(ref_snap.embeddings, dtype=np.float32)
    oos_X = np.asarray(oos_snap.embeddings, dtype=np.float32)
    ref_class_ids = np.asarray(ref_snap.labels)
    oos_class_ids = np.asarray(oos_snap.labels)
    logger.info(
        "Reference: n=%d d_in=%d  |  OOS: n=%d d_in=%d classes=%d",
        ref_X.shape[0], ref_X.shape[1],
        oos_X.shape[0], oos_X.shape[1],
        len(set(oos_class_ids.tolist())),
    )

    logger.info(
        "Fitting ParametricUMAP (n_comp=%d, k=%d, seed=%d) on reference...",
        args.n_components, args.k, seed,
    )
    pumap_state = fit_pumap_and_kmeans(
        ref_X, args.n_components, args.n_neighbors, args.min_dist, args.k, seed,
    )

    logger.info("Projecting %d OOS docs through PUMAP encoder...", oos_X.shape[0])
    projected_oos = project_oos_via_pumap(pumap_state["model"], oos_X)

    logger.info("Calibrating filter on PUMAP reference (mahalanobis_no_correction, α=0.01)...")
    calibration = calibrate_filter(
        pumap_state["ref_embedding"], pumap_state["ref_labels"], projected_oos,
    )

    apply_result = apply_and_assign(
        projected_oos, pumap_state["ref_centroids"], calibration,
    )
    accept_rate = float(apply_result["accepted"].mean())
    logger.info(
        "Overall OOS accept rate: %.3f (%d/%d)",
        accept_rate, int(apply_result["accepted"].sum()),
        len(apply_result["accepted"]),
    )

    # Assemble the data dict in the shape write_outputs expects.
    data = {
        "ref_X": ref_X,
        "ref_Y": pumap_state["ref_embedding"],
        "ref_labels": pumap_state["ref_labels"],
        "ref_class_ids": ref_class_ids,
        "ref_class_names": REFERENCE_20NG_CLASSES,
        "oos_X": oos_X,
        "oos_class_ids": oos_class_ids,
        "oos_class_names": OOS_20NG_CLASSES,
    }
    write_outputs(out_dir, data, apply_result)
    logger.info("PUMAP bridge experiment complete; results in %s/", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
