"""20NG out-of-sample bridge-worlds experiment — the reframed manuscript's headline.

REAP's stated purpose is to build a stable reference semantic space and project
foreign data into it. This script is the most direct empirical test:

    1. Take REAP's consensus on the 8-class 20NG reference (Set A by default).
    2. Train REAP's projection head on
       ``(raw MiniLM embeddings, REAP consensus embedding)`` of the reference.
    3. Calibrate REAP's conformal filter on the reference (LOO Mahalanobis,
       ``mahalanobis_no_correction`` — no peeking at OOS during calibration).
    4. Project the 600-document **OOS** snapshot (12 held-out classes) through
       the trained projection head.
    5. Apply the conformal filter to the projected OOS coordinates.
    6. For accepted OOS docs, assign to the nearest reference cluster centroid.
    7. Build three output tables:

       - ``per_oos_doc.csv``: one row per OOS doc with accept/reject + alpha
         score + assigned reference cluster + distance.
       - ``per_oos_class_summary.csv``: per held-out class, accept rate +
         dominant reference cluster + 95% CI on accept rate.
       - ``cluster_class_matrix.csv``: rows = reference clusters, columns =
         OOS classes, values = count of accepted OOS docs landing there.

Why this test matters
---------------------
- (Q1) Does the conformal filter correctly flag OOD documents?
       Expected: high reject rate on truly OOS classes that have no reference
       analog (e.g., ``misc.forsale``).
- (Q2) For accepted OOS docs, do they map to semantically-related reference
       clusters? Expected: ``rec.sport.baseball`` → near ``rec.sport.hockey``;
       ``talk.religion.misc`` → near ``soc.religion.christian``.

Usage
-----
::

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_20ng_oos_bridge.py \
        --set A --method reap

Side effects
------------
- Loads REAP's consensus artifacts from
  ``results/twenty_newsgroups_reference/reap/set_<X>/``.
- Loads reference + OOS DatasetSnapshots from the REAP cache.
- Trains a torch projection head on CPU (~1-2 min, 5-fold CV).
- Writes three CSVs under
  ``results/twenty_newsgroups_reference/oos_bridge/set_<X>/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from reap.datasets import (  # noqa: E402
    OOS_20NG_CLASSES,
    REFERENCE_20NG_CLASSES,
    load_twenty_newsgroups_oos,
    load_twenty_newsgroups_reference,
)
from reap.filter import apply as filter_apply  # noqa: E402
from reap.filter import calibrate as filter_calibrate  # noqa: E402
from reap.filter import compute_mahalanobis_d2  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_reference_and_oos(
    set_name: str,
    method: str,
    results_root: Path,
) -> dict[str, Any]:
    """Load REAP consensus artifacts + reference + OOS dataset snapshots.

    Returns a dict with:
      - ``ref_X`` (800, 384) raw MiniLM embeddings of the reference docs
      - ``ref_Y`` (800, d_out) REAP consensus embedding (the *target* space)
      - ``ref_labels`` (800,) REAP's cluster assignments
      - ``ref_class_names`` list of 8 reference class names (ground-truth labels)
      - ``ref_class_ids`` (800,) the 0..7 ground-truth class ids
      - ``oos_X`` (600, 384) raw MiniLM embeddings of OOS docs
      - ``oos_class_ids`` (600,) ground-truth class ids (100..111 by convention)
      - ``oos_class_names`` list of 12 OOS class names
    """
    cell_dir = results_root / "twenty_newsgroups_reference" / method / f"set_{set_name}"
    cons_emb_path = cell_dir / "consensus_embedding.npy"
    cons_lbl_path = cell_dir / "consensus_labels.npy"
    if not cons_emb_path.exists() or not cons_lbl_path.exists():
        raise FileNotFoundError(
            f"missing REAP artifacts at {cell_dir} — "
            "did you run scripts/run_paper_benchmark.py --dataset twenty_newsgroups_reference?"
        )

    ref_snap = load_twenty_newsgroups_reference()
    oos_snap = load_twenty_newsgroups_oos()

    return {
        "ref_X": np.asarray(ref_snap.embeddings),
        "ref_Y": np.load(cons_emb_path),
        "ref_labels": np.load(cons_lbl_path).astype(np.int64),
        "ref_class_ids": np.asarray(ref_snap.labels),
        "ref_class_names": REFERENCE_20NG_CLASSES,
        "oos_X": np.asarray(oos_snap.embeddings),
        "oos_class_ids": np.asarray(oos_snap.labels),
        "oos_class_names": OOS_20NG_CLASSES,
    }


def train_head(
    ref_X: np.ndarray, ref_Y: np.ndarray, ref_labels: np.ndarray,
    seed: int = 42,
) -> Any:
    """Train REAP's projection head on (raw input, consensus target).

    Wraps :func:`reap.projection.train_projection_head` with the manuscript-
    pre-registered defaults (5-fold CV, 500 max epochs with early stopping).

    Seeds torch + numpy + Python RNGs before training so the projection head's
    weight initialisation and dropout-mask sequence are reproducible across
    runs. Required by publication-standards.md "Randomness is controlled".
    """
    import os
    import random

    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # MPS / some CPU ops fall back; OK at this seed

    from reap.projection import train_projection_head

    head_result = train_projection_head(
        X=ref_X,
        Y=ref_Y,
        labels=ref_labels,
        seed=None,  # defer to the external np.random.seed + torch.manual_seed above
    )
    final = head_result.get("final_metrics", {})
    cv = head_result.get("cv_metrics", [])
    cv_r2_vals: list[float] = []
    for m in cv:
        if isinstance(m, dict):
            r2 = m.get("r2")
            if isinstance(r2, (int, float)):
                cv_r2_vals.append(float(r2))
    if cv_r2_vals:
        logger.info(
            "Projection head: final R²=%.4f, CV R² mean=%.4f (folds=%s)",
            final.get("r2", float("nan")),
            float(sum(cv_r2_vals) / len(cv_r2_vals)),
            [f"{v:.4f}" for v in cv_r2_vals],
        )
    return head_result["model"]


def project_oos(head: Any, oos_X: np.ndarray) -> np.ndarray:
    """Apply trained projection head to OOS embeddings.

    Returns ``(n_oos, d_out)`` projected coordinates. ``ProjectionHead.forward``
    handles eval mode + ``no_grad`` internally and returns numpy.
    """
    return np.asarray(head.forward(oos_X), dtype=np.float64)


def calibrate_filter(
    ref_Y: np.ndarray, ref_labels: np.ndarray, projected_oos: np.ndarray,
) -> Any:
    """Calibrate the conformal filter using LOO Mahalanobis on the reference only.

    Uses ``mahalanobis_no_correction`` — does not peek at the OOS distribution
    during calibration. The ``oos_coords/oos_labels`` arguments are passed for
    shape-validation purposes (the method ignores their values).
    """
    dummy_oos_labels = np.zeros(projected_oos.shape[0], dtype=np.int64)
    return filter_calibrate(
        reference_coords=ref_Y,
        reference_labels=ref_labels,
        oos_coords=projected_oos,
        oos_labels=dummy_oos_labels,
        alpha=0.01,
        method="mahalanobis_no_correction",
    )


def apply_and_assign(
    projected_oos: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    calibration: Any,
) -> dict[str, np.ndarray]:
    """Apply the conformal filter; for each OOS doc emit the (alpha, assigned_cluster).

    Assignment rule: for *every* OOS doc, find the nearest reference cluster
    centroid in the consensus space (Euclidean). The "accepted" flag is taken
    from the filter; we still record the nearest cluster for rejected docs so
    we can inspect what they would have been assigned to.
    """
    # Assign every OOS doc to its nearest reference cluster centroid.
    cluster_ids = sorted(cluster_centroids)
    centroid_matrix = np.stack([cluster_centroids[c] for c in cluster_ids])
    diffs = projected_oos[:, None, :] - centroid_matrix[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    assigned_idx = np.argmin(dists, axis=1)
    assigned_cluster = np.asarray(
        [cluster_ids[int(i)] for i in assigned_idx], dtype=np.int64
    )
    distance_to_assigned = dists[np.arange(len(projected_oos)), assigned_idx]

    # Filter accept/reject + Mahalanobis D² for each doc. We feed each doc with
    # its nearest-centroid cluster id as the test_labels; this is how a
    # downstream user would actually deploy the filter on truly unlabeled data.
    accepted = filter_apply(
        test_coords=projected_oos,
        test_labels=assigned_cluster,
        calibration=calibration,
    )
    d2 = compute_mahalanobis_d2(
        test_coords=projected_oos,
        test_labels=assigned_cluster,
        calibration=calibration,
    )

    return {
        "assigned_cluster": assigned_cluster,
        "distance_to_assigned": distance_to_assigned,
        "accepted": np.asarray(accepted, dtype=bool),
        "alpha_score": np.asarray(d2, dtype=np.float64),
    }


def write_outputs(
    out_dir: Path,
    data: dict[str, Any],
    apply_result: dict[str, np.ndarray],
) -> None:
    """Write per-doc table, per-OOS-class summary, and cluster-class confusion matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    oos_ids = data["oos_class_ids"]
    oos_class_names = data["oos_class_names"]
    ref_class_names = data["ref_class_names"]
    ref_class_ids = data["ref_class_ids"]
    ref_labels = data["ref_labels"]
    assigned = apply_result["assigned_cluster"]
    accepted = apply_result["accepted"]
    alpha_scores = apply_result["alpha_score"]
    distances = apply_result["distance_to_assigned"]

    # OOS class id (label offset 100) → display name. The loader uses ids
    # 100..111 for the 12 OOS classes.
    oos_id_to_name = {
        100 + i: name for i, name in enumerate(oos_class_names)
    }

    # 1) per_oos_doc.csv
    per_doc_path = out_dir / "per_oos_doc.csv"
    with per_doc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "doc_idx", "oos_class_id", "oos_class_name",
            "assigned_ref_cluster", "distance_to_centroid",
            "accepted", "alpha_score",
        ])
        for i in range(len(oos_ids)):
            cls_id = int(oos_ids[i])
            w.writerow([
                i, cls_id, oos_id_to_name.get(cls_id, "?"),
                int(assigned[i]), f"{float(distances[i]):.6f}",
                "1" if accepted[i] else "0",
                f"{float(alpha_scores[i]):.6f}",
            ])

    # 2) per_oos_class_summary.csv
    summary_path = out_dir / "per_oos_class_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "oos_class_id", "oos_class_name", "n_docs",
            "n_accepted", "accept_rate",
            "dominant_ref_cluster", "dominant_ref_cluster_share",
            "dominant_ref_cluster_name",
        ])
        unique_oos = sorted({int(i) for i in oos_ids.tolist()})
        # Build a map cluster_id -> majority ref class name (for context)
        ref_cluster_to_name: dict[int, str] = {}
        for c in sorted({int(x) for x in ref_labels.tolist()}):
            members = ref_class_ids[ref_labels == c]
            if len(members) == 0:
                ref_cluster_to_name[c] = "?"
            else:
                vals, counts = np.unique(members, return_counts=True)
                ref_cluster_to_name[c] = ref_class_names[int(vals[int(np.argmax(counts))])]
        for cls_id in unique_oos:
            mask = oos_ids == cls_id
            n = int(mask.sum())
            if n == 0:
                continue
            n_acc = int(accepted[mask].sum())
            assignments = assigned[mask]
            vals, counts = np.unique(assignments, return_counts=True)
            dom_idx = int(np.argmax(counts))
            dom_cluster = int(vals[dom_idx])
            dom_share = float(counts[dom_idx]) / n
            w.writerow([
                cls_id, oos_id_to_name.get(cls_id, "?"), n,
                n_acc, f"{n_acc / n:.4f}" if n > 0 else "",
                dom_cluster, f"{dom_share:.4f}",
                ref_cluster_to_name.get(dom_cluster, "?"),
            ])

    # 3) cluster_class_matrix.csv — rows = reference clusters (one per row),
    # columns = OOS classes (one per col). Cell value = count of *accepted*
    # OOS docs of that class assigned to that cluster.
    matrix_path = out_dir / "cluster_class_matrix.csv"
    ref_clusters = sorted({int(x) for x in ref_labels.tolist()})
    oos_classes = sorted({int(x) for x in oos_ids.tolist()})
    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["ref_cluster", "ref_cluster_majority_class"]
        for cls_id in oos_classes:
            header.append(oos_id_to_name.get(cls_id, f"class_{cls_id}"))
        w.writerow(header)
        for c in ref_clusters:
            row: list[str] = [str(c), ref_cluster_to_name.get(c, "?")]
            for cls_id in oos_classes:
                mask = (oos_ids == cls_id) & accepted & (assigned == c)
                row.append(str(int(mask.sum())))
            w.writerow(row)

    logger.info("wrote per_oos_doc.csv (%d rows) -> %s", len(oos_ids), per_doc_path)
    logger.info("wrote per_oos_class_summary.csv -> %s", summary_path)
    logger.info("wrote cluster_class_matrix.csv -> %s", matrix_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="REAP OOS bridge-worlds experiment on 20-Newsgroups.",
    )
    parser.add_argument(
        "--set", default="A", choices=["A", "B", "C"],
        help="Which seed set's REAP artifacts to use (default A).",
    )
    parser.add_argument(
        "--method", default="reap",
        help=(
            "Which method's consensus artifacts to use as the reference. "
            "Default 'reap'. Other methods supported: best_of_n, naive_average, "
            "procrustes, bertopic, parametric_umap."
        ),
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

    results_root = Path(args.output_dir)
    out_dir = (
        results_root / "twenty_newsgroups_reference" / "oos_bridge"
        / f"{args.method}_set_{args.set}"
    )

    logger.info("Loading reference + OOS data + %s artifacts (set %s)", args.method, args.set)
    data = load_reference_and_oos(args.set, args.method, results_root)
    logger.info(
        "Reference: n=%d d_in=%d d_out=%d k=%d  |  OOS: n=%d d_in=%d classes=%d",
        data["ref_X"].shape[0], data["ref_X"].shape[1], data["ref_Y"].shape[1],
        len(set(data["ref_labels"].tolist())),
        data["oos_X"].shape[0], data["oos_X"].shape[1],
        len(set(data["oos_class_ids"].tolist())),
    )

    logger.info("Training projection head (5-fold CV)...")
    head = train_head(data["ref_X"], data["ref_Y"], data["ref_labels"])

    logger.info("Projecting %d OOS docs through the head...", data["oos_X"].shape[0])
    projected_oos = project_oos(head, data["oos_X"])

    logger.info("Calibrating conformal filter on reference (mahalanobis_no_correction, alpha=0.01)...")
    calibration = calibrate_filter(data["ref_Y"], data["ref_labels"], projected_oos)

    # Build per-cluster centroids in the consensus space for assignment.
    cluster_centroids: dict[int, np.ndarray] = {}
    for c in sorted({int(x) for x in data["ref_labels"].tolist()}):
        mask = data["ref_labels"] == c
        if mask.sum() > 0:
            cluster_centroids[c] = data["ref_Y"][mask].mean(axis=0)

    apply_result = apply_and_assign(projected_oos, cluster_centroids, calibration)

    accept_rate = float(apply_result["accepted"].mean())
    logger.info("Overall OOS accept rate: %.3f (%d/%d)",
                accept_rate, int(apply_result["accepted"].sum()),
                len(apply_result["accepted"]))

    write_outputs(out_dir, data, apply_result)
    logger.info("OOS bridge experiment complete; results in %s/", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
