"""NN-centroid oracle — methodologically-clean ceiling for what REAP's projection head should achieve.

For each OOS doc, we project it into the FIXED reference-only consensus
space by:

  1. Finding its k nearest neighbours among the 800 reference docs in raw
     384-d MiniLM space.
  2. Placing the OOS doc at an aggregated (mean / median / distance-weighted)
     position of those neighbours' coordinates in the reference
     consensus_embedding.npy.
  3. Applying the reference-only-calibrated conformal filter.

The reference consensus is NEVER refit on OOS — this is the methodological
correction the user flagged for the deprecated D3 union-oracle.

We sweep:
  - k ∈ {1, 3, 5, 10, 20, 50}
  - aggregation ∈ {mean, median, distance-weighted}
  - distance ∈ {euclidean, cosine}
    (MiniLM outputs are L2-normalised, so the two are equivalent up to
    monotone transform — we include both as a sanity check.)
  - seed sets ∈ {A, B, C} for between-set CI on the ceiling.

We also tune k on a 20% held-out subset of the REFERENCE (not OOS — avoid
leakage). The pre-registered optimum k is "k that maximises accept rate of
held-out reference docs projected by NN-centroid using the other 80%".

Outputs (under results/twenty_newsgroups_reference/nn_centroid_oracle/):
  - sweep_results.csv: one row per (set, k, agg, metric) with overall accept
    rate + per-OOS-class breakdown.
  - k_tuning.csv: held-out reference accept rate per (set, k, agg, metric) — the
    k-selection criterion.
  - per_class_at_best_k.csv: full per-OOS-class accept rate at the pre-registered
    best k.
  - decision_record.json: top-line ceiling + tripwire status.

Tripwire: if best NN-centroid accept rate ≥ trained-head accept rate (5.17%) + 5pp,
escalate to head architecture re-think (per Plan D verifier recommendation).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_20ng_oos_bridge import (  # type: ignore[import-not-found]
    apply_and_assign,
    calibrate_filter,
)

from reap.datasets import (  # noqa: E402
    OOS_20NG_CLASSES,
    load_twenty_newsgroups_oos,
    load_twenty_newsgroups_reference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------


K_VALUES: list[int] = [1, 3, 5, 10, 20, 50]
AGGREGATIONS: list[str] = ["mean", "median", "distance_weighted_mean"]
METRICS: list[str] = ["euclidean", "cosine"]
HELDOUT_FRAC: float = 0.20
HELDOUT_SEED: int = 42
TRAINED_HEAD_ACCEPT_RATE_FOR_TRIPWIRE: float = 0.0517  # canonical set A
TRIPWIRE_GAP_PP: float = 5.0  # percentage points


# ---------------------------------------------------------------------------
# Core NN-centroid logic
# ---------------------------------------------------------------------------


def nn_centroid_project(
    query_raw: np.ndarray,
    ref_raw: np.ndarray,
    ref_consensus: np.ndarray,
    k: int,
    aggregation: str,
    metric: str,
) -> np.ndarray:
    """Project query docs into the fixed reference consensus space via k-NN.

    Parameters
    ----------
    query_raw : (n_query, d_in) raw input embeddings to project.
    ref_raw : (n_ref, d_in) reference raw input embeddings.
    ref_consensus : (n_ref, d_out) reference consensus coords (the FIXED reference space).
    k : Number of nearest neighbours.
    aggregation : "mean" | "median" | "distance_weighted_mean".
    metric : "euclidean" | "cosine".

    Returns
    -------
    (n_query, d_out) projected coordinates in the reference consensus space.
    """
    if metric == "euclidean":
        # ||a-b||² = ||a||² + ||b||² - 2 a·b
        sq_norms_ref = np.sum(ref_raw ** 2, axis=1)[None, :]
        sq_norms_query = np.sum(query_raw ** 2, axis=1)[:, None]
        dists_sq = sq_norms_query + sq_norms_ref - 2.0 * (query_raw @ ref_raw.T)
        dists_sq = np.maximum(dists_sq, 0.0)
        dists = np.sqrt(dists_sq)
    elif metric == "cosine":
        # cosine distance = 1 - cosine_sim. Inputs are L2-normalised so
        # cosine_sim = dot product.
        dists = 1.0 - (query_raw @ ref_raw.T)
    else:
        raise ValueError(f"unknown metric: {metric}")

    k_eff = min(k, ref_raw.shape[0])
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    n_query = query_raw.shape[0]
    projected = np.zeros((n_query, ref_consensus.shape[1]), dtype=np.float64)
    for i in range(n_query):
        idx = nn_idx[i]
        neighbour_coords = ref_consensus[idx]
        if aggregation == "mean":
            projected[i] = neighbour_coords.mean(axis=0)
        elif aggregation == "median":
            projected[i] = np.median(neighbour_coords, axis=0)
        elif aggregation == "distance_weighted_mean":
            d_to_neighbours = dists[i, idx]
            # weight = 1 / (d + eps); normalised
            weights = 1.0 / (d_to_neighbours + 1e-8)
            weights = weights / weights.sum()
            projected[i] = (weights[:, None] * neighbour_coords).sum(axis=0)
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
    return projected


def assign_and_filter(
    projected: np.ndarray,
    ref_consensus: np.ndarray,
    ref_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign each projected doc to nearest reference cluster centroid + apply filter."""
    cluster_centroids: dict[int, np.ndarray] = {}
    for c in sorted({int(x) for x in ref_labels.tolist()}):
        mask = ref_labels == c
        if mask.sum() > 0:
            cluster_centroids[c] = ref_consensus[mask].mean(axis=0)
    calibration = calibrate_filter(ref_consensus, ref_labels, projected)
    return apply_and_assign(projected, cluster_centroids, calibration)


# ---------------------------------------------------------------------------
# k-tuning on held-out reference (no OOS leakage)
# ---------------------------------------------------------------------------


def tune_k_on_heldout_reference(
    ref_raw: np.ndarray,
    ref_consensus: np.ndarray,
    ref_labels: np.ndarray,
    set_name: str,
) -> list[dict[str, Any]]:
    """Hold out 20% of reference docs, project them via NN-centroid using the other 80%,
    apply filter (calibrated on the 80%), and report accept rate. The k that maximises
    accept rate is the pre-registered choice — picked WITHOUT looking at OOS data.

    Returns a list of row dicts (one per k × agg × metric × set).
    """
    rng = np.random.default_rng(HELDOUT_SEED)
    n_ref = ref_raw.shape[0]
    perm = rng.permutation(n_ref)
    n_heldout = int(n_ref * HELDOUT_FRAC)
    heldout_idx = perm[:n_heldout]
    train_idx = perm[n_heldout:]
    train_raw = ref_raw[train_idx]
    train_consensus = ref_consensus[train_idx]
    train_labels = ref_labels[train_idx]
    heldout_raw = ref_raw[heldout_idx]

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for agg in AGGREGATIONS:
            for k in K_VALUES:
                if k > train_raw.shape[0]:
                    continue
                projected = nn_centroid_project(
                    heldout_raw, train_raw, train_consensus, k, agg, metric,
                )
                result = assign_and_filter(projected, train_consensus, train_labels)
                acc_rate = float(result["accepted"].mean())
                rows.append({
                    "set": set_name,
                    "k": k,
                    "aggregation": agg,
                    "metric": metric,
                    "n_heldout": int(n_heldout),
                    "n_accepted_heldout": int(result["accepted"].sum()),
                    "heldout_accept_rate": acc_rate,
                })
    return rows


# ---------------------------------------------------------------------------
# Main sweep on OOS
# ---------------------------------------------------------------------------


def sweep_oos(
    ref_raw: np.ndarray,
    ref_consensus: np.ndarray,
    ref_labels: np.ndarray,
    oos_raw: np.ndarray,
    oos_class_ids: np.ndarray,
    set_name: str,
) -> list[dict[str, Any]]:
    """For each (k, aggregation, metric) combo: project OOS via NN-centroid, filter, record.

    Returns row dicts with overall accept rate + per-OOS-class summary.
    """
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for agg in AGGREGATIONS:
            for k in K_VALUES:
                projected = nn_centroid_project(
                    oos_raw, ref_raw, ref_consensus, k, agg, metric,
                )
                result = assign_and_filter(projected, ref_consensus, ref_labels)
                accepted = result["accepted"]
                acc_rate = float(accepted.mean())
                # Per-OOS-class breakdown
                per_class: dict[str, float] = {}
                for cid in sorted({int(c) for c in oos_class_ids.tolist()}):
                    mask = oos_class_ids == cid
                    if mask.sum() > 0:
                        per_class[str(cid)] = float(accepted[mask].mean())
                rows.append({
                    "set": set_name,
                    "k": k,
                    "aggregation": agg,
                    "metric": metric,
                    "n_oos": int(oos_raw.shape[0]),
                    "n_accepted": int(accepted.sum()),
                    "accept_rate": acc_rate,
                    "per_class_accept_rate_json": json.dumps(per_class),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NN-centroid oracle sweep.")
    parser.add_argument(
        "--sets", default="A,B,C",
        help="Comma-separated seed sets (default A,B,C).",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_root = Path(args.output_dir)
    out_dir = results_root / "twenty_newsgroups_reference" / "nn_centroid_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    # Load data once (datasets are deterministic; loaders return cached snapshots).
    logger.info("Loading reference + OOS snapshots...")
    ref_snap = load_twenty_newsgroups_reference()
    oos_snap = load_twenty_newsgroups_oos()
    ref_raw = np.asarray(ref_snap.embeddings, dtype=np.float64)
    oos_raw = np.asarray(oos_snap.embeddings, dtype=np.float64)
    oos_class_ids = np.asarray(oos_snap.labels)
    logger.info("ref_raw=%s  oos_raw=%s  oos_class_ids unique=%d",
                ref_raw.shape, oos_raw.shape, len(set(oos_class_ids.tolist())))

    all_tuning_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []

    for set_name in requested_sets:
        logger.info("=== Set %s ===", set_name)
        reap_cell = (
            results_root / "twenty_newsgroups_reference" / "reap" / f"set_{set_name}"
        )
        ref_consensus_path = reap_cell / "consensus_embedding.npy"
        ref_labels_path = reap_cell / "consensus_labels.npy"
        if not ref_consensus_path.exists() or not ref_labels_path.exists():
            logger.error("Missing REAP artifacts for set %s: %s", set_name, reap_cell)
            return 2
        ref_consensus = np.load(ref_consensus_path)
        ref_labels = np.load(ref_labels_path).astype(np.int64)

        logger.info("Set %s: k-tuning on held-out reference (20%% = %d docs)...",
                    set_name, int(ref_raw.shape[0] * HELDOUT_FRAC))
        tuning_rows = tune_k_on_heldout_reference(
            ref_raw, ref_consensus, ref_labels, set_name,
        )
        all_tuning_rows.extend(tuning_rows)

        logger.info("Set %s: OOS sweep across %d combos (k × agg × metric)...",
                    set_name,
                    len(K_VALUES) * len(AGGREGATIONS) * len(METRICS))
        sweep_rows = sweep_oos(
            ref_raw, ref_consensus, ref_labels, oos_raw, oos_class_ids, set_name,
        )
        all_sweep_rows.extend(sweep_rows)

    # Write CSVs
    tuning_path = out_dir / "k_tuning.csv"
    if all_tuning_rows:
        with tuning_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_tuning_rows[0]))
            w.writeheader()
            w.writerows(all_tuning_rows)
        logger.info("Wrote k_tuning.csv (%d rows) -> %s", len(all_tuning_rows), tuning_path)

    sweep_path = out_dir / "sweep_results.csv"
    if all_sweep_rows:
        with sweep_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_sweep_rows[0]))
            w.writeheader()
            w.writerows(all_sweep_rows)
        logger.info("Wrote sweep_results.csv (%d rows) -> %s",
                    len(all_sweep_rows), sweep_path)

    # Pre-registered best k: for each (agg, metric), pick the k that
    # MAXIMIZES heldout_accept_rate averaged across sets. Then look up the
    # OOS accept rate at that k (mean across sets).
    decision: dict[str, Any] = {
        "tuning_criterion": "max held-out reference accept rate across sets",
        "configs": [],
    }
    by_combo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in all_tuning_rows:
        by_combo.setdefault((r["aggregation"], r["metric"]), []).append(r)
    for (agg, metric), rows in by_combo.items():
        by_k: dict[int, list[float]] = {}
        for r in rows:
            by_k.setdefault(r["k"], []).append(r["heldout_accept_rate"])
        avg_per_k = {k: float(np.mean(vs)) for k, vs in by_k.items()}
        best_k = max(avg_per_k, key=avg_per_k.get)  # type: ignore[arg-type]
        # OOS accept at this best k
        oos_at_best = [
            r["accept_rate"]
            for r in all_sweep_rows
            if r["aggregation"] == agg and r["metric"] == metric and r["k"] == best_k
        ]
        decision["configs"].append({
            "aggregation": agg,
            "metric": metric,
            "tuned_best_k": int(best_k),
            "heldout_accept_at_best_k": float(avg_per_k[best_k]),
            "oos_accept_mean_across_sets": float(np.mean(oos_at_best)) if oos_at_best else None,
            "oos_accept_per_set": oos_at_best,
        })

    # Pick the single best ceiling = max OOS accept rate across all tuned configs
    best_ceiling = max(
        (c["oos_accept_mean_across_sets"] for c in decision["configs"]
         if c["oos_accept_mean_across_sets"] is not None),
        default=None,
    )
    decision["best_ceiling_oos_accept_rate"] = best_ceiling
    decision["trained_head_accept_rate"] = TRAINED_HEAD_ACCEPT_RATE_FOR_TRIPWIRE
    if best_ceiling is not None:
        gap_pp = (best_ceiling - TRAINED_HEAD_ACCEPT_RATE_FOR_TRIPWIRE) * 100.0
        decision["ceiling_minus_trained_head_pp"] = gap_pp
        decision["tripwire_gap_pp_threshold"] = TRIPWIRE_GAP_PP
        decision["tripwire_triggered"] = bool(gap_pp >= TRIPWIRE_GAP_PP)

    # Per-OOS-class accept rates at the single overall-best config (mean across sets)
    if decision["configs"]:
        best_cfg = max(
            (c for c in decision["configs"] if c["oos_accept_mean_across_sets"] is not None),
            key=lambda c: c["oos_accept_mean_across_sets"],
            default=None,
        )
        if best_cfg:
            per_class_rows: list[dict[str, Any]] = []
            for cid_str in sorted(set(
                k for r in all_sweep_rows
                if r["aggregation"] == best_cfg["aggregation"]
                and r["metric"] == best_cfg["metric"]
                and r["k"] == best_cfg["tuned_best_k"]
                for k in json.loads(r["per_class_accept_rate_json"]).keys()
            )):
                rates_across_sets = []
                for r in all_sweep_rows:
                    if (r["aggregation"] == best_cfg["aggregation"]
                            and r["metric"] == best_cfg["metric"]
                            and r["k"] == best_cfg["tuned_best_k"]):
                        per_class = json.loads(r["per_class_accept_rate_json"])
                        if cid_str in per_class:
                            rates_across_sets.append(per_class[cid_str])
                if rates_across_sets:
                    cid = int(cid_str)
                    cls_idx = cid - 100  # OOS class IDs are 100..111
                    cls_name = (
                        OOS_20NG_CLASSES[cls_idx]
                        if 0 <= cls_idx < len(OOS_20NG_CLASSES)
                        else f"class_{cid}"
                    )
                    per_class_rows.append({
                        "oos_class_id": cid,
                        "oos_class_name": cls_name,
                        "accept_rate_mean": float(np.mean(rates_across_sets)),
                        "accept_rate_std": float(np.std(rates_across_sets)),
                        "n_sets": len(rates_across_sets),
                    })
            per_class_path = out_dir / "per_class_at_best_k.csv"
            with per_class_path.open("w", newline="", encoding="utf-8") as f:
                if per_class_rows:
                    w = csv.DictWriter(f, fieldnames=list(per_class_rows[0]))
                    w.writeheader()
                    w.writerows(per_class_rows)
            logger.info("Wrote per_class_at_best_k.csv -> %s", per_class_path)

    decision_path = out_dir / "decision_record.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    logger.info("=== NN-centroid oracle summary ===\n%s",
                json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
