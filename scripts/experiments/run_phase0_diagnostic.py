"""Phase 0 diagnostic — head-vs-filter root cause for REAP's bridge over-rejection.

Two verifier subagents independently flagged that REAP's filter being too
conservative on semantically-related OOS docs may be a *symptom* of a deeper
issue with the projection head, not a filter geometry problem. This script
runs three independent diagnostics on 20NG set A:

  D1. **Projection-head per-eigendirection CV R²** — for each consensus
      cluster, decompose its sample covariance into eigenvectors, transform
      head residuals into that eigenbasis, compute per-eigendir R². If R²
      collapses on small-eigenvalue directions, the head is biased and the
      filter is treating a symptom.

  D2. **Per-cluster Mardia multivariate skewness + kurtosis** — if REAP's
      clusters are highly non-Gaussian (e.g., filament-like), Mahalanobis is
      mis-specified regardless of regularisation. Report b₁ and b₂ per
      cluster + the Gaussian-null comparison (b₁=0, b₂=p(p+2)).

  D3. **Oracle-head baseline** — re-fit REAP consensus on the union of 800
      reference + 600 OOS = 1400 docs. The resulting consensus places OOS
      docs at TRUE positions (no projection-head error). Apply the conformal
      filter calibrated on the reference-only subset of this oracle
      consensus, and report accept rate.

  Decision rule:
    - If oracle accept rate ≈ 5% (matching trained-head baseline): filter
      geometry is intrinsic to REAP → Plan C (shrinkage / rank truncation
      / α sweep) is the right layer to fix.
    - If oracle accept rate >> 5% (e.g., 30%+): head is the root cause →
      pivot from filter fix to head-loss fix.

Outputs (in results/twenty_newsgroups_reference/phase0_diagnostic/):
  - head_r2_eigenbasis.csv
  - mardia_per_cluster.csv
  - oracle_head_summary.csv
  - decision_record.json — captures the diagnosis verdict + numbers

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_phase0_diagnostic.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Reuse the bridge experiment's helpers for the trained-head baseline + filter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_20ng_oos_bridge import (  # type: ignore[import-not-found]
    apply_and_assign,
    calibrate_filter,
    train_head,
)

from reap.datasets import (  # noqa: E402
    load_twenty_newsgroups_oos,
    load_twenty_newsgroups_reference,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1 — Projection-head per-eigendirection CV R²
# ---------------------------------------------------------------------------


def diagnostic_head_r2_eigenbasis(
    ref_X: np.ndarray,
    ref_Y: np.ndarray,
    ref_labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Per-cluster per-eigendir CV R² of REAP's projection head.

    For each cluster, decomposes the sample covariance of the consensus
    coordinates into eigenvectors. CV-predicts the head's output via 5-fold
    stratified split. Transforms residuals into the eigenbasis and reports
    R² per eigendirection.

    Returns a list of row dicts ready for CSV emission.
    """
    from sklearn.model_selection import StratifiedKFold

    rows: list[dict[str, Any]] = []

    # 5-fold CV — gather out-of-fold predictions for the whole reference set.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    Y_pred_oof = np.zeros_like(ref_Y, dtype=np.float64)
    for fold, (tr, te) in enumerate(skf.split(ref_X, ref_labels)):
        logger.info(
            "D1 head-R² fold %d/%d (train=%d, test=%d)",
            fold + 1, n_folds, len(tr), len(te),
        )
        head = train_head(ref_X[tr], ref_Y[tr], ref_labels[tr], seed=seed + fold)
        Y_pred_oof[te] = np.asarray(head.forward(ref_X[te]), dtype=np.float64)

    # Per-cluster eigenbasis analysis
    cluster_ids = sorted({int(c) for c in ref_labels.tolist()})
    for c in cluster_ids:
        mask = ref_labels == c
        n_c = int(mask.sum())
        if n_c < 5:
            continue
        Y_true_c = ref_Y[mask]
        Y_pred_c = Y_pred_oof[mask]
        mu = Y_true_c.mean(axis=0)
        Y_centered = Y_true_c - mu
        # Sample covariance (n_c, d) -> (d, d)
        cov = np.cov(Y_centered, rowvar=False, bias=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending; reorder descending so eigenvec column 0 is the
        # PRINCIPAL direction.
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Transform true + pred into eigenbasis (centered)
        U_true = (Y_centered) @ eigvecs                            # (n_c, d)
        U_pred = (Y_pred_c - mu) @ eigvecs                         # (n_c, d)
        residual = U_pred - U_true                                  # (n_c, d)

        for k in range(ref_Y.shape[1]):
            ss_res = float(np.sum(residual[:, k] ** 2))
            ss_tot = float(np.sum(U_true[:, k] ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rows.append({
                "cluster_id": c,
                "n_c": n_c,
                "eigendir_rank": k,
                "eigenvalue": _fmt(eigvals[k]),
                "variance_explained_share": _fmt(
                    eigvals[k] / float(np.sum(eigvals)) if np.sum(eigvals) > 0 else 0.0
                ),
                "cv_r2": _fmt(r2),
                "residual_std": _fmt(float(np.std(residual[:, k]))),
            })

    return rows


# ---------------------------------------------------------------------------
# D2 — Per-cluster Mardia skewness + kurtosis
# ---------------------------------------------------------------------------


def diagnostic_mardia(
    ref_Y: np.ndarray, ref_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-cluster Mardia b₁ (skewness), b₂ (kurtosis) vs Gaussian baseline.

    For a p-dim Gaussian sample of size n:
      - b₁ has approximate χ² distribution with p(p+1)(p+2)/6 df
        when scaled by n/6.
      - b₂ has approximate N(p(p+2), 8 p(p+2)/n) distribution.

    Reports b₁, b₂, the χ²/z-statistic, and a deviation flag.
    """
    p = ref_Y.shape[1]
    gaussian_b2 = p * (p + 2)
    rows: list[dict[str, Any]] = []

    cluster_ids = sorted({int(c) for c in ref_labels.tolist()})
    for c in cluster_ids:
        mask = ref_labels == c
        n_c = int(mask.sum())
        if n_c < p + 2:
            continue
        Y_c = ref_Y[mask]
        mu = Y_c.mean(axis=0)
        Y_centered = Y_c - mu
        cov = np.cov(Y_centered, rowvar=False, bias=False)
        try:
            inv_cov = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            continue

        # Mardia's b1: (1/n^2) Σᵢ Σⱼ ((xᵢ-μ)ᵀ Σ⁻¹ (xⱼ-μ))^3
        d_pairs = Y_centered @ inv_cov @ Y_centered.T  # (n_c, n_c)
        b1 = float(np.sum(d_pairs ** 3)) / (n_c * n_c)

        # Mardia's b2: (1/n) Σᵢ ((xᵢ-μ)ᵀ Σ⁻¹ (xᵢ-μ))^2
        d_self = np.diag(d_pairs)
        b2 = float(np.mean(d_self ** 2))

        # b1 normality test statistic (under H0: multivariate normal)
        b1_stat = (n_c / 6.0) * b1
        b1_df = p * (p + 1) * (p + 2) / 6

        # b2 z-statistic: (b2 - p(p+2)) / sqrt(8 p(p+2) / n)
        b2_se = math.sqrt(8 * gaussian_b2 / n_c)
        b2_z = (b2 - gaussian_b2) / b2_se if b2_se > 0 else float("nan")

        rows.append({
            "cluster_id": c,
            "n_c": n_c,
            "p_dim": p,
            "b1_skewness": _fmt(b1),
            "b1_chi2_stat": _fmt(b1_stat),
            "b1_chi2_df": int(b1_df),
            "b2_kurtosis": _fmt(b2),
            "b2_gaussian_expected": _fmt(gaussian_b2),
            "b2_z_stat": _fmt(b2_z),
            "non_gaussian_b2_flag": "1" if abs(b2_z) > 3 else "0",
        })

    return rows


# ---------------------------------------------------------------------------
# D3 — Oracle-head baseline
# ---------------------------------------------------------------------------


def diagnostic_oracle_head(
    ref_X: np.ndarray,
    oos_X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
) -> dict[str, Any]:
    """Re-fit REAP on ref+OOS union → oracle positions for OOS → filter accept rate.

    The key isolating experiment: if REAP rejects OOS docs because the head's
    projection is bad, the oracle (true) positions should produce a much
    higher accept rate. If REAP still rejects OOS at the oracle positions,
    the filter geometry is intrinsic.

    Returns a dict with: oracle_accept_rate, n_accepted, n_total, plus
    per-OOS-class breakdown.
    """
    from reap.benchmarks import run_benchmark_with_artifacts

    union_X = np.vstack([ref_X, oos_X]).astype(np.float32)
    n_ref = ref_X.shape[0]
    n_oos = oos_X.shape[0]

    logger.info(
        "D3 oracle: running REAP on %d-doc union (ref=%d + oos=%d)...",
        n_ref + n_oos, n_ref, n_oos,
    )
    _, artifacts = run_benchmark_with_artifacts(
        X=union_X,
        dataset_name="20NG-union-oracle",
        seeds=seeds,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        k_range=k_range,
        methods=["reap"],
    )
    reap_arts = artifacts.per_method["reap"]
    union_consensus_emb: np.ndarray = np.asarray(reap_arts["consensus_embedding"])
    union_consensus_labels: np.ndarray = np.asarray(reap_arts["consensus_labels"])

    # Split the consensus into reference and oos subsets (oracle positions).
    ref_oracle = union_consensus_emb[:n_ref]
    oos_oracle = union_consensus_emb[n_ref:]
    ref_oracle_labels = union_consensus_labels[:n_ref]

    # Apply filter calibrated on reference oracle. Assign each oracle OOS doc
    # to nearest reference cluster centroid (same scheme as the bridge script).
    cluster_centroids: dict[int, np.ndarray] = {}
    for c in sorted({int(x) for x in ref_oracle_labels.tolist()}):
        mask = ref_oracle_labels == c
        if mask.sum() > 0:
            cluster_centroids[c] = ref_oracle[mask].mean(axis=0)

    calibration = calibrate_filter(ref_oracle, ref_oracle_labels, oos_oracle)
    apply_result = apply_and_assign(oos_oracle, cluster_centroids, calibration)

    accept_rate = float(apply_result["accepted"].mean())
    return {
        "oracle_accept_count": int(apply_result["accepted"].sum()),
        "oracle_accept_rate": accept_rate,
        "n_oos": n_oos,
        "n_ref_clusters": len(cluster_centroids),
        "ref_oracle_label_counts": {
            int(c): int((ref_oracle_labels == c).sum())
            for c in sorted({int(x) for x in ref_oracle_labels.tolist()})
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{float(v):.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 diagnostic.")
    parser.add_argument("--set", default="A", choices=["A", "B", "C"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--skip-d1", action="store_true",
        help="Skip projection-head CV R² diagnostic (saves ~5 min).",
    )
    parser.add_argument(
        "--skip-d3", action="store_true",
        help="Skip oracle-head baseline (saves ~10-15 min compute).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_root = Path(args.output_dir)
    out_dir = (
        results_root / "twenty_newsgroups_reference"
        / "phase0_diagnostic" / f"set_{args.set}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference + OOS data + REAP set A artifacts
    logger.info("Loading reference + OOS snapshots + REAP set %s artifacts...", args.set)
    ref_snap = load_twenty_newsgroups_reference()
    oos_snap = load_twenty_newsgroups_oos()
    ref_X = np.asarray(ref_snap.embeddings, dtype=np.float32)
    oos_X = np.asarray(oos_snap.embeddings, dtype=np.float32)

    reap_cell = (
        results_root / "twenty_newsgroups_reference" / "reap" / f"set_{args.set}"
    )
    ref_Y = np.load(reap_cell / "consensus_embedding.npy")
    ref_labels = np.load(reap_cell / "consensus_labels.npy").astype(np.int64)
    logger.info(
        "Loaded: ref X=%s Y=%s labels=%s; oos X=%s",
        ref_X.shape, ref_Y.shape, ref_labels.shape, oos_X.shape,
    )

    decision: dict[str, Any] = {"set": args.set}

    # D1 — head R² in eigenbasis
    if not args.skip_d1:
        logger.info("=== D1: projection-head per-eigendir CV R² ===")
        r2_rows = diagnostic_head_r2_eigenbasis(ref_X, ref_Y, ref_labels)
        d1_path = out_dir / "head_r2_eigenbasis.csv"
        with d1_path.open("w", newline="", encoding="utf-8") as f:
            if r2_rows:
                w = csv.DictWriter(f, fieldnames=list(r2_rows[0]))
                w.writeheader()
                w.writerows(r2_rows)
        logger.info("D1 wrote %d rows -> %s", len(r2_rows), d1_path)

        # Aggregate verdict: did R² collapse on small eigenvalues?
        small_eig_r2: list[float] = []
        large_eig_r2: list[float] = []
        for row in r2_rows:
            share = float(row["variance_explained_share"])
            r2 = row["cv_r2"]
            if not r2:
                continue
            r2f = float(r2)
            if share < 0.05:
                small_eig_r2.append(r2f)
            elif share > 0.20:
                large_eig_r2.append(r2f)
        decision["d1_small_eig_mean_r2"] = (
            float(np.mean(small_eig_r2)) if small_eig_r2 else None
        )
        decision["d1_large_eig_mean_r2"] = (
            float(np.mean(large_eig_r2)) if large_eig_r2 else None
        )

    # D2 — Mardia
    logger.info("=== D2: per-cluster Mardia skewness + kurtosis ===")
    mardia_rows = diagnostic_mardia(ref_Y, ref_labels)
    d2_path = out_dir / "mardia_per_cluster.csv"
    with d2_path.open("w", newline="", encoding="utf-8") as f:
        if mardia_rows:
            w = csv.DictWriter(f, fieldnames=list(mardia_rows[0]))
            w.writeheader()
            w.writerows(mardia_rows)
    logger.info("D2 wrote %d rows -> %s", len(mardia_rows), d2_path)

    # Aggregate D2 verdict
    non_gauss_flags = sum(1 for r in mardia_rows if r["non_gaussian_b2_flag"] == "1")
    decision["d2_non_gaussian_clusters"] = non_gauss_flags
    decision["d2_total_clusters"] = len(mardia_rows)

    # D3 — oracle-head baseline
    if not args.skip_d3:
        logger.info("=== D3: oracle-head baseline (REAP on ref+OOS union) ===")
        # Load seed manifest for set A
        manifest_path = (
            Path(__file__).resolve().parent.parent.parent
            / "manuscript" / "seeds" / "seed_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        seeds = manifest["sets"][args.set]["seeds"]
        oracle = diagnostic_oracle_head(
            ref_X, oos_X, seeds,
            n_components=5, n_neighbors=15, min_dist=0.1,
            k_range=list(range(5, 21)),
        )
        d3_path = out_dir / "oracle_head_summary.csv"
        with d3_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            for k, v in oracle.items():
                w.writerow([k, json.dumps(v)])
        logger.info("D3 wrote -> %s", d3_path)
        decision["d3_oracle_accept_rate"] = oracle["oracle_accept_rate"]
        decision["d3_oracle_accept_count"] = oracle["oracle_accept_count"]
        decision["d3_n_oos"] = oracle["n_oos"]

    # Decision-rule summary
    decision["trained_head_accept_rate_for_comparison"] = 0.0517  # 31/600, set A canonical
    if "d3_oracle_accept_rate" in decision:
        gap = decision["d3_oracle_accept_rate"] - decision["trained_head_accept_rate_for_comparison"]
        decision["d3_oracle_vs_trained_gap"] = gap
        if gap > 0.20:
            decision["verdict"] = "HEAD_IS_ROOT_CAUSE — pivot to head-loss fix"
        elif gap > 0.05:
            decision["verdict"] = "HEAD_CONTRIBUTES — fix both head + filter"
        else:
            decision["verdict"] = "FILTER_IS_ROOT_CAUSE — proceed with Plan C"

    decision_path = out_dir / "decision_record.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    logger.info("=== Verdict ===\n%s", json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
