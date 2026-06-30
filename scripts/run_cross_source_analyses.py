"""Cross-source analyses on the existing AI-art OOS demo projections (§19).

Runs three pre-registered analyses (no new consensus or head training):

- **A1** — Artist demographic stratification: for each demographic variable
  in the Lovato 2024 survey, the association between demographic and the
  reference cluster the artist's perspective projects into.
- **A2** — Artist-vs-public theme alignment: per-theme cluster-occupancy
  divergence (KL + Wasserstein-1) between the two probe sources.
- **A3** — Base-rate-corrected temporal alignment: per-cluster enrichment
  of artist projections relative to the reference cluster sizes, cross-
  referenced with each cluster's mean year.

Inputs are the canonical set-A projections from
``results/projection_head/ai_art/oos_demo/set_A/`` and the demographic /
theme metadata from the WAMA project + the REAP cache.

Outputs land under
``results/projection_head/ai_art/oos_demo/cross_source/``.

Pre-registered in
``manuscript/proposals/2026-05-21-temporal-holdout-and-cross-source-protocol.md``
(§19) and ``manuscript/evaluation_protocol.md`` §19.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from reap.analysis import (
    compute_baserate_enrichment,
    compute_demographic_contingency,
    compute_kl_divergence,
)
from reap.datasets.ai_art_oos import THEMES, _get_ai_art_year_array

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
WAMA_SOURCE_PATH = Path.home() / "Documents/Berkeley/Research/When-Algorithms-Meet-Artists"
OOS_DEMO_ROOT = RESULTS_ROOT / "projection_head" / "ai_art" / "oos_demo"
CANONICAL_SET = "A"

# Country is constant (all USA) → degenerate contingency, excluded from A1.
DEMOGRAPHIC_COLUMNS = [
    "Art_practice",
    "Age",
    "POC",
    "Gender_identity",
    "AI_models_familiarity",
    "Used_AI_art_models",
    "Professional_artist",
    "Purchase_art",
    "compensation",
]
HOLM_ALPHA = 0.05


def _holm_bonferroni(pvalues: list[float], alpha: float = HOLM_ALPHA) -> list[bool]:
    """Return per-test reject decisions under Holm-Bonferroni at level alpha."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break  # Holm stops at the first non-rejection
    return reject


def _load_artist_demographics() -> dict[str, np.ndarray]:
    """Load the 1,259 artist demographic columns aligned with the projections."""
    import pandas as pd

    df = pd.read_csv(WAMA_SOURCE_PATH / "data/artist_perspectives.csv")
    if len(df) != 1259:
        raise RuntimeError(f"expected 1259 artist rows, got {len(df)}")
    out: dict[str, np.ndarray] = {}
    for col in DEMOGRAPHIC_COLUMNS:
        # Cast to str so NaN becomes "nan" and trailing-space variants stay distinct.
        out[col] = df[col].astype(str).str.strip().to_numpy(dtype=object)
    return out


def run_a1_demographic_stratification(
    artist_assign: np.ndarray, n_clusters: int
) -> dict[str, Any]:
    """A1: demographic → cluster association for each demographic variable."""
    demographics = _load_artist_demographics()
    per_demo: dict[str, Any] = {}
    pvalues: list[float] = []
    demo_order: list[str] = []
    for col, labels in demographics.items():
        result = compute_demographic_contingency(artist_assign, labels, n_clusters)
        per_demo[col] = result
        pvalues.append(result["p_value"])
        demo_order.append(col)

    reject = _holm_bonferroni(pvalues)
    for col, rej in zip(demo_order, reject):
        per_demo[col]["holm_significant"] = bool(rej)

    significant = [c for c, r in zip(demo_order, reject) if r]
    return {
        "analysis": "A1_demographic_stratification",
        "canonical_set": CANONICAL_SET,
        "n_clusters": n_clusters,
        "n_artists": int(len(artist_assign)),
        "holm_alpha": HOLM_ALPHA,
        "demographics": per_demo,
        "significant_demographics": significant,
        "summary": {
            col: {
                "cramers_v": round(per_demo[col]["cramers_v"], 4),
                "p_value": per_demo[col]["p_value"],
                "holm_significant": per_demo[col]["holm_significant"],
                "n_levels": per_demo[col]["n_levels"],
            }
            for col in demo_order
        },
    }


def run_a2_artist_vs_public_alignment(
    artist_assign: np.ndarray,
    artist_themes: np.ndarray,
    public_assign: np.ndarray,
    public_themes: np.ndarray,
    n_clusters: int,
) -> dict[str, Any]:
    """A2: per-theme cluster-occupancy divergence between artist and public probes."""
    from scipy.stats import wasserstein_distance

    per_theme: dict[str, Any] = {}
    for t_idx, theme in enumerate(THEMES):
        art_hist = np.bincount(artist_assign[artist_themes == t_idx], minlength=n_clusters)
        pub_hist = np.bincount(public_assign[public_themes == t_idx], minlength=n_clusters)
        kl = compute_kl_divergence(art_hist, pub_hist)
        # Wasserstein over the cluster-index axis (weak ordinal axis; comparable scalar).
        cluster_axis = np.arange(n_clusters)
        art_w = art_hist / max(art_hist.sum(), 1)
        pub_w = pub_hist / max(pub_hist.sum(), 1)
        wass = float(
            wasserstein_distance(cluster_axis, cluster_axis, u_weights=art_w, v_weights=pub_w)
        )
        per_theme[theme] = {
            "artist_occupancy": art_hist.astype(int).tolist(),
            "public_occupancy": pub_hist.astype(int).tolist(),
            "n_artist": int(art_hist.sum()),
            "n_public": int(pub_hist.sum()),
            "kl_artist_to_public": kl,
            "wasserstein_cluster_axis": wass,
        }

    kl_by_theme = {th: per_theme[th]["kl_artist_to_public"] for th in THEMES}
    most_divergent = max(kl_by_theme, key=lambda k: kl_by_theme[k])
    least_divergent = min(kl_by_theme, key=lambda k: kl_by_theme[k])
    return {
        "analysis": "A2_artist_vs_public_alignment",
        "canonical_set": CANONICAL_SET,
        "n_clusters": n_clusters,
        "per_theme": per_theme,
        "most_divergent_theme": most_divergent,
        "least_divergent_theme": least_divergent,
        "caveat": (
            "Public probes are theme-conditioned subsequence selections of the "
            "reference (round-2 audit). This compares artist stated concerns to "
            "style-controlled discourse passages selected to be theme-relevant, "
            "not two independent corpora."
        ),
    }


def run_a3_baserate_temporal(
    artist_assign: np.ndarray,
    ref_labels: np.ndarray,
    years: np.ndarray,
    n_clusters: int,
) -> dict[str, Any]:
    """A3: per-cluster artist enrichment relative to reference cluster sizes,
    cross-referenced with cluster mean year."""
    ref_counts = np.bincount(ref_labels, minlength=n_clusters)
    base_rate = ref_counts.astype(np.float64) / ref_counts.sum()
    enrichment = compute_baserate_enrichment(artist_assign, base_rate, n_clusters)

    # Per-cluster mean year (reference).
    cluster_mean_year: list[float | None] = []
    for c in range(n_clusters):
        ys = years[ref_labels == c]
        cluster_mean_year.append(float(np.mean(ys)) if len(ys) else None)

    # Which clusters are artist-enriched (ratio > 1), and what are their mean years?
    enriched = [
        {
            "cluster": c,
            "enrichment": enrichment["enrichment_ratio"][c],
            "mean_year": cluster_mean_year[c],
        }
        for c in range(n_clusters)
        if enrichment["enrichment_ratio"][c] is not None
        and enrichment["enrichment_ratio"][c] > 1.0
    ]
    enriched.sort(key=lambda d: -d["enrichment"])

    # Of the enriched clusters, how many have mean year in 2022-2024?
    enriched_recent = [
        e for e in enriched if e["mean_year"] is not None and 2022 <= e["mean_year"] <= 2024
    ]

    return {
        "analysis": "A3_baserate_temporal",
        "canonical_set": CANONICAL_SET,
        "n_clusters": n_clusters,
        "reference_cluster_sizes": ref_counts.astype(int).tolist(),
        "base_rate": base_rate.tolist(),
        "enrichment_ratio": enrichment["enrichment_ratio"],
        "cluster_mean_year": cluster_mean_year,
        "artist_enriched_clusters": enriched,
        "n_enriched_total": len(enriched),
        "n_enriched_in_2022_2024": len(enriched_recent),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    set_dir = OOS_DEMO_ROOT / f"set_{CANONICAL_SET}"
    artist_assign = np.load(set_dir / "assignment_artist.npy")
    public_assign = np.load(set_dir / "assignment_public.npy")
    n_clusters = int(max(artist_assign.max(), public_assign.max())) + 1

    cache = Path.home() / ".cache/reap/datasets"
    artist_themes = np.load(cache / "ai_art_oos_artist/2026-05-21/labels.npy")
    public_themes = np.load(cache / "ai_art_oos_public/2026-05-21/labels.npy")

    ref_labels = np.load(
        RESULTS_ROOT / "ai_art/reap" / f"set_{CANONICAL_SET}" / "consensus_labels.npy"
    )
    years = _get_ai_art_year_array(WAMA_SOURCE_PATH)
    # n_clusters from the reference consensus (authoritative).
    n_clusters = int(ref_labels.max()) + 1

    out_dir = OOS_DEMO_ROOT / "cross_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    a1 = run_a1_demographic_stratification(artist_assign, n_clusters)
    a2 = run_a2_artist_vs_public_alignment(
        artist_assign, artist_themes, public_assign, public_themes, n_clusters
    )
    a3 = run_a3_baserate_temporal(artist_assign, ref_labels, years, n_clusters)

    for name, payload in [
        ("A1_demographic_stratification", a1),
        ("A2_artist_vs_public_alignment", a2),
        ("A3_baserate_temporal", a3),
    ]:
        path = out_dir / f"{name}.json"
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2)
        logger.info("wrote %s", path)

    # Console summary.
    logger.info("=== A1 demographic stratification (Holm alpha=%.2f) ===", HOLM_ALPHA)
    for col, s in a1["summary"].items():
        flag = "***" if s["holm_significant"] else "   "
        logger.info(
            "  %s %-22s cramers_v=%.3f p=%.2e levels=%d",
            flag, col, s["cramers_v"], s["p_value"], s["n_levels"],
        )
    logger.info("  significant: %s", a1["significant_demographics"])

    logger.info("=== A2 artist-vs-public theme alignment ===")
    for theme in THEMES:
        b = a2["per_theme"][theme]
        logger.info(
            "  %-13s KL=%.3f wasserstein=%.3f (n_art=%d n_pub=%d)",
            theme, b["kl_artist_to_public"], b["wasserstein_cluster_axis"],
            b["n_artist"], b["n_public"],
        )
    logger.info(
        "  most divergent: %s | least: %s",
        a2["most_divergent_theme"], a2["least_divergent_theme"],
    )

    logger.info("=== A3 base-rate-corrected temporal ===")
    logger.info(
        "  %d artist-enriched clusters; %d of them have mean year 2022-2024",
        a3["n_enriched_total"], a3["n_enriched_in_2022_2024"],
    )
    for e in a3["artist_enriched_clusters"][:6]:
        logger.info(
            "    cluster %2d: enrichment=%.2fx mean_year=%s",
            e["cluster"], e["enrichment"],
            f"{e['mean_year']:.1f}" if e["mean_year"] is not None else "n/a",
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
