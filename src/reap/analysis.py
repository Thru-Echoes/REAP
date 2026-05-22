"""Cross-source and temporal analysis helpers for REAP projections.

These functions operate on already-projected coordinates (the output of a
trained projection head or a consensus embedding) and quantify how a probe
set distributes relative to a reference set. They are pure functions with
explicit inputs and outputs — no I/O, no global state.

Used by:
- ``scripts/run_ai_art_oos_demo.py`` (cross-source probe projection)
- ``scripts/run_artist_demographic_analysis.py`` (A1)
- ``scripts/run_artist_public_alignment.py`` (A2)
- ``scripts/run_temporal_baserate_check.py`` (A3)
- the temporal-holdout drivers (Phases 4–5 of the 2026-05-21 protocol amendment)

Metric conventions match the pre-registered protocol
(``manuscript/evaluation_protocol.md`` §18, §19).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 1.0) -> float:
    """KL(p || q) between two count vectors with Laplace smoothing.

    Both inputs are non-negative count arrays of equal length. ``alpha`` is
    added to every bin before normalizing, so neither distribution has a
    zero that would make the KL diverge. Uses natural log (nats).

    Side effects: none.
    """
    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {p.shape} vs {q.shape}")
    p_smoothed = p.astype(np.float64) + alpha
    q_smoothed = q.astype(np.float64) + alpha
    p_norm = p_smoothed / p_smoothed.sum()
    q_norm = q_smoothed / q_smoothed.sum()
    return float(np.sum(p_norm * (np.log(p_norm) - np.log(q_norm))))


def compute_cluster_centroids(
    Y_ref: np.ndarray, ref_assign: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Return (n_clusters, d) centroids — the mean of Y_ref grouped by cluster.

    Empty clusters get an all-zero centroid row. Side effects: none.
    """
    centroids = np.zeros((n_clusters, Y_ref.shape[1]), dtype=np.float64)
    for c in range(n_clusters):
        mask = ref_assign == c
        if mask.any():
            centroids[c] = Y_ref[mask].mean(axis=0)
    return centroids


def assign_by_nearest_centroid(Y_probe: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each probe to its nearest centroid (Euclidean) — matches KMeans.predict.

    Side effects: none.
    """
    dists = np.linalg.norm(Y_probe[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


def compute_knn_purity(
    Y_probe: np.ndarray,
    Y_ref: np.ndarray,
    probe_assign: np.ndarray,
    ref_assign: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """For each probe, fraction of its ``k`` nearest reference neighbors that
    share its assigned cluster. Returns (mean, std). Uses cosine distance.

    Side effects: none (imports sklearn lazily).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Y_ref)
    _, idx = nn.kneighbors(Y_probe)
    purities = np.array(
        [float(np.mean(ref_assign[neigh] == probe_assign[i])) for i, neigh in enumerate(idx)]
    )
    return float(purities.mean()), float(purities.std())


def compute_median_centroid_cosine(
    Y_probe: np.ndarray, centroids: np.ndarray, probe_assign: np.ndarray
) -> float:
    """Median cosine between each probe and its assigned cluster centroid.

    Side effects: none.
    """

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    cosines = np.array(
        [_cos(Y_probe[i], centroids[probe_assign[i]]) for i in range(len(Y_probe))]
    )
    return float(np.median(cosines))


def compute_contingency_cramers_v(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, float, float, float]:
    """Build an (n_rows × n_cols) contingency table + chi-square + Cramér's V + p.

    ``row_labels`` and ``col_labels`` are aligned integer arrays. All-zero
    columns are dropped before the chi-square (so scipy doesn't warn / divide
    by zero); Cramér's V uses the column-reduced shape.

    Returns ``(table, chi2_statistic, cramers_v, p_value)``. For a degenerate
    table (single non-zero cell), returns (table, 0.0, 0.0, 1.0).

    Side effects: none (imports scipy lazily).
    """
    from scipy.stats import chi2_contingency

    table = np.zeros((n_rows, n_cols), dtype=np.int64)
    for r in range(n_rows):
        for c in range(n_cols):
            table[r, c] = int(((row_labels == r) & (col_labels == c)).sum())

    nonzero_cols = table.sum(axis=0) > 0
    sub = table[:, nonzero_cols]
    if sub.size == 0 or sub.sum() == 0 or sub.shape[1] < 2 or (sub.sum(axis=1) > 0).sum() < 2:
        return table, 0.0, 0.0, 1.0

    # correction=False disables Yates' continuity correction. Yates only applies
    # to 2x2 tables and would make Cramér's V inconsistent between 2x2 and RxC
    # tables; Cramér's V as an effect size is defined on the uncorrected Pearson
    # chi-square, so we disable it for consistency across all table shapes.
    result = chi2_contingency(sub, correction=False)
    chi2_raw: Any = result[0]  # scipy returns a namedtuple; runtime is a float scalar
    p_raw: Any = result[1]
    chi2_value = float(chi2_raw)
    p_value = float(p_raw)
    n = int(sub.sum())
    r, c = sub.shape
    cramers_v = float(np.sqrt(chi2_value / (n * max(min(r - 1, c - 1), 1))))
    return table, chi2_value, cramers_v, p_value


def compute_theme_cluster_contingency(
    theme_labels: np.ndarray,
    cluster_assign: np.ndarray,
    n_clusters: int,
    n_themes: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Theme × cluster contingency table + chi-square + Cramér's V.

    Thin wrapper over `compute_contingency_cramers_v` that drops the p-value
    (the demo's schema only stores chi2 + Cramér's V). Side effects: none.
    """
    table, chi2_value, cramers_v, _ = compute_contingency_cramers_v(
        theme_labels, cluster_assign, n_themes, n_clusters
    )
    return table, chi2_value, cramers_v


def compute_source_separability_silhouette(
    Y_ref: np.ndarray, Y_probe: np.ndarray
) -> float:
    """Silhouette on {Y_ref ∪ Y_probe} with binary source labels (0=ref, 1=probe).

    A value near zero means the probe set mixes into the reference
    distribution; a high value means the probes form a separable region.
    Side effects: none (imports sklearn lazily).
    """
    from sklearn.metrics import silhouette_score

    Y_concat = np.vstack([Y_ref, Y_probe])
    src_labels = np.concatenate(
        [np.zeros(len(Y_ref), dtype=np.int64), np.ones(len(Y_probe), dtype=np.int64)]
    )
    return float(silhouette_score(Y_concat, src_labels, metric="euclidean"))


def compute_temporal_alignment(
    years: np.ndarray, cluster_assign: np.ndarray, n_clusters: int
) -> dict[str, Any]:
    """Per-cluster year distribution + an ordering of clusters by mean year.

    ``years`` and ``cluster_assign`` are aligned arrays (one per reference
    sample). Returns a dict with ``per_cluster`` (list of per-cluster stats)
    and ``ordering_by_mean_year`` (clusters sorted by mean year, empty
    clusters last). Side effects: none.
    """
    per_cluster: list[dict[str, Any]] = []
    mean_years: list[float] = []
    for c in range(n_clusters):
        mask = cluster_assign == c
        ys = years[mask]
        if len(ys) == 0:
            per_cluster.append(
                {"cluster": c, "n": 0, "mean_year": None, "q25": None, "q75": None, "histogram": {}}
            )
            mean_years.append(float("nan"))
            continue
        hist = {int(y): int(cnt) for y, cnt in zip(*np.unique(ys, return_counts=True))}
        per_cluster.append(
            {
                "cluster": c,
                "n": int(mask.sum()),
                "mean_year": float(np.mean(ys)),
                "median_year": float(np.median(ys)),
                "q25": float(np.percentile(ys, 25)),
                "q75": float(np.percentile(ys, 75)),
                "histogram": hist,
            }
        )
        mean_years.append(float(np.mean(ys)))

    finite_mask = np.isfinite(mean_years)
    finite_idx = np.where(finite_mask)[0]
    sorted_finite = finite_idx[np.argsort(np.array(mean_years)[finite_idx])]
    nan_idx = np.where(~finite_mask)[0]
    ordering = sorted_finite.tolist() + nan_idx.tolist()

    return {"per_cluster": per_cluster, "ordering_by_mean_year": [int(c) for c in ordering]}


def compute_procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Orthogonal-Procrustes disparity between two (n, d) matrices, in [0, 1].

    Uses ``scipy.spatial.procrustes`` (standardizes both inputs, returns the
    residual after the best similarity transform). Lower = more alignable.
    Side effects: none (imports scipy lazily).
    """
    from scipy.spatial import procrustes

    _, _, disparity = procrustes(A, B)
    return float(disparity)


def compute_demographic_contingency(
    cluster_assign: np.ndarray,
    demographic_labels: np.ndarray,
    n_clusters: int,
) -> dict[str, Any]:
    """Association between a demographic variable and cluster assignment.

    ``demographic_labels`` is an array of (string or int) labels aligned with
    ``cluster_assign``. Builds the (n_levels × n_clusters) contingency table,
    computes chi-square + Cramér's V + p-value.

    Returns a dict with ``levels`` (sorted unique demographic values),
    ``table`` (n_levels × n_clusters, list-of-lists), ``chi2``, ``cramers_v``,
    ``p_value``, ``n``. Side effects: none.
    """
    levels = sorted({str(x) for x in demographic_labels.tolist()})
    level_to_idx = {lv: i for i, lv in enumerate(levels)}
    row_labels = np.array(
        [level_to_idx[str(x)] for x in demographic_labels.tolist()], dtype=np.int64
    )
    table, chi2_value, cramers_v, p_value = compute_contingency_cramers_v(
        row_labels, cluster_assign, len(levels), n_clusters
    )
    return {
        "levels": levels,
        "n_levels": len(levels),
        "table": table.astype(int).tolist(),
        "chi2": chi2_value,
        "cramers_v": cramers_v,
        "p_value": p_value,
        "n": int(len(cluster_assign)),
    }


def compute_baserate_enrichment(
    probe_cluster_assign: np.ndarray,
    base_rate_per_cluster: np.ndarray,
    n_clusters: int,
) -> dict[str, Any]:
    """Per-cluster enrichment of a probe set relative to a base-rate distribution.

    ``base_rate_per_cluster`` is a length-``n_clusters`` array of the expected
    fraction of mass in each cluster under the null (e.g., reference cluster
    sizes / total). The enrichment ratio for cluster c is

        (n probes in c / n probes) / base_rate_per_cluster[c]

    A ratio > 1 means the probe set is over-represented in cluster c relative
    to the base rate; < 1 means under-represented. Clusters with zero base
    rate get ``None`` (undefined).

    Returns a dict with ``enrichment_ratio`` (length n_clusters, None where
    base rate is 0), ``probe_density``, ``base_rate``. Side effects: none.
    """
    n_probes = len(probe_cluster_assign)
    probe_counts = np.bincount(probe_cluster_assign, minlength=n_clusters)
    probe_density = probe_counts.astype(np.float64) / max(n_probes, 1)
    base = np.asarray(base_rate_per_cluster, dtype=np.float64)
    if base.shape[0] != n_clusters:
        raise ValueError(
            f"base_rate_per_cluster has length {base.shape[0]}, expected {n_clusters}"
        )
    enrichment: list[float | None] = []
    for c in range(n_clusters):
        if base[c] <= 0.0:
            enrichment.append(None)
        else:
            enrichment.append(float(probe_density[c] / base[c]))
    return {
        "enrichment_ratio": enrichment,
        "probe_density": probe_density.tolist(),
        "base_rate": base.tolist(),
    }
