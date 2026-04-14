"""Benchmark comparison of consensus methods for REAP.

Compares five strategies for combining multi-seed UMAP embeddings:
1. single_seed   — independent seeds, report mean +/- std (instability baseline)
2. best_of_n     — cherry-pick seed with highest silhouette (selection bias baseline)
3. naive_average — average coordinates without alignment (negative control)
4. procrustes    — Procrustes-align then average coordinates (existing baseline)
5. reap          — average pairwise distance matrices, then project (our method)

All methods share the SAME multi-seed UMAP embeddings, computed once. The
difference is only how they combine (or select from) those embeddings. This
ensures fair comparison and saves compute.

Side effects: UMAP + KMeans internal random state, logging.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from reap.clustering import find_best_k
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_consensus,
)
from reap.evaluation import compute_trustworthiness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS: list[str] = [
    "single_seed",
    "best_of_n",
    "naive_average",
    "procrustes",
    "reap",
]

_METHOD_DESCRIPTIONS: dict[str, str] = {
    "single_seed": (
        "Independent single-seed UMAP runs; reports mean and std across seeds"
    ),
    "best_of_n": (
        "Cherry-pick the seed with highest silhouette score"
    ),
    "naive_average": (
        "Average coordinates across seeds without alignment (negative control)"
    ),
    "procrustes": (
        "Procrustes-align to first seed, then average coordinates"
    ),
    "reap": (
        "Average pairwise distance matrices, then project via UMAP (our method)"
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MethodResult(BaseModel):
    """Evaluation results for a single consensus method."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(
        ..., description="Method identifier from ALL_METHODS"
    )
    description: str = Field(
        ..., description="Human-readable method description for tables"
    )
    trustworthiness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean trustworthiness (for single_seed: mean across seeds)",
    )
    trustworthiness_std: float = Field(
        ...,
        ge=0.0,
        description="Std of trustworthiness across seeds (0 for consensus)",
    )
    silhouette: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Silhouette score at best K",
    )
    silhouette_std: float = Field(
        ...,
        ge=0.0,
        description="Std of silhouette across seeds (0 for consensus)",
    )
    best_k: int = Field(
        ..., ge=2, description="Optimal K selected by silhouette search"
    )
    best_k_std: float = Field(
        ...,
        ge=0.0,
        description="Std of best K across seeds (0 for consensus)",
    )
    seed_to_seed_ari_mean: float | None = Field(
        default=None,
        description="Mean of upper triangle of pairwise ARI matrix",
    )
    seed_to_seed_ari_std: float | None = Field(
        default=None,
        description="Std of upper triangle of pairwise ARI matrix",
    )
    seed_to_consensus_ari_mean: float | None = Field(
        default=None,
        description=(
            "Mean ARI between each seed's clustering and this method's "
            "consensus clustering"
        ),
    )
    n_seeds: int = Field(
        ..., ge=2, description="Number of UMAP seeds used"
    )
    runtime_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock time for this method"
    )


class BenchmarkResult(BaseModel):
    """Complete benchmark comparison across all methods for one dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(
        ..., description="Human-readable dataset identifier"
    )
    n_samples: int = Field(
        ..., ge=1, description="Number of data points"
    )
    n_features: int = Field(
        ..., ge=1, description="Original feature dimensionality"
    )
    n_components: int = Field(
        ..., ge=2, description="UMAP target dimensionality"
    )
    n_neighbors: int = Field(
        ..., ge=2, description="UMAP neighborhood size"
    )
    min_dist: float = Field(
        ..., ge=0.0, le=1.0, description="UMAP minimum distance"
    )
    k_range: list[int] = Field(
        ..., min_length=1, description="Candidate K values for clustering"
    )
    n_seeds: int = Field(
        ..., ge=2, description="Number of UMAP seeds"
    )
    methods: list[MethodResult] = Field(
        ..., min_length=1, description="Per-method evaluation results"
    )


class BootstrapCI(BaseModel):
    """Bootstrap confidence interval for a single method-metric pair."""

    model_config = ConfigDict(frozen=True)

    metric: str = Field(
        ..., description="Metric name (trustworthiness, silhouette, best_k)"
    )
    method: str = Field(
        ..., description="Method identifier from ALL_METHODS"
    )
    mean: float = Field(
        ..., description="Mean of bootstrap distribution"
    )
    ci_lower: float = Field(
        ..., description="2.5th percentile (lower bound of 95% CI)"
    )
    ci_upper: float = Field(
        ..., description="97.5th percentile (upper bound of 95% CI)"
    )
    n_bootstrap: int = Field(
        ..., ge=1, description="Number of bootstrap iterations"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _evaluate_embedding(
    X_high: np.ndarray,
    embedding: np.ndarray,
    k_range: list[int],
    input_metric: str = "cosine",
    sil_metric: str = "euclidean",
) -> dict[str, float | int]:
    """Compute trustworthiness, silhouette, and best K for an embedding.

    Pure function (aside from KMeans internal state).

    Parameters
    ----------
    X_high : (n_samples, n_features) original high-dimensional data.
    embedding : (n_samples, n_components) low-dimensional embedding.
    k_range : Candidate K values for silhouette-based K selection.
    input_metric : Distance metric for the high-dimensional space.
    sil_metric : Distance metric for silhouette computation in the
        low-dimensional space.

    Returns
    -------
    Dictionary with keys: trustworthiness, silhouette, best_k.
    """
    tw = compute_trustworthiness(
        X_high, embedding, n_neighbors=15, metric=input_metric
    )
    search_result = find_best_k(embedding, k_range, metric=sil_metric)
    return {
        "trustworthiness": tw,
        "silhouette": search_result.best_score,
        "best_k": search_result.best_k,
    }


def _cluster_embedding(
    embedding: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster an embedding with KMeans at the given K.

    Parameters
    ----------
    embedding : (n_samples, n_components) data to cluster.
    k : Number of clusters.
    random_state : Seed for KMeans.

    Returns
    -------
    (n_samples,) integer label array.
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return np.asarray(model.fit_predict(embedding))


def _compute_seed_stability(
    umap_embeddings: list[np.ndarray],
    consensus_labels: np.ndarray,
    k_range: list[int],
    sil_metric: str = "euclidean",
) -> dict[str, float | None]:
    """Compute seed-to-seed and seed-to-consensus ARI statistics.

    For each seed embedding, find its optimal K via silhouette, cluster it,
    then compute:
    - Pairwise ARI across all seed clusterings (seed-to-seed stability).
    - ARI between each seed's clustering and the consensus clustering
      (seed-to-consensus agreement).

    Parameters
    ----------
    umap_embeddings : List of (n_samples, n_components) per-seed embeddings.
    consensus_labels : (n_samples,) cluster labels from the consensus method.
    k_range : Candidate K values for per-seed clustering.
    sil_metric : Distance metric for silhouette-based K selection.

    Returns
    -------
    Dictionary with seed_to_seed_ari_mean, seed_to_seed_ari_std,
    seed_to_consensus_ari_mean.
    """
    seed_labels_list: list[np.ndarray] = []
    for emb in umap_embeddings:
        search = find_best_k(emb, k_range, metric=sil_metric)
        labels = _cluster_embedding(emb, search.best_k)
        seed_labels_list.append(labels)

    n = len(seed_labels_list)

    # Seed-to-seed: pairwise ARI upper triangle
    ari_values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(seed_labels_list[i], seed_labels_list[j])
            ari_values.append(float(ari))

    ari_arr = np.array(ari_values)
    s2s_mean = float(ari_arr.mean()) if len(ari_arr) > 0 else None
    s2s_std = float(ari_arr.std()) if len(ari_arr) > 0 else None

    # Seed-to-consensus
    s2c_values = [
        float(adjusted_rand_score(sl, consensus_labels))
        for sl in seed_labels_list
    ]
    s2c_mean = float(np.mean(s2c_values)) if s2c_values else None

    return {
        "seed_to_seed_ari_mean": s2s_mean,
        "seed_to_seed_ari_std": s2s_std,
        "seed_to_consensus_ari_mean": s2c_mean,
    }


def _get_naive_average(
    umap_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Average coordinates across seeds without any alignment.

    This is a NEGATIVE CONTROL. UMAP embeddings have rotation/reflection
    ambiguity, so averaging raw coordinates destroys structure. Expected to
    produce low silhouette and low ARI.

    Pure function.

    Parameters
    ----------
    umap_embeddings : List of (n_samples, n_components) per-seed embeddings.

    Returns
    -------
    (n_samples, n_components) naive average embedding.
    """
    return np.mean(np.stack(umap_embeddings), axis=0)


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def _run_single_seed(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Evaluate each seed independently and report mean +/- std.

    This shows how unstable single-seed UMAP is: metrics vary widely across
    seeds. No consensus is applied.

    Parameters
    ----------
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult with per-seed mean/std statistics.
    """
    t0 = time.monotonic()
    n_seeds = len(umap_embeddings)

    tw_list: list[float] = []
    sil_list: list[float] = []
    k_list: list[int] = []

    for i, emb in enumerate(umap_embeddings):
        logger.debug("single_seed: evaluating seed %d/%d", i + 1, n_seeds)
        result = _evaluate_embedding(
            X_high, emb, k_range, input_metric, sil_metric
        )
        tw_list.append(result["trustworthiness"])
        sil_list.append(result["silhouette"])
        k_list.append(int(result["best_k"]))

    # Seed-to-seed ARI: cluster each seed at its best K, compare pairwise
    seed_labels_list: list[np.ndarray] = []
    for emb, k in zip(umap_embeddings, k_list):
        seed_labels_list.append(_cluster_embedding(emb, k))

    ari_values: list[float] = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ari = adjusted_rand_score(
                seed_labels_list[i], seed_labels_list[j]
            )
            ari_values.append(float(ari))

    ari_arr = np.array(ari_values)
    elapsed = time.monotonic() - t0

    return MethodResult(
        method="single_seed",
        description=_METHOD_DESCRIPTIONS["single_seed"],
        trustworthiness=float(np.mean(tw_list)),
        trustworthiness_std=float(np.std(tw_list)),
        silhouette=float(np.mean(sil_list)),
        silhouette_std=float(np.std(sil_list)),
        best_k=int(np.median(k_list)),
        best_k_std=float(np.std(k_list)),
        seed_to_seed_ari_mean=(
            float(ari_arr.mean()) if len(ari_arr) > 0 else None
        ),
        seed_to_seed_ari_std=(
            float(ari_arr.std()) if len(ari_arr) > 0 else None
        ),
        seed_to_consensus_ari_mean=None,
        n_seeds=n_seeds,
        runtime_seconds=elapsed,
    )


def _run_best_of_n(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Pick the seed with the highest silhouette score.

    Demonstrates that cherry-picking the "best" seed does not solve the
    stability problem: the selected seed changes with each run, and the
    resulting clustering is not reproducible.

    Parameters
    ----------
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult for the best single seed.
    """
    t0 = time.monotonic()
    n_seeds = len(umap_embeddings)

    best_idx = -1
    best_sil = -2.0
    eval_results: list[dict[str, float | int]] = []

    for i, emb in enumerate(umap_embeddings):
        result = _evaluate_embedding(
            X_high, emb, k_range, input_metric, sil_metric
        )
        eval_results.append(result)
        if result["silhouette"] > best_sil:
            best_sil = result["silhouette"]
            best_idx = i

    best_result = eval_results[best_idx]
    best_emb = umap_embeddings[best_idx]

    # Compute stability relative to the chosen seed
    consensus_labels = _cluster_embedding(
        best_emb, int(best_result["best_k"])
    )
    stability = _compute_seed_stability(
        umap_embeddings, consensus_labels, k_range, sil_metric
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "best_of_n: selected seed %d/%d (sil=%.4f)",
        best_idx + 1, n_seeds, best_sil,
    )

    return MethodResult(
        method="best_of_n",
        description=_METHOD_DESCRIPTIONS["best_of_n"],
        trustworthiness=float(best_result["trustworthiness"]),
        trustworthiness_std=0.0,
        silhouette=float(best_result["silhouette"]),
        silhouette_std=0.0,
        best_k=int(best_result["best_k"]),
        best_k_std=0.0,
        seed_to_seed_ari_mean=stability["seed_to_seed_ari_mean"],
        seed_to_seed_ari_std=stability["seed_to_seed_ari_std"],
        seed_to_consensus_ari_mean=stability[
            "seed_to_consensus_ari_mean"
        ],
        n_seeds=n_seeds,
        runtime_seconds=elapsed,
    )


def _run_naive_average(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Average coordinates across seeds without alignment.

    Negative control: UMAP embeddings have rotation/reflection ambiguity, so
    averaging raw coordinates destroys structure. Expected to produce degraded
    silhouette and trustworthiness scores.

    Parameters
    ----------
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult for the naive average.
    """
    t0 = time.monotonic()
    n_seeds = len(umap_embeddings)

    naive_emb = _get_naive_average(umap_embeddings)
    result = _evaluate_embedding(
        X_high, naive_emb, k_range, input_metric, sil_metric
    )

    consensus_labels = _cluster_embedding(naive_emb, int(result["best_k"]))
    stability = _compute_seed_stability(
        umap_embeddings, consensus_labels, k_range, sil_metric
    )

    elapsed = time.monotonic() - t0

    return MethodResult(
        method="naive_average",
        description=_METHOD_DESCRIPTIONS["naive_average"],
        trustworthiness=float(result["trustworthiness"]),
        trustworthiness_std=0.0,
        silhouette=float(result["silhouette"]),
        silhouette_std=0.0,
        best_k=int(result["best_k"]),
        best_k_std=0.0,
        seed_to_seed_ari_mean=stability["seed_to_seed_ari_mean"],
        seed_to_seed_ari_std=stability["seed_to_seed_ari_std"],
        seed_to_consensus_ari_mean=stability[
            "seed_to_consensus_ari_mean"
        ],
        n_seeds=n_seeds,
        runtime_seconds=elapsed,
    )


def _run_procrustes(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Procrustes-align to the first seed, then average coordinates.

    Existing baseline from the literature. Addresses rotation/reflection
    ambiguity via orthogonal Procrustes alignment before averaging, but
    residual alignment error degrades cluster structure (ARI ~0.56 vs
    REAP's ~0.75).

    Parameters
    ----------
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult for Procrustes consensus.
    """
    t0 = time.monotonic()
    n_seeds = len(umap_embeddings)

    proc_emb = get_procrustes_consensus(umap_embeddings)
    result = _evaluate_embedding(
        X_high, proc_emb, k_range, input_metric, sil_metric
    )

    consensus_labels = _cluster_embedding(proc_emb, int(result["best_k"]))
    stability = _compute_seed_stability(
        umap_embeddings, consensus_labels, k_range, sil_metric
    )

    elapsed = time.monotonic() - t0

    return MethodResult(
        method="procrustes",
        description=_METHOD_DESCRIPTIONS["procrustes"],
        trustworthiness=float(result["trustworthiness"]),
        trustworthiness_std=0.0,
        silhouette=float(result["silhouette"]),
        silhouette_std=0.0,
        best_k=int(result["best_k"]),
        best_k_std=0.0,
        seed_to_seed_ari_mean=stability["seed_to_seed_ari_mean"],
        seed_to_seed_ari_std=stability["seed_to_seed_ari_std"],
        seed_to_consensus_ari_mean=stability[
            "seed_to_consensus_ari_mean"
        ],
        n_seeds=n_seeds,
        runtime_seconds=elapsed,
    )


def _run_reap(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Average pairwise distance matrices, then project via UMAP.

    The REAP method. Distance matrices are invariant to rotation/reflection,
    so averaging them preserves relational structure. A final UMAP step on the
    precomputed consensus distance matrix produces a stable embedding.

    Parameters
    ----------
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    n_components : UMAP target dimensionality for the final projection.
    n_neighbors : UMAP neighborhood size for the final projection.
    min_dist : UMAP minimum distance for the final projection.
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult for REAP distance-matrix consensus.
    """
    t0 = time.monotonic()
    n_seeds = len(umap_embeddings)

    D_consensus = get_consensus_distance_matrix(umap_embeddings)
    reap_emb, _reducer = get_consensus_embedding(
        D_consensus, n_components, n_neighbors, min_dist
    )

    result = _evaluate_embedding(
        X_high, reap_emb, k_range, input_metric, sil_metric
    )

    consensus_labels = _cluster_embedding(reap_emb, int(result["best_k"]))
    stability = _compute_seed_stability(
        umap_embeddings, consensus_labels, k_range, sil_metric
    )

    elapsed = time.monotonic() - t0

    return MethodResult(
        method="reap",
        description=_METHOD_DESCRIPTIONS["reap"],
        trustworthiness=float(result["trustworthiness"]),
        trustworthiness_std=0.0,
        silhouette=float(result["silhouette"]),
        silhouette_std=0.0,
        best_k=int(result["best_k"]),
        best_k_std=0.0,
        seed_to_seed_ari_mean=stability["seed_to_seed_ari_mean"],
        seed_to_seed_ari_std=stability["seed_to_seed_ari_std"],
        seed_to_consensus_ari_mean=stability[
            "seed_to_consensus_ari_mean"
        ],
        n_seeds=n_seeds,
        runtime_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

MethodName = Literal[
    "single_seed", "best_of_n", "naive_average", "procrustes", "reap"
]


def _dispatch_method(
    method: str,
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    input_metric: str,
    sil_metric: str,
) -> MethodResult:
    """Route to the correct per-method runner.

    Parameters
    ----------
    method : One of ALL_METHODS.
    X_high : Original high-d data.
    umap_embeddings : Per-seed UMAP embeddings (shared).
    k_range : K candidates for clustering.
    n_components : UMAP target dimensionality (used by reap only).
    n_neighbors : UMAP neighborhood size (used by reap only).
    min_dist : UMAP minimum distance (used by reap only).
    input_metric : Metric for high-d trustworthiness.
    sil_metric : Metric for silhouette in low-d.

    Returns
    -------
    MethodResult for the requested method.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == "single_seed":
        return _run_single_seed(
            X_high, umap_embeddings, k_range, input_metric, sil_metric
        )
    elif method == "best_of_n":
        return _run_best_of_n(
            X_high, umap_embeddings, k_range, input_metric, sil_metric
        )
    elif method == "naive_average":
        return _run_naive_average(
            X_high, umap_embeddings, k_range, input_metric, sil_metric
        )
    elif method == "procrustes":
        return _run_procrustes(
            X_high, umap_embeddings, k_range, input_metric, sil_metric
        )
    elif method == "reap":
        return _run_reap(
            X_high,
            umap_embeddings,
            k_range,
            n_components,
            n_neighbors,
            min_dist,
            input_metric,
            sil_metric,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {ALL_METHODS}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_benchmark(
    X: np.ndarray,
    dataset_name: str,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
    methods: list[str] | None = None,
) -> BenchmarkResult:
    """Run all consensus methods on a dataset and compare.

    Workflow:
    1. Compute multi-seed UMAP embeddings once (shared across methods).
    2. For each method, apply its consensus/selection strategy.
    3. Evaluate each method's embedding (trustworthiness, silhouette, K).
    4. Return structured results for all methods.

    Parameters
    ----------
    X : (n_samples, n_features) input data (e.g., sentence embeddings).
    dataset_name : Human-readable identifier for the dataset.
    seeds : List of random seeds for multi-seed UMAP.
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance parameter.
    k_range : Candidate K values for silhouette-based K selection.
    metric : Distance metric for the high-dimensional input space.
        Default "cosine" for sentence embeddings.
    silhouette_metric : Distance metric for silhouette computation in
        the low-dimensional UMAP output space. Default "euclidean"
        (correct per McInnes 2018).
    methods : Which methods to run. Defaults to ALL_METHODS. Must be a
        subset of ALL_METHODS.

    Returns
    -------
    BenchmarkResult containing per-method evaluations.

    Side effects
    ------------
    UMAP and KMeans internal random state. Logging.

    Raises
    ------
    ValueError
        If X is not 2-d, seeds has fewer than 2 entries, or an unknown
        method is requested.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-d input, got shape {X.shape}")
    if len(seeds) < 2:
        raise ValueError(
            f"Need at least 2 seeds for benchmark, got {len(seeds)}"
        )

    selected_methods = methods if methods is not None else ALL_METHODS
    for m in selected_methods:
        if m not in ALL_METHODS:
            raise ValueError(
                f"Unknown method {m!r}. Choose from: {ALL_METHODS}"
            )

    n_samples, n_features = X.shape
    logger.info(
        "Benchmark '%s': %d samples, %d features, %d seeds, "
        "methods=%s",
        dataset_name,
        n_samples,
        n_features,
        len(seeds),
        selected_methods,
    )

    # Step 1: Compute shared multi-seed UMAP embeddings
    logger.info("Computing %d UMAP embeddings (shared)...", len(seeds))
    t_umap = time.monotonic()
    umap_embeddings = get_multi_seed_embeddings(
        X, seeds, n_components, n_neighbors, min_dist, metric
    )
    logger.info(
        "UMAP embeddings computed in %.1fs", time.monotonic() - t_umap
    )

    # Step 2-3: Run and evaluate each method
    method_results: list[MethodResult] = []
    for method_name in selected_methods:
        logger.info("Running method: %s", method_name)
        result = _dispatch_method(
            method_name,
            X,
            umap_embeddings,
            k_range,
            n_components,
            n_neighbors,
            min_dist,
            metric,
            silhouette_metric,
        )
        method_results.append(result)
        logger.info(
            "  %s: TW=%.4f, sil=%.4f, K=%d (%.1fs)",
            method_name,
            result.trustworthiness,
            result.silhouette,
            result.best_k,
            result.runtime_seconds,
        )

    return BenchmarkResult(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        k_range=list(k_range),
        n_seeds=len(seeds),
        methods=method_results,
    )


def compute_bootstrap_ci(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    n_bootstrap: int = 100,
    subsample_fraction: float = 0.8,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
    methods: list[str] | None = None,
    random_state: int = 42,
) -> list[BootstrapCI]:
    """Bootstrap confidence intervals for each method's metrics.

    For each bootstrap iteration:
    1. Subsample data WITHOUT replacement (fraction of original).
    2. Run ``run_benchmark`` on the subsample.
    3. Collect trustworthiness, silhouette, and best_k per method.

    Then compute 95% CI (2.5th and 97.5th percentiles) for each
    method x metric combination.

    Parameters
    ----------
    X : (n_samples, n_features) input data.
    seeds : Random seeds for multi-seed UMAP.
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance parameter.
    k_range : Candidate K values for clustering.
    n_bootstrap : Number of bootstrap iterations. Default 100.
    subsample_fraction : Fraction of data to subsample each iteration.
        Must be in (0, 1). Default 0.8.
    metric : Distance metric for the input space. Default "cosine".
    silhouette_metric : Distance metric for silhouette. Default "euclidean".
    methods : Which methods to run. Defaults to ALL_METHODS.
    random_state : Master seed for reproducible subsampling.

    Returns
    -------
    List of BootstrapCI objects, one per (method, metric) combination.
    Metrics reported: trustworthiness, silhouette, best_k.

    Side effects
    ------------
    UMAP and KMeans internal random state. Logging. Computationally
    expensive — runs ``n_bootstrap`` full benchmarks.

    Raises
    ------
    ValueError
        If subsample_fraction is not in (0, 1) or X is invalid.
    """
    if not 0.0 < subsample_fraction < 1.0:
        raise ValueError(
            f"subsample_fraction must be in (0, 1), got {subsample_fraction}"
        )

    selected_methods = methods if methods is not None else ALL_METHODS
    n_samples = X.shape[0]
    subsample_size = max(
        int(n_samples * subsample_fraction),
        # Need enough samples for clustering — at least max(k_range) + 1
        max(k_range) + 1 if k_range else 10,
    )
    subsample_size = min(subsample_size, n_samples)

    rng = np.random.default_rng(random_state)

    # Accumulators: method_name -> metric_name -> list[float]
    distributions: dict[str, dict[str, list[float]]] = {
        m: {"trustworthiness": [], "silhouette": [], "best_k": []}
        for m in selected_methods
    }

    reported_metrics = ["trustworthiness", "silhouette", "best_k"]

    logger.info(
        "Bootstrap CI: %d iterations, subsample=%d/%d (%.0f%%), "
        "methods=%s",
        n_bootstrap,
        subsample_size,
        n_samples,
        subsample_fraction * 100,
        selected_methods,
    )

    for b in range(n_bootstrap):
        logger.info("Bootstrap iteration %d/%d", b + 1, n_bootstrap)

        # Subsample without replacement
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X[indices]

        # Adjust k_range: exclude K values >= subsample_size
        k_range_sub = [k for k in k_range if k < subsample_size]
        if len(k_range_sub) == 0:
            logger.warning(
                "Bootstrap %d: no valid K values for subsample size %d, "
                "skipping",
                b + 1,
                subsample_size,
            )
            continue

        try:
            bench = run_benchmark(
                X_sub,
                dataset_name=f"bootstrap_{b}",
                seeds=seeds,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                k_range=k_range_sub,
                metric=metric,
                silhouette_metric=silhouette_metric,
                methods=selected_methods,
            )
        except Exception:
            logger.exception(
                "Bootstrap iteration %d failed, skipping", b + 1
            )
            continue

        for mr in bench.methods:
            distributions[mr.method]["trustworthiness"].append(
                mr.trustworthiness
            )
            distributions[mr.method]["silhouette"].append(mr.silhouette)
            distributions[mr.method]["best_k"].append(float(mr.best_k))

    # Compute CIs
    ci_results: list[BootstrapCI] = []
    for method_name in selected_methods:
        for metric_name in reported_metrics:
            values = np.array(distributions[method_name][metric_name])
            if len(values) == 0:
                logger.warning(
                    "No bootstrap samples for %s/%s, skipping CI",
                    method_name,
                    metric_name,
                )
                continue
            ci_results.append(
                BootstrapCI(
                    metric=metric_name,
                    method=method_name,
                    mean=float(values.mean()),
                    ci_lower=float(np.percentile(values, 2.5)),
                    ci_upper=float(np.percentile(values, 97.5)),
                    n_bootstrap=len(values),
                )
            )

    logger.info(
        "Bootstrap complete: %d CIs computed across %d methods",
        len(ci_results),
        len(selected_methods),
    )
    return ci_results
