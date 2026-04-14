"""Ablation and sensitivity analysis for REAP consensus pipelines.

Provides three analysis dimensions:
1. Seed count ablation — how many seeds are needed for stable consensus?
2. Parameter sensitivity sweep — how sensitive is REAP to UMAP hyperparameters?
3. Computational scaling — how does runtime scale with data size and seed count?

Each analysis returns a frozen Pydantic model with all metrics for downstream
reporting and visualization. Functions follow the ``run_*`` naming convention
because they execute multi-step pipelines with side effects (UMAP runs, timing).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans

from reap.clustering import find_best_k
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.evaluation import (
    compute_pairwise_ari,
    compute_silhouette,
    compute_trustworthiness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models — Seed count ablation
# ---------------------------------------------------------------------------


class SeedAblationPoint(BaseModel):
    """Metrics for a single seed-count setting in the ablation study."""

    model_config = ConfigDict(frozen=True)

    n_seeds: int = Field(..., ge=2, description="Number of UMAP seeds used")
    trustworthiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Local neighborhood preservation in [0, 1]",
    )
    silhouette: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Cluster separation quality in [-1, 1]",
    )
    best_k: int = Field(
        ..., ge=2,
        description="Optimal K selected by silhouette search",
    )
    seed_to_seed_ari_mean: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Mean pairwise ARI across per-seed clusterings",
    )
    seed_to_seed_ari_std: float = Field(
        ..., ge=0.0,
        description="Std of pairwise ARI across per-seed clusterings",
    )
    runtime_seconds: float = Field(
        ..., ge=0.0,
        description="Wall-clock time for this seed count (seconds)",
    )


class SeedAblationResult(BaseModel):
    """Result from varying the number of seeds in the REAP consensus pipeline."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(
        ..., description="Identifier for the dataset used",
    )
    n_components: int = Field(
        ..., ge=2, description="UMAP target dimensionality",
    )
    n_neighbors: int = Field(
        ..., ge=2, description="UMAP neighborhood size",
    )
    min_dist: float = Field(
        ..., ge=0.0, le=1.0, description="UMAP minimum distance",
    )
    seed_counts: list[int] = Field(
        ..., min_length=1,
        description="List of seed counts tested",
    )
    results: list[SeedAblationPoint] = Field(
        ..., min_length=1,
        description="Per-seed-count metrics, one per entry in seed_counts",
    )


# ---------------------------------------------------------------------------
# Data models — Parameter sensitivity sweep
# ---------------------------------------------------------------------------


class ParameterSweepPoint(BaseModel):
    """Metrics for a single parameter value in the sensitivity sweep."""

    model_config = ConfigDict(frozen=True)

    parameter_value: float = Field(
        ..., description="Value of the swept parameter for this point",
    )
    trustworthiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Local neighborhood preservation in [0, 1]",
    )
    silhouette: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Cluster separation quality in [-1, 1]",
    )
    best_k: int = Field(
        ..., ge=2,
        description="Optimal K selected by silhouette search",
    )
    seed_to_seed_ari_mean: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Mean pairwise ARI across per-seed clusterings",
    )
    runtime_seconds: float = Field(
        ..., ge=0.0,
        description="Wall-clock time for this parameter value (seconds)",
    )


class ParameterSweepResult(BaseModel):
    """Result from sweeping one UMAP parameter while holding others fixed."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(
        ..., description="Identifier for the dataset used",
    )
    swept_parameter: str = Field(
        ..., description=(
            "Name of the parameter being swept: "
            "'n_components', 'n_neighbors', or 'min_dist'"
        ),
    )
    parameter_values: list[float] = Field(
        ..., min_length=1,
        description="List of parameter values tested",
    )
    fixed_params: dict[str, object] = Field(
        ..., description="Parameters held constant during the sweep",
    )
    results: list[ParameterSweepPoint] = Field(
        ..., min_length=1,
        description="Per-value metrics, one per entry in parameter_values",
    )


# ---------------------------------------------------------------------------
# Data models — Computational scaling
# ---------------------------------------------------------------------------


class ScalingResult(BaseModel):
    """Result from measuring runtime scaling with data size or seed count."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(
        ..., description="Identifier for the dataset used",
    )
    scaling_variable: str = Field(
        ..., description="Variable being scaled: 'n_samples' or 'n_seeds'",
    )
    variable_values: list[int] = Field(
        ..., min_length=1,
        description="Values of the scaling variable tested",
    )
    runtimes: list[float] = Field(
        ..., min_length=1,
        description="Wall-clock time in seconds for each variable value",
    )
    samples_per_second: list[float] = Field(
        ..., min_length=1,
        description="Throughput: samples processed per second",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_seeds(master_seed: int, n_seeds: int) -> list[int]:
    """Generate a reproducible list of random seeds from a master seed.

    Parameters
    ----------
    master_seed : Root seed for the random number generator.
    n_seeds : Number of seeds to generate.

    Returns
    -------
    List of unique integer seeds in [0, 10_000).
    """
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 10_000, size=n_seeds).tolist()


def _run_reap_pipeline_with_metrics(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int] | range,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
) -> dict[str, float | int]:
    """Run the full REAP consensus pipeline and collect all ablation metrics.

    This is an internal helper that centralizes the pipeline logic shared by
    seed ablation, parameter sweeps, and scaling analysis.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    seeds : Random seeds for multi-seed UMAP.
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance.
    k_range : K values to evaluate for clustering.
    metric : Distance metric for the input space.
    silhouette_metric : Metric for silhouette computation in UMAP output space.

    Returns
    -------
    Dictionary with keys: trustworthiness, silhouette, best_k,
    seed_to_seed_ari_mean, seed_to_seed_ari_std, runtime_seconds.

    Side effects: UMAP internal state, logging, wall-clock timing.
    """
    t_start = time.perf_counter()

    # Step 1: Multi-seed UMAP
    umap_embeddings = get_multi_seed_embeddings(
        X, seeds, n_components, n_neighbors, min_dist, metric,
    )

    # Step 2: Distance-matrix consensus
    D_consensus = get_consensus_distance_matrix(umap_embeddings)
    consensus_emb, _ = get_consensus_embedding(
        D_consensus, n_components, n_neighbors, min_dist,
    )

    # Step 3: Trustworthiness
    n_nn = min(n_neighbors, X.shape[0] - 2)
    tw = compute_trustworthiness(
        X, consensus_emb, n_neighbors=n_nn, metric=metric,
    )

    # Step 4: Best K via silhouette
    sil_result = find_best_k(consensus_emb, k_range, metric=silhouette_metric)
    consensus_labels = KMeans(
        n_clusters=sil_result.best_k, random_state=42, n_init="auto",
    ).fit_predict(consensus_emb)
    sil_score = compute_silhouette(
        consensus_emb, consensus_labels, metric=silhouette_metric,
    )

    # Step 5: Seed-to-seed ARI (cluster each individual seed embedding)
    seed_labels_list: list[np.ndarray] = []
    for emb in umap_embeddings:
        seed_sil = find_best_k(emb, k_range, metric=silhouette_metric)
        labels = KMeans(
            n_clusters=seed_sil.best_k, random_state=42, n_init="auto",
        ).fit_predict(emb)
        seed_labels_list.append(np.asarray(labels))

    ari_matrix = compute_pairwise_ari(seed_labels_list)
    triu_indices = np.triu_indices_from(ari_matrix, k=1)
    ari_upper = ari_matrix[triu_indices]

    t_elapsed = time.perf_counter() - t_start

    return {
        "trustworthiness": float(tw),
        "silhouette": float(sil_score),
        "best_k": int(sil_result.best_k),
        "seed_to_seed_ari_mean": float(ari_upper.mean()),
        "seed_to_seed_ari_std": float(ari_upper.std()),
        "runtime_seconds": t_elapsed,
    }


# ---------------------------------------------------------------------------
# Public API — Seed count ablation
# ---------------------------------------------------------------------------


def run_seed_ablation(
    X: np.ndarray,
    dataset_name: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int] | range,
    seed_counts: list[int] | None = None,
    master_seed: int = 42,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
) -> SeedAblationResult:
    """Run REAP with increasing seed counts to show convergence behavior.

    Uses a fixed master seed to generate reproducible seed lists. For each
    count, draws seeds from the master RNG and runs the full REAP pipeline.
    Nested seed lists are cumulative: the 10-seed list is a superset of the
    5-seed list, so improvements reflect additional information rather than
    different randomness.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    dataset_name : Identifier for reporting (e.g., "ai_art", "korean_forest").
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance.
    k_range : K values to evaluate for clustering.
    seed_counts : Number of seeds to test. Default: [5, 10, 15, 20, 25, 30].
    master_seed : Root seed for generating per-count seed lists.
    metric : Distance metric for the input space.
    silhouette_metric : Metric for silhouette in UMAP output space.

    Returns
    -------
    SeedAblationResult with per-count metrics showing convergence.

    Side effects: UMAP internal state, logging, wall-clock timing.
    """
    if seed_counts is None:
        seed_counts = [5, 10, 15, 20, 25, 30]

    # Generate the largest seed list once; slice for smaller counts
    max_seeds = max(seed_counts)
    all_seeds = _generate_seeds(master_seed, max_seeds)

    points: list[SeedAblationPoint] = []
    for count in seed_counts:
        if count < 2:
            logger.warning("Seed count %d < 2 — skipping (need >= 2 for consensus)", count)
            continue

        seeds = all_seeds[:count]
        logger.info(
            "Seed ablation: %d/%d seeds (dataset=%s, d=%d, nn=%d, md=%.4f)",
            count, max_seeds, dataset_name, n_components, n_neighbors, min_dist,
        )

        metrics = _run_reap_pipeline_with_metrics(
            X, seeds, n_components, n_neighbors, min_dist,
            k_range, metric, silhouette_metric,
        )

        points.append(SeedAblationPoint(
            n_seeds=count,
            trustworthiness=metrics["trustworthiness"],
            silhouette=metrics["silhouette"],
            best_k=int(metrics["best_k"]),
            seed_to_seed_ari_mean=metrics["seed_to_seed_ari_mean"],
            seed_to_seed_ari_std=metrics["seed_to_seed_ari_std"],
            runtime_seconds=metrics["runtime_seconds"],
        ))

        logger.info(
            "  → TW=%.3f, Sil=%.3f, K=%d, ARI=%.3f±%.3f (%.1fs)",
            metrics["trustworthiness"],
            metrics["silhouette"],
            metrics["best_k"],
            metrics["seed_to_seed_ari_mean"],
            metrics["seed_to_seed_ari_std"],
            metrics["runtime_seconds"],
        )

    if not points:
        raise ValueError(
            f"No valid seed counts in {seed_counts} (all < 2)"
        )

    # Filter to only valid counts that produced points
    valid_counts = [p.n_seeds for p in points]

    return SeedAblationResult(
        dataset_name=dataset_name,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        seed_counts=valid_counts,
        results=points,
    )


# ---------------------------------------------------------------------------
# Public API — Parameter sensitivity sweep
# ---------------------------------------------------------------------------


_VALID_SWEEP_PARAMS = {"n_components", "n_neighbors", "min_dist"}


def run_parameter_sweep(
    X: np.ndarray,
    dataset_name: str,
    swept_parameter: str,
    parameter_values: list[float] | list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seeds: list[int],
    k_range: list[int] | range,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
) -> ParameterSweepResult:
    """Sweep one UMAP parameter while holding the others fixed.

    Measures how sensitive trustworthiness, silhouette, and seed stability
    are to the swept parameter. Useful for understanding REAP robustness
    across hyperparameter choices.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    dataset_name : Identifier for reporting.
    swept_parameter : One of "n_components", "n_neighbors", or "min_dist".
    parameter_values : Values to try for the swept parameter.
    n_components : UMAP target dimensionality (overridden if swept).
    n_neighbors : UMAP neighborhood size (overridden if swept).
    min_dist : UMAP minimum distance (overridden if swept).
    seeds : Random seeds for multi-seed UMAP.
    k_range : K values to evaluate for clustering.
    metric : Distance metric for the input space.
    silhouette_metric : Metric for silhouette in UMAP output space.

    Returns
    -------
    ParameterSweepResult with per-value metrics.

    Raises
    ------
    ValueError
        If swept_parameter is not one of the three valid UMAP parameters.

    Side effects: UMAP internal state, logging, wall-clock timing.
    """
    if swept_parameter not in _VALID_SWEEP_PARAMS:
        raise ValueError(
            f"swept_parameter must be one of {sorted(_VALID_SWEEP_PARAMS)}, "
            f"got {swept_parameter!r}"
        )

    # Build the fixed-params record (excludes the swept parameter)
    fixed_params: dict[str, object] = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_seeds": len(seeds),
        "metric": metric,
    }
    fixed_params.pop(swept_parameter)

    points: list[ParameterSweepPoint] = []
    for i, value in enumerate(parameter_values):
        # Override the swept parameter
        sweep_nc = int(value) if swept_parameter == "n_components" else n_components
        sweep_nn = int(value) if swept_parameter == "n_neighbors" else n_neighbors
        sweep_md = float(value) if swept_parameter == "min_dist" else min_dist

        logger.info(
            "Parameter sweep %d/%d: %s=%.4g (dataset=%s)",
            i + 1, len(parameter_values), swept_parameter, value, dataset_name,
        )

        metrics = _run_reap_pipeline_with_metrics(
            X, seeds, sweep_nc, sweep_nn, sweep_md,
            k_range, metric, silhouette_metric,
        )

        points.append(ParameterSweepPoint(
            parameter_value=float(value),
            trustworthiness=metrics["trustworthiness"],
            silhouette=metrics["silhouette"],
            best_k=int(metrics["best_k"]),
            seed_to_seed_ari_mean=metrics["seed_to_seed_ari_mean"],
            runtime_seconds=metrics["runtime_seconds"],
        ))

        logger.info(
            "  → TW=%.3f, Sil=%.3f, K=%d, ARI=%.3f (%.1fs)",
            metrics["trustworthiness"],
            metrics["silhouette"],
            metrics["best_k"],
            metrics["seed_to_seed_ari_mean"],
            metrics["runtime_seconds"],
        )

    return ParameterSweepResult(
        dataset_name=dataset_name,
        swept_parameter=swept_parameter,
        parameter_values=[float(v) for v in parameter_values],
        fixed_params=fixed_params,
        results=points,
    )


# ---------------------------------------------------------------------------
# Public API — Computational scaling analysis
# ---------------------------------------------------------------------------


def _get_default_sample_sizes(n_total: int) -> list[int]:
    """Generate reasonable default sample sizes up to the dataset size.

    Produces a logarithmic-ish progression: [100, 200, 500, 1000, ...] capped
    at the actual dataset size. Always includes n_total as the final entry.

    Parameters
    ----------
    n_total : Total number of samples in the dataset.

    Returns
    -------
    Sorted list of sample sizes, all <= n_total, with at least one entry.
    """
    candidates = [100, 200, 500, 1000, 2000, 5000, 10000]
    sizes = [s for s in candidates if s < n_total]
    sizes.append(n_total)
    # Deduplicate and sort
    return sorted(set(sizes))


def run_scaling_analysis(
    X: np.ndarray,
    dataset_name: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seeds: list[int],
    k_range: list[int] | range,
    sample_sizes: list[int] | None = None,
    metric: str = "cosine",
) -> ScalingResult:
    """Measure runtime scaling of the REAP pipeline with data size.

    Subsamples X at each requested size and times the full consensus pipeline
    (multi-seed UMAP + distance-matrix consensus + clustering). Useful for
    characterizing computational complexity and planning resource allocation.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    dataset_name : Identifier for reporting.
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance.
    seeds : Random seeds for multi-seed UMAP.
    k_range : K values to evaluate for clustering.
    sample_sizes : List of n_samples to test. Default: [100, 200, 500, ...]
        up to len(X).
    metric : Distance metric for the input space.

    Returns
    -------
    ScalingResult with runtimes and throughput for each sample size.

    Side effects: UMAP internal state, logging, wall-clock timing.
    """
    n_total = X.shape[0]

    if sample_sizes is None:
        sample_sizes = _get_default_sample_sizes(n_total)

    # Validate and filter sample sizes
    valid_sizes: list[int] = []
    for size in sample_sizes:
        if size < 10:
            logger.warning(
                "Sample size %d too small for meaningful UMAP — skipping", size,
            )
            continue
        if size > n_total:
            logger.warning(
                "Sample size %d > dataset size %d — capping at %d",
                size, n_total, n_total,
            )
            valid_sizes.append(n_total)
        else:
            valid_sizes.append(size)

    # Deduplicate, preserve order
    seen: set[int] = set()
    unique_sizes: list[int] = []
    for s in valid_sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)
    valid_sizes = unique_sizes

    if not valid_sizes:
        raise ValueError(
            f"No valid sample sizes from {sample_sizes} for dataset with "
            f"{n_total} samples"
        )

    # Deterministic subsampling
    rng = np.random.default_rng(42)

    runtimes: list[float] = []
    throughputs: list[float] = []

    for i, size in enumerate(valid_sizes):
        logger.info(
            "Scaling analysis %d/%d: n_samples=%d (dataset=%s)",
            i + 1, len(valid_sizes), size, dataset_name,
        )

        # Subsample without replacement
        if size >= n_total:
            X_sub = X
        else:
            indices = rng.choice(n_total, size=size, replace=False)
            X_sub = X[indices]

        # Adjust n_neighbors if it exceeds subsample size
        effective_nn = min(n_neighbors, size - 1)

        # Adjust k_range so no K exceeds the subsample
        effective_k_range = [k for k in k_range if 2 <= k < size]
        if not effective_k_range:
            effective_k_range = [2]

        t_start = time.perf_counter()

        # Run multi-seed UMAP + consensus (core pipeline, no per-seed clustering)
        umap_embeddings = get_multi_seed_embeddings(
            X_sub, seeds, n_components, effective_nn, min_dist, metric,
        )
        D_consensus = get_consensus_distance_matrix(umap_embeddings)
        get_consensus_embedding(
            D_consensus, n_components, effective_nn, min_dist,
        )

        t_elapsed = time.perf_counter() - t_start

        throughput = size / t_elapsed if t_elapsed > 0 else 0.0
        runtimes.append(t_elapsed)
        throughputs.append(throughput)

        logger.info(
            "  → %.1fs (%.0f samples/sec)", t_elapsed, throughput,
        )

    return ScalingResult(
        dataset_name=dataset_name,
        scaling_variable="n_samples",
        variable_values=valid_sizes,
        runtimes=runtimes,
        samples_per_second=throughputs,
    )


def run_seed_scaling_analysis(
    X: np.ndarray,
    dataset_name: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    seed_counts: list[int] | None = None,
    master_seed: int = 42,
    metric: str = "cosine",
) -> ScalingResult:
    """Measure runtime scaling of the REAP pipeline with seed count.

    Holds data size fixed and varies the number of UMAP seeds to characterize
    the linear cost of additional seeds.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    dataset_name : Identifier for reporting.
    n_components : UMAP target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : UMAP minimum distance.
    seed_counts : Number of seeds to test. Default: [2, 5, 10, 15, 20, 25, 30].
    master_seed : Root seed for generating reproducible seed lists.
    metric : Distance metric for the input space.

    Returns
    -------
    ScalingResult with runtimes and throughput for each seed count.

    Side effects: UMAP internal state, logging, wall-clock timing.
    """
    if seed_counts is None:
        seed_counts = [2, 5, 10, 15, 20, 25, 30]

    max_seeds = max(seed_counts)
    all_seeds = _generate_seeds(master_seed, max_seeds)

    runtimes: list[float] = []
    throughputs: list[float] = []

    for i, count in enumerate(seed_counts):
        if count < 2:
            logger.warning("Seed count %d < 2 — skipping", count)
            continue

        seeds = all_seeds[:count]
        logger.info(
            "Seed scaling %d/%d: n_seeds=%d (dataset=%s)",
            i + 1, len(seed_counts), count, dataset_name,
        )

        t_start = time.perf_counter()

        umap_embeddings = get_multi_seed_embeddings(
            X, seeds, n_components, n_neighbors, min_dist, metric,
        )
        D_consensus = get_consensus_distance_matrix(umap_embeddings)
        get_consensus_embedding(
            D_consensus, n_components, n_neighbors, min_dist,
        )

        t_elapsed = time.perf_counter() - t_start

        n_samples = X.shape[0]
        throughput = n_samples / t_elapsed if t_elapsed > 0 else 0.0
        runtimes.append(t_elapsed)
        throughputs.append(throughput)

        logger.info("  → %.1fs (%.0f samples/sec)", t_elapsed, throughput)

    # Filter to valid counts
    valid_counts = [c for c in seed_counts if c >= 2]

    if not valid_counts:
        raise ValueError(
            f"No valid seed counts in {seed_counts} (all < 2)"
        )

    return ScalingResult(
        dataset_name=dataset_name,
        scaling_variable="n_seeds",
        variable_values=valid_counts,
        runtimes=runtimes,
        samples_per_second=throughputs,
    )
