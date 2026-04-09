"""Data validation gates for the REAP pipeline.

Three validation stages ensure data quality before expensive computations:
1. Pre-embedding: text quality (duplicates, length bounds, encoding artifacts)
2. Post-embedding: embedding quality (shape, NaN/Inf, normalization, uniqueness)
3. Cluster quality: minimum size and source diversity requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error(msg)


def validate_embeddings(
    X: np.ndarray,
    expected_dim: int | None = None,
    check_l2_norm: bool = True,
    check_duplicates: bool = True,
    norm_tolerance: float = 0.01,
) -> ValidationReport:
    """Validate embedding quality before consensus pipeline.

    Parameters
    ----------
    X : (n_samples, n_features) embedding matrix.
    expected_dim : Expected feature dimension (if known).
    check_l2_norm : Verify L2 normalization.
    check_duplicates : Check for identical embedding vectors.
    norm_tolerance : Tolerance for L2 norm deviation from 1.0.

    Returns
    -------
    ValidationReport with pass/fail, warnings, and stats.
    """
    report = ValidationReport()
    report.stats["shape"] = X.shape
    report.stats["dtype"] = str(X.dtype)

    # Dimensionality
    if X.ndim != 2:
        report.add_error(f"Expected 2-d array, got shape {X.shape}")
        return report

    n_samples, n_features = X.shape
    report.stats["n_samples"] = n_samples
    report.stats["n_features"] = n_features

    if expected_dim is not None and n_features != expected_dim:
        report.add_error(f"Expected {expected_dim} features, got {n_features}")

    # NaN / Inf
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0:
        report.add_error(f"{n_nan} NaN values detected")
    if n_inf > 0:
        report.add_error(f"{n_inf} Inf values detected")

    # L2 normalization check
    if check_l2_norm and n_samples > 0:
        norms = np.linalg.norm(X, axis=1)
        report.stats["norm_mean"] = float(norms.mean())
        report.stats["norm_std"] = float(norms.std())
        deviations = np.abs(norms - 1.0)
        n_unnormalized = int((deviations > norm_tolerance).sum())
        if n_unnormalized > 0:
            report.add_warning(
                f"{n_unnormalized}/{n_samples} vectors deviate from unit L2 norm "
                f"(max deviation: {deviations.max():.4f})"
            )

    # Duplicate check
    if check_duplicates and n_samples > 1:
        # Use cosine similarity on a sample for efficiency
        sample_size = min(n_samples, 5000)
        if sample_size < n_samples:
            idx = np.random.default_rng(42).choice(n_samples, sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        from scipy.spatial.distance import pdist
        dists = pdist(X_sample, metric="cosine")  # type: ignore[arg-type]
        n_identical = int((dists < 1e-7).sum())
        if n_identical > 0:
            report.add_error(
                f"{n_identical} near-identical embedding pairs detected "
                f"(cosine distance < 1e-7)"
            )
        report.stats["n_identical_pairs"] = n_identical

    return report


def validate_cluster_sizes(
    labels: np.ndarray,
    min_cluster_size: int = 10,
    min_cluster_fraction: float = 0.0,
    n_total: int | None = None,
) -> ValidationReport:
    """Validate cluster size distribution.

    Parameters
    ----------
    labels : (n_samples,) cluster assignments.
    min_cluster_size : Absolute minimum points per cluster.
    min_cluster_fraction : Minimum fraction of total data per cluster.
    n_total : Total dataset size (for fraction calculation; defaults to len(labels)).

    Returns
    -------
    ValidationReport with cluster size analysis.
    """
    report = ValidationReport()
    n = n_total or len(labels)

    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique.tolist(), counts.tolist()))
    report.stats["n_clusters"] = len(unique)
    report.stats["cluster_sizes"] = sizes
    report.stats["min_size"] = int(counts.min())
    report.stats["max_size"] = int(counts.max())
    report.stats["mean_size"] = float(counts.mean())

    # Check absolute minimum
    small = {k: v for k, v in sizes.items() if v < min_cluster_size}
    if small:
        report.add_warning(
            f"{len(small)} cluster(s) below min_cluster_size={min_cluster_size}: {small}"
        )

    # Check fraction minimum
    if min_cluster_fraction > 0:
        min_count = int(np.ceil(n * min_cluster_fraction))
        fraction_small = {k: v for k, v in sizes.items() if v < min_count}
        if fraction_small:
            report.add_warning(
                f"{len(fraction_small)} cluster(s) below {min_cluster_fraction:.1%} of data "
                f"(need ≥{min_count} points): {fraction_small}"
            )

    # Valid clusters (meet both criteria)
    min_count_frac = int(np.ceil(n * min_cluster_fraction)) if min_cluster_fraction > 0 else 0
    threshold = max(min_cluster_size, min_count_frac)
    n_valid = sum(1 for v in sizes.values() if v >= threshold)
    report.stats["n_valid_clusters"] = n_valid
    report.stats["valid_fraction"] = n_valid / len(unique) if unique.size > 0 else 0.0

    return report
