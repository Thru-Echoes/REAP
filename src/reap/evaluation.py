"""Evaluation metrics for REAP consensus quality and seed stability.

Metrics fall into three categories:
1. Structure preservation — trustworthiness (local neighborhood fidelity)
2. Cluster quality — silhouette score (within-cluster vs between-cluster separation)
3. Stability — seed-to-seed and seed-to-consensus ARI (reproducibility)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger(__name__)


def compute_trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> float:
    """Fraction of true nearest neighbors in high-d that remain neighbors in low-d.

    Values above 0.85 indicate faithful local geometry preservation.

    Parameters
    ----------
    X_high : (n_samples, n_features_high) original high-dimensional data.
    X_low : (n_samples, n_features_low) reduced embedding.
    n_neighbors : Neighborhood size for evaluation.
    metric : Distance metric for the high-dimensional space.

    Returns
    -------
    Trustworthiness score in [0, 1].
    """
    return float(sklearn_trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric=metric))


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Silhouette score measuring cluster separation quality.

    For UMAP output space, use metric="euclidean" (the UMAP output objective
    is Euclidean per McInnes 2018). Higher is better; > 0.7 indicates strong clusters.

    Parameters
    ----------
    X : (n_samples, n_features) embedding coordinates.
    labels : (n_samples,) cluster assignments.
    metric : Distance metric. Use "euclidean" for UMAP output, "cosine" for
             high-d sentence embeddings.

    Returns
    -------
    Mean silhouette score in [-1, 1].
    """
    n_unique = len(set(labels))
    if n_unique < 2:
        logger.warning("Only %d unique label(s) — silhouette undefined, returning -1", n_unique)
        return -1.0
    return float(silhouette_score(X, labels, metric=metric))


def compute_pairwise_ari(
    labels_list: list[np.ndarray],
) -> np.ndarray:
    """Pairwise Adjusted Rand Index between multiple label sets.

    Useful for measuring seed-to-seed clustering agreement.

    Parameters
    ----------
    labels_list : List of (n_samples,) label arrays.

    Returns
    -------
    (n_labelings, n_labelings) symmetric ARI matrix with 1.0 on diagonal.
    """
    n = len(labels_list)
    ari_matrix = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari
    return ari_matrix


def compute_seed_stability(
    seed_labels: list[np.ndarray],
    consensus_labels: np.ndarray,
) -> dict[str, float]:
    """Compute seed-to-seed and seed-to-consensus ARI statistics.

    Parameters
    ----------
    seed_labels : Per-seed cluster assignments.
    consensus_labels : Cluster assignments from the consensus embedding.

    Returns
    -------
    Dictionary with mean/std/median for both s2s and s2c ARI.
    """
    # Seed-to-seed ARI
    ari_matrix = compute_pairwise_ari(seed_labels)
    triu = ari_matrix[np.triu_indices_from(ari_matrix, k=1)]

    # Seed-to-consensus ARI
    s2c = np.array([
        adjusted_rand_score(sl, consensus_labels) for sl in seed_labels
    ])

    return {
        "s2s_ari_mean": float(triu.mean()),
        "s2s_ari_std": float(triu.std()),
        "s2s_ari_median": float(np.median(triu)),
        "s2c_ari_mean": float(s2c.mean()),
        "s2c_ari_std": float(s2c.std()),
        "s2c_ari_median": float(np.median(s2c)),
    }


def compute_distance_correlation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """Pearson correlation between pairwise Euclidean distances.

    Measures how well the projection head preserves relative geometry.
    Values > 0.9 indicate excellent geometric fidelity.

    Parameters
    ----------
    Y_true : (n_samples, d) true consensus coordinates.
    Y_pred : (n_samples, d) predicted coordinates.

    Returns
    -------
    Pearson correlation coefficient in [-1, 1].
    """
    from scipy.spatial.distance import pdist

    d_true = pdist(Y_true, metric="euclidean")
    d_pred = pdist(Y_pred, metric="euclidean")

    if d_true.std() < 1e-12 or d_pred.std() < 1e-12:
        return 0.0

    return float(np.corrcoef(d_true, d_pred)[0, 1])
