"""Clustering utilities for REAP consensus embeddings.

Provides KMeans clustering with automatic K selection via silhouette optimization.
Uses euclidean metric for UMAP output space (correct per McInnes 2018).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from reap._types import SilhouetteSearchResult

logger = logging.getLogger(__name__)


def find_best_k(
    X: np.ndarray,
    k_range: list[int] | range,
    metric: str = "euclidean",
    random_state: int = 42,
) -> SilhouetteSearchResult:
    """Find optimal K for KMeans via silhouette score grid search.

    Parameters
    ----------
    X : (n_samples, n_components) consensus embedding coordinates.
    k_range : Candidate K values to evaluate.
    metric : Distance metric for silhouette computation.
    random_state : Seed for KMeans.

    Returns
    -------
    SilhouetteSearchResult with best_k, best_score, and per-K scores.
    """
    scores: dict[int, float] = {}
    for k in k_range:
        if k < 2:
            continue
        if k >= X.shape[0]:
            logger.warning("K=%d >= n_samples=%d, skipping", k, X.shape[0])
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
        sil = float(silhouette_score(X, labels, metric=metric))
        scores[k] = sil
        logger.debug("K=%d → silhouette=%.4f", k, sil)

    if not scores:
        raise ValueError(f"No valid K in range {list(k_range)} for {X.shape[0]} samples")

    best_k = max(scores, key=lambda k: scores[k])
    return SilhouetteSearchResult(best_k=best_k, best_score=scores[best_k], scores=scores)


def run_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """Fit KMeans and return labels + fitted model.

    Parameters
    ----------
    X : (n_samples, n_components) data to cluster.
    k : Number of clusters.
    random_state : Reproducibility seed.

    Returns
    -------
    (labels, fitted_model) where labels is (n_samples,) int array.
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = model.fit_predict(X)
    return np.asarray(labels), model


def get_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Count the number of points in each cluster.

    Parameters
    ----------
    labels : (n_samples,) cluster assignments.

    Returns
    -------
    Mapping from cluster ID to count, sorted by cluster ID.
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(sorted(zip(unique.tolist(), counts.tolist())))
