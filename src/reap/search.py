"""Grid search framework for REAP hyperparameter optimization.

Implements progressive multi-round grid search with multi-criteria selection.
Selection criteria include equal-weight, cluster-focused, stability-focused,
structure-focused, and Pareto-optimal strategies.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans

from reap._types import GridSearchConfig, GridSearchResult
from reap.clustering import find_best_k
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.evaluation import compute_pairwise_ari, compute_trustworthiness

logger = logging.getLogger(__name__)


def evaluate_config(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int] | range,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
) -> GridSearchResult:
    """Evaluate a single UMAP configuration with full consensus pipeline.

    Runs multi-seed UMAP, computes distance-matrix consensus, finds best K,
    and measures trustworthiness + stability.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    seeds : Random seeds for multi-seed UMAP.
    n_components : Target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : Minimum distance parameter.
    k_range : K values to evaluate for clustering.
    metric : Input space metric.
    silhouette_metric : Metric for silhouette in UMAP output space.

    Returns
    -------
    GridSearchResult with all metrics.

    Side effects: logging, UMAP internal state.
    """
    # Multi-seed UMAP
    umap_embeddings = get_multi_seed_embeddings(
        X, seeds, n_components, n_neighbors, min_dist, metric
    )

    # Distance-matrix consensus
    D_consensus = get_consensus_distance_matrix(umap_embeddings)
    consensus_emb, _ = get_consensus_embedding(
        D_consensus, n_components, n_neighbors, min_dist
    )

    # Trustworthiness
    n_nn = min(n_neighbors, X.shape[0] - 1)
    tw = compute_trustworthiness(
        X, consensus_emb, n_neighbors=n_nn, metric=metric
    )

    # Best K via silhouette
    sil_result = find_best_k(consensus_emb, k_range, metric=silhouette_metric)

    # Per-seed cluster labels for stability
    seed_labels: list[np.ndarray] = []
    seed_k_values: list[int] = []
    for emb in umap_embeddings:
        seed_sil = find_best_k(emb, k_range, metric=silhouette_metric)
        seed_k_values.append(seed_sil.best_k)
        labels = KMeans(n_clusters=seed_sil.best_k, random_state=42, n_init="auto").fit_predict(emb)
        seed_labels.append(labels)

    # Seed-to-seed ARI
    ari_matrix = compute_pairwise_ari(seed_labels)
    triu = ari_matrix[np.triu_indices_from(ari_matrix, k=1)]

    # Seed-to-consensus ARI
    consensus_labels = KMeans(
        n_clusters=sil_result.best_k, random_state=42, n_init="auto"
    ).fit_predict(consensus_emb)
    from sklearn.metrics import adjusted_rand_score
    s2c_aris = [
        adjusted_rand_score(sl, consensus_labels) for sl in seed_labels
    ]

    # Seed K statistics
    seed_k_arr = np.array(seed_k_values)
    from scipy import stats as sp_stats
    mode_result = sp_stats.mode(seed_k_arr, keepdims=True)

    return GridSearchResult(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        trustworthiness=tw,
        best_k=sil_result.best_k,
        best_silhouette=sil_result.best_score,
        mean_ari_seeds=float(triu.mean()),
        std_ari_seeds=float(triu.std()),
        mean_ari_consensus=float(np.mean(s2c_aris)),
        std_ari_consensus=float(np.std(s2c_aris)),
        seed_k_mode=int(mode_result.mode[0]),
        seed_k_agree=int(np.sum(seed_k_arr == sil_result.best_k)),
    )


def grid_search(
    X: np.ndarray,
    config: GridSearchConfig,
) -> list[GridSearchResult]:
    """Run full grid search over UMAP hyperparameters.

    Parameters
    ----------
    X : (n_samples, n_features) input embeddings.
    config : Grid search configuration with parameter lists.

    Returns
    -------
    List of GridSearchResult, one per configuration.

    Side effects: logging, UMAP internal state.
    """
    if config.seeds is not None:
        seeds = config.seeds
    else:
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 10000, size=config.n_seeds).tolist()

    total = (
        len(config.n_components_list)
        * len(config.n_neighbors_list)
        * len(config.min_dist_list)
    )
    logger.info("Grid search: %d configs × %d seeds", total, len(seeds))

    results: list[GridSearchResult] = []
    for nc in config.n_components_list:
        for nn in config.n_neighbors_list:
            for md in config.min_dist_list:
                idx = len(results) + 1
                logger.info("Config %d/%d: d=%d, nn=%d, md=%.4f", idx, total, nc, nn, md)
                result = evaluate_config(
                    X, seeds, nc, nn, md, config.k_range, config.metric,
                )
                results.append(result)

    logger.info("Grid search complete: %d results", len(results))
    return results


def select_best_config(
    results: list[GridSearchResult],
    strategy: str = "equal_weight",
) -> GridSearchResult:
    """Select the best configuration using a named strategy.

    Strategies:
    - "equal_weight": Equal weight to TW, Sil, ARI.
    - "cluster_focused": TW=0.2, Sil=0.4, ARI=0.4.
    - "stability_focused": TW=0.2, Sil=0.2, ARI=0.6.
    - "structure_focused": TW=0.6, Sil=0.2, ARI=0.2.
    - "pareto": Non-dominated set, then equal-weight tiebreak.

    Parameters
    ----------
    results : Grid search results.
    strategy : Selection strategy name.

    Returns
    -------
    The best GridSearchResult.
    """
    weights: dict[str, tuple[float, float, float]] = {
        "equal_weight": (1 / 3, 1 / 3, 1 / 3),
        "cluster_focused": (0.2, 0.4, 0.4),
        "stability_focused": (0.2, 0.2, 0.6),
        "structure_focused": (0.6, 0.2, 0.2),
    }

    def _score(r: GridSearchResult, w: tuple[float, float, float]) -> float:
        return w[0] * r.trustworthiness + w[1] * r.best_silhouette + w[2] * r.mean_ari_seeds

    if strategy == "pareto":
        # Find Pareto-optimal set
        pareto: list[GridSearchResult] = []
        for r in results:
            dominated = False
            for other in results:
                if (
                    other.trustworthiness >= r.trustworthiness
                    and other.best_silhouette >= r.best_silhouette
                    and other.mean_ari_seeds >= r.mean_ari_seeds
                    and (
                        other.trustworthiness > r.trustworthiness
                        or other.best_silhouette > r.best_silhouette
                        or other.mean_ari_seeds > r.mean_ari_seeds
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return max(pareto, key=lambda r: _score(r, (1 / 3, 1 / 3, 1 / 3)))

    if strategy not in weights:
        raise ValueError(f"Unknown strategy: {strategy!r}. Options: {list(weights) + ['pareto']}")

    w = weights[strategy]
    return max(results, key=lambda r: _score(r, w))
