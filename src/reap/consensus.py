"""Core REAP consensus algorithm: distance-matrix averaging for stable dimensionality reduction.

The key insight is that UMAP embeddings are rotationally and reflectively
arbitrary — different random seeds produce geometrically equivalent solutions
differing by rigid transforms. Averaging coordinates (even after Procrustes
alignment) destroys structure. Averaging *pairwise distance matrices* is
invariant to these transforms and preserves relational geometry.

Method comparisons against Procrustes and the other baselines run under the
pre-registered benchmark protocol; the numbers live in committed result
artifacts, not in docstrings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist, squareform
from umap import UMAP

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-seed UMAP
# ---------------------------------------------------------------------------


def get_multi_seed_embeddings(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
) -> list[np.ndarray]:
    """Run UMAP independently with each seed, returning one embedding per seed.

    Pure function. No side effects beyond UMAP internal state.

    Parameters
    ----------
    X : (n_samples, n_features) input data.
    seeds : Random seeds — one UMAP run per seed.
    n_components : Target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : Minimum distance between embedded points.
    metric : Distance metric for the input space.

    Returns
    -------
    list of (n_samples, n_components) arrays, one per seed.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-d input, got shape {X.shape}")
    if len(seeds) < 2:
        raise ValueError(f"Need at least 2 seeds for consensus, got {len(seeds)}")

    embeddings: list[np.ndarray] = []
    for i, seed in enumerate(seeds):
        logger.info("UMAP seed %d/%d (seed=%d)", i + 1, len(seeds), seed)
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        result = reducer.fit_transform(X)
        # float32 is sufficient for individual UMAP embeddings (saves memory);
        # distance matrix accumulation uses float64 for numerical precision.
        embeddings.append(np.asarray(result, dtype=np.float32))

    return embeddings


# ---------------------------------------------------------------------------
# Distance-matrix consensus (PRIMARY METHOD)
# ---------------------------------------------------------------------------


def get_consensus_distance_matrix(
    umap_embeddings: list[np.ndarray],
    metric: str = "euclidean",
) -> np.ndarray:
    """Average pairwise distance matrices across multi-seed UMAP embeddings.

    This is the core REAP contribution. Distance matrices are invariant to the
    rotation/reflection ambiguity of UMAP, so averaging them preserves
    relational structure that coordinate averaging destroys.

    The average of Euclidean distance matrices satisfies the triangle inequality
    (convex combination of metrics is a metric).

    Pure function. No side effects.

    Parameters
    ----------
    umap_embeddings : List of (n_samples, n_components) arrays from different seeds.
    metric : Distance metric in the low-dimensional space (default: euclidean,
             which is correct for UMAP output per McInnes 2018).

    Returns
    -------
    (n_samples, n_samples) symmetric consensus distance matrix.
    """
    if len(umap_embeddings) == 0:
        raise ValueError("umap_embeddings is empty")

    n = umap_embeddings[0].shape[0]
    D_sum = np.zeros((n, n), dtype=np.float64)

    for i, E in enumerate(umap_embeddings):
        if E.shape[0] != n:
            raise ValueError(
                f"Embedding {i} has {E.shape[0]} samples, expected {n}"
            )
        D = squareform(pdist(E, metric=metric))  # type: ignore[arg-type]
        D_sum += D

    D_consensus = D_sum / len(umap_embeddings)
    np.fill_diagonal(D_consensus, 0.0)
    return D_consensus


def get_consensus_embedding(
    D_consensus: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int = 42,
) -> tuple[np.ndarray, UMAP]:
    """Run UMAP on the precomputed consensus distance matrix.

    Side effects: UMAP internal random state.

    Parameters
    ----------
    D_consensus : (n_samples, n_samples) symmetric distance matrix.
    n_components : Target dimensionality of the final embedding.
    n_neighbors : UMAP neighborhood size.
    min_dist : Minimum distance between embedded points.
    random_state : Seed for the final UMAP projection.

    Returns
    -------
    (embedding, fitted_reducer) where embedding is (n_samples, n_components).
    """
    reducer = UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    raw = reducer.fit_transform(D_consensus)
    embedding: np.ndarray = np.asarray(raw, dtype=np.float64)
    return embedding, reducer  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Procrustes consensus (BASELINE — for comparison only)
# ---------------------------------------------------------------------------


def get_procrustes_aligned_embeddings(
    umap_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Align multi-seed UMAP embeddings to the first seed via orthogonal Procrustes.

    BASELINE METHOD ONLY — provided for comparison. Use distance-matrix
    consensus for production workloads.

    Pure function. No side effects.

    Parameters
    ----------
    umap_embeddings : List of (n_samples, n_components) arrays.

    Returns
    -------
    (n_seeds, n_samples, n_components) aligned embeddings.
    """
    from scipy.linalg import orthogonal_procrustes

    reference = umap_embeddings[0]
    ref_centered = reference - reference.mean(axis=0)

    aligned = [ref_centered]
    for emb in umap_embeddings[1:]:
        centered = emb - emb.mean(axis=0)
        R, _ = orthogonal_procrustes(centered, ref_centered)
        aligned.append(centered @ R)

    return np.stack(aligned)


def get_procrustes_consensus(
    umap_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Procrustes-align multi-seed embeddings and average coordinates.

    BASELINE METHOD ONLY — kept as a pre-registered comparison baseline.
    Use `get_consensus_distance_matrix()` + `get_consensus_embedding()` for
    production workloads.

    Pure function. No side effects.

    Returns
    -------
    (n_samples, n_components) consensus embedding.
    """
    aligned = get_procrustes_aligned_embeddings(umap_embeddings)
    return aligned.mean(axis=0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_consensus_pipeline(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
    method: str = "distance_matrix",
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Run the full REAP consensus pipeline: multi-seed UMAP → consensus → embedding.

    Parameters
    ----------
    X : (n_samples, n_features) input data (typically sentence embeddings).
    seeds : Random seeds for multi-seed UMAP.
    n_components : Target dimensionality.
    n_neighbors : UMAP neighborhood size.
    min_dist : Minimum distance parameter.
    metric : Distance metric for the input space (default: cosine).
    method : "distance_matrix" (recommended) or "procrustes" (baseline).
    random_state : Seed for the final projection step.

    Returns
    -------
    (consensus_embedding, metadata) where metadata includes intermediate results.

    Side effects: UMAP internal random state, logging.
    """
    logger.info(
        "REAP pipeline: method=%s, seeds=%d, d=%d, nn=%d, md=%.4f",
        method, len(seeds), n_components, n_neighbors, min_dist,
    )

    # Step 1: Multi-seed UMAP
    umap_embeddings = get_multi_seed_embeddings(
        X, seeds, n_components, n_neighbors, min_dist, metric
    )

    metadata: dict = {
        "method": method,
        "n_seeds": len(seeds),
        "n_samples": X.shape[0],
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "seed_embeddings": umap_embeddings,
    }

    # Step 2: Consensus
    if method == "distance_matrix":
        D_consensus = get_consensus_distance_matrix(umap_embeddings)
        embedding, reducer = get_consensus_embedding(
            D_consensus, n_components, n_neighbors, min_dist, random_state
        )
        metadata["distance_matrix"] = D_consensus
        metadata["reducer"] = reducer
    elif method == "procrustes":
        embedding = get_procrustes_consensus(umap_embeddings)
        metadata["aligned_embeddings"] = get_procrustes_aligned_embeddings(umap_embeddings)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'distance_matrix' or 'procrustes'.")

    metadata["consensus_embedding"] = embedding
    logger.info("REAP pipeline complete: output shape %s", embedding.shape)

    return embedding, metadata
