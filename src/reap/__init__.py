"""REAP: Reproducible Embedding via Averaged Projection.

Stabilizes stochastic dimensionality reduction through distance-matrix
consensus and neural projection. The core algorithm averages pairwise
distance matrices across multi-seed UMAP runs, producing embeddings that
are invariant to rotation/reflection ambiguity.

Quick start::

    from reap import run_consensus_pipeline
    embedding, metadata = run_consensus_pipeline(
        X=embeddings,           # (n_samples, 384) sentence embeddings
        seeds=list(range(30)),  # 30 random seeds
        n_components=18,        # target dimensions
        n_neighbors=19,         # UMAP neighborhood size
        min_dist=0.005,         # minimum distance
    )
"""

__version__ = "0.1.0"

from reap.clustering import find_best_k, get_cluster_sizes, run_kmeans
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_consensus,
    run_consensus_pipeline,
)
from reap.evaluation import (
    compute_distance_correlation,
    compute_pairwise_ari,
    compute_seed_stability,
    compute_silhouette,
    compute_trustworthiness,
)
from reap.validation import validate_cluster_sizes, validate_embeddings

__all__ = [
    # Consensus (core)
    "get_multi_seed_embeddings",
    "get_consensus_distance_matrix",
    "get_consensus_embedding",
    "get_procrustes_consensus",
    "run_consensus_pipeline",
    # Clustering
    "find_best_k",
    "run_kmeans",
    "get_cluster_sizes",
    # Evaluation
    "compute_trustworthiness",
    "compute_silhouette",
    "compute_pairwise_ari",
    "compute_seed_stability",
    "compute_distance_correlation",
    # Validation
    "validate_embeddings",
    "validate_cluster_sizes",
]
