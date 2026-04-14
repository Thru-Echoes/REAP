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

from reap.benchmarks import (
    BenchmarkResult,
    BootstrapCI,
    MethodResult,
    compute_bootstrap_ci,
    run_benchmark,
)
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
from reap.labeling import (
    label_clusters_combined,
    label_clusters_ctfidf,
    label_clusters_llm,
    stratify_points,
)
from reap.reporting import (
    benchmark_to_csv,
    benchmark_to_latex,
    benchmark_to_markdown,
    cross_dataset_latex,
    cross_dataset_table,
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
    # Labeling (c-TF-IDF → LLM → human validation)
    "label_clusters_ctfidf",
    "label_clusters_llm",
    "label_clusters_combined",
    "stratify_points",
    # Benchmarking
    "run_benchmark",
    "compute_bootstrap_ci",
    "BenchmarkResult",
    "MethodResult",
    "BootstrapCI",
    # Reporting
    "benchmark_to_markdown",
    "benchmark_to_latex",
    "benchmark_to_csv",
    "cross_dataset_table",
    "cross_dataset_latex",
    # Validation
    "validate_embeddings",
    "validate_cluster_sizes",
]
