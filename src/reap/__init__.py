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

Benchmark + metrics entry points::

    from reap import run_benchmark, run_benchmark_with_artifacts
    from reap import compare_family, compute_seed_bootstrap_ci
"""

__version__ = "0.1.0"

from reap.analysis import (
    assign_by_nearest_centroid,
    compute_baserate_enrichment,
    compute_cluster_centroids,
    compute_contingency_cramers_v,
    compute_demographic_contingency,
    compute_kl_divergence,
    compute_knn_purity,
    compute_median_centroid_cosine,
    compute_procrustes_disparity,
    compute_source_separability_silhouette,
    compute_temporal_alignment,
    compute_theme_cluster_contingency,
)
from reap.benchmarks import (
    ALL_METHODS,
    BenchmarkArtifacts,
    BenchmarkResult,
    BootstrapCI,
    MethodResult,
    compute_bootstrap_ci,
    compute_subsample_ci,
    run_benchmark,
    run_benchmark_with_artifacts,
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
    compute_ami,
    compute_calinski_harabasz,
    compute_cluster_persistence,
    compute_cluster_purity,
    compute_continuity,
    compute_davies_bouldin,
    compute_distance_correlation,
    compute_external_validity,
    compute_lcmc,
    compute_nmi,
    compute_pairwise_ami,
    compute_pairwise_ari,
    compute_pairwise_nmi,
    compute_r_nx_auc,
    compute_r_nx_curve,
    compute_seed_stability,
    compute_silhouette,
    compute_silhouette_samples,
    compute_topic_coherence,
    compute_topic_diversity,
    compute_topic_exclusivity,
    compute_trustworthiness,
    compute_variation_of_information,
)
from reap.filter import (
    DEFAULT_ALPHA as FILTER_DEFAULT_ALPHA,
)
from reap.filter import (
    DEFAULT_FALLBACK_N_THRESHOLD as FILTER_DEFAULT_FALLBACK_N_THRESHOLD,
)
from reap.filter import (
    DEFAULT_RIDGE as FILTER_DEFAULT_RIDGE,
)
from reap.filter import (
    METHOD_MAHALANOBIS_NO_CORRECTION,
    METHOD_MAHALANOBIS_POOLED,
    FilterCalibration,
)
from reap.filter import (
    apply as apply_filter,
)
from reap.filter import (
    calibrate as calibrate_filter,
)
from reap.filter import (
    calibrate_per_subgroup as calibrate_filter_per_subgroup,
)
from reap.filter import (
    compute_mahalanobis_d2 as compute_filter_mahalanobis_d2,
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
from reap.statistics import (
    BootstrapCIResult,
    FamilyComparisonResult,
    PairedTestResult,
    benjamini_hochberg,
    cliffs_delta,
    compare_family,
    compute_seed_bootstrap_ci,
    holm_bonferroni,
    paired_cohens_d,
    paired_comparison,
    paired_wilcoxon,
)
from reap.validation import validate_cluster_sizes, validate_embeddings

__all__ = [
    # Analysis (cross-source + temporal helpers)
    "compute_kl_divergence",
    "compute_cluster_centroids",
    "assign_by_nearest_centroid",
    "compute_knn_purity",
    "compute_median_centroid_cosine",
    "compute_contingency_cramers_v",
    "compute_theme_cluster_contingency",
    "compute_source_separability_silhouette",
    "compute_temporal_alignment",
    "compute_procrustes_disparity",
    "compute_demographic_contingency",
    "compute_baserate_enrichment",
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
    # Evaluation — structure preservation
    "compute_trustworthiness",
    "compute_continuity",
    "compute_lcmc",
    "compute_r_nx_curve",
    "compute_r_nx_auc",
    # Evaluation — cluster quality
    "compute_silhouette",
    "compute_silhouette_samples",
    "compute_davies_bouldin",
    "compute_calinski_harabasz",
    # Evaluation — clustering agreement
    "compute_pairwise_ari",
    "compute_pairwise_ami",
    "compute_pairwise_nmi",
    "compute_ami",
    "compute_nmi",
    "compute_variation_of_information",
    "compute_seed_stability",
    "compute_distance_correlation",
    # Evaluation — external validity
    "compute_external_validity",
    "compute_cluster_purity",
    # Evaluation — topic + persistence
    "compute_topic_coherence",
    "compute_topic_diversity",
    "compute_topic_exclusivity",
    "compute_cluster_persistence",
    # Labeling
    "label_clusters_ctfidf",
    "label_clusters_llm",
    "label_clusters_combined",
    "stratify_points",
    # Benchmarking
    "run_benchmark",
    "run_benchmark_with_artifacts",
    "compute_bootstrap_ci",
    "compute_subsample_ci",
    "BenchmarkResult",
    "BenchmarkArtifacts",
    "MethodResult",
    "BootstrapCI",
    "ALL_METHODS",
    # Statistics
    "compute_seed_bootstrap_ci",
    "paired_wilcoxon",
    "paired_cohens_d",
    "cliffs_delta",
    "paired_comparison",
    "compare_family",
    "holm_bonferroni",
    "benjamini_hochberg",
    "BootstrapCIResult",
    "PairedTestResult",
    "FamilyComparisonResult",
    # Reporting
    "benchmark_to_markdown",
    "benchmark_to_latex",
    "benchmark_to_csv",
    "cross_dataset_table",
    "cross_dataset_latex",
    # Validation
    "validate_embeddings",
    "validate_cluster_sizes",
    # OOS conformal filter (§3.6)
    "FilterCalibration",
    "calibrate_filter",
    "calibrate_filter_per_subgroup",
    "apply_filter",
    "compute_filter_mahalanobis_d2",
    "METHOD_MAHALANOBIS_POOLED",
    "METHOD_MAHALANOBIS_NO_CORRECTION",
    "FILTER_DEFAULT_ALPHA",
    "FILTER_DEFAULT_RIDGE",
    "FILTER_DEFAULT_FALLBACK_N_THRESHOLD",
]
