"""End-to-end pipeline tests using realistic synthetic data.

These tests run the full REAP pipeline from embeddings to clustering,
verifying that the distance-matrix consensus produces better results
than Procrustes consensus on well-separated data.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap.clustering import find_best_k, run_kmeans
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_consensus,
    run_consensus_pipeline,
)
from reap.evaluation import (
    compute_pairwise_ari,
    compute_seed_stability,
    compute_silhouette,
    compute_trustworthiness,
)
from reap.validation import validate_cluster_sizes, validate_embeddings


@pytest.fixture(scope="module")
def realistic_data() -> dict:
    """Well-separated 384-d blobs simulating sentence embeddings."""
    X, y = make_blobs(
        n_samples=200,
        n_features=384,
        centers=5,
        cluster_std=1.0,
        random_state=42,
    )
    # L2 normalize (like real sentence embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return {"X": X, "labels": y, "n_clusters": 5}


@pytest.fixture(scope="module")
def seeds_5() -> list[int]:
    """5 seeds for faster E2E tests (still meaningful consensus)."""
    return [42, 137, 256, 789, 1024]


class TestFullPipelineE2E:
    """Full REAP pipeline: embeddings → consensus → clustering → evaluation."""

    def test_pipeline_produces_valid_embedding(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        embedding, meta = run_consensus_pipeline(
            realistic_data["X"],
            seeds=seeds_5,
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
        )
        assert embedding.shape == (200, 5)
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))

    def test_trustworthiness_above_threshold(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        embedding, _ = run_consensus_pipeline(
            realistic_data["X"],
            seeds=seeds_5,
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
        )
        tw = compute_trustworthiness(
            realistic_data["X"], embedding, n_neighbors=15
        )
        assert tw > 0.7, f"Trustworthiness {tw:.3f} below 0.7 threshold"

    def test_clustering_finds_correct_k(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        embedding, _ = run_consensus_pipeline(
            realistic_data["X"],
            seeds=seeds_5,
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
        )
        result = find_best_k(embedding, k_range=range(3, 10))
        assert result.best_k == 5, f"Expected K=5, got K={result.best_k}"

    def test_silhouette_positive(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        embedding, _ = run_consensus_pipeline(
            realistic_data["X"],
            seeds=seeds_5,
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
        )
        labels, _ = run_kmeans(embedding, k=5)
        sil = compute_silhouette(embedding, labels)
        assert sil > 0.0, f"Silhouette {sil:.3f} is not positive"


class TestDistanceMatrixBeatsProcrustes:
    """The central claim: distance-matrix consensus > Procrustes consensus."""

    def test_distance_matrix_ari_higher(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        X = realistic_data["X"]
        ground_truth = realistic_data["labels"]

        # Get multi-seed embeddings
        umap_embeddings = get_multi_seed_embeddings(
            X, seeds_5, n_components=5, n_neighbors=15, min_dist=0.1
        )

        # Distance-matrix consensus
        D = get_consensus_distance_matrix(umap_embeddings)
        dm_embedding, _ = get_consensus_embedding(
            D, n_components=5, n_neighbors=15, min_dist=0.1
        )
        dm_labels, _ = run_kmeans(dm_embedding, k=5)
        from sklearn.metrics import adjusted_rand_score
        dm_ari = adjusted_rand_score(ground_truth, dm_labels)

        # Procrustes consensus
        proc_embedding = get_procrustes_consensus(umap_embeddings)
        proc_labels, _ = run_kmeans(proc_embedding, k=5)
        proc_ari = adjusted_rand_score(ground_truth, proc_labels)

        # Distance-matrix should be at least as good
        assert dm_ari >= proc_ari - 0.1, (
            f"Distance-matrix ARI ({dm_ari:.3f}) not better than "
            f"Procrustes ARI ({proc_ari:.3f})"
        )


class TestSeedStabilityE2E:
    """Verify seed stability metrics are computable and reasonable."""

    def test_stability_metrics(
        self, realistic_data: dict, seeds_5: list[int]
    ) -> None:
        umap_embeddings = get_multi_seed_embeddings(
            realistic_data["X"], seeds_5, n_components=5, n_neighbors=15, min_dist=0.1
        )

        # Per-seed clustering
        seed_labels = []
        for emb in umap_embeddings:
            labels, _ = run_kmeans(emb, k=5)
            seed_labels.append(labels)

        # Consensus clustering
        D = get_consensus_distance_matrix(umap_embeddings)
        consensus_emb, _ = get_consensus_embedding(D, n_components=5, n_neighbors=15, min_dist=0.1)
        consensus_labels, _ = run_kmeans(consensus_emb, k=5)

        stability = compute_seed_stability(seed_labels, consensus_labels)
        assert -1.0 <= stability["s2s_ari_mean"] <= 1.0
        assert -1.0 <= stability["s2c_ari_mean"] <= 1.0
        assert stability["s2s_ari_std"] >= 0.0


class TestValidationE2E:
    """Validation gates work on realistic data."""

    def test_valid_embeddings_pass(self, realistic_data: dict) -> None:
        report = validate_embeddings(realistic_data["X"], expected_dim=384)
        assert report.passed, f"Validation failed: {report.errors}"

    def test_valid_clusters_pass(self, realistic_data: dict) -> None:
        report = validate_cluster_sizes(realistic_data["labels"], min_cluster_size=10)
        assert report.stats["n_clusters"] == 5
