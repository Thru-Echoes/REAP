"""Tests for the core REAP consensus algorithm."""

import numpy as np
import pytest

from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_aligned_embeddings,
    get_procrustes_consensus,
    run_consensus_pipeline,
)


class TestMultiSeedEmbeddings:
    def test_returns_correct_count(self, small_blobs: dict, three_seeds: list[int]) -> None:
        result = get_multi_seed_embeddings(
            small_blobs["X"], three_seeds, n_components=3, n_neighbors=10, min_dist=0.1
        )
        assert len(result) == 3

    def test_correct_shape(self, small_blobs: dict, three_seeds: list[int]) -> None:
        result = get_multi_seed_embeddings(
            small_blobs["X"], three_seeds, n_components=3, n_neighbors=10, min_dist=0.1
        )
        for emb in result:
            assert emb.shape == (50, 3)

    def test_different_seeds_produce_different_embeddings(
        self, small_blobs: dict, three_seeds: list[int]
    ) -> None:
        result = get_multi_seed_embeddings(
            small_blobs["X"], three_seeds, n_components=3, n_neighbors=10, min_dist=0.1
        )
        # Embeddings differ (different random initializations)
        assert not np.allclose(result[0], result[1])

    def test_rejects_single_seed(self, small_blobs: dict) -> None:
        with pytest.raises(ValueError, match="at least 2 seeds"):
            get_multi_seed_embeddings(
                small_blobs["X"], [42], n_components=3, n_neighbors=10, min_dist=0.1
            )


class TestConsensusDistanceMatrix:
    def test_symmetric(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_zero_diagonal(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        np.testing.assert_array_equal(np.diag(D), 0.0)

    def test_non_negative(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        assert np.all(D >= 0)

    def test_triangle_inequality(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        """Average of metric distance matrices satisfies triangle inequality."""
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        n = D.shape[0]
        # Check a random subset of triples
        rng = np.random.default_rng(42)
        for _ in range(100):
            i, j, k = rng.choice(n, 3, replace=False)
            assert D[i, k] <= D[i, j] + D[j, k] + 1e-10

    def test_correct_shape(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        n = umap_embeddings_3d[0].shape[0]
        assert D.shape == (n, n)

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            get_consensus_distance_matrix([])

    def test_mismatched_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            get_consensus_distance_matrix([np.zeros((10, 3)), np.zeros((11, 3))])


class TestConsensusEmbedding:
    def test_correct_shape(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        emb, _ = get_consensus_embedding(D, n_components=3, n_neighbors=10, min_dist=0.1)
        assert emb.shape == (umap_embeddings_3d[0].shape[0], 3)

    def test_returns_fitted_reducer(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        D = get_consensus_distance_matrix(umap_embeddings_3d)
        _, reducer = get_consensus_embedding(D, n_components=3, n_neighbors=10, min_dist=0.1)
        assert hasattr(reducer, "embedding_")


class TestProcrustes:
    def test_aligned_shape(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        aligned = get_procrustes_aligned_embeddings(umap_embeddings_3d)
        assert aligned.shape == (3, 50, 3)

    def test_procrustes_consensus_shape(self, umap_embeddings_3d: list[np.ndarray]) -> None:
        consensus = get_procrustes_consensus(umap_embeddings_3d)
        assert consensus.shape == (50, 3)


class TestFullPipeline:
    def test_distance_matrix_method(self, small_blobs: dict, three_seeds: list[int]) -> None:
        emb, meta = run_consensus_pipeline(
            small_blobs["X"],
            seeds=three_seeds,
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            method="distance_matrix",
        )
        assert emb.shape == (50, 3)
        assert meta["method"] == "distance_matrix"
        assert "distance_matrix" in meta
        assert "seed_embeddings" in meta

    def test_procrustes_method(self, small_blobs: dict, three_seeds: list[int]) -> None:
        emb, meta = run_consensus_pipeline(
            small_blobs["X"],
            seeds=three_seeds,
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            method="procrustes",
        )
        assert emb.shape == (50, 3)
        assert meta["method"] == "procrustes"

    def test_unknown_method_raises(self, small_blobs: dict, three_seeds: list[int]) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            run_consensus_pipeline(
                small_blobs["X"],
                seeds=three_seeds,
                n_components=3,
                n_neighbors=10,
                min_dist=0.1,
                method="bogus",
            )
