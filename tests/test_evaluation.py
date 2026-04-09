"""Tests for evaluation metrics."""

import numpy as np
import pytest

from reap.evaluation import (
    compute_distance_correlation,
    compute_pairwise_ari,
    compute_seed_stability,
    compute_silhouette,
    compute_trustworthiness,
)


class TestTrustworthiness:
    def test_perfect_preservation(self) -> None:
        """Identity projection should have TW = 1.0."""
        X = np.random.default_rng(42).standard_normal((50, 10))
        tw = compute_trustworthiness(X, X, n_neighbors=5, metric="euclidean")
        assert tw == pytest.approx(1.0)

    def test_bounded(self) -> None:
        X = np.random.default_rng(42).standard_normal((50, 10))
        X_low = X[:, :3]
        tw = compute_trustworthiness(X, X_low, n_neighbors=5, metric="euclidean")
        assert 0.0 <= tw <= 1.0


class TestSilhouette:
    def test_well_separated_clusters(self) -> None:
        X = np.vstack([
            np.random.default_rng(42).standard_normal((25, 3)) + [10, 0, 0],
            np.random.default_rng(43).standard_normal((25, 3)) + [-10, 0, 0],
        ])
        labels = np.array([0] * 25 + [1] * 25)
        sil = compute_silhouette(X, labels)
        assert sil > 0.5

    def test_single_label_returns_negative(self) -> None:
        X = np.random.default_rng(42).standard_normal((20, 3))
        labels = np.zeros(20, dtype=int)
        sil = compute_silhouette(X, labels)
        assert sil == -1.0


class TestPairwiseARI:
    def test_identical_labels(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        ari_mat = compute_pairwise_ari([labels, labels, labels])
        np.testing.assert_array_almost_equal(ari_mat, np.ones((3, 3)))

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        labels_list = [rng.integers(0, 3, 50) for _ in range(5)]
        ari_mat = compute_pairwise_ari(labels_list)
        np.testing.assert_array_almost_equal(ari_mat, ari_mat.T)

    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(42)
        labels_list = [rng.integers(0, 3, 50) for _ in range(4)]
        ari_mat = compute_pairwise_ari(labels_list)
        np.testing.assert_array_almost_equal(np.diag(ari_mat), np.ones(4))


class TestSeedStability:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(42)
        seed_labels = [rng.integers(0, 3, 50) for _ in range(5)]
        consensus = rng.integers(0, 3, 50)
        result = compute_seed_stability(seed_labels, consensus)
        assert "s2s_ari_mean" in result
        assert "s2c_ari_mean" in result


class TestDistanceCorrelation:
    def test_identical_gives_one(self) -> None:
        Y = np.random.default_rng(42).standard_normal((30, 5))
        assert compute_distance_correlation(Y, Y) == pytest.approx(1.0)

    def test_bounded(self) -> None:
        rng = np.random.default_rng(42)
        Y1 = rng.standard_normal((30, 5))
        Y2 = rng.standard_normal((30, 5))
        dc = compute_distance_correlation(Y1, Y2)
        assert -1.0 <= dc <= 1.0
