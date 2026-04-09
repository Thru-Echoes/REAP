"""Tests for clustering utilities."""

import numpy as np

from reap.clustering import find_best_k, get_cluster_sizes, run_kmeans


class TestFindBestK:
    def test_returns_valid_k(self) -> None:
        X = np.random.default_rng(42).standard_normal((100, 5))
        result = find_best_k(X, k_range=[2, 3, 4, 5])
        assert result.best_k in [2, 3, 4, 5]
        assert -1.0 <= result.best_score <= 1.0
        assert len(result.scores) == 4

    def test_well_separated_finds_correct_k(self) -> None:
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
        result = find_best_k(X, k_range=[2, 3, 4, 5])
        assert result.best_k == 3


class TestRunKMeans:
    def test_returns_correct_labels(self) -> None:
        X = np.random.default_rng(42).standard_normal((50, 3))
        labels, model = run_kmeans(X, k=4)
        assert labels.shape == (50,)
        assert len(set(labels.tolist())) == 4
        assert hasattr(model, "cluster_centers_")


class TestClusterSizes:
    def test_sums_to_total(self) -> None:
        labels = np.array([0, 0, 1, 1, 1, 2])
        sizes = get_cluster_sizes(labels)
        assert sum(sizes.values()) == 6
        assert sizes == {0: 2, 1: 3, 2: 1}
