"""Tests for data validation gates."""

import numpy as np

from reap.validation import validate_cluster_sizes, validate_embeddings


class TestValidateEmbeddings:
    def test_valid_embeddings_pass(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 384))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        report = validate_embeddings(X, expected_dim=384)
        assert report.passed
        assert len(report.errors) == 0

    def test_nan_detected(self) -> None:
        X = np.ones((10, 5))
        X[3, 2] = np.nan
        report = validate_embeddings(X, check_l2_norm=False, check_duplicates=False)
        assert not report.passed
        assert any("NaN" in e for e in report.errors)

    def test_wrong_dim_detected(self) -> None:
        X = np.ones((10, 100))
        report = validate_embeddings(X, expected_dim=384, check_l2_norm=False, check_duplicates=False)
        assert not report.passed

    def test_duplicate_detection(self) -> None:
        X = np.ones((10, 5))  # All identical
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        report = validate_embeddings(X)
        assert any("identical" in e.lower() for e in report.errors)


class TestValidateClusterSizes:
    def test_valid_clusters(self) -> None:
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        report = validate_cluster_sizes(labels, min_cluster_size=10)
        assert report.passed
        assert report.stats["n_clusters"] == 3

    def test_small_cluster_warned(self) -> None:
        labels = np.array([0] * 50 + [1] * 3)
        report = validate_cluster_sizes(labels, min_cluster_size=10)
        assert len(report.warnings) > 0
