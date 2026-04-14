"""Tests for the c-TF-IDF cluster labeling module.

Tests cover: point stratification, c-TF-IDF matrix construction,
cluster labeling, and edge cases (small datasets, few clusters).
LLM-dependent tests are skipped if API keys are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from reap.labeling import build_ctfidf_matrix, label_clusters_ctfidf, stratify_points


@pytest.fixture
def clustered_text_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Synthetic text data with 3 clear clusters."""
    texts = [
        # Cluster 0: climate/environment
        "climate change is accelerating due to carbon emissions worldwide",
        "global warming impacts biodiversity and species extinction rates",
        "greenhouse gas reduction through renewable energy technology",
        "deforestation contributes to rising carbon dioxide levels",
        "environmental sustainability requires carbon neutral policies",
        # Cluster 1: technology/AI
        "artificial intelligence transforms machine learning algorithms",
        "neural networks deep learning revolutionize computer vision",
        "natural language processing enables chatbot applications",
        "quantum computing advances algorithmic efficiency dramatically",
        "cybersecurity threats require advanced artificial intelligence defense",
        # Cluster 2: economics/finance
        "stock market volatility reflects investor confidence metrics",
        "inflation monetary policy central bank interest rate decisions",
        "economic growth unemployment labor market indicators analysis",
        "cryptocurrency blockchain decentralized financial technology innovation",
        "global trade tariffs supply chain disruption economic impact",
    ]
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    rng = np.random.default_rng(42)
    X = rng.standard_normal((15, 10))
    # Make clusters separable
    X[labels == 0] += [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    X[labels == 1] += [0, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    X[labels == 2] += [0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
    return texts, labels, X


class TestStratifyPoints:
    """Tests for core/peripheral point stratification."""

    def test_returns_correct_count(self, clustered_text_data: tuple) -> None:
        _, labels, X = clustered_text_data
        points, summary = stratify_points(X, labels)
        assert len(points) == len(labels)

    def test_all_points_classified(self, clustered_text_data: tuple) -> None:
        _, labels, X = clustered_text_data
        points, summary = stratify_points(X, labels)
        assert summary.core_points + summary.peripheral_points == summary.total_points

    def test_high_threshold_all_peripheral(self, clustered_text_data: tuple) -> None:
        _, labels, X = clustered_text_data
        _, summary = stratify_points(X, labels, silhouette_threshold=2.0)
        assert summary.core_points == 0

    def test_low_threshold_all_core(self, clustered_text_data: tuple) -> None:
        _, labels, X = clustered_text_data
        _, summary = stratify_points(X, labels, silhouette_threshold=-2.0)
        assert summary.core_points == summary.total_points

    def test_point_has_correct_fields(self, clustered_text_data: tuple) -> None:
        _, labels, X = clustered_text_data
        points, _ = stratify_points(X, labels)
        p = points[0]
        assert hasattr(p, "point_index")
        assert hasattr(p, "cluster")
        assert hasattr(p, "silhouette_score")
        assert hasattr(p, "centroid_distance")
        assert p.stratum in ("core", "peripheral")


class TestBuildCtfidfMatrix:
    """Tests for c-TF-IDF matrix construction."""

    def test_correct_shape(self, clustered_text_data: tuple) -> None:
        texts, labels, _ = clustered_text_data
        ctfidf, features, cluster_ids = build_ctfidf_matrix(texts, labels)
        n_clusters = len(set(labels.tolist()))
        assert ctfidf.shape[0] == n_clusters
        assert ctfidf.shape[1] == len(features)
        assert len(cluster_ids) == n_clusters

    def test_non_negative(self, clustered_text_data: tuple) -> None:
        texts, labels, _ = clustered_text_data
        ctfidf, _, _ = build_ctfidf_matrix(texts, labels)
        assert np.all(ctfidf >= 0)

    def test_small_dataset_no_crash(self) -> None:
        """Regression test: min_df=2 with 2 clusters used to crash."""
        texts = ["climate carbon emissions", "machine learning algorithms"]
        labels = np.array([0, 1])
        ctfidf, features, cluster_ids = build_ctfidf_matrix(texts, labels, min_df=2)
        assert ctfidf.shape[0] == 2
        assert len(features) > 0

    def test_single_cluster_per_text(self) -> None:
        """Edge case: one text per cluster."""
        texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
        labels = np.array([0, 1, 2])
        ctfidf, features, _ = build_ctfidf_matrix(texts, labels, min_df=1)
        assert ctfidf.shape[0] == 3


class TestLabelClustersCTfidf:
    """Tests for the full c-TF-IDF labeling pipeline."""

    def test_returns_one_result_per_cluster(self, clustered_text_data: tuple) -> None:
        texts, labels, _ = clustered_text_data
        results = label_clusters_ctfidf(texts, labels)
        assert len(results) == len(set(labels.tolist()))

    def test_each_cluster_has_terms(self, clustered_text_data: tuple) -> None:
        texts, labels, _ = clustered_text_data
        results = label_clusters_ctfidf(texts, labels, top_n=5)
        for r in results:
            assert len(r.top_terms) > 0

    def test_discriminative_terms_differ_between_clusters(
        self, clustered_text_data: tuple
    ) -> None:
        texts, labels, _ = clustered_text_data
        results = label_clusters_ctfidf(texts, labels, top_n=5)
        # Top discriminative terms should differ between at least some clusters
        disc_sets = [
            {t.term for t in r.discriminative_terms[:3]} for r in results
        ]
        # Not all clusters share the same top-3 discriminative terms
        assert not all(s == disc_sets[0] for s in disc_sets)

    def test_correct_item_count(self, clustered_text_data: tuple) -> None:
        texts, labels, _ = clustered_text_data
        results = label_clusters_ctfidf(texts, labels)
        for r in results:
            expected = int((labels == r.cluster).sum())
            assert r.n_items == expected

    def test_discriminativeness_consistent(self, clustered_text_data: tuple) -> None:
        """Verify discriminativeness uses ratio-to-other-clusters metric for both
        top_terms and discriminative_terms."""
        texts, labels, _ = clustered_text_data
        results = label_clusters_ctfidf(texts, labels, top_n=5)
        for r in results:
            for t in r.top_terms:
                assert t.discriminativeness >= 0.0
            for t in r.discriminative_terms:
                assert t.discriminativeness >= 0.0

    def test_small_dataset_regression(self) -> None:
        """Two clusters with min_df=2 should not crash."""
        texts = [
            "renewable energy solar panels",
            "renewable wind power turbines",
            "stock market trading volume",
            "bond yield interest rates",
        ]
        labels = np.array([0, 0, 1, 1])
        results = label_clusters_ctfidf(texts, labels, min_df=2)
        assert len(results) == 2
