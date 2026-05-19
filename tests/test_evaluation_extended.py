"""Tests for the extended evaluation metrics added for the paper benchmark.

Covers: continuity, LCMC, R_NX curve + AUC, Davies-Bouldin, Calinski-Harabasz,
silhouette_samples, AMI/NMI, Variation of Information, pairwise AMI/NMI,
external-validity suite, topic diversity/exclusivity, cluster persistence.

Uses synthetic blobs so tests are fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap.evaluation import (
    compute_ami,
    compute_calinski_harabasz,
    compute_cluster_persistence,
    compute_cluster_purity,
    compute_continuity,
    compute_davies_bouldin,
    compute_external_validity,
    compute_lcmc,
    compute_nmi,
    compute_pairwise_ami,
    compute_pairwise_ari,
    compute_pairwise_nmi,
    compute_r_nx_auc,
    compute_r_nx_curve,
    compute_silhouette_samples,
    compute_topic_diversity,
    compute_topic_exclusivity,
    compute_trustworthiness,
    compute_variation_of_information,
)


@pytest.fixture(scope="module")
def blobs_high_low() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-dim signal blobs projected to high-dim ambient space + ground-truth labels."""
    X_hi, y = make_blobs(
        n_samples=120, n_features=32, centers=4, cluster_std=0.6, random_state=42,
    )
    # Treat the first 4 features as the "low-dim projection"
    X_lo = X_hi[:, :4].astype(np.float64)
    return X_hi, X_lo, y


class TestContinuity:
    def test_range_in_zero_one(self, blobs_high_low):
        X_hi, X_lo, _ = blobs_high_low
        c = compute_continuity(X_hi, X_lo, n_neighbors=10, metric="euclidean")
        assert 0.0 <= c <= 1.0

    def test_identity_is_perfect(self, blobs_high_low):
        _, X_lo, _ = blobs_high_low
        c = compute_continuity(X_lo, X_lo, n_neighbors=10, metric="euclidean")
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_complements_trustworthiness_asymmetric(self, blobs_high_low):
        """Trustworthiness and continuity are in general different."""
        X_hi, X_lo, _ = blobs_high_low
        tw = compute_trustworthiness(X_hi, X_lo, n_neighbors=10, metric="euclidean")
        cn = compute_continuity(X_hi, X_lo, n_neighbors=10, metric="euclidean")
        # Both in [0, 1], but not necessarily equal
        assert 0.0 <= tw <= 1.0
        assert 0.0 <= cn <= 1.0


class TestLCMCAndRNX:
    def test_lcmc_identity_near_unity(self, blobs_high_low):
        _, X_lo, _ = blobs_high_low
        lcmc = compute_lcmc(X_lo, X_lo, n_neighbors=15, metric="euclidean")
        # Identity gives q_nx = 1, so lcmc = 1 - k/(N-1)
        assert lcmc > 0.8

    def test_rnx_curve_shape(self, blobs_high_low):
        X_hi, X_lo, _ = blobs_high_low
        k_values, rnx = compute_r_nx_curve(X_hi, X_lo, k_max=20, metric="euclidean")
        assert k_values.shape == (20,)
        assert rnx.shape == (20,)
        # R_NX should be bounded roughly in [0, 1]; allow minor finite-sample slack
        assert np.all(rnx >= -0.1) and np.all(rnx <= 1.1)

    def test_rnx_auc_returns_float(self, blobs_high_low):
        X_hi, X_lo, _ = blobs_high_low
        auc = compute_r_nx_auc(X_hi, X_lo, k_max=20, metric="euclidean")
        assert isinstance(auc, float)


class TestClusterQualityMetrics:
    def test_davies_bouldin_positive(self, blobs_high_low):
        _, X_lo, y = blobs_high_low
        db = compute_davies_bouldin(X_lo, y)
        assert db > 0.0 and np.isfinite(db)

    def test_davies_bouldin_single_cluster_returns_inf(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        labels = np.zeros(20, dtype=int)
        assert compute_davies_bouldin(X, labels) == float("inf")

    def test_calinski_harabasz_positive(self, blobs_high_low):
        _, X_lo, y = blobs_high_low
        ch = compute_calinski_harabasz(X_lo, y)
        assert ch > 0.0

    def test_calinski_harabasz_single_cluster_zero(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        labels = np.zeros(20, dtype=int)
        assert compute_calinski_harabasz(X, labels) == 0.0

    def test_silhouette_samples_shape(self, blobs_high_low):
        _, X_lo, y = blobs_high_low
        sil = compute_silhouette_samples(X_lo, y, metric="euclidean")
        assert sil.shape == (len(y),)
        assert np.all(sil >= -1.0) and np.all(sil <= 1.0)


class TestAMIAndNMI:
    def test_ami_self_is_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert compute_ami(labels, labels) == pytest.approx(1.0, abs=1e-9)

    def test_nmi_self_is_one(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert compute_nmi(labels, labels) == pytest.approx(1.0, abs=1e-9)

    def test_random_gives_near_zero_ami(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 3, size=1000)
        b = rng.integers(0, 3, size=1000)
        assert abs(compute_ami(a, b)) < 0.05

    def test_ami_symmetric(self):
        rng = np.random.default_rng(1)
        a = rng.integers(0, 3, size=200)
        b = rng.integers(0, 3, size=200)
        assert compute_ami(a, b) == pytest.approx(compute_ami(b, a), abs=1e-12)


class TestVariationOfInformation:
    def test_vi_self_is_zero(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert compute_variation_of_information(labels, labels) == pytest.approx(0.0, abs=1e-9)

    def test_vi_nonnegative(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 1, 0, 1, 0, 1])
        vi = compute_variation_of_information(a, b)
        assert vi >= 0.0

    def test_vi_normalized_in_unit(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 1, 0, 1, 0, 1])
        nvi = compute_variation_of_information(a, b, normalize=True)
        assert 0.0 <= nvi <= 1.0


class TestPairwiseMatrices:
    def test_pairwise_ari_ami_nmi_square(self):
        labels_list = [
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([0, 0, 1, 1, 2, 2]),
        ]
        ari = compute_pairwise_ari(labels_list)
        ami = compute_pairwise_ami(labels_list)
        nmi = compute_pairwise_nmi(labels_list)
        for m in (ari, ami, nmi):
            assert m.shape == (3, 3)
            assert np.allclose(np.diag(m), 1.0)
            assert np.allclose(m, m.T)


class TestExternalValidity:
    def test_external_validity_perfect(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        suite = compute_external_validity(labels, labels)
        for k in ("ari", "ami", "nmi", "homogeneity", "completeness", "v_measure",
                  "fowlkes_mallows", "mean_purity"):
            assert suite[k] == pytest.approx(1.0, abs=1e-9), f"{k} not 1.0"

    def test_purity_breakdown(self):
        true = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 1, 1, 2, 2])  # cluster 0: 100% true-0, cluster 1: 50/50, cluster 2: 100% true-1
        mean_p, per_c = compute_cluster_purity(true, pred)
        assert per_c[0] == pytest.approx(1.0)
        assert per_c[1] == pytest.approx(0.5)
        assert per_c[2] == pytest.approx(1.0)
        assert mean_p == pytest.approx((2 + 1 + 2) / 6)


class TestTopicDiversity:
    def test_diversity_all_unique(self):
        topics = [["a", "b", "c"], ["d", "e", "f"]]
        assert compute_topic_diversity(topics, top_n=3) == pytest.approx(1.0)

    def test_diversity_all_identical(self):
        topics = [["a", "b", "c"], ["a", "b", "c"]]
        assert compute_topic_diversity(topics, top_n=3) == pytest.approx(0.5)

    def test_diversity_empty(self):
        assert compute_topic_diversity([], top_n=10) == 0.0


class TestTopicExclusivity:
    def test_exclusivity_nonnegative(self):
        texts = [
            "apple banana cherry",
            "apple banana",
            "date elderberry fig",
            "date elderberry",
        ]
        labels = np.array([0, 0, 1, 1])
        per_topic = compute_topic_exclusivity(texts, labels, top_n=3)
        assert len(per_topic) == 2
        for v in per_topic:
            assert 0.0 <= v <= 1.0


class TestClusterPersistence:
    def test_persistence_perfect_agreement(self):
        consensus = np.array([0, 0, 1, 1, 2, 2])
        seeds = [consensus.copy() for _ in range(5)]
        result = compute_cluster_persistence(consensus, seeds)
        assert result["mean"] == pytest.approx(1.0)
        assert result["min"] == pytest.approx(1.0)

    def test_persistence_random_low(self):
        rng = np.random.default_rng(0)
        consensus = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        seeds = [rng.permutation(3)[rng.integers(0, 3, size=10)] for _ in range(5)]
        result = compute_cluster_persistence(consensus, seeds, jaccard_threshold=0.9)
        # With random labels above a high threshold, persistence should be low
        assert result["mean"] < 0.5
