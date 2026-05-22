"""Unit tests for reap.analysis cross-source + temporal helpers.

Known-answer tests for each public function. These are pure functions, so
every test is deterministic with no fixtures beyond small hand-built arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

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


class TestKLDivergence:
    def test_identical_distributions_is_zero(self):
        p = np.array([3, 1, 4, 1, 5])
        assert compute_kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_known_answer_with_smoothing(self):
        # p=[0,4], q=[2,2], alpha=1 -> p_norm=[1/6,5/6], q_norm=[1/2,1/2]
        # KL = (1/6)ln(1/3) + (5/6)ln(5/3)
        p = np.array([0, 4])
        q = np.array([2, 2])
        expected = (1 / 6) * np.log(1 / 3) + (5 / 6) * np.log(5 / 3)
        assert compute_kl_divergence(p, q, alpha=1.0) == pytest.approx(expected, abs=1e-12)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.integers(0, 50, size=10)
            q = rng.integers(0, 50, size=10)
            assert compute_kl_divergence(p, q) >= -1e-12

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_kl_divergence(np.array([1, 2]), np.array([1, 2, 3]))


class TestCentroidsAndAssignment:
    def test_centroids_are_cluster_means(self):
        Y = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
        assign = np.array([0, 0, 1])
        centroids = compute_cluster_centroids(Y, assign, n_clusters=2)
        assert centroids[0] == pytest.approx([1.0, 0.0])
        assert centroids[1] == pytest.approx([10.0, 10.0])

    def test_empty_cluster_is_zero(self):
        Y = np.array([[1.0, 1.0]])
        assign = np.array([0])
        centroids = compute_cluster_centroids(Y, assign, n_clusters=3)
        assert centroids[1] == pytest.approx([0.0, 0.0])
        assert centroids[2] == pytest.approx([0.0, 0.0])

    def test_nearest_centroid_assignment(self):
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        Y_probe = np.array([[1.0, 1.0], [9.0, 9.0], [0.5, 0.0]])
        assign = assign_by_nearest_centroid(Y_probe, centroids)
        assert assign.tolist() == [0, 1, 0]


class TestKnnPurity:
    def test_perfect_purity(self):
        # All ref points in cluster 0; probe assigned to 0 -> purity 1.0
        Y_ref = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        ref_assign = np.array([0, 0, 0])
        Y_probe = np.array([[1.0, 0.0]])
        probe_assign = np.array([0])
        mean, std = compute_knn_purity(Y_probe, Y_ref, probe_assign, ref_assign, k=2)
        assert mean == pytest.approx(1.0)
        assert std == pytest.approx(0.0)

    def test_zero_purity(self):
        Y_ref = np.array([[1.0, 0.0], [0.9, 0.1]])
        ref_assign = np.array([1, 1])
        Y_probe = np.array([[1.0, 0.0]])
        probe_assign = np.array([0])  # probe says 0, neighbors are all 1
        mean, _ = compute_knn_purity(Y_probe, Y_ref, probe_assign, ref_assign, k=2)
        assert mean == pytest.approx(0.0)


class TestCentroidCosine:
    def test_collinear_is_one(self):
        centroids = np.array([[2.0, 0.0]])
        Y_probe = np.array([[5.0, 0.0]])
        probe_assign = np.array([0])
        got = compute_median_centroid_cosine(Y_probe, centroids, probe_assign)
        assert got == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        centroids = np.array([[1.0, 0.0]])
        Y_probe = np.array([[0.0, 1.0]])
        probe_assign = np.array([0])
        got = compute_median_centroid_cosine(Y_probe, centroids, probe_assign)
        assert got == pytest.approx(0.0)


class TestContingencyCramersV:
    def test_perfect_association(self):
        # Diagonal table -> Cramér's V = 1
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 0, 1, 1, 2, 2])
        table, chi2, v, _p = compute_contingency_cramers_v(row, col, 3, 3)
        assert table.shape == (3, 3)
        assert v == pytest.approx(1.0, abs=1e-9)
        assert chi2 > 0

    def test_no_association_low_v(self):
        # Uniform spread -> low Cramér's V
        rng = np.random.default_rng(1)
        row = rng.integers(0, 3, size=600)
        col = rng.integers(0, 4, size=600)
        _table, _chi2, v, _p = compute_contingency_cramers_v(row, col, 3, 4)
        assert 0.0 <= v < 0.2

    def test_degenerate_returns_zero(self):
        row = np.array([0, 0, 0])
        col = np.array([0, 0, 0])
        _table, chi2, v, p = compute_contingency_cramers_v(row, col, 1, 1)
        assert chi2 == 0.0 and v == 0.0 and p == 1.0

    def test_table_sums_match_inputs(self):
        row = np.array([0, 1, 2, 0, 1])
        col = np.array([1, 1, 0, 2, 0])
        table, _chi2, _v, _p = compute_contingency_cramers_v(row, col, 3, 3)
        assert int(table.sum()) == 5


class TestThemeClusterContingency:
    def test_shape_and_sum(self):
        themes = np.array([0, 1, 2, 3, 4, 0])
        clusters = np.array([1, 1, 2, 0, 3, 1])
        table, chi2, v = compute_theme_cluster_contingency(
            themes, clusters, n_clusters=5, n_themes=5
        )
        assert table.shape == (5, 5)
        assert int(table.sum()) == 6
        assert chi2 >= 0.0
        assert 0.0 <= v <= 1.0


class TestSourceSeparability:
    def test_well_separated_high(self):
        rng = np.random.default_rng(2)
        Y_ref = rng.normal(0, 0.05, size=(50, 2))
        Y_probe = rng.normal(10, 0.05, size=(50, 2))
        sil = compute_source_separability_silhouette(Y_ref, Y_probe)
        assert sil > 0.9

    def test_overlapping_near_zero(self):
        rng = np.random.default_rng(3)
        Y_ref = rng.normal(0, 1, size=(100, 2))
        Y_probe = rng.normal(0, 1, size=(100, 2))
        sil = compute_source_separability_silhouette(Y_ref, Y_probe)
        assert abs(sil) < 0.2


class TestTemporalAlignment:
    def test_per_cluster_year_stats(self):
        years = np.array([2013, 2015, 2020, 2024, 2024])
        clusters = np.array([0, 0, 1, 1, 1])
        result = compute_temporal_alignment(years, clusters, n_clusters=3)
        assert set(result.keys()) == {"per_cluster", "ordering_by_mean_year"}
        assert len(result["per_cluster"]) == 3
        # cluster 0 mean year = 2014, cluster 1 = (2020+2024+2024)/3
        c0 = result["per_cluster"][0]
        assert c0["mean_year"] == pytest.approx(2014.0)
        assert c0["n"] == 2
        # cluster 2 is empty
        assert result["per_cluster"][2]["n"] == 0
        # ordering is a permutation of 0..2
        assert sorted(result["ordering_by_mean_year"]) == [0, 1, 2]
        # cluster 0 (earlier) before cluster 1 (later); empty cluster 2 last
        assert result["ordering_by_mean_year"][0] == 0
        assert result["ordering_by_mean_year"][-1] == 2


class TestProcrustesDisparity:
    def test_identical_is_zero(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(20, 3))
        assert compute_procrustes_disparity(A, A.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_rotation_is_alignable(self):
        rng = np.random.default_rng(5)
        A = rng.normal(size=(20, 2))
        theta = 0.7
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        B = A @ R.T
        # Procrustes removes rotation -> disparity ~ 0
        assert compute_procrustes_disparity(A, B) == pytest.approx(0.0, abs=1e-9)

    def test_in_unit_interval(self):
        rng = np.random.default_rng(6)
        A = rng.normal(size=(30, 4))
        B = rng.normal(size=(30, 4))
        d = compute_procrustes_disparity(A, B)
        assert 0.0 <= d <= 1.0


class TestDemographicContingency:
    def test_string_labels_associate(self):
        # "young" always -> cluster 0; "old" always -> cluster 1 (perfect)
        demo = np.array(["young", "young", "old", "old", "young", "old"], dtype=object)
        clusters = np.array([0, 0, 1, 1, 0, 1])
        out = compute_demographic_contingency(clusters, demo, n_clusters=2)
        assert out["levels"] == ["old", "young"]
        assert out["cramers_v"] == pytest.approx(1.0, abs=1e-9)
        assert out["n"] == 6
        assert np.array(out["table"]).sum() == 6

    def test_no_association(self):
        rng = np.random.default_rng(7)
        demo = np.array(["a" if x else "b" for x in rng.integers(0, 2, size=400)], dtype=object)
        clusters = rng.integers(0, 5, size=400).astype(np.int64)
        out = compute_demographic_contingency(clusters, demo, n_clusters=5)
        assert 0.0 <= out["cramers_v"] < 0.25


class TestBaserateEnrichment:
    def test_uniform_probe_uniform_base_is_one(self):
        # 2 probes per cluster across 3 clusters; base rate uniform 1/3
        probe_assign = np.array([0, 0, 1, 1, 2, 2])
        base = np.array([1 / 3, 1 / 3, 1 / 3])
        out = compute_baserate_enrichment(probe_assign, base, n_clusters=3)
        for c in range(3):
            assert out["enrichment_ratio"][c] == pytest.approx(1.0)

    def test_enriched_cluster(self):
        # All 4 probes in cluster 0, but base rate says cluster 0 = 0.25
        probe_assign = np.array([0, 0, 0, 0])
        base = np.array([0.25, 0.5, 0.25])
        out = compute_baserate_enrichment(probe_assign, base, n_clusters=3)
        assert out["enrichment_ratio"][0] == pytest.approx(1.0 / 0.25)  # 4x enriched
        assert out["enrichment_ratio"][1] == pytest.approx(0.0)  # depleted

    def test_zero_base_rate_is_none(self):
        probe_assign = np.array([0, 1])
        base = np.array([0.5, 0.5, 0.0])
        out = compute_baserate_enrichment(probe_assign, base, n_clusters=3)
        assert out["enrichment_ratio"][2] is None

    def test_base_rate_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_baserate_enrichment(np.array([0, 1]), np.array([0.5, 0.5]), n_clusters=3)
