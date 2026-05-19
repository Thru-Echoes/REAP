"""Tests for the REAP statistics module (paired tests, corrections, bootstrap)."""

from __future__ import annotations

import numpy as np
import pytest

from reap.statistics import (
    benjamini_hochberg,
    cliffs_delta,
    compare_family,
    compute_seed_bootstrap_ci,
    holm_bonferroni,
    paired_cohens_d,
    paired_comparison,
    paired_wilcoxon,
)


class TestSeedBootstrapCI:
    def test_contains_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=5.0, scale=1.0, size=100)
        ci = compute_seed_bootstrap_ci(values, n_bootstrap=2000)
        assert ci.ci_lower < ci.mean < ci.ci_upper
        assert ci.mean == pytest.approx(values.mean())
        assert ci.n_samples == 100

    def test_tiny_sample_still_works(self):
        ci = compute_seed_bootstrap_ci([1.0, 2.0, 3.0], n_bootstrap=100)
        assert ci.mean == pytest.approx(2.0)
        assert ci.n_samples == 3

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            compute_seed_bootstrap_ci([])

    def test_rejects_invalid_ci_level(self):
        with pytest.raises(ValueError):
            compute_seed_bootstrap_ci([1.0, 2.0], ci_level=1.5)


class TestPairedWilcoxon:
    def test_greater_alt(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=30)
        x = base + 0.5
        y = base
        W, p = paired_wilcoxon(x, y, alternative="greater")
        assert np.isfinite(W)
        assert p < 0.05

    def test_equal_returns_nan(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        W, p = paired_wilcoxon(x, y)
        assert np.isnan(W)
        assert np.isnan(p)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_wilcoxon([1, 2, 3], [1, 2])


class TestEffectSizes:
    def test_cohens_d_sign(self):
        # Paired diffs must have non-zero variance for d_z to be defined.
        x = np.array([1.0, 2.5, 3.0, 4.2])
        y = np.array([0.5, 1.5, 2.7, 3.1])
        d = paired_cohens_d(x, y)
        assert d > 0

    def test_cohens_d_zero_on_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert paired_cohens_d(v, v) == 0.0

    def test_cliffs_delta_bounded(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        d = cliffs_delta(x, y)
        assert -1.0 <= d <= 1.0


class TestCorrections:
    def test_holm_monotone(self):
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        adj = holm_bonferroni(p)
        assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)
        assert np.all(adj >= p)

    def test_holm_clips_to_one(self):
        p = np.array([0.01, 0.5])
        adj = holm_bonferroni(p)
        assert np.all(adj <= 1.0)

    def test_bh_monotone_after_sort(self):
        p = np.array([0.001, 0.01, 0.02, 0.04, 0.05])
        q = benjamini_hochberg(p)
        sorted_q = q[np.argsort(p)]
        assert np.all(np.diff(sorted_q) >= -1e-12)

    def test_bh_less_conservative_than_holm(self):
        p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        q = benjamini_hochberg(p)
        h = holm_bonferroni(p)
        # BH ≤ Holm for positive tests (element-wise)
        assert np.all(q <= h + 1e-12)

    def test_empty_input(self):
        assert holm_bonferroni([]).size == 0
        assert benjamini_hochberg([]).size == 0


class TestFamilyComparison:
    def test_compare_family_structure(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(loc=5.0, size=30)
        baseline = rng.normal(loc=4.0, size=30)
        other = rng.normal(loc=4.5, size=30)
        result = compare_family(
            reference_values=ref,
            baseline_values={"base1": baseline, "base2": other},
            reference_name="reap",
            metric_name="silhouette_C",
            direction="higher_is_better",
        )
        assert result.reference == "reap"
        assert result.metric == "silhouette_C"
        assert len(result.comparisons) == 2
        assert len(result.p_holm) == 2
        assert len(result.p_bh) == 2

    def test_compare_family_detects_improvement(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(loc=0.6, scale=0.05, size=30)
        baseline = rng.normal(loc=0.4, scale=0.05, size=30)
        result = compare_family(
            reference_values=ref,
            baseline_values={"baseline": baseline},
            reference_name="reap",
            metric_name="silhouette_C",
            direction="higher_is_better",
        )
        assert result.comparisons[0].mean_diff > 0.1
        assert result.p_holm[0] < 0.05


class TestPairedComparison:
    def test_direction_lower_is_better(self):
        # DB: lower is better; reap should beat baseline when its values are smaller
        rng = np.random.default_rng(0)
        ref = rng.normal(loc=0.5, scale=0.05, size=30)
        baseline = rng.normal(loc=0.8, scale=0.05, size=30)
        r = paired_comparison(
            ref, baseline, "reap", "baseline", "davies_bouldin",
            direction="lower_is_better",
        )
        assert r.mean_diff < 0  # reap smaller
        assert r.wilcoxon_p < 0.05
