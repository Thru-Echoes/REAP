"""Tests for reap.ablation — seed ablation, parameter sweep, scaling analysis.

Uses small synthetic data (50 samples, 32-d, 3 clusters) with minimal
seed counts and parameter values for fast execution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap.ablation import (
    ParameterSweepResult,
    ScalingResult,
    SeedAblationResult,
    run_parameter_sweep,
    run_scaling_analysis,
    run_seed_ablation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 50
N_FEATURES = 32
N_CLUSTERS = 3
K_RANGE = [2, 3, 4, 5]


@pytest.fixture(scope="module")
def small_data() -> np.ndarray:
    """Small L2-normalized blobs for fast ablation tests."""
    X, _y = make_blobs(  # type: ignore[reportAssignmentType]
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        centers=N_CLUSTERS,
        cluster_std=0.5,
        random_state=42,
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return X


# ---------------------------------------------------------------------------
# Seed ablation tests
# ---------------------------------------------------------------------------


SEED_COUNTS = [2, 3]


@pytest.fixture(scope="module")
def seed_ablation_result(small_data: np.ndarray) -> SeedAblationResult:
    """Run seed ablation once and share across tests."""
    return run_seed_ablation(
        small_data,
        dataset_name="test_blobs",
        n_components=3,
        n_neighbors=10,
        min_dist=0.1,
        k_range=K_RANGE,
        seed_counts=SEED_COUNTS,
        master_seed=42,
        metric="cosine",
        silhouette_metric="euclidean",
    )


class TestSeedAblation:
    """Seed ablation: verify structure and metric properties."""

    def test_returns_correct_type(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        assert isinstance(seed_ablation_result, SeedAblationResult)

    def test_correct_number_of_points(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        assert len(seed_ablation_result.results) == len(SEED_COUNTS)

    def test_seed_counts_match(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        actual_counts = [p.n_seeds for p in seed_ablation_result.results]
        assert actual_counts == SEED_COUNTS

    def test_metrics_are_finite(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        for pt in seed_ablation_result.results:
            assert math.isfinite(pt.trustworthiness), (
                f"n_seeds={pt.n_seeds}: TW not finite"
            )
            assert math.isfinite(pt.silhouette), (
                f"n_seeds={pt.n_seeds}: sil not finite"
            )
            assert math.isfinite(pt.seed_to_seed_ari_mean), (
                f"n_seeds={pt.n_seeds}: ARI mean not finite"
            )
            assert math.isfinite(pt.seed_to_seed_ari_std), (
                f"n_seeds={pt.n_seeds}: ARI std not finite"
            )
            assert math.isfinite(pt.runtime_seconds), (
                f"n_seeds={pt.n_seeds}: runtime not finite"
            )

    def test_more_seeds_better_stability(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        """With more seeds, ARI should generally be at least as good.

        We only require last >= first (soft check, not strictly
        monotonic, since small data can be noisy).
        """
        results = seed_ablation_result.results
        first_ari = results[0].seed_to_seed_ari_mean
        last_ari = results[-1].seed_to_seed_ari_mean
        # Soft check: last should be >= first, with tolerance for noise
        assert last_ari >= first_ari - 0.2, (
            f"Last ARI ({last_ari:.3f}) much worse than first "
            f"({first_ari:.3f})"
        )

    def test_trustworthiness_in_bounds(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        for pt in seed_ablation_result.results:
            assert 0.0 <= pt.trustworthiness <= 1.0

    def test_runtime_positive(
        self, seed_ablation_result: SeedAblationResult
    ) -> None:
        for pt in seed_ablation_result.results:
            assert pt.runtime_seconds > 0.0


# ---------------------------------------------------------------------------
# Parameter sweep tests
# ---------------------------------------------------------------------------


SWEEP_VALUES = [5, 10]


@pytest.fixture(scope="module")
def param_sweep_result(small_data: np.ndarray) -> ParameterSweepResult:
    """Run n_neighbors sweep once and share across tests."""
    return run_parameter_sweep(
        small_data,
        dataset_name="test_blobs",
        swept_parameter="n_neighbors",
        parameter_values=SWEEP_VALUES,
        n_components=3,
        n_neighbors=10,
        min_dist=0.1,
        seeds=[42, 137],
        k_range=K_RANGE,
        metric="cosine",
        silhouette_metric="euclidean",
    )


class TestParameterSweep:
    """Parameter sweep: verify structure and metric properties."""

    def test_returns_correct_type(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        assert isinstance(param_sweep_result, ParameterSweepResult)

    def test_correct_number_of_points(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        assert len(param_sweep_result.results) == len(SWEEP_VALUES)

    def test_parameter_values_match(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        actual = [pt.parameter_value for pt in param_sweep_result.results]
        expected = [float(v) for v in SWEEP_VALUES]
        assert actual == expected

    def test_swept_parameter_recorded(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        assert param_sweep_result.swept_parameter == "n_neighbors"

    def test_metrics_are_finite(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        for pt in param_sweep_result.results:
            assert math.isfinite(pt.trustworthiness)
            assert math.isfinite(pt.silhouette)
            assert math.isfinite(pt.seed_to_seed_ari_mean)
            assert math.isfinite(pt.runtime_seconds)

    def test_runtime_positive(
        self, param_sweep_result: ParameterSweepResult
    ) -> None:
        for pt in param_sweep_result.results:
            assert pt.runtime_seconds > 0.0

    def test_invalid_parameter_raises(
        self, small_data: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="swept_parameter must be"):
            run_parameter_sweep(
                small_data,
                dataset_name="test_blobs",
                swept_parameter="invalid",
                parameter_values=[5, 10],
                n_components=3,
                n_neighbors=10,
                min_dist=0.1,
                seeds=[42, 137],
                k_range=K_RANGE,
            )


# ---------------------------------------------------------------------------
# Scaling analysis tests
# ---------------------------------------------------------------------------


SAMPLE_SIZES = [30, 50]


@pytest.fixture(scope="module")
def scaling_result(small_data: np.ndarray) -> ScalingResult:
    """Run scaling analysis once and share across tests."""
    return run_scaling_analysis(
        small_data,
        dataset_name="test_blobs",
        n_components=3,
        n_neighbors=10,
        min_dist=0.1,
        seeds=[42, 137],
        k_range=K_RANGE,
        sample_sizes=SAMPLE_SIZES,
        metric="cosine",
    )


class TestScalingAnalysis:
    """Scaling analysis: verify structure and runtime properties."""

    def test_returns_scaling_result(
        self, scaling_result: ScalingResult
    ) -> None:
        assert isinstance(scaling_result, ScalingResult)

    def test_runtimes_positive(
        self, scaling_result: ScalingResult
    ) -> None:
        for rt in scaling_result.runtimes:
            assert rt > 0.0

    def test_correct_number_of_points(
        self, scaling_result: ScalingResult
    ) -> None:
        assert len(scaling_result.runtimes) == len(SAMPLE_SIZES)
        assert len(scaling_result.samples_per_second) == len(SAMPLE_SIZES)
        assert len(scaling_result.variable_values) == len(SAMPLE_SIZES)

    def test_variable_values_match(
        self, scaling_result: ScalingResult
    ) -> None:
        assert scaling_result.variable_values == SAMPLE_SIZES

    def test_scaling_variable_is_n_samples(
        self, scaling_result: ScalingResult
    ) -> None:
        assert scaling_result.scaling_variable == "n_samples"

    def test_throughput_positive(
        self, scaling_result: ScalingResult
    ) -> None:
        for sps in scaling_result.samples_per_second:
            assert sps > 0.0

    def test_runtimes_finite(
        self, scaling_result: ScalingResult
    ) -> None:
        for rt in scaling_result.runtimes:
            assert math.isfinite(rt)
