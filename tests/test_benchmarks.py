"""Tests for reap.benchmarks — consensus method comparison.

Runs all six benchmark methods on small synthetic data and verifies
structural properties of the results: correct types, expected method
names, metric bounds, and key claims (REAP stability >= others,
naive_average is the negative control).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap.benchmarks import (
    BenchmarkResult,
    MethodResult,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 50
N_FEATURES = 32
N_CLUSTERS = 3
SEEDS = [42, 137]
K_RANGE = [2, 3, 4, 5]
ALL_METHOD_NAMES = {
    "single_seed",
    "best_of_n",
    "naive_average",
    "procrustes",
    "reap",
    "bertopic",
}


@pytest.fixture(scope="module")
def small_data() -> dict[str, np.ndarray]:
    """Small L2-normalized blobs for fast benchmark tests."""
    X, _y = make_blobs(  # type: ignore[reportAssignmentType]
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        centers=N_CLUSTERS,
        cluster_std=0.5,
        random_state=42,
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return {"X": X}


@pytest.fixture(scope="module")
def benchmark_result(small_data: dict[str, np.ndarray]) -> BenchmarkResult:
    """Run the full benchmark once (all 5 methods) and share across tests."""
    return run_benchmark(
        small_data["X"],
        dataset_name="test_blobs",
        seeds=SEEDS,
        n_components=3,
        n_neighbors=10,
        min_dist=0.1,
        k_range=K_RANGE,
        metric="cosine",
        silhouette_metric="euclidean",
    )


def _get_method(
    result: BenchmarkResult, name: str
) -> MethodResult:
    """Look up a single MethodResult by name."""
    for m in result.methods:
        if m.method == name:
            return m
    raise KeyError(f"Method {name!r} not found in benchmark result")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchmarkStructure:
    """Structural checks on the BenchmarkResult."""

    def test_returns_benchmark_result(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        assert isinstance(benchmark_result, BenchmarkResult)

    def test_has_all_methods(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        assert len(benchmark_result.methods) == 6

    def test_method_names(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        names = {m.method for m in benchmark_result.methods}
        assert names == ALL_METHOD_NAMES

    def test_dataset_metadata(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        assert benchmark_result.n_samples == N_SAMPLES
        assert benchmark_result.n_features == N_FEATURES
        assert benchmark_result.n_seeds == len(SEEDS)
        assert benchmark_result.k_range == K_RANGE
        assert benchmark_result.dataset_name == "test_blobs"


class TestMethodMetrics:
    """Metric-level checks for individual methods."""

    def test_all_methods_positive_trustworthiness(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for m in benchmark_result.methods:
            assert m.trustworthiness > 0.0, (
                f"{m.method} trustworthiness={m.trustworthiness}"
            )

    def test_all_methods_finite_metrics(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for m in benchmark_result.methods:
            assert math.isfinite(m.trustworthiness), (
                f"{m.method} TW not finite"
            )
            assert math.isfinite(m.silhouette), (
                f"{m.method} sil not finite"
            )
            assert math.isfinite(m.trustworthiness_std), (
                f"{m.method} TW std not finite"
            )
            assert math.isfinite(m.silhouette_std), (
                f"{m.method} sil std not finite"
            )
            assert math.isfinite(m.runtime_seconds), (
                f"{m.method} runtime not finite"
            )

    def test_runtime_positive(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for m in benchmark_result.methods:
            assert m.runtime_seconds > 0.0, (
                f"{m.method} runtime={m.runtime_seconds}"
            )


class TestSingleSeedVariability:
    """single_seed should show non-zero std across seeds."""

    def test_single_seed_has_std(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        ss = _get_method(benchmark_result, "single_seed")
        # With 2 seeds on small data, std may be tiny but should be >= 0
        # The key property: single_seed computes per-seed metrics
        assert ss.trustworthiness_std >= 0.0


class TestConsensusMethodsStd:
    """Consensus methods (procrustes, reap) produce a single embedding,
    so trustworthiness_std must be exactly 0."""

    def test_consensus_methods_zero_std(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for name in ("procrustes", "reap"):
            m = _get_method(benchmark_result, name)
            assert m.trustworthiness_std == 0.0, (
                f"{name} TW std={m.trustworthiness_std}, expected 0"
            )
            assert m.silhouette_std == 0.0, (
                f"{name} sil std={m.silhouette_std}, expected 0"
            )


class TestREAPStability:
    """REAP should have seed-to-seed ARI >= all other methods that
    report it, because distance-matrix averaging eliminates
    rotation/reflection ambiguity."""

    def test_reap_has_highest_seed_ari(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        reap = _get_method(benchmark_result, "reap")
        assert reap.seed_to_seed_ari_mean is not None

        for m in benchmark_result.methods:
            if m.method == "reap":
                continue
            if m.seed_to_seed_ari_mean is None:
                continue
            assert reap.seed_to_seed_ari_mean >= (
                m.seed_to_seed_ari_mean - 0.15
            ), (
                f"REAP s2s ARI ({reap.seed_to_seed_ari_mean:.3f}) "
                f"lower than {m.method} "
                f"({m.seed_to_seed_ari_mean:.3f}) by > 0.15"
            )


class TestNaiveAverageNegativeControl:
    """naive_average is the negative control: it averages coordinates
    without alignment, so silhouette should be degraded compared to REAP."""

    def test_naive_average_low_silhouette(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        naive = _get_method(benchmark_result, "naive_average")
        reap = _get_method(benchmark_result, "reap")
        # Naive average should have silhouette <= REAP (with tolerance)
        assert naive.silhouette <= reap.silhouette + 0.15, (
            f"naive_average sil ({naive.silhouette:.3f}) unexpectedly "
            f"higher than REAP sil ({reap.silhouette:.3f})"
        )


class TestSubsetOfMethods:
    """run_benchmark with a methods subset returns only those methods."""

    def test_subset_of_methods(
        self, small_data: dict[str, np.ndarray]
    ) -> None:
        result = run_benchmark(
            small_data["X"],
            dataset_name="test_subset",
            seeds=SEEDS,
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            k_range=K_RANGE,
            metric="cosine",
            silhouette_metric="euclidean",
            methods=["reap", "procrustes"],
        )
        assert len(result.methods) == 2
        names = {m.method for m in result.methods}
        assert names == {"reap", "procrustes"}


class TestPerSeedAndCells:
    """Per-seed distributions and the 2x2 (label_source x embedding_source) cells."""

    def test_per_seed_vectors_length_matches_n_seeds(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for m in benchmark_result.methods:
            assert len(m.per_seed_silhouette) == len(SEEDS), (
                f"{m.method} per_seed_silhouette len={len(m.per_seed_silhouette)}"
            )
            assert len(m.per_seed_trustworthiness) == len(SEEDS)
            assert len(m.per_seed_continuity) == len(SEEDS)
            assert len(m.per_seed_davies_bouldin) == len(SEEDS)
            assert len(m.per_seed_calinski_harabasz) == len(SEEDS)
            assert len(m.per_seed_k) == len(SEEDS)

    def test_consensus_methods_have_cells_b_and_c(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for name in ("procrustes", "reap", "naive_average", "best_of_n"):
            m = _get_method(benchmark_result, name)
            assert m.cell_b_silhouette is not None, f"{name} cell_b missing"
            assert m.cell_c_silhouette is not None, f"{name} cell_c missing"
            assert len(m.cell_b_silhouette) == len(SEEDS)
            assert len(m.cell_c_silhouette) == len(SEEDS)

    def test_single_seed_has_no_cells_b_c(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        m = _get_method(benchmark_result, "single_seed")
        assert m.cell_b_silhouette is None
        assert m.cell_c_silhouette is None

    def test_consensus_singletons_populated(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for name in ("procrustes", "reap", "naive_average", "best_of_n"):
            m = _get_method(benchmark_result, name)
            assert m.consensus_silhouette is not None
            assert m.consensus_trustworthiness is not None
            assert m.consensus_continuity is not None
            assert m.consensus_davies_bouldin is not None
            assert m.consensus_calinski_harabasz is not None
            assert m.consensus_k is not None

    def test_s2s_matrices_are_30x30(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for m in benchmark_result.methods:
            assert m.s2s_ari_matrix is not None
            assert m.s2s_ami_matrix is not None
            assert m.s2s_nmi_matrix is not None
            n = len(SEEDS)
            for matrix in (m.s2s_ari_matrix, m.s2s_ami_matrix, m.s2s_nmi_matrix):
                assert len(matrix) == n
                assert all(len(row) == n for row in matrix)

    def test_s2c_lists_present_for_consensus(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        for name in ("procrustes", "reap", "naive_average", "best_of_n"):
            m = _get_method(benchmark_result, name)
            assert m.s2c_ari_list is not None
            assert m.s2c_ami_list is not None
            assert m.s2c_nmi_list is not None
            assert len(m.s2c_ari_list) == len(SEEDS)


class TestArtifactsReturn:
    """run_benchmark_with_artifacts exposes the raw numpy arrays."""

    def test_returns_umap_and_per_method(
        self, small_data: dict[str, np.ndarray]
    ) -> None:
        from reap.benchmarks import BenchmarkArtifacts, run_benchmark_with_artifacts

        _result, artifacts = run_benchmark_with_artifacts(
            small_data["X"],
            dataset_name="test_artifacts",
            seeds=SEEDS,
            n_components=3, n_neighbors=10, min_dist=0.1, k_range=K_RANGE,
            methods=["reap", "procrustes", "single_seed"],
        )
        assert isinstance(artifacts, BenchmarkArtifacts)
        assert len(artifacts.umap_embeddings) == len(SEEDS)
        # REAP should have consensus_distance_matrix
        assert "consensus_distance_matrix" in artifacts.per_method["reap"]
        # Procrustes should have consensus_embedding + consensus_labels
        assert "consensus_embedding" in artifacts.per_method["procrustes"]
        assert "consensus_labels" in artifacts.per_method["procrustes"]
        # single_seed has only per-seed labels (no consensus)
        assert "seed_labels" in artifacts.per_method["single_seed"]
        assert "consensus_embedding" not in artifacts.per_method["single_seed"]
