"""Tests for the grid search framework.

Tests cover: single config evaluation, grid search, and selection strategies.
Uses small synthetic data with minimal seeds for fast execution.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap._types import GridSearchConfig, GridSearchResult
from reap.search import evaluate_config, grid_search, select_best_config


@pytest.fixture(scope="module")
def search_blobs() -> np.ndarray:
    """Small synthetic data for grid search tests."""
    data = make_blobs(n_samples=80, n_features=32, centers=3, random_state=42)
    return np.asarray(data[0])


class TestEvaluateConfig:
    """Tests for single configuration evaluation."""

    def test_returns_grid_search_result(self, search_blobs: np.ndarray) -> None:
        result = evaluate_config(
            X=search_blobs,
            seeds=[42, 137],
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            k_range=list(range(2, 6)),
        )
        assert isinstance(result, GridSearchResult)

    def test_metrics_in_valid_range(self, search_blobs: np.ndarray) -> None:
        result = evaluate_config(
            X=search_blobs,
            seeds=[42, 137],
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            k_range=list(range(2, 6)),
        )
        assert 0.0 <= result.trustworthiness <= 1.0
        assert -1.0 <= result.best_silhouette <= 1.0
        assert result.best_k >= 2

    def test_has_stability_metrics(self, search_blobs: np.ndarray) -> None:
        result = evaluate_config(
            X=search_blobs,
            seeds=[42, 137],
            n_components=3,
            n_neighbors=10,
            min_dist=0.1,
            k_range=list(range(2, 6)),
        )
        assert result.mean_ari_consensus is not None
        assert result.seed_k_mode is not None
        assert result.seed_k_agree is not None


class TestGridSearch:
    """Tests for the full grid search."""

    def test_returns_correct_number_of_results(self, search_blobs: np.ndarray) -> None:
        config = GridSearchConfig(
            n_components_list=[3],
            n_neighbors_list=[10, 15],
            min_dist_list=[0.1],
            k_range=list(range(2, 5)),
            n_seeds=2,
            seeds=[42, 137],
        )
        results = grid_search(search_blobs, config)
        # 1 n_components × 2 n_neighbors × 1 min_dist = 2
        assert len(results) == 2

    def test_all_results_are_valid(self, search_blobs: np.ndarray) -> None:
        config = GridSearchConfig(
            n_components_list=[3],
            n_neighbors_list=[10],
            min_dist_list=[0.1, 0.5],
            k_range=list(range(2, 5)),
            n_seeds=2,
            seeds=[42, 137],
        )
        results = grid_search(search_blobs, config)
        for r in results:
            assert isinstance(r, GridSearchResult)
            assert r.trustworthiness > 0


class TestSelectBestConfig:
    """Tests for selection strategy functions."""

    @pytest.fixture
    def sample_results(self) -> list[GridSearchResult]:
        return [
            GridSearchResult(
                n_components=3, n_neighbors=10, min_dist=0.1,
                trustworthiness=0.9, best_k=3, best_silhouette=0.7,
                mean_ari_seeds=0.6, std_ari_seeds=0.1,
            ),
            GridSearchResult(
                n_components=5, n_neighbors=15, min_dist=0.05,
                trustworthiness=0.8, best_k=4, best_silhouette=0.8,
                mean_ari_seeds=0.7, std_ari_seeds=0.05,
            ),
            GridSearchResult(
                n_components=3, n_neighbors=20, min_dist=0.01,
                trustworthiness=0.95, best_k=3, best_silhouette=0.5,
                mean_ari_seeds=0.5, std_ari_seeds=0.15,
            ),
        ]

    def test_equal_weight(self, sample_results: list[GridSearchResult]) -> None:
        best = select_best_config(sample_results, strategy="equal_weight")
        assert isinstance(best, GridSearchResult)

    def test_cluster_focused(self, sample_results: list[GridSearchResult]) -> None:
        best = select_best_config(sample_results, strategy="cluster_focused")
        assert isinstance(best, GridSearchResult)

    def test_stability_focused(self, sample_results: list[GridSearchResult]) -> None:
        best = select_best_config(sample_results, strategy="stability_focused")
        assert isinstance(best, GridSearchResult)

    def test_structure_focused(self, sample_results: list[GridSearchResult]) -> None:
        best = select_best_config(sample_results, strategy="structure_focused")
        assert isinstance(best, GridSearchResult)

    def test_pareto(self, sample_results: list[GridSearchResult]) -> None:
        best = select_best_config(sample_results, strategy="pareto")
        assert isinstance(best, GridSearchResult)

    def test_unknown_strategy_raises(self, sample_results: list[GridSearchResult]) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_best_config(sample_results, strategy="nonexistent")
