"""Tests for reap.selection: K by separation and stability, and setting selection."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.datasets import make_blobs

from reap.selection import (
    KScore,
    KSelection,
    SettingEvaluation,
    compute_stability_at_k,
    evaluate_setting,
    find_k_by_separation_and_stability,
    rank_settings,
    select_setting_with_second_draw,
)

K_RANGE = [2, 3, 4, 5, 6, 7]
COMMON = {"silhouette_weight": 0.5, "stability_weight": 0.5, "kmeans_n_init": 10,
          "random_state": 11}


@pytest.fixture(scope="module")
def maps() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Five separated groups in 6-d along distinct directions, plus three noisy 'seed' copies."""
    centers = np.eye(6)[:5] * 10.0
    data = make_blobs(n_samples=200, centers=centers, cluster_std=0.4, random_state=3)
    X, y = np.asarray(data[0]), np.asarray(data[1])
    rng = np.random.default_rng(5)
    seeds = [X + rng.normal(scale=0.3, size=X.shape) for _ in range(3)]
    return X, seeds, y


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_finds_the_true_k_with_either_metric(
    maps: tuple[np.ndarray, list[np.ndarray], np.ndarray], metric: str,
) -> None:
    """Five separated groups give K = 5 under both declared silhouette metrics."""
    X, seeds, _ = maps
    sel = find_k_by_separation_and_stability(X, seeds, K_RANGE, min_cluster_size=10,
                                             silhouette_metric=metric, **COMMON)
    assert sel.best_k == 5
    assert sel.silhouette_metric == metric
    assert [s.k for s in sel.scores] == K_RANGE


def test_accepts_a_custom_metric(maps: tuple[np.ndarray, list[np.ndarray], np.ndarray]) -> None:
    """A callable metric is used and its name recorded."""
    X, seeds, _ = maps
    sel = find_k_by_separation_and_stability(X, seeds, K_RANGE, min_cluster_size=10,
                                             silhouette_metric=cosine_distance, **COMMON)
    assert sel.best_k == 5
    assert sel.silhouette_metric == "cosine"


def test_size_floor_drops_small_clusters(
    maps: tuple[np.ndarray, list[np.ndarray], np.ndarray],
) -> None:
    """Every kept K meets the floor; dropped K carry no score; an impossible floor raises."""
    X, seeds, _ = maps
    sel = find_k_by_separation_and_stability(X, seeds, K_RANGE, min_cluster_size=35,
                                             silhouette_metric="euclidean", **COMMON)
    for s in sel.scores:
        assert s.kept == (s.smallest_cluster >= 35)
        assert (s.score is None) == (not s.kept)
    with pytest.raises(ValueError, match="no K"):
        find_k_by_separation_and_stability(X, seeds, K_RANGE, min_cluster_size=500,
                                           silhouette_metric="euclidean", **COMMON)


def test_rejects_invalid_weights(maps: tuple[np.ndarray, list[np.ndarray], np.ndarray]) -> None:
    """Negative or all-zero weights are an error."""
    X, seeds, _ = maps
    with pytest.raises(ValueError, match="weights"):
        find_k_by_separation_and_stability(
            X, seeds, K_RANGE, min_cluster_size=10, silhouette_metric="euclidean",
            silhouette_weight=0.0, stability_weight=0.0, kmeans_n_init=10, random_state=1)


def test_stability_at_a_common_k(maps: tuple[np.ndarray, list[np.ndarray], np.ndarray]) -> None:
    """Seeds clustered at the same K as the consensus agree almost perfectly on clean groups."""
    X, seeds, _ = maps
    s2c, s2s = compute_stability_at_k(X, seeds, 5, kmeans_n_init=10, random_state=2)
    assert s2c > 0.95 and s2s > 0.95


def _evaluation(nn: int, trust: float, sil: float, s2c: float) -> SettingEvaluation:
    """A hand-made evaluation for ranking tests."""
    sel = KSelection(best_k=3, scores=[KScore(k=3, smallest_cluster=10, kept=True, silhouette=sil,
                                              seed_to_consensus_ari=s2c)],
                     min_cluster_size=5, silhouette_metric="euclidean", silhouette_weight=0.5,
                     stability_weight=0.5, kmeans_n_init=10, random_state=1)
    return SettingEvaluation(n_components=5, n_neighbors=nn, min_dist=0.1, trustworthiness=trust,
                             trust_n_neighbors=15, k_selection=sel, best_k=3, silhouette=sil,
                             seed_to_consensus_ari=s2c, seed_to_seed_ari=s2c, seeds=[1, 2])


def test_rank_settings_scales_before_weighting() -> None:
    """A large raw silhouette does not dominate once metrics are scaled; the floor applies first."""
    evals = [_evaluation(5, 0.80, 0.90, 0.50), _evaluation(15, 0.95, 0.60, 0.90),
             _evaluation(30, 0.90, 0.70, 0.80)]
    ranked = rank_settings(evals, metric_weights={"trustworthiness": 1, "silhouette": 1,
                                                  "seed_to_consensus_ari": 1},
                           trustworthiness_floor=None)
    assert ranked[0].evaluation.n_neighbors == 15
    assert ranked[0].scaled["trustworthiness"] == pytest.approx(1.0)
    floored = rank_settings(evals, metric_weights={"silhouette": 1},
                            trustworthiness_floor="median")
    assert {r.evaluation.n_neighbors for r in floored} == {15, 30}
    with pytest.raises(ValueError, match="unknown metrics"):
        rank_settings(evals, metric_weights={"accuracy": 1}, trustworthiness_floor=None)


def test_second_draw_can_overturn_the_first() -> None:
    """The first draw's winner loses when its second-draw score is low."""
    first = [_evaluation(5, 0.9, 0.9, 0.9), _evaluation(15, 0.8, 0.8, 0.8),
             _evaluation(30, 0.1, 0.1, 0.1)]
    second = [_evaluation(5, 0.1, 0.1, 0.1), _evaluation(15, 0.9, 0.9, 0.9),
              _evaluation(30, 0.5, 0.5, 0.5)]
    weights = {"trustworthiness": 1.0, "silhouette": 1.0}
    chosen, table = select_setting_with_second_draw(first, second, top_n=2,
                                                    metric_weights=weights,
                                                    trustworthiness_floor=None)
    assert chosen.n_neighbors == 15
    assert len(table) == 2
    with pytest.raises(ValueError, match="missing"):
        select_setting_with_second_draw(first, second[1:], top_n=2, metric_weights=weights,
                                        trustworthiness_floor=None)


def test_evaluate_setting_end_to_end() -> None:
    """A real setting: trust at the fixed neighbourhood, K from the declared rule."""
    data = make_blobs(n_samples=150, centers=4, n_features=16, cluster_std=1.0, random_state=9)
    X = np.asarray(data[0], dtype=np.float32)
    ev = evaluate_setting(
        X, [101, 202, 303], 2, 8, 0.1, k_range=[2, 3, 4, 5, 6], input_metric="euclidean",
        trust_n_neighbors=15, min_cluster_size=10, silhouette_metric="euclidean",
        silhouette_weight=0.5, stability_weight=0.5, kmeans_n_init=10, final_map_seed=404,
        kmeans_seed=505)
    assert ev.trust_n_neighbors == 15 and ev.n_neighbors == 8
    assert 0.0 < ev.trustworthiness <= 1.0
    assert ev.best_k in [2, 3, 4, 5, 6]
    assert -1.0 <= ev.seed_to_seed_ari <= 1.0
    assert ev.k_selection.best_k == ev.best_k
