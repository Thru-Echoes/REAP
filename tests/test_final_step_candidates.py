"""Tests for the final-step candidates and the pilot driver (scripts/experiments/).

Candidates a1 and a2 must be deterministic and recover well-separated groups
from a distance matrix; the linkage cut must give exactly K clusters even when
merge heights tie (as co-association distances do); b and c must reduce to the
obvious answer when every final map agrees; and the pilot must run end to end
on the benchmark's synthetic dataset and write every output it promises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

import final_step_candidates as cand  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_final_step_pilot as pilot  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.repro_bundle import read_repro_bundle  # noqa: E402

K_RANGE = [2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four well-separated 5-d blobs: (points, distance matrix, true labels)."""
    data = make_blobs(n_samples=120, centers=4, n_features=5, cluster_std=0.3,
                      center_box=(-10.0, 10.0), random_state=3)
    X = np.asarray(data[0])
    return X, squareform(pdist(X)), np.asarray(data[1])


def test_a1_recovers_separated_groups(blobs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Average linkage on the matrix finds the four blobs and picks K = 4."""
    _, D, y = blobs
    result = cand.run_candidate_a1(D, K_RANGE)
    assert result.k == 4
    assert adjusted_rand_score(y, result.labels) == pytest.approx(1.0)


def test_a2_recovers_separated_groups_and_is_deterministic(
    blobs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """k-medoids finds the blobs, picks K = 4, and gives identical output twice."""
    _, D, y = blobs
    first = cand.run_candidate_a2(D, K_RANGE)
    second = cand.run_candidate_a2(D, K_RANGE)
    assert first.k == 4
    assert adjusted_rand_score(y, first.labels) == pytest.approx(1.0)
    assert np.array_equal(first.labels, second.labels)


def test_kmedoids_medoids_are_members_of_their_clusters(
    blobs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Each medoid belongs to its own cluster, and the greedy order has no repeats."""
    _, D, _ = blobs
    order = cand.get_kmedoids_build_order(D, 6)
    assert len(set(order)) == 6
    labels, medoids = cand.get_kmedoids_labels(D, order[:4])
    for j, m in enumerate(medoids):
        assert labels[m] == j


def test_linkage_cut_gives_exactly_k_with_tied_heights() -> None:
    """Co-association distances tie heavily; every cut still has exactly K clusters."""
    runs = [np.array([0, 0, 1, 1, 2, 2, 3, 3]), np.array([0, 0, 0, 0, 1, 1, 1, 1])]
    distance = cand.get_coassociation_distance(runs)
    cuts = cand.get_average_linkage_labels(distance, [2, 3, 4, 5])
    for k, labels in cuts.items():
        assert len(np.unique(labels)) == k


def test_coassociation_distance_values() -> None:
    """Points always together are at 0, never together at 1, half the time at 0.5."""
    runs = [np.array([0, 0, 1]), np.array([0, 1, 1])]
    distance = cand.get_coassociation_distance(runs)
    assert distance[0, 1] == pytest.approx(0.5)
    assert distance[1, 2] == pytest.approx(0.5)
    assert distance[0, 2] == pytest.approx(1.0)
    assert np.all(np.diag(distance) == 0.0)


def test_b_returns_the_blobs_when_all_maps_are_identical(
    blobs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """With the same map under every seed, b's co-association gives the four blobs."""
    X, _, y = blobs
    b = cand.run_candidate_b({42: X, 0: X, 1: X}, K_RANGE)
    assert b.k == 4
    assert adjusted_rand_score(y, b.labels) == pytest.approx(1.0)


def test_c_prefers_the_smallest_perfectly_repeatable_k(
    blobs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Agreement-based K favours small K: merging whole blobs is repeatable too.

    On four separated blobs, K = 2 and K = 4 both agree perfectly across
    k-means seeds, and the tie goes to the smaller K. This is the known bias
    of choosing K by stability, recorded as a threat in the amendment; the
    test pins it so a change in that behaviour is noticed.
    """
    X, _, _ = blobs
    c = cand.run_candidate_c({42: X, 0: X, 1: X}, K_RANGE, 42)
    best = max(c.k_scores.values())
    assert c.k_scores[c.k] == pytest.approx(best)
    assert c.k == min(k for k, v in c.k_scores.items() if v == pytest.approx(best))
    assert c.k_scores[4] == pytest.approx(1.0)
    assert c.k < 4
    assert np.array_equal(c.labels, cand.get_kmeans_labels(X, c.k, 42))


def test_d_uses_restarts_and_picks_true_k(
    blobs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Candidate d recovers the blobs with ten k-means starts."""
    X, _, y = blobs
    result = cand.run_candidate_d(X, K_RANGE, 42)
    assert result.k == 4
    assert adjusted_rand_score(y, result.labels) == pytest.approx(1.0)


@pytest.mark.slow
def test_pilot_end_to_end_on_synthetic(tmp_path: Path) -> None:
    """Three sets of the synthetic dataset, then the combine step.

    Uses the full 30-seed sets of the manifest, so it is marked slow.
    """
    out = tmp_path / "out"
    bench = tmp_path / "bench" / "reap" / "set_A"
    bench.mkdir(parents=True)
    from run_paper_benchmark import (  # pyright: ignore[reportMissingImports]
        DATASET_CONFIGS,
    )

    c = DATASET_CONFIGS["synthetic"]
    (bench / "bundle.json").write_text(json.dumps({"config": {
        "n_components": c.n_components, "n_neighbors": c.n_neighbors,
        "min_dist": c.min_dist, "k_range": c.k_range}}))
    for s in pilot.SETS:
        assert pilot.main(["--dataset", "synthetic", "--set", s,
                           "--config-from", str(bench / "bundle.json"),
                           "--research-root", str(tmp_path), "--output-dir", str(out)]) == 0
        assert read_repro_bundle(out / f"set_{s}" / "bundle.json").run_status == "completed"
    assert pilot.main(["--dataset", "synthetic", "--combine", "--research-root", str(tmp_path),
                       "--output-dir", str(out)]) == 0
    summary = json.loads((out / "pilot_summary.json").read_text())
    assert set(summary["criteria"]) == set(pilot.CANDIDATES)
    for name in pilot.CANDIDATES:
        margin = summary["criteria"][name]["bootstrap_margin_over_best_baseline"]
        assert margin["lower_95"] <= margin["upper_95"]
    sources = set(pilot.get_label_sources())
    rows = (out / "between_set.csv").read_text().strip().splitlines()[1:]
    assert len(rows) == 3 * len(sources)
    fixed = (out / "fixed_k.csv").read_text().strip().splitlines()[1:]
    assert len(fixed) == len(pilot.FIXED_KS) * 7
