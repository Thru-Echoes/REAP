"""Tests for the hold-out comparison harness (``reap.holdout``).

Real computation throughout (real heads, real KMeans, real metrics) on
synthetic clustered data. Head training uses a short schedule so the suite
stays fast; the schedule does not change what is being checked (completeness,
pairing, recipes, the fold floor, determinism, and the subsample interval).
Parametric UMAP is covered in ``tests/test_parametric_umap_baseline.py``,
which runs in the tensorflow CI job.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_blobs

from reap.holdout import (
    HOLDOUT_METRIC_IDS,
    HeadTrainingSettings,
    HoldoutComparison,
    HoldoutProjection,
    HoldoutRunMetrics,
    compute_holdout_metrics,
    get_nearest_centroid_labels,
    get_paired_subsample_intervals,
    run_holdout_comparison,
)

pytest.importorskip("torch")

FAST_HEADS = HeadTrainingSettings(
    hidden_layers=[32], n_folds=2, max_epochs=150, patience=15, batch_size=32
)
CLUSTER_LOCATIONS = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0], [12.0, 12.0]])


@pytest.fixture(scope="module")
def split() -> dict[str, np.ndarray]:
    """384-d clustered data split 160 in-set / 80 held-out, with a stand-in consensus map."""
    blobs = make_blobs(n_samples=240, n_features=384, centers=4, cluster_std=1.0, random_state=3)
    X, labels = np.asarray(blobs[0]), np.asarray(blobs[1])
    rng = np.random.default_rng(3)
    order = rng.permutation(240)
    in_idx, out_idx = order[:160], order[160:]
    Y_in = CLUSTER_LOCATIONS[labels[in_idx]] + rng.normal(scale=0.3, size=(160, 2))
    return {
        "X_in": X[in_idx],
        "Y_in": Y_in,
        "labels_in": labels[in_idx],
        "X_out": X[out_idx],
        "labels_out": labels[out_idx],
    }


@pytest.fixture(scope="module")
def comparison(split) -> tuple[HoldoutComparison, dict[str, HoldoutProjection]]:
    """One harness run: MLP head, linear head and identity, three training seeds."""
    return run_holdout_comparison(
        split["X_in"], split["Y_in"], split["labels_in"], split["X_out"], split["labels_out"],
        training_seeds=[11, 22, 33], run_id="synthetic/T1",
        methods=["reap_mlp", "linear_head", "identity"], head_settings=FAST_HEADS,
    )


class TestRecipes:
    def test_nearest_centroid_labels(self) -> None:
        Y_in = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0], [10.2, 0.0]])
        labels_in = np.array([5, 5, 9, 9])
        out = get_nearest_centroid_labels(Y_in, labels_in, np.array([[1.0, 0.0], [8.0, 0.0]]))
        assert out.tolist() == [5, 9]

    def test_known_answer_on_a_perfect_placement(self, split) -> None:
        """Held-out points placed at their own cluster's location: ARI 1, high silhouette."""
        rng = np.random.default_rng(0)
        Y_out = CLUSTER_LOCATIONS[split["labels_out"]] + rng.normal(scale=0.3, size=(80, 2))
        values, reasons = compute_holdout_metrics(
            split["X_out"], Y_out, split["Y_in"], split["labels_in"], split["labels_out"],
            k_holdout=4,
        )
        assert reasons == {}
        assert values["ari_adapted_stage10"] == pytest.approx(1.0)
        assert values["silhouette_holdout"] is not None and values["silhouette_holdout"] > 0.8
        assert values["trustworthiness"] is not None and 0.0 <= values["trustworthiness"] <= 1.0
        assert values["distance_correlation"] is not None and values["distance_correlation"] > 0.5

    def test_collapsed_placement_leaves_silhouette_undefined_with_a_reason(self, split) -> None:
        Y_out = np.zeros((80, 2)) + np.random.default_rng(1).normal(scale=1e-3, size=(80, 2))
        values, reasons = compute_holdout_metrics(
            split["X_out"], Y_out, split["Y_in"], split["labels_in"], split["labels_out"],
            k_holdout=4,
        )
        assert values["silhouette_holdout"] is None
        assert "one nearest centroid" in reasons["silhouette_holdout"]

    @pytest.mark.parametrize("n_out", [31, 11])
    def test_fold_below_the_floor_is_undefined_not_computed(self, split, n_out) -> None:
        """v1.5 clause (h): n >= 2k+2 (=32 at k=15) and n >= 3K; below either, no metric."""
        values, reasons = compute_holdout_metrics(
            split["X_out"][:n_out], split["X_out"][:n_out, :2], split["Y_in"],
            split["labels_in"], split["labels_out"][:n_out], k_holdout=4,
        )
        assert all(v is None for v in values.values())
        assert all("fold too small" in reasons[m] for m in HOLDOUT_METRIC_IDS)

    def test_fold_floor_on_cluster_count(self, split) -> None:
        values, reasons = compute_holdout_metrics(
            split["X_out"][:40], split["X_out"][:40, :2], split["Y_in"],
            split["labels_in"], split["labels_out"][:40], k_holdout=14,
        )
        assert values["trustworthiness"] is None
        assert "3*K" in reasons["trustworthiness"]


class TestHarness:
    def test_one_run_per_trained_method_and_seed_and_one_identity_run(self, comparison) -> None:
        result, _ = comparison
        assert len(result.runs) == 2 * 3 + 1
        assert len(result.get_metric_column("reap_mlp", "trustworthiness")) == 3
        assert len(result.get_metric_column("identity", "trustworthiness")) == 1
        assert result.k_holdout == 4 and result.n_out == 80

    def test_every_run_reports_every_metric(self, comparison) -> None:
        result, _ = comparison
        for run in result.runs:
            assert set(run.values) == set(HOLDOUT_METRIC_IDS)
            assert run.values["trustworthiness"] is not None

    def test_identity_lands_held_out_points_in_their_cluster(self, comparison) -> None:
        result, _ = comparison
        (ari,) = result.get_metric_column("identity", "ari_adapted_stage10")
        assert ari == pytest.approx(1.0)

    def test_same_seeds_give_identical_metrics(self, split, comparison) -> None:
        again, _ = run_holdout_comparison(
            split["X_in"], split["Y_in"], split["labels_in"], split["X_out"],
            split["labels_out"], training_seeds=[11, 22, 33], run_id="synthetic/T1",
            methods=["reap_mlp", "linear_head", "identity"], head_settings=FAST_HEADS,
        )
        assert again == comparison[0]

    def test_different_seeds_change_the_trained_runs(self, comparison) -> None:
        result, _ = comparison
        column = result.get_metric_column("reap_mlp", "distance_correlation")
        assert len(set(column)) > 1

    def test_first_run_projections_are_kept_for_intervals(self, comparison) -> None:
        _, projections = comparison
        assert set(projections) == {"reap_mlp", "linear_head", "identity"}
        assert projections["reap_mlp"].seed == 11
        assert projections["identity"].seed is None
        assert projections["reap_mlp"].Y_out.shape == (80, 2)

    def test_round_trips_through_json(self, comparison) -> None:
        result, _ = comparison
        assert HoldoutComparison.model_validate_json(result.model_dump_json()) == result


class TestFailsClosed:
    def test_incomplete_runs_are_rejected(self, comparison) -> None:
        result, _ = comparison
        data = result.model_dump()
        data["runs"] = data["runs"][1:]
        with pytest.raises(ValidationError, match="exactly"):
            HoldoutComparison.model_validate(data)

    def test_undefined_metric_needs_a_reason(self) -> None:
        values = dict.fromkeys(HOLDOUT_METRIC_IDS, 0.5) | {"silhouette_holdout": None}
        with pytest.raises(ValidationError, match="without a reason"):
            HoldoutRunMetrics(method="identity", seed=None, values=values)

    def test_unknown_method(self, split) -> None:
        with pytest.raises(ValueError, match="unknown methods"):
            run_holdout_comparison(
                split["X_in"], split["Y_in"], split["labels_in"], split["X_out"],
                split["labels_out"], [1], "x", methods=["umap_magic"],  # type: ignore[list-item]
            )

    def test_parametric_umap_without_settings(self, split) -> None:
        with pytest.raises(ValueError, match="umap_settings"):
            run_holdout_comparison(
                split["X_in"], split["Y_in"], split["labels_in"], split["X_out"],
                split["labels_out"], [1], "x", methods=["parametric_umap"],
            )

    def test_misaligned_reference_labels(self, split) -> None:
        with pytest.raises(ValueError, match="row-aligned"):
            run_holdout_comparison(
                split["X_in"], split["Y_in"], split["labels_in"], split["X_out"],
                split["labels_out"][:-1], [1], "x", methods=["identity"],
            )


class TestSubsampleIntervals:
    def test_identical_methods_give_exactly_zero(self, split, comparison) -> None:
        _, projections = comparison
        twin = projections["identity"].model_copy(update={"method": "reap_mlp"})
        intervals = get_paired_subsample_intervals(
            split["X_out"], {"reap_mlp": twin, "identity": projections["identity"]},
            split["labels_in"], split["labels_out"], k_holdout=4, seed=5, n_resamples=40,
        )
        assert len(intervals) == len(HOLDOUT_METRIC_IDS)
        for interval in intervals:
            assert interval.point_difference == 0.0
            assert interval.ci_low == 0.0 and interval.ci_high == 0.0
            assert interval.n_defined == 40

    def test_intervals_cover_each_baseline_and_metric(self, split, comparison) -> None:
        _, projections = comparison
        intervals = get_paired_subsample_intervals(
            split["X_out"], projections, split["labels_in"], split["labels_out"],
            k_holdout=4, seed=5, n_resamples=30,
        )
        pairs = {(i.baseline_method, i.metric_id) for i in intervals}
        assert pairs == {(b, m) for b in ("linear_head", "identity") for m in HOLDOUT_METRIC_IDS}
        for interval in intervals:
            assert interval.ci_low is not None and interval.ci_high is not None
            assert interval.ci_low <= interval.ci_high

    def test_same_seed_same_interval(self, split, comparison) -> None:
        _, projections = comparison
        args = (split["X_out"], projections, split["labels_in"], split["labels_out"])
        a = get_paired_subsample_intervals(*args, k_holdout=4, seed=9, n_resamples=20)
        b = get_paired_subsample_intervals(*args, k_holdout=4, seed=9, n_resamples=20)
        assert a == b

    def test_groups_are_kept_or_dropped_whole(self, split, comparison) -> None:
        """With 16 groups of 5 documents, every subsample keeps 13 whole groups (65 rows)."""
        _, projections = comparison
        groups = np.repeat(np.arange(16), 5)
        intervals = get_paired_subsample_intervals(
            split["X_out"], projections, split["labels_in"], split["labels_out"],
            k_holdout=4, seed=5, n_resamples=10, groups=groups,
        )
        assert all(i.n_defined == 10 for i in intervals)

    def test_bad_fraction(self, split, comparison) -> None:
        _, projections = comparison
        with pytest.raises(ValueError, match="fraction"):
            get_paired_subsample_intervals(
                split["X_out"], projections, split["labels_in"], split["labels_out"],
                k_holdout=4, seed=5, fraction=1.0,
            )


def test_metric_undefined_on_every_subsample_gives_an_empty_interval(split, comparison) -> None:
    """A collapsed placement leaves silhouette undefined; its interval is empty, not invented."""
    _, projections = comparison
    rng = np.random.default_rng(2)
    collapsed = projections["identity"].model_copy(
        update={"method": "reap_mlp", "Y_out": rng.normal(scale=1e-3, size=(80, 2))}
    )
    intervals = get_paired_subsample_intervals(
        split["X_out"], {"reap_mlp": collapsed, "identity": projections["identity"]},
        split["labels_in"], split["labels_out"], k_holdout=4, seed=5, n_resamples=10,
    )
    silhouette = next(i for i in intervals if i.metric_id == "silhouette_holdout")
    assert silhouette.n_defined == 0
    assert silhouette.ci_low is None and silhouette.point_difference is None


def test_harness_names_are_exported_from_the_package() -> None:
    """The harness entry points are reachable as ``reap.<name>``."""
    import reap

    for name in ("run_holdout_comparison", "get_paired_subsample_intervals", "HoldoutComparison"):
        assert name in reap.__all__ and hasattr(reap, name)
