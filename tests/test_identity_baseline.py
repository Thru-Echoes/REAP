"""Tests for the identity baseline (``reap.baselines.identity_baseline``).

The baseline copies each new document's nearest reference document's map
coordinate. Tests check it against an independent brute-force loop, pin its
deterministic tie rule, check the similarity it reports for the leakage audit,
check that it fails loudly on bad input, and run it end to end on clustered
384-d synthetic data, where held-out documents must land in their own cluster.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from reap.baselines import IdentityProjectionResult, get_identity_projection


def _get_reference_nearest(X_reference: np.ndarray, X_query: np.ndarray, metric: str) -> list[int]:
    """Independent nearest-neighbour search: a plain Python loop over pairs."""
    nearest = []
    for q in X_query:
        best_index, best_value = -1, None
        for i, r in enumerate(X_reference):
            if metric == "cosine":
                value = -float(q @ r / (np.linalg.norm(q) * np.linalg.norm(r)))
            else:
                value = float(np.sqrt(np.sum((q - r) ** 2)))
            if best_value is None or value < best_value:
                best_index, best_value = i, value
        nearest.append(best_index)
    return nearest


@pytest.fixture(scope="module")
def random_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small random reference set, map coordinates, and queries."""
    rng = np.random.default_rng(20261116)
    return rng.normal(size=(60, 32)), rng.normal(size=(60, 5)), rng.normal(size=(25, 32))


class TestAgainstReference:
    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_matches_brute_force_loop(self, random_inputs, metric) -> None:
        X_ref, Y_ref, X_query = random_inputs
        result = get_identity_projection(X_ref, Y_ref, X_query, metric=metric)
        expected = _get_reference_nearest(X_ref, X_query, metric)
        assert result.nearest_reference_indices.tolist() == expected
        np.testing.assert_array_equal(result.projection, Y_ref[expected])

    def test_query_identical_to_a_reference_row_copies_that_row(self, random_inputs) -> None:
        X_ref, Y_ref, _ = random_inputs
        result = get_identity_projection(X_ref, Y_ref, X_ref[[3, 17, 42]])
        assert result.nearest_reference_indices.tolist() == [3, 17, 42]
        np.testing.assert_array_equal(result.projection, Y_ref[[3, 17, 42]])
        # Unit-norm rounding can leave the self-similarity a few ulps from 1.
        np.testing.assert_allclose(result.nearest_reference_similarity, 1.0, rtol=0, atol=1e-12)

    def test_cosine_and_euclidean_can_disagree(self) -> None:
        X_ref = np.array([[1.0, 0.0], [10.0, 1.0]])
        Y_ref = np.array([[0.0], [1.0]])
        query = np.array([[9.0, 0.0]])
        by_cosine = get_identity_projection(X_ref, Y_ref, query, "cosine")
        by_euclidean = get_identity_projection(X_ref, Y_ref, query, "euclidean")
        assert by_cosine.nearest_reference_indices.tolist() == [0]
        assert by_euclidean.nearest_reference_indices.tolist() == [1]


class TestDeterminism:
    def test_ties_go_to_the_lowest_reference_index(self) -> None:
        X_ref = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])  # rows 0 and 1: same direction
        Y_ref = np.array([[0.0], [1.0], [2.0]])
        result = get_identity_projection(X_ref, Y_ref, np.array([[5.0, 0.0]]))
        assert result.nearest_reference_indices.tolist() == [0]

    def test_repeated_calls_are_identical(self, random_inputs) -> None:
        a = get_identity_projection(*random_inputs)
        b = get_identity_projection(*random_inputs)
        np.testing.assert_array_equal(a.projection, b.projection)

    def test_projection_does_not_alias_the_input(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        result = get_identity_projection(X_ref, Y_ref, X_query)
        result.projection[:] = 0.0
        assert not np.all(Y_ref == 0.0)


class TestSimilarityForLeakageAudit:
    def test_similarity_is_the_cosine_to_the_chosen_neighbour(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        result = get_identity_projection(X_ref, Y_ref, X_query)
        chosen = X_ref[result.nearest_reference_indices]
        expected = np.sum(chosen * X_query, axis=1) / (
            np.linalg.norm(chosen, axis=1) * np.linalg.norm(X_query, axis=1)
        )
        np.testing.assert_allclose(result.nearest_reference_similarity, expected, atol=1e-12)
        assert np.all(np.abs(result.nearest_reference_similarity) <= 1.0 + 1e-12)

    def test_result_type_and_shapes(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        result = get_identity_projection(X_ref, Y_ref, X_query)
        assert isinstance(result, IdentityProjectionResult)
        assert result.method == "identity"
        assert result.projection.shape == (25, 5)
        assert result.nearest_reference_indices.shape == (25,)
        assert result.nearest_reference_indices.dtype == np.int64


class TestFailsLoudly:
    def test_misaligned_reference_rows(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        with pytest.raises(ValueError, match="row-aligned"):
            get_identity_projection(X_ref, Y_ref[:-1], X_query)

    def test_feature_count_mismatch(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        with pytest.raises(ValueError, match="features"):
            get_identity_projection(X_ref, Y_ref, X_query[:, :-1])

    def test_zero_length_row_under_cosine(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        X_query = X_query.copy()
        X_query[4] = 0.0
        with pytest.raises(ValueError, match="zero length"):
            get_identity_projection(X_ref, Y_ref, X_query)

    def test_non_finite_values(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        X_query = X_query.copy()
        X_query[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            get_identity_projection(X_ref, Y_ref, X_query)

    def test_empty_query(self, random_inputs) -> None:
        X_ref, Y_ref, _ = random_inputs
        with pytest.raises(ValueError, match="at least one row"):
            get_identity_projection(X_ref, Y_ref, np.empty((0, 32)))

    def test_one_dimensional_input(self, random_inputs) -> None:
        X_ref, Y_ref, X_query = random_inputs
        with pytest.raises(ValueError, match="2-d"):
            get_identity_projection(X_ref, Y_ref, X_query[0])

    def test_unknown_metric(self, random_inputs) -> None:
        with pytest.raises(ValueError, match="metric"):
            get_identity_projection(*random_inputs, metric="manhattan")  # type: ignore[arg-type]


def test_e2e_held_out_documents_land_in_their_own_cluster() -> None:
    """On well-separated 384-d clusters, every held-out point copies a same-cluster coordinate."""
    blobs = make_blobs(n_samples=300, n_features=384, centers=4, cluster_std=1.0, random_state=7)
    X, labels = np.asarray(blobs[0]), np.asarray(blobs[1])
    rng = np.random.default_rng(7)
    held_out = rng.choice(300, size=60, replace=False)
    in_set = np.setdiff1d(np.arange(300), held_out)
    # Stand-in reference map: each cluster's points sit at a distinct 2-d location.
    cluster_locations = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    Y_in = cluster_locations[labels[in_set]] + rng.normal(scale=0.1, size=(in_set.size, 2))

    result = get_identity_projection(X[in_set], Y_in, X[held_out])
    copied_labels = labels[in_set][result.nearest_reference_indices]
    assert np.array_equal(copied_labels, labels[held_out])
    assert np.all(result.nearest_reference_similarity > 0.5)
