"""Rung 2 - escalating + degenerate ARI verification (verification ladder).

Each test changes exactly one variable from the Rung-1 baseline so that
any failure pinpoints which dimension drove it:

    (a) n         : 40 -> 200 -> 1000
    (b) clusters  : 4  -> 10  -> 25
    (c) noise     : cluster_std 0.3 -> 1.0 -> 2.0
    (d) K-mismatch: 4-blob data clustered with K=4 vs K=7
    (e) degenerate: single-cluster vs structured; n singletons vs structured;
                    both single-cluster; both all-distinct.

For every non-degenerate case the production code (``compute_pairwise_ari``)
and the from-scratch reference (``_reference_ari``) must agree within 1e-12,
and must each lie in [-1, 1]. For degenerate cases the documented convention
is exercised explicitly:

    convention:
        - ARI(all-same, structured) == 0.0
        - ARI(all-singletons, structured) == 0.0
        - ARI(all-same, all-same) == 1.0  (denominator 0/0 by convention)
        - ARI(all-singletons, all-singletons) == 1.0  (denominator 0/0)

Rationale for "0.0 with one trivial side": the contingency-table formula
gives numerator = 0 and denominator > 0 whenever exactly one of the two
partitions is trivial, so ARI = 0 exactly. The from-scratch reference uses
the same convention; sklearn ``adjusted_rand_score`` matches.

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), Task 3.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_ari import reference_ari
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from reap.evaluation import compute_pairwise_ari

TOL = 1e-12


def _make_blobs(n_samples: int, centers: int, cluster_std: float, random_state: int = 0):
    """Return (X, y) with no centers tuple (avoids the make_blobs overload union)."""
    blobs = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False,
    )
    return blobs[0], blobs[1]


# ---------------------------------------------------------------------------
# (a) Scale n: 40, 200, 1000
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [40, 200, 1000])
def test_rung2_scale_n(n_samples):
    """For each n in {40, 200, 1000}, two KMeans labelings: code == reference (1e-12)."""
    X, _ = _make_blobs(n_samples=n_samples, centers=4, cluster_std=0.3)
    s0 = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    s1 = KMeans(n_clusters=4, random_state=1, n_init="auto").fit_predict(X)

    matrix = compute_pairwise_ari([s0, s1])
    ref = reference_ari(s0, s1)

    assert -1.0 <= matrix[0, 1] <= 1.0
    assert -1.0 <= ref <= 1.0
    assert abs(matrix[0, 1] - ref) < TOL, (
        f"n={n_samples}: code={matrix[0, 1]!r} vs reference={ref!r}, delta={matrix[0, 1] - ref!r}"
    )


# ---------------------------------------------------------------------------
# (b) Scale cluster count: 4, 10, 25
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_clusters", [4, 10, 25])
def test_rung2_scale_clusters(n_clusters):
    """For each K in {4, 10, 25}: code == reference on 200-sample blobs (1e-12)."""
    X, _ = _make_blobs(n_samples=200, centers=n_clusters, cluster_std=0.3)
    s0 = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X)
    s1 = KMeans(n_clusters=n_clusters, random_state=1, n_init="auto").fit_predict(X)

    matrix = compute_pairwise_ari([s0, s1])
    ref = reference_ari(s0, s1)
    assert abs(matrix[0, 1] - ref) < TOL, (
        f"K={n_clusters}: code={matrix[0, 1]!r} vs reference={ref!r}, delta={matrix[0, 1] - ref!r}"
    )


# ---------------------------------------------------------------------------
# (c) Scale noise: cluster_std 0.3, 1.0, 2.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.3, 1.0, 2.0])
def test_rung2_scale_noise(noise):
    """For each cluster_std in {0.3, 1.0, 2.0}: code == reference (1e-12).

    As noise rises, KMeans seeds 0 and 1 begin to disagree; the ARI value
    drops, but the code and the reference must continue to match exactly.
    """
    X, _ = _make_blobs(n_samples=200, centers=4, cluster_std=noise)
    s0 = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    s1 = KMeans(n_clusters=4, random_state=1, n_init="auto").fit_predict(X)

    matrix = compute_pairwise_ari([s0, s1])
    ref = reference_ari(s0, s1)
    assert abs(matrix[0, 1] - ref) < TOL, (
        f"noise={noise}: code={matrix[0, 1]!r} vs reference={ref!r}, "
        f"delta={matrix[0, 1] - ref!r}"
    )
    assert -1.0 <= matrix[0, 1] <= 1.0


# ---------------------------------------------------------------------------
# (d) K-mismatch: 4-blob data clustered with K=4 and K=7
# ---------------------------------------------------------------------------


def test_rung2_k_mismatch():
    """4-blob data, KMeans K=4 vs KMeans K=7: code == reference within 1e-12."""
    X, _ = _make_blobs(n_samples=200, centers=4, cluster_std=0.3)
    s_k4 = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    s_k7 = KMeans(n_clusters=7, random_state=0, n_init="auto").fit_predict(X)

    matrix = compute_pairwise_ari([s_k4, s_k7])
    ref = reference_ari(s_k4, s_k7)
    assert abs(matrix[0, 1] - ref) < TOL, (
        f"K-mismatch: code={matrix[0, 1]!r} vs reference={ref!r}, delta={matrix[0, 1] - ref!r}"
    )
    # K=7 has the same data as K=4 but is over-partitioned; ARI should still be > 0.
    assert matrix[0, 1] > 0.0, (
        f"Expected K=7 to retain meaningful agreement with K=4 on well-separated blobs; "
        f"got ARI={matrix[0, 1]!r}"
    )


# ---------------------------------------------------------------------------
# (e) Degenerate: one trivial partition vs structured
# ---------------------------------------------------------------------------


def test_rung2_all_zeros_vs_structured():
    """One-cluster (all zeros) vs structured: ARI = 0 by the contingency formula.

    Why 0 (not undefined): with one trivial side, the contingency table is a
    single row whose row sum equals C(n,2) under C(n_ij, 2); the formula's
    numerator is sum_j C(b_j,2) - sum_j C(b_j,2) = 0 and the denominator is
    a positive scalar, so ARI = 0 exactly.
    """
    _, structured = _make_blobs(n_samples=100, centers=4, cluster_std=0.3)
    structured = np.asarray(structured)
    all_zeros = np.zeros(100, dtype=int)

    matrix = compute_pairwise_ari([all_zeros, structured])
    ref = reference_ari(all_zeros, structured)
    assert abs(matrix[0, 1] - ref) < TOL
    assert matrix[0, 1] == 0.0, f"Expected ARI(all-zeros, structured) == 0; got {matrix[0, 1]!r}"


def test_rung2_all_singletons_vs_structured():
    """Each point in its own cluster vs structured: ARI = 0 by the contingency formula.

    The all-singletons side has row sums all equal to 1, so sum_i C(a_i, 2) = 0;
    the numerator is 0 and the denominator is positive, so ARI = 0 exactly.
    """
    n = 100
    _, structured = _make_blobs(n_samples=n, centers=4, cluster_std=0.3)
    structured = np.asarray(structured)
    all_singletons = np.arange(n, dtype=int)

    matrix = compute_pairwise_ari([all_singletons, structured])
    ref = reference_ari(all_singletons, structured)
    assert abs(matrix[0, 1] - ref) < TOL
    assert matrix[0, 1] == 0.0, (
        f"Expected ARI(all-singletons, structured) == 0; got {matrix[0, 1]!r}"
    )


def test_rung2_both_all_zeros_is_one():
    """Both partitions are a single cluster: ARI = 1.0 by convention (0/0 case)."""
    n = 50
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)

    matrix = compute_pairwise_ari([a, b])
    ref = reference_ari(a, b)
    assert ref == 1.0
    assert matrix[0, 1] == 1.0, (
        f"Expected ARI(all-zeros, all-zeros) == 1.0 by convention; got {matrix[0, 1]!r}"
    )


def test_rung2_both_all_singletons_is_one():
    """Both partitions are n singletons: same partition as set-partitions; ARI = 1.0."""
    n = 50
    a = np.arange(n, dtype=int)
    b = np.arange(n, dtype=int)
    # Relabel b to demonstrate that singleton-as-partition is invariant to label IDs
    rng = np.random.RandomState(0)
    b = rng.permutation(n)

    matrix = compute_pairwise_ari([a, b])
    ref = reference_ari(a, b)
    assert ref == 1.0
    assert matrix[0, 1] == 1.0


# ---------------------------------------------------------------------------
# Cross-check against sklearn directly (third reference path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples,n_clusters,noise", [
    (40, 4, 0.3),
    (200, 4, 1.0),
    (1000, 10, 1.0),
    (200, 25, 2.0),
])
def test_rung2_three_way_sklearn_cross_check(n_samples, n_clusters, noise):
    """``compute_pairwise_ari`` == ``reference_ari`` == ``adjusted_rand_score`` within 1e-12.

    Three-way agreement is the strongest version of the rung's gate; we
    spend an extra sklearn import here to keep the verification protocol
    explicit (the production code uses sklearn internally, so this is not
    full independence — independence is the verifier subagent's job).
    """
    from sklearn.metrics import adjusted_rand_score

    X, _ = _make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=noise)
    s0 = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X)
    s1 = KMeans(n_clusters=n_clusters, random_state=1, n_init="auto").fit_predict(X)

    matrix_val = compute_pairwise_ari([s0, s1])[0, 1]
    ref_val = reference_ari(s0, s1)
    sklearn_val = adjusted_rand_score(s0, s1)

    assert abs(matrix_val - ref_val) < TOL
    assert abs(matrix_val - sklearn_val) < TOL
    assert abs(ref_val - sklearn_val) < TOL
