"""Rung 0 - closed-form known-answer tests for trustworthiness.

Hand-computed expected values. Tolerance 1e-12.

Closed-form cases (all under ``metric="euclidean"``):

    T1.  Identity (X_high == X_low):    T = 1.0     (k-NN perfectly preserved)
    T2.  Scaled preservation:           T = 1.0     (X_low = c * X_high; k-NN preserved)
    T3.  Hand-derived 5-point intruder case (n=5, k=2):
             X_high = [[0],[1],[3],[10],[11]]
             X_low  = [[0],[10],[3],[1],[11]]    (swap of indices 1 and 3 in low-d)
         Total intruder-rank penalty = 7 ⇒ T = 1 - (2/30)*7 = 8/15 = 0.5333...

**Important definitional finding pinned by this rung.**
``reap.evaluation.compute_trustworthiness`` defaults to ``metric="cosine"``,
while ``sklearn.manifold.trustworthiness`` and ``_reference_trustworthiness``
both default to ``metric="euclidean"``. The closed-form expected values
below are derived under Euclidean; tests therefore call the production
function with ``metric="euclidean"`` explicitly. A separate test pins the
cosine-default behaviour so any future implicit change is caught.

Plan reference: trustworthiness sibling ladder, mirroring ARI / silhouette
Rung-0 templates.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_trustworthiness import reference_trustworthiness
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from reap.evaluation import compute_trustworthiness

TOL = 1e-12


# ---------------------------------------------------------------------------
# T1 - identity: trustworthiness = 1.0
# ---------------------------------------------------------------------------


def test_t1_identity_high_equals_low():
    """X_high == X_low ⇒ every k-NN set is preserved exactly ⇒ T = 1.0."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert reference_trustworthiness(X, X, n_neighbors=2, metric="euclidean") == 1.0
    assert compute_trustworthiness(X, X, n_neighbors=2, metric="euclidean") == 1.0


# ---------------------------------------------------------------------------
# T2 - scaled preservation: trustworthiness = 1.0 for any positive scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 2.0, 100.0])
def test_t2_scaled_preservation(scale):
    """X_low = scale * X_high preserves all k-NN sets (Euclidean ranks
    invariant under uniform scaling) ⇒ T = 1.0."""
    X_high = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    X_low = scale * X_high
    assert reference_trustworthiness(X_high, X_low, n_neighbors=2, metric="euclidean") == 1.0


# ---------------------------------------------------------------------------
# T3 - hand-derived 5-point intruder case: T = 8/15
# ---------------------------------------------------------------------------


def test_t3_five_point_intruder_case():
    """X_high = [[0],[1],[3],[10],[11]]; X_low = [[0],[10],[3],[1],[11]] (k=2).

    Hand derivation (full enumeration of intruders and their high-d ranks):
        index 0: high-NN={1,2}; low-NN={3,2}; intruder={3}; rank_high(0,3)=3; +1
        index 1: high-NN={0,2}; low-NN={4,2}; intruder={4}; rank_high(1,4)=4; +2
        index 2: high-NN={1,0}; low-NN={3,0}; intruder={3}; rank_high(2,3)=3; +1
        index 3: high-NN={4,2}; low-NN={0,2}; intruder={0}; rank_high(3,0)=4; +2
        index 4: high-NN={3,2}; low-NN={1,2}; intruder={1}; rank_high(4,1)=3; +1
    Total intruder-rank penalty = 7.
    Normalizer = 2 / (n*k*(2n - 3k - 1)) = 2 / (5*2*(10-6-1)) = 2/30 = 1/15.
    T = 1 - (1/15) * 7 = 8/15 ≈ 0.5333333333.
    """
    X_high = np.array([[0.0], [1.0], [3.0], [10.0], [11.0]])
    X_low = np.array([[0.0], [10.0], [3.0], [1.0], [11.0]])
    expected = 8.0 / 15.0

    ref = reference_trustworthiness(X_high, X_low, n_neighbors=2, metric="euclidean")
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=2, metric="euclidean")
    skl = float(sklearn_trustworthiness(X_high, X_low, n_neighbors=2, metric="euclidean"))

    assert abs(ref - expected) < TOL, (
        f"reference={ref!r} vs expected={expected!r}, delta={ref - expected!r}"
    )
    assert abs(prod - expected) < TOL
    assert abs(skl - expected) < TOL
    assert abs(prod - ref) < TOL
    assert abs(prod - skl) < TOL


# ---------------------------------------------------------------------------
# Sanity: bounds in [0, 1] across closed-form cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "X_high_data,X_low_data,k",
    [
        ([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], [[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], 2),
        ([[0], [1], [3], [10], [11]], [[0], [10], [3], [1], [11]], 2),
    ],
)
def test_trustworthiness_bounded_unit_interval(X_high_data, X_low_data, k):
    """Trustworthiness lies in [0, 1] on every closed-form case."""
    X_high = np.asarray(X_high_data, dtype=np.float64)
    X_low = np.asarray(X_low_data, dtype=np.float64)
    t = reference_trustworthiness(X_high, X_low, n_neighbors=k, metric="euclidean")
    assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# Production / reference / sklearn three-way agreement on a fresh case
# ---------------------------------------------------------------------------


def test_three_way_agreement_random_case():
    """Random correlated low-d embedding: production == reference == sklearn (1e-12)."""
    rng = np.random.RandomState(0)
    X_high = rng.randn(40, 8)
    # Construct a meaningful low-d by random linear projection (preserves some structure).
    A = rng.randn(8, 2)
    X_low = X_high @ A

    ref = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    skl = float(sklearn_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean"))

    assert abs(prod - ref) < TOL
    assert abs(prod - skl) < TOL


# ---------------------------------------------------------------------------
# Domain check: raises when k is too large for n
# ---------------------------------------------------------------------------


def test_reference_raises_when_k_too_large():
    """Reference raises ValueError if 2*n - 3*k - 1 <= 0 (matches sklearn's contract)."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # n=4 ⇒ k must satisfy 2*4 - 3*k - 1 > 0 ⇒ k < 7/3 ⇒ k in {1, 2}
    with pytest.raises(ValueError):
        reference_trustworthiness(X, X, n_neighbors=3, metric="euclidean")


# ---------------------------------------------------------------------------
# Pin compute_trustworthiness's default metric (cosine, not euclidean)
# ---------------------------------------------------------------------------


def test_compute_trustworthiness_default_metric_is_cosine():
    """``compute_trustworthiness`` defaults to metric="cosine", which differs
    from sklearn's default and from `_reference_trustworthiness`'s default
    (both euclidean).

    The test pins this default so a silent change to the wrapper is caught.
    The same call with metric="cosine" passed to sklearn must match the
    production call's no-explicit-metric output.

    Manuscript implication: any prose number coming out of
    ``compute_trustworthiness`` (e.g. published "trustworthiness" in
    `combined_set_*/all_methods.csv`) is by default a *cosine* score, not
    a *Euclidean* score, even on UMAP output (which is Euclidean by
    construction). This needs to be documented in the methods section.
    """
    rng = np.random.RandomState(0)
    X_high = rng.randn(20, 8)
    A = rng.randn(8, 2)
    X_low = X_high @ A

    default_prod = compute_trustworthiness(X_high, X_low, n_neighbors=5)
    explicit_cosine = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="cosine")
    skl_cosine = float(
        sklearn_trustworthiness(X_high, X_low, n_neighbors=5, metric="cosine")
    )

    # The default IS cosine.
    assert abs(default_prod - explicit_cosine) < TOL
    assert abs(default_prod - skl_cosine) < TOL

    # And cosine differs from euclidean in this case (sanity).
    explicit_euclidean = compute_trustworthiness(
        X_high, X_low, n_neighbors=5, metric="euclidean"
    )
    assert abs(explicit_euclidean - default_prod) > 1e-3, (
        "expected cosine vs euclidean to differ noticeably on random data; "
        f"got cosine={default_prod!r}, euclidean={explicit_euclidean!r}"
    )
