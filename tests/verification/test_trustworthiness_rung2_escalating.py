"""Rung 2 - escalating + degenerate trustworthiness verification.

Each case changes exactly one variable so a failure pinpoints which dimension
drove it. Tolerance: 1e-12 under Euclidean, 1e-1 under cosine (sklearn's
``argpartition`` tie-breaking on equal cosine distances diverges from the
``argsort`` reference; this divergence is a known sklearn behaviour, pinned
in Rung 1).

    (a) n         : 40 → 200 → 1000
    (b) k         : 3 → 5 → 10
    (c) noise     : cluster_std 0.3 → 1.0 → 2.0
    (d) high-d dim: 4 → 8 → 32
    (e) extreme k near the upper boundary k = (n-1)//2 - 1

All cases verified under Euclidean only (where ties are rare and tolerance
is tight); a coarse cosine cross-check is included as a sanity loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_trustworthiness import reference_trustworthiness
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from reap.evaluation import compute_trustworthiness

TOL_EUCLIDEAN = 1e-12
TOL_COSINE = 1e-1


def _make_high_low(n_samples: int, n_features: int = 8, cluster_std: float = 0.3,
                    random_state: int = 0):
    blobs = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=4,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False,
    )
    X_high = np.asarray(blobs[0], dtype=np.float64)
    X_low = PCA(n_components=2, random_state=random_state).fit_transform(X_high)
    return X_high, np.asarray(X_low, dtype=np.float64)


# ---------------------------------------------------------------------------
# (a) Scale n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [40, 200, 1000])
def test_rung2_scale_n_euclidean(n_samples):
    """Euclidean: production ↔ reference within 1e-12 across n."""
    X_high, X_low = _make_high_low(n_samples=n_samples)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    assert abs(prod - ref) < TOL_EUCLIDEAN, (
        f"n={n_samples}: prod={prod!r} ref={ref!r} delta={prod - ref!r}"
    )


# ---------------------------------------------------------------------------
# (b) Scale k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 10])
def test_rung2_scale_k_euclidean(k):
    """Euclidean: production ↔ reference within 1e-12 across k."""
    X_high, X_low = _make_high_low(n_samples=200)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=k, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=k, metric="euclidean")
    assert abs(prod - ref) < TOL_EUCLIDEAN


# ---------------------------------------------------------------------------
# (c) Scale noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.3, 1.0, 2.0])
def test_rung2_scale_noise_euclidean(noise):
    X_high, X_low = _make_high_low(n_samples=200, cluster_std=noise)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    assert abs(prod - ref) < TOL_EUCLIDEAN
    assert 0.0 <= prod <= 1.0


# ---------------------------------------------------------------------------
# (d) High-d dimension sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [4, 8, 32])
def test_rung2_scale_features_euclidean(n_features):
    X_high, X_low = _make_high_low(n_samples=200, n_features=n_features)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric="euclidean")
    assert abs(prod - ref) < TOL_EUCLIDEAN


# ---------------------------------------------------------------------------
# (e) Extreme k near the upper boundary
# ---------------------------------------------------------------------------


def test_rung2_k_near_upper_boundary():
    """For n=40, the original Venna-Kaski 2001 formula admits k up to 26
    (i.e. 2n - 3k - 1 > 0). However, sklearn enforces the stricter bound
    ``k < n_samples / 2`` (in _t_sne.py:526), which means sklearn rejects
    k >= 20 for n=40 with a ValueError. The from-scratch reference here
    uses V&K's looser bound; the production wrapper inherits sklearn's.

    Manuscript implication: any application of trustworthiness in REAP is
    constrained to k < n/2 (sklearn's bound), not the looser theoretical
    bound. The benchmark pipeline already respects this via
    ``n_nn = min(15, X_high.shape[0] - 1)`` (benchmarks.py:380).

    Test at k = 18 (well inside both bounds) for clean code-vs-reference
    comparison.
    """
    X_high, X_low = _make_high_low(n_samples=40)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=18, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=18, metric="euclidean")
    assert abs(prod - ref) < TOL_EUCLIDEAN

    # Pin the sklearn-stricter-than-V&K bound:
    with pytest.raises(ValueError):
        compute_trustworthiness(X_high, X_low, n_neighbors=25, metric="euclidean")


# ---------------------------------------------------------------------------
# Coarse cosine cross-check (loose tolerance due to tie-breaking)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [40, 200])
def test_rung2_cosine_loose(n_samples):
    """Cosine: production within 1e-1 of reference (sklearn argpartition
    tie-breaking on equal cosine distances diverges from stable argsort)."""
    X_high, X_low = _make_high_low(n_samples=n_samples)
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=5, metric="cosine")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric="cosine")
    assert abs(prod - ref) < TOL_COSINE, (
        f"cosine n={n_samples}: prod={prod!r} ref={ref!r} delta={prod - ref!r}"
    )
    assert 0.0 <= prod <= 1.0
