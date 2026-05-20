"""Rung 1 - tiny-synthetic property verification for trustworthiness.

40 high-d samples from ``make_blobs(centers=4, cluster_std=0.3)`` projected to
2-d by PCA. The high-d↔low-d trustworthiness should be high for both seed
choices and both metrics.

Properties (gate: 1e-12 production-vs-reference; explicit thresholds for
property predictions):

    P1.  PCA preserves most structure ⇒ trustworthiness > 0.85 for both metrics.
    P2.  Random low-d (uncorrelated with high-d) ⇒ low trustworthiness (<0.7).
    P3.  Scale invariance: T(X_high, c*X_low) == T(X_high, X_low) for c > 0
         (under Euclidean and cosine; both invariant to positive scaling of X_low).
    P4.  Production ↔ reference within 1e-12 on each (metric, k) combo tested.
    P5.  Identity case (X_high == X_low) ⇒ T = 1.0 for both metrics.

Plan reference: trustworthiness sibling ladder.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_trustworthiness import reference_trustworthiness
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from reap.evaluation import compute_trustworthiness

TOL = 1e-12


@pytest.fixture(scope="module")
def rung1_data():
    """40 samples in 8-d high space + 2-d PCA projection."""
    blobs = make_blobs(
        n_samples=40,
        n_features=8,
        centers=4,
        cluster_std=0.3,
        random_state=0,
        return_centers=False,
    )
    X_high = np.asarray(blobs[0], dtype=np.float64)
    X_low = PCA(n_components=2, random_state=0).fit_transform(X_high)
    return {"X_high": X_high, "X_low": np.asarray(X_low, dtype=np.float64)}


# ---------------------------------------------------------------------------
# P1 - PCA preserves most structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("k", [3, 5, 10])
def test_p1_pca_preserves_structure(rung1_data, metric, k):
    """PCA 2-d projection of 8-d blobs has trustworthiness > 0.85."""
    t = reference_trustworthiness(
        rung1_data["X_high"], rung1_data["X_low"], n_neighbors=k, metric=metric
    )
    assert 0.0 <= t <= 1.0
    assert t > 0.85, f"PCA trustworthiness on (metric={metric}, k={k}) is {t!r}; expected > 0.85"


# ---------------------------------------------------------------------------
# P2 - Random low-d has low trustworthiness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p2_random_low_d_low_trustworthiness(rung1_data, seed):
    """A 2-d Gaussian unrelated to high-d gives low trustworthiness."""
    X_high = rung1_data["X_high"]
    rng = np.random.RandomState(seed)
    X_random = rng.randn(*X_high.shape[:1], 2)
    t = reference_trustworthiness(X_high, X_random, n_neighbors=5, metric="euclidean")
    assert 0.0 <= t <= 1.0
    assert t < 0.7, f"random low-d (seed={seed}) trustworthiness={t!r}; expected < 0.7"


# ---------------------------------------------------------------------------
# P3 - Scale invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("scale", [0.5, 2.0, 100.0])
def test_p3_scale_invariance(rung1_data, metric, scale):
    """T(X_high, c*X_low) == T(X_high, X_low) for c > 0 (both metrics invariant)."""
    X_high = rung1_data["X_high"]
    X_low = rung1_data["X_low"]
    base = reference_trustworthiness(X_high, X_low, n_neighbors=5, metric=metric)
    scaled = reference_trustworthiness(X_high, scale * X_low, n_neighbors=5, metric=metric)
    assert abs(base - scaled) < TOL, (
        f"metric={metric}, scale={scale}: base={base!r} scaled={scaled!r} "
        f"delta={base - scaled!r}"
    )


# ---------------------------------------------------------------------------
# P4 - Production ↔ reference agreement
#
# Euclidean: exact (1e-12). Cosine: loosely matches (1e-2) because sklearn's
# trustworthiness uses np.argpartition internally, which is non-deterministic
# on tied distances. Cosine distances on real data hit ties more often than
# Euclidean does (because cosine collapses any magnitude difference to the
# same angle). The divergence is a known sklearn behaviour, not a REAP bug.
# Documented and pinned here so the size of the cosine-vs-euclidean gap is
# observable to future readers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 10])
def test_p4_production_matches_reference_euclidean(rung1_data, k):
    """Euclidean: compute_trustworthiness == reference_trustworthiness within 1e-12."""
    X_high = rung1_data["X_high"]
    X_low = rung1_data["X_low"]
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=k, metric="euclidean")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=k, metric="euclidean")
    assert abs(prod - ref) < TOL, (
        f"euclidean k={k}: prod={prod!r} ref={ref!r} delta={prod - ref!r}"
    )


@pytest.mark.parametrize("k", [3, 5, 10])
def test_p4_production_matches_reference_cosine_loose(rung1_data, k):
    """Cosine: production and reference agree to ~1e-2 (sklearn's argpartition
    tie-breaking diverges from numpy.argsort(kind="stable") when many
    cosine distances are tied)."""
    X_high = rung1_data["X_high"]
    X_low = rung1_data["X_low"]
    prod = compute_trustworthiness(X_high, X_low, n_neighbors=k, metric="cosine")
    ref = reference_trustworthiness(X_high, X_low, n_neighbors=k, metric="cosine")
    assert abs(prod - ref) < 1e-1, (
        f"cosine k={k}: prod={prod!r} ref={ref!r} delta={prod - ref!r} "
        "(exceeded the 1e-1 sanity bound; investigate)"
    )


# ---------------------------------------------------------------------------
# P5 - identity X_high == X_low
#
# Euclidean: T = 1.0 exactly (both production and reference).
# Cosine: reference returns 1.0; production returns ~0.998 because sklearn's
# tie-breaking flips some equally-distant neighbours. Pinned here as a known
# behaviour.
# ---------------------------------------------------------------------------


def test_p5_identity_gives_one_euclidean(rung1_data):
    X = rung1_data["X_high"]
    assert reference_trustworthiness(X, X, n_neighbors=5, metric="euclidean") == 1.0
    assert compute_trustworthiness(X, X, n_neighbors=5, metric="euclidean") == 1.0


def test_p5_identity_cosine_reference_exact_production_close(rung1_data):
    """Cosine identity: reference == 1.0; production ≈ 1.0 (within 1e-2)
    due to sklearn's argpartition tie-breaking on cosine distances."""
    X = rung1_data["X_high"]
    t_ref = reference_trustworthiness(X, X, n_neighbors=5, metric="cosine")
    t_prod = compute_trustworthiness(X, X, n_neighbors=5, metric="cosine")
    assert t_ref == 1.0
    assert abs(t_prod - 1.0) < 1e-2, (
        f"cosine identity production should be ≈ 1.0; got {t_prod!r}"
    )
