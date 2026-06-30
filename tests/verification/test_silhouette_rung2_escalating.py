"""Rung 2 - escalating + degenerate silhouette verification.

Each case changes exactly one variable so a failure pinpoints which dimension
drove it. Production ``compute_silhouette`` and the from-scratch
``reference_silhouette`` (Rung-0 verified) must agree within 1e-12, and the
returned value must lie in [-1, 1].

    (a) n         : 40 → 200 → 1000
    (b) clusters  : 4  → 10  → 25
    (c) noise     : cluster_std 0.3 → 1.0 → 2.0
    (d) K-mismatch: 4-blob data, KMeans K=4 vs K=7
    (e) one cluster with a single point (other clusters with > 1 point)
    (f) three-way agreement with sklearn.silhouette_score on representative configs

Plan reference: silhouette ladder sibling of the ARI Rung-2 plan.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_silhouette import reference_silhouette
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from reap.evaluation import compute_silhouette

TOL = 1e-12


def _make_blobs(n_samples: int, centers: int, cluster_std: float, random_state: int = 0):
    blobs = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        return_centers=False,
    )
    return np.asarray(blobs[0], dtype=np.float64), np.asarray(blobs[1])


# ---------------------------------------------------------------------------
# (a) Scale n
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [40, 200, 1000])
def test_rung2_scale_n(n_samples):
    """compute_silhouette == reference_silhouette within 1e-12 across n."""
    X, _ = _make_blobs(n_samples=n_samples, centers=4, cluster_std=0.3)
    labels = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL, (
        f"n={n_samples}: code={prod!r} vs reference={ref!r}, delta={prod - ref!r}"
    )
    assert -1.0 <= prod <= 1.0


# ---------------------------------------------------------------------------
# (b) Scale clusters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_clusters", [4, 10, 25])
def test_rung2_scale_clusters(n_clusters):
    """For each K in {4, 10, 25}: code == reference within 1e-12."""
    X, _ = _make_blobs(n_samples=200, centers=n_clusters, cluster_std=0.3)
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X)
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL


# ---------------------------------------------------------------------------
# (c) Scale noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise", [0.3, 1.0, 2.0])
def test_rung2_scale_noise(noise):
    """For each cluster_std in {0.3, 1.0, 2.0}: code == reference within 1e-12.

    As noise rises, the silhouette drops; the metric must still agree exactly
    between code and reference.
    """
    X, _ = _make_blobs(n_samples=200, centers=4, cluster_std=noise)
    labels = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL
    assert -1.0 <= prod <= 1.0


# ---------------------------------------------------------------------------
# (d) K-mismatch
# ---------------------------------------------------------------------------


def test_rung2_k_mismatch():
    """4-blob data, KMeans K=4 vs KMeans K=7: code == reference."""
    X, _ = _make_blobs(n_samples=200, centers=4, cluster_std=0.3)
    labels_k4 = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    labels_k7 = KMeans(n_clusters=7, random_state=0, n_init="auto").fit_predict(X)
    for tag, labels in (("K=4", labels_k4), ("K=7", labels_k7)):
        prod = compute_silhouette(X, labels)
        ref = reference_silhouette(X, labels)
        assert abs(prod - ref) < TOL, (
            f"K-mismatch {tag}: code={prod!r} vs reference={ref!r}"
        )


# ---------------------------------------------------------------------------
# (e) One cluster with a singleton point
# ---------------------------------------------------------------------------


def test_rung2_one_singleton_cluster():
    """One cluster has a single point: that point gets s=0 by Rousseeuw;
    other points get standard computation. code == reference within 1e-12."""
    # 3 clusters: 5 points around (0,0), 5 around (10,10), 1 isolated point.
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.normal(0.0, 0.1, size=(5, 2)),
        rng.normal(10.0, 0.1, size=(5, 2)),
        np.array([[20.0, 20.0]]),
    ])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])

    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL


# ---------------------------------------------------------------------------
# (f) Three-way agreement with sklearn directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_samples,n_clusters,noise",
    [
        (40, 4, 0.3),
        (200, 4, 1.0),
        (1000, 10, 1.0),
        (200, 25, 2.0),
    ],
)
def test_rung2_three_way_sklearn_cross_check(n_samples, n_clusters, noise):
    """compute_silhouette ↔ reference_silhouette ↔ sklearn.silhouette_score within 1e-12."""
    X, _ = _make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=noise)
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X)
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    skl = float(silhouette_score(X, labels))

    assert abs(prod - ref) < TOL
    assert abs(prod - skl) < TOL
    assert abs(ref - skl) < TOL


# ---------------------------------------------------------------------------
# Boundedness across all parametric cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [40, 1000])
@pytest.mark.parametrize("n_clusters", [4, 25])
def test_rung2_silhouette_bounded(n_samples, n_clusters):
    """Silhouette always lies in [-1, 1] regardless of n/K/noise."""
    if n_clusters >= n_samples:
        pytest.skip("invalid: n_clusters >= n_samples")
    X, _ = _make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=1.0)
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(X)
    s = reference_silhouette(X, labels)
    assert -1.0 <= s <= 1.0
