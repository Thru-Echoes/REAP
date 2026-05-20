"""Rung 1 - tiny-synthetic property verification for silhouette.

40 samples in 2-d from ``make_blobs(centers=4, cluster_std=0.3, random_state=0)``,
clustered twice with ``KMeans(random_state=0/1, n_init="auto")``. Properties
asserted, every concrete silhouette value cross-checked against
``_reference_silhouette`` (Rung-0 verified).

Properties (gate: 1e-12 within agreement; explicit thresholds for the
property predictions):

    P1.  Both seeds recover well-separated structure ⇒ silhouette(X, seed_i) > 0.5
    P2.  Permutation invariance: silhouette(X, σ(labels)) == silhouette(X, labels)
         for any relabeling σ; tolerance 1e-12.
    P3.  Scale invariance (Euclidean): silhouette(c*X, labels) == silhouette(X, labels)
         for c > 0; tolerance 1e-12.
    P4.  Random labels are uncorrelated with structure: silhouette < 0.15 on
         random labelings (over a few random states).
    P5.  Production ``compute_silhouette`` and from-scratch ``reference_silhouette``
         agree within 1e-12 on every labeling produced here.

A RED here blocks Rung 2; invoke ``superpowers:systematic-debugging``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_silhouette import reference_silhouette
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from reap.evaluation import compute_silhouette

TOL = 1e-12


@pytest.fixture(scope="module")
def rung1_data():
    """40 samples in 2-d + two deterministic KMeans labelings + ground truth."""
    blobs = make_blobs(
        n_samples=40,
        centers=4,
        cluster_std=0.3,
        random_state=0,
        return_centers=False,
    )
    X, ground_truth = blobs[0], blobs[1]
    seed_0 = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    seed_1 = KMeans(n_clusters=4, random_state=1, n_init="auto").fit_predict(X)
    return {
        "X": np.asarray(X, dtype=np.float64),
        "ground_truth": np.asarray(ground_truth),
        "seed_0": np.asarray(seed_0),
        "seed_1": np.asarray(seed_1),
    }


# ---------------------------------------------------------------------------
# P1 - both seeds yield well-separated silhouette (> 0.5)
# ---------------------------------------------------------------------------


def test_p1_each_seed_silhouette_above_half(rung1_data):
    """Well-separated blobs (cluster_std=0.3) ⇒ silhouette > 0.5 for both seeds."""
    X = rung1_data["X"]
    for key in ("seed_0", "seed_1", "ground_truth"):
        labels = rung1_data[key]
        s = reference_silhouette(X, labels)
        assert s > 0.5, f"silhouette({key})={s!r} should exceed 0.5"


# ---------------------------------------------------------------------------
# P2 - permutation invariance
# ---------------------------------------------------------------------------


def test_p2_permutation_invariance(rung1_data):
    """Relabeling does not change silhouette (silhouette depends on partitions,
    not on label identifiers)."""
    X = rung1_data["X"]
    labels = rung1_data["seed_0"]
    base = reference_silhouette(X, labels)

    rng = np.random.RandomState(0)
    permutation = rng.permutation(4)
    relabeled = permutation[labels]
    permuted = reference_silhouette(X, relabeled)

    assert abs(permuted - base) < TOL, (
        f"Label permutation changed silhouette: base={base!r}, permuted={permuted!r}, "
        f"delta={permuted - base!r}"
    )


# ---------------------------------------------------------------------------
# P3 - scale invariance (Euclidean)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 100.0])
def test_p3_scale_invariance_euclidean(rung1_data, scale):
    """Multiplying X by a positive constant leaves the silhouette unchanged
    (under Euclidean distances, a→ca and b→cb, and (cb−ca)/max(ca,cb) = (b−a)/max(a,b))."""
    X = rung1_data["X"]
    labels = rung1_data["seed_0"]
    base = reference_silhouette(X, labels)
    scaled = reference_silhouette(scale * X, labels)
    assert abs(scaled - base) < TOL, (
        f"scale={scale}: silhouette changed under scaling: base={base!r}, "
        f"scaled={scaled!r}, delta={scaled - base!r}"
    )


# ---------------------------------------------------------------------------
# P4 - random labels are uncorrelated with structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_p4_random_labels_low_silhouette(rung1_data, seed):
    """Uniform random labels over 4 clusters ⇒ silhouette is low (< 0.15)."""
    X = rung1_data["X"]
    rng = np.random.RandomState(seed)
    random_labels = rng.randint(0, 4, size=X.shape[0])
    if len(np.unique(random_labels)) < 2:
        pytest.skip("rng produced a single-cluster labeling — skipping")
    s = reference_silhouette(X, random_labels)
    assert s < 0.15, f"random labels (seed={seed}) silhouette={s!r}; expected < 0.15"


# ---------------------------------------------------------------------------
# P5 - production ↔ reference exact on the produced labelings
# ---------------------------------------------------------------------------


def test_p5_production_matches_reference_seed_0(rung1_data):
    """compute_silhouette(X, seed_0) == reference_silhouette(X, seed_0) (1e-12)."""
    X = rung1_data["X"]
    labels = rung1_data["seed_0"]
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL


def test_p5_production_matches_reference_seed_1(rung1_data):
    X = rung1_data["X"]
    labels = rung1_data["seed_1"]
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL


def test_p5_production_matches_reference_ground_truth(rung1_data):
    X = rung1_data["X"]
    labels = rung1_data["ground_truth"]
    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    assert abs(prod - ref) < TOL


# ---------------------------------------------------------------------------
# Boundedness
# ---------------------------------------------------------------------------


def test_silhouette_bounded(rung1_data):
    """Silhouette must lie in [-1, 1] on every labeling considered here."""
    X = rung1_data["X"]
    for key in ("seed_0", "seed_1", "ground_truth"):
        labels = rung1_data[key]
        s = reference_silhouette(X, labels)
        assert -1.0 <= s <= 1.0
