"""Rung 0 - closed-form known-answer tests for silhouette (verification ladder).

Hand-computed expected values; every assertion uses a value re-derivable in
<10 lines of arithmetic from the Rousseeuw definition
(``s = (b - a) / max(a, b)``, ``s = 0`` for singleton clusters). Tolerance 1e-12.

This is the first rung of the *silhouette* sibling ladder. The protocol
mirrors the s2s/s2c ARI ladder: closed-form ⇒ tiny synthetic ⇒ escalating
⇒ 20-Newsgroups ⇒ AI-art + Korean forest. The load-bearing question this
ladder answers is whether REAP's apparent silhouette win (0.67-0.78 vs
0.40-0.49 for baselines on the production CSVs) is a genuine cluster-
quality signal or a *circular* artefact of UMAP's loss objective.

Closed-form cases derived analytically:

    T1.  Perfect separation, 1-d:           silhouette = 1.0
         [0, 0, 10, 10] / labels [0, 0, 1, 1]
    T2.  Unit square, split along x=0.5:    silhouette = 3 - 2*sqrt(2)
         (0,0),(0,1),(1,0),(1,1) / labels [0, 0, 1, 1]
         = 0.17157287525380993...
    T3.  6 points 1-d, three per cluster:   silhouette = 419 / 504
         [0,1,2,8,9,10] / labels [0,0,0,1,1,1]
         = 0.8313492063492063
    T4.  Overlapping clusters (negative):   silhouette = -7/16 = -0.4375
         [0, 10, 1, 9] / labels [0, 0, 1, 1]
    T5.  All singletons:                    silhouette = 0.0
         3 points in any layout / labels [0, 1, 2]
    T6.  <2 unique labels (degenerate):     reference raises ValueError;
                                            production returns -1.0 with warning.

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), reuse-template instantiation for
the silhouette metric.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
from _reference_silhouette import reference_silhouette
from sklearn.metrics import silhouette_score

from reap.evaluation import compute_silhouette

TOL = 1e-12


# ---------------------------------------------------------------------------
# T1 - perfect separation: silhouette = 1.0 exactly
# ---------------------------------------------------------------------------


def test_t1_perfect_separation_one_dimensional():
    """Each cluster has 2 colocated points; clusters are 10 apart.

    For each point i: a(i) = 0 (the other same-cluster point is colocated),
    b(i) = 10 (both points in the other cluster are at distance 10).
    s(i) = (10 - 0) / max(0, 10) = 10 / 10 = 1.0 for every i.
    """
    X = np.array([[0.0], [0.0], [10.0], [10.0]])
    labels = np.array([0, 0, 1, 1])
    assert reference_silhouette(X, labels) == 1.0
    assert compute_silhouette(X, labels) == 1.0


# ---------------------------------------------------------------------------
# T2 - unit-square split: silhouette = 3 - 2*sqrt(2)
# ---------------------------------------------------------------------------


def test_t2_unit_square_split_along_x():
    """4 points at unit-square corners; clusters {(0,0),(0,1)} and {(1,0),(1,1)}.

    For each point: a = 1 (distance to its cluster-mate one unit away on y).
                    b = (1 + sqrt(2)) / 2  (mean distance to the two opposite-cluster points).
                    s = (b - a) / b = (sqrt(2) - 1) / (sqrt(2) + 1)
                                    = 3 - 2*sqrt(2)
                                    = 0.17157287525380993...
    By symmetry every point has the same s.
    """
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    expected = 3.0 - 2.0 * sqrt(2.0)
    assert abs(reference_silhouette(X, labels) - expected) < TOL
    assert abs(compute_silhouette(X, labels) - expected) < TOL


# ---------------------------------------------------------------------------
# T3 - 6 points 1-d, three per cluster: silhouette = 419 / 504
# ---------------------------------------------------------------------------


def test_t3_two_clusters_of_three_one_dimensional():
    """Points [0,1,2,8,9,10]; labels [0,0,0,1,1,1].

    Hand derivation:
      pos 0  | a = mean(1, 2) = 1.5 | b = mean(8, 9, 10) = 9   | s = 7.5/9   = 5/6
      pos 1  | a = mean(1, 1) = 1   | b = mean(7, 8, 9) = 8    | s = 7/8     = 7/8
      pos 2  | a = mean(2, 1) = 1.5 | b = mean(6, 7, 8) = 7    | s = 5.5/7   = 11/14
      pos 8,9,10 by symmetry         |                          | s = 11/14, 7/8, 5/6
    Mean = (5/6 + 7/8 + 11/14) * 2 / 6
         = (5/6 + 7/8 + 11/14) / 3
    Common denominator 168:  5/6 = 140/168;  7/8 = 147/168;  11/14 = 132/168
    Sum = 419/168
    Mean = 419 / (168 * 3) = 419/504
         = 0.8313492063492063...
    """
    X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    expected = 419.0 / 504.0
    assert abs(reference_silhouette(X, labels) - expected) < TOL
    assert abs(compute_silhouette(X, labels) - expected) < TOL


# ---------------------------------------------------------------------------
# T4 - overlapping clusters: silhouette = -7/16 (negative)
# ---------------------------------------------------------------------------


def test_t4_overlapping_clusters_negative_silhouette():
    """Points [0, 10, 1, 9]; labels [0, 0, 1, 1] -- interleaved by position.

    pos 0  (cluster 0): a = dist(0, 10) = 10        b = mean(1, 9) = 5    s = (5-10)/10 = -0.5
    pos 10 (cluster 0): a = 10                       b = mean(9, 1) = 5    s = -0.5
    pos 1  (cluster 1): a = dist(1, 9) = 8           b = mean(1, 9) = 5    s = (5-8)/8 = -3/8
    pos 9  (cluster 1): a = 8                        b = mean(9, 1) = 5    s = -3/8
    Mean = (-1/2 + -1/2 + -3/8 + -3/8) / 4
         = (-8/16 - 8/16 - 6/16 - 6/16) / 4
         = -28/16 / 4
         = -7/16
         = -0.4375
    """
    X = np.array([[0.0], [10.0], [1.0], [9.0]])
    labels = np.array([0, 0, 1, 1])
    expected = -7.0 / 16.0
    assert abs(reference_silhouette(X, labels) - expected) < TOL
    assert abs(compute_silhouette(X, labels) - expected) < TOL


# ---------------------------------------------------------------------------
# T5 - all singletons: reference returns 0 (Rousseeuw); sklearn refuses
# ---------------------------------------------------------------------------


def test_t5_all_singletons_reference_returns_zero_sklearn_refuses():
    """3 points, each in its own cluster.

    Rousseeuw's convention: s(i) = 0 for singleton clusters, so mean s = 0.
    sklearn's contract: ``silhouette_samples`` requires ``2 <= n_unique <= n - 1``
    and raises ValueError otherwise. So the reference returns 0.0; the
    production wrapper (which calls sklearn) raises. Both behaviours are
    pinned here so the divergence is documented.

    Implication for the production pipeline: callers must ensure that
    silhouette is not invoked on partitions where every point is its own
    cluster (e.g. n_clusters == n_samples in degenerate KMeans output).
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 2])
    assert reference_silhouette(X, labels) == 0.0
    with pytest.raises(ValueError):
        compute_silhouette(X, labels)


# ---------------------------------------------------------------------------
# T6 - <2 unique labels: reference raises; production returns -1.0
# ---------------------------------------------------------------------------


def test_t6_single_cluster_reference_raises_production_returns_minus_one():
    """Single-cluster input: reference raises ValueError (matching sklearn).
    Production wrapper returns -1.0 with a logger warning (its documented
    contract). Both behaviours are pinned here so downstream consumers
    cannot silently change them.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        reference_silhouette(X, labels)
    assert compute_silhouette(X, labels) == -1.0


# ---------------------------------------------------------------------------
# Production code matches reference, matches sklearn directly (3-way)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "X_data,labels_data,expected",
    [
        ([[0.0], [0.0], [10.0], [10.0]], [0, 0, 1, 1], 1.0),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 0, 1, 1], 3.0 - 2.0 * sqrt(2.0)),
        ([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]], [0, 0, 0, 1, 1, 1], 419.0 / 504.0),
        ([[0.0], [10.0], [1.0], [9.0]], [0, 0, 1, 1], -7.0 / 16.0),
    ],
)
def test_three_way_agreement_on_closed_form_cases(X_data, labels_data, expected):
    """compute_silhouette ↔ reference_silhouette ↔ sklearn.silhouette_score
    all match the hand value (1e-12).
    """
    X = np.asarray(X_data, dtype=np.float64)
    labels = np.asarray(labels_data)

    prod = compute_silhouette(X, labels)
    ref = reference_silhouette(X, labels)
    skl = float(silhouette_score(X, labels))

    assert abs(prod - expected) < TOL
    assert abs(ref - expected) < TOL
    assert abs(skl - expected) < TOL
    assert abs(prod - ref) < TOL
    assert abs(prod - skl) < TOL


# ---------------------------------------------------------------------------
# Boundedness sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "X_data,labels_data",
    [
        ([[0.0], [0.0], [10.0], [10.0]], [0, 0, 1, 1]),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 0, 1, 1]),
        ([[0.0], [10.0], [1.0], [9.0]], [0, 0, 1, 1]),
    ],
)
def test_silhouette_bounded(X_data, labels_data):
    """Silhouette score lies in [-1, 1] on every closed-form case."""
    X = np.asarray(X_data, dtype=np.float64)
    labels = np.asarray(labels_data)
    s = reference_silhouette(X, labels)
    assert -1.0 <= s <= 1.0
