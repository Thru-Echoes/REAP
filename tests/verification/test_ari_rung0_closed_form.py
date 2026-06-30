"""Rung 0 - closed-form known-answer tests for ARI (verification ladder).

Hand-computed expected values; every assertion uses a value that can be
re-derived in <5 lines of arithmetic from the ARI definition
(``ARI = (S - E) / (M - E)``, see ``_reference_ari``). Tolerance: 1e-12.

Gate: any RED in this file blocks Rung 1. If RED, run
``superpowers:systematic-debugging`` on ``src/reap/evaluation.py`` first,
then restart the ladder from Rung 0 (a fix invalidates lower rungs).

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), Task 1.
"""

from __future__ import annotations

import numpy as np
from _reference_ari import reference_ari

from reap.evaluation import compute_pairwise_ari, compute_seed_stability

TOL = 1e-12


# ---------------------------------------------------------------------------
# T1 - perfect agreement; ARI = 1
# ---------------------------------------------------------------------------


def test_identical_partitions_give_ari_1():
    """Identical partitions: ARI = 1 exactly."""
    assert reference_ari([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_label_permutation_does_not_change_ari():
    """ARI is invariant to label permutation: [0,0,1,1] equiv [1,1,0,0]."""
    assert reference_ari([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0


# ---------------------------------------------------------------------------
# T2 - "maximally independent" 2x2 partitions: ARI = -0.5
# ---------------------------------------------------------------------------


def test_independent_partition_is_minus_half():
    """Hand-derived: ARI([0,0,1,1], [0,1,0,1]) = -0.5.

    Contingency 2x2, all cells = 1; row sums = col sums = 2; n = 4.
        S = 4 * C(1,2) = 0
        Sum_i C(a_i,2) = 2 * C(2,2) = 2
        Sum_j C(b_j,2) = 2 * C(2,2) = 2
        C(n,2) = 6
        E = (2 * 2) / 6 = 2/3
        M = (1/2)(2 + 2) = 2
        ARI = (0 - 2/3) / (2 - 2/3) = (-2/3) / (4/3) = -1/2.
    """
    value = reference_ari([0, 0, 1, 1], [0, 1, 0, 1])
    assert abs(value - (-0.5)) < TOL


# ---------------------------------------------------------------------------
# T3 - partial agreement: ARI = 1.2 / 3.7 = 0.3243243243243243
# ---------------------------------------------------------------------------


def test_partial_agreement_matches_hand_value():
    """Hand-derived: ARI([0,0,0,1,1,1], [0,0,1,1,1,1]) = 1.2/3.7.

    Contingency:                  Row sums: a_0 = 3, a_1 = 3
            b=0  b=1               Col sums: b_0 = 2, b_1 = 4
        a=0   2    1               S = C(2,2)+C(1,2)+C(0,2)+C(3,2) = 1+0+0+3 = 4
        a=1   0    3               Sum_i C(a_i,2) = 2*C(3,2) = 6
                                   Sum_j C(b_j,2) = C(2,2)+C(4,2) = 1+6 = 7
                                   C(n,2) = C(6,2) = 15
                                   E = (6 * 7) / 15 = 2.8
                                   M = (1/2)(6 + 7) = 6.5
                                   ARI = (4 - 2.8) / (6.5 - 2.8)
                                       = 1.2 / 3.7
                                       = 0.3243243243243243.
    """
    value = reference_ari([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1])
    assert abs(value - 1.2 / 3.7) < TOL
    assert abs(value - 0.3243243243243243) < TOL


# ---------------------------------------------------------------------------
# T4 - code under test (compute_pairwise_ari) matches hand values and reference
# ---------------------------------------------------------------------------


def test_compute_pairwise_ari_matches_hand_values_and_reference():
    """A 3-labeling pairwise matrix matches all closed-form pair values exactly."""
    labels_list = [
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
        np.array([0, 1, 0, 1]),
    ]
    matrix = compute_pairwise_ari(labels_list)

    assert matrix.shape == (3, 3)
    assert np.array_equal(matrix, matrix.T)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 2] == 1.0

    assert abs(matrix[0, 1] - 1.0) < TOL
    assert abs(matrix[0, 2] - (-0.5)) < TOL
    assert abs(matrix[1, 2] - (-0.5)) < TOL

    assert abs(matrix[0, 1] - reference_ari(labels_list[0], labels_list[1])) < TOL
    assert abs(matrix[0, 2] - reference_ari(labels_list[0], labels_list[2])) < TOL
    assert abs(matrix[1, 2] - reference_ari(labels_list[1], labels_list[2])) < TOL


# ---------------------------------------------------------------------------
# T5 - compute_seed_stability: all six statistics match hand values exactly
# ---------------------------------------------------------------------------


def test_compute_seed_stability_six_hand_values():
    """seeds = [[0,0,1,1], [0,0,1,1], [0,1,0,1]], consensus = [0,0,1,1].

    Pairwise ARIs over upper triangle (k=1): [(0,1)=1.0, (0,2)=-0.5, (1,2)=-0.5].
        s2s_ari_mean   = (1.0 + (-0.5) + (-0.5)) / 3 = 0.0
        s2s_ari_std    = sqrt(mean((x - mean)^2))
                       = sqrt(((1.0)^2 + (-0.5)^2 + (-0.5)^2) / 3)
                       = sqrt(0.5)
                       = 0.7071067811865476           (population std, ddof=0)
        s2s_ari_median = -0.5                          (sorted: -0.5, -0.5, 1.0)

    s2c (each seed vs consensus): [1.0, 1.0, -0.5].
        s2c_ari_mean   = (1.0 + 1.0 + (-0.5)) / 3 = 0.5
        s2c_ari_std    = sqrt(((1-0.5)^2 + (1-0.5)^2 + (-0.5-0.5)^2) / 3)
                       = sqrt((0.25 + 0.25 + 1.0) / 3)
                       = sqrt(0.5)
                       = 0.7071067811865476
        s2c_ari_median = 1.0                           (sorted: -0.5, 1.0, 1.0)
    """
    seeds = [
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
    ]
    consensus = np.array([0, 0, 1, 1])
    stats = compute_seed_stability(seeds, consensus)

    sqrt_half = 0.7071067811865476

    assert abs(stats["s2s_ari_mean"] - 0.0) < TOL
    assert abs(stats["s2s_ari_std"] - sqrt_half) < TOL
    assert abs(stats["s2s_ari_median"] - (-0.5)) < TOL

    assert abs(stats["s2c_ari_mean"] - 0.5) < TOL
    assert abs(stats["s2c_ari_std"] - sqrt_half) < TOL
    assert abs(stats["s2c_ari_median"] - 1.0) < TOL


# ---------------------------------------------------------------------------
# Sanity - the reference impl on its own pinned closed-form values
# ---------------------------------------------------------------------------


def test_reference_ari_on_closed_form_battery():
    """Roll-up: each closed-form pair returns the exact hand-computed value."""
    assert reference_ari([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert reference_ari([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert abs(reference_ari([0, 0, 1, 1], [0, 1, 0, 1]) - (-0.5)) < TOL
    assert abs(
        reference_ari([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]) - 0.3243243243243243
    ) < TOL
