"""Rung 0 - closed-form known-answer tests for the two distance-correlation recipes.

REAP computes "distance correlation" two subtly different ways under one
everyday name (neither is Székely's distance correlation):

- ``distance_correlation`` (reported metric,
  ``reap.evaluation.compute_distance_correlation``): Pearson r between
  condensed pairwise-distance vectors — unique pairs, no diagonal.
- ``dist_corr_loss`` (training-loss component,
  ``reap.projection.compute_projection_loss``): 1 − Pearson r over the
  flattened full self-distance matrices — diagonal included, pairs twice.

Pearson r is invariant to counting every pair twice (all centered sums scale
by 2, which cancels), so the double-counting is harmless. The n diagonal
zeros are NOT: they shift both means and add n perfectly-agreeing (0, 0)
samples. This module pins each recipe, and their exact gap, on a
hand-computable fixture.

Hand derivation for the 4-point line fixture (D3), Euclidean, 1-D:

    A = [0, 1, 2, 4],   B = [0, 1, 2, 5]

    condensed pair order (i<j): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    d_A = [1, 2, 4, 1, 3, 2]     mean = 13/6
    d_B = [1, 2, 5, 1, 4, 3]     mean = 16/6 = 8/3

    S_ab = Σ(d_A−ā)(d_B−b̄) = (35 + 2 + 77 + 35 + 20 − 1)/18 = 168/18 = 28/3
    S_aa = Σ(d_A−ā)²  = (49 + 1 + 121 + 49 + 25 + 1)/36 = 246/36 = 41/6
    S_bb = Σ(d_B−b̄)²  = (25 + 4 + 49 + 25 + 16 + 1)/9  = 120/9  = 40/3

    r_condensed = (28/3) / sqrt((41/6)·(40/3)) = 14 / sqrt(205) ≈ 0.9778024

    Full-matrix vectors (n=4 → 16 entries: each condensed entry twice plus
    4 diagonal zeros):
    mean_A = 2·13/16 = 13/8,   mean_B = 2·16/16 = 2

    S_ab_full = 2·(23/2) + 4·(13/8)·2      = 23 + 13  = 36
    S_aa_full = 2·(550/64) + 4·(169/64)    = 1776/64  = 111/4
    S_bb_full = 2·16 + 4·4                 = 48
    r_full = 36 / sqrt((111/4)·48) = 36 / sqrt(1332) = 6 / sqrt(37) ≈ 0.9863939

    Closed-form recipe gap on this fixture:
    r_full − r_condensed = 6/√37 − 14/√205 ≈ +0.0085915
    (positive: the diagonal zeros act as n extra perfectly-agreeing samples
    and *raise* the full-matrix correlation on this fixture).

Tolerance 1e-12: both recipes are plain float64 Pearson arithmetic; the only
deviation from the exact fractions is last-ulp rounding in the reductions and
the final square root (same rationale as the sibling rung-0 modules).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _reference_distance_correlation import (
    reference_dist_corr_full_matrix,
    reference_dist_corr_full_matrix_offdiagonal,
    reference_distance_correlation_condensed,
)

from reap.evaluation import compute_distance_correlation

TOL = 1e-12

POINTS_A = np.array([[0.0], [1.0], [2.0], [4.0]])
POINTS_B = np.array([[0.0], [1.0], [2.0], [5.0]])

# Hand-derived closed forms (derivation in the module docstring).
EXPECTED_CONDENSED = 14.0 / math.sqrt(205.0)
EXPECTED_FULL = 6.0 / math.sqrt(37.0)


# ---------------------------------------------------------------------------
# D1 - identity: both recipes give r = 1.0
# ---------------------------------------------------------------------------


def test_d1_identity_condensed():
    """Identical point sets ⇒ condensed vectors identical ⇒ r = 1.0."""
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    assert reference_distance_correlation_condensed(Y, Y) == pytest.approx(1.0, abs=TOL)
    assert compute_distance_correlation(Y, Y) == pytest.approx(1.0, abs=TOL)


def test_d1_identity_full_matrix():
    """Identical point sets ⇒ full flattened matrices identical ⇒ r = 1.0."""
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    assert reference_dist_corr_full_matrix(Y, Y) == pytest.approx(1.0, abs=TOL)


# ---------------------------------------------------------------------------
# D2 - global scaling: distances scale linearly, Pearson r stays 1.0
# ---------------------------------------------------------------------------


def test_d2_scaling_invariance_both_recipes():
    """Y_b = 3·Y_a scales every distance by 3 ⇒ r = 1.0 under BOTH recipes.

    Note this closed form does NOT separate the recipes: the diagonal zeros
    scale to zero, so the full-matrix vectors stay exactly proportional too.
    Separation needs a non-proportional distance change (D3).
    """
    Y = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0], [4.0, 4.0]])
    assert reference_distance_correlation_condensed(Y, 3.0 * Y) == pytest.approx(1.0, abs=TOL)
    assert compute_distance_correlation(Y, 3.0 * Y) == pytest.approx(1.0, abs=TOL)
    assert reference_dist_corr_full_matrix(Y, 3.0 * Y) == pytest.approx(1.0, abs=TOL)


# ---------------------------------------------------------------------------
# D3 - the hand-derived 4-point case: each recipe's value and their gap
# ---------------------------------------------------------------------------


def test_d3_condensed_recipe_closed_form():
    """Condensed recipe on the fixture = 14/√205 (reference and production)."""
    assert reference_distance_correlation_condensed(POINTS_A, POINTS_B) == pytest.approx(
        EXPECTED_CONDENSED, abs=TOL
    )
    assert compute_distance_correlation(POINTS_A, POINTS_B) == pytest.approx(
        EXPECTED_CONDENSED, abs=TOL
    )


def test_d3_full_matrix_recipe_closed_form():
    """Full-matrix recipe on the fixture = 6/√37."""
    assert reference_dist_corr_full_matrix(POINTS_A, POINTS_B) == pytest.approx(
        EXPECTED_FULL, abs=TOL
    )


def test_d3_loss_component_matches_full_matrix_closed_form():
    """Production loss (α=0) on the fixture = 1 − 6/√37, in float64 torch.

    ``compute_projection_loss`` with ``alpha=0`` reduces to the pure
    ``dist_corr_loss`` term, so this pins the production torch calculation —
    not just the numpy reference — to the closed form.
    """
    pytest.importorskip("torch")
    metrics = _projection_loss_metrics(POINTS_B, POINTS_A, alpha=0.0)
    assert metrics["total"] == pytest.approx(1.0 - EXPECTED_FULL, abs=TOL)
    assert metrics["dist_corr_loss"] == pytest.approx(1.0 - EXPECTED_FULL, abs=TOL)


def _projection_loss_metrics(
    Y_pred: np.ndarray, Y_true: np.ndarray, alpha: float
) -> dict[str, float]:
    """Run the production loss on float64 tensors; return its metrics dict."""
    import torch

    from reap.projection import compute_projection_loss

    _, metrics = compute_projection_loss(
        torch.tensor(Y_pred, dtype=torch.float64),
        torch.tensor(Y_true, dtype=torch.float64),
        alpha=alpha,
    )
    return metrics


def test_d3_recipe_gap_closed_form():
    """The two recipes disagree on the fixture by exactly 6/√37 − 14/√205.

    This is the definitional pin: one everyday name, two calculations,
    closed-form different values on the same four points.
    """
    gap = reference_dist_corr_full_matrix(POINTS_A, POINTS_B) - (
        reference_distance_correlation_condensed(POINTS_A, POINTS_B)
    )
    assert gap == pytest.approx(EXPECTED_FULL - EXPECTED_CONDENSED, abs=TOL)
    assert gap > 8e-3  # the disagreement is material, not float noise


def test_d3_double_counting_cancels_diagonal_does_not():
    """Off-diagonal full-matrix r equals the condensed r; the diagonal is the gap.

    Pearson r is invariant to duplicating every sample, so dropping only the
    diagonal from the full-matrix recipe must reproduce the condensed recipe
    to float rounding. The remaining difference between the two registered
    recipes is therefore attributable entirely to the diagonal zeros.
    """
    off_diag = reference_dist_corr_full_matrix_offdiagonal(POINTS_A, POINTS_B)
    condensed = reference_distance_correlation_condensed(POINTS_A, POINTS_B)
    assert off_diag == pytest.approx(condensed, abs=TOL)


# ---------------------------------------------------------------------------
# D4 - degenerate guard: identical points ⇒ 0.0 under both recipes
# ---------------------------------------------------------------------------


def test_d4_degenerate_guard_returns_zero():
    """All-identical points ⇒ zero-variance distances ⇒ guarded 0.0 (both)."""
    Y = np.zeros((5, 3))
    other = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    assert reference_distance_correlation_condensed(Y, other) == 0.0
    assert compute_distance_correlation(Y, other) == 0.0
    assert reference_dist_corr_full_matrix(Y, other) == 0.0
