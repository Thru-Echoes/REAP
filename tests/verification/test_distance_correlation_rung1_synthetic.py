"""Rung 1 - synthetic-data verification for the two distance-correlation recipes.

Verifies, on a fixed seeded synthetic embedding pair, that:

1. the production reported metric (``compute_distance_correlation``) matches
   the from-scratch condensed-recipe reference;
2. the production torch loss component matches the from-scratch full-matrix
   recipe reference (float64 end to end);
3. the two recipes genuinely disagree on realistic inputs, by a measured,
   recorded amount (the parity gap the roadmap requires on a committed
   fixture); and
4. the disagreement is entirely the diagonal: dropping the diagonal from the
   full-matrix recipe reproduces the condensed recipe to float rounding.

Fixture: deterministic ``numpy.random.default_rng(20260702)`` draw —
``Y_TRUE`` is a 64×6 standard-normal point set, ``Y_PRED`` adds 0.15-scaled
normal noise. Committed in code; no artifact files needed.

Tolerances (documented per the scientific-conventions rule):

- 1e-12 between numpy-only paths (production metric vs condensed reference;
  off-diagonal vs condensed): all float64 Pearson arithmetic differing only
  in reduction order (numpy corrcoef vs explicit sums); observed agreement
  is at the 1e-15 scale, so 1e-12 leaves three orders of headroom.
- 1e-9 between the production torch loss and the numpy full-matrix
  reference: ``torch.cdist`` computes Euclidean distances via the
  matrix-multiplication expansion (‖x‖² + ‖y‖² − 2·x·y), a genuinely
  different rounding path from the reference's direct
  sqrt-of-squared-differences; the observed disagreement on this fixture is
  ≈ 9.4e-12. 1e-9 gives two orders of headroom while still pinning every
  digit a manuscript could quote — and the fact that even the *distances*
  round differently is one more reason the loss is registered as its own
  recipe (``dist_corr_loss``) rather than treated as interchangeable with
  the reported metric.
- 1e-9 on the recorded fixture value and gap: differences of float64
  correlations of order 1; platform-to-platform wobble is bounded by the
  same last-ulp effects, far below 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_distance_correlation import (
    reference_dist_corr_full_matrix,
    reference_dist_corr_full_matrix_offdiagonal,
    reference_dist_corr_loss_value,
    reference_distance_correlation_condensed,
)

from reap.evaluation import compute_distance_correlation

REFERENCE_TOL = 1e-12
TORCH_REFERENCE_TOL = 1e-9
GAP_TOL = 1e-9

_rng = np.random.default_rng(20260702)
Y_TRUE = _rng.normal(size=(64, 6))
Y_PRED = Y_TRUE + 0.15 * _rng.normal(size=(64, 6))

# Measured once on the fixture above and recorded at full float64 precision
# (see module docstring for the tolerance rationale). The gap is positive:
# the 64 diagonal zeros act as perfectly-agreeing samples and pull the
# full-matrix correlation upward.
RECORDED_CONDENSED = 0.978644479703091
RECORDED_GAP = 0.0033565004959558165


def test_s1_production_metric_matches_condensed_reference():
    """``compute_distance_correlation`` implements exactly the condensed recipe."""
    ref = reference_distance_correlation_condensed(Y_TRUE, Y_PRED)
    prod = compute_distance_correlation(Y_TRUE, Y_PRED)
    assert prod == pytest.approx(ref, abs=REFERENCE_TOL)


def test_s2_production_loss_matches_full_matrix_reference():
    """The torch loss component implements exactly the full-matrix recipe."""
    torch = pytest.importorskip("torch")
    from reap.projection import compute_projection_loss

    _, metrics = compute_projection_loss(
        torch.tensor(Y_PRED, dtype=torch.float64),
        torch.tensor(Y_TRUE, dtype=torch.float64),
        alpha=0.0,
    )
    ref_loss = reference_dist_corr_loss_value(Y_PRED, Y_TRUE)
    assert metrics["dist_corr_loss"] == pytest.approx(ref_loss, abs=TORCH_REFERENCE_TOL)


def test_s3_recorded_condensed_value():
    """The condensed recipe's value on the fixture, pinned to recorded digits."""
    assert RECORDED_CONDENSED is not None, "measure and record the fixture value"
    ref = reference_distance_correlation_condensed(Y_TRUE, Y_PRED)
    assert ref == pytest.approx(RECORDED_CONDENSED, abs=GAP_TOL)


def test_s4_recipes_disagree_by_recorded_gap():
    """The parity pin: full-matrix minus condensed equals the recorded gap.

    The two recipes are different calculations and must never be quoted
    interchangeably; this records exactly how far apart they sit on a
    realistic fixture (compare the corpus-scale observation of 0.854 vs
    0.892 on identical inputs that motivated the recipe split).
    """
    assert RECORDED_GAP is not None, "measure and record the fixture gap"
    gap = reference_dist_corr_full_matrix(Y_TRUE, Y_PRED) - (
        reference_distance_correlation_condensed(Y_TRUE, Y_PRED)
    )
    assert gap == pytest.approx(RECORDED_GAP, abs=GAP_TOL)
    assert gap > 1e-4  # material disagreement, not accumulated rounding


def test_s5_gap_is_entirely_the_diagonal():
    """Dropping only the diagonal reproduces the condensed recipe (≤ 1e-12)."""
    off_diag = reference_dist_corr_full_matrix_offdiagonal(Y_TRUE, Y_PRED)
    condensed = reference_distance_correlation_condensed(Y_TRUE, Y_PRED)
    assert off_diag == pytest.approx(condensed, abs=REFERENCE_TOL)
