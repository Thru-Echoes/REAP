"""From-scratch ARI reference implementation for the verification ladder.

Imports nothing from ``reap`` and nothing from ``sklearn``. Used by
``tests/verification/test_ari_rung0_*.py`` to independently verify
``reap.evaluation.compute_pairwise_ari`` and ``compute_seed_stability``
against an implementation derived from the contingency-table formula.

Definition (Hubert & Arabie 1985):

    ARI = (S - E) / (M - E),

    S = Sum over the contingency table of C(n_ij, 2),
    E = (Sum_i C(a_i, 2) * Sum_j C(b_j, 2)) / C(n, 2),
    M = 0.5 * (Sum_i C(a_i, 2) + Sum_j C(b_j, 2)),

where ``n_ij`` is the count of points jointly assigned to cluster ``i`` of
partition A and cluster ``j`` of partition B; ``a_i`` and ``b_j`` are the
row and column sums; ``n`` is the total number of points; and
``C(k, 2) = k*(k-1)/2`` is the number of unordered pairs.

Degenerate-case convention (matches scikit-learn):
    - If ``M - E == 0`` (e.g. both partitions trivial), return 1.0
      (perfect agreement up to the chance correction).
    - If ``n < 2`` (fewer than two points), return 1.0.

ARI is invariant to label permutation by construction — the formula only
depends on co-membership counts, never on label identifiers. No
Hungarian / optimal matching may be applied prior to computation; doing
so does not change the value.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np


def _contingency_table(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Build the co-occurrence contingency table of two label arrays.

    Parameters
    ----------
    labels_a, labels_b : 1-d arrays of integer labels with identical length.

    Returns
    -------
    Contingency table of shape ``(n_clusters_a, n_clusters_b)`` whose
    ``[i, j]`` entry is the number of points assigned to cluster ``i``
    in A and cluster ``j`` in B. Rows and columns are indexed in the
    order returned by ``numpy.unique``.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    _, idx_a = np.unique(a, return_inverse=True)
    _, idx_b = np.unique(b, return_inverse=True)
    n_a = int(idx_a.max()) + 1 if idx_a.size else 0
    n_b = int(idx_b.max()) + 1 if idx_b.size else 0
    contingency = np.zeros((n_a, n_b), dtype=np.int64)
    for i, j in zip(idx_a.tolist(), idx_b.tolist()):
        contingency[i, j] += 1
    return contingency


def reference_ari(
    labels_a: Sequence[int] | np.ndarray,
    labels_b: Sequence[int] | np.ndarray,
) -> float:
    """Adjusted Rand Index computed from first principles.

    Parameters
    ----------
    labels_a, labels_b : 1-d label sequences of the same length.

    Returns
    -------
    Adjusted Rand Index in [-0.5, 1.0]. Returns 1.0 by convention when
    the denominator vanishes (degenerate trivial partitions).

    Raises
    ------
    ValueError if the input arrays have different shapes.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(
            f"label arrays must have the same shape; got {a.shape} vs {b.shape}"
        )
    n = int(a.shape[0])
    if n < 2:
        return 1.0

    contingency = _contingency_table(a, b)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    sum_nij2 = sum(comb(int(c), 2) for c in contingency.ravel().tolist())
    sum_ai2 = sum(comb(int(c), 2) for c in row_sums.tolist())
    sum_bj2 = sum(comb(int(c), 2) for c in col_sums.tolist())
    n_choose_2 = comb(n, 2)

    expected_idx = (sum_ai2 * sum_bj2) / n_choose_2
    max_idx = 0.5 * (sum_ai2 + sum_bj2)

    denominator = max_idx - expected_idx
    numerator = sum_nij2 - expected_idx
    if denominator == 0:
        return 1.0
    return numerator / denominator
