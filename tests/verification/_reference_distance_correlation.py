"""From-scratch implementations of REAP's two distance-correlation recipes.

Imports nothing from ``reap``, ``scipy``, ``sklearn``, or ``torch``. Used to
independently verify the two production calculations that share the everyday
name "distance correlation" (neither is Székely's distance correlation):

- ``distance_correlation`` — the *reported metric*
  (``reap.evaluation.compute_distance_correlation``): Pearson correlation
  between **condensed** pairwise-distance vectors — the n·(n−1)/2 unique
  Euclidean self-distances of each point set, no diagonal, each pair once.
- ``dist_corr_loss`` — the *training-loss component*
  (``reap.projection.compute_projection_loss``): 1 − Pearson correlation
  over the **flattened full** n×n self-distance matrices — diagonal zeros
  included, every pair counted twice.

Why they disagree on generic inputs: Pearson correlation is invariant to
duplicating every sample (all centered sums scale by 2, which cancels), so
double-counting the pairs is harmless. The n diagonal zeros are NOT
harmless — they shift both means and contribute n perfectly-agreeing (0, 0)
samples, so the full-matrix correlation differs from the condensed one.
The rung tests pin the exact gap on fixed fixtures.

Degenerate-input guards mirror the production recipes exactly (they are part
of the recipe, not incidental): the condensed recipe returns 0.0 when either
condensed vector has population standard deviation < 1e-12; the full-matrix
recipe returns a correlation of 0.0 when the product of the centered vector
norms is < 1e-12.

Implementation notes
--------------------
- Pairwise Euclidean distances are computed by explicit broadcasting and a
  square root — no library distance function.
- Pearson correlation is computed from first principles (centered dot
  product over the product of centered norms) in float64.
"""

from __future__ import annotations

import numpy as np


def _self_distance_matrix(Y: np.ndarray) -> np.ndarray:
    """Full n×n Euclidean self-distance matrix of a point set, float64."""
    Y = np.asarray(Y, dtype=np.float64)
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _condensed_self_distances(Y: np.ndarray) -> np.ndarray:
    """Unique-pair (i<j) Euclidean self-distances, in row-major pair order.

    Matches the pair ordering of ``scipy.spatial.distance.pdist``:
    (0,1), (0,2), …, (0,n−1), (1,2), …, (n−2,n−1).
    """
    D = _self_distance_matrix(Y)
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return D[iu, ju]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """First-principles Pearson correlation of two equal-length vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c * a_c).sum()) * np.sqrt((b_c * b_c).sum())
    return float((a_c * b_c).sum() / denom)


def reference_distance_correlation_condensed(Y_a: np.ndarray, Y_b: np.ndarray) -> float:
    """The reported-metric recipe: Pearson r of condensed self-distance vectors.

    Mirrors ``reap.evaluation.compute_distance_correlation`` (recipe id
    ``distance_correlation``): unique pairs only, no diagonal, and the
    production guard — 0.0 when either condensed vector's population standard
    deviation is < 1e-12.

    Parameters
    ----------
    Y_a, Y_b : (n, d) point sets whose self-distance structures are compared.

    Returns
    -------
    Pearson correlation in [-1, 1], or 0.0 for degenerate inputs.
    """
    d_a = _condensed_self_distances(Y_a)
    d_b = _condensed_self_distances(Y_b)
    if d_a.std() < 1e-12 or d_b.std() < 1e-12:
        return 0.0
    return _pearson(d_a, d_b)


def reference_dist_corr_full_matrix(Y_a: np.ndarray, Y_b: np.ndarray) -> float:
    """The loss recipe's correlation: Pearson r over flattened full matrices.

    Mirrors the distance-correlation term inside
    ``reap.projection.compute_projection_loss`` (recipe id
    ``dist_corr_loss`` = 1 − this value): the full n×n self-distance
    matrices are flattened with the diagonal zeros included and every pair
    counted twice, and the production guard returns 0.0 when the product of
    the centered vectors' norms is < 1e-12.

    Parameters
    ----------
    Y_a, Y_b : (n, d) point sets whose self-distance structures are compared.

    Returns
    -------
    Pearson correlation in [-1, 1], or 0.0 for degenerate inputs.
    """
    d_a = _self_distance_matrix(Y_a).flatten()
    d_b = _self_distance_matrix(Y_b).flatten()
    a_c = d_a - d_a.mean()
    b_c = d_b - d_b.mean()
    denom = np.sqrt((a_c * a_c).sum()) * np.sqrt((b_c * b_c).sum())
    if denom < 1e-12:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def reference_dist_corr_full_matrix_offdiagonal(Y_a: np.ndarray, Y_b: np.ndarray) -> float:
    """Full-matrix recipe WITHOUT the diagonal (pairs still counted twice).

    Exists to isolate the two recipes' single point of disagreement: Pearson
    correlation is invariant to the double-counting, so this quantity equals
    the condensed recipe up to float rounding — the diagonal zeros are the
    entire gap between ``distance_correlation`` and 1 − ``dist_corr_loss``.

    Parameters
    ----------
    Y_a, Y_b : (n, d) point sets whose self-distance structures are compared.

    Returns
    -------
    Pearson correlation in [-1, 1] over the off-diagonal entries.
    """
    D_a = _self_distance_matrix(Y_a)
    D_b = _self_distance_matrix(Y_b)
    mask = ~np.eye(D_a.shape[0], dtype=bool)
    return _pearson(D_a[mask], D_b[mask])


def reference_dist_corr_loss_value(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    """The ``dist_corr_loss`` recipe value: 1 − full-matrix Pearson r.

    Numpy-float64 mirror of the loss component reported in
    ``compute_projection_loss``'s metrics dict under ``"dist_corr_loss"``.

    Parameters
    ----------
    Y_pred, Y_true : (n, d) point sets (prediction first, matching the loss).

    Returns
    -------
    Loss value in [0, 2] (1.0 for degenerate inputs, via the 0.0-corr guard).
    """
    return 1.0 - reference_dist_corr_full_matrix(Y_pred, Y_true)
