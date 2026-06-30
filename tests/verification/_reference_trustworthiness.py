"""From-scratch trustworthiness score for the verification ladder.

Imports nothing from ``reap`` and nothing from ``sklearn``. Used to
independently verify ``reap.evaluation.compute_trustworthiness`` against an
implementation derived from the Venna & Kaski (2001) definition.

Definition (Venna & Kaski 2001, "Neighborhood Preservation in
Nonlinear Projection Methods: An Experimental Study"):

    T(k) = 1 - (2 / (n*k*(2n - 3k - 1))) * sum_i  sum_{j in U_k(i)}  (r(i,j) - k)

where:
    n        = number of samples
    k        = neighborhood size (``n_neighbors``)
    U_k(i)   = the set of points that are in i's k-nearest-neighbors in
               *low* dimensional space but NOT in its k-nearest-neighbors
               in *high* dimensional space (the "intruders" / false
               neighbors)
    r(i, j)  = the rank of j among the high-dimensional neighbors of i,
               1-indexed (1 = closest), excluding i itself.

Trustworthiness lies in [0, 1] (it is exactly 1 when every k-NN set is
preserved; lower when low-d brings non-neighbors into the local view).

The standard definition requires ``k < (n - 1) / 2`` (equivalently
``2n - 3k - 1 > 0``) for the normalizer to be well-posed; sklearn
enforces ``n_neighbors < n_samples / 2``.

Implementation notes
--------------------
- Distances are computed pairwise. Supported metrics: ``"euclidean"``
  (the production default for UMAP output space) and ``"cosine"`` (for
  high-d sentence embeddings).
- Ties in the high-d ranking are broken by index order (consistent with
  numpy's stable ``argsort`` default).
"""

from __future__ import annotations

import numpy as np


def _pairwise(X: np.ndarray, metric: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1))
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        unit = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
        sim = unit @ unit.T
        return 1.0 - sim
    raise ValueError(f"unsupported metric: {metric!r}")


def reference_trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> float:
    """Trustworthiness score from first principles.

    Parameters
    ----------
    X_high : (n, d_high) original-space coordinates.
    X_low  : (n, d_low)  reduced-space coordinates.
    n_neighbors : neighborhood size k. Must satisfy ``k < n / 2``.
    metric : "euclidean" or "cosine" (matches the metrics used in REAP).

    Returns
    -------
    Trustworthiness in [0, 1].

    Raises
    ------
    ValueError if the normalizer would be non-positive
    (``2*n - 3*k - 1 <= 0``), matching sklearn's contract.
    """
    X_high = np.asarray(X_high, dtype=np.float64)
    X_low = np.asarray(X_low, dtype=np.float64)
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(
            f"X_high and X_low must have the same number of rows; "
            f"got {X_high.shape[0]} vs {X_low.shape[0]}"
        )
    n = X_high.shape[0]
    k = int(n_neighbors)
    if 2 * n - 3 * k - 1 <= 0:
        raise ValueError(
            f"trustworthiness undefined: 2*n - 3*k - 1 = {2 * n - 3 * k - 1} "
            f"<= 0 (need k < (n - 1) / 2; got n={n}, k={k})"
        )

    distances_high = _pairwise(X_high, metric=metric)
    distances_low = _pairwise(X_low, metric=metric)

    # Exclude self from neighborhoods by setting the diagonal to +inf
    np.fill_diagonal(distances_high, np.inf)
    np.fill_diagonal(distances_low, np.inf)

    # rank_high[i, j] = position of j in i's high-d neighbor list (1-indexed,
    # 1 = closest, self is not counted because diagonal is inf).
    rank_high = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        order = np.argsort(distances_high[i], kind="stable")
        # order[0] is the closest non-self; order[1] is second closest; etc.
        # We assign rank 1 to order[0], rank 2 to order[1], ...
        # Note: order also includes i itself (since argsort over diag-inf row
        # puts i last); its rank is n (used by sklearn semantics).
        for pos, idx in enumerate(order, start=1):
            rank_high[i, idx] = pos

    nn_low = np.argsort(distances_low, axis=1, kind="stable")[:, :k]
    nn_high = np.argsort(distances_high, axis=1, kind="stable")[:, :k]
    nn_high_sets = [set(row.tolist()) for row in nn_high]

    total_penalty = 0
    for i in range(n):
        for j in nn_low[i].tolist():
            if j not in nn_high_sets[i]:
                total_penalty += int(rank_high[i, j] - k)

    normalizer = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - normalizer * total_penalty)
