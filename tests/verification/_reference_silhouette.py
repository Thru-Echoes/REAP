"""From-scratch silhouette score for the verification ladder.

Imports nothing from ``reap`` and nothing from ``sklearn``. Used by
``tests/verification/test_silhouette_rung*.py`` to independently verify
``reap.evaluation.compute_silhouette`` against an implementation derived
from the textbook silhouette definition (Rousseeuw 1987).

Definition. For a sample i in cluster C_i = (the cluster containing i),

    a(i) = mean distance from i to other points in C_i
                                                       (cohesion).
    b(i) = min over other clusters C of (mean distance from i to points in C)
                                                       (nearest-cluster separation).
    s(i) = (b(i) - a(i)) / max(a(i), b(i)),   if |C_i| > 1
    s(i) = 0,                                  if |C_i| == 1     (Rousseeuw convention)

The silhouette score is the mean of s(i) over all i.

Implementation notes
--------------------
- Distances are computed pairwise from `X` using the requested `metric`.
  Supported here: ``"euclidean"`` and ``"cosine"`` (matches the metrics
  ``reap.evaluation.compute_silhouette`` actually invokes).
- The Rousseeuw convention ``s(i) = 0 for singleton clusters`` matches
  sklearn 1.x behaviour.
- The score is undefined if fewer than 2 unique cluster labels are present;
  this function raises ``ValueError`` in that case (sklearn does the same).
"""

from __future__ import annotations

import numpy as np


def _pairwise_distance(X: np.ndarray, metric: str) -> np.ndarray:
    """Pairwise distance matrix; supports euclidean and cosine."""
    X = np.asarray(X, dtype=np.float64)
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1))
    if metric == "cosine":
        # Normalize then 1 - cos
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid div-by-zero for zero-vectors (Rousseeuw doesn't address this;
        # neither does sklearn cleanly. Treat zero vectors as orthogonal:
        # cosine distance from any direction is 1).
        unit = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
        sim = unit @ unit.T
        return 1.0 - sim
    raise ValueError(f"unsupported metric: {metric!r}")


def reference_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Mean Rousseeuw silhouette over the rows of ``X`` given ``labels``.

    Parameters
    ----------
    X : (n_samples, n_features) float array.
    labels : (n_samples,) integer labels.
    metric : "euclidean" or "cosine".

    Returns
    -------
    Mean silhouette score in [-1, 1].

    Raises
    ------
    ValueError if fewer than 2 unique labels are present (matches sklearn).
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"silhouette score undefined for fewer than 2 clusters; got "
            f"{unique_labels.size} unique label(s)"
        )

    distances = _pairwise_distance(X, metric=metric)

    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels[i]
        own_mask = (labels == own) & (np.arange(n) != i)
        if own_mask.sum() == 0:
            # Singleton cluster -> Rousseeuw convention s(i) = 0.
            s[i] = 0.0
            continue
        a_i = float(distances[i, own_mask].mean())

        b_values: list[float] = []
        for other in unique_labels:
            if other == own:
                continue
            other_mask = labels == other
            if other_mask.sum() == 0:
                continue
            b_values.append(float(distances[i, other_mask].mean()))
        b_i = min(b_values)

        denom = max(a_i, b_i)
        if denom == 0.0:
            s[i] = 0.0
        else:
            s[i] = (b_i - a_i) / denom
    return float(s.mean())
