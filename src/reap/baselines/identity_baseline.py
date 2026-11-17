"""Identity baseline: place each new document where its nearest reference document sits.

The simplest possible way to put new documents into an existing reference map
needs no training: find each new document's nearest reference document in the
*original* embedding space and copy that reference document's coordinate in the
reference map (the consensus embedding). Protocol §18.d registers this as the
"identity" baseline for the hold-out evaluations: a trained projection head
that cannot beat "copy your nearest neighbour" adds nothing for that corpus
(protocol v1.5 clause e).

The same nearest-neighbour search also gives each new document's highest
similarity to any reference document, which is the quantity the leakage audit
for random hold-outs reports (protocol v1.5 clause f), so the result carries it.

The baseline is deterministic: no random state, and ties go to the lowest
reference index.

Exports: ``IdentityProjectionResult``, ``get_identity_projection``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class IdentityProjectionResult(BaseModel):
    """Output of the identity baseline for one set of new documents."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: Literal["identity"] = Field("identity", description="Baseline identifier.")
    metric: Literal["cosine", "euclidean"] = Field(
        ..., description="Distance used to find the nearest reference document."
    )
    projection: np.ndarray = Field(
        ...,
        description="(n_query, n_components) copied reference-map coordinates, one row "
        "per new document.",
    )
    nearest_reference_indices: np.ndarray = Field(
        ..., description="(n_query,) index of each new document's nearest reference document."
    )
    nearest_reference_similarity: np.ndarray = Field(
        ...,
        description="(n_query,) cosine similarity to that nearest reference document "
        "(always cosine, whatever ``metric`` is; used by the leakage audit).",
    )


def _get_unit_rows(X: np.ndarray, name: str) -> np.ndarray:
    """Return ``X`` with each row scaled to unit length.

    Raises
    ------
    ValueError : If any row has zero length, for which cosine similarity is
        undefined.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        bad = int(np.flatnonzero(norms[:, 0] == 0)[0])
        raise ValueError(f"{name} row {bad} has zero length; cosine similarity is undefined")
    return X / norms


def get_identity_projection(
    X_reference: np.ndarray,
    Y_reference: np.ndarray,
    X_query: np.ndarray,
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> IdentityProjectionResult:
    """Place each query document at its nearest reference document's map coordinate.

    Pure function: no randomness and no side effects. Exact brute-force search,
    so the result does not depend on an approximate index.

    Parameters
    ----------
    X_reference : (n_reference, d) original embeddings of the reference documents.
    Y_reference : (n_reference, n_components) the reference documents' coordinates
        in the reference map, row-aligned with ``X_reference``.
    X_query : (n_query, d) original embeddings of the new documents.
    metric : ``"cosine"`` (protocol §18 default) or ``"euclidean"``.

    Returns
    -------
    IdentityProjectionResult with the copied coordinates, the nearest reference
    index per query, and the cosine similarity to it.

    Raises
    ------
    ValueError : On mismatched shapes, empty inputs, non-finite values, or (for
        cosine) a zero-length row.

    Example
    -------
    >>> import numpy as np
    >>> X_ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> Y_ref = np.array([[10.0, 10.0], [-5.0, 2.0]])
    >>> result = get_identity_projection(X_ref, Y_ref, np.array([[0.9, 0.1]]))
    >>> result.projection.tolist(), result.nearest_reference_indices.tolist()
    ([[10.0, 10.0]], [0])
    """
    X_reference = np.asarray(X_reference, dtype=np.float64)
    Y_reference = np.asarray(Y_reference, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    if X_reference.ndim != 2 or Y_reference.ndim != 2 or X_query.ndim != 2:
        raise ValueError("X_reference, Y_reference and X_query must all be 2-d arrays")
    if X_reference.shape[0] == 0 or X_query.shape[0] == 0:
        raise ValueError("X_reference and X_query must each contain at least one row")
    if X_reference.shape[0] != Y_reference.shape[0]:
        raise ValueError(
            f"X_reference has {X_reference.shape[0]} rows but Y_reference has "
            f"{Y_reference.shape[0]}; they must be row-aligned"
        )
    if X_reference.shape[1] != X_query.shape[1]:
        raise ValueError(
            f"X_reference has {X_reference.shape[1]} features but X_query has "
            f"{X_query.shape[1]}"
        )
    inputs = {"X_reference": X_reference, "Y_reference": Y_reference, "X_query": X_query}
    for name, array in inputs.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")

    similarity = _get_unit_rows(X_query, "X_query") @ _get_unit_rows(
        X_reference, "X_reference"
    ).T
    if metric == "cosine":
        nearest = np.argmax(similarity, axis=1)
    elif metric == "euclidean":
        nearest = np.argmin(cdist(X_query, X_reference, metric="euclidean"), axis=1)
    else:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

    nearest_similarity = similarity[np.arange(X_query.shape[0]), nearest]
    logger.info(
        "Identity baseline: %d queries -> %d reference documents (%s); "
        "median nearest cosine similarity %.4f",
        X_query.shape[0],
        X_reference.shape[0],
        metric,
        float(np.median(nearest_similarity)),
    )
    return IdentityProjectionResult(
        method="identity",
        metric=metric,
        projection=Y_reference[nearest].copy(),
        nearest_reference_indices=nearest.astype(np.int64),
        nearest_reference_similarity=nearest_similarity,
    )
