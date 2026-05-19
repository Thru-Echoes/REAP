"""Out-of-sample conformal filter for REAP-projected coordinates.

Implements the methodology described in manuscript §3.6 and pre-registered
in evaluation_protocol.md §16. Given a reference corpus already embedded
in the consensus space (e.g., the K=8 Korean forest planning corpus) and
out-of-sample (OOS) coordinates produced by REAP's projection head, the
filter decides which OOS points are typical of their assigned cluster.

Algorithm (default ``method="mahalanobis_pooled"``)
---------------------------------------------------

For each cluster ``c`` with reference points ``X_c`` and OOS points ``Y_c``:

1. **Reference covariance.** ``Σ_c = cov(X_c) + ridge·I`` — anchored in
   the reference geometry (anisotropic shape).
2. **Per-cluster OOS centroid (location correction).**
   ``μ̂_c = mean(Y_c)`` — pooled across any subgroups within Y. This
   handles the systematic register/style shift between reference and
   OOS without absorbing topic mismatch (which a per-cluster reference
   centroid would force the filter to ignore).
3. **Empirical leave-one-out threshold.** For each ``x_i ∈ X_c`` compute
   ``M_c^LOO(x_i) = (x_i - μ_{c,-i})^T Σ_{c,-i}^{-1} (x_i - μ_{c,-i})``
   where ``μ_{c,-i}`` and ``Σ_{c,-i}`` are estimated *without* ``x_i``.
   Set ``τ_c`` = ``(1-α)``-quantile of ``{M_c^LOO(x_i)}``.
4. **Apply.** For an OOS point ``y_j ∈ Y_c``, accept iff
   ``(y_j - μ̂_c)^T Σ_c^{-1} (y_j - μ̂_c) ≤ τ_c``.

The ``mahalanobis_no_correction`` variant replaces ``μ̂_c`` with the
reference centroid ``μ_c``; this is the strict-conformal Mahalanobis
filter (no location correction) and yields lower retention when reference
and OOS distributions are register-shifted.

The per-subgroup variant (`calibrate_per_subgroup`) computes a separate
``μ̂_{c,p}`` for each subgroup ``p`` (e.g., president), with a hard
fallback to the pooled centroid when ``|Y_{c,p}| < fallback_n_threshold``.

Public surface
--------------
- `FilterCalibration` — frozen Pydantic model holding per-cluster
  centroids, inverse covariances, thresholds, and bookkeeping.
- `calibrate` — produce a `FilterCalibration` for the pooled or
  no-correction variant.
- `calibrate_per_subgroup` — produce a dict of `FilterCalibration` keyed
  by subgroup label.
- `apply` — apply a `FilterCalibration` to test coordinates, returning a
  boolean keep mask.

This module has no side effects.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Method identifiers. Anything else raises ValueError in calibrate.
METHOD_MAHALANOBIS_POOLED = "mahalanobis_pooled"
METHOD_MAHALANOBIS_NO_CORRECTION = "mahalanobis_no_correction"
_VALID_METHODS = {METHOD_MAHALANOBIS_POOLED, METHOD_MAHALANOBIS_NO_CORRECTION}

DEFAULT_ALPHA = 0.01
DEFAULT_RIDGE = 1e-6
DEFAULT_FALLBACK_N_THRESHOLD = 10


class FilterCalibration(BaseModel):
    """Calibrated state for the OOS conformal filter.

    Per-cluster centroids, inverse covariances, and thresholds are
    keyed by integer cluster label.

    All numpy arrays are immutable to consumers — treat them as read-only
    even though numpy itself does not enforce this. Pydantic's ``frozen``
    setting prevents reassignment of fields but does not prevent in-place
    mutation of the array contents.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    per_cluster_centroids: dict[int, np.ndarray] = Field(
        ...,
        description=(
            "Per-cluster centroid used to compute Mahalanobis distance for "
            "OOS points. For ``mahalanobis_pooled`` this is the per-cluster "
            "OOS mean; for ``mahalanobis_no_correction`` it is the reference "
            "cluster mean."
        ),
    )
    per_cluster_inv_covariances: dict[int, np.ndarray] = Field(
        ...,
        description=(
            "Per-cluster inverse covariance ``Σ_c^{-1}`` estimated on the "
            "reference points only (with ridge regularization)."
        ),
    )
    per_cluster_thresholds: dict[int, float] = Field(
        ...,
        description=(
            "Per-cluster threshold ``τ_c`` = ``(1-α)``-quantile of the "
            "reference leave-one-out Mahalanobis distribution."
        ),
    )
    per_cluster_n_reference: dict[int, int] = Field(
        ...,
        description="Reference points per cluster used during calibration.",
    )
    per_cluster_n_oos: dict[int, int] = Field(
        ...,
        description=(
            "OOS points per cluster used to compute ``μ̂_c`` "
            "(``mahalanobis_pooled``) or zero (``mahalanobis_no_correction``)."
        ),
    )
    alpha: float = Field(..., gt=0.0, lt=1.0)
    method: str = Field(...)
    ridge: float = Field(..., gt=0.0)
    embedding_dim: int = Field(..., ge=2)
    n_reference: int = Field(..., ge=1)
    n_oos: int = Field(..., ge=0)


def calibrate(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    oos_coords: np.ndarray,
    oos_labels: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    method: str = METHOD_MAHALANOBIS_POOLED,
    ridge: float = DEFAULT_RIDGE,
) -> FilterCalibration:
    """Calibrate the conformal filter on reference + OOS coordinates.

    Parameters
    ----------
    reference_coords
        ``(N_ref, d)`` float array of reference points in the consensus
        space.
    reference_labels
        ``(N_ref,)`` int array of cluster assignments for each reference
        point.
    oos_coords
        ``(N_oos, d)`` float array of out-of-sample points (used for the
        per-cluster OOS centroid in ``mahalanobis_pooled``; ignored except
        for shape checks in ``mahalanobis_no_correction``).
    oos_labels
        ``(N_oos,)`` int array of cluster assignments for OOS points.
    alpha
        Significance level. Default 0.01 (99th percentile of reference
        LOO Mahalanobis).
    method
        ``"mahalanobis_pooled"`` (default) or ``"mahalanobis_no_correction"``.
    ridge
        Diagonal ridge added to per-cluster covariances before inversion.
        Default ``1e-6``.

    Returns
    -------
    `FilterCalibration` carrying per-cluster thresholds, centroids, and
    inverse covariances.

    Raises
    ------
    ValueError
        Inputs have inconsistent shapes, ``method`` is unknown, or no
        cluster has enough reference points to compute a covariance.

    Warnings
    --------
    Issues a ``RuntimeWarning`` for each cluster where ``n_c < ⌈1/α⌉``
    or ``n_c < d + 2``. Such clusters get a finite threshold (the maximum
    LOO score, with ridge-regularized covariance), but the empirical
    quantile and conformal coverage are not well-defined.

    Side effects: emits warnings on stderr; otherwise none.
    """
    _check_method(method)
    ref_coords, ref_labels, oos_arr, oos_lbl = _coerce_inputs(
        reference_coords, reference_labels, oos_coords, oos_labels
    )
    if ref_coords.shape[1] != oos_arr.shape[1]:
        raise ValueError(
            f"reference dim {ref_coords.shape[1]} != OOS dim {oos_arr.shape[1]}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if ridge <= 0.0:
        raise ValueError(f"ridge must be positive; got {ridge}")

    d = ref_coords.shape[1]
    cluster_ids = sorted({int(c) for c in np.unique(ref_labels)})
    if not cluster_ids:
        raise ValueError("reference_labels has no clusters")

    centroids: dict[int, np.ndarray] = {}
    inv_covs: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    n_ref_per: dict[int, int] = {}
    n_oos_per: dict[int, int] = {}

    n_min_for_quantile = int(math.ceil(1.0 / alpha))

    for c in cluster_ids:
        ref_in_c = ref_coords[ref_labels == c]
        n_c = int(ref_in_c.shape[0])
        n_ref_per[c] = n_c
        n_oos_per[c] = int((oos_lbl == c).sum())
        if n_c < 2:
            warnings.warn(
                f"cluster {c}: n_c={n_c} < 2 — covariance undefined; "
                "filter will reject every test point in this cluster.",
                RuntimeWarning,
                stacklevel=2,
            )
            centroids[c] = ref_in_c.mean(axis=0) if n_c >= 1 else np.zeros(d)
            inv_covs[c] = np.full((d, d), np.nan)
            thresholds[c] = -np.inf  # always reject
            continue
        if n_c < n_min_for_quantile or n_c < d + 2:
            warnings.warn(
                f"cluster {c}: small cluster (n_c={n_c}, d={d}, alpha={alpha}). "
                f"Empirical {1 - alpha:.4f}-quantile of LOO Mahalanobis is "
                f"defined but is dominated by the maximum value; conformal "
                f"coverage is approximate.",
                RuntimeWarning,
                stacklevel=2,
            )
        mu_ref = ref_in_c.mean(axis=0)
        sigma = np.cov(ref_in_c.T, ddof=1) + ridge * np.eye(d)
        try:
            inv_sigma = np.linalg.inv(sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"cluster {c}: covariance not invertible even after ridge "
                f"regularization (n_c={n_c}, ridge={ridge}): {exc}"
            ) from exc
        # Empirical LOO Mahalanobis distribution
        loo_d2 = _compute_loo_mahalanobis(ref_in_c, ridge)
        valid = np.isfinite(loo_d2)
        if not valid.any():
            raise ValueError(
                f"cluster {c}: every LOO Mahalanobis score is non-finite; "
                "calibration impossible."
            )
        thresholds[c] = float(np.quantile(loo_d2[valid], 1.0 - alpha))
        inv_covs[c] = inv_sigma
        # Centroid choice depends on method
        if method == METHOD_MAHALANOBIS_POOLED:
            oos_in_c = oos_arr[oos_lbl == c]
            if oos_in_c.shape[0] >= 1:
                centroids[c] = oos_in_c.mean(axis=0)
            else:
                # No OOS points — fall back to reference centroid
                centroids[c] = mu_ref
        else:  # METHOD_MAHALANOBIS_NO_CORRECTION
            centroids[c] = mu_ref

    return FilterCalibration(
        per_cluster_centroids=centroids,
        per_cluster_inv_covariances=inv_covs,
        per_cluster_thresholds=thresholds,
        per_cluster_n_reference=n_ref_per,
        per_cluster_n_oos=n_oos_per,
        alpha=alpha,
        method=method,
        ridge=ridge,
        embedding_dim=d,
        n_reference=int(ref_coords.shape[0]),
        n_oos=int(oos_arr.shape[0]),
    )


def calibrate_per_subgroup(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    oos_coords: np.ndarray,
    oos_labels: np.ndarray,
    oos_subgroups: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    fallback_n_threshold: int = DEFAULT_FALLBACK_N_THRESHOLD,
    ridge: float = DEFAULT_RIDGE,
) -> dict[Any, FilterCalibration]:
    """Calibrate one filter per subgroup, with hard fallback to pooled centroid.

    For every (cluster c, subgroup p) cell:
    - if ``|Y_{c,p}| >= fallback_n_threshold``, use ``mean(Y_{c,p})``
      as the centroid for that cell;
    - otherwise, use the pooled per-cluster centroid ``mean(Y_c)``.

    Returns a dict ``{subgroup_id: FilterCalibration}``. Each calibration's
    ``per_cluster_centroids`` reflects the subgroup-or-fallback choice
    cell-by-cell.

    Side effects: may emit calibration warnings via `calibrate`; otherwise none.
    """
    ref_coords, ref_labels, oos_arr, oos_lbl = _coerce_inputs(
        reference_coords, reference_labels, oos_coords, oos_labels
    )
    if oos_subgroups.shape != (oos_arr.shape[0],):
        raise ValueError(
            f"oos_subgroups shape {oos_subgroups.shape} != ({oos_arr.shape[0]},)"
        )
    if fallback_n_threshold < 1:
        raise ValueError(
            f"fallback_n_threshold must be >= 1; got {fallback_n_threshold}"
        )

    # First, compute the pooled calibration to inherit thresholds and
    # inverse covariances; we only override the centroids per subgroup.
    pooled = calibrate(
        ref_coords, ref_labels, oos_arr, oos_lbl,
        alpha=alpha, method=METHOD_MAHALANOBIS_POOLED, ridge=ridge,
    )
    cluster_ids = sorted(pooled.per_cluster_centroids.keys())
    subgroups = sorted({sg for sg in oos_subgroups})  # type: ignore[type-var]

    out: dict[Any, FilterCalibration] = {}
    for sg in subgroups:
        sg_mask = oos_subgroups == sg
        centroids: dict[int, np.ndarray] = {}
        n_oos_per: dict[int, int] = {}
        for c in cluster_ids:
            cell_mask = sg_mask & (oos_lbl == c)
            n_cell = int(cell_mask.sum())
            n_oos_per[c] = n_cell
            if n_cell >= fallback_n_threshold:
                centroids[c] = oos_arr[cell_mask].mean(axis=0)
            else:
                centroids[c] = pooled.per_cluster_centroids[c]
        out[sg] = FilterCalibration(
            per_cluster_centroids=centroids,
            per_cluster_inv_covariances=pooled.per_cluster_inv_covariances,
            per_cluster_thresholds=pooled.per_cluster_thresholds,
            per_cluster_n_reference=pooled.per_cluster_n_reference,
            per_cluster_n_oos=n_oos_per,
            alpha=pooled.alpha,
            method=pooled.method,
            ridge=pooled.ridge,
            embedding_dim=pooled.embedding_dim,
            n_reference=pooled.n_reference,
            n_oos=int(sg_mask.sum()),
        )
    return out


def apply(
    test_coords: np.ndarray,
    test_labels: np.ndarray,
    calibration: FilterCalibration,
) -> np.ndarray:
    """Apply a calibrated filter to test coordinates.

    Returns a ``(N_test,)`` boolean array — ``True`` for points that pass
    the filter (Mahalanobis ≤ threshold), ``False`` for those that fail.
    Points whose cluster does not appear in the calibration are rejected
    (returned as ``False``).

    Side effects: none.
    """
    test_coords = np.ascontiguousarray(test_coords, dtype=np.float64)
    test_labels = np.ascontiguousarray(test_labels)
    if test_coords.ndim != 2:
        raise ValueError(f"test_coords must be 2-d; got shape {test_coords.shape}")
    if test_coords.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f"length mismatch: test_coords {test_coords.shape[0]} vs "
            f"test_labels {test_labels.shape[0]}"
        )
    if test_coords.shape[1] != calibration.embedding_dim:
        raise ValueError(
            f"test_coords dim {test_coords.shape[1]} != calibration "
            f"embedding_dim {calibration.embedding_dim}"
        )
    keep = np.zeros(test_coords.shape[0], dtype=bool)
    for i in range(test_coords.shape[0]):
        c = int(test_labels[i])
        if c not in calibration.per_cluster_centroids:
            continue  # unknown cluster -> reject
        threshold = calibration.per_cluster_thresholds[c]
        if not math.isfinite(threshold):
            continue  # cluster too small to calibrate -> reject
        diff = test_coords[i] - calibration.per_cluster_centroids[c]
        d2 = float(diff @ calibration.per_cluster_inv_covariances[c] @ diff)
        keep[i] = d2 <= threshold
    return keep


def compute_mahalanobis_d2(
    test_coords: np.ndarray,
    test_labels: np.ndarray,
    calibration: FilterCalibration,
) -> np.ndarray:
    """Diagnostic: per-point Mahalanobis D² against the calibrated centroids.

    Returns ``(N_test,)`` float array of D² values. Points whose cluster
    is not in the calibration get ``np.nan``. Useful for plotting the D²
    distribution and for users who want to make their own keep/reject
    decision rather than relying on the per-cluster threshold.

    Side effects: none.
    """
    test_coords = np.ascontiguousarray(test_coords, dtype=np.float64)
    test_labels = np.ascontiguousarray(test_labels)
    if test_coords.shape[1] != calibration.embedding_dim:
        raise ValueError(
            f"test_coords dim {test_coords.shape[1]} != calibration "
            f"embedding_dim {calibration.embedding_dim}"
        )
    out = np.full(test_coords.shape[0], np.nan, dtype=np.float64)
    for i in range(test_coords.shape[0]):
        c = int(test_labels[i])
        if c not in calibration.per_cluster_centroids:
            continue
        diff = test_coords[i] - calibration.per_cluster_centroids[c]
        out[i] = float(diff @ calibration.per_cluster_inv_covariances[c] @ diff)
    return out


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _coerce_inputs(
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    oos_coords: np.ndarray,
    oos_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes/dtypes and convert to canonical layout.

    Side effects: none.
    """
    ref_c = np.ascontiguousarray(reference_coords, dtype=np.float64)
    ref_l = np.asarray(reference_labels)
    oos_c = np.ascontiguousarray(oos_coords, dtype=np.float64)
    oos_l = np.asarray(oos_labels)
    if ref_c.ndim != 2:
        raise ValueError(f"reference_coords must be 2-d; got shape {ref_c.shape}")
    if oos_c.ndim != 2:
        raise ValueError(f"oos_coords must be 2-d; got shape {oos_c.shape}")
    if ref_l.shape != (ref_c.shape[0],):
        raise ValueError(
            f"reference_labels shape {ref_l.shape} != ({ref_c.shape[0]},)"
        )
    if oos_l.shape != (oos_c.shape[0],):
        raise ValueError(
            f"oos_labels shape {oos_l.shape} != ({oos_c.shape[0]},)"
        )
    if not np.issubdtype(ref_l.dtype, np.integer):
        raise ValueError(
            f"reference_labels must be integer-typed; got dtype {ref_l.dtype}"
        )
    if not np.issubdtype(oos_l.dtype, np.integer):
        raise ValueError(
            f"oos_labels must be integer-typed; got dtype {oos_l.dtype}"
        )
    return ref_c, ref_l, oos_c, oos_l


def _compute_loo_mahalanobis(ref_in_c: np.ndarray, ridge: float) -> np.ndarray:
    """Per-point leave-one-out squared Mahalanobis distance.

    For each point ``i``, compute
    ``(x_i - μ_{-i})^T Σ_{-i}^{-1} (x_i - μ_{-i})`` where the centroid
    and covariance are estimated on the other ``n-1`` points with the
    same ridge regularization used at calibration.

    Returns a ``(n,)`` array; entries where ``Σ_{-i}`` is not invertible
    are returned as ``np.nan`` (callers filter these out before computing
    quantiles).

    Side effects: none.
    """
    n, d = ref_in_c.shape
    out = np.empty(n, dtype=np.float64)
    eye = np.eye(d)
    # Sufficient stats for the full sample, used to derive leave-one-out
    # mean and centered-sums-of-squares in O(d^2) per point rather than
    # re-running ``np.cov`` for every i.
    total = ref_in_c.sum(axis=0)
    centered = ref_in_c - ref_in_c.mean(axis=0)
    full_S = centered.T @ centered  # (n-1) * full sample covariance, ddof=1 form
    # full_S = sum_i (x_i - mean) (x_i - mean)^T
    # leave-one-out mean: mu_{-i} = (total - x_i) / (n - 1)
    # leave-one-out centered sum-of-squares:
    #   S_{-i} = full_S - (n / (n-1)) * (x_i - mean)(x_i - mean)^T
    # See Welford-style update (this identity is standard).
    n_loo = n - 1
    if n_loo < 1:
        return np.full(n, np.nan)
    mean_full = ref_in_c.mean(axis=0)
    for i in range(n):
        mu_loo = (total - ref_in_c[i]) / n_loo
        delta = ref_in_c[i] - mean_full
        S_loo = full_S - (n / n_loo) * np.outer(delta, delta)
        # Sample covariance of LOO subset:
        #   Σ_{-i} = S_{-i} / (n_loo - 1) when n_loo - 1 > 0
        #   Add ridge regularization (matches calibration) — even when
        #   n_loo = 1, ridge ensures invertibility.
        ddof = max(n_loo - 1, 1)
        cov_loo = S_loo / ddof + ridge * eye
        try:
            inv_loo = np.linalg.inv(cov_loo)
        except np.linalg.LinAlgError:
            out[i] = np.nan
            continue
        diff = ref_in_c[i] - mu_loo
        out[i] = float(diff @ inv_loo @ diff)
    return out


def _check_method(method: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(_VALID_METHODS)}"
        )


__all__ = [
    "FilterCalibration",
    "calibrate",
    "calibrate_per_subgroup",
    "apply",
    "compute_mahalanobis_d2",
    "METHOD_MAHALANOBIS_POOLED",
    "METHOD_MAHALANOBIS_NO_CORRECTION",
    "DEFAULT_ALPHA",
    "DEFAULT_RIDGE",
    "DEFAULT_FALLBACK_N_THRESHOLD",
]
