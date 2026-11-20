"""Placing new passages on a REAP map: head variants, held-out checks, calibrated inside test.

A projection head maps input embeddings (for example e5 vectors) to the
coordinates of a REAP map, so passages that were not part of the map can be
placed on it. This module offers four heads, each fitted with every setting
declared (no defaults):

- ``fit_reap_head``: REAP's PyTorch MLP (``reap.projection.train_projection_head``).
- ``fit_scaled_mlp_head``: scikit-learn ``MLPRegressor`` on standardised
  inputs and targets, as in the AI-art study.
- ``fit_ridge_head``: one linear map with ridge penalty, chosen from a
  declared grid by leave-one-out error.
- ``fit_knn_head``: no training; a passage's coordinates are the weighted mean
  of the coordinates of its k nearest reference passages in embedding space,
  with k chosen from a declared grid by inner cross-validation.

To judge a head, ``get_heldout_predictions`` places every reference passage
with a head trained without it (k-fold). The held-out placements also
calibrate the inside test. A passage counts as inside the reference when its
nearest-reference distance is within ``get_conformal_cutoff`` of the held-out
reference passages' nearest-reference distances
(``get_nearest_reference_distances``, each held-out passage's own position
excluded). A new passage that resembles the reference then passes with
probability of about ``1 - alpha``. Fold heads see fewer passages than the
final head, so the test leans conservative; it is a cross-validation
calibration, not a finite-sample guarantee.

Exports: ``HeadFit``, ``REAP_HEAD_SETTING_KEYS``, ``fit_reap_head``,
``fit_scaled_mlp_head``, ``fit_ridge_head``, ``fit_knn_head``,
``get_heldout_predictions``, ``get_nearest_reference_distances``,
``get_conformal_cutoff``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Predictor = Callable[[np.ndarray], np.ndarray]

REAP_HEAD_SETTING_KEYS = frozenset({
    "hidden_layers", "dropout", "alpha", "n_folds", "batch_size", "max_epochs", "patience",
    "lr", "weight_decay", "device",
})


class HeadFit(BaseModel):
    """A fitted head: its kind, what it chose, and its predict function."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: str = Field(..., description="Head kind: reap, scaled_mlp, ridge or knn.")
    chosen: dict[str, Any] = Field(..., description="Values the head chose (k, alpha), if any.")
    predict: Predictor = Field(..., description="Maps (n, d) embeddings to (n, m) coordinates.")


def fit_reap_head(
    X: np.ndarray, Y: np.ndarray, labels: np.ndarray, *, settings: dict[str, Any], seed: int,
) -> HeadFit:
    """REAP's PyTorch MLP head, trained on all rows.

    ``settings`` must hold exactly ``REAP_HEAD_SETTING_KEYS``. The training
    loss does not use ``labels``; they set the head's internal cross-validation
    folds, which consume random state, so the fitted head can still change
    with the labels. Side effects: seeds torch with ``seed``, trains a network.
    """
    from reap.projection import train_projection_head

    if set(settings) != REAP_HEAD_SETTING_KEYS:
        raise ValueError(f"settings must have exactly {sorted(REAP_HEAD_SETTING_KEYS)}")
    head = train_projection_head(X=X, Y=Y, labels=labels, seed=seed, **settings)
    model = head["model"]
    return HeadFit(kind="reap", chosen={},
                   predict=lambda Z: np.asarray(model.forward(Z), dtype=np.float64))


def fit_scaled_mlp_head(
    X: np.ndarray, Y: np.ndarray, *, hidden_layer_sizes: Sequence[int], activation: str,
    alpha: float, learning_rate_init: float, max_iter: int, early_stopping: bool,
    validation_fraction: float, random_state: int,
) -> HeadFit:
    """scikit-learn MLPRegressor on standardised inputs and targets (mean-squared error only).

    Side effects: none beyond fitting (warnings from scikit-learn pass through).
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
    mlp = MLPRegressor(hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation,  # pyright: ignore[reportArgumentType]
                       alpha=alpha, learning_rate_init=learning_rate_init, max_iter=max_iter,
                       early_stopping=early_stopping, validation_fraction=validation_fraction,
                       random_state=random_state)
    mlp.fit(sx.transform(X), sy.transform(Y))
    return HeadFit(kind="scaled_mlp", chosen={"n_iter": int(mlp.n_iter_)},
                   predict=lambda Z: np.asarray(
                       sy.inverse_transform(np.asarray(mlp.predict(sx.transform(Z))).reshape(
                           len(Z), -1)), dtype=np.float64))


def fit_ridge_head(X: np.ndarray, Y: np.ndarray, *, alpha_grid: Sequence[float]) -> HeadFit:
    """A linear map with intercept; the ridge penalty is chosen from ``alpha_grid``.

    The penalty minimises leave-one-out error (scikit-learn ``RidgeCV``).
    Side effects: none.
    """
    from sklearn.linear_model import RidgeCV

    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must not be empty")
    model = RidgeCV(alphas=np.asarray(alpha_grid, dtype=np.float64)).fit(X, Y)
    return HeadFit(kind="ridge", chosen={"alpha": float(model.alpha_)},  # pyright: ignore[reportArgumentType]
                   predict=lambda Z: np.asarray(model.predict(Z), dtype=np.float64))


def fit_knn_head(
    X: np.ndarray, Y: np.ndarray, *, k_grid: Sequence[int], weights: Literal["uniform", "distance"],
    metric: str, n_inner_folds: int, random_state: int,
) -> HeadFit:
    """Coordinates as the weighted mean of the k nearest reference passages' coordinates.

    k minimises mean squared error over ``n_inner_folds`` inner folds (ties go
    to the smaller k); every k must fit in an inner training fold. Side effects: none.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    smallest_train = len(X) - math.ceil(len(X) / n_inner_folds)
    if not k_grid or max(k_grid) > smallest_train or min(k_grid) < 1:
        raise ValueError(f"every k must be in 1..{smallest_train}, got {list(k_grid)}")
    folds = KFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

    def _model(k: int) -> Any:
        return KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric, algorithm="brute")

    mse = {k: float(-cross_val_score(_model(k), X, Y, cv=folds,
                                     scoring="neg_mean_squared_error").mean()) for k in k_grid}
    best = min(mse, key=lambda k: (mse[k], k))
    model = _model(best).fit(X, Y)
    return HeadFit(kind="knn", chosen={"k": best, "inner_mse": mse},
                   predict=lambda Z: np.asarray(model.predict(Z), dtype=np.float64))


def get_heldout_predictions(
    X: np.ndarray, fit: Callable[[np.ndarray], HeadFit], *, n_folds: int, random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[HeadFit]]:
    """Place every row with a head trained without it (shuffled k-fold, not label-stratified).

    ``fit`` receives the training row indices and returns a ``HeadFit``.
    Returns the held-out placements (n, m), each row's fold number, and the
    fold heads. Side effects: whatever ``fit`` does.
    """
    from sklearn.model_selection import KFold

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_of_row = np.full(len(X), -1, dtype=np.int64)
    parts: dict[int, np.ndarray] = {}
    fits: list[HeadFit] = []
    for f, (train, test) in enumerate(folds.split(X)):
        head = fit(train)
        parts[f] = head.predict(X[test])
        fold_of_row[test] = f
        fits.append(head)
    first = parts[0]
    Y_hat = np.empty((len(X), first.shape[1]), dtype=np.float64)
    for f, (_, test) in enumerate(folds.split(X)):
        Y_hat[test] = parts[f]
    return Y_hat, fold_of_row, fits


def get_nearest_reference_distances(
    Y_ref: np.ndarray, Y_query: np.ndarray, *, own_rows: np.ndarray | None,
) -> np.ndarray:
    """Distance from each query point to its nearest reference position.

    With ``own_rows``, query ``i`` is reference row ``own_rows[i]`` placed by
    a head, and its own true position is skipped. Side effects: none.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1 if own_rows is None else 2).fit(Y_ref)
    dist, idx = nn.kneighbors(Y_query)
    if own_rows is None:
        return dist[:, 0]
    if len(own_rows) != len(Y_query):
        raise ValueError("own_rows must have one entry per query point")
    first_is_own = idx[:, 0] == own_rows
    return np.where(first_is_own, dist[:, 1], dist[:, 0])


def get_conformal_cutoff(scores: np.ndarray, *, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score (split-conformal rank).

    Raises ``ValueError`` when there are too few scores for ``alpha``.
    Side effects: none.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"{n} scores are too few for alpha={alpha}; need at least {rank}")
    return float(np.sort(np.asarray(scores, dtype=np.float64))[rank - 1])
