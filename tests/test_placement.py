"""Tests for reap.placement: head variants, held-out placements, calibrated inside test."""

from __future__ import annotations

import numpy as np
import pytest

from reap.placement import (
    REAP_HEAD_SETTING_KEYS,
    HeadFit,
    fit_knn_head,
    fit_reap_head,
    fit_ridge_head,
    fit_scaled_mlp_head,
    get_conformal_cutoff,
    get_heldout_predictions,
    get_nearest_reference_distances,
)


def _linear_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inputs in 20-d and 3-d targets that are a fixed linear map of them plus small noise."""
    rng = np.random.default_rng(seed)
    W = np.random.default_rng(0).normal(size=(20, 3))
    X = rng.normal(size=(n, 20))
    return X, X @ W + rng.normal(scale=0.05, size=(n, 3))


def test_ridge_head_recovers_a_linear_map() -> None:
    """Held-out ridge placements land near the truth on linear data; alpha is from the grid."""
    X, Y = _linear_data(300, 1)
    Y_hat, folds, fits = get_heldout_predictions(
        X, lambda tr: fit_ridge_head(X[tr], Y[tr], alpha_grid=[1e-3, 1e-1, 10.0]),
        n_folds=5, random_state=3)
    assert set(folds) == set(range(5))
    assert np.median(np.linalg.norm(Y_hat - Y, axis=1)) < 0.2
    assert all(f.chosen["alpha"] in (1e-3, 1e-1, 10.0) for f in fits)
    with pytest.raises(ValueError, match="empty"):
        fit_ridge_head(X, Y, alpha_grid=[])


def test_knn_head_picks_k_from_the_grid() -> None:
    """kNN chooses a k from the grid and refuses a k larger than an inner training fold."""
    X, Y = _linear_data(120, 2)
    head = fit_knn_head(X, Y, k_grid=[1, 3, 5, 10], weights="distance", metric="cosine",
                        n_inner_folds=5, random_state=4)
    assert head.kind == "knn" and head.chosen["k"] in (1, 3, 5, 10)
    assert head.predict(X[:7]).shape == (7, 3)
    with pytest.raises(ValueError, match="every k"):
        fit_knn_head(X, Y, k_grid=[200], weights="uniform", metric="euclidean",
                     n_inner_folds=5, random_state=4)


def test_scaled_mlp_head_predicts_in_target_units() -> None:
    """The scaled MLP returns coordinates in the targets' own units and shape."""
    X, Y = _linear_data(200, 5)
    head = fit_scaled_mlp_head(X, Y, hidden_layer_sizes=[32], activation="relu", alpha=1e-4,
                               learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                               validation_fraction=0.1, random_state=6)
    pred = head.predict(X)
    assert pred.shape == Y.shape
    assert np.median(np.linalg.norm(pred - Y, axis=1)) < np.median(np.linalg.norm(Y, axis=1))


def test_reap_head_requires_every_setting() -> None:
    """The REAP head trains with declared settings and refuses a missing one."""
    pytest.importorskip("torch")
    X, Y = _linear_data(80, 7)
    labels = (Y[:, 0] > np.median(Y[:, 0])).astype(int)
    settings = {"hidden_layers": [16], "dropout": 0.0, "alpha": 0.7, "n_folds": 2,
                "batch_size": 32, "max_epochs": 3, "patience": 2, "lr": 1e-3,
                "weight_decay": 1e-4, "device": "cpu"}
    assert set(settings) == REAP_HEAD_SETTING_KEYS
    head = fit_reap_head(X.astype(np.float32), Y.astype(np.float32), labels, settings=settings,
                         seed=8)
    assert head.predict(X[:5].astype(np.float32)).shape == (5, 3)
    with pytest.raises(ValueError, match="exactly"):
        fit_reap_head(X, Y, labels, settings={k: settings[k] for k in list(settings)[1:]}, seed=8)


def test_heldout_rows_never_train_their_own_head() -> None:
    """Each row is placed by a head whose training rows exclude it."""
    X = np.arange(50, dtype=np.float64).reshape(-1, 1)

    def fit(train: np.ndarray) -> HeadFit:
        seen = set(train.tolist())
        return HeadFit(kind="probe", chosen={},
                       predict=lambda Z: np.array([[float(int(z) in seen)] for z in Z[:, 0]]))

    Y_hat, folds, _ = get_heldout_predictions(X, fit, n_folds=5, random_state=1)
    assert np.all(Y_hat == 0.0)
    assert np.all(folds >= 0)


def test_nearest_distance_skips_own_position() -> None:
    """A held-out passage's own true position is skipped; other positions count."""
    Y_ref = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    Y_query = np.array([[0.1, 0.0], [3.0, 0.0]])
    plain = get_nearest_reference_distances(Y_ref, Y_query, own_rows=None)
    assert plain == pytest.approx([0.1, 0.0])
    own = get_nearest_reference_distances(Y_ref, Y_query, own_rows=np.array([0, 1]))
    assert own == pytest.approx([2.9, 3.0])


def test_conformal_cutoff_rank() -> None:
    """The cutoff is the ceil((n+1)(1-alpha))-th smallest score; too few scores raise."""
    scores = np.arange(1, 100, dtype=float)[::-1]
    assert get_conformal_cutoff(scores, alpha=0.1) == 90.0
    with pytest.raises(ValueError, match="too few"):
        get_conformal_cutoff(np.ones(5), alpha=0.01)


def test_calibrated_cutoff_accepts_fresh_reference_like_passages() -> None:
    """New passages from the reference's own distribution pass at about 1 - alpha."""
    X, Y = _linear_data(400, 11)
    fit = lambda tr: fit_ridge_head(X[tr], Y[tr], alpha_grid=[1e-2])  # noqa: E731
    Y_hat, _, _ = get_heldout_predictions(X, fit, n_folds=5, random_state=2)
    cutoff = get_conformal_cutoff(
        get_nearest_reference_distances(Y, Y_hat, own_rows=np.arange(len(Y))), alpha=0.1)
    X_new, _ = _linear_data(400, 12)
    nn_new = get_nearest_reference_distances(Y, fit(np.arange(len(X))).predict(X_new),
                                             own_rows=None)
    assert 0.8 <= float(np.mean(nn_new <= cutoff)) <= 1.0
