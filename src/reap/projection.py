"""Neural projection head for mapping new data into an established REAP consensus space.

The projection head learns a mapping from high-dimensional sentence embeddings
(e.g., 384-d or 1024-d) to the consensus UMAP coordinate space, eliminating
the need to re-run the expensive multi-seed UMAP procedure for new data.

Architecture: MLP with BatchNorm + GELU + Dropout.
Loss: alpha * MSE + (1 - alpha) * (1 - distance_correlation).

Requires the `torch` optional dependency: pip install reap-topics[projection]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch import nn

logger = logging.getLogger(__name__)

# The reporting-side metric clustering (KMeans over the projected points) uses its
# own FIXED seed, kept separate from the model seed, so a reported metric such as
# the ARI of those labels is deterministic regardless of which model seed produced
# the points being clustered.
_METRIC_KMEANS_RANDOM_STATE = 42


def _check_torch() -> None:
    """Raise ImportError with install instructions if torch is missing.

    Side effects: imports torch module.
    """
    try:
        __import__("torch")
    except ImportError:
        raise ImportError(
            "PyTorch is required for the projection head. "
            "Install with: pip install reap-embeddings[projection]"
        ) from None


class _BaseProjectionHead:
    """Shared machinery for projection heads.

    Handles device placement, the forward pass, train/eval modes, parameter
    access, and save/load. A subclass builds ``self._model`` (a torch module) and
    sets ``self.input_dim`` / ``self.output_dim`` in its ``__init__``. This base
    holds no architecture of its own, so the MLP head and the linear head share
    one interface and one set of behaviours.
    """

    # Set by each subclass's __init__.
    _model: nn.Module
    _device: torch.device
    input_dim: int
    output_dim: int

    def to(self, device: str) -> _BaseProjectionHead:
        """Move the model to a torch device. Returns self for chaining."""
        import torch

        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def parameters(self):  # type: ignore[no-untyped-def]
        """Return model parameters for an optimizer."""
        return self._model.parameters()

    def train_mode(self) -> None:
        """Set training mode (enables dropout + batchnorm tracking, if present)."""
        self._model.train()

    def eval_mode(self) -> None:
        """Set evaluation mode (disables dropout, uses running batchnorm stats)."""
        self._model.eval()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Project embeddings to consensus space.

        Parameters
        ----------
        X : (n_samples, input_dim) embeddings.

        Returns
        -------
        (n_samples, output_dim) consensus coordinates.
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X.astype(np.float32)).to(self._device)
            out = self._model(tensor)
            return out.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights to disk. Side effect: writes a file at ``path``."""
        import torch

        torch.save(self._model.state_dict(), path)
        logger.info("Projection head saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk. Side effect: mutates this model in place."""
        import torch

        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        logger.info("Projection head loaded from %s", path)


class ProjectionHead(_BaseProjectionHead):
    """MLP that maps high-d embeddings to consensus UMAP coordinates.

    Default architecture for 384-d → 18-d:
        Linear(384, 128) → BatchNorm → GELU → Dropout(0.3)
        Linear(128, 64)  → BatchNorm → GELU → Dropout(0.3)
        Linear(64, 18)   [no activation]

    ~59K parameters. Trained with combined MSE + distance correlation loss.
    """

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 18,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        _check_torch()
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden = hidden_layers or [128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self._model = nn.Sequential(*layers)
        self._device = torch.device("cpu")


class LinearProjectionHead(_BaseProjectionHead):
    """Single affine map from input embeddings to consensus coordinates.

    This is the pre-registered §11 linear baseline head: one
    ``Linear(input_dim, output_dim)`` with no hidden layers, batchnorm, or
    dropout. It shares the ``ProjectionHead`` interface, so
    ``train_projection_head`` and the OOS comparison harness can build, train,
    and evaluate it exactly as they do the MLP head.
    """

    def __init__(self, input_dim: int = 384, output_dim: int = 18) -> None:
        _check_torch()
        import torch
        import torch.nn as nn

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._model = nn.Linear(input_dim, output_dim)
        self._device = torch.device("cpu")


def compute_projection_loss(
    Y_pred: object,
    Y_true: object,
    alpha: float = 0.7,
) -> tuple[object, dict[str, float]]:
    """Combined MSE + distance-correlation loss for projection-head training.

    Loss = alpha * MSE(Y_pred, Y_true) + (1 - alpha) * (1 - dist_corr(Y_pred, Y_true))

    The distance-correlation term is the registered loss recipe
    ``dist_corr_loss`` (kind: ``adapted``): a Pearson correlation over the
    *full* flattened self-distance matrices of the two batches — diagonal
    zeros included, every pair counted twice — computed with ``torch.cdist``.
    Despite the everyday name it is **not** Székely's distance correlation,
    and it is a different calculation from the reported metric recipe
    ``distance_correlation`` (``reap.evaluation
    .compute_distance_correlation``), which uses condensed unique-pair
    vectors: the diagonal zeros shift the correlation, so the two recipes
    disagree on generic inputs and must never be quoted interchangeably (the
    exact gap is pinned in
    ``tests/verification/test_distance_correlation_rung*``).

    The loss numerics are calibration-frozen: the golden projection
    thresholds and the OOS scripts' externally-seeded calibrations were
    validated against exactly this op sequence, so refactors must keep it
    byte-identical — ``tests/verification/test_dist_corr_loss_regression.py``
    fails on any numeric change. The deferred Linux-side recalibration is
    the only sanctioned occasion to move it.

    Parameters
    ----------
    Y_pred : (batch, d) predicted tensor.
    Y_true : (batch, d) target tensor.
    alpha : Weight for MSE term. Distance correlation gets (1 - alpha).

    Returns
    -------
    (loss_tensor, metrics_dict) where metrics_dict has 'mse',
    'dist_corr_loss', 'total'. The 'dist_corr_loss' entry is 1 minus the
    full-matrix correlation (exactly 1.0 for degenerate inputs, via the
    0.0-correlation guard).
    """
    import torch

    Y_pred_t = Y_pred if isinstance(Y_pred, torch.Tensor) else torch.tensor(Y_pred)
    Y_true_t = Y_true if isinstance(Y_true, torch.Tensor) else torch.tensor(Y_true)

    mse = torch.nn.functional.mse_loss(Y_pred_t, Y_true_t)

    # Pairwise distance correlation
    d_pred = torch.cdist(Y_pred_t, Y_pred_t).flatten()
    d_true = torch.cdist(Y_true_t, Y_true_t).flatten()

    d_pred_centered = d_pred - d_pred.mean()
    d_true_centered = d_true - d_true.mean()

    denom = d_pred_centered.norm() * d_true_centered.norm()
    if denom < 1e-12:
        dist_corr = torch.tensor(0.0, device=Y_pred_t.device)
    else:
        dist_corr = (d_pred_centered * d_true_centered).sum() / denom

    dist_corr_loss = 1.0 - dist_corr
    total = alpha * mse + (1.0 - alpha) * dist_corr_loss

    return total, {
        "mse": float(mse.item()),
        "dist_corr_loss": float(dist_corr_loss.item()),
        "total": float(total.item()),
    }


def train_projection_head(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.3,
    alpha: float = 0.7,
    n_folds: int = 5,
    batch_size: int = 64,
    max_epochs: int = 500,
    patience: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int | None = 42,
    head_factory: Callable[[int, int], _BaseProjectionHead] | None = None,
) -> dict:
    """Train projection head with stratified cross-validation.

    Parameters
    ----------
    X : (n_samples, input_dim) high-d embeddings.
    Y : (n_samples, output_dim) consensus UMAP targets.
    labels : (n_samples,) cluster labels for stratified splitting.
    hidden_layers : Hidden layer sizes (default: [128, 64]).
    dropout : Dropout probability.
    alpha : MSE weight in combined loss.
    n_folds : Number of CV folds.
    batch_size : Training batch size.
    max_epochs : Maximum training epochs per fold.
    patience : Early stopping patience.
    lr : Learning rate.
    weight_decay : L2 regularization.
    device : Torch device ("cpu", "cuda", "mps").
    seed : Base random seed (default 42). An int controls torch weight init,
        per-fold shuffling, and the stratified fold splits, so training is
        reproducible out of the box. Pass None to instead defer to the caller's
        global torch seed and keep the fold splits at their fixed default — use
        this to reproduce externally-seeded calibrations (the golden fixtures and
        the OOS scripts seed torch themselves and pass seed=None). The
        reporting-side metric clustering always uses its own fixed seed,
        independent of this one.
    head_factory : Builds the head from (input_dim, output_dim). Defaults to the
        MLP ProjectionHead; pass ``lambda i, o: LinearProjectionHead(i, o)`` for
        the linear baseline.

    Returns
    -------
    Dictionary with 'model' (ProjectionHead), 'cv_metrics', 'final_metrics'.

    Side effects: trains neural network, logs progress.
    """
    _check_torch()
    import torch
    from sklearn.model_selection import StratifiedKFold

    from reap.evaluation import (
        compute_distance_correlation,
        compute_silhouette,
        compute_trustworthiness,
    )

    if seed is not None:
        torch.manual_seed(seed)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    hidden = hidden_layers or [128, 64]

    def _default_head(in_dim: int, out_dim: int) -> _BaseProjectionHead:
        return ProjectionHead(in_dim, out_dim, hidden, dropout)

    make_head = head_factory if head_factory is not None else _default_head

    kfold_random_state = 42 if seed is None else seed
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=kfold_random_state)
    cv_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels)):
        logger.info("Fold %d/%d", fold + 1, n_folds)

        if seed is not None:
            torch.manual_seed(seed + fold)
        head = make_head(input_dim, output_dim).to(device)
        head.train_mode()
        optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=15, min_lr=1e-6
        )

        X_train = torch.from_numpy(X[train_idx].astype(np.float32)).to(device)
        Y_train = torch.from_numpy(Y[train_idx].astype(np.float32)).to(device)
        X_val_np, Y_val_np = X[val_idx], Y[val_idx]
        X_val_t = torch.from_numpy(X_val_np.astype(np.float32)).to(device)
        Y_val_t = torch.from_numpy(Y_val_np.astype(np.float32)).to(device)

        best_val_loss = float("inf")
        best_state: dict[str, object] | None = None
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            head.train_mode()
            perm = torch.randperm(len(X_train))

            for start in range(0, len(X_train), batch_size):
                idx = perm[start : start + batch_size]
                loss, _ = compute_projection_loss(
                    head._model(X_train[idx]), Y_train[idx], alpha
                )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[union-attr]
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()

            # Early stopping on VALIDATION loss, not training loss
            head.eval_mode()
            with torch.no_grad():
                val_loss, _ = compute_projection_loss(
                    head._model(X_val_t), Y_val_t, alpha
                )
            val_loss_value = float(val_loss.item())  # type: ignore[union-attr]
            scheduler.step(val_loss_value)

            if val_loss_value < best_val_loss - 1e-6:
                best_val_loss = val_loss_value
                best_state = {k: v.clone() for k, v in head._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best model weights from validation
        if best_state is not None:
            head._model.load_state_dict(best_state)

        # Evaluate fold
        Y_pred = head.forward(X_val_np)
        labels_val = labels[val_idx]
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score as _ari

        n_clusters = len(set(labels))
        km = KMeans(
            n_clusters=n_clusters,
            random_state=_METRIC_KMEANS_RANDOM_STATE,
            n_init="auto",
        )
        pred_labels = km.fit_predict(Y_pred)

        n_nn = min(15, len(X_val_np) - 1)
        fold_metrics = {
            "mse": float(np.mean((Y_pred - Y_val_np) ** 2)),
            "trustworthiness": compute_trustworthiness(
                X_val_np, Y_pred, n_neighbors=n_nn
            ),
            "silhouette": compute_silhouette(Y_pred, labels_val),
            "ari": float(_ari(labels_val, pred_labels)),
            "distance_correlation": compute_distance_correlation(Y_val_np, Y_pred),
        }
        cv_metrics.append(fold_metrics)
        logger.info("Fold %d metrics: %s", fold + 1, fold_metrics)

    # Train final model on all data
    logger.info("Training final model on all %d samples", len(X))
    if seed is not None:
        torch.manual_seed(seed + n_folds)
    final_head = make_head(input_dim, output_dim).to(device)
    final_head.train_mode()
    final_optimizer = torch.optim.Adam(final_head.parameters(), lr=lr, weight_decay=weight_decay)
    final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer, factor=0.5, patience=15, min_lr=1e-6
    )

    X_all = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_all = torch.from_numpy(Y.astype(np.float32)).to(device)

    best_final_loss = float("inf")
    best_final_state: dict[str, object] | None = None
    final_no_improve = 0

    for epoch in range(max_epochs):
        final_head.train_mode()
        perm = torch.randperm(len(X_all))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_all), batch_size):
            idx = perm[start : start + batch_size]
            loss, _ = compute_projection_loss(final_head._model(X_all[idx]), Y_all[idx], alpha)
            final_optimizer.zero_grad()
            loss.backward()  # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(final_head.parameters(), max_norm=1.0)
            final_optimizer.step()
            epoch_loss += loss.item()  # type: ignore[union-attr]
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        final_scheduler.step(avg_loss)

        if avg_loss < best_final_loss - 1e-6:
            best_final_loss = avg_loss
            best_final_state = {
                k: v.clone() for k, v in final_head._model.state_dict().items()
            }
            final_no_improve = 0
        else:
            final_no_improve += 1
            if final_no_improve >= patience:
                logger.info("Final model early stopping at epoch %d", epoch + 1)
                break

    # Restore best weights
    if best_final_state is not None:
        final_head._model.load_state_dict(best_final_state)

    Y_pred_final = final_head.forward(X)
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.metrics import adjusted_rand_score as _ari_final

    n_clusters = len(set(labels))
    km_final = _KMeans(
        n_clusters=n_clusters,
        random_state=_METRIC_KMEANS_RANDOM_STATE,
        n_init="auto",
    )
    pred_labels_final = km_final.fit_predict(Y_pred_final)

    n_nn = min(15, len(X) - 1)
    final_metrics = {
        "mse": float(np.mean((Y_pred_final - Y) ** 2)),
        "trustworthiness": compute_trustworthiness(
            X, Y_pred_final, n_neighbors=n_nn
        ),
        "silhouette": compute_silhouette(Y_pred_final, labels),
        "ari": float(_ari_final(labels, pred_labels_final)),
        "distance_correlation": compute_distance_correlation(Y, Y_pred_final),
    }

    return {
        "model": final_head,
        "cv_metrics": cv_metrics,
        "final_metrics": final_metrics,
        "config": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_layers": hidden,
            "dropout": dropout,
            "alpha": alpha,
            "n_folds": n_folds,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }
