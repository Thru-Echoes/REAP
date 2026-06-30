"""Projection head — CV-R² distribution and predicted-vs-target scatter.

Trains the REAP projection head on the 20newsgroups golden text fixture
(§6.b of the evaluation protocol) using the same 10-seed Set A
configuration as `tests/test_projection_golden.py`, then renders:

Panel (a) — R² across 3 stratified CV folds as a boxplot, overlaid
   with per-fold dots. R² per fold is computed as 1 − MSE / variance
   of the target consensus coordinates (same definition as in the
   golden test). The pre-registered floor (CV R² ≥ 0.60 in protocol
   §6.c) is drawn as a horizontal reference line.

Panel (b) — predicted vs target consensus coordinates on a single
   held-out CV fold (fold 0). One panel per output dimension (top 4
   dimensions by target variance) with the identity line overlaid.

This uses the package code (`reap.projection.train_projection_head`),
not a custom training loop, so the figure is regenerated against
whatever shipped MLP/loss is current.

Inputs
------
20newsgroups golden text fixture (auto-cached by
`reap.datasets.load_golden_text`; ~/.cache/reap/datasets/...).

Outputs
-------
manuscript/figures/projection_head_scatter.png
manuscript/figures/projection_head_scatter.pdf

Notes
-----
* Runs ~30-60 s on CPU (no GPU dependency).
* Deterministic via `np.random.seed`, `torch.manual_seed`, and the same
  fold-splitting random_state=42 used by `train_projection_head`.
* If the golden text snapshot or the torch dependency is unavailable,
  the script raises with a clear message naming what is missing so the
  caller can decide whether to skip this figure.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PNG = REPO_ROOT / "manuscript" / "figures" / "projection_head_scatter.png"
OUT_PDF = REPO_ROOT / "manuscript" / "figures" / "projection_head_scatter.pdf"
SEED_MANIFEST_PATH = REPO_ROOT / "manuscript" / "seeds" / "seed_manifest.json"

# Mirror the golden-test config so the figure tracks the validated regime.
GOLDEN_K = 8
GOLDEN_N_COMPONENTS = 8
GOLDEN_N_NEIGHBORS = 15
GOLDEN_MIN_DIST = 0.1
GOLDEN_N_SEEDS = 10
PROJECTION_N_FOLDS = 3
PROJECTION_MAX_EPOCHS = 150
PROJECTION_PATIENCE = 30
PROJECTION_BATCH_SIZE = 32
PROJECTION_LR = 1e-3
TORCH_SEED = 42

# Pre-registered floor from manuscript/evaluation_protocol.md §6.c.
CV_R2_FLOOR = 0.60


def _set_torch_deterministic(seed: int) -> None:
    """Seed torch (and CUDA if available) for reproducible training."""
    import torch  # local import so script-import doesn't require torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_set_a_seeds(n: int) -> list[int]:
    """Load the first n seeds from Set A of the project seed manifest."""
    manifest = json.loads(SEED_MANIFEST_PATH.read_text())
    seeds = manifest["sets"]["A"]["seeds"][:n]
    return [int(s) for s in seeds]


def get_reap_consensus(
    X: np.ndarray,
    seeds: list[int],
) -> np.ndarray:
    """Run the REAP consensus pipeline at the golden config.

    Returns the (N, d) consensus embedding.
    """
    from reap.consensus import (
        get_consensus_distance_matrix,
        get_consensus_embedding,
        get_multi_seed_embeddings,
    )
    embs = get_multi_seed_embeddings(
        X, seeds,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    D = get_consensus_distance_matrix(embs)
    Y, _ = get_consensus_embedding(
        D,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    return Y


def get_cv_r2_per_fold(
    train_result: dict,
    consensus_full: np.ndarray,
) -> list[float]:
    """Per-fold R² using the same definition as test_projection_golden.py.

    R² = 1 − MSE_fold / var(consensus_full). Matches the headline metric
    enforced by the pre-registered floor (§6.c).
    """
    y_var = float(np.var(consensus_full))
    if y_var <= 0:
        raise RuntimeError("Consensus targets have zero variance — degenerate.")
    return [1.0 - float(m["mse"]) / y_var for m in train_result["cv_metrics"]]


def get_first_fold_held_out_predictions(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-train the head on fold 0's training partition; predict on fold 0's val.

    Note: `train_projection_head` does not expose per-fold val predictions
    on its public surface (it computes per-fold metrics internally). We
    reproduce its first fold here using the same stratified split
    (`random_state=42`, the package's hard-coded value) so the scatter
    is taken from the same held-out partition the harness scores on.
    """
    import torch

    from reap.projection import ProjectionHead, compute_projection_loss

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(X, labels)))
    X_train_np = X[train_idx].astype(np.float32)
    Y_train_np = Y[train_idx].astype(np.float32)
    X_val_np = X[val_idx].astype(np.float32)
    Y_val_np = Y[val_idx].astype(np.float32)

    _set_torch_deterministic(TORCH_SEED)
    head = ProjectionHead(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_layers=[128, 64],
        dropout=0.3,
    )
    optimizer = torch.optim.Adam(
        head.parameters(), lr=PROJECTION_LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6,
    )

    X_t = torch.from_numpy(X_train_np)
    Y_t = torch.from_numpy(Y_train_np)
    X_val_t = torch.from_numpy(X_val_np)
    Y_val_t = torch.from_numpy(Y_val_np)

    best_val = float("inf")
    best_state: dict[str, object] | None = None
    no_improve = 0
    for epoch in range(PROJECTION_MAX_EPOCHS):
        head.train_mode()
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), PROJECTION_BATCH_SIZE):
            idx = perm[start : start + PROJECTION_BATCH_SIZE]
            loss, _ = compute_projection_loss(
                head._model(X_t[idx]), Y_t[idx], alpha=0.7,
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[union-attr]
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
        head.eval_mode()
        with torch.no_grad():
            val_loss, _ = compute_projection_loss(
                head._model(X_val_t), Y_val_t, alpha=0.7,
            )
        v = float(val_loss.item())  # type: ignore[union-attr]
        scheduler.step(v)
        if v < best_val - 1e-6:
            best_val = v
            best_state = {k: t.clone() for k, t in head._model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PROJECTION_PATIENCE:
                logger.info("Held-out fold early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        head._model.load_state_dict(best_state)
    Y_pred = head.forward(X_val_np)
    return Y_val_np, Y_pred


def plot_cv_r2_distribution(
    ax: Axes,
    r2_per_fold: list[float],
) -> None:
    """Boxplot of CV R² with individual fold dots overlaid."""
    box = ax.boxplot(
        [r2_per_fold], vert=True, widths=0.45,
        patch_artist=True,
        medianprops=dict(color="#222", linewidth=1.5),
        boxprops=dict(facecolor="#9ec5e8", edgecolor="#1f4e79"),
        whiskerprops=dict(color="#1f4e79"),
        capprops=dict(color="#1f4e79"),
    )
    _ = box
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(r2_per_fold))
    ax.scatter(1 + jitter, r2_per_fold, color="#1f4e79",
               s=46, zorder=3, edgecolor="white", linewidth=0.6)
    for i, val in enumerate(r2_per_fold):
        ax.annotate(
            f"fold {i + 1}: {val:.3f}",
            xy=(1 + jitter[i], val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color="#222",
        )

    ax.axhline(CV_R2_FLOOR, color="#bf3a3a", linestyle="--", linewidth=1,
               label=f"pre-registered floor R² $\\geq$ {CV_R2_FLOOR:.2f}")
    ax.set_xticks([1])
    ax.set_xticklabels(["3-fold CV"])
    ax.set_ylabel("R²")
    ax.set_ylim(min(0.5, min(r2_per_fold) - 0.05),
                max(1.0, max(r2_per_fold) + 0.05))
    ax.set_title("(a) Projection head CV R² distribution")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_predicted_vs_target_scatter(
    fig: Figure,
    gs: object,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    base_row: int,
    base_col: int,
) -> None:
    """Predicted vs target scatter for the top-4 high-variance dimensions."""
    # Pick the four output dimensions with the highest target variance to
    # make the visual signal stronger.
    var_per_dim = Y_true.var(axis=0)
    top_dims = np.argsort(var_per_dim)[::-1][:4]
    for k, dim in enumerate(top_dims):
        r = base_row + k // 2
        c = base_col + (k % 2)
        ax = fig.add_subplot(gs[r, c])  # type: ignore[arg-type]
        y_t = Y_true[:, dim]
        y_p = Y_pred[:, dim]
        ax.scatter(y_t, y_p, s=18, color="#1f4e79", alpha=0.55,
                   edgecolor="none")
        lim_low = float(min(y_t.min(), y_p.min()))
        lim_high = float(max(y_t.max(), y_p.max()))
        ax.plot([lim_low, lim_high], [lim_low, lim_high],
                color="#bf3a3a", linewidth=1, linestyle="--")
        # Per-dim Pearson correlation (signed) as a numeric anchor.
        if y_t.std() > 0 and y_p.std() > 0:
            r_val = float(np.corrcoef(y_t, y_p)[0, 1])
            ax.text(0.04, 0.95, f"r = {r_val:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    color="#222")
        ax.set_title(f"output dim {int(dim)}", fontsize=10)
        ax.set_xlabel("target")
        ax.set_ylabel("predicted")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def main() -> None:
    np.random.seed(0)

    try:
        from reap.datasets import load_golden_text
    except Exception as exc:
        raise RuntimeError(
            "reap.datasets.load_golden_text unavailable: " + str(exc)
        ) from exc

    snap = load_golden_text()
    seeds = _load_set_a_seeds(GOLDEN_N_SEEDS)
    logger.info(
        "Loaded golden text fixture: N=%d, d_in=%d, K_classes=%d",
        snap.embeddings.shape[0], snap.embeddings.shape[1],
        int(np.unique(snap.labels).size) if snap.labels is not None else -1,
    )

    Y_consensus = get_reap_consensus(snap.embeddings, seeds)

    from reap.projection import train_projection_head
    _set_torch_deterministic(TORCH_SEED)
    train_result = train_projection_head(
        snap.embeddings.astype(np.float32),
        Y_consensus.astype(np.float32),
        snap.labels,  # type: ignore[arg-type]
        n_folds=PROJECTION_N_FOLDS,
        max_epochs=PROJECTION_MAX_EPOCHS,
        patience=PROJECTION_PATIENCE,
        batch_size=PROJECTION_BATCH_SIZE,
        lr=PROJECTION_LR,
    )
    r2_per_fold = get_cv_r2_per_fold(train_result, Y_consensus)
    logger.info("CV R² per fold: %s (mean %.3f)",
                [f"{v:.3f}" for v in r2_per_fold], float(np.mean(r2_per_fold)))

    Y_true_held, Y_pred_held = get_first_fold_held_out_predictions(
        snap.embeddings, Y_consensus, snap.labels,  # type: ignore[arg-type]
        n_folds=PROJECTION_N_FOLDS,
    )

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.30,
                          width_ratios=[1.0, 1.0, 1.0])
    ax_box = fig.add_subplot(gs[:, 0])
    plot_cv_r2_distribution(ax_box, r2_per_fold)

    plot_predicted_vs_target_scatter(
        fig, gs, Y_true_held, Y_pred_held, base_row=0, base_col=1,
    )

    suptitle = (
        "Projection head validation on the 20ng golden fixture "
        "(N=400, d_in=384, d_out=8)"
    )
    fig.suptitle(suptitle, fontsize=11.5, y=1.0)
    # The right-hand 2x2 grid is the scatter; add a label spanning them.
    fig.text(
        0.70, 0.97,
        "(b) Predicted vs target (CV fold 1 held-out)",
        ha="center", va="top", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s and %s", OUT_PNG, OUT_PDF)


if __name__ == "__main__":
    main()
