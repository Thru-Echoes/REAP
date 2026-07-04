"""Byte-identical regression guard for the ``dist_corr_loss`` recipe.

The projection-training loss numerics are calibration-frozen: the golden
projection thresholds and the OOS scripts' externally-seeded calibrations
were validated against exactly today's op sequence, and the deferred
Linux-side recalibration is the only sanctioned occasion to move them. A
refactor that changes the loss arithmetic — even one that is mathematically
equivalent — silently shifts trained heads and every downstream calibrated
number. This module makes any such change fail loudly on every platform.

Mechanism: ``_frozen_projection_loss`` is a deliberately frozen replica of
the exact op sequence inside ``reap.projection.compute_projection_loss``
(torch ``cdist`` → ``flatten`` → mean-centering → norm-product guard →
``1 − r`` → ``alpha``-blend with MSE). On fixed inputs the production
function must produce a **bitwise-identical** loss tensor
(``torch.equal``) and an exactly equal metrics dict. Because production and
replica run in the same process on the same inputs, this holds on any
platform — no platform-gated golden constants needed — and any numeric
refactor of the production function breaks it immediately.

Do NOT "clean up" the replica to share code with production, and do not
relax ``torch.equal`` to a tolerance: byte-identity IS the contract. If the
loss must ever change, that change lands together with the golden-threshold
recalibration, and this replica is updated in the same commit with a
CHANGELOG entry (see the scientific-conventions randomness/tolerance rules).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from reap.projection import compute_projection_loss  # noqa: E402


def _frozen_projection_loss(Y_pred, Y_true, alpha):
    """Frozen replica of compute_projection_loss's exact op sequence.

    Intentionally duplicated (see module docstring) — must never be
    refactored to call the production function or share helpers with it.
    """
    mse = torch.nn.functional.mse_loss(Y_pred, Y_true)

    d_pred = torch.cdist(Y_pred, Y_pred).flatten()
    d_true = torch.cdist(Y_true, Y_true).flatten()

    d_pred_centered = d_pred - d_pred.mean()
    d_true_centered = d_true - d_true.mean()

    denom = d_pred_centered.norm() * d_true_centered.norm()
    if denom < 1e-12:
        dist_corr = torch.tensor(0.0, device=Y_pred.device)
    else:
        dist_corr = (d_pred_centered * d_true_centered).sum() / denom

    dist_corr_loss = 1.0 - dist_corr
    total = alpha * mse + (1.0 - alpha) * dist_corr_loss

    return total, {
        "mse": float(mse.item()),
        "dist_corr_loss": float(dist_corr_loss.item()),
        "total": float(total.item()),
    }


def _fixed_batch(dtype) -> tuple:
    """Deterministic 32×18 prediction/target pair in the requested dtype."""
    rng = np.random.default_rng(42)
    Y_true = rng.normal(size=(32, 18))
    Y_pred = Y_true + 0.2 * rng.normal(size=(32, 18))
    return (
        torch.tensor(Y_pred, dtype=dtype),
        torch.tensor(Y_true, dtype=dtype),
    )


@pytest.mark.parametrize("alpha", [0.7, 0.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
def test_loss_tensor_is_byte_identical_to_frozen_replica(dtype, alpha):
    """Fixed inputs ⇒ production loss tensor bitwise-equals the frozen replica."""
    Y_pred, Y_true = _fixed_batch(dtype)
    total_prod, metrics_prod = compute_projection_loss(Y_pred, Y_true, alpha=alpha)
    total_frozen, metrics_frozen = _frozen_projection_loss(Y_pred, Y_true, alpha=alpha)
    assert isinstance(total_prod, torch.Tensor)
    assert total_prod.dtype == dtype
    assert torch.equal(total_prod, total_frozen)
    assert metrics_prod == metrics_frozen  # exact float equality, not approx


def test_degenerate_guard_path_is_byte_identical():
    """The denom < 1e-12 guard branch is part of the frozen recipe too."""
    Y_pred = torch.zeros((6, 4), dtype=torch.float32)
    Y_true, _ = _fixed_batch(torch.float32)
    Y_true = Y_true[:6, :4].contiguous()
    total_prod, metrics_prod = compute_projection_loss(Y_pred, Y_true, alpha=0.7)
    total_frozen, metrics_frozen = _frozen_projection_loss(Y_pred, Y_true, alpha=0.7)
    assert isinstance(total_prod, torch.Tensor)
    assert torch.equal(total_prod, total_frozen)
    assert metrics_prod == metrics_frozen
    assert metrics_prod["dist_corr_loss"] == 1.0  # guard yields corr 0.0 ⇒ loss exactly 1
