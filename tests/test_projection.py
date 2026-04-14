"""Tests for the neural projection head module.

Tests cover: model construction, forward pass, save/load round-trip,
loss computation, and training pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for projection tests")

from reap.projection import (  # noqa: E402
    ProjectionHead,
    compute_projection_loss,
    train_projection_head,
)


class TestProjectionHead:
    """Tests for ProjectionHead model construction and forward pass."""

    def test_default_architecture(self) -> None:
        head = ProjectionHead(input_dim=32, output_dim=5)
        assert head.input_dim == 32
        assert head.output_dim == 5

    def test_forward_shape(self) -> None:
        head = ProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        out = head.forward(X)
        assert out.shape == (20, 5)

    def test_forward_no_nan(self) -> None:
        head = ProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        out = head.forward(X)
        assert np.all(np.isfinite(out))

    def test_custom_hidden_layers(self) -> None:
        head = ProjectionHead(input_dim=64, output_dim=3, hidden_layers=[32, 16, 8])
        X = np.random.default_rng(42).standard_normal((10, 64)).astype(np.float32)
        out = head.forward(X)
        assert out.shape == (10, 3)

    def test_eval_mode_deterministic(self) -> None:
        head = ProjectionHead(input_dim=32, output_dim=5, dropout=0.5)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        out1 = head.forward(X)
        out2 = head.forward(X)
        np.testing.assert_array_equal(out1, out2)


class TestSaveLoad:
    """Tests for model serialization."""

    def test_round_trip_preserves_weights(self) -> None:
        head = ProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        out_before = head.forward(X)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        head.save(path)

        head2 = ProjectionHead(input_dim=32, output_dim=5)
        head2.load(path)
        out_after = head2.forward(X)

        np.testing.assert_array_almost_equal(out_before, out_after, decimal=6)
        Path(path).unlink()


class TestProjectionLoss:
    """Tests for the combined MSE + distance correlation loss."""

    def test_zero_loss_for_identical_inputs(self) -> None:
        Y = torch.randn(20, 5)
        _, metrics = compute_projection_loss(Y, Y, alpha=0.7)
        assert metrics["mse"] < 1e-6
        assert metrics["total"] < 1e-6

    def test_loss_is_positive_for_different_inputs(self) -> None:
        Y_pred = torch.randn(20, 5)
        Y_true = torch.randn(20, 5)
        _, metrics = compute_projection_loss(Y_pred, Y_true, alpha=0.7)
        assert metrics["total"] > 0

    def test_alpha_one_is_pure_mse(self) -> None:
        Y_pred = torch.randn(20, 5)
        Y_true = torch.randn(20, 5)
        _, metrics = compute_projection_loss(Y_pred, Y_true, alpha=1.0)
        assert abs(metrics["total"] - metrics["mse"]) < 1e-6

    def test_alpha_zero_is_pure_dist_corr(self) -> None:
        Y_pred = torch.randn(20, 5)
        Y_true = torch.randn(20, 5)
        _, metrics = compute_projection_loss(Y_pred, Y_true, alpha=0.0)
        assert abs(metrics["total"] - metrics["dist_corr_loss"]) < 1e-6

    def test_loss_is_finite(self) -> None:
        Y_pred = torch.randn(50, 5)
        Y_true = torch.randn(50, 5)
        _, metrics = compute_projection_loss(Y_pred, Y_true)
        assert np.isfinite(metrics["total"])
        assert np.isfinite(metrics["mse"])
        assert np.isfinite(metrics["dist_corr_loss"])


class TestTrainProjectionHead:
    """Tests for the full training pipeline."""

    @pytest.fixture(scope="class")
    def training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synthetic training data: 100 samples, 32-d input, 5-d output, 3 clusters."""
        rng = np.random.default_rng(42)
        n_samples = 100
        input_dim = 32
        output_dim = 5

        X = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
        Y = rng.standard_normal((n_samples, output_dim)).astype(np.float32)
        labels = np.array([i % 3 for i in range(n_samples)])
        return X, Y, labels

    def test_returns_expected_keys(
        self, training_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = training_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=5, patience=3, batch_size=32
        )
        assert "model" in result
        assert "cv_metrics" in result
        assert "final_metrics" in result
        assert "config" in result

    def test_model_is_projection_head(
        self, training_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = training_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=5, patience=3, batch_size=32
        )
        assert isinstance(result["model"], ProjectionHead)

    def test_cv_metrics_per_fold(
        self, training_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = training_data
        n_folds = 3
        result = train_projection_head(
            X, Y, labels, n_folds=n_folds, max_epochs=5, patience=3, batch_size=32
        )
        assert len(result["cv_metrics"]) == n_folds
        for fold_metrics in result["cv_metrics"]:
            assert "mse" in fold_metrics
            assert "trustworthiness" in fold_metrics
            assert "silhouette" in fold_metrics
            assert "ari" in fold_metrics
            assert "distance_correlation" in fold_metrics

    def test_final_metrics_have_expected_keys(
        self, training_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = training_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=5, patience=3, batch_size=32
        )
        final = result["final_metrics"]
        assert "mse" in final
        assert "trustworthiness" in final
        assert final["mse"] >= 0.0

    def test_trained_model_produces_correct_shape(
        self, training_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = training_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=5, patience=3, batch_size=32
        )
        model: ProjectionHead = result["model"]
        out = model.forward(X)
        assert out.shape == Y.shape
