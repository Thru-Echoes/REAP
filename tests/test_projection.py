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
    LinearProjectionHead,
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


class TestLinearProjectionHead:
    """Tests for the single-layer linear projection head (the §11 linear baseline).

    The linear head is a pre-registered comparison point for the MLP head: it maps
    the input to the consensus space with one affine transform and no hidden layers.
    """

    def test_construction_records_dims(self) -> None:
        head = LinearProjectionHead(input_dim=32, output_dim=5)
        assert head.input_dim == 32
        assert head.output_dim == 5

    def test_forward_shape(self) -> None:
        head = LinearProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(42).standard_normal((20, 32)).astype(np.float32)
        assert head.forward(X).shape == (20, 5)

    def test_is_a_single_affine_map(self) -> None:
        # A linear head has exactly one weight matrix (in*out) plus one bias vector
        # (out) — no hidden layers, batchnorm, or dropout parameters.
        head = LinearProjectionHead(input_dim=8, output_dim=3)
        n_params = sum(int(p.numel()) for p in head.parameters())
        assert n_params == 8 * 3 + 3

    def test_forward_is_finite(self) -> None:
        head = LinearProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(0).standard_normal((15, 32)).astype(np.float32)
        assert np.all(np.isfinite(head.forward(X)))

    def test_eval_is_deterministic(self) -> None:
        head = LinearProjectionHead(input_dim=32, output_dim=5)
        X = np.random.default_rng(1).standard_normal((10, 32)).astype(np.float32)
        np.testing.assert_array_equal(head.forward(X), head.forward(X))

    def test_save_load_round_trip(self) -> None:
        head = LinearProjectionHead(input_dim=16, output_dim=4)
        X = np.random.default_rng(2).standard_normal((12, 16)).astype(np.float32)
        before = head.forward(X)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        head.save(path)
        head2 = LinearProjectionHead(input_dim=16, output_dim=4)
        head2.load(path)
        np.testing.assert_array_almost_equal(before, head2.forward(X), decimal=6)
        Path(path).unlink()


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


class TestHeadFactoryAndSeed:
    """Tests for the pluggable head factory and reproducible seeding.

    train_projection_head must (a) default to the MLP head, (b) train whatever
    head a caller supplies via head_factory, and (c) be reproducible: the same
    seed yields the same trained model, a different seed yields a different one.
    """

    @pytest.fixture(scope="class")
    def small_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        n_samples, input_dim, output_dim = 100, 16, 4
        X = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
        Y = rng.standard_normal((n_samples, output_dim)).astype(np.float32)
        labels = np.array([i % 3 for i in range(n_samples)])
        return X, Y, labels

    def test_default_factory_trains_mlp_head(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = small_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=3, patience=2, batch_size=32, seed=0
        )
        assert isinstance(result["model"], ProjectionHead)

    def test_head_factory_trains_linear_head(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = small_data
        result = train_projection_head(
            X,
            Y,
            labels,
            n_folds=2,
            max_epochs=3,
            patience=2,
            batch_size=32,
            seed=0,
            head_factory=lambda in_dim, out_dim: LinearProjectionHead(in_dim, out_dim),
        )
        assert isinstance(result["model"], LinearProjectionHead)

    def test_same_seed_is_reproducible(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = small_data
        kwargs = dict(n_folds=2, max_epochs=8, patience=5, batch_size=32, seed=123)
        r1 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        r2 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(r1["model"].forward(X), r2["model"].forward(X))

    def test_different_seed_changes_result(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = small_data
        kwargs = dict(n_folds=2, max_epochs=8, patience=5, batch_size=32)
        r1 = train_projection_head(X, Y, labels, seed=1, **kwargs)  # type: ignore[arg-type]
        r2 = train_projection_head(X, Y, labels, seed=2, **kwargs)  # type: ignore[arg-type]
        assert not np.allclose(r1["model"].forward(X), r2["model"].forward(X))

    def test_config_records_seed(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, Y, labels = small_data
        result = train_projection_head(
            X, Y, labels, n_folds=2, max_epochs=3, patience=2, batch_size=32, seed=99
        )
        assert result["config"]["seed"] == 99

    def test_seed_none_defers_to_external_torch_seed(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        # With seed=None, training must NOT reseed torch internally — it uses the
        # global torch seed the caller set beforehand. This is the contract the
        # pre-calibrated golden fixtures and the OOS scripts rely on: they seed
        # torch externally and expect the result to depend on that seed. So the
        # same external seed reproduces the result, and a different one changes it.
        import torch

        X, Y, labels = small_data
        kwargs = dict(n_folds=2, max_epochs=8, patience=5, batch_size=32, seed=None)
        torch.manual_seed(2024)
        r1 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        torch.manual_seed(2024)
        r2 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(r1["model"].forward(X), r2["model"].forward(X))
        torch.manual_seed(777)
        r3 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        assert not np.allclose(r1["model"].forward(X), r3["model"].forward(X))

    def test_default_is_deterministic(
        self, small_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        # Out of the box — no seed argument and no external torch seeding —
        # training is reproducible: the default seed makes two independent calls
        # produce the same model. This is the determinism-by-default contract.
        X, Y, labels = small_data
        kwargs = dict(n_folds=2, max_epochs=8, patience=5, batch_size=32)
        r1 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        r2 = train_projection_head(X, Y, labels, **kwargs)  # type: ignore[arg-type]
        np.testing.assert_array_equal(r1["model"].forward(X), r2["model"].forward(X))


class TestBatchOfOneRow:
    """Training must not stop when the row count leaves a last batch of one row.

    ``ProjectionHead`` normalises each hidden layer with ``BatchNorm1d``, which in
    training mode refuses a batch of one row ("Expected more than 1 value per
    channel when training"). A training set whose size is one more than a
    multiple of ``batch_size`` used to produce exactly such a batch, in the final
    fit on all rows and in the training part of each inner fold.
    """

    @staticmethod
    def _make_data(
        n_samples: int, n_classes: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random 16-d inputs, 4-d targets, and balanced labels, seeded."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((n_samples, 16)).astype(np.float32)
        Y = rng.standard_normal((n_samples, 4)).astype(np.float32)
        labels = np.array([i % n_classes for i in range(n_samples)])
        return X, Y, labels

    @staticmethod
    def _inner_train_sizes(labels: np.ndarray, n_folds: int, seed: int) -> list[int]:
        """Training-part sizes of the inner folds, split as train_projection_head does."""
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        dummy_X = np.zeros((len(labels), 1))
        return [len(train_idx) for train_idx, _ in skf.split(dummy_X, labels)]

    def test_final_fit_with_remainder_of_one(self) -> None:
        """129 rows with batch_size=64: the final fit's last batch would be one row."""
        X, Y, labels = self._make_data(n_samples=129, n_classes=3)
        n_folds, seed, batch_size = 3, 0, 64
        # Precondition: the inner folds do not hit the case, so this test
        # isolates the final fit on all rows.
        assert all(
            size % batch_size != 1 for size in self._inner_train_sizes(labels, n_folds, seed)
        )
        assert len(X) % batch_size == 1

        result = train_projection_head(
            X, Y, labels, n_folds=n_folds, batch_size=batch_size,
            max_epochs=3, patience=2, seed=seed,
        )
        assert isinstance(result["model"], ProjectionHead)
        assert np.all(np.isfinite(result["model"].forward(X)))

    def test_inner_fold_with_remainder_of_one(self) -> None:
        """130 rows, two folds: each inner training part has 65 rows (64 + 1)."""
        X, Y, labels = self._make_data(n_samples=130, n_classes=2)
        n_folds, seed, batch_size = 2, 0, 64
        assert self._inner_train_sizes(labels, n_folds, seed) == [65, 65]
        assert len(X) % batch_size != 1

        result = train_projection_head(
            X, Y, labels, n_folds=n_folds, batch_size=batch_size,
            max_epochs=3, patience=2, seed=seed,
        )
        assert len(result["cv_metrics"]) == n_folds
        assert np.all(np.isfinite(result["model"].forward(X)))

    def test_single_batch_of_batch_size_plus_one(self) -> None:
        """65 rows, batch_size=64, in the final fit: one batch of 65 rows."""
        X, Y, labels = self._make_data(n_samples=65, n_classes=2)
        result = train_projection_head(
            X, Y, labels, n_folds=2, batch_size=64, max_epochs=2, patience=2, seed=0,
        )
        assert np.all(np.isfinite(result["model"].forward(X)))


class TestGetBatchBounds:
    """Unit tests for the batch splitter used by both training loops."""

    def test_even_split_is_unchanged(self) -> None:
        from reap.projection import _get_batch_bounds

        assert _get_batch_bounds(128, 64) == [(0, 64), (64, 128)]

    def test_ordinary_remainder_is_unchanged(self) -> None:
        from reap.projection import _get_batch_bounds

        assert _get_batch_bounds(130, 64) == [(0, 64), (64, 128), (128, 130)]

    def test_remainder_of_one_joins_previous_batch(self) -> None:
        from reap.projection import _get_batch_bounds

        assert _get_batch_bounds(129, 64) == [(0, 64), (64, 129)]
        assert _get_batch_bounds(4801, 64)[-1] == (4736, 4801)

    def test_fewer_rows_than_batch_size(self) -> None:
        from reap.projection import _get_batch_bounds

        assert _get_batch_bounds(10, 64) == [(0, 10)]
        assert _get_batch_bounds(1, 64) == [(0, 1)]

    def test_batch_size_one_is_not_merged(self) -> None:
        from reap.projection import _get_batch_bounds

        assert _get_batch_bounds(3, 1) == [(0, 1), (1, 2), (2, 3)]

    @pytest.mark.parametrize("n_rows", list(range(0, 200)))
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
    def test_every_row_once_and_no_batch_of_one(self, n_rows: int, batch_size: int) -> None:
        """Bounds tile 0..n_rows contiguously; a batch of one only when unavoidable."""
        from reap.projection import _get_batch_bounds

        bounds = _get_batch_bounds(n_rows, batch_size)
        covered = [i for start, stop in bounds for i in range(start, stop)]
        assert covered == list(range(n_rows))
        if batch_size > 1 and n_rows > 1:
            assert all(stop - start >= 2 for start, stop in bounds)
        assert all(stop - start <= batch_size + 1 for start, stop in bounds)


class TestProjectionExports:
    """The projection head is REAP's headline out-of-sample feature, so its public
    names must be reachable from the top-level ``reap`` package, not just from the
    submodule."""

    def test_projection_names_are_public(self) -> None:
        import reap

        for name in (
            "ProjectionHead",
            "LinearProjectionHead",
            "train_projection_head",
            "compute_projection_loss",
        ):
            assert name in reap.__all__, f"{name} missing from reap.__all__"
            assert hasattr(reap, name), f"{name} not importable from reap"


class TestTorchInstallHint:
    """The torch-missing error must name the real installable package."""

    def test_missing_torch_hint_names_real_package(self, monkeypatch) -> None:
        """_check_torch's ImportError points at reap-topics[projection].

        The hint is what a user actually types: the distribution on PyPI is
        ``reap-topics``, so any other name in the message sends them to a
        package that does not exist (or worse, someone else's).
        """
        import builtins

        from reap.projection import _check_torch

        real_import = builtins.__import__

        def _no_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked: torch not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        with pytest.raises(ImportError, match=r"reap-topics\[projection\]"):
            _check_torch()
