"""Unit tests for the Parametric UMAP baseline wrapper.

Verifies the (embedding, random_state) contract on tiny synthetic blobs.
Skips cleanly when ``tensorflow`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

pytest.importorskip("tensorflow")
pytest.importorskip("umap.parametric_umap")


def _make_synthetic(
    n_samples: int = 80, n_features: int = 32, n_centers: int = 4
) -> np.ndarray:
    """Synthetic blob embeddings, L2-normalised to mimic sentence-encoder output."""
    X, _ = make_blobs(  # type: ignore[misc]
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=0.5,
        random_state=42,
        return_centers=False,
    )
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_parametric_umap_baseline_shape_and_no_nan() -> None:
    """Output embedding has the right shape, is finite, and carries the seed."""
    from reap.baselines import ParametricUMAPResult, run_parametric_umap_baseline

    X = _make_synthetic(n_samples=80, n_features=32, n_centers=4)

    result = run_parametric_umap_baseline(
        embeddings=X,
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=123,
        n_epochs=10,  # quick for unit test
    )

    assert isinstance(result, ParametricUMAPResult)
    assert result.method == "parametric_umap"
    assert result.random_state == 123

    assert result.embedding.shape == (80, 2)
    assert np.all(np.isfinite(result.embedding)), "Embedding contains NaN/inf"


def test_parametric_umap_baseline_is_deterministic_given_seed() -> None:
    """Same input + same seed → byte-identical embedding (within float tolerance)."""
    from reap.baselines import run_parametric_umap_baseline

    X = _make_synthetic(n_samples=60, n_features=16, n_centers=3)

    result_a = run_parametric_umap_baseline(
        embeddings=X, n_components=2, n_neighbors=10, min_dist=0.1,
        random_state=7, n_epochs=10,
    )
    result_b = run_parametric_umap_baseline(
        embeddings=X, n_components=2, n_neighbors=10, min_dist=0.1,
        random_state=7, n_epochs=10,
    )

    # Determinism — TF op_determinism is enabled in _seed_tensorflow, so
    # outputs should match within tight tolerance.
    np.testing.assert_allclose(
        result_a.embedding, result_b.embedding, rtol=1e-4, atol=1e-4,
    )


def test_parametric_umap_holdout_projects_unseen_documents() -> None:
    """Fit on the in-set split only; held-out rows get finite coordinates of the right shape."""
    from reap.baselines import ParametricUMAPHoldoutResult, run_parametric_umap_holdout

    X = _make_synthetic(n_samples=100, n_features=32, n_centers=4)
    result = run_parametric_umap_holdout(
        X[:80], X[80:], n_components=2, n_neighbors=10, min_dist=0.1,
        random_state=7, n_epochs=10,
    )
    assert isinstance(result, ParametricUMAPHoldoutResult)
    assert result.embedding_in.shape == (80, 2)
    assert result.embedding_out.shape == (20, 2)
    assert np.all(np.isfinite(result.embedding_out))
    assert result.random_state == 7


def test_parametric_umap_holdout_is_deterministic_given_seed() -> None:
    """Same inputs and seed give the same held-out projection."""
    from reap.baselines import run_parametric_umap_holdout

    X = _make_synthetic(n_samples=100, n_features=32, n_centers=4)
    kwargs = dict(n_components=2, n_neighbors=10, min_dist=0.1, random_state=7, n_epochs=10)
    a = run_parametric_umap_holdout(X[:80], X[80:], **kwargs)  # type: ignore[arg-type]
    b = run_parametric_umap_holdout(X[:80], X[80:], **kwargs)  # type: ignore[arg-type]
    # TF CPU kernels are not guaranteed bit-identical between runs; 1e-4 matches the
    # tolerance of the fit_transform determinism test above.
    np.testing.assert_allclose(a.embedding_out, b.embedding_out, rtol=1e-4, atol=1e-4)


def test_parametric_umap_holdout_rejects_feature_mismatch() -> None:
    """Mismatched feature counts fail before any fitting."""
    from reap.baselines import run_parametric_umap_holdout

    X = _make_synthetic(n_samples=100, n_features=32, n_centers=4)
    with pytest.raises(ValueError, match="features"):
        run_parametric_umap_holdout(
            X[:80], X[80:, :16], n_components=2, n_neighbors=10, min_dist=0.1
        )


def test_holdout_harness_runs_parametric_umap_once_per_seed() -> None:
    """The hold-out harness trains parametric UMAP per seed and scores it on unseen rows."""
    from reap.holdout import ParametricUMAPSettings, run_holdout_comparison

    blobs = make_blobs(n_samples=150, n_features=32, centers=3, cluster_std=0.5, random_state=4)
    X = np.asarray(blobs[0])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.asarray(blobs[1])
    Y_in = np.eye(3)[labels[:110]][:, :2] * 10.0
    comparison, projections = run_holdout_comparison(
        X[:110], Y_in, labels[:110], X[110:], labels[110:],
        training_seeds=[3, 4], run_id="synthetic/pumap",
        methods=["parametric_umap", "identity"],
        umap_settings=ParametricUMAPSettings(
            n_components=2, n_neighbors=10, min_dist=0.1, n_epochs=5
        ),
    )
    assert len(comparison.get_metric_column("parametric_umap", "trustworthiness")) == 2
    assert projections["parametric_umap"].Y_in_space.shape == (110, 2)
    assert projections["parametric_umap"].Y_out.shape == (40, 2)
