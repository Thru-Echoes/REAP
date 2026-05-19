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
