"""Unit tests for the BERTopic baseline wrapper (src/reap/baselines/bertopic.py).

Verifies the (embedding, labels, centroids) contract on tiny synthetic
blobs. Real BERTopic on n=80, d=32 is fast (<10 s on Apple Silicon) but
not instant — these tests stay inside the unit-test envelope of the
existing suite (50–200 samples, 32-d) and skip cleanly when the
``baselines`` extra is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

pytest.importorskip("bertopic")
pytest.importorskip("hdbscan")


def _make_synthetic(
    n_samples: int = 80, n_features: int = 32, n_centers: int = 4
) -> tuple[np.ndarray, list[str]]:
    """Synthetic blobs + per-document text placeholders.

    Texts encode the ground-truth cluster id so c-TF-IDF inside BERTopic
    has meaningful tokens to work with. Returned embeddings are
    L2-normalized to mimic sentence-encoder output.
    """
    X, y = make_blobs(  # type: ignore[misc]
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=0.5,
        random_state=42,
        return_centers=False,
    )
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    texts = [
        f"document about topic {int(y[i])} word_{int(y[i])} content here"
        for i in range(n_samples)
    ]
    return X, texts


def test_bertopic_baseline_shape_and_no_nan() -> None:
    """Embedding/labels/centroids must have the right shapes and be finite."""
    from reap.baselines import BaselineResult, run_bertopic_baseline

    X, texts = _make_synthetic(n_samples=80, n_features=32, n_centers=4)

    result = run_bertopic_baseline(
        embeddings=X,
        texts=texts,
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        min_cluster_size=5,
    )

    assert isinstance(result, BaselineResult)
    assert result.method == "bertopic"

    # Embedding: (n_samples, n_components)
    assert result.embedding.shape == (80, 2)
    assert np.all(np.isfinite(result.embedding)), "Embedding contains NaN/inf"

    # Labels: (n_samples,), integer, no NaN
    assert result.labels.shape == (80,)
    assert result.labels.dtype.kind in ("i", "u")
    # All non-noise labels are >= 0
    assert (result.labels >= -1).all()

    # Centroids: (n_clusters, n_components), one row per non-noise cluster
    assert result.centroids.ndim == 2
    assert result.centroids.shape[1] == 2
    assert result.centroids.shape[0] == result.n_clusters
    if result.n_clusters > 0:
        assert np.all(np.isfinite(result.centroids)), "Centroids contain NaN/inf"

    # Noise count consistency
    assert result.n_noise == int(np.sum(result.labels == -1))
    assert result.n_noise + sum(
        int(np.sum(result.labels == c)) for c in range(result.n_clusters)
    ) == 80


def test_bertopic_baseline_sensible_cluster_count() -> None:
    """On well-separated blobs HDBSCAN should find ≥2 clusters (real signal)."""
    from reap.baselines import run_bertopic_baseline

    X, texts = _make_synthetic(n_samples=100, n_features=32, n_centers=4)

    result = run_bertopic_baseline(
        embeddings=X,
        texts=texts,
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        min_cluster_size=5,
    )

    # Well-separated 4-blob data should produce ≥2 clusters (not all noise);
    # not pinning to exactly 4 because HDBSCAN's behaviour at n=100 is sensitive
    # to min_cluster_size and we want a stable test.
    assert result.n_clusters >= 2, (
        f"Expected ≥2 clusters on 4-blob synthetic, got {result.n_clusters} "
        f"(n_noise={result.n_noise})"
    )
    # Most points should be assigned (allow up to 30% noise on tiny synthetic).
    assert result.n_noise < 0.30 * X.shape[0], (
        f"Too many noise points: {result.n_noise}/{X.shape[0]}"
    )
