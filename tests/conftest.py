"""Shared test fixtures for REAP test suite.

Uses synthetic data with known structure for deterministic, fast tests.
"""

import platform

import numpy as np
import pytest
from sklearn.datasets import make_blobs


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip ``reference_platform``-marked tests off the Linux reference environment.

    Tests marked ``reference_platform`` assert pre-registered metric ranges
    (``manuscript/evaluation_protocol.md`` §13) calibrated on the Linux reference
    environment used by the dedicated golden-validation job. The
    sentence-transformers / UMAP / torch float behavior on macOS diverges enough to
    move these metrics below the pre-registered floors, so they are asserted on
    Linux only; macOS still runs every other test. Side effect: skips the item.
    """
    if item.get_closest_marker("reference_platform") and platform.system() != "Linux":
        pytest.skip(
            "pre-registered metric range (evaluation_protocol §13) — calibrated on the "
            f"Linux reference environment; ML-stack float behavior diverges on "
            f"{platform.system()}, so it is asserted on Linux only"
        )


@pytest.fixture(scope="session")
def synthetic_blobs() -> dict:
    """Well-separated blobs in 384-d with known ground truth."""
    X, y = make_blobs(
        n_samples=200,
        n_features=384,
        centers=5,
        cluster_std=1.0,
        random_state=42,
    )
    # L2 normalize (like sentence embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return {"X": X, "labels": y, "n_clusters": 5, "n_samples": 200, "n_features": 384}


@pytest.fixture(scope="session")
def small_blobs() -> dict:
    """Tiny blobs for fast unit tests (50 samples, 32-d)."""
    X, y = make_blobs(
        n_samples=50,
        n_features=32,
        centers=3,
        cluster_std=0.5,
        random_state=42,
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return {"X": X, "labels": y, "n_clusters": 3, "n_samples": 50, "n_features": 32}


@pytest.fixture(scope="session")
def three_seeds() -> list[int]:
    """Minimal seed list for fast consensus tests."""
    return [42, 137, 256]


@pytest.fixture(scope="session")
def umap_embeddings_3d(small_blobs: dict, three_seeds: list[int]) -> list[np.ndarray]:
    """Pre-computed 3-seed UMAP embeddings for consensus tests."""
    from reap.consensus import get_multi_seed_embeddings

    return get_multi_seed_embeddings(
        small_blobs["X"],
        seeds=three_seeds,
        n_components=3,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
    )
