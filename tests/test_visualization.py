"""Tests for the PCA-based visualization module.

Tests cover: PCA fitting, 2D transformation, save/load round-trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

from reap.visualization import (
    fit_pca_visualization,
    load_visualization_reducer,
    save_visualization_reducer,
    transform_to_2d,
)


@pytest.fixture
def consensus_embedding() -> np.ndarray:
    """Synthetic consensus embedding: 50 samples, 10 dimensions."""
    return np.random.default_rng(42).standard_normal((50, 10))


class TestFitPcaVisualization:
    """Tests for fitting the PCA visualization reducer."""

    def test_returns_pca_and_metadata(self, consensus_embedding: np.ndarray) -> None:
        pca, metadata = fit_pca_visualization(consensus_embedding)
        assert isinstance(pca, PCA)
        assert isinstance(metadata, dict)

    def test_metadata_has_variance_info(self, consensus_embedding: np.ndarray) -> None:
        pca, metadata = fit_pca_visualization(consensus_embedding)
        assert "pc1_variance_ratio" in metadata
        assert "pc2_variance_ratio" in metadata
        assert "total_variance_ratio" in metadata
        assert 0.0 < metadata["total_variance_ratio"] <= 1.0

    def test_custom_components(self, consensus_embedding: np.ndarray) -> None:
        pca, metadata = fit_pca_visualization(consensus_embedding, n_components=3)
        assert pca.n_components == 3
        assert "pc3_variance_ratio" in metadata


class TestTransformTo2d:
    """Tests for 2D projection."""

    def test_output_shape(self, consensus_embedding: np.ndarray) -> None:
        pca, _ = fit_pca_visualization(consensus_embedding)
        result = transform_to_2d(pca, consensus_embedding)
        assert result.shape == (50, 2)

    def test_deterministic(self, consensus_embedding: np.ndarray) -> None:
        pca, _ = fit_pca_visualization(consensus_embedding)
        r1 = transform_to_2d(pca, consensus_embedding)
        r2 = transform_to_2d(pca, consensus_embedding)
        np.testing.assert_array_equal(r1, r2)

    def test_additive_projection(self, consensus_embedding: np.ndarray) -> None:
        """New data doesn't change existing point positions."""
        pca, _ = fit_pca_visualization(consensus_embedding)
        original_2d = transform_to_2d(pca, consensus_embedding[:30])

        # Adding new data through the same PCA
        combined_2d = transform_to_2d(pca, consensus_embedding)
        np.testing.assert_array_almost_equal(original_2d, combined_2d[:30])


class TestSaveLoadReducer:
    """Tests for PCA serialization."""

    def test_round_trip(self, consensus_embedding: np.ndarray) -> None:
        pca, _ = fit_pca_visualization(consensus_embedding)
        before = transform_to_2d(pca, consensus_embedding)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        save_visualization_reducer(pca, path)
        loaded = load_visualization_reducer(path)
        after = transform_to_2d(loaded, consensus_embedding)

        np.testing.assert_array_almost_equal(before, after)
        Path(path).unlink()

    def test_load_wrong_type_raises(self) -> None:
        import pickle

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a pca"}, f)
            path = f.name

        with pytest.raises(TypeError, match="Expected PCA"):
            load_visualization_reducer(path)
        Path(path).unlink()
