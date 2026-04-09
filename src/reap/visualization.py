"""PCA-based 2D visualization for REAP consensus embeddings.

Uses PCA (not UMAP) for the final visualization step because:
1. The consensus space has strong linear structure (PCA Sil=0.645 vs UMAP Sil=-0.018)
2. PCA is deterministic — no random seed needed
3. PCA.transform() is additive — new data never changes existing point positions
4. PCA explains ~67.7% variance in first 2 PCs for typical consensus spaces

The visualization reducer should be fit once on reference data and reused
via transform() for all subsequent projections.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def fit_pca_visualization(
    X_consensus: np.ndarray,
    n_components: int = 2,
) -> tuple[PCA, dict[str, float]]:
    """Fit a PCA visualization reducer on reference consensus embeddings.

    Parameters
    ----------
    X_consensus : (n_samples, d) reference consensus UMAP coordinates.
    n_components : Number of visualization dimensions (default: 2).

    Returns
    -------
    (fitted_pca, metadata) where metadata includes explained variance info.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_consensus)

    explained = pca.explained_variance_ratio_
    metadata = {
        f"pc{i+1}_variance_ratio": float(v) for i, v in enumerate(explained)
    }
    metadata["total_variance_ratio"] = float(sum(explained))

    logger.info(
        "PCA fit: %d components, %.1f%% variance explained",
        n_components,
        metadata["total_variance_ratio"] * 100,
    )
    return pca, metadata


def transform_to_2d(
    pca: PCA,
    X_consensus: np.ndarray,
) -> np.ndarray:
    """Project consensus coordinates to 2D using a pre-fitted PCA.

    Uses transform() (NOT fit_transform()) to ensure additive projection:
    reference points keep their positions when new data is added.

    Parameters
    ----------
    pca : Pre-fitted PCA from fit_pca_visualization().
    X_consensus : (n_samples, d) consensus UMAP coordinates.

    Returns
    -------
    (n_samples, 2) visualization coordinates.
    """
    return pca.transform(X_consensus)


def save_visualization_reducer(pca: PCA, path: str | Path) -> None:
    """Save the fitted PCA reducer to disk for reuse.

    Parameters
    ----------
    pca : Fitted PCA object.
    path : Output file path (typically .pkl).
    """
    with open(path, "wb") as f:
        pickle.dump(pca, f)
    logger.info("Visualization reducer saved to %s", path)


def load_visualization_reducer(path: str | Path) -> PCA:
    """Load a previously fitted PCA reducer.

    Parameters
    ----------
    path : Path to saved PCA pickle file.

    Returns
    -------
    Fitted PCA object ready for transform().
    """
    with open(path, "rb") as f:
        pca = pickle.load(f)  # noqa: S301
    if not isinstance(pca, PCA):
        raise TypeError(f"Expected PCA, got {type(pca).__name__}")
    return pca
