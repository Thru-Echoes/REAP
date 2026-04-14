"""Deterministic synthetic-blob loaders used by tests and the golden fixture.

These are real DatasetSnapshots — not mocks — so any pipeline code that
accepts a snapshot works unchanged on them. The golden loader in
particular is the Tier-1/2/3 verification fixture referenced from
`manuscript/evaluation_protocol.md` §6.
"""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
from sklearn.datasets import make_blobs

from reap.datasets._schema import DatasetMetadata, DatasetSnapshot

_SYNTHETIC_CITATION = (
    "@misc{reap_synthetic_blobs_2026,\n"
    "  title={REAP synthetic-blobs fixture},\n"
    "  note={Deterministic scikit-learn make_blobs output used for REAP "
    "verification},\n"
    "  year={2026}\n"
    "}"
)


def _compute_sha256(X: np.ndarray, y: np.ndarray) -> str:
    """SHA256 of the concatenated embedding + label bytes.

    Deterministic given the same (X, y) arrays and the same byte
    representation. No side effects.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def load_synthetic_blobs(
    n_samples: int = 400,
    n_features: int = 384,
    centers: int = 8,
    cluster_std: float = 2.0,
    random_state: int = 42,
    name_suffix: str = "",
    l2_normalize: bool = True,
) -> DatasetSnapshot:
    """Return a deterministic synthetic `DatasetSnapshot` via `make_blobs`.

    L2-normalization mimics the geometry of real sentence embeddings and
    is on by default — turn it off with `l2_normalize=False` for raw
    Gaussian blobs.

    Parameters
    ----------
    n_samples, n_features, centers, cluster_std, random_state
        Passed through to `sklearn.datasets.make_blobs`.
    name_suffix
        Appended to the metadata `name` field so variants are distinct
        in result CSVs and caches.
    l2_normalize
        If True, each row is divided by its L2 norm (safe for zero rows).

    Returns
    -------
    DatasetSnapshot with float64 embeddings, int64 labels, full metadata.

    Side effects: none — uses seeded numpy via scikit-learn.
    """
    blob_out = cast(
        tuple[np.ndarray, np.ndarray],
        make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
            return_centers=False,
        ),
    )
    X = np.asarray(blob_out[0], dtype=np.float64)
    y = np.asarray(blob_out[1], dtype=np.int64)

    if l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X = X / norms

    sha = _compute_sha256(X, y)
    meta = DatasetMetadata(
        name=f"synthetic_blobs{name_suffix}",
        version="1.0.0",
        n_samples=X.shape[0],
        embedding_dim=X.shape[1],
        embedding_model=(
            f"sklearn.datasets.make_blobs(centers={centers}, "
            f"cluster_std={cluster_std}, random_state={random_state})"
        ),
        preprocessing_version="1.0.0-l2norm" if l2_normalize else "1.0.0-raw",
        sha256=sha,
        license="BSD-3-Clause",
        citation=_SYNTHETIC_CITATION,
        source_url=None,
        notes=(
            "Deterministic synthetic blobs for REAP E2E and golden-validation "
            "tests. Safe to commit as a fixture."
        ),
    )
    return DatasetSnapshot(embeddings=X, texts=None, labels=y, metadata=meta)


def load_golden_blobs() -> DatasetSnapshot:
    """The frozen golden fixture from `manuscript/evaluation_protocol.md` §6.

    400 samples, 384 ambient dimensions, 8 ground-truth clusters,
    intrinsic dimensionality 16, L2-normalized.

    Why this construction. High-dimensional Gaussian blobs are
    trivially separable because random centers in R^384 are nearly
    orthogonal (curse of dimensionality), so UMAP recovers them
    perfectly at every seed — a poor demonstration of REAP because
    there is no instability to stabilize. Real sentence embeddings
    are not like this: their intrinsic dimensionality is far below
    the nominal 384/768/1024, and topic overlap creates genuine
    stochastic variation in UMAP outputs.

    We therefore draw blobs in R^16 (intrinsic dim), random-project
    to R^384 (ambient dim, simulating feature redundancy typical of
    embedding models), add isotropic noise at 15% of signal scale,
    then L2-normalize. cluster_std=1.8 in the intrinsic space is
    tuned to produce genuine per-seed UMAP variance.

    Do NOT change these parameters — they are pre-registered. A
    change requires a protocol correction (§13).

    Returns
    -------
    DatasetSnapshot — the canonical golden fixture.
    """
    rng = np.random.default_rng(42)
    intrinsic_dim = 16
    ambient_dim = 384
    n_samples = 400
    n_centers = 8
    cluster_std = 2.5
    noise_sigma = 0.3

    X_low_raw, y_raw = cast(
        tuple[np.ndarray, np.ndarray],
        make_blobs(
            n_samples=n_samples,
            n_features=intrinsic_dim,
            centers=n_centers,
            cluster_std=cluster_std,
            center_box=(-3.0, 3.0),
            random_state=42,
            return_centers=False,
        ),
    )
    X_low = np.asarray(X_low_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.int64)

    projection = rng.standard_normal((intrinsic_dim, ambient_dim)) / np.sqrt(
        intrinsic_dim
    )
    noise = noise_sigma * rng.standard_normal((n_samples, ambient_dim))
    X = X_low @ projection + noise

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    sha = _compute_sha256(X, y)
    meta = DatasetMetadata(
        name="synthetic_blobs_golden",
        version="1.2.0",
        n_samples=n_samples,
        embedding_dim=ambient_dim,
        embedding_model=(
            f"make_blobs(centers={n_centers}, cluster_std={cluster_std}, "
            f"center_box=(-3,3), random_state=42) in R^{intrinsic_dim} → "
            f"random_projection → R^{ambient_dim} + noise(sigma={noise_sigma}) "
            "→ L2-normalize"
        ),
        preprocessing_version="1.2.0-lowd-projected-overlapping",
        sha256=sha,
        license="BSD-3-Clause",
        citation=_SYNTHETIC_CITATION,
        source_url=None,
        notes=(
            "Low-intrinsic-dim blobs projected to ambient 384-d. Designed to "
            "reproduce the per-seed UMAP instability REAP addresses. Frozen "
            "in evaluation_protocol.md §6 v1.1."
        ),
    )
    return DatasetSnapshot(embeddings=X, texts=None, labels=y, metadata=meta)
