"""BERTopic baseline for REAP.

Wraps BERTopic's default pipeline (UMAP → HDBSCAN → c-TF-IDF) so the
methods paper can compare REAP head-to-head against the de-facto
topic-modeling baseline (pre-registered as method #5 in
``manuscript/evaluation_protocol.md`` §3).

Design contract — match REAP's other consensus methods:

1. **Same input.** Accepts the *same* pre-computed embeddings REAP
   receives (``np.ndarray`` of shape ``(n_samples, d_in)``). BERTopic is
   *not* allowed to re-embed text internally — the fair comparison
   requires identical inputs across methods.
2. **Same output shape.** Returns a frozen :class:`BaselineResult`
   carrying ``embedding`` (the fitted UMAP projection), ``labels``
   (per-document cluster index), and ``centroids`` (one centroid per
   non-noise cluster, computed from the BERTopic UMAP embedding). The
   benchmarks harness in :mod:`reap.benchmarks` then clusters that
   embedding with KMeans at the matched K so cross-method paired tests
   stay seed-aligned.
3. **Noise handling.** HDBSCAN's noise label ``-1`` is preserved as-is
   in ``labels`` (consumers can choose to drop or re-cluster). Centroids
   are computed only over non-noise clusters.

Side effects: UMAP / HDBSCAN internal random state, logging at INFO
level on cluster counts, and (transitively) BERTopic's own logging.

References
----------
Grootendorst, M. (2022). BERTopic: Neural topic modeling with a
class-based TF-IDF procedure. arXiv:2203.05794.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # Avoid a hard import at type-check time; runtime guard below handles missing deps.
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BERTopic default hyperparameters — kept here as named constants so the
# paper's pre-registration is auditable. Sourced from BERTopic v0.17.x
# library defaults plus the values used in REAP's existing internal call
# site in benchmarks.py (n_components / n_neighbors / min_dist are passed
# in by the caller so REAP and BERTopic share UMAP geometry).
# ---------------------------------------------------------------------------

DEFAULT_MIN_CLUSTER_SIZE: int = 15  # BERTopic / HDBSCAN library default

DEFAULT_RANDOM_STATE: int = 42  # Aligns with REAP's other deterministic-seed code


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class BaselineResult(BaseModel):
    """Output of a baseline method run.

    Mirrors the (embedding, labels, centroids) trio that REAP's internal
    methods produce after clustering: ``embedding`` comes from the
    method's dimensionality-reduction step, ``labels`` from its
    clustering step, and ``centroids`` from per-cluster averaging of the
    embedding rows.

    All arrays are stored as plain ``np.ndarray`` (Pydantic
    ``arbitrary_types_allowed``) — same convention as
    :class:`reap.benchmarks.BenchmarkArtifacts`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: str = Field(..., description="Baseline identifier (e.g., 'bertopic')")
    embedding: np.ndarray = Field(
        ...,
        description="(n_samples, n_components) low-dim embedding produced by the baseline",
    )
    labels: np.ndarray = Field(
        ...,
        description=(
            "(n_samples,) integer per-document cluster index. HDBSCAN noise points "
            "carry the label -1; non-noise clusters are contiguous non-negative ints."
        ),
    )
    centroids: np.ndarray = Field(
        ...,
        description=(
            "(n_clusters, n_components) per-cluster mean of the embedding, computed "
            "over non-noise points only. Empty (n_clusters=0) when every point was noise."
        ),
    )
    n_noise: int = Field(
        ...,
        ge=0,
        description="Number of documents assigned to HDBSCAN's noise cluster (-1)",
    )
    n_clusters: int = Field(
        ...,
        ge=0,
        description="Number of non-noise clusters discovered",
    )


# ---------------------------------------------------------------------------
# Public baseline runner
# ---------------------------------------------------------------------------


def run_bertopic_baseline(
    embeddings: np.ndarray,
    texts: list[str] | None,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> BaselineResult:
    """Run BERTopic's default pipeline on pre-computed embeddings.

    Fits a BERTopic model (UMAP → HDBSCAN → c-TF-IDF) using the supplied
    embeddings *as is*; no re-embedding occurs inside BERTopic. The UMAP
    hyperparameters are passed through so REAP and BERTopic share
    geometry — only the consensus / clustering step differs.

    Pure-ish: BERTopic mutates its own internal state but the function
    has no on-disk side effects.

    Parameters
    ----------
    embeddings : (n_samples, d_in) document embeddings (e.g., e5-large-v2,
        MiniLM). Must be 2-d and finite. Cast to float32 internally
        because BERTopic/HDBSCAN prefer that dtype.
    texts : Optional list of per-document strings, length ``n_samples``.
        BERTopic requires *some* document list to fit; when ``texts`` is
        ``None`` we pass placeholder strings (``"doc_{i}"``) so the model
        can run, but downstream c-TF-IDF labels will be meaningless.
        Callers who care about the topic labels should always supply
        real texts.
    n_components, n_neighbors, min_dist : UMAP hyperparameters, matched
        to REAP's run for fair comparison.
    metric : Input-space distance metric for UMAP (default ``"cosine"``).
    min_cluster_size : HDBSCAN ``min_cluster_size`` (default 15, the
        BERTopic library default).
    random_state : UMAP seed for reproducibility.

    Returns
    -------
    :class:`BaselineResult` with the fitted embedding, per-document
    cluster labels (HDBSCAN ``-1`` denotes noise), and per-cluster
    centroids in the embedding space.

    Raises
    ------
    ImportError
        If ``bertopic`` or ``hdbscan`` are not installed. Install with
        ``pip install reap-topics[baselines]``.
    ValueError
        If ``embeddings`` is not 2-d or ``texts`` length disagrees with
        ``embeddings.shape[0]``.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-d (n_samples, d_in), got shape {embeddings.shape}"
        )
    n_samples = embeddings.shape[0]
    if texts is not None and len(texts) != n_samples:
        raise ValueError(
            f"texts length {len(texts)} != embeddings.shape[0] {n_samples}"
        )

    try:
        from bertopic import BERTopic  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "BERTopic baseline requires bertopic. "
            "Install with: pip install 'reap-topics[baselines]'"
        ) from exc

    try:
        from hdbscan import HDBSCAN  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "BERTopic baseline requires hdbscan. "
            "Install with: pip install 'reap-topics[baselines]'"
        ) from exc

    from umap import UMAP

    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        prediction_data=True,
    )

    # BERTopic requires a document list. When the caller didn't supply
    # one, hand it placeholders — the topic labels will be useless but
    # the clustering geometry (the only thing we report here) is fine.
    docs: list[str] = (
        texts if texts is not None else [f"doc_{i}" for i in range(n_samples)]
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=False,
    )

    embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
    topics_list: list[int]
    topics_list, _ = topic_model.fit_transform(docs, embeddings=embeddings_f32)
    labels = np.asarray(topics_list, dtype=np.int64)

    # UMAP embedding is exposed on the fitted UMAP sub-model. Cast to
    # float64 to match the convention in reap.consensus.get_consensus_embedding.
    fitted_umap: Any = topic_model.umap_model
    bertopic_embedding = np.asarray(fitted_umap.embedding_, dtype=np.float64)

    centroids, n_clusters = _compute_centroids(bertopic_embedding, labels)
    n_noise = int(np.sum(labels == -1))

    logger.info(
        "BERTopic baseline: %d non-noise clusters, %d noise points (%.1f%%) "
        "[n=%d, d_in=%d, d_out=%d]",
        n_clusters,
        n_noise,
        100.0 * n_noise / n_samples if n_samples > 0 else 0.0,
        n_samples,
        embeddings.shape[1],
        n_components,
    )

    return BaselineResult(
        method="bertopic",
        embedding=bertopic_embedding,
        labels=labels,
        centroids=centroids,
        n_noise=n_noise,
        n_clusters=n_clusters,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_centroids(
    embedding: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compute per-cluster centroids in the embedding space, excluding noise.

    Pure function.

    Parameters
    ----------
    embedding : (n_samples, n_components) low-dim embedding.
    labels : (n_samples,) integer cluster index. HDBSCAN noise is ``-1``.

    Returns
    -------
    centroids : (n_clusters, n_components) per-cluster mean, ordered by
        ascending non-noise cluster id. When every point is noise, returns
        a ``(0, n_components)`` array.
    n_clusters : Number of non-noise clusters (== ``centroids.shape[0]``).
    """
    unique_ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    n_components = embedding.shape[1]
    if not unique_ids:
        return np.empty((0, n_components), dtype=embedding.dtype), 0

    centroids = np.stack(
        [embedding[labels == cid].mean(axis=0) for cid in unique_ids]
    )
    return centroids.astype(embedding.dtype, copy=False), len(unique_ids)
