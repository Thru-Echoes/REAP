"""Parametric UMAP baseline for REAP.

Wraps Sainburg et al. 2021's parametric UMAP — UMAP's neural-network
variant in which a small encoder MLP learns to reproduce UMAP's
non-linear embedding, enabling deterministic out-of-sample projection.

Why this baseline matters
-------------------------
Parametric UMAP is the **direct prior art** for REAP's projection-head
contribution: "single-seed UMAP + a trained NN that can project new
data into the same space." A reviewer evaluating REAP's projection-head
claim will look for this head-to-head comparison first. REAP differs by
training the head against a *consensus* embedding (averaged across 30
seeds via distance-matrix consensus) rather than a single stochastic
UMAP run — so REAP's projection head inherits the consensus stability
that Parametric UMAP cannot, by construction, have.

Design contract — match the other consensus baselines (see
:mod:`reap.baselines.bertopic_baseline`):

1. **Same input.** Accepts the pre-computed embeddings the rest of the
   harness uses (``np.ndarray`` of shape ``(n_samples, d_in)``). No
   re-embedding.
2. **Same output shape.** Returns a frozen :class:`BaselineResult`
   carrying ``embedding`` (the fitted ParametricUMAP projection in the
   target ``n_components`` space), ``labels`` (KMeans applied at the
   matched ``k`` — the harness wires this up), and ``centroids``. Because
   ParametricUMAP does not produce its own labels (unlike BERTopic /
   HDBSCAN), the harness's KMeans pass is what labels documents.
3. **Single-seed by nature.** The internal optimization is deterministic
   given a fixed ``random_state`` (Sainburg 2021 §3). Parameterising
   ``random_state`` with the per-set first seed makes the cross-set
   between-set ARI honestly reflect Parametric UMAP's seed sensitivity.

Side effects
------------
- TensorFlow Keras model fitting (silent unless ``verbose=True`` is set
  in the constructor; we keep it silent for benchmark runs).
- Random state in TF / Keras / numpy / Python.

References
----------
Sainburg, T., McInnes, L., & Gentner, T. Q. (2021). Parametric UMAP
embeddings for representation and semisupervised learning. Neural
Computation 33(11): 2881-2907.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — kept here so the paper's pre-registration is auditable.
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 64
DEFAULT_N_EPOCHS: int = 50  # Sainburg 2021 §4 reports plateau at ~50 epochs
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ParametricUMAPResult(BaseModel):
    """Output of a Parametric UMAP fit.

    Mirrors :class:`reap.baselines.BaselineResult` so the harness can
    consume it interchangeably. ``labels`` is left as ``None`` here
    because Parametric UMAP does not produce labels — the benchmark
    harness applies KMeans to the returned ``embedding`` at the matched
    cluster count, exactly as it does for ``naive_average`` / ``reap``.

    Side effects: arrays held as plain ``np.ndarray`` via
    ``arbitrary_types_allowed`` (same convention as
    :class:`reap.baselines.BaselineResult` and
    :class:`reap.benchmarks.BenchmarkArtifacts`).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: str = Field(..., description="Baseline identifier — always 'parametric_umap'")
    embedding: np.ndarray = Field(
        ...,
        description=(
            "(n_samples, n_components) low-dim embedding from the fitted "
            "Parametric UMAP encoder"
        ),
    )
    random_state: int = Field(
        ...,
        description="The random_state used for the fit (per-set first seed; reproducibility hook)",
    )


# ---------------------------------------------------------------------------
# Public baseline runner
# ---------------------------------------------------------------------------


def run_parametric_umap_baseline(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
    random_state: int = DEFAULT_RANDOM_STATE,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ParametricUMAPResult:
    """Fit Sainburg et al. 2021 Parametric UMAP on pre-computed embeddings.

    Trains a small Keras encoder MLP whose target is the same fuzzy
    simplicial-set objective UMAP optimizes. UMAP hyperparameters
    ``n_components``, ``n_neighbors``, ``min_dist``, ``metric`` are passed
    through unchanged so REAP and Parametric UMAP share input-space
    geometry — only the consensus / projection step differs.

    Pure-ish: TF / Keras internal random state; no on-disk side effects.

    Parameters
    ----------
    embeddings : (n_samples, d_in) document embeddings (e.g.,
        e5-large-v2, MiniLM). Cast to float32 internally because
        TF / Keras prefers that dtype.
    n_components, n_neighbors, min_dist : UMAP hyperparameters, matched
        to REAP's run for fair comparison.
    metric : Input-space distance metric (default ``"cosine"``).
    random_state : Seed for TF / numpy / Python RNGs.
    n_epochs, batch_size : Training schedule. Defaults follow Sainburg
        2021 §4.

    Returns
    -------
    :class:`ParametricUMAPResult` with the fitted embedding and the
    ``random_state`` used.

    Raises
    ------
    ImportError
        If ``tensorflow`` or the ``umap.parametric_umap`` submodule is
        not installed. Install with
        ``pip install reap-topics[parametric_umap]``.
    ValueError
        If ``embeddings`` is not 2-d.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-d (n_samples, d_in), got shape {embeddings.shape}"
        )

    try:
        from umap.parametric_umap import ParametricUMAP  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Parametric UMAP baseline requires umap.parametric_umap (+tensorflow). "
            "Install with: pip install 'reap-topics[parametric_umap]'"
        ) from exc

    # TF random_state — set BEFORE constructing ParametricUMAP so weight
    # initialisation honors the seed.
    _seed_tensorflow(random_state)

    model: Any = ParametricUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=random_state,
        verbose=False,
    )

    embeddings_f32 = np.asarray(embeddings, dtype=np.float32)
    embedding_out: np.ndarray = model.fit_transform(embeddings_f32)
    embedding_out = np.asarray(embedding_out, dtype=np.float64)

    logger.info(
        "Parametric UMAP baseline: n=%d, d_in=%d -> d_out=%d, random_state=%d, epochs=%d",
        embeddings.shape[0],
        embeddings.shape[1],
        n_components,
        random_state,
        n_epochs,
    )

    return ParametricUMAPResult(
        method="parametric_umap",
        embedding=embedding_out,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _seed_tensorflow(seed: int) -> None:
    """Seed TF / Keras / numpy / Python RNGs so the Parametric UMAP fit is reproducible.

    Side effects: sets random.seed, np.random.seed, tf.random.set_seed.
    """
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore[import-untyped]

        tf.random.set_seed(seed)
        # Deterministic ops (TF >=2.10): improves repeatability at the
        # cost of some speed; acceptable for a benchmark run.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass
