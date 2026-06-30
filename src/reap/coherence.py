"""Topic coherence — gensim-backed UMass / NPMI / c_v on REAP clusters.

A high-level convenience layer over the three pre-registered topic
coherence metrics (``manuscript/evaluation_protocol.md`` §7). Each
metric measures how internally consistent a cluster's top discriminative
terms are with respect to the reference corpus:

* **UMass** (Mimno et al. 2011) — log of co-document frequency. Negative
  values; closer to 0 is better.
* **NPMI** (Bouma 2009; Lau et al. 2014) — normalised pointwise mutual
  information. Bounded in [-1, 1]; higher is better.
* **c_v** (Röder et al. 2015) — sliding-window pointwise mutual
  information with cosine similarity. Bounded in [0, 1]; higher is better.

Why this module exists alongside ``reap.evaluation.compute_topic_coherence``:

* :func:`reap.evaluation.compute_topic_coherence` is the *low-level*
  primitive — it takes pre-extracted topic word-lists and returns one
  score per topic. It is wired into the benchmark harness via private
  helpers (UMass/NPMI are native; c_v dispatches to gensim).
* This module is the *high-level* convenience surface for paper-track
  work. It accepts ``labels + texts`` directly (running c-TF-IDF
  internally), exposes all three metrics through one entry point, and
  returns a frozen :class:`CoherenceResult` carrying per-topic vectors
  and aggregate means — the shape downstream reporting tables expect.

Gensim is required for **every** metric exposed here so all three
metrics use a single, well-vetted implementation (the pre-registered
choice in §7 names gensim's c_v explicitly; UMass and NPMI under the
same library avoid cross-implementation drift). ``ImportError`` is
raised with an install hint if gensim is missing.

References
----------
Mimno, D., et al. (2011). Optimizing semantic coherence in topic models.
Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction.
Lau, J. H., Newman, D., Baldwin, T. (2014). Machine reading tea leaves.
Röder, M., Both, A., Hinneburg, A. (2015). Exploring the space of topic
coherence measures.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from reap.labeling import label_clusters_ctfidf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public metric names + defaults
# ---------------------------------------------------------------------------

CoherenceMetric = Literal["u_mass", "c_npmi", "c_v"]

#: All three pre-registered coherence metrics (manuscript §7).
ALL_COHERENCE_METRICS: tuple[CoherenceMetric, ...] = ("u_mass", "c_npmi", "c_v")

#: Default number of top terms per cluster, matching benchmarks.py top_n=10.
DEFAULT_TOP_N: int = 10


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class CoherenceResult(BaseModel):
    """Topic coherence scores across all three pre-registered metrics.

    Reports per-cluster vectors (one float per non-noise cluster) and the
    arithmetic mean across clusters. ``cluster_ids`` records the order of
    clusters in the per-cluster vectors so callers can map back to the
    label space.

    All three metrics carry NaN entries for clusters whose top-N word
    lists were too short or fully out-of-vocab (gensim's behaviour for
    such degenerate inputs); aggregate means use ``numpy.nanmean`` and
    therefore ignore those clusters. ``n_clusters_scored`` reports the
    count of clusters that contributed a finite value to each aggregate.
    """

    model_config = ConfigDict(frozen=True)

    cluster_ids: list[int] = Field(
        ...,
        description="Non-noise cluster ids, ordered to match the per-cluster lists",
    )
    top_n: int = Field(
        ..., ge=1, description="Top-N term cutoff used when computing each metric"
    )

    umass_per_cluster: list[float] = Field(
        ..., description="UMass coherence per cluster (≤ 0; closer to 0 = better)"
    )
    npmi_per_cluster: list[float] = Field(
        ..., description="NPMI coherence per cluster (in [-1, 1]; higher = better)"
    )
    cv_per_cluster: list[float] = Field(
        ..., description="c_v coherence per cluster (in [0, 1]; higher = better)"
    )

    umass_mean: float = Field(
        ..., description="Arithmetic mean of finite UMass scores"
    )
    npmi_mean: float = Field(
        ..., description="Arithmetic mean of finite NPMI scores"
    )
    cv_mean: float = Field(
        ..., description="Arithmetic mean of finite c_v scores"
    )

    n_clusters_scored: dict[str, int] = Field(
        ...,
        description=(
            "Count of clusters that contributed a finite (non-NaN) value to each "
            "aggregate. Keys: 'u_mass', 'c_npmi', 'c_v'."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coherence(
    labels: np.ndarray,
    texts: list[str],
    metric: CoherenceMetric,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> list[float]:
    """Compute a single coherence metric per non-noise cluster.

    Runs c-TF-IDF internally to extract per-cluster top terms, then
    dispatches to gensim's :class:`~gensim.models.coherencemodel.CoherenceModel`
    for the chosen metric.

    Pure-ish: imports gensim + sklearn on first call; no on-disk side
    effects.

    Parameters
    ----------
    labels : (n_samples,) integer cluster index. ``-1`` is treated as
        noise (HDBSCAN convention) and skipped.
    texts : Length-``n_samples`` list of source documents. Used both as
        the c-TF-IDF input and as the reference corpus for coherence
        co-occurrence statistics.
    metric : Which coherence metric to compute. One of ``"u_mass"``,
        ``"c_npmi"``, ``"c_v"``.
    top_n : Number of top discriminative terms per cluster (default 10,
        matching the benchmark harness).

    Returns
    -------
    List of floats, one per non-noise cluster, ordered by ascending
    cluster id. Entries that gensim cannot score (no valid terms) are
    returned as ``float("nan")`` so the caller can decide how to
    aggregate.

    Raises
    ------
    ImportError
        If gensim is not installed. Install with
        ``pip install 'reap-topics[coherence]'``.
    ValueError
        If ``metric`` is not one of the three pre-registered metrics,
        ``labels`` and ``texts`` lengths disagree, or ``top_n < 1``.
    """
    _validate_inputs(labels, texts, metric, top_n)

    topics, _ = _get_topic_word_lists(labels, texts, top_n)
    if not topics:
        logger.warning(
            "compute_coherence(%s): no non-noise clusters in labels — returning empty list",
            metric,
        )
        return []

    return _score_topics_with_gensim(topics, texts, metric)


def compute_all_coherence(
    labels: np.ndarray,
    texts: list[str],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> CoherenceResult:
    """Compute all three pre-registered coherence metrics in one pass.

    Single c-TF-IDF pass shared across metrics. Returns a frozen
    :class:`CoherenceResult` with per-cluster vectors, aggregate means,
    and the count of clusters that produced finite scores for each
    metric.

    Parameters
    ----------
    labels, texts, top_n : See :func:`compute_coherence`.

    Returns
    -------
    :class:`CoherenceResult`.

    Raises
    ------
    ImportError
        If gensim is not installed.
    ValueError
        If inputs disagree on length or ``top_n < 1``.
    """
    if labels.shape[0] != len(texts):
        raise ValueError(
            f"labels length {labels.shape[0]} != texts length {len(texts)}"
        )
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    topics, cluster_ids = _get_topic_word_lists(labels, texts, top_n)
    if not topics:
        logger.warning("compute_all_coherence: no non-noise clusters — returning zeros")
        return CoherenceResult(
            cluster_ids=[],
            top_n=top_n,
            umass_per_cluster=[],
            npmi_per_cluster=[],
            cv_per_cluster=[],
            umass_mean=float("nan"),
            npmi_mean=float("nan"),
            cv_mean=float("nan"),
            n_clusters_scored={"u_mass": 0, "c_npmi": 0, "c_v": 0},
        )

    umass = _score_topics_with_gensim(topics, texts, "u_mass")
    npmi = _score_topics_with_gensim(topics, texts, "c_npmi")
    cv = _score_topics_with_gensim(topics, texts, "c_v")

    return CoherenceResult(
        cluster_ids=cluster_ids,
        top_n=top_n,
        umass_per_cluster=umass,
        npmi_per_cluster=npmi,
        cv_per_cluster=cv,
        umass_mean=_finite_mean(umass),
        npmi_mean=_finite_mean(npmi),
        cv_mean=_finite_mean(cv),
        n_clusters_scored={
            "u_mass": _count_finite(umass),
            "c_npmi": _count_finite(npmi),
            "c_v": _count_finite(cv),
        },
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    labels: np.ndarray,
    texts: list[str],
    metric: str,
    top_n: int,
) -> None:
    """Argument validation shared across public entry points."""
    if metric not in ALL_COHERENCE_METRICS:
        raise ValueError(
            f"Unknown coherence metric {metric!r}. "
            f"Choose from: {list(ALL_COHERENCE_METRICS)}"
        )
    if labels.shape[0] != len(texts):
        raise ValueError(
            f"labels length {labels.shape[0]} != texts length {len(texts)}"
        )
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")


def _get_topic_word_lists(
    labels: np.ndarray,
    texts: list[str],
    top_n: int,
) -> tuple[list[list[str]], list[int]]:
    """Extract per-cluster top terms via c-TF-IDF.

    Returns ``(topics, cluster_ids)`` where ``topics[i]`` is the top-N
    word list for cluster ``cluster_ids[i]``. Skips HDBSCAN noise
    (``-1``). Output order is the order returned by
    :func:`reap.labeling.label_clusters_ctfidf`, which sorts ascending
    by cluster id.
    """
    ctfidf_results = label_clusters_ctfidf(texts, labels, top_n=top_n)
    topics: list[list[str]] = []
    cluster_ids: list[int] = []
    for cr in ctfidf_results:
        if cr.cluster == -1:
            continue
        topics.append([term.term for term in cr.top_terms[:top_n]])
        cluster_ids.append(int(cr.cluster))
    return topics, cluster_ids


def _score_topics_with_gensim(
    topics: list[list[str]],
    texts: list[str],
    metric: str,
) -> list[float]:
    """Score topics against the reference corpus via gensim.

    Centralised so all three metrics share one Dictionary build and one
    tokenization pass — and so the ImportError message lives in one place.

    Side effects: builds a gensim Dictionary and corpus in memory.
    """
    try:
        from gensim.corpora import Dictionary  # type: ignore[import-untyped]
        from gensim.models.coherencemodel import CoherenceModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            f"Coherence metric {metric!r} requires gensim. "
            "Install with: pip install 'reap-topics[coherence]'"
        ) from exc

    tokenized = [t.split() for t in texts]
    dictionary = Dictionary(tokenized)
    cm = CoherenceModel(
        topics=topics,
        texts=tokenized,
        dictionary=dictionary,
        coherence=metric,
    )
    per_topic = cm.get_coherence_per_topic()
    return [float(v) for v in per_topic]


def _finite_mean(values: list[float]) -> float:
    """Mean over the finite entries; NaN if no finite values exist."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _count_finite(values: list[float]) -> int:
    """Number of finite (non-NaN, non-inf) entries."""
    arr = np.asarray(values, dtype=np.float64)
    return int(np.sum(np.isfinite(arr)))
