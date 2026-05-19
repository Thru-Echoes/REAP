"""Benchmark comparison of consensus methods for REAP.

Compares six strategies for combining multi-seed UMAP embeddings:

1. ``single_seed``   — independent seeds; full per-seed distribution is kept.
2. ``best_of_n``     — cherry-pick the highest-silhouette seed.
3. ``naive_average`` — coordinate average without alignment (negative control).
4. ``procrustes``    — Procrustes-align then average coordinates (baseline).
5. ``reap``          — average pairwise distance matrices, then project (ours).
6. ``bertopic``      — BERTopic's default pipeline (UMAP + HDBSCAN + c-TF-IDF).

All six share the *same* 30 multi-seed UMAP embeddings computed once per
call; the difference is only how they combine or select from those
embeddings. This ensures cross-method comparisons are fair and saves compute.

Every method emits a per-seed distribution for the quality metrics
(silhouette, trustworthiness, continuity, Davies-Bouldin, Calinski-Harabasz),
and the consensus methods additionally emit the 2 x 2 ``(label_source,
embedding_source)`` decomposition:

- **Cell A**: ``(seed labels, seed emb)``  — 30 values per method.
- **Cell B**: ``(seed labels, consensus emb)`` — 30 values, consensus methods only.
- **Cell C**: ``(consensus labels, seed emb)`` — 30 values, consensus methods only.
  This is the crucial one for paired statistical tests: it produces a
  seed-matched distribution for every consensus method, so Wilcoxon /
  Cohen's d can compare REAP vs Procrustes on matched seeds.
- **Cell D**: ``(consensus labels, consensus emb)`` — the single headline
  number.

The full 30x30 pairwise ARI / AMI / NMI matrices and the 30-length seed-to-
consensus vectors are persisted alongside the headline numbers so downstream
scripts can compute paired tests and seed-bootstrap CIs without re-running
the expensive UMAP stage.

Optional external validity (vs ``ground_truth_labels``) and topic-level
metrics (coherence / diversity / exclusivity, via ``texts``) are computed
per method when their inputs are provided.

Side effects: UMAP + KMeans internal random state, logging, ``tracemalloc``
peak-memory tracking when ``track_peak_memory=True`` (default).
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans

from reap.clustering import find_best_k
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_consensus,
)
from reap.evaluation import (
    _build_binary_doc_term_matrix,  # type: ignore[reportPrivateUsage]
    _coherence_npmi,  # type: ignore[reportPrivateUsage]
    _coherence_umass,  # type: ignore[reportPrivateUsage]
    compute_calinski_harabasz,
    compute_cluster_persistence,
    compute_continuity,
    compute_davies_bouldin,
    compute_external_validity,
    compute_pairwise_ami,
    compute_pairwise_ari,
    compute_pairwise_nmi,
    compute_silhouette,
    compute_topic_diversity,
    compute_topic_exclusivity,
    compute_trustworthiness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS: list[str] = [
    "single_seed",
    "best_of_n",
    "naive_average",
    "procrustes",
    "reap",
    "bertopic",
    "parametric_umap",
]

_METHOD_DESCRIPTIONS: dict[str, str] = {
    "single_seed": (
        "Independent single-seed UMAP runs; reports mean and std across seeds"
    ),
    "best_of_n": (
        "Cherry-pick the seed with highest silhouette score"
    ),
    "naive_average": (
        "Average coordinates across seeds without alignment (negative control)"
    ),
    "procrustes": (
        "Procrustes-align to first seed, then average coordinates"
    ),
    "reap": (
        "Average pairwise distance matrices, then project via UMAP (our method)"
    ),
    "bertopic": (
        "BERTopic default pipeline (single-seed UMAP + HDBSCAN + c-TF-IDF)"
    ),
    "parametric_umap": (
        "Sainburg 2021 parametric UMAP (Keras encoder MLP, single-seed, projectable)"
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MethodResult(BaseModel):
    """Evaluation results for a single consensus method.

    The "per_seed_*" vectors always have length ``n_seeds``. The
    ``cell_b_*`` / ``cell_c_*`` vectors are populated for methods in
    :data:`_CONSENSUS_METHODS` (including BERTopic) and ``None`` otherwise.
    """

    model_config = ConfigDict(frozen=True)

    method: str = Field(..., description="Method identifier from ALL_METHODS")
    description: str = Field(..., description="Human-readable method description")
    n_seeds: int = Field(..., ge=2, description="Number of UMAP seeds used")
    runtime_seconds: float = Field(
        ..., ge=0.0, description="Wall-clock time for this method"
    )
    peak_memory_mb: float | None = Field(
        default=None, ge=0.0, description="Peak memory delta during this method (MB)"
    )

    # ----- Headline summary (legacy-compatible single numbers) -----
    trustworthiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Headline trustworthiness (consensus for consensus methods, mean-of-seeds otherwise)",
    )
    trustworthiness_std: float = Field(
        ..., ge=0.0,
        description="Std of per-seed trustworthiness (0 for consensus headlines)",
    )
    silhouette: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Headline silhouette (Cell D for consensus methods)",
    )
    silhouette_std: float = Field(
        ..., ge=0.0,
        description="Std of per-seed silhouette (0 for consensus headlines)",
    )
    best_k: int = Field(..., ge=2, description="Selected K (consensus K for consensus methods)")
    best_k_std: float = Field(
        ..., ge=0.0, description="Std of per-seed K (0 for consensus)"
    )
    seed_to_seed_ari_mean: float | None = Field(default=None, description="Mean upper-tri ARI")
    seed_to_seed_ari_std: float | None = Field(default=None, description="Std upper-tri ARI")
    seed_to_consensus_ari_mean: float | None = Field(default=None, description="Mean s2c ARI")

    # ----- Per-seed distributions (Cell A: seed labels on seed emb) -----
    per_seed_silhouette: list[float] = Field(default_factory=list)
    per_seed_trustworthiness: list[float] = Field(default_factory=list)
    per_seed_continuity: list[float] = Field(default_factory=list)
    per_seed_davies_bouldin: list[float] = Field(default_factory=list)
    per_seed_calinski_harabasz: list[float] = Field(default_factory=list)
    per_seed_k: list[int] = Field(default_factory=list)

    # ----- 2x2 decomposition Cells B and C (consensus methods only) -----
    cell_b_silhouette: list[float] | None = Field(
        default=None,
        description="Silhouette(seed_labels_i, consensus_emb) for each seed i",
    )
    cell_c_silhouette: list[float] | None = Field(
        default=None,
        description="Silhouette(consensus_labels, seed_emb_i) — 30 values usable as paired distribution",
    )

    # ----- Consensus singletons (Cell D + singletons) -----
    consensus_silhouette: float | None = None
    consensus_trustworthiness: float | None = None
    consensus_continuity: float | None = None
    consensus_davies_bouldin: float | None = None
    consensus_calinski_harabasz: float | None = None
    consensus_k: int | None = None

    # ----- Pairwise agreement matrices (30x30) and summaries -----
    s2s_ari_matrix: list[list[float]] | None = None
    s2s_ami_matrix: list[list[float]] | None = None
    s2s_nmi_matrix: list[list[float]] | None = None
    s2s_ami_mean: float | None = None
    s2s_ami_std: float | None = None
    s2s_nmi_mean: float | None = None
    s2s_nmi_std: float | None = None

    # ----- Seed-to-consensus agreement (length-30 vectors) -----
    s2c_ari_list: list[float] | None = None
    s2c_ami_list: list[float] | None = None
    s2c_nmi_list: list[float] | None = None
    s2c_ari_std: float | None = None
    s2c_ami_mean: float | None = None
    s2c_ami_std: float | None = None
    s2c_nmi_mean: float | None = None
    s2c_nmi_std: float | None = None

    # ----- External validity (only if ground_truth_labels provided) -----
    ext_ari: float | None = None
    ext_ami: float | None = None
    ext_nmi: float | None = None
    ext_homogeneity: float | None = None
    ext_completeness: float | None = None
    ext_v_measure: float | None = None
    ext_fowlkes_mallows: float | None = None
    ext_mean_purity: float | None = None
    ext_purity_per_cluster: dict[str, float] | None = None

    # ----- Topic metrics (only if texts provided) -----
    topic_coherence_umass: float | None = None
    topic_coherence_npmi: float | None = None
    topic_coherence_cv: float | None = None
    topic_diversity: float | None = None
    topic_exclusivity: float | None = None

    # ----- Cluster persistence (only for consensus methods with seed labels) -----
    cluster_persistence_mean: float | None = None
    cluster_persistence_min: float | None = None
    cluster_persistence_median: float | None = None
    cluster_persistence_per_cluster: dict[str, float] | None = None


class BenchmarkResult(BaseModel):
    """Complete benchmark comparison across all methods for one dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Human-readable dataset identifier")
    n_samples: int = Field(..., ge=1)
    n_features: int = Field(..., ge=1)
    n_components: int = Field(..., ge=2)
    n_neighbors: int = Field(..., ge=2)
    min_dist: float = Field(..., ge=0.0, le=1.0)
    k_range: list[int] = Field(..., min_length=1)
    n_seeds: int = Field(..., ge=2)
    k_fixed: int | None = Field(
        default=None,
        description="If set, all methods use this K (matched-K mode).",
    )
    input_metric: str = Field(default="cosine")
    silhouette_metric: str = Field(default="euclidean")
    seeds: list[int] = Field(
        default_factory=list,
        description="Seed list actually used (may be empty for tests that construct BenchmarkResult by hand).",
    )
    methods: list[MethodResult] = Field(..., min_length=1)


class BootstrapCI(BaseModel):
    """Subsample-based confidence interval (data-resampling). See compute_subsample_ci."""

    model_config = ConfigDict(frozen=True)

    metric: str
    method: str
    mean: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int = Field(..., ge=1)


class BenchmarkArtifacts(BaseModel):
    """Raw numpy artifacts from a benchmark run, for downstream analysis.

    Sidecar to :class:`BenchmarkResult`. Holds the per-seed UMAP embeddings
    (shared across methods) and, per method, the consensus embedding, the
    consensus labels, and (for REAP only) the consensus distance matrix.
    Used by the between-seed-set reliability script and by statistical
    inference scripts that need the raw arrays.

    Stored as plain ``dict`` / ``list`` fields (with
    ``arbitrary_types_allowed=True``) rather than typed numpy — Pydantic's
    numpy handling is finicky, and serialization is handled via CSV / .npy
    by the caller.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    umap_embeddings: list[np.ndarray] = Field(
        ..., description="Per-seed UMAP embeddings, length n_seeds, each (N, d)"
    )
    per_method: dict[str, dict[str, Any]] = Field(
        ...,
        description=(
            "method_name -> {consensus_embedding, consensus_labels, "
            "seed_labels, consensus_distance_matrix (REAP only)}"
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers — pure-ish computations shared across methods
# ---------------------------------------------------------------------------


def _cluster_embedding(
    embedding: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster an embedding with KMeans at the given K.

    Parameters
    ----------
    embedding : (n_samples, n_components) data to cluster.
    k : Number of clusters.
    random_state : Seed for KMeans.

    Returns
    -------
    (n_samples,) integer label array.
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return np.asarray(model.fit_predict(embedding))


def _get_naive_average(
    umap_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Average coordinates across seeds without any alignment.

    Pure function. Documented as a negative control for the benchmark.
    """
    return np.mean(np.stack(umap_embeddings), axis=0)


def _select_seed_k(
    embedding: np.ndarray,
    k_range: list[int],
    k_fixed: int | None,
    sil_metric: str,
) -> int:
    """Select K for a per-seed embedding, respecting matched-K mode."""
    if k_fixed is not None:
        return k_fixed
    return int(find_best_k(embedding, k_range, metric=sil_metric).best_k)


def _compute_per_seed_cell_a(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    k_fixed: int | None,
    input_metric: str,
    sil_metric: str,
) -> dict[str, list]:
    """Cell A: per-seed metrics computed on each seed's own embedding and its own KMeans labels.

    Returns a dict of 30-length vectors:
    silhouette, trustworthiness, continuity, davies_bouldin,
    calinski_harabasz, k_per_seed, and the list of label arrays.
    """
    n_seeds = len(umap_embeddings)
    sil_list: list[float] = []
    tw_list: list[float] = []
    cont_list: list[float] = []
    db_list: list[float] = []
    ch_list: list[float] = []
    k_list: list[int] = []
    seed_labels_list: list[np.ndarray] = []

    n_nn = min(15, X_high.shape[0] - 1)
    for i, emb in enumerate(umap_embeddings):
        k = _select_seed_k(emb, k_range, k_fixed, sil_metric)
        labels = _cluster_embedding(emb, k)
        sil_list.append(compute_silhouette(emb, labels, metric=sil_metric))
        tw_list.append(compute_trustworthiness(X_high, emb, n_neighbors=n_nn, metric=input_metric))
        cont_list.append(compute_continuity(X_high, emb, n_neighbors=n_nn, metric=input_metric))
        db_list.append(compute_davies_bouldin(emb, labels))
        ch_list.append(compute_calinski_harabasz(emb, labels))
        k_list.append(int(k))
        seed_labels_list.append(labels)
        logger.debug("Cell A seed %d/%d: K=%d, sil=%.4f", i + 1, n_seeds, k, sil_list[-1])

    return {
        "silhouette": sil_list,
        "trustworthiness": tw_list,
        "continuity": cont_list,
        "davies_bouldin": db_list,
        "calinski_harabasz": ch_list,
        "k": k_list,
        "labels": seed_labels_list,
    }


def _compute_cell_b(
    consensus_emb: np.ndarray,
    seed_labels_list: list[np.ndarray],
    sil_metric: str,
) -> list[float]:
    """Cell B: silhouette of each seed's KMeans labels applied to the consensus embedding.

    Answers "does the seed's partition still look clean in the consensus space?"
    """
    values: list[float] = []
    for labels in seed_labels_list:
        if len(set(labels.tolist())) < 2:
            values.append(float("nan"))
        else:
            values.append(compute_silhouette(consensus_emb, labels, metric=sil_metric))
    return values


def _compute_cell_c(
    umap_embeddings: list[np.ndarray],
    consensus_labels: np.ndarray,
    sil_metric: str,
) -> list[float]:
    """Cell C: silhouette of the consensus labels applied to each seed's embedding.

    This is the distribution used for paired statistical tests between
    consensus methods: both REAP and Procrustes produce a 30-length vector
    here, matched by seed index.
    """
    values: list[float] = []
    for emb in umap_embeddings:
        if len(set(consensus_labels.tolist())) < 2:
            values.append(float("nan"))
        else:
            values.append(compute_silhouette(emb, consensus_labels, metric=sil_metric))
    return values


def _compute_consensus_singletons(
    X_high: np.ndarray,
    consensus_emb: np.ndarray,
    consensus_labels: np.ndarray,
    input_metric: str,
    sil_metric: str,
) -> dict[str, float]:
    """Consensus-level metrics (Cell D + trust + continuity + DB + CH)."""
    n_nn = min(15, X_high.shape[0] - 1)
    return {
        "silhouette": compute_silhouette(consensus_emb, consensus_labels, metric=sil_metric),
        "trustworthiness": compute_trustworthiness(
            X_high, consensus_emb, n_neighbors=n_nn, metric=input_metric
        ),
        "continuity": compute_continuity(
            X_high, consensus_emb, n_neighbors=n_nn, metric=input_metric
        ),
        "davies_bouldin": compute_davies_bouldin(consensus_emb, consensus_labels),
        "calinski_harabasz": compute_calinski_harabasz(consensus_emb, consensus_labels),
    }


def _compute_pairwise_agreement(
    seed_labels_list: list[np.ndarray],
    consensus_labels: np.ndarray | None,
    compute_matrices: bool,
) -> dict[str, Any]:
    """Pairwise s2s matrices (ARI, AMI, NMI) and s2c vectors.

    Returns a flat dict with keys:
      s2s_ari_matrix, s2s_ami_matrix, s2s_nmi_matrix (lists of lists or None),
      s2s_ari_mean, s2s_ari_std, s2s_ami_mean, s2s_ami_std, s2s_nmi_mean, s2s_nmi_std,
      s2c_ari_list, s2c_ami_list, s2c_nmi_list,
      s2c_ari_mean, s2c_ari_std, s2c_ami_mean, s2c_ami_std, s2c_nmi_mean, s2c_nmi_std.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

    result: dict[str, object] = {}

    # s2s — pairwise matrices
    if compute_matrices:
        ari_m = compute_pairwise_ari(seed_labels_list)
        ami_m = compute_pairwise_ami(seed_labels_list)
        nmi_m = compute_pairwise_nmi(seed_labels_list)
    else:
        ari_m = None
        ami_m = None
        nmi_m = None

    # Upper-triangle summaries (always computed)
    def _tri_stats(m: np.ndarray | None) -> tuple[float, float]:
        if m is None:
            return (0.0, 0.0)
        tri = m[np.triu_indices_from(m, k=1)]
        return (float(tri.mean()), float(tri.std()))

    # If matrices weren't stored, compute summaries directly from label list
    if ari_m is None:
        ari_m = compute_pairwise_ari(seed_labels_list)
    if ami_m is None:
        ami_m = compute_pairwise_ami(seed_labels_list)
    if nmi_m is None:
        nmi_m = compute_pairwise_nmi(seed_labels_list)

    ari_mean, ari_std = _tri_stats(ari_m)
    ami_mean, ami_std = _tri_stats(ami_m)
    nmi_mean, nmi_std = _tri_stats(nmi_m)

    result["s2s_ari_mean"] = ari_mean
    result["s2s_ari_std"] = ari_std
    result["s2s_ami_mean"] = ami_mean
    result["s2s_ami_std"] = ami_std
    result["s2s_nmi_mean"] = nmi_mean
    result["s2s_nmi_std"] = nmi_std

    if compute_matrices:
        result["s2s_ari_matrix"] = ari_m.tolist()
        result["s2s_ami_matrix"] = ami_m.tolist()
        result["s2s_nmi_matrix"] = nmi_m.tolist()
    else:
        result["s2s_ari_matrix"] = None
        result["s2s_ami_matrix"] = None
        result["s2s_nmi_matrix"] = None

    # s2c
    if consensus_labels is None:
        result["s2c_ari_list"] = None
        result["s2c_ami_list"] = None
        result["s2c_nmi_list"] = None
        result["s2c_ari_mean"] = None
        result["s2c_ari_std"] = None
        result["s2c_ami_mean"] = None
        result["s2c_ami_std"] = None
        result["s2c_nmi_mean"] = None
        result["s2c_nmi_std"] = None
        return result

    s2c_ari = [float(adjusted_rand_score(sl, consensus_labels)) for sl in seed_labels_list]
    s2c_ami = [
        float(adjusted_mutual_info_score(sl, consensus_labels))
        for sl in seed_labels_list
    ]
    s2c_nmi = [
        float(normalized_mutual_info_score(sl, consensus_labels))
        for sl in seed_labels_list
    ]
    result["s2c_ari_list"] = s2c_ari
    result["s2c_ami_list"] = s2c_ami
    result["s2c_nmi_list"] = s2c_nmi
    result["s2c_ari_mean"] = float(np.mean(s2c_ari))
    result["s2c_ari_std"] = float(np.std(s2c_ari))
    result["s2c_ami_mean"] = float(np.mean(s2c_ami))
    result["s2c_ami_std"] = float(np.std(s2c_ami))
    result["s2c_nmi_mean"] = float(np.mean(s2c_nmi))
    result["s2c_nmi_std"] = float(np.std(s2c_nmi))
    return result


def _compute_external_fields(
    ground_truth_labels: np.ndarray | None,
    consensus_labels: np.ndarray | None,
) -> dict[str, Any]:
    """External validity suite vs ground truth, or all-None if inputs missing."""
    if ground_truth_labels is None or consensus_labels is None:
        return {
            "ext_ari": None, "ext_ami": None, "ext_nmi": None,
            "ext_homogeneity": None, "ext_completeness": None,
            "ext_v_measure": None, "ext_fowlkes_mallows": None,
            "ext_mean_purity": None, "ext_purity_per_cluster": None,
        }
    suite = compute_external_validity(ground_truth_labels, consensus_labels)
    pc = suite["purity_per_cluster"]
    return {
        "ext_ari": suite["ari"],
        "ext_ami": suite["ami"],
        "ext_nmi": suite["nmi"],
        "ext_homogeneity": suite["homogeneity"],
        "ext_completeness": suite["completeness"],
        "ext_v_measure": suite["v_measure"],
        "ext_fowlkes_mallows": suite["fowlkes_mallows"],
        "ext_mean_purity": suite["mean_purity"],
        # Pydantic dict[str, float] — coerce int keys to str for JSON safety
        "ext_purity_per_cluster": (
            {str(k): float(v) for k, v in pc.items()} if isinstance(pc, dict) else None
        ),
    }


def _compute_topic_fields(
    texts: list[str] | None,
    consensus_labels: np.ndarray | None,
    top_n: int,
) -> dict[str, Any]:
    """Topic-level metrics (coherence, diversity, exclusivity).

    Requires ``texts`` and consensus labels. Computes UMass, NPMI, c_v
    (gensim; skipped with a warning if gensim is missing), plus diversity
    and exclusivity. Top-N terms come from a local c-TF-IDF pass so no
    LLM API calls are made.
    """
    if texts is None or consensus_labels is None:
        return {
            "topic_coherence_umass": None,
            "topic_coherence_npmi": None,
            "topic_coherence_cv": None,
            "topic_diversity": None,
            "topic_exclusivity": None,
        }
    from reap.labeling import label_clusters_ctfidf

    ctfidf_results = label_clusters_ctfidf(texts, consensus_labels, top_n=top_n)
    topics = [[t.term for t in r.top_terms] for r in ctfidf_results]

    # UMass / NPMI (native — no extra deps)
    doc_term, vocab_index = _build_binary_doc_term_matrix(texts)
    umass_scores = [
        _coherence_umass(t[:top_n], doc_term, vocab_index, 1e-12) for t in topics
    ]
    npmi_scores = [
        _coherence_npmi(t[:top_n], doc_term, vocab_index, 1e-12) for t in topics
    ]

    # c_v (optional, requires gensim)
    cv_scores: list[float] | None
    try:
        from reap.evaluation import compute_topic_coherence

        cv_scores = compute_topic_coherence(topics, texts, measure="c_v", top_n=top_n)
    except ImportError:
        cv_scores = None
        logger.warning("gensim not installed; c_v coherence skipped. pip install gensim")

    diversity = compute_topic_diversity(topics, top_n=top_n)
    exclusivity = compute_topic_exclusivity(texts, consensus_labels, top_n=top_n)

    return {
        "topic_coherence_umass": float(np.mean(umass_scores)) if umass_scores else None,
        "topic_coherence_npmi": float(np.mean(npmi_scores)) if npmi_scores else None,
        "topic_coherence_cv": float(np.mean(cv_scores)) if cv_scores else None,
        "topic_diversity": float(diversity),
        "topic_exclusivity": float(np.mean(exclusivity)) if exclusivity else None,
    }


def _compute_persistence_fields(
    consensus_labels: np.ndarray | None,
    seed_labels_list: list[np.ndarray],
    jaccard_threshold: float = 0.5,
) -> dict[str, Any]:
    """Cluster persistence across per-seed clusterings, or all-None if no consensus."""
    if consensus_labels is None or not seed_labels_list:
        return {
            "cluster_persistence_mean": None,
            "cluster_persistence_min": None,
            "cluster_persistence_median": None,
            "cluster_persistence_per_cluster": None,
        }
    persistence = compute_cluster_persistence(
        consensus_labels, seed_labels_list, jaccard_threshold
    )
    pc = persistence["per_cluster"]
    return {
        "cluster_persistence_mean": float(persistence["mean"]),  # type: ignore[arg-type]
        "cluster_persistence_min": float(persistence["min"]),  # type: ignore[arg-type]
        "cluster_persistence_median": float(persistence["median"]),  # type: ignore[arg-type]
        "cluster_persistence_per_cluster": (
            {str(k): float(v) for k, v in pc.items()} if isinstance(pc, dict) else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def _run_single_seed(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    k_fixed: int | None,
    input_metric: str,
    sil_metric: str,
    ground_truth_labels: np.ndarray | None,
    texts: list[str] | None,
    artifacts_collector: dict[str, dict[str, Any]] | None = None,
) -> MethodResult:
    """Per-seed-independent evaluation. No consensus — emits Cell A only."""
    t0 = time.monotonic()
    tracemalloc.start()

    cell_a = _compute_per_seed_cell_a(
        X_high, umap_embeddings, k_range, k_fixed, input_metric, sil_metric
    )
    seed_labels_list: list[np.ndarray] = cell_a["labels"]  # type: ignore[assignment]

    # Pairwise s2s summaries (no consensus labels → s2c stays None)
    agreement = _compute_pairwise_agreement(
        seed_labels_list, consensus_labels=None, compute_matrices=True
    )

    # External validity for single_seed: compute against the median seed
    ext_fields: dict[str, object] = {
        "ext_ari": None, "ext_ami": None, "ext_nmi": None,
        "ext_homogeneity": None, "ext_completeness": None,
        "ext_v_measure": None, "ext_fowlkes_mallows": None,
        "ext_mean_purity": None, "ext_purity_per_cluster": None,
    }
    if ground_truth_labels is not None and seed_labels_list:
        # Aggregate ARI/AMI/NMI vs ground truth across seeds; report means.
        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

        ari_vs_gt = [float(adjusted_rand_score(ground_truth_labels, sl)) for sl in seed_labels_list]
        ami_vs_gt = [float(adjusted_mutual_info_score(ground_truth_labels, sl)) for sl in seed_labels_list]
        nmi_vs_gt = [float(normalized_mutual_info_score(ground_truth_labels, sl)) for sl in seed_labels_list]
        ext_fields["ext_ari"] = float(np.mean(ari_vs_gt))
        ext_fields["ext_ami"] = float(np.mean(ami_vs_gt))
        ext_fields["ext_nmi"] = float(np.mean(nmi_vs_gt))
        # Leave h/c/v/purity None for single_seed since they're cluster-specific

    # No topic metrics for single_seed (no single "consensus" to label)
    topic_fields = _compute_topic_fields(
        texts=None, consensus_labels=None, top_n=10
    )

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    elapsed = time.monotonic() - t0

    if artifacts_collector is not None:
        artifacts_collector["single_seed"] = {
            "seed_labels": [np.asarray(sl, dtype=np.int64) for sl in seed_labels_list],
        }

    tw_arr = np.array(cell_a["trustworthiness"])
    sil_arr = np.array(cell_a["silhouette"])
    k_arr = np.array(cell_a["k"])

    return MethodResult(
        method="single_seed",
        description=_METHOD_DESCRIPTIONS["single_seed"],
        n_seeds=len(umap_embeddings),
        runtime_seconds=elapsed,
        peak_memory_mb=float(peak_bytes) / (1024 * 1024),
        trustworthiness=float(tw_arr.mean()),
        trustworthiness_std=float(tw_arr.std()),
        silhouette=float(sil_arr.mean()),
        silhouette_std=float(sil_arr.std()),
        best_k=int(np.median(k_arr)),
        best_k_std=float(k_arr.std()),
        seed_to_seed_ari_mean=agreement["s2s_ari_mean"],  # type: ignore[arg-type]
        seed_to_seed_ari_std=agreement["s2s_ari_std"],  # type: ignore[arg-type]
        seed_to_consensus_ari_mean=None,
        per_seed_silhouette=cell_a["silhouette"],  # type: ignore[arg-type]
        per_seed_trustworthiness=cell_a["trustworthiness"],  # type: ignore[arg-type]
        per_seed_continuity=cell_a["continuity"],  # type: ignore[arg-type]
        per_seed_davies_bouldin=cell_a["davies_bouldin"],  # type: ignore[arg-type]
        per_seed_calinski_harabasz=cell_a["calinski_harabasz"],  # type: ignore[arg-type]
        per_seed_k=cell_a["k"],  # type: ignore[arg-type]
        cell_b_silhouette=None,
        cell_c_silhouette=None,
        consensus_silhouette=None,
        consensus_trustworthiness=None,
        consensus_continuity=None,
        consensus_davies_bouldin=None,
        consensus_calinski_harabasz=None,
        consensus_k=None,
        **{k: v for k, v in agreement.items() if k.startswith("s2s_") or k.startswith("s2c_")},
        **ext_fields,
        **topic_fields,
        cluster_persistence_mean=None,
        cluster_persistence_min=None,
        cluster_persistence_median=None,
        cluster_persistence_per_cluster=None,
    )


def _run_consensus_method(
    method_name: str,
    consensus_emb: np.ndarray,
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    k_fixed: int | None,
    input_metric: str,
    sil_metric: str,
    ground_truth_labels: np.ndarray | None,
    texts: list[str] | None,
    t0: float,
    peak_bytes: float | None,
    artifacts_collector: dict[str, dict[str, Any]] | None = None,
) -> MethodResult:
    """Shared runner for methods that produce a single consensus embedding.

    Used by naive_average, procrustes, reap, bertopic (the latter passes its
    own embedding in via ``consensus_emb``).
    """
    # Cell A (always — so every method has a per-seed distribution)
    cell_a = _compute_per_seed_cell_a(
        X_high, umap_embeddings, k_range, k_fixed, input_metric, sil_metric
    )
    seed_labels_list: list[np.ndarray] = cell_a["labels"]  # type: ignore[assignment]

    # Consensus K
    consensus_k = (
        k_fixed
        if k_fixed is not None
        else int(find_best_k(consensus_emb, k_range, metric=sil_metric).best_k)
    )
    consensus_labels = _cluster_embedding(consensus_emb, consensus_k)

    # Cell D + trust/continuity/DB/CH at consensus
    singletons = _compute_consensus_singletons(
        X_high, consensus_emb, consensus_labels, input_metric, sil_metric
    )
    # Cell B and Cell C
    cell_b = _compute_cell_b(consensus_emb, seed_labels_list, sil_metric)
    cell_c = _compute_cell_c(umap_embeddings, consensus_labels, sil_metric)

    # Agreement
    agreement = _compute_pairwise_agreement(
        seed_labels_list, consensus_labels, compute_matrices=True
    )

    # External validity
    ext_fields = _compute_external_fields(ground_truth_labels, consensus_labels)

    # Topic metrics
    topic_fields = _compute_topic_fields(texts, consensus_labels, top_n=10)

    # Cluster persistence
    persistence_fields = _compute_persistence_fields(
        consensus_labels, seed_labels_list
    )

    if artifacts_collector is not None:
        bucket = artifacts_collector.setdefault(method_name, {})
        bucket["consensus_embedding"] = np.asarray(consensus_emb, dtype=np.float64)
        bucket["consensus_labels"] = np.asarray(consensus_labels, dtype=np.int64)
        bucket["seed_labels"] = [np.asarray(sl, dtype=np.int64) for sl in seed_labels_list]
        bucket["consensus_k"] = int(consensus_k)

    elapsed = time.monotonic() - t0
    peak_mb = (
        float(peak_bytes) / (1024 * 1024) if peak_bytes is not None else None
    )

    return MethodResult(
        method=method_name,
        description=_METHOD_DESCRIPTIONS[method_name],
        n_seeds=len(umap_embeddings),
        runtime_seconds=elapsed,
        peak_memory_mb=peak_mb,
        # Legacy-compatible headline fields use the consensus values
        trustworthiness=float(singletons["trustworthiness"]),
        trustworthiness_std=0.0,
        silhouette=float(singletons["silhouette"]),
        silhouette_std=0.0,
        best_k=int(consensus_k),
        best_k_std=0.0,
        seed_to_seed_ari_mean=agreement["s2s_ari_mean"],  # type: ignore[arg-type]
        seed_to_seed_ari_std=agreement["s2s_ari_std"],  # type: ignore[arg-type]
        seed_to_consensus_ari_mean=agreement["s2c_ari_mean"],  # type: ignore[arg-type]
        per_seed_silhouette=cell_a["silhouette"],  # type: ignore[arg-type]
        per_seed_trustworthiness=cell_a["trustworthiness"],  # type: ignore[arg-type]
        per_seed_continuity=cell_a["continuity"],  # type: ignore[arg-type]
        per_seed_davies_bouldin=cell_a["davies_bouldin"],  # type: ignore[arg-type]
        per_seed_calinski_harabasz=cell_a["calinski_harabasz"],  # type: ignore[arg-type]
        per_seed_k=cell_a["k"],  # type: ignore[arg-type]
        cell_b_silhouette=cell_b,
        cell_c_silhouette=cell_c,
        consensus_silhouette=float(singletons["silhouette"]),
        consensus_trustworthiness=float(singletons["trustworthiness"]),
        consensus_continuity=float(singletons["continuity"]),
        consensus_davies_bouldin=float(singletons["davies_bouldin"]),
        consensus_calinski_harabasz=float(singletons["calinski_harabasz"]),
        consensus_k=int(consensus_k),
        **{
            k: v for k, v in agreement.items()
            if k.startswith("s2s_") or k.startswith("s2c_")
        },
        **ext_fields,
        **topic_fields,
        **persistence_fields,
    )


def _run_best_of_n(
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    k_fixed: int | None,
    input_metric: str,
    sil_metric: str,
    ground_truth_labels: np.ndarray | None,
    texts: list[str] | None,
    artifacts_collector: dict[str, dict[str, Any]] | None = None,
) -> MethodResult:
    """Pick the seed with the highest silhouette score as the consensus."""
    t0 = time.monotonic()
    tracemalloc.start()

    cell_a = _compute_per_seed_cell_a(
        X_high, umap_embeddings, k_range, k_fixed, input_metric, sil_metric
    )
    sil_arr = np.array(cell_a["silhouette"])
    best_idx = int(np.argmax(sil_arr))
    best_emb = umap_embeddings[best_idx]

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    logger.info(
        "best_of_n: selected seed %d/%d (sil=%.4f)",
        best_idx + 1, len(umap_embeddings), sil_arr[best_idx],
    )

    return _run_consensus_method(
        method_name="best_of_n",
        consensus_emb=best_emb,
        X_high=X_high,
        umap_embeddings=umap_embeddings,
        k_range=k_range,
        k_fixed=k_fixed,
        input_metric=input_metric,
        sil_metric=sil_metric,
        ground_truth_labels=ground_truth_labels,
        texts=texts,
        t0=t0,
        peak_bytes=peak_bytes,
        artifacts_collector=artifacts_collector,
    )


def _run_naive_average(
    X_high, umap_embeddings, k_range, k_fixed,
    input_metric, sil_metric, ground_truth_labels, texts,
    artifacts_collector=None,
):
    """Average coordinates across seeds without alignment (negative control)."""
    t0 = time.monotonic()
    tracemalloc.start()
    naive_emb = _get_naive_average(umap_embeddings)
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return _run_consensus_method(
        "naive_average", naive_emb, X_high, umap_embeddings, k_range, k_fixed,
        input_metric, sil_metric, ground_truth_labels, texts, t0, peak_bytes,
        artifacts_collector=artifacts_collector,
    )


def _run_procrustes(
    X_high, umap_embeddings, k_range, k_fixed,
    input_metric, sil_metric, ground_truth_labels, texts,
    artifacts_collector=None,
):
    """Procrustes-align to first seed, then average coordinates."""
    t0 = time.monotonic()
    tracemalloc.start()
    proc_emb = get_procrustes_consensus(umap_embeddings)
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return _run_consensus_method(
        "procrustes", proc_emb, X_high, umap_embeddings, k_range, k_fixed,
        input_metric, sil_metric, ground_truth_labels, texts, t0, peak_bytes,
        artifacts_collector=artifacts_collector,
    )


def _run_reap(
    X_high, umap_embeddings, k_range, k_fixed,
    n_components, n_neighbors, min_dist,
    input_metric, sil_metric, ground_truth_labels, texts,
    artifacts_collector=None,
):
    """Distance-matrix consensus + UMAP projection (the proposed method)."""
    t0 = time.monotonic()
    tracemalloc.start()
    D_consensus = get_consensus_distance_matrix(umap_embeddings)
    reap_emb, _ = get_consensus_embedding(
        D_consensus, n_components, n_neighbors, min_dist
    )
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    result = _run_consensus_method(
        "reap", reap_emb, X_high, umap_embeddings, k_range, k_fixed,
        input_metric, sil_metric, ground_truth_labels, texts, t0, peak_bytes,
        artifacts_collector=artifacts_collector,
    )
    # REAP additionally persists the consensus distance matrix for between-set
    # Frobenius comparisons.
    if artifacts_collector is not None:
        artifacts_collector["reap"]["consensus_distance_matrix"] = np.asarray(
            D_consensus, dtype=np.float64
        )
    return result


def _run_bertopic(
    X_high, umap_embeddings, k_range, k_fixed,
    n_components, n_neighbors, min_dist,
    input_metric, sil_metric, ground_truth_labels, texts,
    seeds: list[int],
    artifacts_collector=None,
):
    """BERTopic's default pipeline, evaluated through our KMeans + metrics harness.

    Delegates the BERTopic fit itself to :func:`reap.baselines.run_bertopic_baseline`
    (the module-level home of the baseline); this wrapper handles the timing,
    memory tracking, and downstream KMeans-on-embedding evaluation that all
    other consensus methods share.

    BERTopic is fundamentally a single-seed method (one UMAP fit + one HDBSCAN
    fit per call). We parameterise its internal UMAP ``random_state`` with
    ``seeds[0]`` — the first seed of the current set. This way the between-set
    reliability comparison reflects BERTopic's actual seed sensitivity (its
    output WILL differ across sets A/B/C because the first seed differs)
    rather than being artificially 1.0 from a hardcoded library default.
    """
    from reap.baselines import run_bertopic_baseline

    t0 = time.monotonic()
    tracemalloc.start()

    bertopic_seed = int(seeds[0])
    baseline_result = run_bertopic_baseline(
        embeddings=X_high,
        texts=texts,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=input_metric,
        random_state=bertopic_seed,
    )
    bertopic_emb = baseline_result.embedding

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return _run_consensus_method(
        "bertopic", bertopic_emb, X_high, umap_embeddings, k_range, k_fixed,
        input_metric, sil_metric, ground_truth_labels, texts, t0, peak_bytes,
        artifacts_collector=artifacts_collector,
    )


def _run_parametric_umap(
    X_high, umap_embeddings, k_range, k_fixed,
    n_components, n_neighbors, min_dist,
    input_metric, sil_metric, ground_truth_labels, texts,
    seeds: list[int],
    artifacts_collector=None,
):
    """Sainburg 2021 Parametric UMAP, evaluated through our KMeans + metrics harness.

    Direct prior art for REAP's projection-head contribution: a neural-network
    encoder trained against UMAP's fuzzy-simplicial-set objective. Like BERTopic,
    Parametric UMAP is fundamentally single-seed — we pass ``seeds[0]`` as its
    ``random_state`` so the between-set ARI comparison honestly reflects its
    seed sensitivity (output differs across A/B/C because seeds[0] differs).
    """
    from reap.baselines.parametric_umap_baseline import run_parametric_umap_baseline

    t0 = time.monotonic()
    tracemalloc.start()

    param_seed = int(seeds[0])
    result = run_parametric_umap_baseline(
        embeddings=X_high,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=input_metric,
        random_state=param_seed,
    )
    param_emb = result.embedding

    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return _run_consensus_method(
        "parametric_umap", param_emb, X_high, umap_embeddings, k_range, k_fixed,
        input_metric, sil_metric, ground_truth_labels, texts, t0, peak_bytes,
        artifacts_collector=artifacts_collector,
    )


# ---------------------------------------------------------------------------
# Dispatch + public API
# ---------------------------------------------------------------------------


MethodName = Literal[
    "single_seed", "best_of_n", "naive_average", "procrustes",
    "reap", "bertopic", "parametric_umap",
]


def _dispatch_method(
    method: str,
    X_high: np.ndarray,
    umap_embeddings: list[np.ndarray],
    k_range: list[int],
    k_fixed: int | None,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    input_metric: str,
    sil_metric: str,
    ground_truth_labels: np.ndarray | None,
    texts: list[str] | None,
    seeds: list[int],
    artifacts_collector: dict[str, dict[str, Any]] | None = None,
) -> MethodResult:
    """Route to the correct per-method runner.

    ``seeds`` is the per-set seed list. Most runners consume it implicitly
    via ``umap_embeddings`` (already computed from these seeds upstream),
    but BERTopic — which fits its own UMAP — needs the seed list itself
    so its internal ``random_state`` varies across A/B/C.
    """
    if method == "single_seed":
        return _run_single_seed(
            X_high, umap_embeddings, k_range, k_fixed,
            input_metric, sil_metric, ground_truth_labels, texts,
            artifacts_collector=artifacts_collector,
        )
    if method == "best_of_n":
        return _run_best_of_n(
            X_high, umap_embeddings, k_range, k_fixed,
            input_metric, sil_metric, ground_truth_labels, texts,
            artifacts_collector=artifacts_collector,
        )
    if method == "naive_average":
        return _run_naive_average(
            X_high, umap_embeddings, k_range, k_fixed,
            input_metric, sil_metric, ground_truth_labels, texts,
            artifacts_collector=artifacts_collector,
        )
    if method == "procrustes":
        return _run_procrustes(
            X_high, umap_embeddings, k_range, k_fixed,
            input_metric, sil_metric, ground_truth_labels, texts,
            artifacts_collector=artifacts_collector,
        )
    if method == "reap":
        return _run_reap(
            X_high, umap_embeddings, k_range, k_fixed,
            n_components, n_neighbors, min_dist,
            input_metric, sil_metric, ground_truth_labels, texts,
            artifacts_collector=artifacts_collector,
        )
    if method == "bertopic":
        return _run_bertopic(
            X_high, umap_embeddings, k_range, k_fixed,
            n_components, n_neighbors, min_dist,
            input_metric, sil_metric, ground_truth_labels, texts,
            seeds,
            artifacts_collector=artifacts_collector,
        )
    if method == "parametric_umap":
        return _run_parametric_umap(
            X_high, umap_embeddings, k_range, k_fixed,
            n_components, n_neighbors, min_dist,
            input_metric, sil_metric, ground_truth_labels, texts,
            seeds,
            artifacts_collector=artifacts_collector,
        )
    raise ValueError(
        f"Unknown method {method!r}. Choose from: {ALL_METHODS}"
    )


def run_benchmark_with_artifacts(
    X: np.ndarray,
    dataset_name: str,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
    methods: list[str] | None = None,
    *,
    ground_truth_labels: np.ndarray | None = None,
    texts: list[str] | None = None,
    k_fixed: int | None = None,
) -> tuple[BenchmarkResult, BenchmarkArtifacts]:
    """Same as :func:`run_benchmark` but also returns the raw numpy artifacts.

    Useful for downstream analyses that need the consensus embeddings, labels,
    or the REAP consensus distance matrix (between-seed-set reliability,
    statistical-inference scripts, persistence-on-disk for reproducibility).

    Returns
    -------
    ``(BenchmarkResult, BenchmarkArtifacts)`` where the artifacts bundle
    carries ``umap_embeddings`` (list of N x d arrays) and per-method
    ``consensus_embedding``, ``consensus_labels``, ``seed_labels``, and
    (REAP only) ``consensus_distance_matrix``.
    """
    collector: dict[str, dict[str, Any]] = {}
    result = _run_benchmark_impl(
        X, dataset_name, seeds, n_components, n_neighbors, min_dist, k_range,
        metric, silhouette_metric, methods,
        ground_truth_labels=ground_truth_labels, texts=texts, k_fixed=k_fixed,
        artifacts_collector=collector,
    )
    # Pull umap_embeddings out of the implementation by running it once — we
    # can't get them from collector because single_seed etc. only store labels.
    # Instead, the impl stores them under the sentinel key "__umap_embeddings__".
    umap_embs = collector.pop("__umap_embeddings__", {}).get("embeddings", [])
    artifacts = BenchmarkArtifacts(
        umap_embeddings=umap_embs,
        per_method=collector,
    )
    return result, artifacts


def run_benchmark(
    X: np.ndarray,
    dataset_name: str,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
    methods: list[str] | None = None,
    *,
    ground_truth_labels: np.ndarray | None = None,
    texts: list[str] | None = None,
    k_fixed: int | None = None,
) -> BenchmarkResult:
    """Run all consensus methods on a dataset and compare.

    Shares the 30-seed UMAP embedding set across all methods, then computes
    per-method metrics including the 2 x 2 per-seed x consensus decomposition
    for silhouette, plus full 30x30 pairwise ARI/AMI/NMI matrices, plus
    optional external validity and topic metrics.

    Parameters
    ----------
    X : (n_samples, n_features) input data (e.g., sentence embeddings).
    dataset_name : Human-readable dataset identifier.
    seeds : Random seeds for multi-seed UMAP.
    n_components, n_neighbors, min_dist : UMAP params.
    k_range : Candidate K values for silhouette-based K selection.
    metric : Distance metric for the high-dimensional input space
        (default "cosine" for sentence embeddings).
    silhouette_metric : Metric for silhouette computation in low-d (default
        "euclidean" — correct for UMAP output per McInnes 2018).
    methods : Subset of ALL_METHODS to run. Defaults to all.
    ground_truth_labels : Optional (N,) array for external validity metrics.
    texts : Optional list of N documents for topic coherence / diversity /
        exclusivity.
    k_fixed : If set, every method uses this K (matched-K mode). If None,
        each method picks K via silhouette (default).

    Returns
    -------
    :class:`BenchmarkResult` with per-method evaluations.

    Side effects
    ------------
    UMAP and KMeans internal random state. Logging. ``tracemalloc``
    peak-memory tracking.

    Raises
    ------
    ValueError
        If ``X`` is not 2-d, fewer than 2 seeds, or an unknown method.
    """
    return _run_benchmark_impl(
        X, dataset_name, seeds, n_components, n_neighbors, min_dist, k_range,
        metric, silhouette_metric, methods,
        ground_truth_labels=ground_truth_labels, texts=texts, k_fixed=k_fixed,
        artifacts_collector=None,
    )


def _run_benchmark_impl(
    X: np.ndarray,
    dataset_name: str,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    metric: str,
    silhouette_metric: str,
    methods: list[str] | None,
    *,
    ground_truth_labels: np.ndarray | None,
    texts: list[str] | None,
    k_fixed: int | None,
    artifacts_collector: dict[str, dict[str, Any]] | None,
) -> BenchmarkResult:
    """Internal implementation shared by run_benchmark and run_benchmark_with_artifacts."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2-d input, got shape {X.shape}")
    if len(seeds) < 2:
        raise ValueError(f"Need at least 2 seeds, got {len(seeds)}")

    if ground_truth_labels is not None:
        if len(ground_truth_labels) != X.shape[0]:
            raise ValueError(
                f"ground_truth_labels length {len(ground_truth_labels)} "
                f"!= X.shape[0] {X.shape[0]}"
            )
    if texts is not None and len(texts) != X.shape[0]:
        raise ValueError(
            f"texts length {len(texts)} != X.shape[0] {X.shape[0]}"
        )

    selected = methods if methods is not None else ALL_METHODS
    for m in selected:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method {m!r}. Choose from: {ALL_METHODS}")

    n_samples, n_features = X.shape
    logger.info(
        "Benchmark '%s': n=%d d=%d seeds=%d methods=%s k_fixed=%s",
        dataset_name, n_samples, n_features, len(seeds), selected, k_fixed,
    )

    logger.info("Computing %d UMAP embeddings (shared across methods)...", len(seeds))
    t_umap = time.monotonic()
    umap_embeddings = get_multi_seed_embeddings(
        X, seeds, n_components, n_neighbors, min_dist, metric
    )
    logger.info("UMAP embeddings computed in %.1fs", time.monotonic() - t_umap)

    if artifacts_collector is not None:
        # Sentinel key; extracted by run_benchmark_with_artifacts.
        artifacts_collector["__umap_embeddings__"] = {
            "embeddings": [np.asarray(e, dtype=np.float32) for e in umap_embeddings]
        }

    method_results: list[MethodResult] = []
    for method_name in selected:
        logger.info("Running method: %s", method_name)
        result = _dispatch_method(
            method_name, X, umap_embeddings, k_range, k_fixed,
            n_components, n_neighbors, min_dist,
            metric, silhouette_metric,
            ground_truth_labels, texts,
            seeds,
            artifacts_collector=artifacts_collector,
        )
        method_results.append(result)
        logger.info(
            "  %s: TW=%.4f sil=%.4f K=%d (%.1fs, peak %.1fMB)",
            method_name, result.trustworthiness, result.silhouette,
            result.best_k, result.runtime_seconds,
            result.peak_memory_mb or 0.0,
        )

    return BenchmarkResult(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        k_range=list(k_range),
        n_seeds=len(seeds),
        k_fixed=k_fixed,
        input_metric=metric,
        silhouette_metric=silhouette_metric,
        seeds=list(seeds),
        methods=method_results,
    )


# ---------------------------------------------------------------------------
# Subsample-based confidence intervals (legacy path, renamed for clarity)
# ---------------------------------------------------------------------------


def compute_subsample_ci(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    k_range: list[int],
    n_bootstrap: int = 100,
    subsample_fraction: float = 0.8,
    metric: str = "cosine",
    silhouette_metric: str = "euclidean",
    methods: list[str] | None = None,
    random_state: int = 42,
) -> list[BootstrapCI]:
    """Subsample-based "bootstrap" — resamples data rows (not seeds).

    Answers "how stable is the metric under data resampling?" This is
    complementary to :func:`reap.statistics.compute_seed_bootstrap_ci`,
    which resamples the seed distribution and is what the pre-registered
    protocol §8 calls for when reporting per-seed metric CIs.

    Previously named ``compute_bootstrap_ci``. Renamed to avoid conflating
    the two different uncertainty questions (data vs seed).

    Parameters
    ----------
    X : (n_samples, n_features) input data.
    seeds, n_components, n_neighbors, min_dist, k_range, metric,
    silhouette_metric, methods : As in :func:`run_benchmark`.
    n_bootstrap : Number of subsample iterations.
    subsample_fraction : Fraction of rows to subsample each iteration.
    random_state : Master seed for reproducible subsampling.

    Returns
    -------
    List of :class:`BootstrapCI`, one per (method, metric) combination.
    """
    if not 0.0 < subsample_fraction < 1.0:
        raise ValueError(
            f"subsample_fraction must be in (0, 1), got {subsample_fraction}"
        )

    selected = methods if methods is not None else ALL_METHODS
    n_samples = X.shape[0]
    subsample_size = min(
        max(
            int(n_samples * subsample_fraction),
            (max(k_range) + 1) if k_range else 10,
        ),
        n_samples,
    )
    rng = np.random.default_rng(random_state)

    distributions: dict[str, dict[str, list[float]]] = {
        m: {"trustworthiness": [], "silhouette": [], "best_k": []}
        for m in selected
    }

    logger.info(
        "Subsample CI: %d iters, subsample=%d/%d, methods=%s",
        n_bootstrap, subsample_size, n_samples, selected,
    )

    for b in range(n_bootstrap):
        logger.info("Subsample iteration %d/%d", b + 1, n_bootstrap)
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X[indices]
        k_range_sub = [k for k in k_range if k < subsample_size]
        if not k_range_sub:
            logger.warning("subsample %d: no valid K, skipping", b + 1)
            continue
        try:
            bench = run_benchmark(
                X_sub, dataset_name=f"subsample_{b}", seeds=seeds,
                n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                k_range=k_range_sub, metric=metric,
                silhouette_metric=silhouette_metric, methods=selected,
            )
        except Exception:
            logger.exception("subsample %d failed, skipping", b + 1)
            continue
        for mr in bench.methods:
            distributions[mr.method]["trustworthiness"].append(mr.trustworthiness)
            distributions[mr.method]["silhouette"].append(mr.silhouette)
            distributions[mr.method]["best_k"].append(float(mr.best_k))

    ci_results: list[BootstrapCI] = []
    for m in selected:
        for metric_name in ("trustworthiness", "silhouette", "best_k"):
            values = np.array(distributions[m][metric_name])
            if values.size == 0:
                continue
            ci_results.append(
                BootstrapCI(
                    metric=metric_name, method=m,
                    mean=float(values.mean()),
                    ci_lower=float(np.percentile(values, 2.5)),
                    ci_upper=float(np.percentile(values, 97.5)),
                    n_bootstrap=int(values.size),
                )
            )
    logger.info("Subsample CIs: %d computed", len(ci_results))
    return ci_results


# Backwards-compat alias (kept pre-v1 without deprecation warning; see
# feedback memory: no version bumps yet).
compute_bootstrap_ci = compute_subsample_ci
