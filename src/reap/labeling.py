"""Cluster labeling via c-TF-IDF and optional LLM refinement.

Three independent labeling methods:
- Method A: c-TF-IDF (statistical, no API needed)
- Method B: LLM-only (requires openai or anthropic)
- Method C: Combined (c-TF-IDF terms fed to LLM as context)

Point stratification (core vs peripheral) can be applied before any method
to focus labels on the most representative points.

Requires the `labeling` optional dependency for LLM features:
pip install reap-embeddings[labeling]
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples

from reap._types import CTfidfClusterResult, CTfidfTerm, PointStratification, StratificationSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point stratification
# ---------------------------------------------------------------------------


def stratify_points(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_threshold: float = 0.0,
) -> tuple[list[PointStratification], StratificationSummary]:
    """Split points into core (high silhouette) and peripheral (low silhouette).

    Parameters
    ----------
    X : (n_samples, d) embedding coordinates.
    labels : (n_samples,) cluster assignments.
    silhouette_threshold : Points with silhouette >= threshold are core.

    Returns
    -------
    (point_list, summary) with per-point classification and aggregate stats.
    """
    sil_scores = silhouette_samples(X, labels, metric="euclidean")

    # Centroid distances
    centroids: dict[int, np.ndarray] = {}
    for label in set(labels.tolist()):
        mask = labels == label
        centroids[label] = X[mask].mean(axis=0)

    points: list[PointStratification] = []
    for i in range(len(X)):
        label = int(labels[i])
        dist = float(np.linalg.norm(X[i] - centroids[label]))
        stratum = "core" if sil_scores[i] >= silhouette_threshold else "peripheral"
        points.append(PointStratification(
            point_index=i,
            cluster=label,
            silhouette_score=float(sil_scores[i]),
            centroid_distance=dist,
            stratum=stratum,
        ))

    core_mask = np.array([p.stratum == "core" for p in points])
    core_labels = labels[core_mask]
    core_sizes = dict(Counter(core_labels.tolist()))

    summary = StratificationSummary(
        silhouette_threshold=silhouette_threshold,
        total_points=len(points),
        core_points=int(core_mask.sum()),
        peripheral_points=int((~core_mask).sum()),
        core_silhouette_mean=float(sil_scores[core_mask].mean()) if core_mask.any() else 0.0,
        all_silhouette_mean=float(sil_scores.mean()),
        core_cluster_sizes=core_sizes,
        smallest_core_cluster=min(core_sizes.values()) if core_sizes else 0,
    )
    return points, summary


# ---------------------------------------------------------------------------
# Method A: c-TF-IDF labeling
# ---------------------------------------------------------------------------


def build_ctfidf_matrix(
    texts: list[str],
    labels: np.ndarray,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str | list[str] | None = "english",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a c-TF-IDF matrix for cluster labeling.

    c-TF-IDF (class-based TF-IDF) treats each cluster as a single document,
    then weights terms by their frequency within the cluster relative to
    their frequency across all clusters.

    Parameters
    ----------
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    ngram_range : N-gram sizes to extract.
    min_df : Minimum document frequency for terms.
    max_df : Maximum document frequency fraction.
    stop_words : Stop word list or "english" for built-in.

    Returns
    -------
    (ctfidf_matrix, feature_names, cluster_ids) where ctfidf_matrix is
    (n_clusters, n_terms).
    """
    unique_labels = sorted(set(labels.tolist()))

    # Concatenate texts per cluster
    cluster_docs: list[str] = []
    for label in unique_labels:
        mask = labels == label
        cluster_docs.append(" ".join(t for t, m in zip(texts, mask) if m))

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
    )
    raw_matrix = vectorizer.fit_transform(cluster_docs)
    tf_matrix = np.asarray(raw_matrix.toarray(), dtype=np.float64)  # type: ignore[union-attr]
    feature_names = vectorizer.get_feature_names_out().tolist()

    # L1-normalize TF per cluster
    row_sums = tf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tf_norm = tf_matrix / row_sums

    # IDF: log(1 + N / df) where df = number of clusters containing the term
    n_clusters = len(unique_labels)
    df = (tf_matrix > 0).sum(axis=0).astype(np.float64)
    df[df == 0] = 1.0
    idf = np.log1p(n_clusters / df)

    ctfidf = tf_norm * idf
    return ctfidf, feature_names, np.array(unique_labels)


def label_clusters_ctfidf(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 2,
    max_df: float = 0.95,
    stop_words: str | list[str] | None = "english",
) -> list[CTfidfClusterResult]:
    """Label clusters using c-TF-IDF term extraction.

    Parameters
    ----------
    texts : Raw text for each data point.
    labels : (n_samples,) cluster assignments.
    top_n : Number of top terms to return per cluster.
    ngram_range : N-gram sizes.
    min_df : Minimum document frequency.
    max_df : Maximum document frequency fraction.
    stop_words : Stop words to exclude.

    Returns
    -------
    List of CTfidfClusterResult, one per cluster.
    """
    ctfidf, features, cluster_ids = build_ctfidf_matrix(
        texts, labels, ngram_range, min_df, max_df, stop_words
    )

    results: list[CTfidfClusterResult] = []
    for i, cluster_id in enumerate(cluster_ids):
        scores = ctfidf[i]
        sorted_idx = np.argsort(scores)[::-1]

        # Top terms by c-TF-IDF score
        top_terms: list[CTfidfTerm] = []
        for j in sorted_idx[:top_n]:
            if scores[j] <= 0:
                break
            term = features[j]
            top_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(scores[j] / max(ctfidf[:, j].max(), 1e-12)),
                cluster=int(cluster_id),
            ))

        # Discriminative terms (highest ratio of this-cluster vs max-other)
        max_other = np.zeros(len(features))
        for k in range(len(cluster_ids)):
            if k != i:
                max_other = np.maximum(max_other, ctfidf[k])
        disc_ratio = np.divide(scores, max_other + 1e-12)
        disc_idx = np.argsort(disc_ratio)[::-1]

        disc_terms: list[CTfidfTerm] = []
        for j in disc_idx[:top_n]:
            if scores[j] <= 0:
                break
            term = features[j]
            disc_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(disc_ratio[j]),
                cluster=int(cluster_id),
            ))

        # Shared terms (high score in multiple clusters)
        shared_terms: list[CTfidfTerm] = []
        multi_cluster = (ctfidf[:, :] > 0).sum(axis=0) > 1
        shared_idx = np.argsort(scores * multi_cluster)[::-1]
        for j in shared_idx[:top_n]:
            if scores[j] <= 0 or not multi_cluster[j]:
                break
            term = features[j]
            shared_terms.append(CTfidfTerm(
                term=term,
                ngram_size=len(term.split()),
                ctfidf_score=float(scores[j]),
                discriminativeness=float(disc_ratio[j]),
                cluster=int(cluster_id),
            ))

        n_items = int((labels == cluster_id).sum())
        results.append(CTfidfClusterResult(
            cluster=int(cluster_id),
            n_items=n_items,
            top_terms=top_terms,
            discriminative_terms=disc_terms,
            shared_terms=shared_terms,
        ))

    return results
