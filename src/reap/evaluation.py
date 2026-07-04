"""Evaluation metrics for REAP consensus quality and seed stability.

Metrics are grouped into eight families:

1. **Structure preservation** (DR quality): trustworthiness (false neighbors),
   continuity (missed neighbors, dual of trustworthiness), LCMC (Chen & Buja
   2009), R_NX curve (Lee et al. 2013) and its area-under-curve summary.
2. **Cluster quality** (internal): silhouette, Davies-Bouldin (lower is
   better), Calinski-Harabasz (higher is better).
3. **Clustering agreement** (pairwise): ARI, AMI, NMI; the full pairwise
   30x30 matrices for seed-to-seed analysis; Variation of Information (Meila
   2007), which is a proper metric on partitions and does not assume matched K.
4. **Seed stability**: seed-to-seed (s2s) and seed-to-consensus (s2c) agreement
   distributions. All three agreement measures are reported; full distributions
   are kept, not just mean/std.
5. **External validity** (vs ground truth): ARI, AMI, NMI, homogeneity,
   completeness, V-measure, Fowlkes-Mallows, and per-cluster purity.
6. **Topic metrics**: coherence (UMass, NPMI, c_v), diversity (Dieng et al.
   2020), and exclusivity (Bischof & Airoldi 2012 adaptation).
7. **Cluster persistence**: for each consensus cluster, the fraction of
   per-seed clusterings that contain a Hungarian-matched counterpart above a
   Jaccard threshold.
8. **Distance correlation**: Pearson correlation of pairwise Euclidean
   distances (projection-head geometry preservation).

Every function has a single-responsibility signature. Wrappers over sklearn
are thin on purpose — explicit names make the evaluation protocol traceable
in `manuscript/evaluation_protocol.md` Sections 7 and 8.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def compute_trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> float:
    """Fraction of true nearest neighbors in high-d that remain neighbors in low-d.

    Values above 0.85 indicate faithful local geometry preservation.

    Parameters
    ----------
    X_high : (n_samples, n_features_high) original high-dimensional data.
    X_low : (n_samples, n_features_low) reduced embedding.
    n_neighbors : Neighborhood size for evaluation.
    metric : Distance metric for the high-dimensional space.

    Returns
    -------
    Trustworthiness score in [0, 1].
    """
    return float(sklearn_trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric=metric))


def compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Silhouette score measuring cluster separation quality.

    For UMAP output space, use metric="euclidean" (the UMAP output objective
    is Euclidean per McInnes 2018). Higher is better; > 0.7 indicates strong clusters.

    Parameters
    ----------
    X : (n_samples, n_features) embedding coordinates.
    labels : (n_samples,) cluster assignments.
    metric : Distance metric. Use "euclidean" for UMAP output, "cosine" for
             high-d sentence embeddings.

    Returns
    -------
    Mean silhouette score in [-1, 1].
    """
    n_unique = len(set(labels))
    if n_unique < 2:
        logger.warning("Only %d unique label(s) — silhouette undefined, returning -1", n_unique)
        return -1.0
    return float(silhouette_score(X, labels, metric=metric))


def compute_pairwise_ari(
    labels_list: list[np.ndarray],
) -> np.ndarray:
    """Pairwise Adjusted Rand Index between multiple label sets.

    Useful for measuring seed-to-seed clustering agreement.

    Parameters
    ----------
    labels_list : List of (n_samples,) label arrays.

    Returns
    -------
    (n_labelings, n_labelings) symmetric ARI matrix with 1.0 on diagonal.
    """
    n = len(labels_list)
    ari_matrix = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari
    return ari_matrix


def compute_seed_stability(
    seed_labels: list[np.ndarray],
    consensus_labels: np.ndarray,
) -> dict[str, float]:
    """Compute seed-to-seed and seed-to-consensus ARI statistics.

    Parameters
    ----------
    seed_labels : Per-seed cluster assignments.
    consensus_labels : Cluster assignments from the consensus embedding.

    Returns
    -------
    Dictionary with mean/std/median for both s2s and s2c ARI.
    """
    # Seed-to-seed ARI
    ari_matrix = compute_pairwise_ari(seed_labels)
    triu = ari_matrix[np.triu_indices_from(ari_matrix, k=1)]

    # Seed-to-consensus ARI
    s2c = np.array([
        adjusted_rand_score(sl, consensus_labels) for sl in seed_labels
    ])

    return {
        "s2s_ari_mean": float(triu.mean()),
        "s2s_ari_std": float(triu.std()),
        "s2s_ari_median": float(np.median(triu)),
        "s2c_ari_mean": float(s2c.mean()),
        "s2c_ari_std": float(s2c.std()),
        "s2c_ari_median": float(np.median(s2c)),
    }


def compute_distance_correlation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """Pearson correlation between condensed pairwise-distance vectors.

    Registered metric recipe ``distance_correlation`` (kind: ``adapted``):
    the Euclidean self-distances of each point set are collected once per
    unique pair (scipy's condensed ``pdist`` form — no diagonal, no
    double-counting), and the two condensed vectors are compared with a
    plain Pearson correlation. Despite the everyday name, this is **not**
    Székely's distance correlation — it is an adapted, simpler statistic
    answering the same practical question: how well is relative geometry
    preserved? Values > 0.9 indicate excellent geometric fidelity.

    This is a different calculation from the training-loss recipe
    ``dist_corr_loss`` in ``reap.projection.compute_projection_loss``,
    which correlates the *full* flattened self-distance matrices (diagonal
    zeros included, every pair counted twice). The two recipes disagree on
    generic inputs and must never be quoted interchangeably; the exact gap
    is pinned in ``tests/verification/test_distance_correlation_rung*``.

    Parameters
    ----------
    Y_true : (n_samples, d) true consensus coordinates.
    Y_pred : (n_samples, d) predicted coordinates.

    Returns
    -------
    Pearson correlation coefficient in [-1, 1]; 0.0 when either condensed
    distance vector is degenerate (standard deviation < 1e-12) or the
    correlation is non-finite.
    """
    from scipy.spatial.distance import pdist

    d_true = pdist(Y_true, metric="euclidean")
    d_pred = pdist(Y_pred, metric="euclidean")

    if d_true.std() < 1e-12 or d_pred.std() < 1e-12:
        return 0.0

    corr = float(np.corrcoef(d_true, d_pred)[0, 1])
    return corr if np.isfinite(corr) else 0.0


# ---------------------------------------------------------------------------
# Topic coherence (UMass, NPMI, c_v)
# ---------------------------------------------------------------------------


def compute_topic_coherence(
    topics: list[list[str]],
    texts: list[str],
    measure: str = "u_mass",
    top_n: int | None = None,
    epsilon: float = 1e-12,
) -> list[float]:
    """Compute topic coherence for each topic's word list against a corpus.

    Dispatches to UMass (Mimno et al. 2011), NPMI (Bouma 2009; Lau et al.
    2014), or c_v (Röder et al. 2015). UMass and NPMI are implemented
    natively; c_v requires gensim (``pip install gensim``).

    Parameters
    ----------
    topics : List of word lists, one per topic. Each inner list contains
        the top discriminative terms for that topic (e.g., from c-TF-IDF).
    texts : Corpus documents used as the reference for co-occurrence
        counting. These should be the same documents the topics were
        derived from.
    measure : ``"u_mass"``, ``"c_npmi"``, or ``"c_v"``.
    top_n : If given, truncate each topic's word list to the first
        ``top_n`` terms before computing coherence. Useful when topics
        have variable-length term lists and you want comparable scores.
    epsilon : Smoothing constant for log-probability terms (avoids
        log(0)). Default 1e-12.

    Returns
    -------
    List of floats, one coherence score per topic. Higher is better for
    all three measures. UMass values are typically negative; NPMI values
    are in [-1, 1]; c_v values are in [0, 1].

    Side effects: none for UMass/NPMI (pure computation). c_v imports
    gensim and builds an internal dictionary/corpus.
    """
    if measure in ("u_mass", "c_npmi"):
        doc_term, vocab_index = _build_binary_doc_term_matrix(texts)
        results: list[float] = []
        for words in topics:
            w = words[:top_n] if top_n is not None else words
            if measure == "u_mass":
                results.append(_coherence_umass(w, doc_term, vocab_index, epsilon))
            else:
                results.append(_coherence_npmi(w, doc_term, vocab_index, epsilon))
        return results
    elif measure == "c_v":
        return _coherence_cv(topics, texts, top_n)
    else:
        raise ValueError(
            f"Unknown coherence measure {measure!r}. "
            "Use 'u_mass', 'c_npmi', or 'c_v'."
        )


def _build_binary_doc_term_matrix(
    texts: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a binary document-term matrix from raw texts.

    Returns (doc_term_binary, vocab_index) where doc_term_binary is
    (n_docs, n_vocab) with 1/0 entries and vocab_index maps term →
    column index.

    Side effects: none (uses sklearn CountVectorizer internally).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    dtm = cv.fit_transform(texts)  # type: ignore[reportAttributeAccessIssue]
    vocab: dict[str, int] = cv.vocabulary_
    return np.asarray(dtm.toarray(), dtype=np.float64), vocab  # type: ignore[union-attr]


def _coherence_umass(
    words: list[str],
    doc_term: np.ndarray,
    vocab_index: dict[str, int],
    epsilon: float,
) -> float:
    """UMass coherence (Mimno et al. 2011) for a single topic.

    UMass = (2 / (N*(N-1))) * sum_{i=2..N, j=1..i-1}
        log( (D(w_i, w_j) + 1) / D(w_j) )

    where D(w) = document frequency of w, D(w_i, w_j) = co-document
    frequency. The +1 smoothing prevents log(0). More negative =
    less coherent; closer to 0 = more coherent.
    """
    indices = [vocab_index[w] for w in words if w in vocab_index]
    if len(indices) < 2:
        return 0.0

    n = len(indices)
    total = 0.0
    for i in range(1, n):
        for j in range(i):
            col_i = doc_term[:, indices[i]]
            col_j = doc_term[:, indices[j]]
            d_ij = float((col_i * col_j).sum())
            d_j = float(col_j.sum())
            if d_j < epsilon:
                continue
            total += np.log((d_ij + 1.0) / d_j)

    n_pairs = n * (n - 1) / 2
    return total / n_pairs if n_pairs > 0 else 0.0


def _coherence_npmi(
    words: list[str],
    doc_term: np.ndarray,
    vocab_index: dict[str, int],
    epsilon: float,
) -> float:
    """NPMI coherence (Bouma 2009; Lau et al. 2014) for a single topic.

    NPMI(w_i, w_j) = ( log(P(w_i,w_j) / (P(w_i)*P(w_j))) )
                     / ( -log(P(w_i,w_j)) )

    Topic NPMI = mean over all unique pairs. Values in [-1, 1]; higher
    indicates stronger positive association between the top terms.
    """
    n_docs = float(doc_term.shape[0])
    indices = [vocab_index[w] for w in words if w in vocab_index]
    if len(indices) < 2:
        return 0.0

    n = len(indices)
    npmi_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            col_i = doc_term[:, indices[i]]
            col_j = doc_term[:, indices[j]]
            p_i = float(col_i.sum()) / n_docs
            p_j = float(col_j.sum()) / n_docs
            p_ij = float((col_i * col_j).sum()) / n_docs

            if p_ij < epsilon or p_i < epsilon or p_j < epsilon:
                npmi_sum += -1.0
            else:
                pmi = np.log(p_ij / (p_i * p_j))
                neg_log_p_ij = -np.log(p_ij)
                if neg_log_p_ij < epsilon:
                    npmi_sum += 0.0
                else:
                    npmi_sum += pmi / neg_log_p_ij
            count += 1

    return npmi_sum / count if count > 0 else 0.0


def _coherence_cv(
    topics: list[list[str]],
    texts: list[str],
    top_n: int | None,
) -> list[float]:
    """c_v coherence (Röder et al. 2015) via gensim's CoherenceModel.

    Requires ``pip install gensim``. Raises ImportError with install
    instructions if gensim is not available.

    Side effects: builds a gensim Dictionary and corpus internally.
    """
    try:
        from gensim.corpora import Dictionary  # type: ignore[import-untyped]
        from gensim.models.coherencemodel import CoherenceModel  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "c_v coherence requires gensim. Install with: pip install gensim"
        ) from None

    tokenized = [text.split() for text in texts]
    dictionary = Dictionary(tokenized)

    topic_words = [
        (w[:top_n] if top_n is not None else w) for w in topics
    ]

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    per_topic = cm.get_coherence_per_topic()
    return [float(v) for v in per_topic]


# ---------------------------------------------------------------------------
# Structure preservation — continuity, LCMC, R_NX curve
# ---------------------------------------------------------------------------


def compute_continuity(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> float:
    """Continuity: fraction of true high-d neighbors that remain neighbors in low-d.

    The dual of trustworthiness (Venna & Kaski 2001). Trustworthiness catches
    *false* neighbors introduced by the projection; continuity catches *true*
    neighbors that got pushed apart. Reporting both is standard in DR papers
    (Lee & Verleysen 2009).

    Implementation note: computed by swapping the arguments to sklearn's
    trustworthiness, which is exactly equivalent to continuity when the same
    metric is used in both spaces. REAP uses cosine for high-d (sentence
    embeddings) and the same metric for low-d rankings for consistency with
    ``compute_trustworthiness``.

    Parameters
    ----------
    X_high : (n_samples, n_features_high) original high-dimensional data.
    X_low : (n_samples, n_features_low) reduced embedding.
    n_neighbors : Neighborhood size for evaluation.
    metric : Distance metric used for both spaces' neighbor orderings.

    Returns
    -------
    Continuity score in [0, 1].
    """
    return float(
        sklearn_trustworthiness(
            X_low, X_high, n_neighbors=n_neighbors, metric=metric
        )
    )


def _compute_rank_matrices(
    X_high: np.ndarray,
    X_low: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full rank-of-neighbor matrices for both spaces.

    Returns ``(rank_hi, rank_lo)`` where ``rank_hi[i, j]`` is the rank of
    ``j`` in ``i``'s sorted neighbor list (0 = nearest, excluding self). For
    ``i == j`` the entry is ``n_samples`` (a sentinel larger than any valid
    rank). Used by :func:`compute_lcmc` and :func:`compute_r_nx_curve`.

    Memory: allocates two ``int32`` arrays of shape ``(N, N)``, i.e.
    ``8 * N^2`` bytes total. At ``N=2000`` that is ~32 MB.

    Pure function. Side effects: none.
    """
    from sklearn.metrics import pairwise_distances

    n = X_high.shape[0]

    d_hi = pairwise_distances(X_high, metric=metric)
    d_lo = pairwise_distances(X_low, metric=metric)
    np.fill_diagonal(d_hi, np.inf)
    np.fill_diagonal(d_lo, np.inf)

    # argsort along each row gives neighbor indices in increasing-distance order.
    # Drop the last column (which corresponds to the self entry we set to inf).
    knn_hi = np.argsort(d_hi, axis=1)[:, :-1]
    knn_lo = np.argsort(d_lo, axis=1)[:, :-1]

    rank_hi = np.full((n, n), n, dtype=np.int32)
    rank_lo = np.full((n, n), n, dtype=np.int32)
    rows = np.arange(n)[:, None]
    ranks_vec = np.arange(n - 1, dtype=np.int32)[None, :]
    rank_hi[rows, knn_hi] = ranks_vec
    rank_lo[rows, knn_lo] = ranks_vec
    return rank_hi, rank_lo


def compute_lcmc(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> float:
    """Local Continuity Meta-Criterion (Chen & Buja 2009).

    Chance-corrected fraction of k-NN preserved between high-d and low-d:
    ``LCMC(k) = Q_NX(k) - k/(N-1)`` where ``Q_NX(k) = mean_i |N_k^hi(i) ∩
    N_k^lo(i)| / k``. Values range roughly in ``[0, 1 - k/(N-1)]``; higher is
    better. A random projection yields ``LCMC ≈ 0``; a perfect projection
    yields ``LCMC = 1 - k/(N-1)``.

    Parameters
    ----------
    X_high : (n_samples, n_features_high) original high-dimensional data.
    X_low : (n_samples, n_features_low) reduced embedding.
    n_neighbors : k for the k-NN comparison.
    metric : Distance metric for both spaces' neighbor orderings.

    Returns
    -------
    LCMC score. Higher is better.
    """
    n = X_high.shape[0]
    if n_neighbors >= n:
        raise ValueError(f"n_neighbors={n_neighbors} must be < n_samples={n}")

    rank_hi, rank_lo = _compute_rank_matrices(X_high, X_low, metric)
    both_within = (rank_hi < n_neighbors) & (rank_lo < n_neighbors)
    q_nx = float(both_within.sum()) / (n * n_neighbors)
    return float(q_nx - n_neighbors / (n - 1))


def compute_r_nx_curve(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_max: int | None = None,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Rescaled neighbor-rank preservation curve R_NX(k) (Lee et al. 2013).

    ``R_NX(k) = ((N-1) Q_NX(k) - k) / (N - 1 - k)`` rescales LCMC so that
    random projections map to 0 and perfect projections to 1, across all k.
    Reporting the full R_NX(k) curve (often on a log-scaled k axis) is the
    modern way to characterize DR quality across scales rather than at a
    single k.

    Parameters
    ----------
    X_high : (n_samples, n_features_high) original data.
    X_low : (n_samples, n_features_low) reduced embedding.
    k_max : Maximum k. Defaults to ``min(100, N // 2)`` to keep memory and
        time bounded; pass a larger value (up to ``N - 2``) for the full
        curve on small datasets.
    metric : Distance metric for both spaces.

    Returns
    -------
    ``(k_values, r_nx_values)`` with ``k_values = [1, 2, ..., k_max]`` and
    corresponding ``r_nx_values`` in ``[0, 1]`` (approximately; may exceed
    slightly for finite-sample noise).
    """
    n = X_high.shape[0]
    if k_max is None:
        k_max = min(100, n // 2)
    k_max = min(k_max, n - 2)
    if k_max < 1:
        raise ValueError(f"k_max={k_max} must be >= 1; n_samples={n}")

    rank_hi, rank_lo = _compute_rank_matrices(X_high, X_low, metric)
    max_rank = np.maximum(rank_hi, rank_lo)

    k_values = np.arange(1, k_max + 1, dtype=np.int32)
    r_nx = np.zeros(k_max, dtype=np.float64)
    for ki, k in enumerate(k_values):
        q_nx = float((max_rank < k).sum()) / (n * k)
        r_nx[ki] = ((n - 1) * q_nx - k) / (n - 1 - k)
    return k_values, r_nx


def compute_r_nx_auc(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k_max: int | None = None,
    metric: str = "cosine",
) -> float:
    """Log-scale area under the R_NX(k) curve (Lee et al. 2015).

    Summary statistic used when a single number is needed alongside the full
    curve. Weights small-k (local) preservation more than large-k (global) by
    integrating on a log-k axis.

    Parameters
    ----------
    X_high, X_low, k_max, metric : See :func:`compute_r_nx_curve`.

    Returns
    -------
    Weighted mean of ``R_NX(k) / k`` normalized by ``sum(1/k)``.
    """
    k_values, r_nx = compute_r_nx_curve(X_high, X_low, k_max, metric)
    weights = 1.0 / k_values.astype(np.float64)
    return float(np.sum(r_nx * weights) / np.sum(weights))


# ---------------------------------------------------------------------------
# Cluster quality — Davies-Bouldin, Calinski-Harabasz
# ---------------------------------------------------------------------------


def compute_davies_bouldin(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Davies-Bouldin cluster-quality index. Lower is better.

    Mean ratio of within-cluster scatter to between-cluster separation over
    each cluster's worst-case sibling. Sensitive to cluster shape in
    complementary ways to silhouette: DB penalizes overlapping dense clusters
    that silhouette can rate highly when well-centered.

    Parameters
    ----------
    X : (n_samples, n_features) embedding coordinates.
    labels : (n_samples,) cluster assignments.

    Returns
    -------
    DB index (``>= 0``). Returns ``np.inf`` when only one cluster exists
    (undefined).
    """
    if len(set(labels.tolist())) < 2:
        logger.warning(
            "Davies-Bouldin requires >= 2 clusters — returning inf"
        )
        return float("inf")
    return float(davies_bouldin_score(X, labels))


def compute_calinski_harabasz(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Calinski-Harabasz (variance-ratio) cluster-quality index. Higher is better.

    Ratio of between-cluster dispersion to within-cluster dispersion, scaled
    by the degrees of freedom. Favors clusterings with compact, well-separated
    clusters; values can be arbitrarily large (no upper bound).

    Parameters
    ----------
    X : (n_samples, n_features) embedding coordinates.
    labels : (n_samples,) cluster assignments.

    Returns
    -------
    CH index (``>= 0``). Returns ``0.0`` when only one cluster exists.
    """
    if len(set(labels.tolist())) < 2:
        logger.warning(
            "Calinski-Harabasz requires >= 2 clusters — returning 0.0"
        )
        return 0.0
    return float(calinski_harabasz_score(X, labels))


def compute_silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """Per-sample silhouette scores.

    Wrapper around sklearn to expose the full per-point distribution used by
    point stratification and for reporting percentile-based silhouette tables
    (not just the mean).

    Parameters
    ----------
    X : (n_samples, n_features) embedding coordinates.
    labels : (n_samples,) cluster assignments.
    metric : Distance metric (default "euclidean" for UMAP output).

    Returns
    -------
    (n_samples,) array of silhouette scores in ``[-1, 1]``.
    """
    return np.asarray(silhouette_samples(X, labels, metric=metric))


# ---------------------------------------------------------------------------
# Clustering agreement — AMI, NMI, VI, pairwise matrices
# ---------------------------------------------------------------------------


def compute_ami(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """Adjusted Mutual Information (Vinh, Epps & Bailey 2010).

    Chance-corrected mutual information. More robust than ARI when cluster
    sizes are imbalanced (Romano et al. 2016). Range: ``[-1, 1]``; 1 is
    perfect agreement; 0 is chance.

    Parameters
    ----------
    labels_a, labels_b : (n_samples,) label arrays.
    average_method : Normalization method passed through to sklearn
        (``"arithmetic"``, ``"geometric"``, ``"min"``, or ``"max"``). The
        arithmetic default matches sklearn's convention and is what we use
        throughout the REAP benchmark.

    Returns
    -------
    AMI score.
    """
    return float(
        adjusted_mutual_info_score(
            labels_a, labels_b, average_method=average_method
        )
    )


def compute_nmi(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """Normalized Mutual Information (uncorrected for chance).

    Range: ``[0, 1]``; 1 is perfect agreement. Unlike AMI it does not correct
    for chance, so it is biased upward as the number of clusters grows.
    Reported alongside AMI for continuity with older literature and because
    reviewers often expect both.

    Parameters
    ----------
    labels_a, labels_b : (n_samples,) label arrays.
    average_method : Normalization method passed to sklearn.

    Returns
    -------
    NMI score.
    """
    return float(
        normalized_mutual_info_score(
            labels_a, labels_b, average_method=average_method
        )
    )


def compute_variation_of_information(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    normalize: bool = False,
) -> float:
    """Variation of Information between two clusterings (Meila 2007).

    ``VI(A, B) = H(A|B) + H(B|A) = H(A) + H(B) - 2 * I(A; B)``, where H is
    Shannon entropy (nats) and I is mutual information. VI is a proper
    metric on the space of partitions; unlike ARI / AMI it does *not* require
    matched K and is bounded above by ``log(N)``. Lower is better.

    Parameters
    ----------
    labels_a, labels_b : (n_samples,) label arrays.
    normalize : If ``True``, return ``VI / log(N)`` in ``[0, 1]``.

    Returns
    -------
    VI in nats (or normalized VI if ``normalize=True``). Lower is better.
    """
    n = len(labels_a)
    if n == 0:
        return 0.0
    if len(labels_b) != n:
        raise ValueError(
            f"Label arrays must have same length; got {n} and {len(labels_b)}"
        )

    # Contingency table
    a_ids, a_inv = np.unique(labels_a, return_inverse=True)
    b_ids, b_inv = np.unique(labels_b, return_inverse=True)
    cont = np.zeros((len(a_ids), len(b_ids)), dtype=np.float64)
    for ai, bi in zip(a_inv, b_inv):
        cont[ai, bi] += 1.0
    p_ab = cont / n

    p_a = p_ab.sum(axis=1)
    p_b = p_ab.sum(axis=0)

    def _entropy(p: np.ndarray) -> float:
        p_nz = p[p > 0]
        return float(-np.sum(p_nz * np.log(p_nz)))

    h_a = _entropy(p_a)
    h_b = _entropy(p_b)

    mi = 0.0
    for i in range(len(a_ids)):
        for j in range(len(b_ids)):
            pij = p_ab[i, j]
            if pij > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += pij * np.log(pij / (p_a[i] * p_b[j]))

    vi = h_a + h_b - 2.0 * mi
    if normalize:
        log_n = float(np.log(n)) if n > 1 else 1.0
        return vi / log_n if log_n > 0 else 0.0
    return vi


def compute_pairwise_ami(
    labels_list: list[np.ndarray],
) -> np.ndarray:
    """Pairwise Adjusted Mutual Information matrix.

    Parameters
    ----------
    labels_list : List of (n_samples,) label arrays.

    Returns
    -------
    (n_labelings, n_labelings) symmetric AMI matrix with 1.0 on diagonal.
    """
    n = len(labels_list)
    m = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = compute_ami(labels_list[i], labels_list[j])
            m[i, j] = v
            m[j, i] = v
    return m


def compute_pairwise_nmi(
    labels_list: list[np.ndarray],
) -> np.ndarray:
    """Pairwise Normalized Mutual Information matrix.

    Parameters
    ----------
    labels_list : List of (n_samples,) label arrays.

    Returns
    -------
    (n_labelings, n_labelings) symmetric NMI matrix with 1.0 on diagonal.
    """
    n = len(labels_list)
    m = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = compute_nmi(labels_list[i], labels_list[j])
            m[i, j] = v
            m[j, i] = v
    return m


# ---------------------------------------------------------------------------
# External validity (labels_true vs labels_pred)
# ---------------------------------------------------------------------------


def compute_cluster_purity(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> tuple[float, dict[int, float]]:
    """Per-cluster and mean purity against ground-truth labels.

    For each predicted cluster, purity is the fraction of points whose true
    label matches the cluster's modal true label. Mean purity is the
    size-weighted average (equivalent to ``#correctly_majority_matched / N``).

    Parameters
    ----------
    labels_true : (n_samples,) ground-truth labels.
    labels_pred : (n_samples,) predicted cluster labels.

    Returns
    -------
    ``(mean_purity, per_cluster_purity)``.
    """
    n = len(labels_true)
    per_cluster: dict[int, float] = {}
    total_correct = 0
    for pid in np.unique(labels_pred):
        mask = labels_pred == pid
        if mask.sum() == 0:
            continue
        _, counts = np.unique(labels_true[mask], return_counts=True)
        dominant = int(counts.max())
        per_cluster[int(pid)] = dominant / int(mask.sum())
        total_correct += dominant
    mean = total_correct / n if n > 0 else 0.0
    return float(mean), per_cluster


def compute_external_validity(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> dict[str, float | dict[int, float]]:
    """Full external-validity suite against ground-truth labels.

    Returns every commonly reported agreement metric in one call so that
    paper tables can be generated without weaving several sklearn calls.

    Parameters
    ----------
    labels_true : (n_samples,) ground-truth labels.
    labels_pred : (n_samples,) predicted cluster labels.

    Returns
    -------
    Dict with keys ``ari``, ``ami``, ``nmi``, ``homogeneity``,
    ``completeness``, ``v_measure``, ``fowlkes_mallows``, ``mean_purity``,
    ``purity_per_cluster``.
    """
    h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
    mean_purity, per_cluster_purity = compute_cluster_purity(
        labels_true, labels_pred
    )
    return {
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "ami": compute_ami(labels_true, labels_pred),
        "nmi": compute_nmi(labels_true, labels_pred),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "fowlkes_mallows": float(
            fowlkes_mallows_score(labels_true, labels_pred)
        ),
        "mean_purity": mean_purity,
        "purity_per_cluster": per_cluster_purity,
    }


# ---------------------------------------------------------------------------
# Topic-level — diversity and exclusivity
# ---------------------------------------------------------------------------


def compute_topic_diversity(
    topics: list[list[str]],
    top_n: int = 25,
) -> float:
    """Fraction of unique words across all topics' top-N term lists.

    Per Dieng, Ruiz & Blei (2020): ``TD = #unique_terms / (top_n * n_topics)``.
    Range ``[1 / n_topics, 1]``; higher means topics are more distinct. A
    corpus with redundant near-duplicate topics scores low even if each
    individual topic is coherent.

    Parameters
    ----------
    topics : Per-topic word lists, e.g. from c-TF-IDF or LDA.
    top_n : How many top words per topic to consider.

    Returns
    -------
    Topic diversity in ``[0, 1]``.
    """
    if not topics:
        return 0.0
    unique_terms: set[str] = set()
    total = 0
    for words in topics:
        w = words[:top_n]
        unique_terms.update(w)
        total += len(w)
    return float(len(unique_terms) / total) if total > 0 else 0.0


def compute_topic_exclusivity(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 2,
    stop_words: str | list[str] | None = "english",
) -> list[float]:
    """Per-topic exclusivity via c-TF-IDF (Bischof & Airoldi 2012 adaptation).

    For each cluster ``k`` with its top-n c-TF-IDF terms, exclusivity is the
    mean, over those terms, of ``ctfidf[k, term] / sum_j ctfidf[j, term]``.
    A term that is highly weighted in *only* cluster ``k`` contributes a
    value near 1; a term that is common across clusters contributes near
    ``1/K``. Topic exclusivity is the mean exclusivity of its top-n terms.

    Reported alongside coherence because coherence alone can be gamed by
    small, redundant topics that share a handful of high-frequency terms.

    Parameters
    ----------
    texts : Raw documents (one per sample).
    labels : (n_samples,) cluster assignments.
    top_n : Number of top terms per cluster to score.
    ngram_range : N-gram sizes for c-TF-IDF.
    min_df : Minimum document frequency for the c-TF-IDF vocabulary.
    stop_words : Passed through to ``CountVectorizer``.

    Returns
    -------
    List of exclusivity values in ``[0, 1]``, one per cluster (sorted by
    cluster ID ascending).
    """
    from reap.labeling import build_ctfidf_matrix

    ctfidf, _, cluster_ids = build_ctfidf_matrix(
        texts, labels, ngram_range, min_df, 0.95, stop_words
    )
    col_sums = ctfidf.sum(axis=0)

    per_topic: list[float] = []
    for i in range(len(cluster_ids)):
        row = ctfidf[i]
        # top-n indices by c-TF-IDF score (descending), positive scores only
        top_idx = np.argsort(row)[::-1][:top_n]
        vals: list[float] = []
        for v in top_idx:
            if row[v] <= 0:
                break
            if col_sums[v] > 1e-12:
                vals.append(float(row[v] / col_sums[v]))
        per_topic.append(float(np.mean(vals)) if vals else 0.0)
    return per_topic


# ---------------------------------------------------------------------------
# Cluster persistence — Hungarian-matched Jaccard across per-seed clusterings
# ---------------------------------------------------------------------------


def compute_cluster_persistence(
    consensus_labels: np.ndarray,
    seed_labels_list: list[np.ndarray],
    jaccard_threshold: float = 0.5,
) -> dict[str, float | dict[int, float]]:
    """Per-cluster persistence across per-seed clusterings.

    For each consensus cluster ``c``, compute the fraction of seed
    clusterings that contain a Hungarian-matched counterpart with Jaccard
    overlap ``>= jaccard_threshold``. Answers the user-facing question
    "which topics are stable across seeds?" in a way ARI alone does not: a
    dataset with one stable core topic and many unstable fringe topics will
    show that contrast in per-cluster persistence while ARI reports a single
    aggregate number.

    Parameters
    ----------
    consensus_labels : (n_samples,) consensus cluster labels.
    seed_labels_list : List of (n_samples,) per-seed cluster labelings.
    jaccard_threshold : Minimum Jaccard overlap for a seed-cluster to count
        as a "match" for a consensus cluster.

    Returns
    -------
    Dict with ``per_cluster`` (mapping cluster id -> persistence in
    ``[0, 1]``), ``mean``, ``min``, ``max``, ``median``.
    """
    from scipy.optimize import linear_sum_assignment

    consensus_ids = np.unique(consensus_labels)
    n_consensus = len(consensus_ids)
    n_seeds = len(seed_labels_list)

    match_counts: dict[int, int] = {int(c): 0 for c in consensus_ids}

    if n_consensus == 0 or n_seeds == 0:
        return {
            "per_cluster": {int(c): 0.0 for c in consensus_ids},
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    # Precompute consensus cluster masks
    consensus_masks: list[np.ndarray] = [
        (consensus_labels == c) for c in consensus_ids
    ]

    for seed_labels in seed_labels_list:
        seed_ids = np.unique(seed_labels)
        n_seed = len(seed_ids)
        if n_seed == 0:
            continue

        seed_masks = [(seed_labels == s) for s in seed_ids]

        # Jaccard(C, S) = |C & S| / |C | S|
        j_matrix = np.zeros((n_consensus, n_seed), dtype=np.float64)
        for ci, c_mask in enumerate(consensus_masks):
            for sj, s_mask in enumerate(seed_masks):
                inter = int((c_mask & s_mask).sum())
                if inter == 0:
                    continue
                union = int((c_mask | s_mask).sum())
                j_matrix[ci, sj] = inter / union if union > 0 else 0.0

        # Hungarian: maximize total Jaccard by negating
        row_ind, col_ind = linear_sum_assignment(-j_matrix)
        for r, c in zip(row_ind, col_ind):
            if j_matrix[r, c] >= jaccard_threshold:
                match_counts[int(consensus_ids[r])] += 1

    per_cluster = {k: v / n_seeds for k, v in match_counts.items()}
    values = np.asarray(list(per_cluster.values()))
    return {
        "per_cluster": per_cluster,
        "mean": float(values.mean()) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
    }
