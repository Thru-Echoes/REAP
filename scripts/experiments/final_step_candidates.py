"""Candidate replacements for REAP's final clustering step (research code).

REAP's current final step maps the averaged distance matrix with one UMAP run
(seed 42), picks K by silhouette, and fits k-means once (``n_init="auto"``,
a single k-means++ start). The candidates below are the ones registered in
the final-step amendment draft (``docs/proposals/``). None of them is used by
the package; REAP's default does not change until an amendment is ratified.

- ``a1`` average linkage on the averaged matrix, K by silhouette on that matrix;
- ``a2`` k-medoids on the averaged matrix (deterministic greedy start, then
  alternating updates), K by silhouette on that matrix;
- ``b``  co-association over several final maps: each map gets the current
  K rule and k-means, the share of maps that put two documents together is
  turned into a distance, and average linkage cuts it at the median K;
- ``c``  K by agreement: for each K, k-means on several final maps, K is the
  one whose labelings agree most (mean pairwise ARI); labels are the seed-42
  map's k-means at that K;
- ``d``  the current step with ten k-means starts instead of one, for the
  K search and the final fit (applied to every method, since the clustering
  harness is shared).

Every function is pure apart from seeded randomness passed in explicitly.
Ties between candidate K values go to the smaller K, as in
``reap.clustering.find_best_k``.

Exports: ``CandidateResult``, ``get_average_linkage_labels``,
``get_kmedoids_build_order``, ``get_kmedoids_labels``, ``run_candidate_a1``,
``run_candidate_a2``, ``get_coassociation_distance``, ``run_candidate_b``,
``run_candidate_c``, ``get_kmeans_restart_labels``, ``get_k_search_restarts``,
``run_candidate_d``.
"""

from __future__ import annotations

import numpy as np
from final_step import get_best_and_second, get_k_search, get_kmeans_labels, get_pairwise_ari
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Number of k-means starts for candidate d. sklearn's own default before
# n_init="auto" was 10; it is the smallest common choice that is not one start.
KMEANS_RESTARTS = 10
# Cap on alternating k-medoids updates; each update cannot raise the total
# distance, so the loop ends well before this on real data. Reaching it raises.
KMEDOIDS_MAX_ITER = 300


class CandidateResult(BaseModel):
    """Labels and the K-rule trace of one candidate on one averaged matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    candidate: str = Field(..., description="Candidate id: a1, a2, b, c or d.")
    k: int = Field(..., description="K the candidate chose.")
    labels: np.ndarray = Field(..., description="(N,) integer labels at that K.")
    k_scores: dict[int, float] = Field(
        ..., description="The candidate's K-rule score at each candidate K."
    )
    k_rule: str = Field(..., description="Plain statement of the K rule used.")


def _get_condensed(D: np.ndarray) -> np.ndarray:
    """Condensed form of a symmetric distance matrix with zero diagonal."""
    return squareform(D, checks=False)


def get_average_linkage_labels(D: np.ndarray, k_range: list[int]) -> dict[int, np.ndarray]:
    """Average-linkage tree on ``D`` cut at each K (labels 0..K-1).

    The tree is cut by merge order (the last K-1 merges are undone), which
    gives exactly K clusters even when merge heights tie, as they do for
    co-association distances. Deterministic. Raises ``ValueError`` if a cut
    still does not have K clusters.
    """
    tree = linkage(_get_condensed(D), method="average")
    cuts = cut_tree(tree, n_clusters=list(k_range))
    out: dict[int, np.ndarray] = {}
    for col, k in enumerate(k_range):
        labels = np.asarray(cuts[:, col], dtype=np.int64)
        if len(np.unique(labels)) != k:
            raise ValueError(f"average-linkage cut at K={k} gave {len(np.unique(labels))}")
        out[k] = labels
    return out


def _get_silhouette_on_matrix(D: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette of labels with ``D`` as the precomputed distance."""
    return float(silhouette_score(D, labels, metric="precomputed"))


def _pick_k(scores: dict[int, float]) -> int:
    """Highest score, ties to the smaller K (insertion order is ascending K)."""
    return get_best_and_second(scores)[0]


def run_candidate_a1(D: np.ndarray, k_range: list[int]) -> CandidateResult:
    """Candidate a1: average linkage on the averaged matrix, K by silhouette on it."""
    cuts = get_average_linkage_labels(D, k_range)
    scores = {k: _get_silhouette_on_matrix(D, lab) for k, lab in cuts.items()}
    k = _pick_k(scores)
    return CandidateResult(candidate="a1", k=k, labels=cuts[k], k_scores=scores,
                           k_rule="silhouette on the averaged matrix")


def get_kmedoids_build_order(D: np.ndarray, k_max: int) -> list[int]:
    """Greedy medoid order: the first K entries are the greedy start for K.

    The first medoid minimises the total distance to all points; each next
    one is the point that lowers that total the most. Ties go to the lowest
    index. Deterministic.
    """
    n = D.shape[0]
    if k_max > n:
        raise ValueError(f"k_max={k_max} exceeds n={n}")
    order = [int(np.argmin(D.sum(axis=1)))]
    nearest = D[:, order[0]].copy()
    for _ in range(1, k_max):
        gain = np.maximum(nearest[:, None] - D, 0.0).sum(axis=0)
        gain[order] = -np.inf
        nxt = int(np.argmax(gain))
        order.append(nxt)
        nearest = np.minimum(nearest, D[:, nxt])
    return order


def get_kmedoids_labels(D: np.ndarray, start: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Alternating k-medoids from a given start; returns (labels, medoids).

    Each round assigns every point to its nearest medoid, then moves each
    medoid to the member with the smallest total distance to its cluster.
    Stops when the medoids stop changing; raises ``RuntimeError`` if that
    has not happened after ``KMEDOIDS_MAX_ITER`` rounds.
    """
    medoids = np.asarray(start, dtype=np.int64)
    for _ in range(KMEDOIDS_MAX_ITER):
        labels = np.argmin(D[:, medoids], axis=1)
        updated = medoids.copy()
        for j in range(len(medoids)):
            members = np.flatnonzero(labels == j)
            within = D[np.ix_(members, members)].sum(axis=1)
            updated[j] = members[int(np.argmin(within))]
        if np.array_equal(updated, medoids):
            return np.asarray(labels, dtype=np.int64), medoids
        medoids = updated
    raise RuntimeError(f"k-medoids did not converge in {KMEDOIDS_MAX_ITER} rounds")


def run_candidate_a2(D: np.ndarray, k_range: list[int]) -> CandidateResult:
    """Candidate a2: k-medoids on the averaged matrix, K by silhouette on it."""
    order = get_kmedoids_build_order(D, max(k_range))
    by_k = {k: get_kmedoids_labels(D, order[:k])[0] for k in k_range}
    scores = {k: _get_silhouette_on_matrix(D, lab) for k, lab in by_k.items()}
    k = _pick_k(scores)
    return CandidateResult(candidate="a2", k=k, labels=by_k[k], k_scores=scores,
                           k_rule="silhouette on the averaged matrix")


def get_coassociation_distance(label_runs: list[np.ndarray]) -> np.ndarray:
    """One minus the share of runs that put each pair of points together."""
    n = label_runs[0].shape[0]
    together = np.zeros((n, n), dtype=np.float64)
    for labels in label_runs:
        together += labels[:, None] == labels[None, :]
    distance = 1.0 - together / len(label_runs)
    np.fill_diagonal(distance, 0.0)
    return distance


def run_candidate_b(
    maps: dict[int, np.ndarray], k_range: list[int]
) -> CandidateResult:
    """Candidate b: co-association over the final maps, cut at the median K.

    Each map (keyed by its seed) gets the current K rule and k-means with the
    same seed. The median of their K values (the lower middle value for an
    even count) is cut from the average-linkage tree of the co-association
    distance. The scores reported are the number of maps that chose each K.
    """
    per_map_k: list[int] = []
    runs: list[np.ndarray] = []
    for seed, emb in maps.items():
        k = _pick_k(get_k_search(emb, k_range, seed))
        per_map_k.append(k)
        runs.append(get_kmeans_labels(emb, k, seed))
    k_median = int(sorted(per_map_k)[(len(per_map_k) - 1) // 2])
    distance = get_coassociation_distance(runs)
    labels = get_average_linkage_labels(distance, [k_median])[k_median]
    counts = {k: float(per_map_k.count(k)) for k in k_range}
    return CandidateResult(candidate="b", k=k_median, labels=labels, k_scores=counts,
                           k_rule="median of the per-map silhouette K")


def run_candidate_c(
    maps: dict[int, np.ndarray], k_range: list[int], display_seed: int
) -> CandidateResult:
    """Candidate c: K whose k-means labelings agree most across the final maps.

    For each K, k-means on each map with that map's seed; the score is the
    mean pairwise ARI of those labelings. Labels are the display map's
    k-means at the chosen K.
    """
    scores: dict[int, float] = {}
    at_k: dict[int, dict[int, np.ndarray]] = {}
    for k in k_range:
        at_k[k] = {seed: get_kmeans_labels(emb, k, seed) for seed, emb in maps.items()}
        scores[k] = float(np.mean(get_pairwise_ari(list(at_k[k].values()))))
    k = _pick_k(scores)
    return CandidateResult(candidate="c", k=k, labels=at_k[k][display_seed], k_scores=scores,
                           k_rule="mean pairwise ARI across final maps")


def get_kmeans_restart_labels(
    embedding: np.ndarray, k: int, seed: int, n_init: int = KMEANS_RESTARTS
) -> np.ndarray:
    """K-means labels with several starts (the best of ``n_init`` fits is kept)."""
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)  # pyright: ignore[reportArgumentType]
    return np.asarray(model.fit_predict(embedding), dtype=np.int64)


def get_k_search_restarts(
    embedding: np.ndarray, k_range: list[int], seed: int
) -> dict[int, float]:
    """Silhouette at each K with ``KMEANS_RESTARTS`` k-means starts (candidate d)."""
    scores: dict[int, float] = {}
    for k in k_range:
        labels = get_kmeans_restart_labels(embedding, k, seed)
        scores[k] = float(silhouette_score(embedding, labels, metric="euclidean"))
    return scores


def run_candidate_d(embedding: np.ndarray, k_range: list[int], seed: int) -> CandidateResult:
    """Candidate d: the current K rule and k-means, with ten starts instead of one.

    Applies to any consensus map, so the baselines are re-clustered the same way.
    """
    scores = get_k_search_restarts(embedding, k_range, seed)
    k = _pick_k(scores)
    labels = get_kmeans_restart_labels(embedding, k, seed)
    return CandidateResult(candidate="d", k=k, labels=labels,
                           k_scores=scores, k_rule="silhouette, ten k-means starts")
