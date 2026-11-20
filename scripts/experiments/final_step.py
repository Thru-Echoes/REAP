"""Shared helpers for experiments on REAP's final step (UMAP map, K rule, k-means).

REAP averages the pairwise distance matrices of many seeded UMAP runs and then
runs one final UMAP on that average, picks K by silhouette, and clusters with
k-means. The experiments in this folder hold the averaged matrix fixed and vary
only that final step, so these helpers reproduce it exactly as
``reap.benchmarks`` does (same functions, same defaults) while letting the
caller choose the seeds.

Nothing here changes REAP's default behaviour; it is research code under
``scripts/`` and imports the package read-only.

Exports: ``BENCHMARK_FINAL_SEED``, ``FinalStepConfig``, ``FinalStepRun``,
``get_final_embedding``, ``get_k_search``, ``get_kmeans_labels``,
``get_pairwise_ari``, ``get_ari_summary``, ``get_cross_ari``,
``get_file_sha256``.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from reap.clustering import find_best_k
from reap.consensus import get_consensus_embedding

# The seed reap.benchmarks uses for the final UMAP, the K search and k-means
# (the default ``random_state`` of get_consensus_embedding, find_best_k and
# _cluster_embedding). A run with this seed everywhere reproduces the benchmark.
BENCHMARK_FINAL_SEED = 42


class FinalStepConfig(BaseModel):
    """Settings of REAP's final step for one dataset, as the benchmark used them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_components: int = Field(..., ge=1, description="Dimensions of the final UMAP map.")
    n_neighbors: int = Field(..., ge=2, description="UMAP neighbourhood size.")
    min_dist: float = Field(..., ge=0.0, description="UMAP minimum distance.")
    k_range: list[int] = Field(..., min_length=1, description="Candidate K values.")


class FinalStepRun(BaseModel):
    """One final-step run on a fixed averaged distance matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str = Field(..., description="Which seeds vary: joint, map_only or cluster_only.")
    map_seed: int = Field(..., description="Seed of the final UMAP.")
    cluster_seed: int = Field(..., description="Seed of k-means in the K search and the final fit.")
    k: int = Field(..., description="K chosen by the silhouette rule.")
    best_silhouette: float = Field(..., description="Silhouette at the chosen K.")
    second_silhouette: float = Field(
        ..., description="Highest silhouette among the other candidate K values."
    )
    second_k: int = Field(..., description="K with the second-highest silhouette.")


def get_final_embedding(
    D_consensus: np.ndarray, config: FinalStepConfig, map_seed: int
) -> np.ndarray:
    """Run REAP's final UMAP on a fixed averaged distance matrix.

    Inputs: the (N, N) averaged distance matrix, the final-step settings, and
    the UMAP seed. Output: the (N, n_components) map. Calls
    ``reap.consensus.get_consensus_embedding`` unchanged.
    Side effects: none beyond UMAP's own seeded random state.
    """
    embedding, _ = get_consensus_embedding(
        D_consensus,
        config.n_components,
        config.n_neighbors,
        config.min_dist,
        random_state=map_seed,
    )
    return embedding


def get_k_search(
    embedding: np.ndarray, k_range: list[int], cluster_seed: int
) -> dict[int, float]:
    """Silhouette at each candidate K, as ``reap.clustering.find_best_k`` computes it.

    Inputs: the map, candidate K values, and the k-means seed. Output: a
    mapping from K to silhouette (euclidean, the benchmark's setting).
    Side effects: none.
    """
    return dict(find_best_k(embedding, k_range, random_state=cluster_seed).scores)


def get_kmeans_labels(embedding: np.ndarray, k: int, cluster_seed: int) -> np.ndarray:
    """K-means labels exactly as ``reap.benchmarks._cluster_embedding`` fits them.

    Inputs: the map, K, and the k-means seed. Output: (N,) integer labels.
    Side effects: none.
    """
    model = KMeans(n_clusters=k, random_state=cluster_seed, n_init="auto")
    return np.asarray(model.fit_predict(embedding))


def get_best_and_second(scores: dict[int, float]) -> tuple[int, float, int, float]:
    """Return (best K, its silhouette, second-best K, its silhouette).

    Ties go to the smaller K, matching ``max`` over an insertion-ordered dict in
    ``find_best_k``. Raises ``ValueError`` with fewer than two candidates.
    """
    if len(scores) < 2:
        raise ValueError("need at least two candidate K values to report a runner-up")
    best_k = max(scores, key=lambda k: scores[k])
    rest = {k: v for k, v in scores.items() if k != best_k}
    second_k = max(rest, key=lambda k: rest[k])
    return best_k, scores[best_k], second_k, rest[second_k]


def get_pairwise_ari(label_runs: list[np.ndarray]) -> list[float]:
    """ARI for every unordered pair of runs (n * (n - 1) / 2 values)."""
    return [
        float(adjusted_rand_score(a, b)) for a, b in itertools.combinations(label_runs, 2)
    ]


def get_cross_ari(
    runs_a: list[np.ndarray], runs_b: list[np.ndarray], matched: bool
) -> list[float]:
    """ARI between runs on two different matrices.

    ``matched=True`` pairs run i with run i (same final seeds on both sides);
    ``matched=False`` pairs every i with every j != i.
    """
    if len(runs_a) != len(runs_b):
        raise ValueError("both sides need the same number of runs")
    if matched:
        return [float(adjusted_rand_score(a, b)) for a, b in zip(runs_a, runs_b)]
    return [
        float(adjusted_rand_score(runs_a[i], runs_b[j]))
        for i in range(len(runs_a))
        for j in range(len(runs_b))
        if i != j
    ]


def get_ari_summary(values: list[float]) -> dict[str, float | int]:
    """Mean, median, minimum, maximum and count of a list of ARI values."""
    if not values:
        raise ValueError("no ARI values to summarise")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": int(arr.size),
    }


def get_file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes. Side effects: reads the file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
