"""Choosing K and choosing settings, with every choice declared (no defaults).

These functions follow the rules of REAP's two predecessor studies, so a
search scores settings the way those studies did:

- **Trustworthiness at a fixed neighbourhood.** Each setting's map is scored
  with the same trustworthiness neighbourhood (``trust_n_neighbors``), not
  with the setting's own ``n_neighbors``, which would favour small
  neighbourhoods.
- **K by separation and stability together.** For every K in a declared
  range, the consensus map and every per-seed map are clustered at that same
  K. A K whose smallest consensus cluster is below ``min_cluster_size`` is
  dropped. Over the remaining K, silhouette and the mean seed-to-consensus
  ARI are min-max scaled, and K maximises
  ``silhouette_weight * scaled silhouette + stability_weight * scaled ARI``.
  (Trustworthiness cannot choose K: it depends on the map, not the labels.)
- **Stability at a common K.** Seed-to-consensus and seed-to-seed agreement
  are measured with every seed clustered at the chosen K.
- **Settings on scaled metrics, confirmed on a second draw.** Metrics are
  min-max scaled across the evaluated settings before they are weighted, an
  optional trustworthiness floor removes settings first, and the top settings
  of one seed draw are re-scored on a second draw and chosen by their mean.

Silhouette is computed with a declared metric: ``"euclidean"``,
``"cosine"``, any metric name scikit-learn accepts, or a callable. Ties
between K values go to the smaller K; ties between settings to the earlier
one in the input order.

The older ``reap.search`` functions are unchanged.

Exports: ``SilhouetteMetric``, ``KScore``, ``KSelection``, ``SettingEvaluation``,
``RankedSetting``, ``find_k_by_separation_and_stability``,
``compute_stability_at_k``, ``evaluate_setting``, ``rank_settings``,
``select_setting_with_second_draw``.

Example
-------
>>> selection = find_k_by_separation_and_stability(  # doctest: +SKIP
...     consensus_embedding, seed_embeddings, range(5, 31), min_cluster_size=10,
...     silhouette_metric="euclidean", silhouette_weight=0.5, stability_weight=0.5,
...     kmeans_n_init=10, random_state=seed)
>>> selection.best_k  # doctest: +SKIP
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.evaluation import compute_trustworthiness

logger = logging.getLogger(__name__)

SilhouetteMetric = str | Callable[[np.ndarray, np.ndarray], float]
SETTING_METRICS = ("trustworthiness", "silhouette", "seed_to_consensus_ari", "seed_to_seed_ari")


class KScore(BaseModel):
    """One candidate K: its raw and scaled scores and whether it passed the floor."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(..., description="Number of clusters.")
    smallest_cluster: int = Field(..., description="Size of the smallest consensus cluster.")
    kept: bool = Field(..., description="Whether the smallest cluster meets the floor.")
    silhouette: float = Field(..., description="Silhouette of the consensus labels.")
    seed_to_consensus_ari: float = Field(
        ..., description="Mean ARI between each seed's labels and the consensus labels at K."
    )
    scaled_silhouette: float | None = Field(
        default=None, description="Min-max scaled silhouette over the kept K (None if dropped)."
    )
    scaled_seed_to_consensus_ari: float | None = Field(
        default=None, description="Min-max scaled ARI over the kept K (None if dropped)."
    )
    score: float | None = Field(default=None, description="Weighted score (None if dropped).")


class KSelection(BaseModel):
    """The chosen K, every candidate's scores, and the declared choices behind them."""

    model_config = ConfigDict(frozen=True)

    best_k: int = Field(..., description="The chosen K.")
    scores: list[KScore] = Field(..., description="Every candidate K, in the order tried.")
    min_cluster_size: int = Field(..., description="Floor on the smallest consensus cluster.")
    silhouette_metric: str = Field(..., description="Name of the silhouette metric used.")
    silhouette_weight: float = Field(..., description="Weight of the scaled silhouette.")
    stability_weight: float = Field(..., description="Weight of the scaled stability.")
    kmeans_n_init: int = Field(..., description="k-means starts per fit.")
    random_state: int = Field(..., description="Seed of every k-means fit.")


class SettingEvaluation(BaseModel):
    """One UMAP setting scored under the declared rules."""

    model_config = ConfigDict(frozen=True)

    n_components: int = Field(..., description="UMAP dimensions.")
    n_neighbors: int = Field(..., description="UMAP neighbourhood size.")
    min_dist: float = Field(..., description="UMAP minimum distance.")
    trustworthiness: float = Field(..., description="Trustworthiness at trust_n_neighbors.")
    trust_n_neighbors: int = Field(..., description="Neighbourhood of the trustworthiness.")
    k_selection: KSelection = Field(..., description="How K was chosen for this setting.")
    best_k: int = Field(..., description="The chosen K.")
    silhouette: float = Field(..., description="Consensus silhouette at the chosen K.")
    seed_to_consensus_ari: float = Field(..., description="Mean seed-to-consensus ARI at K.")
    seed_to_seed_ari: float = Field(..., description="Mean pairwise seed-to-seed ARI at K.")
    seeds: list[int] = Field(..., description="The per-seed UMAP seeds of this draw.")


class RankedSetting(BaseModel):
    """A setting's scaled metrics and score within one ranking."""

    model_config = ConfigDict(frozen=True)

    evaluation: SettingEvaluation = Field(..., description="The setting and its raw metrics.")
    scaled: dict[str, float] = Field(..., description="Min-max scaled metric values.")
    score: float = Field(..., description="Weighted sum of the scaled metrics.")


def _get_metric_name(metric: SilhouetteMetric) -> str:
    """A printable name for a silhouette metric."""
    return metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")


def _get_min_max_scaled(values: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1]; all-equal values scale to 1."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _get_kmeans_labels(X: np.ndarray, k: int, n_init: int, random_state: int) -> np.ndarray:
    """k-means labels at K with an explicit number of starts and seed."""
    model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)  # pyright: ignore[reportArgumentType]
    return np.asarray(model.fit_predict(X), dtype=np.int64)


def _check_weights(*weights: float) -> None:
    """Weights must be non-negative and not all zero."""
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"weights must be non-negative and not all zero, got {weights}")


def find_k_by_separation_and_stability(
    consensus_embedding: np.ndarray,
    seed_embeddings: Sequence[np.ndarray],
    k_range: Iterable[int],
    *,
    min_cluster_size: int,
    silhouette_metric: SilhouetteMetric,
    silhouette_weight: float,
    stability_weight: float,
    kmeans_n_init: int,
    random_state: int,
) -> KSelection:
    """Choose K by scaled silhouette and scaled seed-to-consensus ARI, after a size floor.

    Inputs: the consensus map, the per-seed maps of the same documents, the
    candidate K values, and every declared choice (keyword-only, no defaults).
    Output: a ``KSelection`` with every candidate's scores. Raises
    ``ValueError`` when no K passes the floor or the weights are invalid.
    Side effects: none.
    """
    _check_weights(silhouette_weight, stability_weight)
    rows: list[dict[str, float | int | bool]] = []
    for k in k_range:
        if k < 2 or k >= consensus_embedding.shape[0]:
            raise ValueError(f"K={k} is outside 2..n-1 for {consensus_embedding.shape[0]} rows")
        consensus_labels = _get_kmeans_labels(consensus_embedding, k, kmeans_n_init, random_state)
        smallest = int(np.bincount(consensus_labels).min())
        sil = float(silhouette_score(consensus_embedding, consensus_labels,
                                     metric=silhouette_metric))  # pyright: ignore[reportArgumentType]
        ari = float(np.mean([
            adjusted_rand_score(_get_kmeans_labels(e, k, kmeans_n_init, random_state),
                                consensus_labels)
            for e in seed_embeddings]))
        rows.append({"k": k, "smallest_cluster": smallest, "kept": smallest >= min_cluster_size,
                     "silhouette": sil, "seed_to_consensus_ari": ari})
    kept = [r for r in rows if r["kept"]]
    if not kept:
        raise ValueError(f"no K in the range has a smallest cluster of {min_cluster_size} or more")
    s_sil = _get_min_max_scaled([float(r["silhouette"]) for r in kept])
    s_ari = _get_min_max_scaled([float(r["seed_to_consensus_ari"]) for r in kept])
    scored = {int(r["k"]): (a, b, silhouette_weight * a + stability_weight * b)
              for r, a, b in zip(kept, s_sil, s_ari)}
    best_k = max(scored, key=lambda k: (scored[k][2], -k))
    scores = [KScore(**{**r, **({"scaled_silhouette": scored[int(r["k"])][0],
                                 "scaled_seed_to_consensus_ari": scored[int(r["k"])][1],
                                 "score": scored[int(r["k"])][2]} if r["kept"] else {})})
              for r in rows]
    return KSelection(best_k=best_k, scores=scores, min_cluster_size=min_cluster_size,
                      silhouette_metric=_get_metric_name(silhouette_metric),
                      silhouette_weight=silhouette_weight, stability_weight=stability_weight,
                      kmeans_n_init=kmeans_n_init, random_state=random_state)


def compute_stability_at_k(
    consensus_embedding: np.ndarray, seed_embeddings: Sequence[np.ndarray], k: int, *,
    kmeans_n_init: int, random_state: int,
) -> tuple[float, float]:
    """Mean seed-to-consensus ARI and mean pairwise seed-to-seed ARI, all at the same K.

    Side effects: none.
    """
    consensus_labels = _get_kmeans_labels(consensus_embedding, k, kmeans_n_init, random_state)
    seed_labels = [_get_kmeans_labels(e, k, kmeans_n_init, random_state) for e in seed_embeddings]
    s2c = float(np.mean([adjusted_rand_score(s, consensus_labels) for s in seed_labels]))
    s2s = float(np.mean([adjusted_rand_score(a, b)
                         for a, b in itertools.combinations(seed_labels, 2)]))
    return s2c, s2s


def evaluate_setting(
    X: np.ndarray,
    seeds: list[int],
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    *,
    k_range: Iterable[int],
    input_metric: str,
    trust_n_neighbors: int,
    min_cluster_size: int,
    silhouette_metric: SilhouetteMetric,
    silhouette_weight: float,
    stability_weight: float,
    kmeans_n_init: int,
    final_map_seed: int,
    kmeans_seed: int,
) -> SettingEvaluation:
    """Build REAP's map for one setting and score it under the declared rules.

    Inputs: the embeddings, one seed draw, the UMAP setting, and every
    declared choice (keyword-only, no defaults). Output: a
    ``SettingEvaluation``. Side effects: UMAP's seeded random state, logging.
    """
    if len(seeds) < 2:
        raise ValueError("a setting needs at least two per-seed maps")
    maps = get_multi_seed_embeddings(X, seeds, n_components, n_neighbors, min_dist, input_metric)
    D = get_consensus_distance_matrix(maps)
    consensus, _ = get_consensus_embedding(D, n_components, n_neighbors, min_dist,
                                           random_state=final_map_seed)
    trust = compute_trustworthiness(X, consensus, n_neighbors=trust_n_neighbors,
                                    metric=input_metric)
    selection = find_k_by_separation_and_stability(
        consensus, maps, k_range, min_cluster_size=min_cluster_size,
        silhouette_metric=silhouette_metric, silhouette_weight=silhouette_weight,
        stability_weight=stability_weight, kmeans_n_init=kmeans_n_init,
        random_state=kmeans_seed)
    s2c, s2s = compute_stability_at_k(consensus, maps, selection.best_k,
                                      kmeans_n_init=kmeans_n_init, random_state=kmeans_seed)
    best = next(s for s in selection.scores if s.k == selection.best_k)
    logger.info("setting %d/%d/%.4f: trust %.3f, K %d", n_components, n_neighbors, min_dist,
                trust, selection.best_k)
    return SettingEvaluation(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
        trustworthiness=trust, trust_n_neighbors=trust_n_neighbors, k_selection=selection,
        best_k=selection.best_k, silhouette=best.silhouette, seed_to_consensus_ari=s2c,
        seed_to_seed_ari=s2s, seeds=list(seeds))


def rank_settings(
    evaluations: Sequence[SettingEvaluation], *,
    metric_weights: dict[str, float],
    trustworthiness_floor: Literal["median"] | float | None,
) -> list[RankedSetting]:
    """Rank settings on min-max scaled metrics, after an optional trustworthiness floor.

    ``metric_weights`` maps any of ``SETTING_METRICS`` to a weight. The floor
    is a number, ``"median"`` (the median trustworthiness of ``evaluations``),
    or None. Returns the passing settings best first. Raises ``ValueError``
    on an unknown metric, invalid weights, or when no setting passes.
    """
    unknown = set(metric_weights) - set(SETTING_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {SETTING_METRICS}")
    _check_weights(*metric_weights.values())
    if trustworthiness_floor == "median":
        floor = float(np.median([e.trustworthiness for e in evaluations]))
    else:
        floor = trustworthiness_floor
    passing = [e for e in evaluations if floor is None or e.trustworthiness >= floor]
    if not passing:
        raise ValueError("no setting passes the trustworthiness floor")
    scaled = {m: _get_min_max_scaled([float(getattr(e, m)) for e in passing])
              for m in metric_weights}
    ranked = [RankedSetting(evaluation=e, scaled={m: scaled[m][i] for m in metric_weights},
                            score=sum(w * scaled[m][i] for m, w in metric_weights.items()))
              for i, e in enumerate(passing)]
    order = sorted(range(len(ranked)), key=lambda i: (-ranked[i].score, i))
    return [ranked[i] for i in order]


def select_setting_with_second_draw(
    first_draw: Sequence[SettingEvaluation],
    second_draw: Sequence[SettingEvaluation],
    *,
    top_n: int,
    metric_weights: dict[str, float],
    trustworthiness_floor: Literal["median"] | float | None,
) -> tuple[SettingEvaluation, list[dict[str, float]]]:
    """Re-score the first draw's top settings on a second seed draw; choose by the mean.

    Both draws are ranked with the same weights and floor. Among the first
    draw's ``top_n``, the chosen setting has the highest mean of its two
    scores; a top setting missing from, or removed by the floor in, the second
    draw raises ``ValueError``. Returns the chosen setting (first-draw
    evaluation) and a table of each candidate's two scores and mean.
    """
    first = rank_settings(first_draw, metric_weights=metric_weights,
                          trustworthiness_floor=trustworthiness_floor)[:top_n]
    second = {(r.evaluation.n_components, r.evaluation.n_neighbors, r.evaluation.min_dist): r
              for r in rank_settings(second_draw, metric_weights=metric_weights,
                                     trustworthiness_floor=trustworthiness_floor)}
    table: list[dict[str, float]] = []
    for r in first:
        key = (r.evaluation.n_components, r.evaluation.n_neighbors, r.evaluation.min_dist)
        if key not in second:
            raise ValueError(f"top setting {key} is missing from the second draw's ranking")
        table.append({"n_components": key[0], "n_neighbors": key[1], "min_dist": key[2],
                      "first_score": r.score, "second_score": second[key].score,
                      "mean_score": (r.score + second[key].score) / 2})
    best = max(range(len(table)), key=lambda i: (table[i]["mean_score"], -i))
    return first[best].evaluation, table
