"""Hold-out comparison harness: every method, every training seed, one fixed split.

Protocol §18 asks how well each method places documents it never saw into a
reference map built from the other documents. For one hold-out split (one
"cell" of the run matrix) this module:

1. trains each trained method once per training seed against the one in-set
   consensus — the REAP MLP projection head, the linear head, and parametric
   UMAP — and runs the deterministic identity baseline once (protocol v1.6
   clause b);
2. computes the four registered hold-out test metrics on the held-out
   documents for every (method, seed) run (protocol §18.e, recipes below);
3. optionally computes the pre-designated paired subsample intervals that
   report generalization uncertainty on run 1 (v1.6 clause b, item 6).

Paired significance tests and the Holm correction are *not* computed here:
the multiple-comparison families span many cells, so they belong to the
aggregation step that reads every cell.

Metric recipes (each held-out metric compares the held-out documents' raw
embeddings ``X_out`` with their placement ``Y_out`` by one method):

- ``trustworthiness`` — ``reap.evaluation.compute_trustworthiness(X_out,
  Y_out)``, k = ``n_neighbors`` (15), cosine in the raw space.
- ``distance_correlation`` — ``reap.evaluation.compute_distance_correlation(
  X_out, Y_out)``: Pearson r between condensed Euclidean pairwise-distance
  vectors of raw and placed held-out documents (registered recipe
  ``distance_correlation``; raw-vs-placed, as §18.e defines it — not the
  target-vs-prediction comparison inside ``train_projection_head``).
- ``silhouette_holdout`` — Euclidean silhouette of ``Y_out`` under
  nearest-centroid labels, the centroids being the method's own in-set
  placement averaged per in-set cluster.
- ``ari_adapted_stage10`` — adjusted Rand index between a KMeans partition of
  ``Y_out`` (K = the in-set cluster count, fixed once per cell; KMeans seed
  ``METRIC_KMEANS_RANDOM_STATE``, separate from every training seed) and the
  caller-supplied reference partition of the held-out documents from the
  full-data rerun.

A metric that cannot be computed is recorded as ``None`` with a reason, never
as a sentinel number. A held-out set smaller than the v1.5 clause (h) floors
(n ≥ 2k + 2 for trustworthiness at k neighbours, and n ≥ 3K) leaves every
metric undefined with reason "fold too small".

Exports: ``HOLDOUT_METRIC_IDS``, ``TRAINED_METHODS``, ``DETERMINISTIC_METHODS``,
``METRIC_KMEANS_RANDOM_STATE``, ``HeadTrainingSettings``,
``ParametricUMAPSettings``, ``HoldoutRunMetrics``, ``HoldoutComparison``,
``HoldoutProjection``, ``SubsampleInterval``, ``get_nearest_centroid_labels``,
``compute_holdout_metrics``, ``run_holdout_comparison``,
``get_paired_subsample_intervals``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from reap.baselines.identity_baseline import get_identity_projection
from reap.evaluation import compute_distance_correlation, compute_trustworthiness

logger = logging.getLogger(__name__)

HOLDOUT_METRIC_IDS: tuple[str, ...] = (
    "trustworthiness",
    "distance_correlation",
    "silhouette_holdout",
    "ari_adapted_stage10",
)
TRAINED_METHODS: tuple[str, ...] = ("reap_mlp", "linear_head", "parametric_umap")
DETERMINISTIC_METHODS: tuple[str, ...] = ("identity",)
ALL_METHODS: tuple[str, ...] = TRAINED_METHODS + DETERMINISTIC_METHODS

# Fixed seed for the KMeans inside the adapted ARI. A separate stream from every
# training seed, so a reported metric does not move with the model's seed
# (same value and rationale as reap.projection's metric clustering).
METRIC_KMEANS_RANDOM_STATE = 42
_METRIC_KMEANS_N_INIT: int = 10

MethodName = Literal["reap_mlp", "linear_head", "parametric_umap", "identity"]


class HeadTrainingSettings(BaseModel):
    """Training settings shared by the MLP and linear projection heads.

    Defaults are the hold-out proposal's settings (§2.1 step 4 of
    ``2026-05-21-temporal-holdout-and-cross-source-protocol.md``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_layers: list[int] = Field(default=[128, 64], description="MLP hidden sizes.")
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0, description="MLP dropout.")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="MSE weight in the loss.")
    n_folds: int = Field(default=5, ge=2, description="Cross-validation folds inside training.")
    batch_size: int = Field(default=64, ge=1, description="Mini-batch size.")
    max_epochs: int = Field(default=500, ge=1, description="Maximum epochs.")
    patience: int = Field(default=50, ge=1, description="Early-stopping patience.")
    lr: float = Field(default=1e-3, gt=0.0, description="Learning rate.")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="L2 regularization.")


class ParametricUMAPSettings(BaseModel):
    """Settings for the parametric UMAP baseline, matched to the cell's REAP run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_components: int = Field(..., ge=1, description="Output dimension (= consensus dim).")
    n_neighbors: int = Field(..., ge=2, description="UMAP neighbourhood size.")
    min_dist: float = Field(..., ge=0.0, description="UMAP min_dist.")
    metric: str = Field(default="cosine", description="Input-space metric.")
    n_epochs: int = Field(default=50, ge=1, description="Training epochs.")
    batch_size: int = Field(default=64, ge=1, description="Mini-batch size.")


class HoldoutRunMetrics(BaseModel):
    """The four hold-out metrics for one (method, training seed) run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: MethodName = Field(..., description="Method id.")
    seed: int | None = Field(
        ..., description="Training seed; None for the deterministic identity baseline."
    )
    values: dict[str, float | None] = Field(
        ..., description="Metric id -> value, None when undefined."
    )
    undefined_reasons: dict[str, str] = Field(
        default_factory=dict, description="Metric id -> why it is undefined."
    )

    @model_validator(mode="after")
    def _check_metrics(self) -> HoldoutRunMetrics:
        """Every registered metric is present; undefined ones say why."""
        if set(self.values) != set(HOLDOUT_METRIC_IDS):
            raise ValueError(f"run must report exactly {HOLDOUT_METRIC_IDS}")
        for metric_id, value in self.values.items():
            if value is None and metric_id not in self.undefined_reasons:
                raise ValueError(f"{metric_id} is undefined without a reason")
            if value is not None and metric_id in self.undefined_reasons:
                raise ValueError(f"{metric_id} has both a value and an undefined reason")
        return self


class HoldoutComparison(BaseModel):
    """Every method's hold-out metrics for one cell, one row per (method, seed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(..., min_length=1, description="Run-matrix cell id, e.g. 'ai_art/T1'.")
    n_in: int = Field(..., ge=1, description="In-set document count.")
    n_out: int = Field(..., ge=1, description="Held-out document count.")
    k_holdout: int = Field(..., ge=2, description="In-set cluster count used for the ARI.")
    n_neighbors: int = Field(..., ge=1, description="Trustworthiness neighbourhood size.")
    training_seeds: list[int] = Field(..., min_length=1, description="Training seeds, in order.")
    methods: list[MethodName] = Field(..., min_length=1, description="Methods run.")
    runs: list[HoldoutRunMetrics] = Field(..., description="One entry per (method, seed).")

    @model_validator(mode="after")
    def _check_complete(self) -> HoldoutComparison:
        """Trained methods have one run per seed; deterministic ones exactly one run."""
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds contains duplicates")
        expected = set()
        for method in self.methods:
            if method in DETERMINISTIC_METHODS:
                expected.add((method, None))
            else:
                expected.update((method, seed) for seed in self.training_seeds)
        observed = [(run.method, run.seed) for run in self.runs]
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError("runs do not match methods x training seeds exactly")
        return self

    def get_metric_column(self, method: str, metric_id: str) -> list[float | None]:
        """Return one method's values of one metric, in training-seed order.

        A deterministic method returns its single value.
        """
        by_seed = {run.seed: run.values[metric_id] for run in self.runs if run.method == method}
        if not by_seed:
            raise KeyError(f"method {method!r} was not run")
        if method in DETERMINISTIC_METHODS:
            return [by_seed[None]]
        return [by_seed[seed] for seed in self.training_seeds]


class HoldoutProjection(BaseModel):
    """One method's run-1 placement, kept for the subsample intervals."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: MethodName = Field(..., description="Method id.")
    seed: int | None = Field(..., description="Training seed of this run (None for identity).")
    Y_out: np.ndarray = Field(..., description="(n_out, d) held-out placement.")
    Y_in_space: np.ndarray = Field(
        ..., description="(n_in, d) the method's in-set placement, for nearest-centroid labels."
    )


class SubsampleInterval(BaseModel):
    """Paired 95% subsample interval of (reference method − baseline) for one metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_id: str = Field(..., description="Metric id.")
    reference_method: str = Field(..., description="Method the difference is taken from.")
    baseline_method: str = Field(..., description="Method subtracted.")
    point_difference: float | None = Field(
        ..., description="Difference on the full held-out set (None if undefined)."
    )
    ci_low: float | None = Field(..., description="2.5th percentile of the differences.")
    ci_high: float | None = Field(..., description="97.5th percentile of the differences.")
    n_resamples: int = Field(..., description="Subsamples drawn.")
    n_defined: int = Field(..., description="Subsamples on which the difference was defined.")
    fraction: float = Field(..., description="Share of held-out units kept per subsample.")


def get_nearest_centroid_labels(
    Y_in_space: np.ndarray, labels_in: np.ndarray, Y_out: np.ndarray
) -> np.ndarray:
    """Label each held-out point with the in-set cluster whose centroid is nearest.

    Centroids are the mean in-set placement of each in-set cluster. Pure function.
    Returns (n_out,) labels drawn from ``labels_in``'s values; ties go to the
    lowest cluster label.
    """
    clusters = np.unique(labels_in)
    centroids = np.stack([Y_in_space[labels_in == c].mean(axis=0) for c in clusters])
    squared = ((Y_out[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return clusters[np.argmin(squared, axis=1)]


def _get_fold_too_small_reason(n_out: int, n_neighbors: int, k_holdout: int) -> str | None:
    """Return why a held-out set is below the v1.5 clause (h) floors, or None."""
    if n_out < 2 * n_neighbors + 2:
        return f"fold too small: n_out={n_out} < 2*{n_neighbors}+2 for trustworthiness"
    if n_out < 3 * k_holdout:
        return f"fold too small: n_out={n_out} < 3*K={3 * k_holdout}"
    return None


def compute_holdout_metrics(
    X_out: np.ndarray,
    Y_out: np.ndarray,
    Y_in_space: np.ndarray,
    labels_in: np.ndarray,
    reference_labels_out: np.ndarray,
    k_holdout: int,
    n_neighbors: int = 15,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Compute the four registered hold-out metrics for one placement.

    Pure apart from KMeans, which uses the fixed ``METRIC_KMEANS_RANDOM_STATE``.
    Recipes are in the module docstring.

    Returns
    -------
    (values, undefined_reasons): metric id -> value (None when undefined), and
    metric id -> reason for each undefined metric.
    """
    reasons: dict[str, str] = {}
    too_small = _get_fold_too_small_reason(X_out.shape[0], n_neighbors, k_holdout)
    if too_small is not None:
        return dict.fromkeys(HOLDOUT_METRIC_IDS), dict.fromkeys(HOLDOUT_METRIC_IDS, too_small)

    values: dict[str, float | None] = {
        "trustworthiness": compute_trustworthiness(X_out, Y_out, n_neighbors=n_neighbors),
        "distance_correlation": compute_distance_correlation(X_out, Y_out),
    }
    nearest = get_nearest_centroid_labels(Y_in_space, labels_in, Y_out)
    if len(np.unique(nearest)) < 2:
        values["silhouette_holdout"] = None
        reasons["silhouette_holdout"] = "all held-out points share one nearest centroid"
    else:
        values["silhouette_holdout"] = float(silhouette_score(Y_out, nearest, metric="euclidean"))
    kmeans = KMeans(
        n_clusters=k_holdout,
        random_state=METRIC_KMEANS_RANDOM_STATE,
        n_init=_METRIC_KMEANS_N_INIT,  # type: ignore[arg-type]  # sklearn stubs say str
    )
    predicted = kmeans.fit_predict(Y_out)
    values["ari_adapted_stage10"] = float(adjusted_rand_score(reference_labels_out, predicted))
    return values, reasons


def _get_head_factory(method: str, settings: HeadTrainingSettings):  # type: ignore[no-untyped-def]
    """Return the ``head_factory`` for ``train_projection_head`` (MLP or linear)."""
    from reap.projection import LinearProjectionHead, ProjectionHead

    if method == "linear_head":
        return lambda in_dim, out_dim: LinearProjectionHead(in_dim, out_dim)
    return lambda in_dim, out_dim: ProjectionHead(
        in_dim, out_dim, settings.hidden_layers, settings.dropout
    )


def _run_trained_method(
    method: str,
    seed: int,
    X_in: np.ndarray,
    Y_in: np.ndarray,
    labels_in: np.ndarray,
    X_out: np.ndarray,
    head_settings: HeadTrainingSettings,
    umap_settings: ParametricUMAPSettings | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one method with one seed; return (Y_out, Y_in_space).

    Side effects: trains a model (torch for the heads, tensorflow for
    parametric UMAP), which sets that framework's random state.
    """
    if method == "parametric_umap":
        if umap_settings is None:
            raise ValueError("parametric_umap requested without umap_settings")
        from reap.baselines.parametric_umap_baseline import run_parametric_umap_holdout

        fitted = run_parametric_umap_holdout(
            X_in,
            X_out,
            n_components=umap_settings.n_components,
            n_neighbors=umap_settings.n_neighbors,
            min_dist=umap_settings.min_dist,
            metric=umap_settings.metric,
            random_state=seed,
            n_epochs=umap_settings.n_epochs,
            batch_size=umap_settings.batch_size,
        )
        return fitted.embedding_out, fitted.embedding_in
    from reap.projection import train_projection_head

    trained = train_projection_head(
        X_in,
        Y_in,
        labels_in,
        hidden_layers=head_settings.hidden_layers,
        dropout=head_settings.dropout,
        alpha=head_settings.alpha,
        n_folds=head_settings.n_folds,
        batch_size=head_settings.batch_size,
        max_epochs=head_settings.max_epochs,
        patience=head_settings.patience,
        lr=head_settings.lr,
        weight_decay=head_settings.weight_decay,
        seed=seed,
        head_factory=_get_head_factory(method, head_settings),
    )
    return np.asarray(trained["model"].forward(X_out), dtype=np.float64), Y_in


def run_holdout_comparison(
    X_in: np.ndarray,
    Y_in: np.ndarray,
    labels_in: np.ndarray,
    X_out: np.ndarray,
    reference_labels_out: np.ndarray,
    training_seeds: Sequence[int],
    run_id: str,
    methods: Sequence[MethodName] = ALL_METHODS,  # type: ignore[assignment]
    head_settings: HeadTrainingSettings | None = None,
    umap_settings: ParametricUMAPSettings | None = None,
    n_neighbors: int = 15,
) -> tuple[HoldoutComparison, dict[str, HoldoutProjection]]:
    """Run every method on one hold-out split and score every run.

    Side effects: trains 30-ish models per trained method (one per seed); logs
    progress. Fails closed: any method error propagates — a cell with a missing
    method is never returned.

    Parameters
    ----------
    X_in : (n_in, d) in-set raw embeddings.
    Y_in : (n_in, c) the in-set consensus embedding (the reference map).
    labels_in : (n_in,) in-set consensus cluster labels; their count fixes K.
    X_out : (n_out, d) held-out raw embeddings.
    reference_labels_out : (n_out,) held-out labels from the full-data rerun,
        the reference partition for ``ari_adapted_stage10``.
    training_seeds : One training seed per run, e.g. Set D (v1.6 clause b).
    run_id : Run-matrix cell id.
    methods : Methods to run (default: all four).
    head_settings : Projection-head settings (default: ``HeadTrainingSettings()``).
    umap_settings : Required when ``parametric_umap`` is in ``methods``.
    n_neighbors : Trustworthiness neighbourhood size.

    Returns
    -------
    (comparison, run-1 projections by method) — the projections feed
    ``get_paired_subsample_intervals``.

    Raises
    ------
    ValueError : On misaligned inputs, an unknown method, or missing settings.
    """
    seeds = [int(s) for s in training_seeds]
    unknown = {str(m) for m in methods} - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if not (X_in.shape[0] == Y_in.shape[0] == labels_in.shape[0]):
        raise ValueError("X_in, Y_in and labels_in must be row-aligned")
    if X_out.shape[0] != reference_labels_out.shape[0]:
        raise ValueError("X_out and reference_labels_out must be row-aligned")
    if not seeds:
        raise ValueError("training_seeds must not be empty")
    head_settings = head_settings or HeadTrainingSettings()
    k_holdout = int(len(np.unique(labels_in)))

    runs: list[HoldoutRunMetrics] = []
    first_runs: dict[str, HoldoutProjection] = {}

    def _record(method: str, seed: int | None, Y_out: np.ndarray, Y_space: np.ndarray) -> None:
        values, reasons = compute_holdout_metrics(
            X_out, Y_out, Y_space, labels_in, reference_labels_out, k_holdout, n_neighbors
        )
        runs.append(
            HoldoutRunMetrics(
                method=method, seed=seed, values=values, undefined_reasons=reasons  # type: ignore[arg-type]
            )
        )
        if method not in first_runs:
            first_runs[method] = HoldoutProjection(
                method=method, seed=seed, Y_out=Y_out, Y_in_space=Y_space  # type: ignore[arg-type]
            )

    for method in methods:
        if method in DETERMINISTIC_METHODS:
            identity = get_identity_projection(X_in, Y_in, X_out)
            _record(method, None, identity.projection, Y_in)
            continue
        for index, seed in enumerate(seeds, start=1):
            Y_out, Y_space = _run_trained_method(
                method, seed, X_in, Y_in, labels_in, X_out, head_settings, umap_settings
            )
            _record(method, seed, Y_out, Y_space)
            logger.info("%s: %s run %d/%d (seed %d) done", run_id, method, index, len(seeds), seed)

    comparison = HoldoutComparison(
        run_id=run_id,
        n_in=int(X_in.shape[0]),
        n_out=int(X_out.shape[0]),
        k_holdout=k_holdout,
        n_neighbors=n_neighbors,
        training_seeds=seeds,
        methods=list(methods),
        runs=runs,
    )
    return comparison, first_runs


def get_paired_subsample_intervals(
    X_out: np.ndarray,
    projections: dict[str, HoldoutProjection],
    labels_in: np.ndarray,
    reference_labels_out: np.ndarray,
    k_holdout: int,
    seed: int,
    reference_method: str = "reap_mlp",
    n_resamples: int = 2000,
    fraction: float = 0.8,
    groups: np.ndarray | None = None,
    n_neighbors: int = 15,
) -> list[SubsampleInterval]:
    """Paired subsample intervals of (reference − baseline) for every metric.

    Each subsample keeps ``fraction`` of the held-out units, drawn without
    replacement (the same subsample for every method), where a unit is a
    document or, when ``groups`` is given, a whole group such as a source
    document (protocol v1.6 clause b, item 6). The interval is the 2.5th–97.5th
    percentile of the differences over the subsamples on which both metrics
    are defined. Descriptive only; not part of any Holm family.

    Side effects: none beyond KMeans inside the ARI (fixed seed). ``seed``
    drives the subsample draws through a local generator.
    """
    if reference_method not in projections:
        raise ValueError(f"reference method {reference_method!r} has no projection")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    units = np.arange(X_out.shape[0]) if groups is None else np.unique(groups)
    n_keep = int(round(fraction * units.size))
    rng = np.random.default_rng(seed)

    def _score(method: str, rows: np.ndarray) -> dict[str, float | None]:
        projection = projections[method]
        values, _ = compute_holdout_metrics(
            X_out[rows], projection.Y_out[rows], projection.Y_in_space, labels_in,
            reference_labels_out[rows], k_holdout, n_neighbors,
        )
        return values

    all_rows = np.arange(X_out.shape[0])
    full = {method: _score(method, all_rows) for method in projections}
    baselines = [m for m in projections if m != reference_method]
    differences: dict[tuple[str, str], list[float]] = {
        (b, metric): [] for b in baselines for metric in HOLDOUT_METRIC_IDS
    }
    for _ in range(n_resamples):
        kept = rng.choice(units, size=n_keep, replace=False)
        rows = kept if groups is None else np.flatnonzero(np.isin(groups, kept))
        rows = np.sort(rows)
        scores = {method: _score(method, rows) for method in projections}
        for b in baselines:
            for metric in HOLDOUT_METRIC_IDS:
                ref_value, base_value = scores[reference_method][metric], scores[b][metric]
                if ref_value is not None and base_value is not None:
                    differences[(b, metric)].append(ref_value - base_value)

    intervals = []
    for b in baselines:
        for metric in HOLDOUT_METRIC_IDS:
            diffs = np.asarray(differences[(b, metric)])
            ref_full, base_full = full[reference_method][metric], full[b][metric]
            intervals.append(
                SubsampleInterval(
                    metric_id=metric,
                    reference_method=reference_method,
                    baseline_method=b,
                    point_difference=(
                        None if ref_full is None or base_full is None else ref_full - base_full
                    ),
                    ci_low=float(np.percentile(diffs, 2.5)) if diffs.size else None,
                    ci_high=float(np.percentile(diffs, 97.5)) if diffs.size else None,
                    n_resamples=n_resamples,
                    n_defined=int(diffs.size),
                    fraction=fraction,
                )
            )
    return intervals
