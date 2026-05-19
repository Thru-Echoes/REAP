"""Statistical inference for the REAP benchmark.

Implements the inferential layer required by ``manuscript/evaluation_protocol.md``
§8:

- **Seed-bootstrap CI** (:func:`compute_seed_bootstrap_ci`) — nonparametric
  bootstrap over a 1D sample of per-seed metric values, the correct
  uncertainty source for the "mean metric across the pre-declared seed set"
  claim. Distinct from data-subsampling bootstrap, which answers a different
  question.
- **Paired Wilcoxon signed-rank test** (:func:`paired_wilcoxon`) — compares
  two methods matched by seed index; makes no distributional assumption.
- **Effect sizes**: paired Cohen's ``d_z`` (:func:`paired_cohens_d`) and
  Cliff's delta (:func:`cliffs_delta`), the latter nonparametric.
- **Multiple-comparison corrections**: Holm-Bonferroni
  (:func:`holm_bonferroni`) as the primary family-wise-error control and
  Benjamini-Hochberg FDR (:func:`benjamini_hochberg`) as a sensitivity check.
- **Family comparison helper** (:func:`compare_family`) — runs all pairwise
  tests of one reference method vs a dict of baselines and applies both
  corrections; emits a ready-to-render :class:`FamilyComparisonResult`.

All functions are pure (no side effects) unless noted. Results are returned
as frozen Pydantic models so downstream callers can serialize them directly
to CSV / JSON without extra glue.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap CI over a 1D distribution
# ---------------------------------------------------------------------------


class BootstrapCIResult(BaseModel):
    """Bootstrap confidence interval for a 1D sample."""

    model_config = ConfigDict(frozen=True)

    mean: float = Field(..., description="Sample mean (unchanged by bootstrap)")
    median: float = Field(..., description="Sample median")
    ci_lower: float = Field(
        ..., description="Lower bound of the CI (percentile)"
    )
    ci_upper: float = Field(
        ..., description="Upper bound of the CI (percentile)"
    )
    ci_level: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Confidence level (default 0.95 = 95% CI)",
    )
    n_bootstrap: int = Field(..., ge=1, description="Number of bootstrap draws")
    n_samples: int = Field(..., ge=1, description="Number of input values")


def compute_seed_bootstrap_ci(
    values: np.ndarray | list[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapCIResult:
    """Nonparametric bootstrap CI for the mean of a 1D sample.

    Resamples ``values`` with replacement and computes the mean of each
    resample, then reports percentile CIs at ``ci_level``. Appropriate for
    the "mean metric across the seed set" uncertainty per protocol §8.

    Unlike :func:`reap.benchmarks.compute_subsample_ci` (which resamples
    data rows), this function resamples the seed distribution and is what
    the pre-registered analysis calls for.

    Parameters
    ----------
    values : 1D array or list of per-seed metric values.
    n_bootstrap : Number of bootstrap resamples.
    ci_level : Confidence level; must be in ``(0, 1)``.
    random_state : Seed for reproducible resampling.

    Returns
    -------
    :class:`BootstrapCIResult` with mean, median, CI bounds, and bookkeeping.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = len(arr)
    if n == 0:
        raise ValueError("values is empty")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level={ci_level} must be in (0, 1)")

    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(arr[idx].mean())

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return BootstrapCIResult(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        ci_lower=lower,
        ci_upper=upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Paired statistical tests and effect sizes
# ---------------------------------------------------------------------------


class PairedTestResult(BaseModel):
    """Full paired comparison of two methods on matched seeds."""

    model_config = ConfigDict(frozen=True)

    reference: str = Field(..., description="Reference method name")
    comparator: str = Field(..., description="Comparator method name")
    metric: str = Field(..., description="Metric name")
    n: int = Field(..., ge=0, description="Number of matched pairs")
    mean_diff: float = Field(..., description="Mean of (reference - comparator)")
    median_diff: float = Field(..., description="Median of (reference - comparator)")
    wilcoxon_W: float = Field(
        ..., description="Wilcoxon signed-rank statistic (NaN if undefined)"
    )
    wilcoxon_p: float = Field(
        ..., description="Wilcoxon p-value (NaN if undefined)"
    )
    cohens_d: float = Field(
        ..., description="Paired Cohen's d_z on reference - comparator"
    )
    cliffs_delta: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Nonparametric effect size in [-1, 1]",
    )
    direction: Literal["higher_is_better", "lower_is_better"] = Field(
        ..., description="How the metric is oriented for the alternative"
    )


def paired_wilcoxon(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired observations.

    Pairs are aligned by index. Observations with zero difference are dropped
    before the test per Pratt's convention (scipy's default handling issues
    a warning); ``(NaN, NaN)`` is returned when fewer than 2 non-zero
    differences remain.

    Parameters
    ----------
    x, y : Paired sample arrays of equal length.
    alternative : ``"two-sided"``, ``"greater"`` (H_a: x > y), or ``"less"``.

    Returns
    -------
    ``(W_statistic, p_value)``.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if len(xa) != len(ya):
        raise ValueError(
            f"Paired arrays must have same length; got {len(xa)} and {len(ya)}"
        )
    diffs = xa - ya
    nonzero = diffs != 0
    if int(nonzero.sum()) < 2:
        return float("nan"), float("nan")
    res = stats.wilcoxon(xa[nonzero], ya[nonzero], alternative=alternative)
    # scipy returns a WilcoxonResult namedtuple; type stubs don't expose it fully.
    return float(res.statistic), float(res.pvalue)  # type: ignore[attr-defined]


def paired_cohens_d(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> float:
    """Paired Cohen's ``d_z``: ``mean(x - y) / std(x - y, ddof=1)``.

    Standard effect size for matched-pair designs (Lakens 2013). Positive
    values indicate ``x > y`` on average. Returns ``0.0`` when the paired
    differences have near-zero variance (no meaningful effect).

    Parameters
    ----------
    x, y : Paired sample arrays.

    Returns
    -------
    Cohen's ``d_z``.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if len(xa) != len(ya):
        raise ValueError(
            f"Paired arrays must have same length; got {len(xa)} and {len(ya)}"
        )
    diffs = xa - ya
    if len(diffs) < 2:
        return 0.0
    sd = float(diffs.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(diffs.mean() / sd)


def cliffs_delta(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> float:
    """Cliff's delta — nonparametric effect size in ``[-1, 1]``.

    ``delta = (#{i,j : x_i > y_j} - #{i,j : x_i < y_j}) / (n_x * n_y)``.
    Unlike Cohen's d, makes no assumption about the shape of the
    distributions; recommended for small-N paired designs per Romano et al.
    (2006). Computed in ``O((n_x + n_y) log(n_x + n_y))`` via ranks.

    Parameters
    ----------
    x, y : Sample arrays (need not be paired nor equal length).

    Returns
    -------
    Cliff's delta.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    nx, ny = len(xa), len(ya)
    if nx == 0 or ny == 0:
        return 0.0
    combined = np.concatenate([xa, ya])
    ranks = stats.rankdata(combined)
    rx = ranks[:nx]
    U = float(rx.sum()) - nx * (nx + 1) / 2.0
    return float(2.0 * U / (nx * ny) - 1.0)


def paired_comparison(
    reference: np.ndarray | list[float],
    comparator: np.ndarray | list[float],
    reference_name: str,
    comparator_name: str,
    metric_name: str,
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
) -> PairedTestResult:
    """Full paired comparison between two methods on matched seeds.

    Runs the Wilcoxon signed-rank test with an alternative derived from
    ``direction``, and computes paired Cohen's ``d_z`` and Cliff's delta as
    complementary effect sizes.

    Parameters
    ----------
    reference, comparator : Matched per-seed metric vectors.
    reference_name, comparator_name : Method labels (e.g., ``"reap"``,
        ``"procrustes"``).
    metric_name : Metric label for bookkeeping (e.g., ``"silhouette_C"``).
    direction : ``"higher_is_better"`` -> ``H_a: reference > comparator``;
        ``"lower_is_better"`` -> ``H_a: reference < comparator``.

    Returns
    -------
    :class:`PairedTestResult`.
    """
    alt: Literal["greater", "less"] = (
        "greater" if direction == "higher_is_better" else "less"
    )
    ref_arr = np.asarray(reference, dtype=np.float64).ravel()
    comp_arr = np.asarray(comparator, dtype=np.float64).ravel()
    w, p = paired_wilcoxon(ref_arr, comp_arr, alternative=alt)
    diffs = ref_arr - comp_arr
    return PairedTestResult(
        reference=reference_name,
        comparator=comparator_name,
        metric=metric_name,
        n=len(ref_arr),
        mean_diff=float(diffs.mean()) if len(diffs) else 0.0,
        median_diff=float(np.median(diffs)) if len(diffs) else 0.0,
        wilcoxon_W=w,
        wilcoxon_p=p,
        cohens_d=paired_cohens_d(ref_arr, comp_arr),
        cliffs_delta=cliffs_delta(ref_arr, comp_arr),
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down FWER correction.

    Controls the family-wise error rate at the same level as Bonferroni but
    is uniformly more powerful (Holm 1979). Returns adjusted p-values; compare
    to the unadjusted alpha (e.g., 0.05) as usual.

    Parameters
    ----------
    p_values : Array of raw p-values.

    Returns
    -------
    Array of adjusted p-values, same length / same order as input.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    p_sorted = p[order]
    # Multipliers (n - k) for k = 0, 1, ..., n-1
    multipliers = (n - np.arange(n)).astype(np.float64)
    adj_sorted = multipliers * p_sorted
    # Enforce monotonicity (step-down): adj_k = max(adj_0, ..., adj_k)
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(p)
    adj[order] = adj_sorted
    return adj


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR correction.

    Controls the false-discovery rate at the same level as the raw alpha
    (Benjamini & Hochberg 1995). More powerful than Holm when many tests are
    expected to reject the null. Returns adjusted p-values (q-values).

    Parameters
    ----------
    p_values : Array of raw p-values.

    Returns
    -------
    Array of q-values, same length / same order as input.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adj_sorted = p_sorted * n / ranks
    # Step-up: enforce monotonicity from the largest p downwards
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(p)
    adj[order] = adj_sorted
    return adj


# ---------------------------------------------------------------------------
# Family comparison: reference method vs each baseline
# ---------------------------------------------------------------------------


class FamilyComparisonResult(BaseModel):
    """Ready-to-render table of one reference method vs each baseline."""

    model_config = ConfigDict(frozen=True)

    reference: str = Field(..., description="Reference method (e.g., 'reap')")
    metric: str = Field(..., description="Metric compared")
    comparisons: list[PairedTestResult] = Field(
        ..., description="One paired test per baseline, order preserved"
    )
    p_holm: list[float] = Field(
        ..., description="Holm-Bonferroni adjusted p-values, aligned with comparisons"
    )
    p_bh: list[float] = Field(
        ..., description="Benjamini-Hochberg adjusted p-values, aligned with comparisons"
    )


def compare_family(
    reference_values: np.ndarray | list[float],
    baseline_values: dict[str, np.ndarray | list[float]],
    reference_name: str,
    metric_name: str,
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better",
) -> FamilyComparisonResult:
    """Run paired tests of ``reference`` against each baseline and correct.

    Applies Holm-Bonferroni (primary) and BH-FDR (sensitivity) corrections
    across the family of baseline comparisons. Missing / undefined p-values
    are treated as 1.0 for correction purposes so they never produce
    spurious "significant after correction" results.

    Parameters
    ----------
    reference_values : Per-seed metric vector for the reference method.
    baseline_values : Mapping from baseline name to per-seed metric vector.
        All arrays must have the same length as ``reference_values``.
    reference_name : Name to attach to the reference (e.g., ``"reap"``).
    metric_name : Metric label for bookkeeping.
    direction : Orientation of the metric; passed to paired tests.

    Returns
    -------
    :class:`FamilyComparisonResult` containing per-baseline paired tests and
    both corrections.
    """
    comparisons: list[PairedTestResult] = []
    for name, values in baseline_values.items():
        comparisons.append(
            paired_comparison(
                reference_values,
                values,
                reference_name,
                name,
                metric_name,
                direction,
            )
        )
    raw_p = np.array(
        [c.wilcoxon_p for c in comparisons], dtype=np.float64
    )
    # NaN p-values are treated as 1.0 for correction so they cannot become
    # spuriously "significant". The raw NaN is preserved in each
    # PairedTestResult for transparency.
    raw_p_clean = np.where(np.isnan(raw_p), 1.0, raw_p)
    p_holm = holm_bonferroni(raw_p_clean).tolist()
    p_bh = benjamini_hochberg(raw_p_clean).tolist()
    return FamilyComparisonResult(
        reference=reference_name,
        metric=metric_name,
        comparisons=comparisons,
        p_holm=[float(x) for x in p_holm],
        p_bh=[float(x) for x in p_bh],
    )
