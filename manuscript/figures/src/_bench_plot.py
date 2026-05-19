"""Shared helpers for the per-dataset metrics bar charts.

These plots read `results/<dataset>/combined_set_{A,B,C}/all_methods.csv`
— the same CSVs the manuscript Tables 1/1b and 2/2b cite — and render
REAP vs the baselines on the headline consensus metrics. Because each
(method, seed-set) yields a single consensus value, cross-method
"95% CI from set A/B/C" is undefined; we instead report the cross-set
mean ± SD across the three disjoint seed sets, matching the
inference unit declared in §4.6 of the experiments section.

Inputs
------
combined_set_{A,B,C}/all_methods.csv per dataset, as written by
`scripts/run_paper_benchmark.py`.

Outputs
-------
None directly; consumed by the per-dataset scripts.

Side effects
------------
None.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


# Mapping from result-CSV column to (label, source-tier).
# "consensus_*" columns hold one value per (method, seed-set).
# "per_seed_*" columns are mean across the set's seeds (so the
# baseline pairwise-identity caveat in §4.6 applies — they will show
# small or zero differences against single_seed by construction).
# Topic-coherence columns may not be in the CSV yet (Lane A in progress).
HEADLINE_METRICS: dict[str, tuple[str, str]] = {
    "consensus_silhouette": ("Consensus silhouette", "consensus"),
    "consensus_trustworthiness": ("Consensus trustworthiness", "consensus"),
    "topic_coherence_umass": ("Topic coherence (UMass)", "consensus"),
    "topic_coherence_npmi": ("Topic coherence (NPMI)", "consensus"),
    "topic_coherence_cv": ("Topic coherence ($C_v$)", "consensus"),
}

PER_SEED_REFERENCE_METRIC: tuple[str, str] = (
    "trustworthiness",
    "Per-seed trustworthiness",
)

METHOD_ORDER: list[str] = [
    "single_seed",
    "best_of_n",
    "naive_average",
    "procrustes",
    "bertopic",
    "reap",
]

METHOD_DISPLAY: dict[str, str] = {
    "single_seed": "single seed",
    "best_of_n": "best-of-N",
    "naive_average": "naive avg",
    "procrustes": "Procrustes",
    "bertopic": "BERTopic",
    "reap": "REAP",
}

# REAP highlighted; other baselines grey-blue gradient.
METHOD_COLOUR: dict[str, str] = {
    "single_seed":  "#9ec5e8",
    "best_of_n":    "#7aa9d1",
    "naive_average": "#7f7f7f",
    "procrustes":   "#4f7fa5",
    "bertopic":     "#bf9d3a",
    "reap":         "#2ca02c",
}


@dataclass(frozen=True)
class _MetricStats:
    """Per-method aggregated stats across seed sets."""

    method: str
    metric: str
    values_by_set: dict[str, float]
    mean: float
    sd: float
    n_sets: int


def _read_all_methods_csv(path: Path) -> dict[str, dict[str, str]]:
    """Read a `combined_set_<X>/all_methods.csv`.

    Returns
    -------
    dict mapping method-name -> column-name -> raw string value.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["method"]: row for row in reader}


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def get_metric_stats_across_sets(
    dataset_dir: Path,
    metric_columns: list[str],
    method_order: list[str],
) -> dict[tuple[str, str], _MetricStats]:
    """Aggregate per-method values across combined_set_{A,B,C}.

    Parameters
    ----------
    dataset_dir : path containing combined_set_A/B/C subdirs.
    metric_columns : column names to aggregate.
    method_order : methods to look for. Missing rows are skipped silently.

    Returns
    -------
    dict keyed by (method, metric) with _MetricStats; missing
    (method, metric) combinations are absent from the output.
    """
    set_to_rows: dict[str, dict[str, dict[str, str]]] = {}
    for set_name in ("A", "B", "C"):
        csv_path = dataset_dir / f"combined_set_{set_name}" / "all_methods.csv"
        if not csv_path.exists():
            logger.warning("Missing %s — skipping seed set %s", csv_path, set_name)
            continue
        set_to_rows[set_name] = _read_all_methods_csv(csv_path)

    out: dict[tuple[str, str], _MetricStats] = {}
    for method in method_order:
        for metric in metric_columns:
            values_by_set: dict[str, float] = {}
            for set_name, rows in set_to_rows.items():
                if method not in rows:
                    continue
                value = _parse_optional_float(rows[method].get(metric))
                if value is None:
                    continue
                values_by_set[set_name] = value
            if not values_by_set:
                continue
            arr = np.array(list(values_by_set.values()), dtype=float)
            out[(method, metric)] = _MetricStats(
                method=method,
                metric=metric,
                values_by_set=values_by_set,
                mean=float(arr.mean()),
                sd=float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                n_sets=len(arr),
            )
    return out


def get_available_metric_columns(
    dataset_dir: Path,
    requested: dict[str, tuple[str, str]],
) -> list[tuple[str, str, str]]:
    """Return [(column, label, source-tier)] for metrics actually present in CSVs.

    Logs a warning per requested column missing from set A's CSV (the
    union of present columns is the same across sets in this repo).
    """
    csv_path = dataset_dir / "combined_set_A" / "all_methods.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing reference CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    header_set = set(header)
    available: list[tuple[str, str, str]] = []
    for col, (label, source) in requested.items():
        if col in header_set:
            # Check at least one row has a non-empty value for this column.
            any_value = False
            rows = _read_all_methods_csv(csv_path)
            for row in rows.values():
                if _parse_optional_float(row.get(col)) is not None:
                    any_value = True
                    break
            if any_value:
                available.append((col, label, source))
            else:
                logger.warning(
                    "Column %s present in %s but all rows empty — skipping.",
                    col, csv_path,
                )
        else:
            logger.warning(
                "Column %s missing from %s — skipping this metric.",
                col, csv_path,
            )
    return available


def plot_metrics_panel(
    ax: Axes,
    stats: dict[tuple[str, str], _MetricStats],
    metric_column: str,
    metric_label: str,
    method_order: list[str],
    higher_is_better: bool = True,
) -> None:
    """Draw one bar chart panel for one metric.

    Methods on the x axis (those present in `stats`), bar heights at the
    cross-set mean, error bars at one cross-set standard deviation.
    """
    methods_present = [
        m for m in method_order if (m, metric_column) in stats
    ]
    means = [stats[(m, metric_column)].mean for m in methods_present]
    sds = [stats[(m, metric_column)].sd for m in methods_present]
    colours = [METHOD_COLOUR.get(m, "#777") for m in methods_present]
    labels = [METHOD_DISPLAY.get(m, m) for m in methods_present]

    x = np.arange(len(methods_present))
    bars = ax.bar(x, means, yerr=sds, capsize=4, color=colours,
                  edgecolor="#222", linewidth=0.5, alpha=0.92)

    # Annotate each bar with the mean (3 d.p.).
    for bar_obj, mean_val in zip(bars, means):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + (max(means) - min(means) + 1e-3) * 0.04,
            f"{mean_val:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=10)
    arrow = "$\\uparrow$ higher is better" if higher_is_better else "$\\downarrow$ lower is better"
    ax.set_title(f"{metric_label}  ({arrow})", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_dataset_metrics_bars(
    dataset_dir: Path,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> dict[tuple[str, str], _MetricStats]:
    """Render the metrics bar-chart figure for one dataset.

    Returns the stats dict for downstream callers (used by the
    figures/README.md cross-walk). Side effect: writes PNG + PDF.
    """
    requested = dict(HEADLINE_METRICS)
    available_consensus = get_available_metric_columns(dataset_dir, requested)
    available_per_seed = get_available_metric_columns(
        dataset_dir,
        {PER_SEED_REFERENCE_METRIC[0]: (PER_SEED_REFERENCE_METRIC[1], "per_seed")},
    )

    columns = [col for col, _, _ in available_consensus + available_per_seed]
    stats = get_metric_stats_across_sets(dataset_dir, columns, METHOD_ORDER)

    panels: list[tuple[str, str, bool]] = []
    for col, label, _ in available_consensus:
        # Trustworthiness is "higher better" in REAP's harness even though
        # for consensus REAP intentionally trades off some trustworthiness
        # for silhouette — the metric itself is still higher-is-better.
        panels.append((col, label, True))
    for col, label, _ in available_per_seed:
        panels.append((col, label, True))

    n = len(panels)
    if n == 0:
        raise RuntimeError(
            f"No headline metrics found for {dataset_dir} — "
            "expected at least consensus_silhouette + consensus_trustworthiness."
        )
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig_height = 4.4 * n_rows + 0.6
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.7 * n_cols, fig_height),
        squeeze=False,
    )
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    for ax, (col, label, higher_better) in zip(axes.flat, panels):
        plot_metrics_panel(
            ax, stats, col, label, METHOD_ORDER,
            higher_is_better=higher_better,
        )

    subtitle = (
        "Cross-set mean ± SD across 3 disjoint seed sets "
        "(A/B/C, 30 seeds each). Source: "
        "results/<dataset>/combined_set_{A,B,C}/all_methods.csv."
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s and %s", out_png, out_pdf)
    return stats
