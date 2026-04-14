"""Publication-ready table generation for REAP benchmark results.

Converts benchmark, ablation, and bootstrap results into formatted tables
for inclusion in manuscripts and supplementary materials. Supported formats:
LaTeX (booktabs), Markdown (GitHub-flavored), and CSV.

All functions are pure (no side effects) except ``benchmark_to_csv`` which
writes to disk.

Exports
-------
- benchmark_to_markdown, benchmark_to_latex, benchmark_to_csv
- cross_dataset_table, cross_dataset_latex
- ablation_to_markdown, ablation_to_latex
- sweep_to_markdown, sweep_to_latex
- bootstrap_to_markdown, bootstrap_to_latex
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from reap.ablation import (
    ParameterSweepResult,
    SeedAblationResult,
)
from reap.benchmarks import BenchmarkResult, BootstrapCI, MethodResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------


def _format_metric(value: float, std: float = 0.0, decimals: int = 3) -> str:
    """Format a metric value, optionally with standard deviation.

    Parameters
    ----------
    value : The metric value.
    std : Standard deviation. If > 0, appended as " +/- std".
    decimals : Number of decimal places.

    Returns
    -------
    Formatted string, e.g. ``'0.892'`` or ``'0.892 +/- 0.031'``.
    """
    formatted = f"{value:.{decimals}f}"
    if std > 0:
        formatted += f" \u00b1 {std:.{decimals}f}"
    return formatted


def _format_metric_latex(value: float, std: float = 0.0, decimals: int = 3) -> str:
    r"""Format a metric value for LaTeX, optionally with standard deviation.

    Parameters
    ----------
    value : The metric value.
    std : Standard deviation. If > 0, appended as ``$\\pm$ std``.
    decimals : Number of decimal places.

    Returns
    -------
    LaTeX-formatted string, e.g. ``'0.892'`` or ``'0.892 $\\pm$ 0.031'``.
    """
    formatted = f"{value:.{decimals}f}"
    if std > 0:
        formatted += f" $\\pm$ {std:.{decimals}f}"
    return formatted


def _format_ci(mean: float, ci_lower: float, ci_upper: float, decimals: int = 3) -> str:
    """Format a value with a confidence interval for Markdown.

    Parameters
    ----------
    mean : Point estimate.
    ci_lower : Lower bound of the confidence interval.
    ci_upper : Upper bound of the confidence interval.
    decimals : Number of decimal places.

    Returns
    -------
    Formatted string, e.g. ``'0.892 [0.871, 0.913]'``.
    """
    return (
        f"{mean:.{decimals}f} "
        f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
    )


def _format_ci_latex(
    mean: float, ci_lower: float, ci_upper: float, decimals: int = 3,
) -> str:
    r"""Format a value with a confidence interval for LaTeX.

    Parameters
    ----------
    mean : Point estimate.
    ci_lower : Lower bound of the confidence interval.
    ci_upper : Upper bound of the confidence interval.
    decimals : Number of decimal places.

    Returns
    -------
    LaTeX-formatted string, e.g. ``'0.892 [0.871, 0.913]'``.
    """
    return (
        f"{mean:.{decimals}f} "
        f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
    )


def _bold_markdown(text: str) -> str:
    """Wrap text in ``**`` for Markdown bold.

    Parameters
    ----------
    text : Plain text to bold.

    Returns
    -------
    Markdown bold string, e.g. ``'**0.892**'``.
    """
    return f"**{text}**"


def _bold_latex(text: str) -> str:
    r"""Wrap text in ``\\textbf{}`` for LaTeX bold.

    Parameters
    ----------
    text : Plain text to bold.

    Returns
    -------
    LaTeX bold string, e.g. ``'\\textbf{0.892}'``.
    """
    return f"\\textbf{{{text}}}"


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Parameters
    ----------
    text : Raw text that may contain LaTeX specials.

    Returns
    -------
    Escaped text safe for inclusion in LaTeX documents.
    """
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _format_runtime(seconds: float) -> str:
    """Format a runtime duration as a human-readable string.

    Parameters
    ----------
    seconds : Duration in seconds.

    Returns
    -------
    Formatted string: ``'1.2s'`` for < 60s, ``'2m 30s'`` for >= 60s.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}m {remaining:.0f}s"


def _na_str() -> str:
    """Return the string used for missing / not-applicable values.

    Returns
    -------
    The literal string ``'--'``.
    """
    return "--"


# ---------------------------------------------------------------------------
# Method-level cell extraction
# ---------------------------------------------------------------------------


def _get_method_cells(
    method: MethodResult, decimals: int = 3,
) -> dict[str, tuple[float | None, str]]:
    """Extract (sortable_value, display_string) pairs for each metric column.

    Parameters
    ----------
    method : A single method's benchmark results.
    decimals : Number of decimal places for formatting.

    Returns
    -------
    Mapping from column name to (numeric_value_or_None, display_string).
    Numeric value is ``None`` when the metric is unavailable.
    """
    cells: dict[str, tuple[float | None, str]] = {}

    cells["TW"] = (
        method.trustworthiness,
        _format_metric(method.trustworthiness, method.trustworthiness_std, decimals),
    )
    cells["Sil"] = (
        method.silhouette,
        _format_metric(method.silhouette, method.silhouette_std, decimals),
    )
    cells["K"] = (
        float(method.best_k),
        _format_metric(float(method.best_k), method.best_k_std, decimals=0),
    )

    if method.seed_to_seed_ari_mean is not None:
        cells["S2S ARI"] = (
            method.seed_to_seed_ari_mean,
            _format_metric(
                method.seed_to_seed_ari_mean,
                method.seed_to_seed_ari_std or 0.0,
                decimals,
            ),
        )
    else:
        cells["S2S ARI"] = (None, _na_str())

    if method.seed_to_consensus_ari_mean is not None:
        cells["S2C ARI"] = (
            method.seed_to_consensus_ari_mean,
            _format_metric(method.seed_to_consensus_ari_mean, decimals=decimals),
        )
    else:
        cells["S2C ARI"] = (None, _na_str())

    cells["Time"] = (method.runtime_seconds, _format_runtime(method.runtime_seconds))
    return cells


def _get_method_cells_latex(
    method: MethodResult, decimals: int = 3,
) -> dict[str, tuple[float | None, str]]:
    """Extract (sortable_value, latex_display_string) pairs for each metric column.

    Parameters
    ----------
    method : A single method's benchmark results.
    decimals : Number of decimal places for formatting.

    Returns
    -------
    Mapping from column name to (numeric_value_or_None, latex_display_string).
    """
    cells: dict[str, tuple[float | None, str]] = {}

    cells["TW"] = (
        method.trustworthiness,
        _format_metric_latex(method.trustworthiness, method.trustworthiness_std, decimals),
    )
    cells["Sil"] = (
        method.silhouette,
        _format_metric_latex(method.silhouette, method.silhouette_std, decimals),
    )
    cells["K"] = (
        float(method.best_k),
        _format_metric_latex(float(method.best_k), method.best_k_std, decimals=0),
    )

    if method.seed_to_seed_ari_mean is not None:
        cells["S2S ARI"] = (
            method.seed_to_seed_ari_mean,
            _format_metric_latex(
                method.seed_to_seed_ari_mean,
                method.seed_to_seed_ari_std or 0.0,
                decimals,
            ),
        )
    else:
        cells["S2S ARI"] = (None, _na_str())

    if method.seed_to_consensus_ari_mean is not None:
        cells["S2C ARI"] = (
            method.seed_to_consensus_ari_mean,
            _format_metric_latex(method.seed_to_consensus_ari_mean, decimals=decimals),
        )
    else:
        cells["S2C ARI"] = (None, _na_str())

    cells["Time"] = (method.runtime_seconds, _format_runtime(method.runtime_seconds))
    return cells


# ---------------------------------------------------------------------------
# Best-value detection
# ---------------------------------------------------------------------------

# For these columns, higher is better. Time is lower-is-better.
_HIGHER_IS_BETTER: set[str] = {"TW", "Sil", "S2S ARI", "S2C ARI"}
_LOWER_IS_BETTER: set[str] = {"Time"}
# K is not bolded (no universal "better" direction)


def _find_best_per_column(
    all_cells: dict[str, dict[str, tuple[float | None, str]]],
) -> dict[str, str | None]:
    """Identify the best method for each column.

    Parameters
    ----------
    all_cells : Mapping from method_name -> column_name -> (value, display).

    Returns
    -------
    Mapping from column_name -> method_name that holds the best value,
    or ``None`` if no valid values exist for that column.
    """
    best: dict[str, str | None] = {}
    columns = {"TW", "Sil", "S2S ARI", "S2C ARI", "Time"}

    for col in columns:
        candidates: list[tuple[str, float]] = []
        for method_name, cells in all_cells.items():
            val, _ = cells[col]
            if val is not None:
                candidates.append((method_name, val))

        if not candidates:
            best[col] = None
            continue

        if col in _HIGHER_IS_BETTER:
            best[col] = max(candidates, key=lambda x: x[1])[0]
        elif col in _LOWER_IS_BETTER:
            best[col] = min(candidates, key=lambda x: x[1])[0]
        else:
            best[col] = None

    return best


# ---------------------------------------------------------------------------
# 1. Method comparison tables
# ---------------------------------------------------------------------------


def benchmark_to_markdown(
    result: BenchmarkResult,
    bold_best: bool = True,
    decimals: int = 3,
) -> str:
    """Generate a Markdown table comparing all methods in a benchmark.

    Columns: Method | TW | Sil | K | S2S ARI | S2C ARI | Time

    Parameters
    ----------
    result : Complete benchmark result with one or more methods.
    bold_best : If ``True``, wrap the best value in each column with ``**``.
        Higher is better for TW, Sil, S2S ARI, S2C ARI; lower for Time.
        K is never bolded (no universal direction).
    decimals : Number of decimal places for metric values.

    Returns
    -------
    Markdown-formatted table string including a header line with dataset info.
    """
    columns = ["Method", "TW", "Sil", "K", "S2S ARI", "S2C ARI", "Time"]
    header_info = (
        f"**{result.dataset_name}** "
        f"(n={result.n_samples}, d={result.n_features}, "
        f"d_UMAP={result.n_components}, nn={result.n_neighbors}, "
        f"md={result.min_dist}, seeds={result.n_seeds})"
    )

    # Collect cells for every method
    all_cells: dict[str, dict[str, tuple[float | None, str]]] = {}
    for method in result.methods:
        all_cells[method.method] = _get_method_cells(method, decimals)

    # Determine best per column
    best_per_col = _find_best_per_column(all_cells) if bold_best else {}

    # Build rows
    rows: list[list[str]] = []
    for method in result.methods:
        cells = all_cells[method.method]
        row = [method.method]
        for col in columns[1:]:
            _, display = cells[col]
            if bold_best and best_per_col.get(col) == method.method:
                display = _bold_markdown(display)
            row.append(display)
        rows.append(row)

    # Compute column widths for alignment
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Render
    lines: list[str] = [header_info, ""]
    header_line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        row_line = "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def benchmark_to_latex(
    result: BenchmarkResult,
    bold_best: bool = True,
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> str:
    r"""Generate a LaTeX table comparing all methods in a benchmark.

    Produces a complete ``table`` environment with ``booktabs`` rules,
    ``\textbf{}`` for best values, and ``$\pm$`` for standard deviations.

    Parameters
    ----------
    result : Complete benchmark result with one or more methods.
    bold_best : If ``True``, wrap best values with ``\textbf{}``.
    caption : Optional table caption. If ``None``, a default is generated.
    label : Optional ``\label{}`` key. If ``None``, a default is generated
        from the dataset name.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    LaTeX source string for a complete ``table`` environment.
    """
    columns_display = ["Method", "TW", "Sil", "$K$", "S2S ARI", "S2C ARI", "Time"]
    col_keys = ["Method", "TW", "Sil", "K", "S2S ARI", "S2C ARI", "Time"]

    # Collect cells for every method
    all_cells: dict[str, dict[str, tuple[float | None, str]]] = {}
    for method in result.methods:
        all_cells[method.method] = _get_method_cells_latex(method, decimals)

    best_per_col = _find_best_per_column(all_cells) if bold_best else {}

    # Build data rows
    data_rows: list[str] = []
    for method in result.methods:
        cells = all_cells[method.method]
        parts = [_escape_latex(method.method)]
        for col in col_keys[1:]:
            _, display = cells[col]
            if bold_best and best_per_col.get(col) == method.method:
                display = _bold_latex(display)
            parts.append(display)
        data_rows.append("        " + " & ".join(parts) + " \\\\")

    # Default caption/label
    if caption is None:
        caption = (
            f"Method comparison on {result.dataset_name} "
            f"($n$={result.n_samples}, $d$={result.n_features}, "
            f"$d_{{\\text{{UMAP}}}}$={result.n_components}, "
            f"$k_{{\\text{{nn}}}}$={result.n_neighbors}, "
            f"$d_{{\\min}}$={result.min_dist}, "
            f"seeds={result.n_seeds})."
        )
    safe_name = result.dataset_name.lower().replace(" ", "_").replace("-", "_")
    if label is None:
        label = f"tab:{safe_name}_comparison"

    # Column alignment: left for method, centered for metrics
    col_spec = "l" + "c" * (len(col_keys) - 1)

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(columns_display) + " \\\\",
        "        \\midrule",
    ]
    lines.extend(data_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"


def benchmark_to_csv(
    result: BenchmarkResult,
    path: str | Path,
    decimals: int = 3,
) -> None:
    """Export benchmark results to a CSV file.

    One row per method with columns for all metrics. Values are written as
    plain numbers (no formatting or bold).

    Parameters
    ----------
    result : Complete benchmark result.
    path : Output file path.
    decimals : Number of decimal places for metric values.

    Side effects
    -------------
    Writes a CSV file to ``path``.
    """
    fieldnames = [
        "method",
        "description",
        "trustworthiness",
        "trustworthiness_std",
        "silhouette",
        "silhouette_std",
        "best_k",
        "best_k_std",
        "seed_to_seed_ari_mean",
        "seed_to_seed_ari_std",
        "seed_to_consensus_ari_mean",
        "n_seeds",
        "runtime_seconds",
    ]

    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method in result.methods:
            row: dict[str, str] = {
                "method": method.method,
                "description": method.description,
                "trustworthiness": f"{method.trustworthiness:.{decimals}f}",
                "trustworthiness_std": f"{method.trustworthiness_std:.{decimals}f}",
                "silhouette": f"{method.silhouette:.{decimals}f}",
                "silhouette_std": f"{method.silhouette_std:.{decimals}f}",
                "best_k": str(method.best_k),
                "best_k_std": f"{method.best_k_std:.{decimals}f}",
                "seed_to_seed_ari_mean": (
                    f"{method.seed_to_seed_ari_mean:.{decimals}f}"
                    if method.seed_to_seed_ari_mean is not None
                    else ""
                ),
                "seed_to_seed_ari_std": (
                    f"{method.seed_to_seed_ari_std:.{decimals}f}"
                    if method.seed_to_seed_ari_std is not None
                    else ""
                ),
                "seed_to_consensus_ari_mean": (
                    f"{method.seed_to_consensus_ari_mean:.{decimals}f}"
                    if method.seed_to_consensus_ari_mean is not None
                    else ""
                ),
                "n_seeds": str(method.n_seeds),
                "runtime_seconds": f"{method.runtime_seconds:.{decimals}f}",
            }
            writer.writerow(row)

    logger.info("Benchmark CSV written to %s (%d methods)", path, len(result.methods))


# ---------------------------------------------------------------------------
# 2. Cross-dataset comparison
# ---------------------------------------------------------------------------


def _get_method_from_benchmark(
    result: BenchmarkResult, method: str,
) -> MethodResult | None:
    """Find a method by name in a benchmark result.

    Parameters
    ----------
    result : Benchmark result to search.
    method : Method name (case-insensitive).

    Returns
    -------
    The matching ``MethodResult``, or ``None`` if not found.
    """
    method_lower = method.lower()
    for m in result.methods:
        if m.method.lower() == method_lower:
            return m
    return None


def cross_dataset_table(
    results: list[BenchmarkResult],
    method: str = "reap",
    decimals: int = 3,
) -> str:
    """Markdown table comparing a single method across multiple datasets.

    Columns: Dataset | n | d_embed | d_UMAP | K | TW | Sil | S2S ARI

    Parameters
    ----------
    results : List of benchmark results, one per dataset.
    method : Name of the method to extract from each benchmark
        (case-insensitive).
    decimals : Number of decimal places for metric values.

    Returns
    -------
    Markdown-formatted table string. Datasets where the method is not found
    are included with ``'--'`` for all metric columns.
    """
    columns = ["Dataset", "n", "d_embed", "d_UMAP", "K", "TW", "Sil", "S2S ARI"]

    rows: list[list[str]] = []
    for bench in results:
        m = _get_method_from_benchmark(bench, method)
        if m is not None:
            row = [
                bench.dataset_name,
                str(bench.n_samples),
                str(bench.n_features),
                str(bench.n_components),
                str(m.best_k),
                _format_metric(m.trustworthiness, m.trustworthiness_std, decimals),
                _format_metric(m.silhouette, m.silhouette_std, decimals),
                (
                    _format_metric(
                        m.seed_to_seed_ari_mean,
                        m.seed_to_seed_ari_std or 0.0,
                        decimals,
                    )
                    if m.seed_to_seed_ari_mean is not None
                    else _na_str()
                ),
            ]
        else:
            logger.warning(
                "Method %r not found in benchmark for %s", method, bench.dataset_name,
            )
            row = [bench.dataset_name, str(bench.n_samples), str(bench.n_features),
                   str(bench.n_components)] + [_na_str()] * 4
        rows.append(row)

    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = []
    header_line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        row_line = "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def cross_dataset_latex(
    results: list[BenchmarkResult],
    method: str = "reap",
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> str:
    r"""LaTeX table comparing a single method across multiple datasets.

    Columns: Dataset | $n$ | $d_{\text{embed}}$ | $d_{\text{UMAP}}$ | $K$ |
    TW | Sil | S2S ARI

    Parameters
    ----------
    results : List of benchmark results, one per dataset.
    method : Name of the method to extract (case-insensitive).
    caption : Optional table caption.
    label : Optional ``\label{}`` key.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    LaTeX source string for a complete ``table`` environment.
    """
    columns_display = [
        "Dataset", "$n$", "$d_{\\text{embed}}$", "$d_{\\text{UMAP}}$",
        "$K$", "TW", "Sil", "S2S ARI",
    ]
    col_spec = "l" + "r" * 3 + "c" * 4

    data_rows: list[str] = []
    for bench in results:
        m = _get_method_from_benchmark(bench, method)
        if m is not None:
            parts = [
                _escape_latex(bench.dataset_name),
                str(bench.n_samples),
                str(bench.n_features),
                str(bench.n_components),
                str(m.best_k),
                _format_metric_latex(m.trustworthiness, m.trustworthiness_std, decimals),
                _format_metric_latex(m.silhouette, m.silhouette_std, decimals),
                (
                    _format_metric_latex(
                        m.seed_to_seed_ari_mean,
                        m.seed_to_seed_ari_std or 0.0,
                        decimals,
                    )
                    if m.seed_to_seed_ari_mean is not None
                    else _na_str()
                ),
            ]
        else:
            parts = [
                _escape_latex(bench.dataset_name),
                str(bench.n_samples),
                str(bench.n_features),
                str(bench.n_components),
            ] + [_na_str()] * 4
        data_rows.append("        " + " & ".join(parts) + " \\\\")

    if caption is None:
        caption = f"Cross-dataset comparison for {_escape_latex(method)} method."
    if label is None:
        label = "tab:cross_dataset_comparison"

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(columns_display) + " \\\\",
        "        \\midrule",
    ]
    lines.extend(data_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 3. Ablation tables
# ---------------------------------------------------------------------------


def ablation_to_markdown(result: SeedAblationResult, decimals: int = 3) -> str:
    """Markdown table showing metrics as a function of seed count.

    Columns: Seeds | TW | Sil | K | S2S ARI | Time

    Parameters
    ----------
    result : Seed ablation result containing per-seed-count metrics.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    Markdown-formatted table string with a header line.
    """
    columns = ["Seeds", "TW", "Sil", "K", "S2S ARI", "Time"]
    header_info = f"**Seed ablation: {result.dataset_name}**"

    rows: list[list[str]] = []
    for point in result.results:
        tw_str = _format_metric(point.trustworthiness, decimals=decimals)
        sil_str = _format_metric(point.silhouette, decimals=decimals)
        k_str = str(point.best_k)
        s2s_str = _format_metric(
            point.seed_to_seed_ari_mean,
            point.seed_to_seed_ari_std,
            decimals,
        )
        time_str = _format_runtime(point.runtime_seconds)
        rows.append([str(point.n_seeds), tw_str, sil_str, k_str, s2s_str, time_str])

    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = [header_info, ""]
    header_line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        row_line = "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def ablation_to_latex(
    result: SeedAblationResult,
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> str:
    r"""LaTeX table showing metrics as a function of seed count.

    Parameters
    ----------
    result : Seed ablation result.
    caption : Optional table caption.
    label : Optional ``\label{}`` key.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    LaTeX source string for a complete ``table`` environment.
    """
    columns_display = ["Seeds", "TW", "Sil", "$K$", "S2S ARI", "Time"]
    col_spec = "r" + "c" * 5

    data_rows: list[str] = []
    for point in result.results:
        tw_str = _format_metric_latex(point.trustworthiness, decimals=decimals)
        sil_str = _format_metric_latex(point.silhouette, decimals=decimals)
        k_str = str(point.best_k)
        s2s_str = _format_metric_latex(
            point.seed_to_seed_ari_mean,
            point.seed_to_seed_ari_std,
            decimals,
        )
        time_str = _format_runtime(point.runtime_seconds)
        parts = [str(point.n_seeds), tw_str, sil_str, k_str, s2s_str, time_str]
        data_rows.append("        " + " & ".join(parts) + " \\\\")

    safe_name = result.dataset_name.lower().replace(" ", "_").replace("-", "_")
    if caption is None:
        caption = f"Seed ablation on {_escape_latex(result.dataset_name)}."
    if label is None:
        label = f"tab:{safe_name}_seed_ablation"

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(columns_display) + " \\\\",
        "        \\midrule",
    ]
    lines.extend(data_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"


def sweep_to_markdown(result: ParameterSweepResult, decimals: int = 3) -> str:
    """Markdown table showing metrics as a function of a swept parameter.

    The first column is the swept parameter name and its values. Remaining
    columns are the standard metric set.

    Parameters
    ----------
    result : Parameter sweep result.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    Markdown-formatted table string with a header line.
    """
    param_col = result.swept_parameter
    columns = [param_col, "TW", "Sil", "K", "S2S ARI", "Time"]
    header_info = f"**Parameter sweep ({param_col}): {result.dataset_name}**"

    rows: list[list[str]] = []
    for point in result.results:
        tw_str = _format_metric(point.trustworthiness, decimals=decimals)
        sil_str = _format_metric(point.silhouette, decimals=decimals)
        k_str = str(point.best_k)
        s2s_str = _format_metric(point.seed_to_seed_ari_mean, decimals=decimals)
        time_str = _format_runtime(point.runtime_seconds)
        # Format the parameter value: use float formatting for decimals,
        # integer formatting for whole numbers.
        param_val = point.parameter_value
        if param_val == int(param_val):
            param_str = str(int(param_val))
        else:
            param_str = f"{param_val:.{decimals}f}".rstrip("0").rstrip(".")
        rows.append([param_str, tw_str, sil_str, k_str, s2s_str, time_str])

    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = [header_info, ""]
    header_line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        row_line = "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def sweep_to_latex(
    result: ParameterSweepResult,
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> str:
    r"""LaTeX table showing metrics as a function of a swept parameter.

    Parameters
    ----------
    result : Parameter sweep result.
    caption : Optional table caption.
    label : Optional ``\label{}`` key.
    decimals : Number of decimal places for metric values.

    Returns
    -------
    LaTeX source string for a complete ``table`` environment.
    """
    param_col = result.swept_parameter
    param_col_latex = _escape_latex(param_col)
    columns_display = [param_col_latex, "TW", "Sil", "$K$", "S2S ARI", "Time"]
    col_spec = "r" + "c" * 5

    data_rows: list[str] = []
    for point in result.results:
        tw_str = _format_metric_latex(point.trustworthiness, decimals=decimals)
        sil_str = _format_metric_latex(point.silhouette, decimals=decimals)
        k_str = str(point.best_k)
        s2s_str = _format_metric_latex(point.seed_to_seed_ari_mean, decimals=decimals)
        time_str = _format_runtime(point.runtime_seconds)
        param_val = point.parameter_value
        if param_val == int(param_val):
            param_str = str(int(param_val))
        else:
            param_str = f"{param_val:.{decimals}f}".rstrip("0").rstrip(".")
        parts = [param_str, tw_str, sil_str, k_str, s2s_str, time_str]
        data_rows.append("        " + " & ".join(parts) + " \\\\")

    safe_name = result.dataset_name.lower().replace(" ", "_").replace("-", "_")
    safe_param = param_col.lower().replace(" ", "_").replace("-", "_")
    if caption is None:
        caption = (
            f"Parameter sweep ({param_col_latex}) on "
            f"{_escape_latex(result.dataset_name)}."
        )
    if label is None:
        label = f"tab:{safe_name}_{safe_param}_sweep"

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(columns_display) + " \\\\",
        "        \\midrule",
    ]
    lines.extend(data_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 4. Bootstrap CI tables
# ---------------------------------------------------------------------------


def bootstrap_to_markdown(cis: list[BootstrapCI], decimals: int = 3) -> str:
    """Markdown table with bootstrap confidence intervals per method per metric.

    Rows are methods; columns are metrics with 95% CIs formatted as
    ``mean [lower, upper]``.

    Parameters
    ----------
    cis : List of bootstrap CI results. Each entry is one (method, metric) pair.
    decimals : Number of decimal places.

    Returns
    -------
    Markdown-formatted table string.
    """
    # Group CIs by method, preserving insertion order
    methods_order: list[str] = []
    metrics_order: list[str] = []
    ci_lookup: dict[tuple[str, str], BootstrapCI] = {}

    for ci in cis:
        if ci.method not in methods_order:
            methods_order.append(ci.method)
        if ci.metric not in metrics_order:
            metrics_order.append(ci.metric)
        ci_lookup[(ci.method, ci.metric)] = ci

    # Build column headers: "Metric [95% CI]"
    n_bootstrap = cis[0].n_bootstrap if cis else 0
    columns = ["Method"] + [f"{m} [95% CI]" for m in metrics_order]

    rows: list[list[str]] = []
    for method in methods_order:
        row = [method]
        for metric in metrics_order:
            key = (method, metric)
            if key in ci_lookup:
                ci = ci_lookup[key]
                row.append(_format_ci(ci.mean, ci.ci_lower, ci.ci_upper, decimals))
            else:
                row.append(_na_str())
        rows.append(row)

    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = []
    if n_bootstrap > 0:
        lines.append(f"*Bootstrap CIs (n={n_bootstrap})*")
        lines.append("")
    header_line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        row_line = "| " + " | ".join(
            row[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"
        lines.append(row_line)

    return "\n".join(lines) + "\n"


def bootstrap_to_latex(
    cis: list[BootstrapCI],
    caption: str | None = None,
    label: str | None = None,
    decimals: int = 3,
) -> str:
    r"""LaTeX table with bootstrap confidence intervals per method per metric.

    Parameters
    ----------
    cis : List of bootstrap CI results.
    caption : Optional table caption.
    label : Optional ``\label{}`` key.
    decimals : Number of decimal places.

    Returns
    -------
    LaTeX source string for a complete ``table`` environment.
    """
    # Group CIs by method
    methods_order: list[str] = []
    metrics_order: list[str] = []
    ci_lookup: dict[tuple[str, str], BootstrapCI] = {}

    for ci in cis:
        if ci.method not in methods_order:
            methods_order.append(ci.method)
        if ci.metric not in metrics_order:
            metrics_order.append(ci.metric)
        ci_lookup[(ci.method, ci.metric)] = ci

    n_bootstrap = cis[0].n_bootstrap if cis else 1000
    columns_display = ["Method"] + [f"{m} [95\\% CI]" for m in metrics_order]
    col_spec = "l" + "c" * len(metrics_order)

    data_rows: list[str] = []
    for method in methods_order:
        parts = [_escape_latex(method)]
        for metric in metrics_order:
            key = (method, metric)
            if key in ci_lookup:
                ci = ci_lookup[key]
                parts.append(
                    _format_ci_latex(ci.mean, ci.ci_lower, ci.ci_upper, decimals)
                )
            else:
                parts.append(_na_str())
        data_rows.append("        " + " & ".join(parts) + " \\\\")

    if caption is None:
        caption = f"Bootstrap confidence intervals ($n$={n_bootstrap})."
    if label is None:
        label = "tab:bootstrap_ci"

    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        " + " & ".join(columns_display) + " \\\\",
        "        \\midrule",
    ]
    lines.extend(data_rows)
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ]

    return "\n".join(lines) + "\n"
