"""OOS conformal filter — five-variant retention and α-sensitivity.

The pre-registered evaluation protocol (§16.b) freezes five filter
variants for the head-to-head comparison reported in §4.7:

    1. Euclidean (no correction)
    2. Euclidean + pooled correction
    3. Mahalanobis (no correction)
    4. Mahalanobis + pooled correction        (default §3.6)
    5. Mahalanobis + per-president correction

This figure renders a two-panel summary:

Panel (a) — retention at the default α = 0.01 for all five variants on
the Korean forest OOS corpus (1,662 pledges, source
`five_variant_retention.csv`). The pre-registered observed band of
45-55% (protocol §16.d) is drawn as a horizontal span; the default
variant should land inside it.

Panel (b) — α-sensitivity sweep for the empirical-LOO and chi-squared
thresholds with Mahalanobis + pooled correction (source
`higher_retention_exploration.csv`). The α = 0.01 default is marked
with a vertical line. The chi-squared series shows the threshold is
systematically tighter than empirical LOO at matched α (§4.7.2,
manuscript).

Inputs
------
Searches in order:
  1. <repo>/results/korean_forest_oos/five_variant_retention.csv
  2. <repo>/results/korean_forest_oos/higher_retention_exploration.csv
  3. green-narrative/hye_in/for_hyein/park_moon_results_2026-05-07_v2/
     {five_variant_retention.csv, higher_retention_exploration.csv}

The protocol (§16.d) requires the manuscript numbers to come from a
locked snapshot under `results/korean_forest_oos/` for the released
paper; the sibling-project CSVs are honoured here as the v1.2-era
source of record while the snapshot is being prepared. The bundle is
recorded in the cached dataset at
`~/.cache/reap/datasets/korean_forest_oos/2026-05-09/metadata.json`.

Outputs
-------
manuscript/figures/filter_retention.png
manuscript/figures/filter_retention.pdf

Behaviour
---------
* If neither input location contains the expected CSVs, the script logs
  one warning per missing input and skips the affected panel rather
  than rendering with invented numbers.
* When both source files are missing entirely, raises so the missing
  data is surfaced loudly.
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
logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PNG = REPO_ROOT / "manuscript" / "figures" / "filter_retention.png"
OUT_PDF = REPO_ROOT / "manuscript" / "figures" / "filter_retention.pdf"

REAP_RESULTS_DIR = REPO_ROOT / "results" / "korean_forest_oos"
SIBLING_DIR = Path(
    "/Users/echoes/Documents/Berkeley/Research/green-narrative/hye_in/"
    "for_hyein/park_moon_results_2026-05-07_v2"
)

# Pre-registered observed band for overall retention at α = 0.01
# (Mahalanobis + pooled correction); protocol §16.d.
OBSERVED_BAND_LOW = 0.45
OBSERVED_BAND_HIGH = 0.55
DEFAULT_ALPHA = 0.01

# Variant display order (matches protocol §16.b).
VARIANT_ORDER: list[str] = [
    "Euclidean (no correction)",
    "Euclidean + pooled correction",
    "Mahalanobis (no correction)",
    "Mahalanobis + pooled correction",
    "Mahalanobis + per-president correction",
]
DEFAULT_VARIANT = "Mahalanobis + pooled correction"
VARIANT_COLOUR: dict[str, str] = {
    "Euclidean (no correction)":               "#cccccc",
    "Euclidean + pooled correction":           "#9ec5e8",
    "Mahalanobis (no correction)":             "#7f7f7f",
    "Mahalanobis + pooled correction":         "#2ca02c",
    "Mahalanobis + per-president correction":  "#bf9d3a",
}


@dataclass(frozen=True)
class _VariantRetention:
    """One row of the five-variant retention table."""

    variant: str
    overall_pct: float          # percent in [0, 100]
    per_president: dict[str, float]


@dataclass(frozen=True)
class _AlphaPoint:
    """One row of the alpha-sensitivity table."""

    alpha: float
    method: str
    overall_pct: float


def _try_paths(name: str) -> Path | None:
    candidates = [REAP_RESULTS_DIR / name, SIBLING_DIR / name]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_five_variant_retention() -> list[_VariantRetention] | None:
    """Read the five-variant retention CSV.

    Returns None if no source is found; logs a warning.
    """
    path = _try_paths("five_variant_retention.csv")
    if path is None:
        logger.warning(
            "five_variant_retention.csv not found in either %s or %s; "
            "panel (a) will be skipped.",
            REAP_RESULTS_DIR, SIBLING_DIR,
        )
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: list[_VariantRetention] = []
    for r in rows:
        try:
            out.append(_VariantRetention(
                variant=r["variant"],
                overall_pct=float(r["Overall_pct"]),
                per_president={
                    "Lee":  float(r["Lee_pct"]),
                    "Park": float(r["Park_pct"]),
                    "Moon": float(r["Moon_pct"]),
                },
            ))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed row in %s: %s (%s)", path, r, exc)
    logger.info("Loaded %d variants from %s", len(out), path)
    return out


def get_alpha_sensitivity() -> list[_AlphaPoint] | None:
    """Read the α-sensitivity exploration CSV.

    Returns None if no source is found; logs a warning.
    """
    path = _try_paths("higher_retention_exploration.csv")
    if path is None:
        logger.warning(
            "higher_retention_exploration.csv not found in either %s or %s; "
            "panel (b) will be skipped.",
            REAP_RESULTS_DIR, SIBLING_DIR,
        )
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: list[_AlphaPoint] = []
    for r in rows:
        try:
            out.append(_AlphaPoint(
                alpha=float(r["alpha"]),
                method=r["method"],
                overall_pct=float(r["Overall_pct"]),
            ))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed row in %s: %s (%s)", path, r, exc)
    logger.info("Loaded %d α-sweep points from %s", len(out), path)
    return out


def plot_five_variant_bar(
    ax: Axes,
    rows: list[_VariantRetention],
) -> None:
    """Bar chart of five-variant overall retention at α = 0.01."""
    ordered_rows = [r for v in VARIANT_ORDER for r in rows if r.variant == v]
    labels = [r.variant for r in ordered_rows]
    means = [r.overall_pct / 100.0 for r in ordered_rows]
    colours = [VARIANT_COLOUR.get(v, "#777") for v in labels]
    x = np.arange(len(ordered_rows))

    # Observed band (protocol §16.d) drawn first so bars sit on top.
    ax.axhspan(
        OBSERVED_BAND_LOW, OBSERVED_BAND_HIGH,
        alpha=0.18, color="#2ca02c",
        label=f"observed band {int(OBSERVED_BAND_LOW * 100)}-"
              f"{int(OBSERVED_BAND_HIGH * 100)}% (protocol §16.d)",
        zorder=0,
    )

    bars = ax.bar(x, means, color=colours, edgecolor="#222",
                  linewidth=0.6, alpha=0.95, zorder=1)
    for bar_obj, val in zip(bars, means):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.012,
            f"{val * 100:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    # Per-president dots overlaid as a sanity check on the cross-president
    # spread documented in §4.7.
    pres_offsets = {"Lee": -0.15, "Park": 0.0, "Moon": 0.15}
    pres_markers = {"Lee": "o", "Park": "s", "Moon": "^"}
    for r, xi in zip(ordered_rows, x):
        for pres, pct in r.per_president.items():
            ax.scatter(
                float(xi) + pres_offsets[pres], pct / 100.0,
                marker=pres_markers[pres], s=22,
                color="#333", alpha=0.85, zorder=3,
                label=f"{pres}" if (r is ordered_rows[0]) else None,
            )

    ax.set_xticks(x)
    short_labels = [
        v.replace(" + ", "\n+ ").replace("Mahalanobis", "Mahal.")
        for v in labels
    ]
    ax.set_xticklabels(short_labels, fontsize=8.5, rotation=0)
    ax.set_ylabel("Retention fraction (α = 0.01)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(a) Five-variant retention on Korean forest OOS")

    # Highlight the default variant with a star above the bar.
    for r, xi in zip(ordered_rows, x):
        if r.variant == DEFAULT_VARIANT:
            ax.text(float(xi), r.overall_pct / 100.0 + 0.10, "default",
                    ha="center", fontsize=8, color="#2ca02c",
                    fontweight="bold")

    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_alpha_sensitivity(
    ax: Axes,
    points: list[_AlphaPoint],
) -> None:
    """Line plot of retention vs α for empirical-LOO and chi-squared."""
    methods_to_plot = [
        ("Mahalanobis+pooled (empirical LOO)", "#2ca02c", "o"),
        ("Mahalanobis+pooled (chi-squared theoretical)", "#1f77b4", "s"),
    ]
    for method, colour, marker in methods_to_plot:
        method_points = sorted(
            [p for p in points if p.method == method],
            key=lambda p: p.alpha,
        )
        if not method_points:
            logger.warning("No α-sweep rows for method %s", method)
            continue
        xs = np.array([p.alpha for p in method_points])
        ys = np.array([p.overall_pct / 100.0 for p in method_points])
        # Plot in reverse-α order so x ascends left to right.
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=marker,
                color=colour, label=method.replace("Mahalanobis+pooled", "Mahal.+pooled"))

    # Default α marker.
    ax.axvline(DEFAULT_ALPHA, color="#aaa", linestyle="--", linewidth=1,
               label=f"default α = {DEFAULT_ALPHA}")
    ax.axhspan(OBSERVED_BAND_LOW, OBSERVED_BAND_HIGH,
               alpha=0.15, color="#2ca02c", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("Retention fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(b) α-sensitivity (Mahal. + pooled)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    np.random.seed(0)
    rows = get_five_variant_retention()
    sweep = get_alpha_sensitivity()
    if rows is None and sweep is None:
        raise FileNotFoundError(
            "Neither five_variant_retention.csv nor "
            "higher_retention_exploration.csv is available. "
            "See protocol §16.d for the expected snapshot path."
        )

    n_panels = sum(x is not None for x in (rows, sweep))
    fig, axes_obj = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 5.0),
        squeeze=False,
    )
    axes = list(axes_obj[0])
    next_ax = 0
    if rows is not None:
        plot_five_variant_bar(axes[next_ax], rows)
        next_ax += 1
    if sweep is not None:
        plot_alpha_sensitivity(axes[next_ax], sweep)
        next_ax += 1

    suptitle = (
        "Korean forest OOS conformal filter — pre-registered five-variant "
        "comparison and α-sensitivity"
    )
    fig.suptitle(suptitle, fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s and %s", OUT_PNG, OUT_PDF)


if __name__ == "__main__":
    main()
