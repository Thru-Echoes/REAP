"""Collect the final-step diagnosis into one table (CSV and Markdown).

Reads ``set_summary.json`` and ``cross_set_summary.json`` for every dataset
named on the command line, under the diagnosis output folder written by
``run_final_step_diagnosis.py``, plus the benchmark's own
``between_set_reliability.csv`` for the baselines' between-set ARI. A
missing set or file raises: the table is never built from a partial run.

Outputs, in the diagnosis folder: ``diagnosis_summary.csv`` (full precision)
and ``diagnosis_summary.md`` (three decimals, for the protocol amendment).

Usage::

    python scripts/experiments/summarize_final_step_diagnosis.py \\
        twenty_newsgroups_reference=results/twenty_newsgroups_reference \\
        calls_presidential_e5=results/corporate_presidential/reap_runs/full/calls_presidential_e5

Each argument is ``<dataset>=<benchmark output folder>``, relative to the
research root unless absolute.

Side effects: reads JSON and CSV files; writes the two summary files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from research_root import get_research_root  # noqa: E402

SETS = ("A", "B", "C")
PAIRS = ("A-B", "A-C", "B-C")
BASELINES = ("best_of_n", "naive_average", "procrustes", "bertopic", "parametric_umap")


def get_between_set_means(csv_path: Path) -> dict[str, float]:
    """Mean between-set ARI per method over the three set pairs.

    Raises when a method listed in the file does not have all three pairs.
    Side effects: reads the file.
    """
    by_method: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_method.setdefault(row["method"], []).append(float(row["ari"]))
    for method, values in by_method.items():
        if len(values) != len(PAIRS):
            raise ValueError(f"{csv_path}: {method} has {len(values)} set pairs, expected 3")
    return {m: float(np.mean(v)) for m, v in by_method.items()}


def get_dataset_rows(dataset: str, diag_dir: Path, benchmark_dir: Path) -> list[dict[str, Any]]:
    """One row per seed set plus one cross-set row for a dataset.

    Raises ``FileNotFoundError`` when any expected output is missing.
    """
    rows: list[dict[str, Any]] = []
    for s in SETS:
        path = diag_dir / f"set_{s}" / "set_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing diagnosis output: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary["reproduces_benchmark_labels_exactly"]:
            raise ValueError(f"{path}: the seed-42 run did not reproduce the benchmark")
        arms = summary["arms"]
        joint_ks = [int(k) for k in arms["joint"]["k_counts"]]
        rows.append({
            "dataset": dataset,
            "scope": f"set {s}",
            "reference_k": summary["reference_k"],
            "joint_k_min": min(joint_ks),
            "joint_k_max": max(joint_ks),
            "within_joint_mean": arms["joint"]["ari_chosen_k"]["mean"],
            "within_joint_min": arms["joint"]["ari_chosen_k"]["min"],
            "within_map_only_mean": arms["map_only"]["ari_chosen_k"]["mean"],
            "within_cluster_only_mean": arms["cluster_only"]["ari_chosen_k"]["mean"],
            "within_joint_reference_k_mean": arms["joint"]["ari_reference_k"]["mean"],
            "within_cluster_only_reference_k_mean": arms["cluster_only"]["ari_reference_k"]["mean"],
        })
    cross_path = diag_dir / "cross_set_summary.json"
    if not cross_path.exists():
        raise FileNotFoundError(f"missing diagnosis output: {cross_path}")
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    missing = [p for p in PAIRS if p not in cross]
    if missing:
        raise ValueError(f"{cross_path}: missing set pairs {missing}")
    means = get_between_set_means(benchmark_dir / "between_set_reliability.csv")
    baselines = {m: v for m, v in means.items() if m in BASELINES}
    best = max(baselines, key=lambda m: baselines[m])
    rows.append({
        "dataset": dataset,
        "scope": "between sets",
        "benchmark_reap_mean": float(np.mean([cross[p]["benchmark_seed42"] for p in PAIRS])),
        "cross_matched_mean": float(np.mean([cross[p]["matched_final_seeds"]["mean"]
                                             for p in PAIRS])),
        "cross_unmatched_mean": float(np.mean([cross[p]["unmatched_final_seeds"]["mean"]
                                               for p in PAIRS])),
        "best_baseline": best,
        "best_baseline_mean": baselines[best],
    })
    return rows


FIELDS = [
    "dataset", "scope", "reference_k", "joint_k_min", "joint_k_max",
    "within_joint_mean", "within_joint_min", "within_map_only_mean",
    "within_cluster_only_mean", "within_joint_reference_k_mean",
    "within_cluster_only_reference_k_mean", "benchmark_reap_mean",
    "cross_matched_mean", "cross_unmatched_mean", "best_baseline", "best_baseline_mean",
]


def _fmt(value: Any) -> str:
    """Three decimals for floats, plain text otherwise, blank for missing."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def get_markdown(rows: list[dict[str, Any]]) -> str:
    """Two Markdown tables: within one averaged matrix, and between seed sets."""
    lines = [
        "Within one averaged matrix (mean pairwise ARI over the 45 pairs of 10 final-step runs):",
        "",
        "| Dataset | Set | Seed-42 K | K range (joint) | Joint | Joint min | Map only "
        "| Clustering only | Joint at seed-42 K | Clustering only at seed-42 K |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["scope"].startswith("set"):
            lines.append(
                f"| {r['dataset']} | {r['scope'][-1]} | {r['reference_k']} | "
                f"{r['joint_k_min']}–{r['joint_k_max']} | {_fmt(r['within_joint_mean'])} | "
                f"{_fmt(r['within_joint_min'])} | {_fmt(r['within_map_only_mean'])} | "
                f"{_fmt(r['within_cluster_only_mean'])} | "
                f"{_fmt(r['within_joint_reference_k_mean'])} | "
                f"{_fmt(r['within_cluster_only_reference_k_mean'])} |"
            )
    lines += [
        "",
        "Between seed sets (mean over the pairs A–B, A–C, B–C):",
        "",
        "| Dataset | REAP, benchmark (seed 42) | Joint runs, same final seed "
        "| Joint runs, different final seeds | Best baseline | Best baseline ARI |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["scope"] == "between sets":
            lines.append(
                f"| {r['dataset']} | {_fmt(r['benchmark_reap_mean'])} | "
                f"{_fmt(r['cross_matched_mean'])} | {_fmt(r['cross_unmatched_mean'])} | "
                f"{r['best_baseline']} | {_fmt(r['best_baseline_mean'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Build the summary; return 0. Raises on any missing input."""
    p = argparse.ArgumentParser(description="Summarise the final-step diagnosis.")
    p.add_argument("datasets", nargs="+", help="<dataset>=<benchmark output folder>")
    p.add_argument("--research-root", default=None)
    p.add_argument("--diagnosis-dir", default=None)
    args = p.parse_args(argv)
    root = get_research_root(args.research_root)
    diag_root = (Path(args.diagnosis_dir) if args.diagnosis_dir
                 else root / "results" / "final_step_repeatability" / "diagnosis")
    rows: list[dict[str, Any]] = []
    for item in args.datasets:
        dataset, _, bench = item.partition("=")
        if not bench:
            raise ValueError(f"expected <dataset>=<folder>, got {item!r}")
        bench_dir = Path(bench) if Path(bench).is_absolute() else root / bench
        rows += get_dataset_rows(dataset, diag_root / dataset, bench_dir)
    with (diag_root / "diagnosis_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    (diag_root / "diagnosis_summary.md").write_text(get_markdown(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
