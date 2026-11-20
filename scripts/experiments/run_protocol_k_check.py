"""Between-set agreement under the protocol's K rule, from saved benchmark maps.

Protocol §9 picks K once, on REAP's Set A consensus map, and freezes it for
every method and seed set. The benchmark instead picks K by silhouette
separately for each method and set. Because trustworthiness does not depend
on the labels, §9's score (0.5 × silhouette + 0.5 × trustworthiness) is
maximised over K by silhouette alone on that one map, so the protocol's K is
the K REAP's benchmark run chose on Set A. This check clusters every
method's saved consensus map for sets A, B and C at that K, with the
harness's own k-means (``random_state=42``, ``n_init="auto"``, as
``reap.benchmarks._cluster_embedding``), and reports the mean between-set ARI
next to the benchmark's per-set-K value.

Outputs ``protocol_k_check.csv`` (full precision) under
``<research root>/results/final_step_repeatability/diagnosis/``.

Usage::

    python scripts/experiments/run_protocol_k_check.py \\
        twenty_newsgroups_reference=results/twenty_newsgroups_reference ...

Each argument is ``<dataset>=<benchmark output folder>``. A missing method
map or set raises.

Side effects: reads saved ``.npy`` and CSV files; writes one CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_step import BENCHMARK_FINAL_SEED, get_kmeans_labels  # noqa: E402
from research_root import get_research_root  # noqa: E402
from summarize_final_step_diagnosis import get_between_set_means  # noqa: E402

METHODS = ("reap", "best_of_n", "naive_average", "procrustes", "bertopic", "parametric_umap")
SETS = ("A", "B", "C")


def get_protocol_k(benchmark_dir: Path) -> int:
    """K of REAP's saved Set A consensus labels (the protocol's frozen K)."""
    labels = np.load(benchmark_dir / "reap" / "set_A" / "consensus_labels.npy")
    return int(len(np.unique(labels)))


def get_rows(dataset: str, benchmark_dir: Path) -> list[dict[str, object]]:
    """One row per method: frozen-K and benchmark between-set ARI.

    Raises ``FileNotFoundError`` when a method's consensus map is missing for
    any set.
    """
    k = get_protocol_k(benchmark_dir)
    bench = get_between_set_means(benchmark_dir / "between_set_reliability.csv")
    rows: list[dict[str, object]] = []
    for method in METHODS:
        labels = {}
        for s in SETS:
            path = benchmark_dir / method / f"set_{s}" / "consensus_embedding.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing consensus map: {path}")
            labels[s] = get_kmeans_labels(np.load(path), k, BENCHMARK_FINAL_SEED)
        aris = [float(adjusted_rand_score(labels[a], labels[b]))
                for a, b in itertools.combinations(SETS, 2)]
        rows.append({
            "dataset": dataset, "method": method, "protocol_k": k,
            "between_set_ari_protocol_k": repr(float(np.mean(aris))),
            "between_set_ari_protocol_k_min": repr(min(aris)),
            "between_set_ari_benchmark": repr(bench[method]),
        })
    return rows


def get_averaged_matrix(dataset: str, benchmark_dir: Path, diagnosis_dir: Path,
                        seed_set: str) -> np.ndarray:
    """REAP's averaged matrix: the benchmark's saved copy, else the diagnosis's rebuilt one.

    The rebuilt copy is only written after its seed-42 final step reproduced
    the saved labels exactly. Raises ``FileNotFoundError`` when neither exists.
    """
    for path in (benchmark_dir / "reap" / f"set_{seed_set}" / "consensus_distance_matrix.npy",
                 diagnosis_dir / dataset / f"set_{seed_set}" / "consensus_distance_matrix.npy"):
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"no averaged matrix for {dataset} set {seed_set}")


def get_rows_by_final_seed(
    dataset: str, benchmark_dir: Path, diagnosis_dir: Path, final_seeds: list[int]
) -> list[dict[str, object]]:
    """Protocol-K between-set ARI for each final seed, per method.

    For final seed s, REAP's final map is rebuilt from its averaged matrix
    with UMAP seed s and clustered at the protocol K with k-means seed s; each
    baseline's saved map is clustered at that K with k-means seed s. Seed 42
    reproduces ``get_rows``.
    """
    from final_step import get_final_embedding
    from run_final_step_diagnosis import get_config

    k = get_protocol_k(benchmark_dir)
    config = get_config(dataset, benchmark_dir / "reap" / "set_A")
    saved_maps = {(m, s): np.load(benchmark_dir / m / f"set_{s}" / "consensus_embedding.npy")
                  for m in METHODS if m != "reap" for s in SETS}
    matrices = {s: get_averaged_matrix(dataset, benchmark_dir, diagnosis_dir, s) for s in SETS}
    rows: list[dict[str, object]] = []
    for seed in final_seeds:
        reap_maps = {s: get_final_embedding(matrices[s], config, seed) for s in SETS}
        for method in METHODS:
            labels = {s: get_kmeans_labels(reap_maps[s] if method == "reap"
                                           else saved_maps[(method, s)], k, seed)
                      for s in SETS}
            aris = [float(adjusted_rand_score(labels[a], labels[b]))
                    for a, b in itertools.combinations(SETS, 2)]
            rows.append({"dataset": dataset, "method": method, "protocol_k": k,
                         "final_seed": seed, "between_set_ari": repr(float(np.mean(aris)))})
    return rows


def main(argv: list[str] | None = None) -> int:
    """Write the protocol-K table; return 0. Raises on any missing input."""
    p = argparse.ArgumentParser(description="Between-set ARI under the protocol K rule.")
    p.add_argument("datasets", nargs="+", help="<dataset>=<benchmark output folder>")
    p.add_argument("--research-root", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--final-seeds", nargs="+", type=int, default=None,
                   help="Also repeat the check for each of these final seeds.")
    args = p.parse_args(argv)
    root = get_research_root(args.research_root)
    diag_dir = root / "results" / "final_step_repeatability" / "diagnosis"
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for item in args.datasets:
        dataset, _, folder = item.partition("=")
        if not folder:
            raise ValueError(f"expected <dataset>=<folder>, got {item!r}")
        bench_dir = Path(folder) if Path(folder).is_absolute() else root / folder
        rows += get_rows(dataset, bench_dir)
        if args.final_seeds:
            seed_rows += get_rows_by_final_seed(dataset, bench_dir, diag_dir, args.final_seeds)
    if seed_rows:
        with (diag_dir / "protocol_k_by_final_seed.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
    out = Path(args.output) if args.output else diag_dir / "protocol_k_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
