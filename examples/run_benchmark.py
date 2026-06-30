#!/usr/bin/env python
"""Run the REAP benchmark comparison on synthetic or real datasets.

Demonstrates the full publication workflow:
1. Load embeddings (synthetic, AI-art, or Korean forest)
2. Run REAP vs four baselines (single-seed, best-of-N, naive-avg, Procrustes)
3. Run a seed-ablation study
4. Emit publication-quality tables (LaTeX, Markdown, CSV)

Real-dataset params are pinned to the validated configs from each
sibling source project:

- ``ai_art`` — `nn=53, md=0.01, n_components=5, K=20` (WAMA
  ``figures/config_comparison/all_results.json`` "Best combined").
- ``korean_forest`` — `nn=19, md=0.005, n_components=18` (Hye In's
  ``data/consensus_analysis/analysis_metadata.json`` winner).

Examples
--------
Smoke run on Korean forest (5 seeds, < 1 min on laptop)::

    python examples/run_benchmark.py --dataset korean_forest --seeds 5

Default synthetic demo (preserves prior behaviour)::

    python examples/run_benchmark.py

Full 30-seed publication run on AI-art (slow, ~30 min)::

    python examples/run_benchmark.py --dataset ai_art --seeds 30 --ablation

Side effects
------------
- Reads from the REAP cache (real datasets) or generates blobs (synthetic).
- Writes tables under ``--output-dir`` (default ``benchmark_output/<dataset>/``).
- Prints summary tables to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

# Add parent to path if running as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reap.ablation import run_seed_ablation
from reap.benchmarks import run_benchmark
from reap.datasets import (
    DatasetSnapshot,
    load_ai_art,
    load_korean_forest,
)
from reap.reporting import (
    ablation_to_latex,
    ablation_to_markdown,
    benchmark_to_csv,
    benchmark_to_latex,
    benchmark_to_markdown,
    cross_dataset_latex,
    cross_dataset_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Per-dataset benchmark configuration.

    Encapsulates the validated UMAP parameters, K search range, and
    display name to feed into ``run_benchmark`` and ``run_seed_ablation``.
    """

    display_name: str
    n_components: int
    n_neighbors: int
    min_dist: float
    k_range: list[int]
    loader: Callable[[], DatasetSnapshot] | None
    """`None` = synthetic generator path; non-None = real-data loader."""


SYNTHETIC = BenchmarkConfig(
    display_name="Synthetic (demo)",
    n_components=5,
    n_neighbors=15,
    min_dist=0.01,
    k_range=list(range(3, 10)),
    loader=None,
)

AI_ART = BenchmarkConfig(
    display_name="AI-art discourse",
    n_components=5,
    n_neighbors=53,
    min_dist=0.01,
    k_range=list(range(5, 31)),
    loader=load_ai_art,
)

KOREAN_FOREST = BenchmarkConfig(
    display_name="Korean forest policy",
    n_components=18,
    n_neighbors=19,
    min_dist=0.005,
    k_range=list(range(5, 31)),
    loader=load_korean_forest,
)

CONFIGS: dict[str, BenchmarkConfig] = {
    "synthetic": SYNTHETIC,
    "ai_art": AI_ART,
    "korean_forest": KOREAN_FOREST,
}


def generate_synthetic_dataset(
    n_samples: int = 200,
    n_features: int = 384,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate L2-normalized synthetic blobs for the demo path.

    Side effects: none.
    """
    blob = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
        return_centers=False,
    )
    X = np.asarray(blob[0], dtype=np.float64)
    labels = np.asarray(blob[1], dtype=np.int64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms
    return X, labels


def load_dataset_embeddings(name: str) -> tuple[np.ndarray, BenchmarkConfig]:
    """Resolve a dataset name to (X, config). Side effects: cache reads."""
    cfg = CONFIGS[name]
    if cfg.loader is None:
        X, _y = generate_synthetic_dataset()
        return X, cfg
    snap = cfg.loader()
    return snap.embeddings, cfg


def run_for_dataset(
    name: str,
    seeds: list[int],
    output_dir: Path,
    do_ablation: bool,
    ablation_seed_counts: list[int],
):
    """Run the full benchmark workflow for one dataset.

    Side effects: writes LaTeX, Markdown, and CSV under ``output_dir``;
    logs progress; prints tables to stdout.
    """
    logger.info("loading dataset %s", name)
    try:
        X, cfg = load_dataset_embeddings(name)
    except FileNotFoundError as exc:
        logger.error(
            "cannot load %s: %s. Build the cache with `python "
            "scripts/build_datasets.py --dataset %s --source-path ...`",
            name, exc, name,
        )
        raise

    logger.info(
        "%s: n=%d, d=%d, seeds=%d, params=(d=%d, nn=%d, md=%g)",
        cfg.display_name, X.shape[0], X.shape[1], len(seeds),
        cfg.n_components, cfg.n_neighbors, cfg.min_dist,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("running benchmark (REAP vs 4 baselines)…")
    result = run_benchmark(
        X=X,
        dataset_name=cfg.display_name,
        seeds=seeds,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        k_range=cfg.k_range,
    )

    md = benchmark_to_markdown(result)
    print("\n" + md)

    (output_dir / "benchmark_table.md").write_text(md, encoding="utf-8")
    (output_dir / "benchmark_table.tex").write_text(
        benchmark_to_latex(
            result,
            caption=f"Method comparison on {cfg.display_name}",
            label=f"tab:{name}-benchmark",
        ),
        encoding="utf-8",
    )
    benchmark_to_csv(result, str(output_dir / "benchmark_results.csv"))
    logger.info("benchmark tables saved to %s/", output_dir)

    if do_ablation:
        logger.info("running seed ablation at counts=%s", ablation_seed_counts)
        ablation = run_seed_ablation(
            X=X,
            dataset_name=cfg.display_name,
            n_components=cfg.n_components,
            n_neighbors=cfg.n_neighbors,
            min_dist=cfg.min_dist,
            k_range=cfg.k_range,
            seed_counts=ablation_seed_counts,
        )
        abl_md = ablation_to_markdown(ablation)
        print("\n" + abl_md)
        (output_dir / "ablation_table.md").write_text(abl_md, encoding="utf-8")
        (output_dir / "ablation_table.tex").write_text(
            ablation_to_latex(
                ablation,
                caption=f"Effect of seed count on REAP quality ({cfg.display_name})",
                label=f"tab:{name}-ablation",
            ),
            encoding="utf-8",
        )
        logger.info("ablation tables saved to %s/", output_dir)

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Side effects: reads cache or generates blobs, writes tables, prints stdout.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.seeds < 2:
        parser.error("--seeds must be >= 2")

    seeds = list(range(args.seeds))
    output_root = Path(args.output_dir)
    datasets = args.dataset or ["synthetic"]

    results = []
    for name in datasets:
        out_dir = output_root / name
        result = run_for_dataset(
            name=name,
            seeds=seeds,
            output_dir=out_dir,
            do_ablation=args.ablation,
            ablation_seed_counts=args.ablation_seed_counts,
        )
        results.append(result)

    if len(results) > 1:
        cross_md = cross_dataset_table(results)
        print("\n" + cross_md)
        (output_root / "cross_dataset_table.md").write_text(cross_md, encoding="utf-8")
        (output_root / "cross_dataset_table.tex").write_text(
            cross_dataset_latex(
                results,
                caption="REAP performance across datasets",
                label="tab:cross-dataset",
            ),
            encoding="utf-8",
        )
        logger.info("cross-dataset tables saved to %s/", output_root)

    logger.info("done; outputs in %s/", output_root)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run REAP benchmark vs baselines on synthetic or real "
            "(AI-art / Korean forest) datasets."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(CONFIGS),
        help=(
            "Dataset to benchmark. Pass multiple times to run several "
            "datasets and emit a cross-dataset table. Defaults to "
            "'synthetic' if omitted (preserves prior demo behaviour)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds (default 10; use 30 for publication runs).",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_output",
        help="Root directory for output tables (default 'benchmark_output').",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also run a seed-ablation study and emit ablation tables.",
    )
    parser.add_argument(
        "--ablation-seed-counts",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[3, 5, 10],
        help=(
            "Comma-separated seed counts for the ablation "
            "(default '3,5,10'; use '5,10,15,20,25,30' for publication)."
        ),
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
