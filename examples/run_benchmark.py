#!/usr/bin/env python
"""Example: run REAP benchmark comparison on a dataset.

This script demonstrates the full publication workflow:
1. Load or generate embeddings
2. Run REAP vs 4 baselines
3. Run seed ablation analysis
4. Generate publication-quality tables (LaTeX + Markdown)

Usage:
    python examples/run_benchmark.py

To use with your own data, replace the synthetic blob generation
with your actual embeddings:

    embeddings = np.load("path/to/embeddings.npy")
    result = run_benchmark(
        X=embeddings,
        dataset_name="My Dataset",
        seeds=list(range(30)),
        n_components=18,
        n_neighbors=19,
        min_dist=0.005,
        k_range=list(range(3, 15)),
    )
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs

# Add parent to path if running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reap.ablation import run_seed_ablation
from reap.benchmarks import run_benchmark
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


def generate_synthetic_dataset(
    n_samples: int = 500,
    n_features: int = 384,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sentence embeddings for demonstration.

    Parameters
    ----------
    n_samples : Number of data points.
    n_features : Embedding dimensionality (384 = MiniLM, 1024 = e5-large).
    n_clusters : Number of ground-truth clusters.
    random_state : Reproducibility seed.

    Returns
    -------
    (X, labels) where X is (n_samples, n_features) normalized embeddings.
    """
    X, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=random_state,
    )
    # L2-normalize to mimic sentence embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    return X, labels


def main() -> None:
    """Run the full benchmark workflow."""
    output_dir = Path("benchmark_output")
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Generate synthetic data (replace with real embeddings)
    # ---------------------------------------------------------------
    logger.info("Generating synthetic dataset...")
    X, _labels = generate_synthetic_dataset(
        n_samples=200, n_features=384, n_clusters=5
    )
    logger.info("Dataset: %d samples, %d features", X.shape[0], X.shape[1])

    # ---------------------------------------------------------------
    # Step 2: Run benchmark (REAP vs 4 baselines)
    # ---------------------------------------------------------------
    logger.info("Running benchmark comparison...")
    result = run_benchmark(
        X=X,
        dataset_name="Synthetic (demo)",
        seeds=list(range(10)),  # use 30 for publication
        n_components=5,
        n_neighbors=15,
        min_dist=0.01,
        k_range=list(range(3, 10)),
    )

    # Print markdown table
    md_table = benchmark_to_markdown(result)
    print("\n" + md_table)

    # Save LaTeX table
    latex_table = benchmark_to_latex(
        result,
        caption="Method comparison on synthetic data",
        label="tab:synthetic-benchmark",
    )
    latex_path = output_dir / "benchmark_table.tex"
    latex_path.write_text(latex_table)
    logger.info("LaTeX table saved to %s", latex_path)

    # Save CSV
    csv_path = output_dir / "benchmark_results.csv"
    benchmark_to_csv(result, str(csv_path))
    logger.info("CSV saved to %s", csv_path)

    # ---------------------------------------------------------------
    # Step 3: Seed ablation (how many seeds are needed?)
    # ---------------------------------------------------------------
    logger.info("Running seed ablation...")
    ablation = run_seed_ablation(
        X=X,
        dataset_name="Synthetic (demo)",
        n_components=5,
        n_neighbors=15,
        min_dist=0.01,
        k_range=list(range(3, 10)),
        seed_counts=[3, 5, 10],  # use [5, 10, 15, 20, 25, 30] for publication
    )

    # Print ablation table
    abl_md = ablation_to_markdown(ablation)
    print("\n" + abl_md)

    # Save ablation LaTeX
    abl_latex = ablation_to_latex(
        ablation,
        caption="Effect of seed count on REAP quality",
        label="tab:seed-ablation",
    )
    abl_path = output_dir / "ablation_table.tex"
    abl_path.write_text(abl_latex)
    logger.info("Ablation LaTeX saved to %s", abl_path)

    # ---------------------------------------------------------------
    # Step 4: Cross-dataset comparison (with multiple datasets)
    # ---------------------------------------------------------------
    # Run a second dataset for cross-comparison demo
    X2, _ = generate_synthetic_dataset(
        n_samples=150, n_features=32, n_clusters=3, random_state=99
    )
    result2 = run_benchmark(
        X=X2,
        dataset_name="Synthetic-2 (demo)",
        seeds=list(range(10)),
        n_components=3,
        n_neighbors=10,
        min_dist=0.05,
        k_range=list(range(2, 6)),
    )

    # Cross-dataset table
    cross_md = cross_dataset_table([result, result2])
    print("\n" + cross_md)

    cross_latex = cross_dataset_latex(
        [result, result2],
        caption="REAP performance across datasets",
        label="tab:cross-dataset",
    )
    cross_path = output_dir / "cross_dataset_table.tex"
    cross_path.write_text(cross_latex)
    logger.info("Cross-dataset LaTeX saved to %s", cross_path)

    logger.info("All outputs saved to %s/", output_dir)


if __name__ == "__main__":
    main()
