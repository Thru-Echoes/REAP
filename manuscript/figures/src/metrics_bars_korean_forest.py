"""Korean-forest metrics bar chart (REAP vs baselines, headline consensus metrics).

Reads `results/korean_forest/combined_set_{A,B,C}/all_methods.csv` and
plots each method's cross-set mean ± SD on every headline metric present.

Inputs
------
results/korean_forest/combined_set_{A,B,C}/all_methods.csv

Outputs
-------
manuscript/figures/metrics_bars_korean_forest.png
manuscript/figures/metrics_bars_korean_forest.pdf

Behaviour
---------
* If a requested column (e.g., `topic_coherence_umass`,
  `topic_coherence_cv`, BERTopic rows) is absent or empty in the CSVs,
  the script logs a one-line warning via `logging.getLogger(__name__)`
  and plots only the metrics/methods that are available.
* Cross-set SD is reported with N=3 — consistent with Table 1b in
  manuscript/sections/experiments.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from _bench_plot import plot_dataset_metrics_bars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "results" / "korean_forest"
OUT_PNG = REPO_ROOT / "manuscript" / "figures" / "metrics_bars_korean_forest.png"
OUT_PDF = REPO_ROOT / "manuscript" / "figures" / "metrics_bars_korean_forest.pdf"


def main() -> None:
    np.random.seed(0)
    plot_dataset_metrics_bars(
        dataset_dir=DATASET_DIR,
        title="Korean forest policy (MiniLM, $N=905$)",
        out_png=OUT_PNG,
        out_pdf=OUT_PDF,
    )


if __name__ == "__main__":
    main()
