"""AI-art metrics bar chart (REAP vs baselines, headline consensus metrics).

Reads `results/ai_art/combined_set_{A,B,C}/all_methods.csv` and plots
each method's cross-set mean ± SD on every headline metric present.

Inputs
------
results/ai_art/combined_set_{A,B,C}/all_methods.csv

Outputs
-------
manuscript/figures/metrics_bars_ai_art.png
manuscript/figures/metrics_bars_ai_art.pdf

Behaviour
---------
* If a requested column (e.g., `topic_coherence_umass`,
  `topic_coherence_cv`, BERTopic rows) is absent or empty in the CSVs,
  the script logs a one-line warning via `logging.getLogger(__name__)`
  and plots only the metrics/methods that are available.
* Cross-set SD is reported with N=3 — consistent with Table 2b in
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
DATASET_DIR = REPO_ROOT / "results" / "ai_art"
OUT_PNG = REPO_ROOT / "manuscript" / "figures" / "metrics_bars_ai_art.png"
OUT_PDF = REPO_ROOT / "manuscript" / "figures" / "metrics_bars_ai_art.pdf"


def main() -> None:
    np.random.seed(0)
    plot_dataset_metrics_bars(
        dataset_dir=DATASET_DIR,
        title="AI-art discourse (e5-large-v2, $N=1{,}736$)",
        out_png=OUT_PNG,
        out_pdf=OUT_PDF,
    )


if __name__ == "__main__":
    main()
