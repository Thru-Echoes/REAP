"""Run the Parametric UMAP baseline only, preserving the other 6 methods' results.

Mirrors ``scripts/run_bertopic_only.py`` exactly except for the target method.
Same merge semantics: writes per-method artifacts to
``results/<dataset>/parametric_umap/set_<X>/`` and merges the row into
``results/<dataset>/combined_set_<X>/all_methods.csv``, preserving every other
method's row byte-identical.

Usage
-----
Smoke test (5 seeds of Set A, ~5-8 min)::

    python scripts/run_parametric_umap_only.py --dataset korean_forest \
        --seed-sets A --quick --skip-reliability

Full sweep (both datasets × A/B/C, ~30-60 min wall depending on epoch count)::

    python scripts/run_parametric_umap_only.py --dataset korean_forest --dataset ai_art

Side effects
------------
- Reads from ``~/.cache/reap/datasets/`` (cached snapshots).
- Writes to ``--output-dir`` (default ``results/``); merges into
  ``combined_set_<X>/all_methods.csv``.
- CPU/GPU-bound (TensorFlow). Set ``KMP_DUPLICATE_LIB_OK=TRUE`` on macOS.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Reuse the helpers from the paper-grade runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bertopic_only import (  # noqa: E402  (sibling script reuse)
    _merge_method_row_into_combined_csv,  # type: ignore[attr-defined]
    run_one_dataset_seed_set_method_only,
)
from run_paper_benchmark import (  # noqa: E402
    DATASET_CONFIGS,
    SEED_MANIFEST_PATH,
    _load_seed_sets,
    compute_between_set_reliability,
    compute_pairwise_tests,
)

# Add src/ for the package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reap.benchmarks import ALL_METHODS  # noqa: E402

logger = logging.getLogger(__name__)

ONLY_METHOD = "parametric_umap"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Parametric UMAP baseline only; merge into existing "
            "all_methods.csv."
        ),
    )
    parser.add_argument(
        "--dataset", action="append", choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset(s) to benchmark. Pass multiple times for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output root directory (default 'results/').",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke mode: first 5 seeds of each requested set.",
    )
    parser.add_argument(
        "--seed-sets", default="A,B,C",
        help="Comma-separated seed-set names (default 'A,B,C').",
    )
    parser.add_argument(
        "--skip-reliability", action="store_true",
        help="Skip recomputing between-set reliability + pairwise tests after the sweep.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if ONLY_METHOD not in ALL_METHODS:
        logger.error(
            "method %r is not in ALL_METHODS=%s — abort",
            ONLY_METHOD, ALL_METHODS,
        )
        return 2

    try:
        manifest_sets = _load_seed_sets(SEED_MANIFEST_PATH)
    except Exception as exc:
        logger.error("could not load seed manifest at %s: %s", SEED_MANIFEST_PATH, exc)
        return 2

    output_dir = Path(args.output_dir)
    requested_sets = [s.strip() for s in args.seed_sets.split(",") if s.strip()]

    for dataset_name in args.dataset:
        if dataset_name == "synthetic":
            logger.error(
                "synthetic dataset is not supported here — "
                "use run_paper_benchmark.py for that"
            )
            return 2
        for set_name in requested_sets:
            if set_name not in manifest_sets:
                logger.error(
                    "unknown seed set %r (available: %s)",
                    set_name, list(manifest_sets),
                )
                return 2
            seeds = manifest_sets[set_name]
            if args.quick:
                seeds = seeds[:5]
                logger.info(
                    "--quick: using first %d seeds of Set %s", len(seeds), set_name,
                )
            run_one_dataset_seed_set_method_only(
                dataset_name, set_name, seeds, output_dir, ONLY_METHOD,
            )

        if not args.skip_reliability:
            dataset_root = output_dir / dataset_name
            compute_between_set_reliability(dataset_root, list(ALL_METHODS), requested_sets)
            compute_pairwise_tests(
                dataset_root, list(ALL_METHODS), requested_sets, reference="reap",
            )

    logger.info("Parametric UMAP sweep complete; results in %s/", output_dir)
    return 0


# Silence unused-import warning in case ruff doesn't pick it up
_ = _merge_method_row_into_combined_csv


if __name__ == "__main__":
    raise SystemExit(main())
