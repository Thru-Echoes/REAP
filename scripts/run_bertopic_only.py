"""Run the BERTopic baseline only, preserving the other 5 methods' results.

This script exists because:

1. The 5 non-BERTopic methods (``single_seed``, ``best_of_n``,
   ``naive_average``, ``procrustes``, ``reap``) have already been benchmarked
   on AI-art and Korean forest at 30 seeds × 3 seed sets. The resulting CSV
   values are quoted verbatim in ``manuscript/sections/experiments.md``.
2. ``scripts/run_paper_benchmark.py`` overwrites
   ``results/<dataset>/combined_set_<X>/all_methods.csv`` on every
   invocation. Re-running the full sweep risks tiny floating-point drift
   relative to the already-quoted numbers, which would silently desync the
   prose from the data on disk.

This script writes only:

- ``results/<dataset>/bertopic/set_<X>/`` — new directory, no conflict.
- ``results/<dataset>/combined_set_<X>/all_methods.csv`` — read existing
  file, drop any prior ``bertopic`` row, append the new ``bertopic`` row,
  write back. The other 5 methods' rows are byte-identical preserved.

After all requested seed sets for a dataset are processed, the script
re-runs ``compute_between_set_reliability`` and ``compute_pairwise_tests``
so those summary tables reflect the now-6-method state.

Usage
-----
Smoke test (5 seeds of Set A only, ~3-5 min)::

    python scripts/run_bertopic_only.py --dataset korean_forest --seed-sets A --quick --skip-reliability

Full sweep (both datasets × A/B/C, ~1-2 hours)::

    python scripts/run_bertopic_only.py --dataset korean_forest --dataset ai_art

Side effects
------------
- Reads from ``~/.cache/reap/datasets/`` (cached snapshots).
- Writes to ``--output-dir`` (default ``results/``); merges into
  ``combined_set_<X>/all_methods.csv``.
- CPU-bound. n=905 Korean forest at 30 seeds takes ~5-10 min/set;
  n=1736 AI-art takes ~10-20 min/set.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import sys
from pathlib import Path

# Reuse all the helpers from the paper-grade runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paper_benchmark import (  # noqa: E402
    DATASET_CONFIGS,
    SEED_MANIFEST_PATH,
    _load_seed_sets,
    _persist_method_artifacts,
    _write_all_methods_summary,
    compute_between_set_reliability,
    compute_pairwise_tests,
)

# Add src/ for the package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reap.benchmarks import (  # noqa: E402
    ALL_METHODS,
    BenchmarkResult,
    run_benchmark_with_artifacts,
)
from reap.datasets import DatasetSnapshot  # noqa: E402

logger = logging.getLogger(__name__)

ONLY_METHOD = "bertopic"


def _merge_method_row_into_combined_csv(
    result: BenchmarkResult,
    combined_csv: Path,
    method_name: str,
) -> None:
    """Merge the single-method ``result`` row into an existing combined CSV.

    Pure-ish (filesystem write): reads ``combined_csv`` if present, drops any
    existing row whose ``method`` column equals ``method_name``, then writes
    a fresh row for ``method_name`` produced from ``result``. The remaining
    rows from the existing CSV are preserved byte-identical.

    If ``combined_csv`` does not exist, the function falls through to
    :func:`_write_all_methods_summary` which writes a brand-new file
    containing only the new method's row.
    """
    if not combined_csv.exists():
        logger.info(
            "%s does not exist — writing fresh single-method CSV via _write_all_methods_summary",
            combined_csv,
        )
        combined_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_all_methods_summary(result, combined_csv)
        return

    # Read the existing rows, preserving order and column set.
    with combined_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows: list[dict[str, str]] = list(reader)
        fieldnames: list[str] = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"{combined_csv} has no header — refusing to merge.")

    # Write the new method to a temp CSV via _write_all_methods_summary so we
    # produce a row that exactly matches the harness's canonical formatting.
    tmp_path = combined_csv.with_suffix(".tmp_new_method.csv")
    _write_all_methods_summary(result, tmp_path)
    with tmp_path.open(encoding="utf-8") as f:
        new_reader = csv.DictReader(f)
        new_rows: list[dict[str, str]] = list(new_reader)
        new_fieldnames: list[str] = list(new_reader.fieldnames or [])
    tmp_path.unlink()

    # Union the column sets in case the harness has added new columns since
    # the existing CSV was written (we don't want to lose any column).
    union_fields: list[str] = list(fieldnames)
    for fn in new_fieldnames:
        if fn not in union_fields:
            union_fields.append(fn)

    # Drop any existing row for the method we're merging in; keep the rest.
    kept_rows = [r for r in existing_rows if r.get("method") != method_name]
    merged_rows = kept_rows + new_rows

    # Backfill missing columns to keep the writer happy.
    for r in merged_rows:
        for fn in union_fields:
            r.setdefault(fn, "")

    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=union_fields)
        w.writeheader()
        w.writerows(merged_rows)

    logger.info(
        "merged %s row into %s (kept %d existing row(s), now %d total)",
        method_name, combined_csv, len(kept_rows), len(merged_rows),
    )


def run_one_dataset_seed_set_method_only(
    dataset_name: str,
    seed_set_name: str,
    seeds: list[int],
    output_dir: Path,
    method_name: str,
) -> BenchmarkResult:
    """Run the benchmark for one (dataset, seed_set) on ONLY ``method_name``.

    Mirrors :func:`run_paper_benchmark.run_one_dataset_seed_set` but writes
    the combined CSV via a *merge* rather than overwrite, so the other
    methods' rows are preserved.

    Side effects
    ------------
    - Writes ``results/<dataset>/<method>/set_<seed_set>/*`` (overwrites any
      prior single-method artifacts for that cell).
    - Merges one row into ``results/<dataset>/combined_set_<seed_set>/all_methods.csv``.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    logger.info(
        "=== %s / Set %s / method=%s (%d seeds) ===",
        cfg.display_name, seed_set_name, method_name, len(seeds),
    )

    snap: DatasetSnapshot = cfg.loader()
    X = snap.embeddings
    texts = snap.texts
    labels = snap.labels

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result, artifacts = run_benchmark_with_artifacts(
        X=X,
        dataset_name=cfg.display_name,
        seeds=seeds,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        k_range=cfg.k_range,
        methods=[method_name],
        ground_truth_labels=labels,
        texts=texts,
    )

    # Per-method artifacts (no conflict with other methods' subdirs).
    dataset_root = output_dir / dataset_name
    for m in result.methods:
        method_dir = dataset_root / m.method / f"set_{seed_set_name}"
        _persist_method_artifacts(
            m, artifacts, method_dir, seeds, dataset_name, seed_set_name,
            cfg, started_at,
        )

    # Merge into the combined CSV instead of overwriting.
    combined_csv = dataset_root / f"combined_set_{seed_set_name}" / "all_methods.csv"
    _merge_method_row_into_combined_csv(result, combined_csv, method_name)

    logger.info(
        "%s / Set %s / %s done. Results in %s/",
        cfg.display_name, seed_set_name, method_name, dataset_root,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the BERTopic baseline only; merge into existing all_methods.csv.",
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

    # Defensive: ensure the BERTopic method is actually registered.
    if ONLY_METHOD not in ALL_METHODS:
        logger.error(
            "method %r is not in ALL_METHODS=%s — abort",
            ONLY_METHOD, ALL_METHODS,
        )
        return 2

    # Load the canonical seed manifest (no synthetic fallback — this script
    # is for real datasets).
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
                "synthetic dataset is not supported here — use run_paper_benchmark.py for that",
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
                logger.info("--quick: using first %d seeds of Set %s", len(seeds), set_name)
            run_one_dataset_seed_set_method_only(
                dataset_name, set_name, seeds, output_dir, ONLY_METHOD,
            )

        if not args.skip_reliability:
            dataset_root = output_dir / dataset_name
            # Use the FULL method list so reliability + pairwise compare
            # bertopic against all 5 other methods.
            compute_between_set_reliability(dataset_root, list(ALL_METHODS), requested_sets)
            compute_pairwise_tests(
                dataset_root, list(ALL_METHODS), requested_sets, reference="reap",
            )

    logger.info("BERTopic-only sweep complete; results in %s/", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
