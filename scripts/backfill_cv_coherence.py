"""Backfill ``topic_coherence_cv`` for the 4 non-single_seed consensus methods.

Why this script exists
----------------------
The recent BERTopic sweep populated ``topic_coherence_cv`` for ``bertopic``
on every (dataset, set) cell. The four other consensus methods
(``best_of_n``, ``naive_average``, ``procrustes``, ``reap``) still have
empty c_v values in their ``topic_metrics.json`` files and in
``combined_set_<X>/all_methods.csv`` — their benchmark runs predate the
c_v addition to the harness.

We do *not* want a full re-run (it would risk floating-point drift on the
UMass + NPMI values that ``manuscript/sections/experiments.md`` already
quotes). Instead, this script re-computes ONLY ``topic_coherence_cv`` for
the 4 methods × 2 datasets × 3 sets = 24 cells, using the EXACT same code
path that ``reap.benchmarks._run_bertopic`` used when it computed
BERTopic's c_v.

Code path reused: :func:`reap.benchmarks._compute_topic_fields` with
``top_n=10``. That function:

  1. calls :func:`reap.labeling.label_clusters_ctfidf` to extract the
     top-N words per cluster (no LLM calls);
  2. calls ``reap.evaluation.compute_topic_coherence(topics, texts,
     measure='c_v', top_n=10)`` (gensim ``CoherenceModel`` under the hood)
     to get per-cluster c_v scores;
  3. aggregates via ``float(np.mean(cv_scores))``.

What this script DOES NOT change
--------------------------------
- ``topic_coherence_umass``, ``topic_coherence_npmi``, ``topic_diversity``,
  ``topic_exclusivity`` are preserved byte-identical.
- ``single_seed`` is excluded (no consensus_labels by design).
- ``bertopic`` is excluded (already populated; preserving its value
  guarantees the cross-method c_v comparison is internally consistent).
- All other CSV columns, JSON fields, and ``.npy`` files are untouched.

Usage
-----
Dry-run (compute + print, no writes; ~15-20 min)::

    python scripts/backfill_cv_coherence.py --dataset korean_forest --dataset ai_art --dry-run

Full backfill (writes; ~15-20 min)::

    python scripts/backfill_cv_coherence.py --dataset korean_forest --dataset ai_art

Side effects (without ``--dry-run``)
- Updates ``results/<dataset>/<method>/set_<X>/topic_metrics.json``
  in-place (only the ``topic_coherence_cv`` field).
- Updates ``results/<dataset>/combined_set_<X>/all_methods.csv`` in-place
  (only the ``topic_coherence_cv`` cell for the affected method row).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add src/ for the package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reap.benchmarks import _compute_topic_fields  # noqa: E402  (private but stable)
from reap.datasets import DatasetSnapshot, load_ai_art, load_korean_forest  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_LOADERS: dict[str, Any] = {
    "korean_forest": load_korean_forest,
    "ai_art": load_ai_art,
}

# Methods to backfill. Deliberately excludes:
#   - single_seed (has no consensus_labels by design)
#   - bertopic (already populated in the recent overnight sweep)
METHODS_TO_BACKFILL: list[str] = [
    "best_of_n",
    "naive_average",
    "procrustes",
    "reap",
]

# Matches the harness call site in reap.benchmarks._run_bertopic.
TOP_N: int = 10


def backfill_cell(
    dataset_root: Path,
    method: str,
    set_name: str,
    texts: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill ``topic_coherence_cv`` for one (dataset, method, set) cell.

    Pure-ish: reads ``consensus_labels.npy`` + existing ``topic_metrics.json``
    + ``combined_set_<X>/all_methods.csv``; in non-dry-run mode writes the
    JSON and CSV in-place with only the ``topic_coherence_cv`` field
    updated. All other fields, rows, and columns remain byte-identical.

    Parameters
    ----------
    dataset_root : ``results/<dataset>/`` path.
    method : Method name (one of :data:`METHODS_TO_BACKFILL`).
    set_name : Seed set name (``"A"``, ``"B"``, or ``"C"``).
    texts : Document texts, length ``n_samples``. Must align with the
        ``consensus_labels.npy`` array shape.
    dry_run : If True, compute the new c_v value but do not write files.

    Returns
    -------
    dict with keys:
      - ``status``: ``"written"`` | ``"dry_run"`` | ``"skipped"``
      - ``old_cv``: prior value in ``topic_metrics.json`` (``float`` or ``None``)
      - ``new_cv``: freshly-computed value (``float`` or ``None``)
      - ``reason``: present only when ``status == "skipped"``
    """
    cell_dir = dataset_root / method / f"set_{set_name}"
    if not cell_dir.exists():
        return {"status": "skipped", "reason": f"missing dir {cell_dir}"}

    labels_path = cell_dir / "consensus_labels.npy"
    if not labels_path.exists():
        return {"status": "skipped", "reason": f"missing {labels_path}"}

    labels = np.load(labels_path)
    if labels.shape[0] != len(texts):
        return {
            "status": "skipped",
            "reason": (
                f"labels.shape[0]={labels.shape[0]} != len(texts)={len(texts)} "
                f"for {cell_dir}"
            ),
        }

    # The actual computation — uses the same code path the harness used for
    # BERTopic, so c_v values are bit-identical methodology.
    topic_fields = _compute_topic_fields(texts, labels, top_n=TOP_N)
    new_cv = topic_fields["topic_coherence_cv"]
    if not isinstance(new_cv, (int, float, type(None))):
        raise RuntimeError(
            f"_compute_topic_fields returned non-numeric c_v for {cell_dir}: {new_cv!r}"
        )

    # Read the existing topic_metrics.json so we can preserve every other field.
    topic_metrics_path = cell_dir / "topic_metrics.json"
    if topic_metrics_path.exists():
        existing = json.loads(topic_metrics_path.read_text(encoding="utf-8"))
        old_cv = existing.get("topic_coherence_cv")
    else:
        existing = {}
        old_cv = None

    if dry_run:
        return {
            "status": "dry_run",
            "old_cv": old_cv,
            "new_cv": new_cv,
        }

    # Write topic_metrics.json: only c_v changes; every other field is preserved.
    existing["topic_coherence_cv"] = new_cv
    topic_metrics_path.write_text(
        json.dumps(existing, indent=2) + "\n",
        encoding="utf-8",
    )

    # Update combined_set_<X>/all_methods.csv: only the c_v cell of this method's row.
    combined_csv = dataset_root / f"combined_set_{set_name}" / "all_methods.csv"
    if combined_csv.exists():
        _update_csv_cv_cell(combined_csv, method, new_cv)
    else:
        logger.warning(
            "combined CSV missing for %s set %s: %s (topic_metrics.json updated only)",
            method, set_name, combined_csv,
        )

    return {"status": "written", "old_cv": old_cv, "new_cv": new_cv}


def _update_csv_cv_cell(csv_path: Path, method: str, new_cv: float | None) -> None:
    """Update ``topic_coherence_cv`` for one method's row in an all_methods.csv.

    Reads the CSV, finds exactly one row whose ``method`` column equals
    ``method``, overwrites its ``topic_coherence_cv`` cell, and writes the
    CSV back. Every other row and every other column is preserved as the
    string value that ``csv.DictReader`` returned (no re-formatting).
    """
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise RuntimeError(f"{csv_path} has no header — refusing to update.")
    if "topic_coherence_cv" not in fieldnames:
        raise RuntimeError(
            f"{csv_path} has no 'topic_coherence_cv' column — refusing to update."
        )

    updated_indices: list[int] = []
    for idx, row in enumerate(rows):
        if row.get("method") == method:
            updated_indices.append(idx)

    if len(updated_indices) == 0:
        raise RuntimeError(f"Method {method!r} not found in {csv_path}.")
    if len(updated_indices) > 1:
        raise RuntimeError(
            f"Method {method!r} appears {len(updated_indices)} times in {csv_path}."
        )

    rows[updated_indices[0]]["topic_coherence_cv"] = (
        f"{float(new_cv):.6f}" if new_cv is not None else ""
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill topic_coherence_cv post-hoc for the 4 non-single_seed methods.",
    )
    parser.add_argument(
        "--dataset", action="append", choices=sorted(DATASET_LOADERS), required=True,
        help="Dataset(s) to backfill. Pass multiple times for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Result root (default 'results/').",
    )
    parser.add_argument(
        "--seed-sets", default="A,B,C",
        help="Comma-separated seed-set names (default 'A,B,C').",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute the new c_v values and log them, but do NOT write any file.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    requested_sets = [s.strip() for s in args.seed_sets.split(",") if s.strip()]
    output_dir = Path(args.output_dir)

    n_total = 0
    n_written = 0
    n_skipped = 0
    for dataset_name in args.dataset:
        logger.info("Loading texts for %s...", dataset_name)
        snap: DatasetSnapshot = DATASET_LOADERS[dataset_name]()
        if snap.texts is None:
            logger.error("Dataset %s has no texts — cannot compute c_v.", dataset_name)
            return 2
        texts = snap.texts
        dataset_root = output_dir / dataset_name
        for set_name in requested_sets:
            for method in METHODS_TO_BACKFILL:
                n_total += 1
                result = backfill_cell(dataset_root, method, set_name, texts, args.dry_run)
                status = result["status"]
                old_cv = result.get("old_cv")
                new_cv = result.get("new_cv")
                logger.info(
                    "%s / %s / set_%s : status=%s  old_cv=%s  new_cv=%s",
                    dataset_name, method, set_name,
                    status,
                    f"{old_cv:.4f}" if isinstance(old_cv, (int, float)) else "None",
                    f"{new_cv:.4f}" if isinstance(new_cv, (int, float)) else "None",
                )
                if status == "skipped":
                    n_skipped += 1
                    logger.warning("  reason: %s", result.get("reason"))
                elif status == "written":
                    n_written += 1

    suffix = "computed (dry-run)" if args.dry_run else "written"
    logger.info(
        "Backfill complete: %d/%d cells %s; %d skipped",
        n_written if not args.dry_run else (n_total - n_skipped),
        n_total, suffix, n_skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
