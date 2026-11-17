"""Combine every hold-out cell's results into the paper's tables, fail-closed.

Reads the per-cell results written by ``scripts/run_holdout_cell.py`` under the
research root and the plan in ``docs/run_matrix_v1_6.json``, then applies the
aggregation rules of protocol v1.6 clause (e):

- every in-scope cell (every replicate) must have a completed record file from
  merged code (protocol ≥ 1.6, no uncommitted changes); a missing, failed, or
  extra run stops the aggregation with an error — nothing is averaged around;
- per family member (cell × baseline × test metric) the REAP column is paired
  with the baseline's column by seed index, after the per-seed mean over
  replicates; the identity baseline's single value is repeated across seeds;
- two-sided Wilcoxon signed-rank, mean paired difference, paired Cohen's d_z,
  and a 95% seed-bootstrap interval for every method's mean (10,000 resamples,
  generator seed 42);
- an untestable member stays in its family with p = 1 and its reason, so the
  family size equals the registered size; Holm within each family, with
  Benjamini–Hochberg alongside;
- the v1.6 acceptance expectations are evaluated and reported, not gated on.

``--status`` skips the statistics and writes only the accounting table: every
registered cell marked completed, failed, pending, or deferred with its reason.
That table is what the chapter uses to account for every registered analysis.

Outputs (under ``<research root>/results/temporal_holdout/aggregate/``):
``accounting.csv``, ``method_summaries.csv``, ``paired_tests.csv``,
``acceptance.json``, ``bundle.json``.

Usage::

    python scripts/aggregate_holdout_results.py --status
    python scripts/aggregate_holdout_results.py

Side effects: reads result files; writes the outputs above; logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_holdout_cell as cell_driver  # noqa: E402
from research_root import get_research_root  # noqa: E402

from reap.holdout import DETERMINISTIC_METHODS, HoldoutComparison  # noqa: E402
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    read_repro_bundle,
    validate_protocol_version,
    write_repro_bundle,
)
from reap.statistics import (  # noqa: E402
    benjamini_hochberg,
    compute_seed_bootstrap_ci,
    holm_bonferroni,
    paired_cohens_d,
    paired_wilcoxon,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_METHOD = "reap_mlp"
FAMILY_BY_CORPUS = {"ai_art": "ai_art_holdout", "korean_forest": "korean_forest_holdout"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
ACCEPTANCE_FLOOR = 0.70  # v1.5 clause (a): trustworthiness and distance correlation
K1_TRUSTWORTHINESS_FLOOR = 0.65
K1_STRATEGY = "K1"
# v1.6 clause (c): against the linear head, either metric counts.
LINEAR_HEAD_WINS = {("linear_head", "trustworthiness"), ("linear_head", "distance_correlation")}


class IncompleteResultsError(RuntimeError):
    """Raised when the results on disk do not match the plan exactly."""


class CellResult(BaseModel):
    """One cell's validated results: one comparison per replicate."""

    model_config = ConfigDict(frozen=True)

    cell_id: str = Field(..., description="Run-matrix cell id.")
    corpus: str = Field(..., description="Corpus id.")
    strategy: str = Field(..., description="Strategy id.")
    comparisons: list[HoldoutComparison] = Field(..., description="One per replicate, in order.")
    bundles: list[ReproBundle] = Field(..., description="Matching record files.")


def get_expected_dirs(matrix: dict, research_root: Path) -> dict[str, list[Path]]:
    """Map every in-scope cell id to its expected result directories (one per replicate)."""
    expected: dict[str, list[Path]] = {}
    for cell in matrix["cells"]:
        if cell["status"] != "in_scope":
            continue
        replicates = cell["replicates"]
        if replicates > 1:
            expected[cell["cell_id"]] = [
                cell_driver.get_output_dir(research_root, cell, r)
                for r in range(1, replicates + 1)
            ]
        else:
            expected[cell["cell_id"]] = [cell_driver.get_output_dir(research_root, cell, None)]
    return expected


def get_accounting_rows(matrix: dict, research_root: Path) -> list[dict[str, str]]:
    """One row per registered cell: completed, failed, pending, or deferred.

    A cell is completed only when every replicate has a completed bundle;
    failed when any replicate's bundle failed; pending otherwise. Pure apart
    from reading bundle files.
    """
    expected = get_expected_dirs(matrix, research_root)
    rows = []
    for cell in matrix["cells"]:
        if cell["status"] == "deferred":
            rows.append({"cell_id": cell["cell_id"], "status": "deferred",
                         "detail": cell["deferral_reason"]})
            continue
        statuses = []
        for directory in expected[cell["cell_id"]]:
            bundle_path = directory / "bundle.json"
            statuses.append(read_repro_bundle(bundle_path).run_status
                            if bundle_path.exists() else "pending")
        if all(s == "completed" for s in statuses):
            status, detail = "completed", f"{len(statuses)} replicate(s)"
        elif "failed" in statuses:
            status, detail = "failed", ", ".join(statuses)
        else:
            status, detail = "pending", ", ".join(statuses)
        rows.append({"cell_id": cell["cell_id"], "status": status, "detail": detail})
    return rows


def _get_found_result_dirs(research_root: Path) -> set[Path]:
    """Every directory under the hold-out results tree that holds a bundle."""
    tree = research_root / "results" / "temporal_holdout"
    return {p.parent for p in tree.rglob("bundle.json") if "aggregate" not in p.parts}


def load_cell_results(matrix: dict, research_root: Path) -> list[CellResult]:
    """Load and validate every in-scope cell's results; fail on any mismatch.

    Raises
    ------
    IncompleteResultsError : On a missing, failed, unmerged, dirty, or extra run.
    """
    expected = get_expected_dirs(matrix, research_root)
    problems: list[str] = []
    expected_dirs = {d for dirs in expected.values() for d in dirs}
    for extra in sorted(_get_found_result_dirs(research_root) - expected_dirs):
        problems.append(f"extra result not in the plan: {extra}")
    cells = {c["cell_id"]: c for c in matrix["cells"]}
    results: list[CellResult] = []
    for cell_id, directories in expected.items():
        comparisons, bundles = [], []
        for directory in directories:
            bundle_path = directory / "bundle.json"
            if not bundle_path.exists():
                problems.append(f"missing run: {directory}")
                continue
            bundle = read_repro_bundle(bundle_path)
            if bundle.run_status != "completed":
                problems.append(f"failed run: {directory}: {bundle.failure_reason}")
                continue
            if bundle.code.dirty:
                problems.append(f"run from uncommitted code: {directory}")
                continue
            try:
                validate_protocol_version(bundle.protocol_version, "1.6")
            except ValueError as error:
                problems.append(f"{directory}: {error}")
                continue
            comparisons.append(HoldoutComparison.model_validate_json(
                (directory / "comparison_metrics.json").read_text()))
            bundles.append(bundle)
        if len(comparisons) == len(directories):
            cell = cells[cell_id]
            results.append(CellResult(cell_id=cell_id, corpus=cell["corpus"],
                                      strategy=cell["strategy"], comparisons=comparisons,
                                      bundles=bundles))
    if problems:
        raise IncompleteResultsError("results do not match the plan:\n  " + "\n  ".join(problems))
    return results


def get_seed_column(result: CellResult, method: str, metric: str) -> list[float | None]:
    """Per-seed values for one method and metric, averaged over replicates.

    A seed whose value is undefined in any replicate is undefined. The identity
    baseline's single value per replicate is averaged over replicates and
    repeated across the seeds.
    """
    seeds = result.comparisons[0].training_seeds
    per_replicate = [c.get_metric_column(method, metric) for c in result.comparisons]
    if method in DETERMINISTIC_METHODS:
        values = [column[0] for column in per_replicate]
        value = None if any(v is None for v in values) else float(np.mean(values))  # type: ignore[arg-type]
        return [value] * len(seeds)
    column: list[float | None] = []
    for index in range(len(seeds)):
        values = [rep[index] for rep in per_replicate]
        column.append(None if any(v is None for v in values) else float(np.mean(values)))  # type: ignore[arg-type]
    return column


def get_paired_test_row(
    result: CellResult, baseline: str, metric: str, family: str
) -> dict[str, object]:
    """Test REAP against one baseline on one metric (one family member)."""
    reference = get_seed_column(result, REFERENCE_METHOD, metric)
    comparator = get_seed_column(result, baseline, metric)
    row: dict[str, object] = {"family": family, "cell_id": result.cell_id,
                              "baseline": baseline, "metric": metric,
                              "n_seeds": len(reference)}
    if any(v is None for v in reference) or any(v is None for v in comparator):
        return {**row, "p_raw": 1.0, "testable": False,
                "untestable_reason": "metric undefined on at least one seed"}
    x, y = np.asarray(reference, dtype=float), np.asarray(comparator, dtype=float)
    statistic, p_value = paired_wilcoxon(x, y)
    row.update(mean_reference=float(x.mean()), mean_baseline=float(y.mean()),
               mean_difference=float((x - y).mean()), cohens_dz=paired_cohens_d(x, y))
    if not np.isfinite(p_value):
        return {**row, "p_raw": 1.0, "testable": False,
                "untestable_reason": "fewer than two non-zero paired differences"}
    return {**row, "statistic": float(statistic), "p_raw": float(p_value), "testable": True,
            "untestable_reason": ""}


def get_paired_tests(results: list[CellResult], matrix: dict) -> list[dict[str, object]]:
    """Every family member's test, with Holm and BH applied within each family.

    Raises
    ------
    IncompleteResultsError : If a family's size differs from the registered size.
    """
    rows = [
        get_paired_test_row(result, baseline, metric, FAMILY_BY_CORPUS[result.corpus])
        for result in results
        for baseline in matrix["paired_baselines"]
        for metric in matrix["primary_test_metrics"]
    ]
    for family in sorted({str(r["family"]) for r in rows}):
        members = [r for r in rows if r["family"] == family]
        registered = matrix["declared_family_sizes"][family]
        if len(members) != registered:
            raise IncompleteResultsError(
                f"family {family} has {len(members)} members, registered {registered}"
            )
        p_raw = [float(r["p_raw"]) for r in members]  # type: ignore[arg-type]
        for row, p_holm, p_bh in zip(members, holm_bonferroni(p_raw), benjamini_hochberg(p_raw)):
            row["p_holm"] = float(p_holm)
            row["p_bh"] = float(p_bh)
            row["significant_holm"] = bool(row["testable"]) and float(p_holm) < ALPHA
    return rows


def get_method_summaries(results: list[CellResult], matrix: dict) -> list[dict[str, object]]:
    """Mean and 95% seed-bootstrap interval for every cell, method and metric."""
    rows = []
    for result in results:
        for method in matrix["methods"]:
            for metric in matrix["primary_test_metrics"]:
                column = get_seed_column(result, method, metric)
                row: dict[str, object] = {"cell_id": result.cell_id, "method": method,
                                          "metric": metric}
                if any(v is None for v in column):
                    rows.append({**row, "defined": False})
                    continue
                if method in DETERMINISTIC_METHODS:
                    rows.append({**row, "defined": True, "mean": column[0],
                                 "ci_low": column[0], "ci_high": column[0], "n": 1})
                    continue
                ci = compute_seed_bootstrap_ci(column, n_bootstrap=BOOTSTRAP_RESAMPLES,  # type: ignore[arg-type]
                                               random_state=BOOTSTRAP_SEED)
                rows.append({**row, "defined": True, "mean": ci.mean, "ci_low": ci.ci_lower,
                             "ci_high": ci.ci_upper, "n": ci.n_samples})
    return rows


def get_acceptance(summaries: list[dict[str, object]], tests: list[dict[str, object]],
                   results: list[CellResult]) -> dict[str, object]:
    """Evaluate the v1.6 acceptance expectations (reported, never gated on)."""
    def _mean(cell_id: str, metric: str) -> float | None:
        for row in summaries:
            key = (row["cell_id"], row["method"], row["metric"])
            if key == (cell_id, REFERENCE_METHOD, metric):
                return row.get("mean")  # type: ignore[return-value]
        return None

    report: dict[str, object] = {}
    for corpus in ("ai_art", "korean_forest"):
        cells = [r for r in results if r.corpus == corpus]
        meeting = [
            r.cell_id for r in cells
            if all((m := _mean(r.cell_id, metric)) is not None and m > ACCEPTANCE_FLOOR
                   for metric in ("trustworthiness", "distance_correlation"))
        ]
        by_cell: dict[str, set[tuple[str, str]]] = {}
        for t in tests:
            if t["family"] == FAMILY_BY_CORPUS[corpus] and t["significant_holm"]:
                if float(t["mean_difference"]) > 0:  # type: ignore[arg-type]
                    by_cell.setdefault(str(t["cell_id"]), set()).add(
                        (str(t["baseline"]), str(t["metric"])))
        significance_cells = [
            cell for cell, wins in by_cell.items()
            if ("parametric_umap", "trustworthiness") in wins
            and bool(LINEAR_HEAD_WINS & wins)
        ]
        report[corpus] = {
            "n_cells": len(cells),
            "cells_meeting_both_floors": meeting,
            "required": 3,
            "floor_expectation_met": len(meeting) >= 3,
            "cells_meeting_significance_rule": sorted(significance_cells),
            "significance_expectation_met": bool(significance_cells),
        }
    k1 = [r for r in results if r.strategy == K1_STRATEGY]
    if k1:
        value = _mean(k1[0].cell_id, "trustworthiness")
        report["korean_forest_K1_trustworthiness"] = {
            "mean": value, "floor": K1_TRUSTWORTHINESS_FLOOR,
            "met": value is not None and value > K1_TRUSTWORTHINESS_FLOOR,
        }
    return report


def _write_csv(path: Path, rows: list[dict[str, str]] | list[dict[str, object]]) -> None:
    """Write rows (union of keys, sorted columns after the first row's) as CSV, atomically."""
    columns = list(rows[0]) if rows else []
    for row in rows:
        columns.extend(k for k in row if k not in columns)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, restval="")
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()

    def _save(tmp: Path) -> None:
        tmp.write_text(text, encoding="utf-8")

    cell_driver._write_atomic(path, _save)


def _get_inputs_sha256(research_root: Path, matrix: dict) -> str:
    """SHA-256 over every input comparison file, in plan order."""
    digest = hashlib.sha256()
    for directories in get_expected_dirs(matrix, research_root).values():
        for directory in directories:
            digest.update((directory / "comparison_metrics.json").read_bytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code (2 on incomplete results).

    Side effects: see the module docstring.
    """
    parser = argparse.ArgumentParser(description="Combine hold-out results, fail-closed.")
    parser.add_argument("--research-root", default=None)
    parser.add_argument("--status", action="store_true", help="Write the accounting table only.")
    parser.add_argument("--run-matrix", default=str(cell_driver.RUN_MATRIX_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    matrix = json.loads(Path(args.run_matrix).read_text())
    validate_protocol_version(matrix["protocol_version"], "1.6")
    research_root = get_research_root(args.research_root)
    out_dir = research_root / "results" / "temporal_holdout" / "aggregate"
    accounting = get_accounting_rows(matrix, research_root)
    _write_csv(out_dir / "accounting.csv", accounting)
    for row in accounting:
        logger.info("%-60s %-9s %s", row["cell_id"], row["status"], row["detail"])
    if args.status:
        return 0

    started = datetime.now(timezone.utc)
    try:
        results = load_cell_results(matrix, research_root)
    except IncompleteResultsError as error:
        logger.error("%s", error)
        return 2
    tests = get_paired_tests(results, matrix)
    summaries = get_method_summaries(results, matrix)
    acceptance = get_acceptance(summaries, tests, results)
    _write_csv(out_dir / "paired_tests.csv", tests)
    _write_csv(out_dir / "method_summaries.csv", summaries)
    cell_driver._write_json(out_dir / "acceptance.json", acceptance)
    commits: list[JsonValue] = [*sorted({b.code.commit for r in results for b in r.bundles})]
    config: dict[str, JsonValue] = {
        "run_matrix": "docs/run_matrix_v1_6.json", "result_commits": commits,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "alpha": ALPHA, "reference_method": REFERENCE_METHOD,
    }
    bundle = ReproBundle(
        protocol_version=matrix["protocol_version"], code=get_code_state(REPO_ROOT),
        script="scripts/aggregate_holdout_results.py", run_id="temporal_holdout/aggregate",
        dataset="temporal_holdout_results", dataset_version=matrix["protocol_version"],
        dataset_sha256=_get_inputs_sha256(research_root, matrix),
        seed_set="D", seeds=results[0].comparisons[0].training_seeds,
        seed_manifest=cell_driver.SEED_MANIFEST_RELPATH,
        config=config,
        evaluation_mode="holdout", metric_recipe_ids=list(matrix["primary_test_metrics"]),
        environment=get_environment_snapshot(), run_status="completed",
        started_at=started, finished_at=datetime.now(timezone.utc),
    )
    write_repro_bundle(bundle, out_dir / "bundle.json")
    logger.info("wrote %d paired tests and %d summaries to %s", len(tests), len(summaries),
                out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
