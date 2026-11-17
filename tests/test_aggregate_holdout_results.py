"""Tests for the fail-closed hold-out aggregator (``scripts/aggregate_holdout_results.py``).

Builds a complete synthetic results tree for every in-scope cell of the real
v1.6 run matrix (constructed metric values, no training), then checks that:
the aggregator accepts it and reproduces the registered family sizes; it stops
on a missing, failed, uncommitted-code, or extra run; an untestable member
stays in its family with p = 1; replicates are averaged per seed; a planted
real difference is detected and a planted null is not; and the accounting
table lists every registered cell, deferred ones with their reason.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_holdout_results as agg  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_holdout_cell as cell_driver  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.holdout import HOLDOUT_METRIC_IDS, HoldoutComparison, HoldoutRunMetrics  # noqa: E402
from reap.repro_bundle import (  # noqa: E402
    CodeState,
    ReproBundle,
    get_environment_snapshot,
    read_repro_bundle,
    write_repro_bundle,
)

MATRIX = json.loads(cell_driver.RUN_MATRIX_PATH.read_text())
SEEDS = list(range(1000, 1030))
ENVIRONMENT = get_environment_snapshot()
# Planted effects: REAP beats parametric UMAP on trustworthiness by 0.05 in every
# cell; REAP and the linear head are drawn from the same distribution (a null).
REAP_TRUST = 0.80
PUMAP_TRUST = 0.75


def _get_comparison(cell_id: str, replicate_shift: float, rng: np.random.Generator,
                    undefined: str | None = None) -> HoldoutComparison:
    """A synthetic comparison for one replicate with planted effects."""
    runs = []
    for method in ("reap_mlp", "linear_head", "parametric_umap"):
        for seed in SEEDS:
            base = {"reap_mlp": REAP_TRUST, "linear_head": REAP_TRUST,
                    "parametric_umap": PUMAP_TRUST}[method]
            values = {m: float(base + replicate_shift + rng.normal(scale=0.01))
                      for m in HOLDOUT_METRIC_IDS}
            reasons = {}
            if undefined is not None and method == "parametric_umap" and seed == SEEDS[0]:
                values[undefined] = None  # type: ignore[assignment]
                reasons[undefined] = "all held-out points share one nearest centroid"
            runs.append(HoldoutRunMetrics(method=method, seed=seed, values=values,  # type: ignore[arg-type]
                                          undefined_reasons=reasons))
    runs.append(HoldoutRunMetrics(method="identity", seed=None,
                                  values=dict.fromkeys(HOLDOUT_METRIC_IDS, 0.6)))
    return HoldoutComparison(run_id=cell_id, n_in=100, n_out=40, k_holdout=4, n_neighbors=15,
                             training_seeds=SEEDS,
                             methods=["reap_mlp", "linear_head", "parametric_umap", "identity"],
                             runs=runs)


def _write_run(directory: Path, comparison: HoldoutComparison, status: str = "completed",
               dirty: bool = False) -> None:
    """Write one replicate's comparison and record file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison_metrics.json").write_text(comparison.model_dump_json())
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    write_repro_bundle(
        ReproBundle(
            protocol_version="1.6", code=CodeState(commit="d" * 40, dirty=dirty),
            script="scripts/run_holdout_cell.py", run_id=comparison.run_id, dataset="d",
            dataset_version="v", dataset_sha256="e" * 64, seed_set="D", seeds=SEEDS,
            seed_manifest="m", config={}, evaluation_mode="holdout",
            metric_recipe_ids=list(HOLDOUT_METRIC_IDS), environment=ENVIRONMENT,
            run_status=status,  # type: ignore[arg-type]
            failure_reason="crash" if status == "failed" else None,
            started_at=now, finished_at=now,
        ),
        directory / "bundle.json",
    )


@pytest.fixture()
def research_root(tmp_path: Path) -> Path:
    """A complete synthetic results tree for every in-scope cell (T5 replicates differ)."""
    rng = np.random.default_rng(7)
    for cell_id, directories in agg.get_expected_dirs(MATRIX, tmp_path).items():
        for index, directory in enumerate(directories):
            _write_run(directory, _get_comparison(cell_id, 0.01 * index, rng))
    return tmp_path


def _first_dir(root: Path, cell_id: str) -> Path:
    return agg.get_expected_dirs(MATRIX, root)[cell_id][0]


class TestCompleteTree:
    def test_families_have_the_registered_sizes(self, research_root: Path) -> None:
        tests = agg.get_paired_tests(agg.load_cell_results(MATRIX, research_root), MATRIX)
        counts = {f: sum(t["family"] == f for t in tests) for f in agg.FAMILY_BY_CORPUS.values()}
        assert counts == {"ai_art_holdout": 60, "korean_forest_holdout": 72}

    def test_planted_difference_is_found_and_planted_null_is_not(self, research_root) -> None:
        tests = agg.get_paired_tests(agg.load_cell_results(MATRIX, research_root), MATRIX)
        vs_pumap = [t for t in tests if t["baseline"] == "parametric_umap"
                    and t["metric"] == "trustworthiness"]
        assert all(t["significant_holm"] for t in vs_pumap)
        assert all(float(t["mean_difference"]) > 0.04 for t in vs_pumap)
        vs_linear = [t for t in tests if t["baseline"] == "linear_head"]
        assert sum(bool(t["significant_holm"]) for t in vs_linear) <= 2

    def test_identity_is_tested_against_its_single_value(self, research_root) -> None:
        tests = agg.get_paired_tests(agg.load_cell_results(MATRIX, research_root), MATRIX)
        vs_identity = [t for t in tests if t["baseline"] == "identity"]
        assert all(t["testable"] and t["mean_baseline"] == pytest.approx(0.6)
                   for t in vs_identity)

    def test_replicates_are_averaged_per_seed(self, research_root: Path) -> None:
        results = {r.cell_id: r for r in agg.load_cell_results(MATRIX, research_root)}
        t5 = results["ai_art/T5"]
        assert len(t5.comparisons) == 3
        column = agg.get_seed_column(t5, "reap_mlp", "trustworthiness")
        by_replicate = [c.get_metric_column("reap_mlp", "trustworthiness") for c in t5.comparisons]
        assert column[0] == pytest.approx(np.mean([rep[0] for rep in by_replicate]))

    def test_main_writes_every_output(self, research_root: Path) -> None:
        assert agg.main(["--research-root", str(research_root)]) == 0
        out = research_root / "results" / "temporal_holdout" / "aggregate"
        for name in ("accounting.csv", "paired_tests.csv", "method_summaries.csv",
                     "acceptance.json", "bundle.json"):
            assert (out / name).exists(), name
        assert read_repro_bundle(out / "bundle.json").run_status == "completed"
        acceptance = json.loads((out / "acceptance.json").read_text())
        assert acceptance["ai_art"]["floor_expectation_met"] is True
        assert acceptance["korean_forest"]["n_cells"] == 6


class TestFailsClosed:
    def test_missing_replicate_stops_aggregation(self, research_root: Path) -> None:
        shutil.rmtree(agg.get_expected_dirs(MATRIX, research_root)["ai_art/T5"][1])
        with pytest.raises(agg.IncompleteResultsError, match="missing run"):
            agg.load_cell_results(MATRIX, research_root)
        assert agg.main(["--research-root", str(research_root)]) == 2

    def test_failed_run_stops_aggregation(self, research_root: Path) -> None:
        directory = _first_dir(research_root, "ai_art/T1")
        _write_run(directory, _get_comparison("ai_art/T1", 0.0, np.random.default_rng(1)),
                   status="failed")
        with pytest.raises(agg.IncompleteResultsError, match="failed run"):
            agg.load_cell_results(MATRIX, research_root)

    def test_run_from_uncommitted_code_stops_aggregation(self, research_root: Path) -> None:
        directory = _first_dir(research_root, "ai_art/T2")
        _write_run(directory, _get_comparison("ai_art/T2", 0.0, np.random.default_rng(1)),
                   dirty=True)
        with pytest.raises(agg.IncompleteResultsError, match="uncommitted"):
            agg.load_cell_results(MATRIX, research_root)

    def test_extra_run_stops_aggregation(self, research_root: Path) -> None:
        extra = research_root / "results" / "temporal_holdout" / "ai_art" / "T9"
        _write_run(extra, _get_comparison("ai_art/T9", 0.0, np.random.default_rng(1)))
        with pytest.raises(agg.IncompleteResultsError, match="extra result"):
            agg.load_cell_results(MATRIX, research_root)


def test_untestable_member_stays_in_its_family_with_p_one(research_root: Path) -> None:
    directory = _first_dir(research_root, "ai_art/T3")
    _write_run(directory, _get_comparison("ai_art/T3", 0.0, np.random.default_rng(2),
                                          undefined="silhouette_holdout"))
    tests = agg.get_paired_tests(agg.load_cell_results(MATRIX, research_root), MATRIX)
    member = next(t for t in tests if t["cell_id"] == "ai_art/T3"
                  and t["baseline"] == "parametric_umap" and t["metric"] == "silhouette_holdout")
    assert member["testable"] is False and member["p_raw"] == 1.0 and member["p_holm"] == 1.0
    assert sum(t["family"] == "ai_art_holdout" for t in tests) == 60


def test_accounting_lists_every_registered_cell(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    _write_run(_first_dir(tmp_path, "ai_art/T1"), _get_comparison("ai_art/T1", 0.0, rng))
    rows = agg.get_accounting_rows(MATRIX, tmp_path)
    assert len(rows) == 23
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    assert by_status == {"completed": 1, "pending": 10, "deferred": 12}
    assert all(row["detail"] for row in rows if row["status"] == "deferred")
    assert agg.main(["--research-root", str(tmp_path), "--status"]) == 0
    with (tmp_path / "results/temporal_holdout/aggregate/accounting.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 23
