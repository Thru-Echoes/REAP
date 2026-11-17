"""Tests for the hold-out cell driver (``scripts/run_holdout_cell.py``).

The driver is how a hold-out number reaches the paper, so it is tested like
package code: the scope and protocol gates, every split rule (disjoint,
complete, group-aware, deterministic), the leakage audit on a known answer,
the no-overwrite rule, and one full replicate end to end on tiny synthetic data
— checking that every artifact and a valid record file are written, and that a
crash leaves a failed record. Real-data split checks run when the source
projects and caches are present and skip visibly otherwise.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_holdout_cell as driver  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.holdout import HeadTrainingSettings, HoldoutComparison  # noqa: E402
from reap.repro_bundle import (  # noqa: E402
    CodeState,
    ReproBundle,
    get_environment_snapshot,
    read_repro_bundle,
    write_repro_bundle,
)

pytest.importorskip("torch")

MATRIX = json.loads(driver.RUN_MATRIX_PATH.read_text())
CELLS = {cell["cell_id"]: cell for cell in MATRIX["cells"]}
FAST_HEADS = HeadTrainingSettings(hidden_layers=[16], n_folds=2, max_epochs=40, patience=8)


def _get_variables(n_documents: int = 60, chunks_per_document: int = 4) -> driver.SplitVariables:
    """Synthetic corpus: documents with several chunks each, years 2013-2025."""
    years = np.repeat(2013 + np.arange(n_documents) % 13, chunks_per_document)
    groups = np.repeat(np.arange(n_documents), chunks_per_document)
    return driver.SplitVariables(strata=years, groups=groups)


class TestGates:
    def test_in_scope_cell_is_returned(self) -> None:
        assert driver.get_cell(MATRIX, "ai_art/T1")["strategy"] == "T1"

    def test_deferred_cell_is_refused(self) -> None:
        with pytest.raises(ValueError, match="deferred"):
            driver.get_cell(MATRIX, "korean_forest/K1/intfloat/multilingual-e5-large")

    def test_unknown_cell_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown cell"):
            driver.get_cell(MATRIX, "ai_art/T9")

    def test_old_protocol_is_refused(self) -> None:
        with pytest.raises(ValueError, match="predates"):
            driver.get_cell({**MATRIX, "protocol_version": "1.5"}, "ai_art/T1")

    def test_set_d_has_thirty_distinct_seeds(self) -> None:
        seeds = driver.get_set_d_seeds(REPO_ROOT / driver.SEED_MANIFEST_RELPATH)
        assert len(seeds) == len(set(seeds)) == 30


class TestSplits:
    @pytest.mark.parametrize("strategy", ["T1", "T2", "T3", "T4"])
    def test_year_splits_hold_out_exactly_the_registered_years(self, strategy) -> None:
        variables = _get_variables()
        (split,) = driver.get_splits({"strategy": strategy, "master_seeds": []}, variables)
        held_years = set(variables.strata[split.out_idx].tolist())
        assert held_years == driver.AI_ART_YEAR_STRATEGIES[strategy] & set(
            variables.strata.tolist()
        )
        assert not set(variables.strata[split.in_idx].tolist()) & held_years

    @pytest.mark.parametrize(
        "strategy,master_seeds",
        [("T5", [20260521, 20260522, 20260523]), ("K2-T4", [20260524]), ("K2-T5", [20260525])],
    )
    def test_random_splits_are_disjoint_complete_and_group_aware(
        self, strategy, master_seeds
    ) -> None:
        variables = _get_variables()
        splits = driver.get_splits({"strategy": strategy, "master_seeds": master_seeds}, variables)
        assert len(splits) == len(master_seeds)
        n = variables.strata.size
        for split in splits:
            assert not set(split.in_idx) & set(split.out_idx)
            assert sorted(np.concatenate([split.in_idx, split.out_idx]).tolist()) == list(range(n))
            assert not set(variables.groups[split.in_idx]) & set(variables.groups[split.out_idx])
            assert 0.1 < split.out_idx.size / n < 0.3
        if len(splits) > 1:
            assert splits[0].replicate == 1 and splits[-1].replicate == len(splits)
            assert not np.array_equal(splits[0].out_idx, splits[1].out_idx)

    def test_random_splits_are_deterministic(self) -> None:
        cell = {"strategy": "T5", "master_seeds": [20260521]}
        a = driver.get_splits(cell, _get_variables())
        b = driver.get_splits(cell, _get_variables())
        assert np.array_equal(a[0].out_idx, b[0].out_idx)

    def test_president_split(self) -> None:
        presidents = np.asarray(["Moon", "Park", "Lee"] * 20)
        variables = driver.SplitVariables(strata=presidents, groups=np.arange(60))
        (split,) = driver.get_splits({"strategy": "K2-T2", "master_seeds": []}, variables)
        assert set(presidents[split.out_idx]) == {"Lee"}
        assert "Lee" not in set(presidents[split.in_idx])

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="no within-corpus split rule"):
            driver.get_splits({"strategy": "K1", "master_seeds": []}, _get_variables())


def test_leakage_audit_flags_a_duplicated_document() -> None:
    rng = np.random.default_rng(0)
    X_in = rng.normal(size=(40, 16))
    X_out = np.vstack([X_in[3], rng.normal(size=(9, 16))])
    audit = driver.get_leakage_audit(X_in, X_out)
    assert audit["n_held_out"] == 10
    assert audit["n_above_threshold"] == 1
    assert audit["q100"] == pytest.approx(1.0)


def _write_bundle(path: Path, status: str) -> None:
    """Write a minimal valid bundle with the given status."""
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    write_repro_bundle(
        ReproBundle(
            protocol_version="1.6", code=CodeState(commit="a" * 40, dirty=False), script="s",
            run_id="r", dataset="d", dataset_version="v", dataset_sha256="b" * 64,
            seed_set="D", seeds=[1], seed_manifest="m", config={}, evaluation_mode="holdout",
            metric_recipe_ids=["trustworthiness"], environment=get_environment_snapshot(),
            run_status=status,  # type: ignore[arg-type]
            failure_reason="crash" if status == "failed" else None,
            started_at=now, finished_at=now,
        ),
        path / "bundle.json",
    )


class TestNoOverwrite:
    def test_completed_result_is_protected(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "completed")
        with pytest.raises(FileExistsError, match="already holds a completed result"):
            driver.validate_output_dir_is_free(tmp_path)

    def test_failed_or_empty_directory_may_be_rerun(self, tmp_path: Path) -> None:
        driver.validate_output_dir_is_free(tmp_path / "empty")
        _write_bundle(tmp_path, "failed")
        driver.validate_output_dir_is_free(tmp_path)


def test_output_dir_layout() -> None:
    root = Path("/research")
    kf = CELLS["korean_forest/K2-T4/paraphrase-multilingual-MiniLM-L12-v2"]
    assert driver.get_output_dir(root, kf, None) == root / (
        "results/temporal_holdout/korean_forest/K2-T4/minilm"
    )
    assert driver.get_output_dir(root, CELLS["ai_art/T5"], 2) == root / (
        "results/temporal_holdout/ai_art/T5/replicate_2"
    )


@pytest.fixture(scope="module")
def tiny_cell_data() -> dict:
    """120 clustered 32-d documents split 90 / 30, as one hold-out replicate."""
    from sklearn.datasets import make_blobs

    blobs = make_blobs(n_samples=120, n_features=32, centers=3, cluster_std=0.6, random_state=5)
    X = np.asarray(blobs[0])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    split = driver.Split(
        replicate=None, master_seed=20260524, in_idx=np.arange(90), out_idx=np.arange(90, 120)
    )
    return {"X": X, "split": split}


def _run_tiny(tmp_path: Path, data: dict, X_out: np.ndarray | None = None) -> ReproBundle:
    """Run one replicate on the tiny data with two training seeds and no parametric UMAP."""
    original = driver.UMAP_SETTINGS["ai_art"]
    driver.UMAP_SETTINGS["ai_art"] = {"n_components": 2, "n_neighbors": 8, "min_dist": 0.1}
    original_k = driver.K_RANGE
    driver.K_RANGE = range(2, 6)
    try:
        X = data["X"]
        return driver.run_replicate(
            cell={"cell_id": "ai_art/T5", "corpus": "ai_art", "strategy": "T5",
                  "encoder": "intfloat/e5-large-v2"},
            split=data["split"], X_in=X[:90], X_out=X[90:] if X_out is None else X_out,
            set_d_seeds=[101, 202], output_dir=tmp_path,
            dataset_meta={"version": "synthetic", "sha256": "c" * 64},
            n_resamples=5, groups_out=None, head_settings=FAST_HEADS, umap_epochs=5,
            code_root=REPO_ROOT, methods=("reap_mlp", "linear_head", "identity"),
        )
    finally:
        driver.UMAP_SETTINGS["ai_art"] = original
        driver.K_RANGE = original_k


class TestEndToEnd:
    def test_one_replicate_writes_every_artifact_and_a_valid_record(
        self, tmp_path: Path, tiny_cell_data: dict
    ) -> None:
        bundle = _run_tiny(tmp_path, tiny_cell_data)
        for name in (
            "retained_indices.npy", "held_out_indices.npy", "consensus_embedding.npy",
            "consensus_labels.npy", "reference_held_out_labels.npy", "comparison_metrics.json",
            "subsample_intervals.json", "leakage_audit.json", "bundle.json",
            "run1_reap_mlp_held_out_projection.npy",
        ):
            assert (tmp_path / name).exists(), name
        assert read_repro_bundle(tmp_path / "bundle.json") == bundle
        assert bundle.run_status == "completed"
        assert bundle.evaluation_mode == "holdout"
        assert bundle.seeds == [101, 202]
        assert bundle.config["k_holdout"] in range(2, 6)
        comparison = HoldoutComparison.model_validate_json(
            (tmp_path / "comparison_metrics.json").read_text()
        )
        assert len(comparison.get_metric_column("reap_mlp", "trustworthiness")) == 2
        assert np.array_equal(np.load(tmp_path / "held_out_indices.npy"), np.arange(90, 120))

    def test_a_crash_leaves_a_failed_record(self, tmp_path: Path, tiny_cell_data: dict) -> None:
        bad_out = tiny_cell_data["X"][90:, :16]  # wrong feature count
        with pytest.raises(ValueError):
            _run_tiny(tmp_path, tiny_cell_data, X_out=bad_out)
        bundle = read_repro_bundle(tmp_path / "bundle.json")
        assert bundle.run_status == "failed"
        assert bundle.failure_reason


def test_real_splits_match_the_preflight_counts() -> None:
    """On the real corpora, the registered splits have the expected sizes and no shared groups."""
    try:
        ai_art = driver._load_cell_data(CELLS["ai_art/T5"])
        kf_cell = CELLS["korean_forest/K2-T4/paraphrase-multilingual-MiniLM-L12-v2"]
        kf = driver._load_cell_data(kf_cell)
    except FileNotFoundError as missing:
        pytest.skip(f"real datasets or source projects not on this machine: {missing}")
    (t1,) = driver.get_splits(CELLS["ai_art/T1"], ai_art["variables"])
    assert t1.out_idx.size == 960  # protocol §18.b
    for split in driver.get_splits(CELLS["ai_art/T5"], ai_art["variables"]):
        groups = ai_art["variables"].groups
        assert not set(groups[split.in_idx]) & set(groups[split.out_idx])
    (moon,) = driver.get_splits(
        CELLS["korean_forest/K2-T1/paraphrase-multilingual-MiniLM-L12-v2"], kf["variables"]
    )
    assert moon.out_idx.size == 390
