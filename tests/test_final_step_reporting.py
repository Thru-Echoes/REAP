"""Tests for the final-step reporting scripts (scripts/experiments/).

Each script turns saved outputs into a table that feeds the protocol
amendment, so each test builds a small set of outputs in the expected layout,
runs the script's ``main``, and checks the numbers against a direct
computation and that a missing input stops the script.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

import final_step  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_protocol_k_check as pk  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_umap_settings_diagnosis as us  # noqa: E402  # pyright: ignore[reportMissingImports]
import summarize_final_step_diagnosis as summ  # noqa: E402  # pyright: ignore[reportMissingImports]

SETS = ("A", "B", "C")


def _blob_map(seed: int, n: int = 60) -> np.ndarray:
    """A small 2-d map of three blobs, shifted slightly per seed."""
    data = make_blobs(n_samples=n, centers=[[0, 0], [6, 0], [0, 6]], cluster_std=0.6,
                      random_state=0)
    rng = np.random.default_rng(seed)
    return np.asarray(data[0]) + rng.normal(scale=0.05, size=(n, 2))


def _write_between_set_csv(path: Path, methods: list[str]) -> None:
    """A benchmark-style between-set file with arbitrary but complete rows."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "set_a", "set_b", "ari", "ami", "nmi", "frobenius_relative"])
        for i, m in enumerate(methods):
            for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
                w.writerow([m, a, b, 0.5 + 0.05 * i, 0.5, 0.5, ""])


@pytest.fixture()
def benchmark_dir(tmp_path: Path) -> Path:
    """Saved benchmark outputs for every method and set, with REAP's matrix and config."""
    root = tmp_path / "bench"
    for i, m in enumerate(pk.METHODS):
        for j, s in enumerate(SETS):
            d = root / m / f"set_{s}"
            d.mkdir(parents=True)
            emb = _blob_map(10 * i + j)
            np.save(d / "consensus_embedding.npy", emb)
            if m == "reap":
                np.save(d / "consensus_labels.npy", final_step.get_kmeans_labels(emb, 3, 42))
                np.save(d / "consensus_distance_matrix.npy", squareform(pdist(emb)))
                (d / "bundle.json").write_text(json.dumps({"config": {
                    "n_components": 2, "n_neighbors": 10, "min_dist": 0.1,
                    "k_range": [2, 3, 4]}}))
    _write_between_set_csv(root / "between_set_reliability.csv", list(pk.METHODS))
    return root


def test_protocol_k_check_matches_direct_computation(benchmark_dir: Path, tmp_path: Path) -> None:
    """The frozen K is REAP's Set A K, and each method's ARI is computed at it."""
    out = tmp_path / "out.csv"
    assert pk.main([f"toy={benchmark_dir}", "--research-root", str(tmp_path),
                    "--output", str(out)]) == 0
    rows = {r["method"]: r for r in csv.DictReader(out.open())}
    assert set(rows) == set(pk.METHODS)
    assert all(int(r["protocol_k"]) == 3 for r in rows.values())
    labels = {s: final_step.get_kmeans_labels(
        np.load(benchmark_dir / "procrustes" / f"set_{s}" / "consensus_embedding.npy"), 3, 42)
        for s in SETS}
    expected = np.mean([adjusted_rand_score(labels[a], labels[b])
                        for a, b in (("A", "B"), ("A", "C"), ("B", "C"))])
    assert float(rows["procrustes"]["between_set_ari_protocol_k"]) == pytest.approx(expected)


def test_protocol_k_by_final_seed_writes_every_seed(benchmark_dir: Path, tmp_path: Path) -> None:
    """With final seeds, one row per method and seed is written next to the main table."""
    diag = tmp_path / "results" / "final_step_repeatability" / "diagnosis"
    diag.mkdir(parents=True)
    assert pk.main([f"toy={benchmark_dir}", "--research-root", str(tmp_path),
                    "--final-seeds", "42", "0"]) == 0
    rows = list(csv.DictReader((diag / "protocol_k_by_final_seed.csv").open()))
    assert len(rows) == len(pk.METHODS) * 2
    assert {int(r["final_seed"]) for r in rows} == {42, 0}


def test_protocol_k_check_stops_on_a_missing_map(benchmark_dir: Path, tmp_path: Path) -> None:
    """A missing consensus map raises instead of skipping the method."""
    (benchmark_dir / "bertopic" / "set_C" / "consensus_embedding.npy").unlink()
    with pytest.raises(FileNotFoundError):
        pk.main([f"toy={benchmark_dir}", "--research-root", str(tmp_path),
                 "--output", str(tmp_path / "x.csv")])


def _write_set_summary(path: Path, k: int) -> None:
    """A diagnosis set summary in the layout run_final_step_diagnosis writes."""
    arm = {"k_counts": {str(k): 9, str(k + 1): 1},
           "ari_chosen_k": {"mean": 0.9, "median": 0.9, "min": 0.8, "max": 1.0, "n_pairs": 45},
           "ari_reference_k": {"mean": 0.95, "median": 0.95, "min": 0.9, "max": 1.0,
                               "n_pairs": 45}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reference_k": k, "reproduces_benchmark_labels_exactly": True,
                                "arms": {a: arm for a in ("joint", "map_only",
                                                          "cluster_only")}}))


def test_summary_table_rows_and_best_baseline(tmp_path: Path) -> None:
    """Three set rows plus one between-set row; the best baseline is the highest mean."""
    diag = tmp_path / "diag"
    for s in SETS:
        _write_set_summary(diag / "toy" / f"set_{s}" / "set_summary.json", 5)
    pair = {"benchmark_seed42": 0.8, "matched_final_seeds": {"mean": 0.85},
            "unmatched_final_seeds": {"mean": 0.84}}
    (diag / "toy" / "cross_set_summary.json").write_text(json.dumps(
        {p: pair for p in ("A-B", "A-C", "B-C")}))
    bench = tmp_path / "bench"
    bench.mkdir()
    _write_between_set_csv(bench / "between_set_reliability.csv",
                           ["reap", "best_of_n", "procrustes"])
    assert summ.main([f"toy={bench}", "--research-root", str(tmp_path),
                      "--diagnosis-dir", str(diag)]) == 0
    rows = list(csv.DictReader((diag / "diagnosis_summary.csv").open()))
    assert len(rows) == 4
    between = rows[-1]
    assert between["best_baseline"] == "procrustes"
    assert float(between["best_baseline_mean"]) == pytest.approx(0.6)
    assert "| toy |" in (diag / "diagnosis_summary.md").read_text()


def test_summary_table_stops_on_a_missing_set(tmp_path: Path) -> None:
    """A dataset with a missing set raises instead of summarising the rest."""
    diag = tmp_path / "diag"
    _write_set_summary(diag / "toy" / "set_A" / "set_summary.json", 5)
    with pytest.raises(FileNotFoundError):
        summ.get_dataset_rows("toy", diag / "toy", tmp_path)


def test_settings_combine_covers_the_whole_grid(tmp_path: Path) -> None:
    """Every grid cell and method gets a row; a missing cell raises."""
    out = tmp_path / "settings"
    for nn in us.N_NEIGHBORS_GRID:
        for md in us.MIN_DIST_GRID:
            for j, s in enumerate(SETS):
                d = out / us.get_cell_name(nn, md) / f"set_{s}"
                d.mkdir(parents=True)
                methods = {}
                for i, m in enumerate(us.METHODS):
                    emb = _blob_map(100 * i + j)
                    np.save(d / f"embedding_{m}.npy", emb)
                    np.save(d / f"labels_{m}.npy", final_step.get_kmeans_labels(emb, 3, 42))
                    methods[m] = {"k": 3, "silhouette": 0.5, "trustworthiness": 0.9,
                                  "ari_expert": None}
                (d / "record.json").write_text(json.dumps({
                    "methods": methods,
                    "reap_within_matrix": {"ari": {"mean": 0.9}}}))
    rows = us.run_combine(out)
    assert len(rows) == len(us.N_NEIGHBORS_GRID) * len(us.MIN_DIST_GRID) * len(us.METHODS)
    assert all(r["protocol_k"] == 3 for r in rows)
    fixed = list(csv.DictReader((out / "fixed_k.csv").open()))
    assert len(fixed) == len(rows) * len(us.FIXED_KS)
    (out / us.get_cell_name(50, 0.0) / "set_B" / "record.json").unlink()
    with pytest.raises(FileNotFoundError):
        us.run_combine(out)
