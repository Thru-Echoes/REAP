"""End-to-end test of the reference-plus-probes driver (scripts/run_reference_probes.py).

A small synthetic reference (filtered by a table column) and one probe set
run through the whole driver: REAP map, K, final-step repeats, projection
head, probe projection and assignment, and the long-format tables. A second
seed set reuses the first run's K. Probes must never change the reference.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_reference_probes as driver  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.repro_bundle import read_repro_bundle  # noqa: E402

N_REF, N_PROBE, DIM = 400, 40, 32


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Reference .npz with a two-variant table, and a probe folder (emb.npy + ids.txt)."""
    root = tmp_path_factory.mktemp("inputs")
    data = make_blobs(n_samples=N_REF + N_PROBE, centers=4, n_features=DIM, cluster_std=1.0,
                      random_state=7)
    X = np.asarray(data[0], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ref_ids = np.array([f"r{i}" for i in range(N_REF)])
    np.savez(root / "ref.npz", ids=ref_ids, vectors=X[:N_REF])
    with (root / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "corner"])
        for i, rid in enumerate(ref_ids):
            w.writerow([rid, "PS" if i % 4 else "PW"])
    probe = root / "probe"
    probe.mkdir()
    np.save(probe / "emb.npy", X[N_REF:])
    (probe / "ids.txt").write_text("\n".join(f"p{i}" for i in range(N_PROBE)) + "\n")
    return {"root": root, "ref": root / "ref.npz", "table": root / "table.csv", "probe": probe}


def _args(inputs: dict[str, Path], out: Path, seed_set: str, extra: list[str]) -> list[str]:
    """Command line for a small run."""
    return ["--label", "toy_ps", "--reference", str(inputs["ref"]),
            "--reference-table", str(inputs["table"]), "--where", "corner=PS",
            "--probe", f"calls={inputs['probe']}", "--seed-set", seed_set, "--n-seeds", "3",
            "--k-min", "2", "--k-max", "6", "--final-seeds", "42", "0",
            "--output-dir", str(out), *extra]


def test_driver_end_to_end(inputs: dict[str, Path], tmp_path: Path) -> None:
    """Every output exists with the promised shape, and set B reuses set A's K."""
    out_a = tmp_path / "A"
    assert driver.main(_args(inputs, out_a, "A", [])) == 0
    n_ps = sum(1 for i in range(N_REF) if i % 4)
    ref = np.load(out_a / "reference.npz")
    assert ref["consensus"].shape == (n_ps, 5)
    assert ref["repeat_labels"].shape == (2, n_ps)
    assert np.array_equal(ref["repeat_labels"][0], ref["labels"])
    record = json.loads((out_a / "run_record.json").read_text())
    k = record["k"]
    assert 2 <= k <= 6 and len(np.unique(ref["labels"])) == k
    probe = np.load(out_a / "probe_calls.npz")
    assert probe["consensus"].shape == (N_PROBE, 5)
    assert probe["assign_knn_vote"].max() < k
    rows = list(csv.DictReader((out_a / "probe_assignments.csv").open()))
    assert len(rows) == 2 * 2 * 2 * N_PROBE
    assert {r["rule"] for r in rows} == {"nearest_centroid", "knn_vote"}
    assert len(list(csv.DictReader((out_a / "reference_labels.csv").open()))) == 2 * n_ps
    assert len(list(csv.DictReader((out_a / "layout_2d.csv").open()))) == n_ps + N_PROBE
    assert "cv_metrics" in json.loads((out_a / "head_cv.json").read_text())
    placement = list(csv.DictReader((out_a / "probe_placement.csv").open()))
    assert len(placement) == N_PROBE
    assert all(float(r["nearest_reference_distance"]) >= 0 for r in placement)
    assert {r["accepted_strict"] for r in placement} <= {"True", "False"}
    assert (out_a / "head.pt").exists()
    assert record["probes"]["calls"]["accepted_strict_share"] <= 1.0
    assert read_repro_bundle(out_a / "bundle.json").run_status == "completed"

    out_b = tmp_path / "B"
    assert driver.main(_args(inputs, out_b, "B", ["--k-from",
                                                  str(out_a / "run_record.json")])) == 0
    record_b = json.loads((out_b / "run_record.json").read_text())
    assert record_b["k"] == k
    assert record_b["k_rule"].startswith("frozen from")


def test_probes_do_not_change_the_reference(inputs: dict[str, Path], tmp_path: Path) -> None:
    """The reference map and labels are identical with and without probes."""
    with_probe, without = tmp_path / "with", tmp_path / "without"
    driver.main(_args(inputs, with_probe, "A", []))
    args = _args(inputs, without, "A", [])
    i = args.index("--probe")
    del args[i:i + 2]
    driver.main(args)
    a, b = np.load(with_probe / "reference.npz"), np.load(without / "reference.npz")
    assert np.array_equal(a["consensus"], b["consensus"])
    assert np.array_equal(a["labels"], b["labels"])


def test_missing_reference_ids_raise(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A table id with no vector stops the run."""
    table = tmp_path / "bad.csv"
    table.write_text("id,corner\nr1,PS\nmissing,PS\n")
    args = _args(inputs, tmp_path / "out", "A", [])
    args[args.index(str(inputs["table"]))] = str(table)
    with pytest.raises(ValueError, match="no vector"):
        driver.main(args)
