"""End-to-end tests of the reference-plus-probes driver (scripts/run_reference_probes.py).

A small synthetic reference (filtered by a table column) and one probe set
run through the whole driver with settings read from a parameter-search
record made on exactly those rows: REAP map, K, final-step repeats,
projection head, probe projection and assignment, placement, and the
long-format tables. A second seed set reuses the first run's K. The driver
must refuse to run without a matching search record unless it is a
labelled code check, and probes must never change the reference.
"""

from __future__ import annotations

import csv
import datetime
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
import search_record as sr  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.repro_bundle import read_repro_bundle  # noqa: E402

N_REF, N_PROBE, DIM = 400, 40, 32
SETTINGS = {"n_components": 5, "n_neighbors": 15, "min_dist": 0.1}
K_RANGE = [2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Reference .npz with a two-variant table, a probe folder, and a search record."""
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
    _, X_ps = driver.get_reference(root / "ref.npz", root / "table.csv", ["corner=PS"])
    _write_record(root / "search_ps.json", X_ps)
    _, X_pw = driver.get_reference(root / "ref.npz", root / "table.csv", ["corner=PW"])
    _write_record(root / "search_pw.json", X_pw)
    return {"root": root, "ref": root / "ref.npz", "table": root / "table.csv", "probe": probe,
            "record": root / "search_ps.json", "other_record": root / "search_pw.json"}


def _write_record(path: Path, X: np.ndarray) -> None:
    """A search record for exactly the rows X (as a parameter search would write it)."""
    record = sr.SearchRecord(
        dataset="toy", variant=path.stem, encoder="synthetic", method="reap",
        input_sha256=sr.get_input_sha256(X), n_rows=len(X), **SETTINGS,
        k_rule="silhouette over k_range on the Set A seed-42 map", k_range=K_RANGE,
        search=sr.SearchProvenance(
            script="test", commit="0" * 40, run_id=f"search/{path.stem}",
            grid={"n_neighbors": [15]}, selection_rule="the only grid point", seeds=[1, 2, 3],
            finished_at=datetime.datetime.now(datetime.timezone.utc)))
    path.write_text(record.model_dump_json(indent=2))


def _args(inputs: dict[str, Path], out: Path, seed_set: str, extra: list[str]) -> list[str]:
    """Command line for a small run (no settings flags; add a record or a code check)."""
    return ["--label", "toy_ps", "--reference", str(inputs["ref"]),
            "--reference-table", str(inputs["table"]), "--where", "corner=PS",
            "--probe", f"calls={inputs['probe']}", "--seed-set", seed_set, "--n-seeds", "3",
            "--final-seeds", "42", "0", "--output-dir", str(out), *extra]


def test_driver_end_to_end_from_a_search_record(inputs: dict[str, Path], tmp_path: Path) -> None:
    """Every output exists, settings come from the record, and set B reuses set A's K."""
    out_a = tmp_path / "A"
    assert driver.main(_args(inputs, out_a, "A", ["--search-record", str(inputs["record"])])) == 0
    n_ps = sum(1 for i in range(N_REF) if i % 4)
    ref = np.load(out_a / "reference.npz")
    assert ref["consensus"].shape == (n_ps, SETTINGS["n_components"])
    assert ref["repeat_labels"].shape == (2, n_ps)
    assert np.array_equal(ref["repeat_labels"][0], ref["labels"])
    record = json.loads((out_a / "run_record.json").read_text())
    assert record["code_check"] is False
    assert record["settings"] == SETTINGS
    assert record["parameter_source"]["search_run_id"] == "search/search_ps"
    k = record["k"]
    assert k in K_RANGE and len(np.unique(ref["labels"])) == k
    assert not (out_a / "CODE_CHECK.txt").exists()
    probe = np.load(out_a / "probe_calls.npz")
    assert probe["consensus"].shape == (N_PROBE, SETTINGS["n_components"])
    rows = list(csv.DictReader((out_a / "probe_assignments.csv").open()))
    assert len(rows) == 2 * 2 * 2 * N_PROBE
    assert len(list(csv.DictReader((out_a / "reference_labels.csv").open()))) == 2 * n_ps
    assert len(list(csv.DictReader((out_a / "layout_2d.csv").open()))) == n_ps + N_PROBE
    placement = list(csv.DictReader((out_a / "probe_placement.csv").open()))
    assert len(placement) == N_PROBE
    assert (out_a / "head.pt").exists()
    assert "cv_metrics" in json.loads((out_a / "head_cv.json").read_text())
    bundle = read_repro_bundle(out_a / "bundle.json")
    assert bundle.config["code_check"] is False

    out_b = tmp_path / "B"
    assert driver.main(_args(inputs, out_b, "B", [
        "--search-record", str(inputs["record"]),
        "--k-from", str(out_a / "run_record.json")])) == 0
    record_b = json.loads((out_b / "run_record.json").read_text())
    assert record_b["k"] == k
    assert record_b["k_rule"].startswith("frozen from")

    probe_free = tmp_path / "no_probe"
    args = _args(inputs, probe_free, "A", ["--search-record", str(inputs["record"])])
    i = args.index("--probe")
    del args[i:i + 2]
    driver.main(args)
    again = np.load(probe_free / "reference.npz")
    assert np.array_equal(again["consensus"], ref["consensus"])
    assert np.array_equal(again["labels"], ref["labels"])


def test_refuses_to_run_without_a_search_record(inputs: dict[str, Path], tmp_path: Path) -> None:
    """No record and no code-check flag is an error: REAP has no defaults."""
    with pytest.raises(ValueError, match="no default parameters"):
        driver.main(_args(inputs, tmp_path / "out", "A", []))


def test_refuses_a_record_made_on_other_rows(inputs: dict[str, Path], tmp_path: Path) -> None:
    """Settings searched on the written rows do not carry over to the spoken rows."""
    with pytest.raises(ValueError, match="different input"):
        driver.main(_args(inputs, tmp_path / "out", "A",
                          ["--search-record", str(inputs["other_record"])]))


def test_refuses_settings_flags_next_to_a_record(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A record and command-line settings together is an error."""
    with pytest.raises(ValueError, match="come from the search record"):
        driver.main(_args(inputs, tmp_path / "out", "A",
                          ["--search-record", str(inputs["record"]), "--n-neighbors", "30"]))


def test_code_check_is_stamped_and_needs_every_setting(
    inputs: dict[str, Path], tmp_path: Path,
) -> None:
    """A code check runs only with explicit settings and marks every output as a code check."""
    with pytest.raises(ValueError, match="every setting given explicitly"):
        driver.main(_args(inputs, tmp_path / "x", "A", ["--code-check", "--n-components", "5"]))
    out = tmp_path / "check"
    flags = ["--code-check", "--n-components", "5", "--n-neighbors", "15", "--min-dist", "0.1",
             "--k-min", "2", "--k-max", "6"]
    assert driver.main(_args(inputs, out, "A", flags)) == 0
    assert (out / "CODE_CHECK.txt").read_text().startswith("CODE CHECK")
    record = json.loads((out / "run_record.json").read_text())
    assert record["code_check"] is True
    assert read_repro_bundle(out / "bundle.json").run_id.startswith("code_check/")
    with pytest.raises(ValueError, match="same parameter search"):
        driver.main(_args(inputs, tmp_path / "B", "B", [
            "--search-record", str(inputs["record"]), "--k-from", str(out / "run_record.json")]))


def test_missing_reference_ids_raise(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A table id with no vector stops the run."""
    table = tmp_path / "bad.csv"
    table.write_text("id,corner\nr1,PS\nmissing,PS\n")
    args = _args(inputs, tmp_path / "out", "A", ["--search-record", str(inputs["record"])])
    args[args.index(str(inputs["table"]))] = str(table)
    with pytest.raises(ValueError, match="no vector"):
        driver.main(args)


def test_search_record_rejects_unknown_fields(tmp_path: Path) -> None:
    """The record format is strict: an unexpected field fails on read."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"dataset": "toy", "unexpected": 1}))
    with pytest.raises(Exception, match="unexpected"):
        sr.read_search_record(path)
