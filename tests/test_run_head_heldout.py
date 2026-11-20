"""End-to-end test of scripts/run_head_heldout.py on a small saved map."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_head_heldout as script  # noqa: E402  # pyright: ignore[reportMissingImports]

SETTINGS = {
    "n_folds": 4, "fold_seed": 3, "alpha": 0.1,
    "heads": {"ridge": {"alpha_grid": [0.1, 1.0]},
              "knn": {"k_grid": [1, 3, 5], "weights": "distance", "metric": "cosine",
                      "n_inner_folds": 4, "random_state": 3}},
}


@pytest.fixture()
def inputs(tmp_path: Path) -> dict[str, Path]:
    """A 160-passage reference from two sources, its saved 2-d map, and one probe set."""
    data = make_blobs(n_samples=200, centers=4, n_features=12, cluster_std=1.0, random_state=1)
    X, y = np.asarray(data[0], dtype=np.float32), np.asarray(data[1])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = np.array([f"{'a' if i % 2 else 'b'}-{i}" for i in range(160)])
    np.savez(tmp_path / "ref.npz", ids=ids, vectors=X[:160])
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "reference.npz", ids=ids, consensus=X[:160, :2] * 5.0, labels=y[:160])
    np.savez(tmp_path / "probe.npz", ids=np.array([f"p-{i}" for i in range(40)]),
             vectors=X[160:])
    (tmp_path / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8")
    return {"ref": tmp_path / "ref.npz", "run": run, "probe": tmp_path / "probe.npz",
            "settings": tmp_path / "settings.json", "out": tmp_path / "out"}


def _argv(p: dict[str, Path]) -> list[str]:
    """Command-line arguments for the fixture."""
    return ["--reference", str(p["ref"]), "--run-dir", str(p["run"]),
            "--probe", f"probe={p['probe']}", "--head-settings", str(p["settings"]),
            "--source-separator", "-", "--output-dir", str(p["out"])]


def test_writes_tables_and_calibrated_summary(inputs: dict[str, Path]) -> None:
    """Every head gets a held-out table, a probe table and a summary calibrated at alpha."""
    assert script.main(_argv(inputs)) == 0
    summary = json.loads((inputs["out"] / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["heads"]) == {"ridge", "knn"}
    for kind, head in summary["heads"].items():
        held = head["heldout"]
        assert held["inside_share"] >= 1 - SETTINGS["alpha"] - 1e-9
        assert set(held["miss_median_by_source"]) == {"a", "b"}
        assert head["probes"]["probe"]["n"] == 40
        with (inputs["out"] / f"heldout_{kind}.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 160 and {r["fold"] for r in rows} == {"0", "1", "2", "3"}
        with (inputs["out"] / f"probe_placement_{kind}.csv").open(encoding="utf-8") as fh:
            assert len(list(csv.DictReader(fh))) == 40
        with np.load(inputs["out"] / f"placements_{kind}.npz") as placed:
            assert placed["heldout"].shape == (160, 2)
            assert placed["probe__probe"].shape == (40, 2)
            assert len(placed["probe_ids__probe"]) == 40
    assert summary["heads"]["knn"]["chosen_full"]["k"] in (1, 3, 5)


def test_refuses_overwrite_and_bad_settings(inputs: dict[str, Path]) -> None:
    """A second run into the same folder, a missing setting or an unknown head stop it."""
    assert script.main(_argv(inputs)) == 0
    with pytest.raises(SystemExit):
        script.main(_argv(inputs))
    inputs["settings"].write_text(json.dumps({"n_folds": 4, "heads": {}}), encoding="utf-8")
    inputs["out"] = inputs["out"].with_name("out2")
    with pytest.raises(SystemExit):
        script.main(_argv(inputs))
    inputs["settings"].write_text(json.dumps({**SETTINGS, "heads": {"forest": {}}}),
                                  encoding="utf-8")
    with pytest.raises(SystemExit):
        script.main(_argv(inputs))
