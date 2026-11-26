"""Guards for the protocol v1.8 run matrix (presidential environmental reference map).

The v1.8 amendment (``docs/proposals/2026-11-protocol-v1.8-amendment.md``)
registers the third case study after analysis. These tests keep its run
matrix consistent with the amendment, so the registered runs cannot drift:

- the matrix names protocol 1.8, says it was registered after analysis, and
  links an amendment file that exists;
- the seed manifest it names is REAP's and has the recorded SHA-256;
- every stage has exactly one cell per seed set and method, and the counts
  match the declared counts;
- every cell names a seed set, a method and a result path, and ids are unique.

Inputs are tracked files, so this runs in code-only CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "docs" / "run_matrix_v1_8.json"


@pytest.fixture(scope="module")
def matrix() -> dict:
    """Load the v1.8 run matrix; a missing file is a hard failure."""
    assert MATRIX_PATH.exists(), f"missing run matrix: {MATRIX_PATH}"
    return json.loads(MATRIX_PATH.read_text())


def test_version_and_amendment_link(matrix: dict) -> None:
    """The matrix names protocol 1.8, is marked post-analysis, and its amendment exists."""
    assert matrix["protocol_version"] == "1.8"
    assert matrix["registered_after_analysis"] is True
    assert (REPO_ROOT / matrix["amendment"]).exists()


def test_seed_manifest_hash(matrix: dict) -> None:
    """The seed manifest is REAP's own and unchanged since the runs."""
    manifest = REPO_ROOT / matrix["seed_manifest"]
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == matrix["seed_manifest_sha256"]
    manifest_sets = json.loads(manifest.read_text())["sets"]
    for seed_set in matrix["seed_sets"]:
        assert len(manifest_sets[seed_set]["seeds"]) == matrix["n_seeds_per_set"], seed_set


def test_one_cell_per_stage_method_and_set(matrix: dict) -> None:
    """Each stage covers every seed set once per method, and counts match the declaration."""
    seed_sets = set(matrix["seed_sets"])
    expected_methods = {
        "seed_agreement": {"reap"},
        "benchmark": set(matrix["benchmark_methods"]),
        "default_head_check": {"reap_default_head"},
    }
    for stage, methods in expected_methods.items():
        pairs = [
            (c["method"], c["seed_set"]) for c in matrix["cells"] if c["stage"] == stage
        ]
        assert len(pairs) == len(set(pairs)), stage
        assert set(pairs) == {(m, s) for m in methods for s in seed_sets}, stage
        assert len(pairs) == matrix["declared_cell_counts"][stage], stage
    assert {c["stage"] for c in matrix["cells"]} == set(expected_methods)


def test_cells_are_complete(matrix: dict) -> None:
    """Every cell has a non-empty id, method, seed set and result path; ids are unique."""
    ids = [c["cell_id"] for c in matrix["cells"]]
    assert len(ids) == len(set(ids))
    for cell in matrix["cells"]:
        for field in ("cell_id", "stage", "method", "seed_set", "result_path"):
            assert cell.get(field), (cell.get("cell_id"), field)
        assert cell["seed_set"] in matrix["seed_sets"]
        assert cell["result_path"].endswith(f"set_{cell['seed_set']}")
