"""Guards for the protocol v1.6 run matrix (hold-out scope and per-seed unit).

The v1.6 amendment (``docs/proposals/2026-11-protocol-v1.6-amendment.md``)
runs 11 of the 23 registered hold-out cells first and defers the 12
Korean-forest cells on the two non-canonical encoders as a separate family.
These tests keep that reduction mechanical, so it cannot drift into a silent
deviation:

- the v1.6 matrix still lists all 23 v1.5 cells, with the same ids and
  settings, each marked ``in_scope`` or ``deferred`` with a reason;
- exactly the canonical-encoder cells are in scope;
- family sizes recomputed from the in-scope and deferred cells equal both the
  JSON's declared sizes and the amendment's ``family_size(...) = N`` lines;
- the per-seed unit is the one the amendment defines (30 training runs per
  trained method, pairing by seed index, identity deterministic).

Inputs are tracked files, so this runs in code-only CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_V1_6_PATH = REPO_ROOT / "docs" / "run_matrix_v1_6.json"
MATRIX_V1_5_PATH = REPO_ROOT / "docs" / "run_matrix_v1_5.json"
AMENDMENT_PATH = REPO_ROOT / "docs" / "proposals" / "2026-11-protocol-v1.6-amendment.md"

CANONICAL_KF_ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def matrix() -> dict:
    """Load the v1.6 run matrix; a missing file is a hard failure."""
    assert MATRIX_V1_6_PATH.exists(), f"missing run matrix: {MATRIX_V1_6_PATH}"
    return json.loads(MATRIX_V1_6_PATH.read_text())


@pytest.fixture(scope="module")
def matrix_v1_5() -> dict:
    """Load the v1.5 registration the v1.6 matrix must stay faithful to."""
    return json.loads(MATRIX_V1_5_PATH.read_text())


def _get_cells(matrix: dict, corpus: str, status: str) -> list[dict]:
    """Return the cells of one corpus with one status."""
    return [c for c in matrix["cells"] if c["corpus"] == corpus and c["status"] == status]


def test_version_and_amendment_link(matrix: dict) -> None:
    """The matrix names protocol 1.6 and an amendment file that exists."""
    assert matrix["protocol_version"] == "1.6"
    assert (REPO_ROOT / matrix["amendment"]).exists()


def test_all_v1_5_cells_kept_unchanged_apart_from_status(
    matrix: dict, matrix_v1_5: dict
) -> None:
    """No registered cell is dropped, added, or re-specified — only marked."""
    v1_6 = {c["cell_id"]: c for c in matrix["cells"]}
    v1_5 = {c["cell_id"]: c for c in matrix_v1_5["cells"]}
    assert v1_6.keys() == v1_5.keys()
    for cell_id, old in v1_5.items():
        new = {k: v for k, v in v1_6[cell_id].items() if k not in ("status", "deferral_reason")}
        assert new == old, cell_id


def test_every_cell_has_a_status_and_deferred_cells_a_reason(matrix: dict) -> None:
    """Statuses are in_scope or deferred; a deferred cell always says why."""
    for cell in matrix["cells"]:
        assert cell["status"] in {"in_scope", "deferred"}, cell["cell_id"]
        if cell["status"] == "deferred":
            assert cell.get("deferral_reason"), cell["cell_id"]
        else:
            assert "deferral_reason" not in cell, cell["cell_id"]


def test_exactly_the_canonical_encoder_cells_are_in_scope(matrix: dict) -> None:
    """In scope: all 5 AI-art cells and the 6 canonical-encoder KF cells."""
    assert len(_get_cells(matrix, "ai_art", "in_scope")) == 5
    assert _get_cells(matrix, "ai_art", "deferred") == []
    kf_in = _get_cells(matrix, "korean_forest", "in_scope")
    kf_deferred = _get_cells(matrix, "korean_forest", "deferred")
    assert len(kf_in) == 6
    assert len(kf_deferred) == 12
    assert {c["encoder"] for c in kf_in} == {CANONICAL_KF_ENCODER}
    assert CANONICAL_KF_ENCODER not in {c["encoder"] for c in kf_deferred}
    assert {c["strategy"] for c in kf_in} == {c["strategy"] for c in kf_deferred}


def test_family_sizes_recompute_from_cells(matrix: dict) -> None:
    """Each declared family = its cells × baselines × test metrics."""
    per_cell = len(matrix["paired_baselines"]) * len(matrix["primary_test_metrics"])
    declared = matrix["declared_family_sizes"]
    assert declared["ai_art_holdout"] == len(_get_cells(matrix, "ai_art", "in_scope")) * per_cell
    assert declared["korean_forest_holdout"] == (
        len(_get_cells(matrix, "korean_forest", "in_scope")) * per_cell
    )
    assert declared["korean_forest_holdout_deferred_encoders"] == (
        len(_get_cells(matrix, "korean_forest", "deferred")) * per_cell
    )


def test_family_sizes_match_amendment_stated_counts(matrix: dict) -> None:
    """The JSON's declared sizes and the amendment's stated counts agree."""
    text = AMENDMENT_PATH.read_text()
    stated = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"family_size\((\w+)\)\s*=\s*(\d+)", text)
    }
    assert stated == matrix["declared_family_sizes"]


def test_per_seed_unit_is_the_amended_one(matrix: dict) -> None:
    """30 training runs per trained method, paired by seed; identity is deterministic."""
    unit = matrix["per_seed_unit"]
    assert unit["training_runs_per_method"] == matrix["n_seeds"] == 30
    assert set(unit["trained_methods"]) | set(unit["deterministic_methods"]) == set(
        matrix["methods"]
    )
    assert unit["deterministic_methods"] == ["identity"]
    assert unit["pairing"] == "by seed index i"


def test_generalization_interval_is_outside_the_holm_families(matrix: dict) -> None:
    """The resampling interval is descriptive, pre-designated, and not a family member."""
    interval = matrix["generalization_interval"]
    assert interval["in_holm_family"] is False
    assert interval["resamples"] == 2000
    assert "run 1" in interval["run"]


def test_cross_encoder_metric_is_deferred(matrix: dict) -> None:
    """With one encoder in scope, the cross-encoder metric cannot be computed."""
    assert matrix["kf_only_supporting_metrics"] == []
    assert "cross_encoder_procrustes_disparity" in matrix["deferred_metrics"]
