"""Guards for the protocol v1.5 machine-readable run matrix.

The v1.5 amendment (``docs/proposals/2026-07-protocol-v1.5-amendment.md``)
defines the Holm multiple-comparison families ONCE and commits the full
temporal-holdout run enumeration as ``docs/run_matrix_v1_5.json``. The
Holm-corrected significance threshold changes by an order of magnitude with
the family size, so a drifting or re-derivable-by-hand-only family is
family-shopping waiting to happen. These tests make the counts mechanical:

- the per-corpus family sizes recomputed from the enumerated cells must
  equal both the JSON's declared sizes and the counts stated in the
  amendment text (parsed from its ``family_size(...) = N`` lines);
- every pre-registered paired baseline and scalar test metric appears
  exactly once per run cell (no duplication, no omission, no per-cell
  cherry-picking);
- the pre-registered constants the matrix repeats from protocol §18 —
  strategy counts, encoders, random-split master seeds, seed set D — match
  the protocol's values, and Set D actually exists in the seed manifest.

Both inputs are tracked files, so this runs in code-only CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_MATRIX_PATH = REPO_ROOT / "docs" / "run_matrix_v1_5.json"
AMENDMENT_PATH = REPO_ROOT / "docs" / "proposals" / "2026-07-protocol-v1.5-amendment.md"
SEED_MANIFEST_PATH = REPO_ROOT / "manuscript" / "seeds" / "seed_manifest.json"

KF_ENCODERS = {
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
}
AI_ART_ENCODER = "intfloat/e5-large-v2"

# Pre-registered in protocol §18.b/§18.c; repeated here so a silent edit to
# the matrix (or the protocol) fails loudly.
EXPECTED_MASTER_SEEDS = {
    "ai_art/T5": [20260521, 20260522, 20260523],
    "korean_forest/K2-T4": [20260524],
    "korean_forest/K2-T5": [20260525],
}


@pytest.fixture(scope="module")
def matrix() -> dict:
    """Load the committed run matrix; a missing file is a hard failure."""
    assert RUN_MATRIX_PATH.exists(), f"missing pre-registered run matrix: {RUN_MATRIX_PATH}"
    return json.loads(RUN_MATRIX_PATH.read_text())


@pytest.fixture(scope="module")
def amendment_family_sizes() -> dict[str, int]:
    """Parse the ``family_size(<name>) = <N>`` lines from the amendment."""
    assert AMENDMENT_PATH.exists(), f"missing amendment draft: {AMENDMENT_PATH}"
    text = AMENDMENT_PATH.read_text()
    matches = re.finditer(r"family_size\((\w+)\)\s*=\s*(\d+)", text)
    sizes = {m.group(1): int(m.group(2)) for m in matches}
    assert sizes, "amendment states no machine-readable family_size(...) = N lines"
    return sizes


def test_cell_counts_match_protocol_18a(matrix: dict) -> None:
    """5 AI-art cells and 18 Korean-forest cells (6 strategies × 3 encoders)."""
    cells = matrix["cells"]
    ai_art = [c for c in cells if c["corpus"] == "ai_art"]
    kf = [c for c in cells if c["corpus"] == "korean_forest"]
    assert len(ai_art) == 5
    assert len(kf) == 18
    assert len(cells) == 23
    kf_strategies = {c["strategy"] for c in kf}
    assert len(kf_strategies) == 6
    for strategy in kf_strategies:
        encoders = {c["encoder"] for c in kf if c["strategy"] == strategy}
        assert encoders == KF_ENCODERS, f"{strategy} missing an encoder: {encoders}"
    assert {c["encoder"] for c in ai_art} == {AI_ART_ENCODER}


def test_family_sizes_recompute_from_cells(matrix: dict) -> None:
    """Declared family sizes equal n_cells × n_baselines × n_test_metrics."""
    n_baselines = len(matrix["paired_baselines"])
    n_metrics = len(matrix["primary_test_metrics"])
    declared = matrix["declared_family_sizes"]
    for corpus, family_key in [
        ("ai_art", "ai_art_holdout"),
        ("korean_forest", "korean_forest_holdout"),
    ]:
        n_cells = sum(1 for c in matrix["cells"] if c["corpus"] == corpus)
        assert declared[family_key] == n_cells * n_baselines * n_metrics, family_key


def test_family_sizes_match_amendment_stated_counts(
    matrix: dict, amendment_family_sizes: dict[str, int]
) -> None:
    """The JSON's declared sizes and the amendment's stated counts agree."""
    assert amendment_family_sizes == matrix["declared_family_sizes"]


def test_every_baseline_and_metric_exactly_once_per_cell(matrix: dict) -> None:
    """No run cell drops, duplicates, or adds a baseline or test metric."""
    baselines = matrix["paired_baselines"]
    metrics = matrix["primary_test_metrics"]
    assert len(set(baselines)) == len(baselines)
    assert len(set(metrics)) == len(metrics)
    for cell in matrix["cells"]:
        assert sorted(cell["paired_baselines"]) == sorted(baselines), cell["cell_id"]
        assert sorted(cell["primary_test_metrics"]) == sorted(metrics), cell["cell_id"]


def test_random_split_master_seeds_are_the_preregistered_ones(matrix: dict) -> None:
    """T5 / K2-T4 / K2-T5 carry exactly the §18 master seeds; others none."""
    by_key: dict[str, list[int]] = {}
    for cell in matrix["cells"]:
        key = f"{cell['corpus']}/{cell['strategy']}"
        seeds = cell.get("master_seeds", [])
        if key in by_key:
            assert by_key[key] == seeds, f"inconsistent master seeds within {key}"
        by_key[key] = seeds
        assert cell["replicates"] == max(1, len(seeds)), cell["cell_id"]
    for key, expected in EXPECTED_MASTER_SEEDS.items():
        assert by_key.get(key) == expected, f"{key}: {by_key.get(key)} != {expected}"
    for key, seeds in by_key.items():
        if key not in EXPECTED_MASTER_SEEDS:
            assert seeds == [], f"unexpected master seeds on non-random strategy {key}"


def test_seed_set_d_exists_and_matches(matrix: dict) -> None:
    """The matrix runs on Set D and Set D has 30 seeds in the manifest."""
    assert matrix["seed_set"] == "D"
    assert matrix["n_seeds"] == 30
    manifest = json.loads(SEED_MANIFEST_PATH.read_text())
    assert len(manifest["sets"]["D"]["seeds"]) == 30


def test_a1_demographics_family_is_nine_without_country(matrix: dict) -> None:
    """Clause (l): Country is constant (all-USA) and excluded; family m=9."""
    demographics = matrix["a1_demographics"]
    assert len(demographics) == 9
    assert len(set(demographics)) == 9
    assert "Country" not in demographics
    assert matrix["declared_family_sizes"]["a1_demographics"] == 9


def test_core_methods_family_is_five(matrix: dict) -> None:
    """§8's core family: REAP vs the five §3 baselines, per dataset per metric."""
    core = matrix["core_paired_baselines"]
    assert len(core) == 5
    assert len(set(core)) == 5
    assert matrix["declared_family_sizes"]["core_methods_per_dataset_per_metric"] == 5


def test_distance_correlation_means_the_condensed_recipe(matrix: dict) -> None:
    """The matrix's distance_correlation metric is the reported-metric recipe.

    Protocol v1.5 clause (m) registers two recipes under the everyday name;
    the run matrix must bind its test metric to the condensed-vector recipe
    id so no aggregation step can quietly substitute the loss variant.
    """
    note = matrix["metric_recipe_notes"]["distance_correlation"]
    assert "condensed" in note
    assert "dist_corr_loss" in note  # names the excluded sibling recipe
