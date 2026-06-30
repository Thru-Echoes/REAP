"""Smoke test for the projection-head OOS metrics.

Confirms that the JSON artifacts produced by
``scripts/run_projection_head_oos.py`` exist for all (corpus, set) tuples
and contain the expected schema + sensible ranges. Does NOT re-train the
projection head (slow + duplicates the script's work); the reproducibility
guarantee is encoded in the script's deterministic-seed setup.

The load-bearing claim this test enforces:
    - per_fold list has 5 entries with the expected metric keys
    - cv_summary has mean/std/min/max for each metric
    - trustworthiness CV mean is in [0, 1]
    - distance_correlation CV mean is in [-1, 1]
    - ARI CV mean is in [-1, 1]
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
PROJ_HEAD_ROOT = REPO_ROOT / "results" / "projection_head"

CORPORA = ("twenty_newsgroups_reference", "ai_art", "korean_forest")
SETS = ("A", "B", "C")

REQUIRED_METRIC_KEYS = (
    "mse",
    "trustworthiness",
    "silhouette",
    "ari",
    "distance_correlation",
)


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_oos_metrics_file_exists_and_well_formed(corpus, set_letter):
    """The OOS metrics JSON exists and has the required schema."""
    path = PROJ_HEAD_ROOT / corpus / f"set_{set_letter}" / "oos_metrics.json"
    assert path.exists(), f"missing: {path}"

    with open(path) as fp:
        payload = json.load(fp)

    for key in ("corpus", "set", "git_commit", "seed", "config", "per_fold", "cv_summary"):
        assert key in payload, f"{path} missing key '{key}'"

    assert payload["corpus"] == corpus
    assert payload["set"] == set_letter
    assert isinstance(payload["per_fold"], list)
    assert len(payload["per_fold"]) == 5  # 5-fold CV
    for fold in payload["per_fold"]:
        for metric_key in REQUIRED_METRIC_KEYS:
            assert metric_key in fold, f"fold missing metric '{metric_key}'"

    for metric_key in REQUIRED_METRIC_KEYS:
        summary = payload["cv_summary"][metric_key]
        for stat in ("mean", "std", "min", "max"):
            assert stat in summary, f"cv_summary[{metric_key}] missing '{stat}'"


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("set_letter", SETS)
def test_oos_metrics_ranges(corpus, set_letter):
    """Metrics fall in their theoretical ranges."""
    path = PROJ_HEAD_ROOT / corpus / f"set_{set_letter}" / "oos_metrics.json"
    with open(path) as fp:
        payload = json.load(fp)
    cv = payload["cv_summary"]

    assert 0.0 <= cv["trustworthiness"]["mean"] <= 1.0
    assert -1.0 <= cv["distance_correlation"]["mean"] <= 1.0
    assert -1.0 <= cv["ari"]["mean"] <= 1.0
    assert -1.0 <= cv["silhouette"]["mean"] <= 1.0
    assert cv["mse"]["mean"] >= 0.0


@pytest.mark.parametrize("corpus", CORPORA)
def test_oos_trustworthiness_above_minimum_threshold(corpus):
    """Trustworthiness CV mean exceeds the minimum needed for the OOS-angle
    claim across all three sets for each corpus.

    Threshold 0.70 reflects an honest sanity floor: any number below this on
    the production corpora would refute the "neighborhood is preserved OOS"
    claim. Observed values are 0.77–0.86; 0.70 sits comfortably below them
    while still catching a significant regression.
    """
    for set_letter in SETS:
        path = PROJ_HEAD_ROOT / corpus / f"set_{set_letter}" / "oos_metrics.json"
        with open(path) as fp:
            payload = json.load(fp)
        cv_trust = payload["cv_summary"]["trustworthiness"]["mean"]
        assert cv_trust > 0.70, (
            f"{corpus} set {set_letter}: trustworthiness CV mean = {cv_trust:.4f}; "
            "expected > 0.70 for the OOS-projection claim to hold"
        )
