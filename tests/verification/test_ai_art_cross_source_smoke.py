"""Smoke test for the AI-art cross-source analyses (A1/A2/A3, §19).

Validates that ``scripts/run_cross_source_analyses.py`` produced the three
JSON artifacts with the expected schema and value ranges. Does not re-run
the analyses.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
CROSS_SOURCE_ROOT = (
    REPO_ROOT / "results" / "projection_head" / "ai_art" / "oos_demo" / "cross_source"
)
THEMES = ("compensation", "ownership", "threat", "transparency", "utility")


def test_a1_demographic_stratification_schema():
    path = CROSS_SOURCE_ROOT / "A1_demographic_stratification.json"
    assert path.exists(), f"missing: {path}"
    d = json.loads(path.read_text())
    assert d["analysis"] == "A1_demographic_stratification"
    assert d["n_artists"] == 1259
    assert "demographics" in d and "significant_demographics" in d
    for col, block in d["demographics"].items():
        assert 0.0 <= block["cramers_v"] <= 1.0, f"{col}: Cramér's V out of [0,1]"
        assert 0.0 <= block["p_value"] <= 1.0, f"{col}: p out of [0,1]"
        assert isinstance(block["holm_significant"], bool)
        # contingency table rows sum to the n artists
        table = block["table"]
        assert sum(sum(row) for row in table) == 1259


def test_a2_artist_vs_public_alignment_schema():
    path = CROSS_SOURCE_ROOT / "A2_artist_vs_public_alignment.json"
    assert path.exists(), f"missing: {path}"
    d = json.loads(path.read_text())
    assert d["analysis"] == "A2_artist_vs_public_alignment"
    assert set(d["per_theme"].keys()) == set(THEMES)
    for theme in THEMES:
        b = d["per_theme"][theme]
        assert b["kl_artist_to_public"] >= 0.0, f"{theme}: KL must be non-negative"
        assert b["wasserstein_cluster_axis"] >= 0.0
        # artist themes are balanced ~252 (251 for threat); public exactly 150
        assert b["n_public"] == 150
        assert b["n_artist"] in (251, 252)
        assert len(b["artist_occupancy"]) == len(b["public_occupancy"])
    assert d["most_divergent_theme"] in THEMES
    assert d["least_divergent_theme"] in THEMES


def test_a3_baserate_temporal_schema():
    path = CROSS_SOURCE_ROOT / "A3_baserate_temporal.json"
    assert path.exists(), f"missing: {path}"
    d = json.loads(path.read_text())
    assert d["analysis"] == "A3_baserate_temporal"
    n_clusters = d["n_clusters"]
    assert len(d["enrichment_ratio"]) == n_clusters
    assert len(d["cluster_mean_year"]) == n_clusters
    assert sum(d["reference_cluster_sizes"]) == 1736
    # base rate is a probability distribution summing to 1
    assert abs(sum(d["base_rate"]) - 1.0) < 1e-9
    # enrichment ratios are non-negative where defined
    for r in d["enrichment_ratio"]:
        assert r is None or r >= 0.0
    assert d["n_enriched_total"] >= 0
    assert d["n_enriched_in_2022_2024"] <= d["n_enriched_total"]
    # enriched clusters all have enrichment > 1
    for e in d["artist_enriched_clusters"]:
        assert e["enrichment"] > 1.0
        if e["mean_year"] is not None:
            assert 2013 <= e["mean_year"] <= 2025
