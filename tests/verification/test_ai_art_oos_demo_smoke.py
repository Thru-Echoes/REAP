"""Smoke test for the AI-art OOS empirical demo.

Validates that ``scripts/run_ai_art_oos_demo.py`` produces a complete,
well-formed bundle of artifacts under
``results/projection_head/ai_art/oos_demo/`` for all three consensus sets
(A, B, C), plus a top-level ``cross_set_consistency.json``. Does NOT
re-train the projection head; the deterministic-seed scaffolding in the
demo script is the reproducibility guarantee.

The load-bearing claims this test enforces:

- Per-set: head weights, projections, assignments, comparison_metrics,
  temporal_alignment all exist with expected shapes.
- Probe cluster assignments live in the 20-cluster reference label space.
- All comparison metrics fall in their theoretical ranges (KL >= 0,
  knn_purity in [0, 1], silhouette in [-1, 1], Cramer's V in [0, 1]).
- A pre-registered "honest sanity floor": at least one reference cluster
  contains BOTH artist and public probes (i.e., the head doesn't
  trivially route a single probe source into one corner of the space).
- Cross-set Procrustes disparity is finite and non-negative for all three
  set pairs and both probe sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path("/Users/echoes/Documents/Berkeley/Research/REAP")
OOS_DEMO_ROOT = REPO_ROOT / "results" / "projection_head" / "ai_art" / "oos_demo"
SETS = ("A", "B", "C")

EXPECTED_N_ARTIST = 1259
EXPECTED_N_PUBLIC = 750
EXPECTED_OUTPUT_DIM = 5
# Per-set effective K from the consensus pipeline (TRACE evt_002):
# the pipeline targets k=20 but drops empty clusters, so sets A/B/C have 20/19/18.
EXPECTED_N_REF_CLUSTERS_PER_SET = {"A": 20, "B": 19, "C": 18}


REQUIRED_PER_SOURCE_KEYS = (
    "n",
    "input_dim",
    "output_dim",
    "probe_input_sha256",
    "occupancy_hist",
    "kl_to_reference",
    "knn_purity_mean",
    "knn_purity_std",
    "centroid_cosine_median",
    "theme_cluster_contingency",
    "chi2_statistic",
    "cramers_v",
    "source_separability_silhouette",
    "probe_source_name",
)


def _load_comparison(set_letter: str) -> dict:
    path = OOS_DEMO_ROOT / f"set_{set_letter}" / "comparison_metrics.json"
    assert path.exists(), f"missing: {path}"
    with open(path) as fp:
        return json.load(fp)


@pytest.mark.parametrize("set_letter", SETS)
def test_per_set_artifacts_exist(set_letter):
    """Every per-set output file exists with the expected name."""
    set_dir = OOS_DEMO_ROOT / f"set_{set_letter}"
    for name in (
        "head_state_dict.pt",
        "head_config.json",
        "projection_artist.npy",
        "projection_public.npy",
        "assignment_artist.npy",
        "assignment_public.npy",
        "comparison_metrics.json",
        "temporal_alignment.json",
    ):
        assert (set_dir / name).exists(), f"missing artifact: {set_dir / name}"


@pytest.mark.parametrize("set_letter", SETS)
def test_projection_shapes_and_finiteness(set_letter):
    """Projected coords have the expected shape, dtype, and are finite."""
    set_dir = OOS_DEMO_ROOT / f"set_{set_letter}"
    artist = np.load(set_dir / "projection_artist.npy")
    public = np.load(set_dir / "projection_public.npy")
    assert artist.shape == (EXPECTED_N_ARTIST, EXPECTED_OUTPUT_DIM)
    assert public.shape == (EXPECTED_N_PUBLIC, EXPECTED_OUTPUT_DIM)
    assert np.isfinite(artist).all()
    assert np.isfinite(public).all()
    # 5-d UMAP outputs are bounded — anything wildly large is a red flag.
    artist_max = float(np.abs(artist).max())
    public_max = float(np.abs(public).max())
    assert artist_max < 100.0, f"artist coords out of expected range: max={artist_max}"
    assert public_max < 100.0, f"public coords out of expected range: max={public_max}"


@pytest.mark.parametrize("set_letter", SETS)
def test_assignments_in_cluster_label_space(set_letter):
    """Probe assignments are integers in [0, K-1] for the set's effective K."""
    n_clusters = EXPECTED_N_REF_CLUSTERS_PER_SET[set_letter]
    set_dir = OOS_DEMO_ROOT / f"set_{set_letter}"
    artist_a = np.load(set_dir / "assignment_artist.npy")
    public_a = np.load(set_dir / "assignment_public.npy")
    assert artist_a.shape == (EXPECTED_N_ARTIST,)
    assert public_a.shape == (EXPECTED_N_PUBLIC,)
    assert np.issubdtype(artist_a.dtype, np.integer)
    assert np.issubdtype(public_a.dtype, np.integer)
    assert artist_a.min() >= 0
    assert public_a.min() >= 0
    assert artist_a.max() < n_clusters
    assert public_a.max() < n_clusters


@pytest.mark.parametrize("set_letter", SETS)
def test_comparison_metrics_schema(set_letter):
    """The comparison_metrics.json file matches the documented top-level schema."""
    n_clusters = EXPECTED_N_REF_CLUSTERS_PER_SET[set_letter]
    payload = _load_comparison(set_letter)

    for key in (
        "corpus",
        "set",
        "git_commit",
        "seed",
        "n_ref",
        "n_ref_clusters",
        "ref_input_sha256",
        "ref_target_sha256",
        "head_config",
        "ref_occupancy",
        "probes",
    ):
        assert key in payload, f"set {set_letter}: missing top-level key {key!r}"

    assert payload["corpus"] == "ai_art"
    assert payload["set"] == set_letter
    assert payload["n_ref"] == 1736
    assert payload["n_ref_clusters"] == n_clusters
    assert len(payload["ref_occupancy"]) == n_clusters
    assert sum(payload["ref_occupancy"]) == 1736

    assert set(payload["probes"].keys()) == {"artist", "public"}
    for source in ("artist", "public"):
        block = payload["probes"][source]
        for key in REQUIRED_PER_SOURCE_KEYS:
            assert key in block, f"set {set_letter} {source}: missing key {key!r}"


@pytest.mark.parametrize("set_letter", SETS)
def test_comparison_metric_ranges(set_letter):
    """Each comparison metric falls in its theoretical range."""
    n_clusters = EXPECTED_N_REF_CLUSTERS_PER_SET[set_letter]
    payload = _load_comparison(set_letter)
    for source in ("artist", "public"):
        b = payload["probes"][source]
        assert b["kl_to_reference"] >= 0.0, f"{source}: KL must be non-negative"
        assert 0.0 <= b["knn_purity_mean"] <= 1.0, f"{source}: knn_purity_mean out of [0,1]"
        assert 0.0 <= b["knn_purity_std"] <= 1.0, f"{source}: knn_purity_std out of [0,1]"
        assert -1.0 <= b["centroid_cosine_median"] <= 1.0, (
            f"{source}: centroid_cosine_median out of [-1,1]"
        )
        assert b["chi2_statistic"] >= 0.0, f"{source}: chi2 must be non-negative"
        assert 0.0 <= b["cramers_v"] <= 1.0, f"{source}: Cramer's V out of [0,1]"
        assert -1.0 <= b["source_separability_silhouette"] <= 1.0, (
            f"{source}: source_separability_silhouette out of [-1,1]"
        )
        assert b["n"] == (EXPECTED_N_ARTIST if source == "artist" else EXPECTED_N_PUBLIC)
        assert b["input_dim"] == 1024
        assert b["output_dim"] == EXPECTED_OUTPUT_DIM

        # Occupancy histogram covers the per-set effective K and counts the n probes.
        assert len(b["occupancy_hist"]) == n_clusters
        assert sum(b["occupancy_hist"]) == b["n"]

        # Theme-cluster contingency: 5 rows x K cols summing to the n probes.
        contingency = np.array(b["theme_cluster_contingency"], dtype=np.int64)
        assert contingency.shape == (5, n_clusters)
        assert int(contingency.sum()) == b["n"]


@pytest.mark.parametrize("set_letter", SETS)
def test_honest_sanity_floor_both_sources_share_a_cluster(set_letter):
    """At least one reference cluster must contain BOTH artist and public probes.

    A head that trivially routed all of one source into one corner of the
    consensus space (zero overlap with the other source's clusters) would be
    failing OOS — fix the head, do not relax this test.
    """
    set_dir = OOS_DEMO_ROOT / f"set_{set_letter}"
    artist_a = np.load(set_dir / "assignment_artist.npy")
    public_a = np.load(set_dir / "assignment_public.npy")
    artist_clusters = set(artist_a.tolist())
    public_clusters = set(public_a.tolist())
    shared = artist_clusters & public_clusters
    assert len(shared) >= 1, (
        f"set {set_letter}: no reference cluster contains both probe sources; "
        f"artist clusters = {sorted(artist_clusters)}, public clusters = "
        f"{sorted(public_clusters)}"
    )


@pytest.mark.parametrize("set_letter", SETS)
def test_temporal_alignment_schema_and_ranges(set_letter):
    """temporal_alignment.json contains per-cluster year stats + an ordering."""
    n_clusters = EXPECTED_N_REF_CLUSTERS_PER_SET[set_letter]
    set_dir = OOS_DEMO_ROOT / f"set_{set_letter}"
    with open(set_dir / "temporal_alignment.json") as fp:
        payload = json.load(fp)
    assert set(payload.keys()) == {"per_cluster", "ordering_by_mean_year"}
    assert len(payload["per_cluster"]) == n_clusters
    assert len(payload["ordering_by_mean_year"]) == n_clusters
    assert sorted(payload["ordering_by_mean_year"]) == list(range(n_clusters))

    for entry in payload["per_cluster"]:
        assert "cluster" in entry and 0 <= entry["cluster"] < n_clusters
        assert "n" in entry and entry["n"] >= 0
        if entry["n"] > 0:
            assert 2013 <= entry["mean_year"] <= 2025
            assert 2013 <= entry["q25"] <= entry["q75"] <= 2025


def test_cross_set_consistency_file():
    """cross_set_consistency.json exists and has Procrustes disparities for all 3 pairs."""
    path = OOS_DEMO_ROOT / "cross_set_consistency.json"
    assert path.exists(), f"missing: {path}"
    with open(path) as fp:
        payload = json.load(fp)

    assert set(payload.keys()) >= {"git_commit", "seed", "pairs"}
    assert set(payload["pairs"].keys()) == {"A_vs_B", "A_vs_C", "B_vs_C"}
    for pair_key, pair_block in payload["pairs"].items():
        for source in ("artist", "public"):
            assert source in pair_block, f"{pair_key}: missing source {source}"
            disp = pair_block[source]["procrustes_disparity"]
            assert disp >= 0.0, f"{pair_key} {source}: Procrustes disparity must be non-negative"
            assert disp < 1.0, (
                f"{pair_key} {source}: Procrustes disparity {disp:.4f} is suspiciously high; "
                "two probe coordinate sets from sister consensus runs should be alignable. "
                "scipy.spatial.procrustes disparity is in [0, 1] when both inputs are well-formed."
            )


def test_showcase_figure_exists():
    """The showcase scatter PNG was generated for set A."""
    fig_path = OOS_DEMO_ROOT / "set_A" / "showcase_figure.png"
    assert fig_path.exists(), f"missing figure: {fig_path}"
    assert fig_path.stat().st_size > 10_000, (
        f"figure {fig_path} is suspiciously small ({fig_path.stat().st_size} bytes)"
    )
