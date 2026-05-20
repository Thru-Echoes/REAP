"""Rung 1 - tiny-synthetic property verification for ARI (verification ladder).

40 samples in 2-d from ``sklearn.datasets.make_blobs(n_samples=40, centers=4,
cluster_std=0.3, random_state=0)``, clustered twice with
``KMeans(n_clusters=4, random_state=0/1)``. Closed-form properties are
asserted on the resulting label arrays; every concrete ARI value is
cross-checked against ``_reference_ari.reference_ari`` (which Rung 0
already proved correct on hand-computed values).

Properties at this rung (gate: 1e-12 within agreement, 1e-9 within property
predictions):

    P1.  Both seeds recover the true blob structure: ARI(seed_i, gt) > 0.95.
    P2.  The two seeds agree with each other: ARI(seed_0, seed_1) > 0.95.
    P3.  ARI is permutation-invariant at scale: relabeling seed_1's
         label identifiers preserves ARI to within 1e-12.
    P4.  Random labels are uncorrelated with structure:
         ARI(rand_labels, gt) ~ 0; assert |ARI| < 0.15 across 5 random seeds.
    P5.  ``compute_pairwise_ari`` / ``compute_seed_stability`` agree with
         ``reference_ari`` on the produced label arrays (1e-12 tolerance).
    P6.  Tri-view per-seed has ground truth at this rung; s2s has only one
         pair (one value); s2c is reported as n/a (consensus undefined for
         a 2-seed rung; first defined at Rung 2).

A RED here blocks Rung 2: invoke ``superpowers:systematic-debugging`` on
``src/reap/evaluation.py`` and restart the ladder from Rung 0.

Plan reference: ``docs/superpowers/plans/2026-05-18-metric-correctness-
verification-ladder.md`` (gitignored), Task 2.
"""

from __future__ import annotations

import numpy as np
import pytest
from _reference_ari import reference_ari
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from reap.evaluation import compute_pairwise_ari, compute_seed_stability

TOL = 1e-12


@pytest.fixture(scope="module")
def rung1_data():
    """Well-separated blobs + two deterministic KMeans labelings + GT."""
    blobs = make_blobs(
        n_samples=40,
        centers=4,
        cluster_std=0.3,
        random_state=0,
        return_centers=False,
    )
    X, ground_truth = blobs[0], blobs[1]
    seed_0_labels = KMeans(n_clusters=4, random_state=0, n_init="auto").fit_predict(X)
    seed_1_labels = KMeans(n_clusters=4, random_state=1, n_init="auto").fit_predict(X)
    return {
        "X": X,
        "ground_truth": ground_truth,
        "seed_0": seed_0_labels,
        "seed_1": seed_1_labels,
    }


# ---------------------------------------------------------------------------
# P1 - Both seeds recover the true blob structure
# ---------------------------------------------------------------------------


def test_p1_each_seed_recovers_structure(rung1_data):
    """Well-separated blobs (cluster_std=0.3) ⇒ ARI(seed_i, GT) > 0.95."""
    gt = rung1_data["ground_truth"]
    s0 = rung1_data["seed_0"]
    s1 = rung1_data["seed_1"]

    ari_s0_gt = reference_ari(s0, gt)
    ari_s1_gt = reference_ari(s1, gt)

    assert ari_s0_gt > 0.95, f"seed_0 vs GT ARI={ari_s0_gt!r} should be > 0.95"
    assert ari_s1_gt > 0.95, f"seed_1 vs GT ARI={ari_s1_gt!r} should be > 0.95"


# ---------------------------------------------------------------------------
# P2 - The two seeds agree with each other
# ---------------------------------------------------------------------------


def test_p2_two_seeds_agree(rung1_data):
    """Both deterministic KMeans runs on the same well-separated blobs ⇒
    seed-to-seed ARI > 0.95."""
    s0 = rung1_data["seed_0"]
    s1 = rung1_data["seed_1"]
    ari = reference_ari(s0, s1)
    assert ari > 0.95, f"seed_0 vs seed_1 ARI={ari!r} should be > 0.95"


# ---------------------------------------------------------------------------
# P3 - Permutation invariance at scale
# ---------------------------------------------------------------------------


def test_p3_permutation_invariance_at_scale(rung1_data):
    """Relabeling seed_1's cluster ids does not change ARI(seed_0, seed_1)."""
    s0 = rung1_data["seed_0"]
    s1 = rung1_data["seed_1"]

    base = reference_ari(s0, s1)
    rng = np.random.RandomState(0)
    permutation = rng.permutation(4)
    s1_permuted = permutation[s1]
    permuted = reference_ari(s0, s1_permuted)

    assert abs(permuted - base) < TOL, (
        f"Label permutation changed ARI: base={base!r}, permuted={permuted!r}, "
        f"delta={permuted - base!r}"
    )


# ---------------------------------------------------------------------------
# P4 - Random labels are uncorrelated with structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_p4_random_labels_have_small_ari(rung1_data, seed):
    """Uniform random labels over 4 classes vs GT ⇒ |ARI| < 0.15."""
    gt = rung1_data["ground_truth"]
    rng = np.random.RandomState(seed)
    random_labels = rng.randint(0, 4, size=gt.shape[0])
    ari = reference_ari(random_labels, gt)
    assert abs(ari) < 0.15, f"random labels (seed={seed}) ARI={ari!r}; expected |ARI| < 0.15"


# ---------------------------------------------------------------------------
# P5 - Code under test agrees with reference impl on the produced labels
# ---------------------------------------------------------------------------


def test_p5_compute_pairwise_ari_matches_reference(rung1_data):
    """``compute_pairwise_ari`` on [seed_0, seed_1] equals ``reference_ari`` (1e-12)."""
    labels_list = [rung1_data["seed_0"], rung1_data["seed_1"]]
    matrix = compute_pairwise_ari(labels_list)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0 and matrix[1, 1] == 1.0
    assert np.array_equal(matrix, matrix.T)

    ref = reference_ari(labels_list[0], labels_list[1])
    assert abs(matrix[0, 1] - ref) < TOL, (
        f"compute_pairwise_ari[0,1]={matrix[0, 1]!r} vs reference={ref!r}, "
        f"delta={matrix[0, 1] - ref!r}"
    )


def test_p5_compute_seed_stability_matches_reference(rung1_data):
    """``compute_seed_stability`` (s2s component) matches ``reference_ari`` (1e-12)."""
    seeds = [rung1_data["seed_0"], rung1_data["seed_1"]]
    consensus_dummy = rung1_data["ground_truth"]  # arbitrary; we only check s2s here
    stats = compute_seed_stability(seeds, consensus_dummy)

    ari_pair = reference_ari(seeds[0], seeds[1])
    assert abs(stats["s2s_ari_mean"] - ari_pair) < TOL, (
        f"s2s_ari_mean={stats['s2s_ari_mean']!r} vs single-pair reference={ari_pair!r}"
    )
    assert stats["s2s_ari_std"] == 0.0, (
        f"single-pair s2s std should be 0; got {stats['s2s_ari_std']!r}"
    )
    assert abs(stats["s2s_ari_median"] - ari_pair) < TOL


# ---------------------------------------------------------------------------
# P6 - Tri-view at Rung 1: per-seed (with GT) + s2s (1 pair) + s2c = n/a
# ---------------------------------------------------------------------------


def test_p6_tri_view_per_seed_has_ground_truth(rung1_data):
    """At Rung 1 with ground truth, per-seed ARI vs GT is the defined view.
    Report (and assert) the per-seed values and the s2s single pair; s2c is
    n/a at this rung (consensus not defined for 2 seeds).
    """
    gt = rung1_data["ground_truth"]
    s0 = rung1_data["seed_0"]
    s1 = rung1_data["seed_1"]

    per_seed_ari_to_gt = [reference_ari(s0, gt), reference_ari(s1, gt)]
    s2s_ari = [reference_ari(s0, s1)]
    s2c_ari = None  # n/a (consensus undefined at this rung)
    s2c_reason = "consensus undefined for a 2-seed rung; first defined at Rung 2"

    assert all(v > 0.95 for v in per_seed_ari_to_gt), (
        f"per-seed ARIs vs GT must all exceed 0.95: {per_seed_ari_to_gt}"
    )
    assert len(s2s_ari) == 1 and s2s_ari[0] > 0.95, (
        f"s2s should be a single ARI > 0.95: {s2s_ari}"
    )
    assert s2c_ari is None and s2c_reason  # explicit n/a, never silent
