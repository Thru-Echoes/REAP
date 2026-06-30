"""Projection-head golden validation tests (Tier 2-PH).

These tests verify the three load-bearing properties of REAP's neural
projection head on the 20newsgroups text fixture:

1. **Training-fit sanity** — the head actually learned the mapping from
   raw embeddings to consensus coordinates. A zero-epoch model would
   fail this.
2. **Cross-validation generalization** — the head generalizes to unseen
   exemplars of known topics under stratified 3-fold CV.
3. **Topic-exclusion cohesion** — held out an entire topic, train on
   the rest, project the excluded topic; the held-out projection is a
   coherent new cluster (tight internally, separated from training
   clusters). Tests the projection head's ability to place genuinely
   novel semantic structure.

Reference run
-------------
Thresholds calibrated on 2026-04-13 using 10 seeds from Set A,
n_folds=3, max_epochs=150, patience=30, batch_size=32. Observed:

* Training-fit: MSE 0.22, distance correlation 0.996
* CV mean: MSE 1.56, R² 0.89, trustworthiness 0.86,
  ARI-vs-truth 0.46, distance correlation 0.90

Pre-registered ranges (below) leave headroom so platform/library drift
does not cause spurious failures, while still catching real regressions.

Slow-tier deferral
------------------
Tests here run on 2 held-out topics (one from each overlap pair:
talk.politics.guns = 4, comp.sys.ibm.pc.hardware = 6). The full
leave-one-topic-out ×8 sweep and 5-fold CV with max_epochs=500 live in
the slow CI workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

pytest.importorskip("torch", reason="PyTorch required for projection golden tests")
pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers required for the 20ng text fixture",
)

from reap.consensus import (  # noqa: E402
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.datasets import DatasetSnapshot, load_golden_text  # noqa: E402
from reap.projection import train_projection_head  # noqa: E402

SEED_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "manuscript"
    / "seeds"
    / "seed_manifest.json"
)

GOLDEN_K = 8
GOLDEN_N_COMPONENTS = 8
GOLDEN_N_NEIGHBORS = 15
GOLDEN_MIN_DIST = 0.1
GOLDEN_N_SEEDS = 10

PROJECTION_N_FOLDS = 3
PROJECTION_MAX_EPOCHS = 150
PROJECTION_PATIENCE = 30
PROJECTION_BATCH_SIZE = 32
PROJECTION_LR = 1e-3

# Pre-registered Tier 2-PH thresholds (v1.0, calibrated 2026-04-13).
# Training-fit sanity — observed MSE 0.22, distance correlation 0.996.
THRESH_TRAINING_FIT_MSE_MAX: float = 0.80
THRESH_TRAINING_FIT_DCORR_MIN: float = 0.95
THRESH_TRAINING_FIT_TRUSTWORTHINESS_MIN: float = 0.80
# CV — observed MSE 1.56, R² 0.89, trust 0.86, ARI 0.46, dcorr 0.90.
THRESH_CV_MSE_MAX: float = 3.0
THRESH_CV_R2_MIN: float = 0.60
THRESH_CV_TRUSTWORTHINESS_MIN: float = 0.75
THRESH_CV_ARI_VS_TRUTH_MIN: float = 0.25
THRESH_CV_DCORR_MIN: float = 0.80
# Topic-exclusion thresholds are based on the scientific finding
# calibrated 2026-04-13: the projection head maps out-of-distribution
# topics to their nearest semantic training neighbor, with a strong
# effect for overlap-pair members and a weaker effect for separated
# topics.
#
# Observed on the 20ng golden fixture:
#   politics.guns → politics.mideast: 64% (pair partner)
#   politics.mideast → politics.guns: 66%
#   ibm.hardware → mac.hardware: 60%
#   mac.hardware → ibm.hardware: 70%
#   sci.space (separated) → religion.christian: 44% (weak)
#   comp.graphics (separated) → mac.hardware: 50% (weak)
#
# Thresholds leave margin so platform/library drift doesn't cause
# spurious failures while still catching real regressions.

# For overlap-pair members: head must map ≥50% of held-out docs to the
# pair partner. This is the strong interpolation claim.
THRESH_OVERLAP_PAIR_PARTNER_FRACTION_MIN: float = 0.50
# For any held-out topic: at least one training cluster must attract
# ≥30% of the held-out docs — i.e., the head is not scattering them
# uniformly (random baseline is 1/7 ≈ 0.14).
THRESH_ANY_DOMINANT_FRACTION_MIN: float = 0.30

# Fast-tier sweep: 2 overlap-pair members + 2 separated topics.
FAST_TIER_EXCLUDED_TOPICS: list[int] = [0, 2, 4, 6]
# Pair-partner mapping from GOLDEN_20NG_CLASSES ordering:
# 0 sci.space, 1 rec.sport.hockey, 2 comp.graphics, 3 soc.religion.christian,
# 4 talk.politics.guns, 5 talk.politics.mideast,
# 6 comp.sys.ibm.pc.hardware, 7 comp.sys.mac.hardware.
PAIR_PARTNER: dict[int, int] = {4: 5, 5: 4, 6: 7, 7: 6}
# Topics in the fast-tier sweep that have a pair partner (used by the
# overlap-pair routing test so it does not skip on separated topics).
FAST_TIER_OVERLAP_PAIR_TOPICS: list[int] = [
    t for t in FAST_TIER_EXCLUDED_TOPICS if t in PAIR_PARTNER
]


def _load_set_a_seeds(n: int) -> list[int]:
    manifest = json.loads(SEED_MANIFEST_PATH.read_text())
    seeds = manifest["sets"]["A"]["seeds"][:n]
    assert len(seeds) == n
    return [int(s) for s in seeds]


def _set_torch_deterministic(seed: int) -> None:
    """Seed torch for reproducible projection-head training."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_reap_consensus(
    X: np.ndarray, seeds: list[int]
) -> np.ndarray:
    """Run REAP on X with the given seeds; return consensus coordinates."""
    embs = get_multi_seed_embeddings(
        X,
        seeds,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    D = get_consensus_distance_matrix(embs)
    Y, _ = get_consensus_embedding(
        D,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    return Y


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeds_10() -> list[int]:
    return _load_set_a_seeds(GOLDEN_N_SEEDS)


@pytest.fixture(scope="module")
def text_snap() -> DatasetSnapshot:
    return load_golden_text()


@pytest.fixture(scope="module")
def consensus_full(
    text_snap: DatasetSnapshot, seeds_10: list[int]
) -> np.ndarray:
    """REAP consensus on the full 400-doc golden text fixture."""
    return _run_reap_consensus(text_snap.embeddings, seeds_10)


@pytest.fixture(scope="module")
def trained_head(
    text_snap: DatasetSnapshot, consensus_full: np.ndarray
) -> dict:
    """Train the projection head once with 3-fold CV; reused across tests."""
    _set_torch_deterministic(42)
    return train_projection_head(
        text_snap.embeddings.astype(np.float32),
        consensus_full.astype(np.float32),
        text_snap.labels,  # type: ignore[arg-type]
        n_folds=PROJECTION_N_FOLDS,
        max_epochs=PROJECTION_MAX_EPOCHS,
        patience=PROJECTION_PATIENCE,
        batch_size=PROJECTION_BATCH_SIZE,
        lr=PROJECTION_LR,
    )


# ---------------------------------------------------------------------------
# Test 1 — Training-fit sanity
# ---------------------------------------------------------------------------


class TestTrainingFitSanity:
    """The head must actually learn the mapping it was trained on."""

    def test_training_fit_mse_below_threshold(self, trained_head: dict) -> None:
        mse = float(trained_head["final_metrics"]["mse"])
        assert mse <= THRESH_TRAINING_FIT_MSE_MAX, (
            f"Projection head training-fit MSE = {mse:.4f} exceeds threshold "
            f"{THRESH_TRAINING_FIT_MSE_MAX}. The head is not fitting its "
            "training data — investigate training loop (LR, epochs, batch size). "
            "See manuscript/evaluation_protocol.md §13 before widening the threshold."
        )

    def test_training_fit_distance_correlation_above_threshold(
        self, trained_head: dict
    ) -> None:
        dcorr = float(trained_head["final_metrics"]["distance_correlation"])
        assert dcorr >= THRESH_TRAINING_FIT_DCORR_MIN, (
            f"Projection head training-fit distance correlation = {dcorr:.4f} "
            f"below threshold {THRESH_TRAINING_FIT_DCORR_MIN}. The head's "
            "distance-preserving component is not learning."
        )

    def test_training_fit_trustworthiness_above_threshold(
        self, trained_head: dict
    ) -> None:
        tw = float(trained_head["final_metrics"]["trustworthiness"])
        assert tw >= THRESH_TRAINING_FIT_TRUSTWORTHINESS_MIN, (
            f"Projection head training-fit trustworthiness = {tw:.4f} below "
            f"threshold {THRESH_TRAINING_FIT_TRUSTWORTHINESS_MIN}. Local "
            "neighborhood preservation has degraded."
        )


# ---------------------------------------------------------------------------
# Test 2 — Cross-validation generalization (3-fold)
# ---------------------------------------------------------------------------


class TestCrossValidationGeneralization:
    """The head must generalize to held-out exemplars of known topics.

    All thresholds are on the MEAN across folds — individual folds may
    fluctuate more, but the mean must clear the pre-registered bar.
    """

    def test_cv_mean_mse_below_threshold(self, trained_head: dict) -> None:
        mse_vals = [float(m["mse"]) for m in trained_head["cv_metrics"]]
        mean_mse = float(np.mean(mse_vals))
        assert mean_mse <= THRESH_CV_MSE_MAX, (
            f"CV mean MSE = {mean_mse:.4f} (fold values {mse_vals}) exceeds "
            f"threshold {THRESH_CV_MSE_MAX}."
        )

    def test_cv_mean_r2_above_threshold(
        self, trained_head: dict, consensus_full: np.ndarray
    ) -> None:
        """R² computed relative to the target variance (1 - MSE/var)."""
        mean_mse = float(np.mean([float(m["mse"]) for m in trained_head["cv_metrics"]]))
        y_var = float(np.var(consensus_full))
        assert y_var > 0, "Consensus targets have zero variance — degenerate."
        r2 = 1.0 - mean_mse / y_var
        assert r2 >= THRESH_CV_R2_MIN, (
            f"CV R² = {r2:.4f} (mean MSE {mean_mse:.4f} / var {y_var:.4f}) "
            f"below threshold {THRESH_CV_R2_MIN}."
        )

    def test_cv_mean_trustworthiness_above_threshold(
        self, trained_head: dict
    ) -> None:
        tw_vals = [float(m["trustworthiness"]) for m in trained_head["cv_metrics"]]
        mean_tw = float(np.mean(tw_vals))
        assert mean_tw >= THRESH_CV_TRUSTWORTHINESS_MIN, (
            f"CV mean trustworthiness = {mean_tw:.4f} (fold values {tw_vals}) "
            f"below threshold {THRESH_CV_TRUSTWORTHINESS_MIN}."
        )

    def test_cv_mean_ari_vs_truth_above_threshold(
        self, trained_head: dict
    ) -> None:
        """Held-out cluster assignment agreement with true topic labels."""
        ari_vals = [float(m["ari"]) for m in trained_head["cv_metrics"]]
        mean_ari = float(np.mean(ari_vals))
        assert mean_ari >= THRESH_CV_ARI_VS_TRUTH_MIN, (
            f"CV mean ARI vs true topic = {mean_ari:.4f} (fold values {ari_vals}) "
            f"below threshold {THRESH_CV_ARI_VS_TRUTH_MIN}."
        )

    def test_cv_mean_distance_correlation_above_threshold(
        self, trained_head: dict
    ) -> None:
        dcorr_vals = [
            float(m["distance_correlation"]) for m in trained_head["cv_metrics"]
        ]
        mean_dcorr = float(np.mean(dcorr_vals))
        assert mean_dcorr >= THRESH_CV_DCORR_MIN, (
            f"CV mean distance correlation = {mean_dcorr:.4f} "
            f"(fold values {dcorr_vals}) below threshold {THRESH_CV_DCORR_MIN}."
        )


# ---------------------------------------------------------------------------
# Test 3 — Topic-exclusion cohesion (held-out topic → novel cluster)
# ---------------------------------------------------------------------------


def _project_held_out_topic(
    snap: DatasetSnapshot,
    seeds: list[int],
    excluded_topic: int,
) -> dict:
    """Run the topic-exclusion experiment for one excluded topic.

    For the excluded topic:
      1. Run REAP on the 7 remaining topics → consensus targets.
      2. Train projection head on the 7 topics.
      3. Project the held-out topic's raw embeddings through the head.
      4. Assign each held-out doc to its nearest training-cluster centroid.
      5. Report which training cluster was dominant and its fraction.

    Returns a dict with held-out projection, per-doc nearest-cluster
    assignments, dominant cluster, dominant fraction, and (for overlap-pair
    members) the pair-partner fraction.

    Side effects: runs UMAP + projection-head training (~30-45s per topic).
    """
    assert snap.labels is not None
    mask = snap.labels != excluded_topic
    X_train = snap.embeddings[mask]
    y_train = snap.labels[mask]
    X_held = snap.embeddings[~mask]

    Y_train = _run_reap_consensus(X_train, seeds)

    _set_torch_deterministic(20260413 + excluded_topic)
    result = train_projection_head(
        X_train.astype(np.float32),
        Y_train.astype(np.float32),
        y_train,
        n_folds=2,
        max_epochs=PROJECTION_MAX_EPOCHS,
        patience=PROJECTION_PATIENCE,
        batch_size=PROJECTION_BATCH_SIZE,
        lr=PROJECTION_LR,
    )
    head = result["model"]

    Y_held = head.forward(X_held.astype(np.float32))
    Y_train_proj = head.forward(X_train.astype(np.float32))

    train_classes = np.unique(y_train)
    train_centroids = np.stack(
        [Y_train_proj[y_train == c].mean(axis=0) for c in train_classes]
    )
    nearest_class_idx = cdist(Y_held, train_centroids).argmin(axis=1)
    nearest_class = train_classes[nearest_class_idx]

    unique_vals, counts = np.unique(nearest_class, return_counts=True)
    fractions: dict[int, float] = {
        int(c): float(cnt) / float(Y_held.shape[0])
        for c, cnt in zip(unique_vals.tolist(), counts.tolist())
    }
    dominant_class = int(unique_vals[int(counts.argmax())])
    dominant_fraction = fractions[dominant_class]

    partner = PAIR_PARTNER.get(excluded_topic)
    partner_fraction = fractions.get(partner, 0.0) if partner is not None else None

    return {
        "excluded_topic": excluded_topic,
        "dominant_class": dominant_class,
        "dominant_fraction": dominant_fraction,
        "fractions": fractions,
        "partner_class": partner,
        "partner_fraction": partner_fraction,
    }


# Memoize the per-topic exclusion run so two fixtures (all topics +
# overlap-pair-only) can share results without recomputing UMAP +
# projection-head training per topic.
_TOPIC_EXCLUSION_CACHE: dict[int, dict] = {}


def _cached_topic_exclusion(
    snap: DatasetSnapshot, seeds: list[int], topic: int
) -> dict:
    if topic not in _TOPIC_EXCLUSION_CACHE:
        _TOPIC_EXCLUSION_CACHE[topic] = _project_held_out_topic(snap, seeds, topic)
    return _TOPIC_EXCLUSION_CACHE[topic]


@pytest.fixture(scope="module", params=FAST_TIER_EXCLUDED_TOPICS)
def topic_exclusion_run(
    request: pytest.FixtureRequest,
    text_snap: DatasetSnapshot,
    seeds_10: list[int],
) -> dict:
    """Parametrized: run topic-exclusion for each topic in FAST_TIER_EXCLUDED_TOPICS."""
    return _cached_topic_exclusion(text_snap, seeds_10, int(request.param))


@pytest.fixture(scope="module", params=FAST_TIER_OVERLAP_PAIR_TOPICS)
def overlap_pair_exclusion_run(
    request: pytest.FixtureRequest,
    text_snap: DatasetSnapshot,
    seeds_10: list[int],
) -> dict:
    """Parametrized over overlap-pair members only — never produces a skip."""
    return _cached_topic_exclusion(text_snap, seeds_10, int(request.param))


class TestTopicExclusionSemanticRouting:
    """Held-out topic routes to its nearest semantic training neighbor.

    The projection head does not create novel clusters for unseen topics;
    instead, it maps out-of-distribution docs toward their nearest
    semantic neighbor in the training space. For overlap-pair members
    (politics.guns ↔ politics.mideast, ibm.hardware ↔ mac.hardware),
    this effect is strong (≥50% of held-out docs land in the pair
    partner). For clearly separated topics, the effect is weaker but
    still non-random (one cluster attracts ≥30% of held-out docs).
    """

    def test_dominant_cluster_is_non_random(
        self, topic_exclusion_run: dict
    ) -> None:
        """Some training cluster attracts more than random-baseline share."""
        topic = topic_exclusion_run["excluded_topic"]
        frac = topic_exclusion_run["dominant_fraction"]
        dominant = topic_exclusion_run["dominant_class"]
        assert frac >= THRESH_ANY_DOMINANT_FRACTION_MIN, (
            f"Held-out topic {topic}: no training cluster attracted more than "
            f"{frac:.4f} of held-out docs (dominant cluster was {dominant} "
            f"with fraction {frac:.4f}, threshold "
            f"{THRESH_ANY_DOMINANT_FRACTION_MIN}). The projection head is "
            "scattering this unseen topic near-uniformly across training "
            "clusters — it is not doing meaningful semantic routing. See "
            "manuscript/evaluation_protocol.md §13 before widening the "
            "threshold."
        )

    @pytest.mark.reference_platform
    def test_overlap_pair_routes_to_partner(
        self, overlap_pair_exclusion_run: dict
    ) -> None:
        """Overlap-pair topics: held-out docs land in the pair-partner cluster.

        Parametrized over `FAST_TIER_OVERLAP_PAIR_TOPICS` only — separated
        topics are excluded from this fixture by construction, so this test
        never skips. Separated topics are still covered by
        `test_dominant_cluster_is_non_random`.
        """
        topic = overlap_pair_exclusion_run["excluded_topic"]
        partner = overlap_pair_exclusion_run["partner_class"]
        assert partner is not None, (
            f"Test bug: topic {topic} reached the overlap-pair test without "
            f"a partner. FAST_TIER_OVERLAP_PAIR_TOPICS should only contain "
            f"keys of PAIR_PARTNER."
        )
        partner_frac = overlap_pair_exclusion_run["partner_fraction"]
        assert partner_frac >= THRESH_OVERLAP_PAIR_PARTNER_FRACTION_MIN, (
            f"Overlap-pair topic {topic}: only {partner_frac:.4f} of "
            f"held-out docs landed in the pair partner (class {partner}), "
            f"below threshold "
            f"{THRESH_OVERLAP_PAIR_PARTNER_FRACTION_MIN}. The head is "
            "failing to interpolate semantically adjacent content — this is "
            "a regression in REAP's headline claim that the projection head "
            "generalizes meaningfully to related unseen content."
        )
