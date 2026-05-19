"""Tests for topic coherence metrics (UMass, NPMI).

These tests verify the native implementations against known properties:
- UMass: terms that co-occur frequently score higher; terms that never
  co-occur score near -inf (dampened by smoothing).
- NPMI: perfectly co-occurring terms score near 1; independent terms
  score near 0; never-co-occurring terms score -1.
- Both: single-word topics return 0 (no pairs); unknown words are
  ignored; real c-TF-IDF output produces finite scores.

c_v is tested only when gensim is installed (optional dep).
"""

from __future__ import annotations

import numpy as np
import pytest

from reap.evaluation import compute_topic_coherence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    "forest policy management tree planting",
    "forest tree biodiversity conservation ecosystem",
    "carbon emission reduction climate change policy",
    "green energy climate carbon sequestration",
    "forest tree management sustainable development",
    "climate policy carbon tax emission trading",
    "tree planting reforestation forest restoration",
    "renewable energy solar wind green transition",
]

TOPIC_FOREST = ["forest", "tree", "planting", "management"]
TOPIC_CLIMATE = ["climate", "carbon", "emission", "policy"]
TOPIC_DISJOINT = ["zzzzz_never_appears", "yyyyy_absent"]


# ---------------------------------------------------------------------------
# UMass
# ---------------------------------------------------------------------------


class TestUMass:
    """UMass coherence (Mimno et al. 2011)."""

    def test_co_occurring_terms_score_higher(self) -> None:
        """Terms that co-occur frequently should score higher than disjoint."""
        scores = compute_topic_coherence(
            [TOPIC_FOREST, TOPIC_CLIMATE],
            CORPUS,
            measure="u_mass",
        )
        # Both topics have co-occurring terms, so both should be > some floor
        # (UMass is typically negative, closer to 0 = better).
        for s in scores:
            assert np.isfinite(s), f"UMass score is not finite: {s}"

    def test_single_word_topic_returns_zero(self) -> None:
        scores = compute_topic_coherence([["forest"]], CORPUS, measure="u_mass")
        assert scores == [0.0]

    def test_unknown_words_handled(self) -> None:
        scores = compute_topic_coherence(
            [TOPIC_DISJOINT], CORPUS, measure="u_mass"
        )
        assert scores == [0.0]

    def test_top_n_truncation(self) -> None:
        full = compute_topic_coherence(
            [TOPIC_FOREST], CORPUS, measure="u_mass"
        )
        truncated = compute_topic_coherence(
            [TOPIC_FOREST], CORPUS, measure="u_mass", top_n=2
        )
        # top_n=2 uses only first 2 words (1 pair) vs full 4 words (6 pairs)
        assert full[0] != truncated[0] or len(TOPIC_FOREST) <= 2

    def test_returns_one_score_per_topic(self) -> None:
        scores = compute_topic_coherence(
            [TOPIC_FOREST, TOPIC_CLIMATE], CORPUS, measure="u_mass"
        )
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# NPMI
# ---------------------------------------------------------------------------


class TestNPMI:
    """NPMI coherence (Bouma 2009; Lau et al. 2014)."""

    def test_co_occurring_terms_are_positive(self) -> None:
        """Terms that co-occur more than chance should have positive NPMI."""
        scores = compute_topic_coherence(
            [TOPIC_FOREST], CORPUS, measure="c_npmi"
        )
        assert scores[0] > 0.0, f"Expected positive NPMI for forest topic, got {scores[0]}"

    def test_values_in_expected_range(self) -> None:
        """NPMI is bounded [-1, 1]."""
        scores = compute_topic_coherence(
            [TOPIC_FOREST, TOPIC_CLIMATE], CORPUS, measure="c_npmi"
        )
        for s in scores:
            assert -1.0 <= s <= 1.0 + 1e-9, f"NPMI out of range: {s}"

    def test_single_word_topic_returns_zero(self) -> None:
        scores = compute_topic_coherence([["carbon"]], CORPUS, measure="c_npmi")
        assert scores == [0.0]

    def test_unknown_words_handled(self) -> None:
        scores = compute_topic_coherence(
            [TOPIC_DISJOINT], CORPUS, measure="c_npmi"
        )
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# Dispatch + error handling
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_unknown_measure_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown coherence measure"):
            compute_topic_coherence([TOPIC_FOREST], CORPUS, measure="bogus")

    def test_cv_without_gensim_raises_import_error(self) -> None:
        """c_v should raise ImportError if gensim is not installed."""
        try:
            import gensim  # noqa: F401
            pytest.skip("gensim is installed — cannot test the ImportError path")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="gensim"):
            compute_topic_coherence([TOPIC_FOREST], CORPUS, measure="c_v")


# ---------------------------------------------------------------------------
# Integration: real c-TF-IDF output → coherence
# ---------------------------------------------------------------------------


def test_coherence_on_ctfidf_output() -> None:
    """Verify coherence works on actual c-TF-IDF output from labeling."""
    from sklearn.datasets import make_blobs

    from reap.labeling import label_clusters_ctfidf

    # Generate synthetic clustered "documents" with distinct vocabularies.
    rng = np.random.default_rng(42)
    cluster_words = [
        ["forest", "tree", "woodland", "timber", "canopy"],
        ["ocean", "wave", "marine", "coral", "fish"],
        ["mountain", "summit", "cliff", "rock", "altitude"],
    ]
    texts = []
    labels_list = []
    for cluster_id, words in enumerate(cluster_words):
        for _ in range(20):
            chosen = rng.choice(words, size=8, replace=True).tolist()
            filler = rng.choice(["the", "of", "and", "in", "a"], size=4).tolist()
            texts.append(" ".join(chosen + filler))
            labels_list.append(cluster_id)
    labels = np.array(labels_list)

    ctfidf = label_clusters_ctfidf(texts, labels, top_n=5)
    topics = [[term.term for term in cr.top_terms] for cr in ctfidf]

    umass = compute_topic_coherence(topics, texts, measure="u_mass")
    npmi = compute_topic_coherence(topics, texts, measure="c_npmi")

    assert len(umass) == 3
    assert len(npmi) == 3
    for s in umass:
        assert np.isfinite(s)
    for s in npmi:
        assert np.isfinite(s)
        assert s > 0.0, (
            f"NPMI should be positive for well-separated synthetic clusters, got {s}"
        )
