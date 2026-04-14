"""Golden validation for the c-TF-IDF → LLM cluster labeling pipeline.

Exercises the full labeling pipeline on the 20newsgroups golden text
fixture: KMeans cluster labels from the REAP consensus embedding →
c-TF-IDF candidate terms → LLM refinement into human-readable labels.

Providers tested
----------------
* **Anthropic Claude Opus 4.6** (`claude-opus-4-6`) — matches the
  "Claude = yourself" directive for labeling pipelines in the paper.
* **OpenAI gpt-5.4-mini** — the current-generation OpenAI endpoint.

API key handling
----------------
These tests make real API calls. They skip with a clear message when
the relevant key is not set:
  * `ANTHROPIC_API_KEY` required for the Anthropic pipeline.
  * `OPENAI_API_KEY` required for the OpenAI pipeline.

Marked with `@pytest.mark.llm` so CI can opt-in (e.g., a nightly or
manual workflow) rather than paying for every PR run. To invoke:
  pytest tests/test_labeling_golden.py -v -m llm

Validation criteria
-------------------
For each of 8 KMeans clusters from the golden-text REAP consensus:
  * Each LLM label is non-empty and distinct from every other cluster's label.
  * Each label's `confidence` is in [0, 1].
  * The label or short_description contains at least one topic-appropriate
    keyword for the cluster's dominant ground-truth topic.

The last criterion is the substantive test: "did the LLM get it right?".
Keyword dictionaries are intentionally permissive (many plausible terms
per topic) so correct-but-phrased-differently labels still pass.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from reap.clustering import run_kmeans
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.datasets import GOLDEN_20NG_CLASSES, DatasetSnapshot, load_golden_text
from reap.labeling import (
    label_clusters_combined,
    label_clusters_ctfidf,
)

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

ANTHROPIC_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.4-mini"

# Topic-appropriate keyword sets for the 8 ground-truth classes. Kept
# permissive: a correct LLM label phrased differently still passes.
# All comparisons are case-insensitive and substring-based.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "sci.space": [
        "space", "nasa", "astronaut", "rocket", "orbit", "planet",
        "mission", "shuttle", "astronomy", "satellite", "cosmos",
        "spacecraft", "lunar", "mars", "solar",
    ],
    "rec.sport.hockey": [
        "hockey", "nhl", "game", "team", "player", "season", "sport",
        "puck", "ice", "rink", "goal", "playoff", "stanley cup",
    ],
    "comp.graphics": [
        "graphics", "image", "3d", "rendering", "visual", "computer graphic",
        "pixel", "raster", "vector graphic", "vga", "ray trac", "bitmap",
        "jpeg", "gif", "animation",
    ],
    "soc.religion.christian": [
        "christian", "religion", "god", "church", "faith", "bible",
        "jesus", "christ", "worship", "spiritual", "theology", "gospel",
        "salvation", "scripture",
    ],
    "talk.politics.guns": [
        "gun", "firearm", "weapon", "rifle", "pistol", "shoot",
        "second amendment", "2nd amendment", "nra", "militia",
        "concealed carry", "ammunition",
    ],
    "talk.politics.mideast": [
        "mideast", "middle east", "israel", "palestin", "arab", "jew",
        "zion", "turk", "armenian", "muslim", "gaza", "west bank",
        "jerusalem", "lebanon",
    ],
    "comp.sys.ibm.pc.hardware": [
        "ibm", "pc", "hardware", "dos", "486", "386", "motherboard",
        "bios", "isa", "ide", "personal computer", "controller",
        "driver", "pc hardware",
    ],
    "comp.sys.mac.hardware": [
        "mac", "apple", "macintosh", "powerbook", "quadra", "performa",
        "scsi", "appletalk", "system 7", "mac hardware", "mac os",
    ],
}


def _load_set_a_seeds(n: int) -> list[int]:
    return [int(s) for s in json.loads(SEED_MANIFEST_PATH.read_text())["sets"]["A"]["seeds"][:n]]


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _dominant_topic_per_kmeans_cluster(
    kmeans_labels: np.ndarray, ground_truth: np.ndarray
) -> dict[int, int]:
    """For each KMeans cluster, return the dominant ground-truth topic ID.

    KMeans returns arbitrary cluster IDs; this maps each back to the
    20ng class that most of its members belong to.
    """
    mapping: dict[int, int] = {}
    for c in np.unique(kmeans_labels):
        mask = kmeans_labels == c
        topic_ids = ground_truth[mask]
        counter = Counter(topic_ids.tolist())
        dominant, _ = counter.most_common(1)[0]
        mapping[int(c)] = int(dominant)
    return mapping


def _label_contains_any_keyword(label_text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match: label_text contains any of the keywords."""
    haystack = label_text.lower()
    return any(kw.lower() in haystack for kw in keywords)


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped so expensive steps run once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def text_snap() -> DatasetSnapshot:
    """Load the 20ng golden text snapshot once for the module."""
    pytest.importorskip("sentence_transformers")
    return load_golden_text()


@pytest.fixture(scope="module")
def kmeans_labels_and_dominant(text_snap: DatasetSnapshot) -> dict:
    """Run REAP on full 20ng golden, KMeans at K=8, return labels + dominant-topic mapping."""
    seeds = _load_set_a_seeds(GOLDEN_N_SEEDS)
    embs = get_multi_seed_embeddings(
        text_snap.embeddings,
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
    labels, _ = run_kmeans(Y, k=GOLDEN_K)
    assert text_snap.labels is not None
    dominant = _dominant_topic_per_kmeans_cluster(labels, text_snap.labels)
    return {"labels": labels, "dominant_topic": dominant}


# ---------------------------------------------------------------------------
# Stage 1 — c-TF-IDF on real text (no API dependency; always runs)
# ---------------------------------------------------------------------------


class TestCtfidfOnGoldenText:
    """c-TF-IDF statistical term extraction on the 20ng golden clusters.

    No API keys needed; this is the foundation on which LLM labeling builds.
    """

    @pytest.fixture(scope="class")
    def ctfidf_results(
        self,
        text_snap: DatasetSnapshot,
        kmeans_labels_and_dominant: dict,
    ) -> list:
        """Run c-TF-IDF once; reused across the tests in this class."""
        assert text_snap.texts is not None
        return label_clusters_ctfidf(
            text_snap.texts,
            kmeans_labels_and_dominant["labels"],
            top_n=10,
            ngram_range=(1, 3),
            min_df=2,
        )

    def test_one_result_per_cluster(self, ctfidf_results: list) -> None:
        assert len(ctfidf_results) == GOLDEN_K, (
            f"Expected {GOLDEN_K} cluster results, got {len(ctfidf_results)}"
        )

    def test_each_cluster_has_top_terms(self, ctfidf_results: list) -> None:
        for r in ctfidf_results:
            assert len(r.top_terms) > 0, (
                f"Cluster {r.cluster} has zero top terms — c-TF-IDF produced "
                "nothing usable for LLM labeling"
            )

    def test_each_cluster_has_discriminative_terms(
        self, ctfidf_results: list
    ) -> None:
        for r in ctfidf_results:
            assert len(r.discriminative_terms) > 0, (
                f"Cluster {r.cluster} has zero discriminative terms — "
                "clusters are not distinguishable via c-TF-IDF"
            )

    def test_top_term_distinctness_across_clusters(
        self, ctfidf_results: list
    ) -> None:
        """Different clusters should have different #1 top terms.

        Some overlap is acceptable (common corpus vocabulary) but if ≥3
        clusters share the same top term, c-TF-IDF is not discriminating.
        """
        top_terms = [r.top_terms[0].term for r in ctfidf_results if r.top_terms]
        counter = Counter(top_terms)
        most_common_count = counter.most_common(1)[0][1] if counter else 0
        assert most_common_count < 3, (
            f"Top c-TF-IDF terms are too similar across clusters: "
            f"{counter}. The pipeline is not discriminating between clusters."
        )

    def test_top_terms_match_dominant_topic_for_clear_clusters(
        self,
        ctfidf_results: list,
        kmeans_labels_and_dominant: dict,
    ) -> None:
        """At least one top c-TF-IDF term per CLEARLY-SEPARATED cluster
        should match the cluster's dominant-topic keyword set.

        Overlap-pair clusters (politics, hardware) are excluded because
        c-TF-IDF for those depends on which subtle distinguishing terms
        survived the 2000-char truncation; the LLM stage smooths this out.
        """
        dominant_map = kmeans_labels_and_dominant["dominant_topic"]
        clear_classes = {0, 1, 2, 3}  # sci.space, hockey, graphics, christian
        failures: list[str] = []
        for r in ctfidf_results:
            dom_id = dominant_map[r.cluster]
            if dom_id not in clear_classes:
                continue
            topic_name = GOLDEN_20NG_CLASSES[dom_id]
            keywords = TOPIC_KEYWORDS[topic_name]
            top_term_texts = " ".join(t.term for t in r.top_terms[:10])
            if not _label_contains_any_keyword(top_term_texts, keywords):
                failures.append(
                    f"cluster {r.cluster} (dominant topic {topic_name}): "
                    f"top terms {[t.term for t in r.top_terms[:5]]} contain "
                    f"none of {keywords[:5]}"
                )
        assert not failures, (
            "c-TF-IDF top terms did not match dominant-topic keywords for "
            f"clearly-separated clusters: {failures}"
        )


# ---------------------------------------------------------------------------
# Stage 2 — LLM refinement via Anthropic Claude Opus 4.6
# ---------------------------------------------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(
    not _has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set — set it to run the Claude labeling tests",
)
class TestAnthropicLabeling:
    """End-to-end c-TF-IDF → Claude Opus 4.6 on the 20ng golden fixture."""

    @pytest.fixture(scope="class")
    def anthropic_results(
        self,
        text_snap: DatasetSnapshot,
        kmeans_labels_and_dominant: dict,
    ) -> dict:
        """Run the combined pipeline once with Claude Opus 4.6; reuse for tests."""
        assert text_snap.texts is not None
        ctfidf_results, llm_labels = label_clusters_combined(
            text_snap.texts,
            kmeans_labels_and_dominant["labels"],
            provider="anthropic",
            model=ANTHROPIC_MODEL,
            n_sample_texts=5,
        )
        return {
            "ctfidf_results": ctfidf_results,
            "llm_labels": llm_labels,
            "dominant_topic": kmeans_labels_and_dominant["dominant_topic"],
        }

    def test_one_label_per_cluster(self, anthropic_results: dict) -> None:
        assert len(anthropic_results["llm_labels"]) == GOLDEN_K

    def test_labels_are_non_empty(self, anthropic_results: dict) -> None:
        for lbl in anthropic_results["llm_labels"]:
            assert lbl.label.strip(), (
                f"Cluster {lbl.cluster}: Anthropic returned an empty label"
            )

    def test_labels_are_distinct(self, anthropic_results: dict) -> None:
        label_texts = [lbl.label.strip().lower() for lbl in anthropic_results["llm_labels"]]
        assert len(set(label_texts)) == len(label_texts), (
            f"Anthropic produced duplicate cluster labels: {label_texts}"
        )

    def test_confidence_in_valid_range(self, anthropic_results: dict) -> None:
        for lbl in anthropic_results["llm_labels"]:
            assert 0.0 <= lbl.confidence <= 1.0, (
                f"Cluster {lbl.cluster}: confidence {lbl.confidence} "
                "outside [0, 1]"
            )

    def test_label_matches_dominant_topic(self, anthropic_results: dict) -> None:
        """Each label or short_description contains ≥1 topic-appropriate keyword."""
        dominant_map = anthropic_results["dominant_topic"]
        failures: list[str] = []
        for lbl in anthropic_results["llm_labels"]:
            dom_id = dominant_map[lbl.cluster]
            topic_name = GOLDEN_20NG_CLASSES[dom_id]
            keywords = TOPIC_KEYWORDS[topic_name]
            haystack = f"{lbl.label} {lbl.short_description} {lbl.evidence_summary}"
            if not _label_contains_any_keyword(haystack, keywords):
                failures.append(
                    f"cluster {lbl.cluster} (dominant={topic_name}): "
                    f"label={lbl.label!r} description={lbl.short_description!r} "
                    f"contains none of {keywords[:6]}"
                )
        assert not failures, (
            "Anthropic labels did not match dominant-topic keywords: "
            f"{failures}"
        )


# ---------------------------------------------------------------------------
# Stage 2 — LLM refinement via OpenAI gpt-5.4-mini
# ---------------------------------------------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(
    not _has_openai_key(),
    reason="OPENAI_API_KEY not set — set it to run the OpenAI labeling tests",
)
class TestOpenAILabeling:
    """End-to-end c-TF-IDF → OpenAI gpt-5.4-mini on the 20ng golden fixture."""

    @pytest.fixture(scope="class")
    def openai_results(
        self,
        text_snap: DatasetSnapshot,
        kmeans_labels_and_dominant: dict,
    ) -> dict:
        """Run the combined pipeline once with gpt-5.4-mini; reuse for tests."""
        assert text_snap.texts is not None
        ctfidf_results, llm_labels = label_clusters_combined(
            text_snap.texts,
            kmeans_labels_and_dominant["labels"],
            provider="openai",
            model=OPENAI_MODEL,
            n_sample_texts=5,
        )
        return {
            "ctfidf_results": ctfidf_results,
            "llm_labels": llm_labels,
            "dominant_topic": kmeans_labels_and_dominant["dominant_topic"],
        }

    def test_one_label_per_cluster(self, openai_results: dict) -> None:
        assert len(openai_results["llm_labels"]) == GOLDEN_K

    def test_labels_are_non_empty(self, openai_results: dict) -> None:
        for lbl in openai_results["llm_labels"]:
            assert lbl.label.strip(), (
                f"Cluster {lbl.cluster}: OpenAI returned an empty label"
            )

    def test_labels_are_distinct(self, openai_results: dict) -> None:
        label_texts = [lbl.label.strip().lower() for lbl in openai_results["llm_labels"]]
        assert len(set(label_texts)) == len(label_texts), (
            f"OpenAI produced duplicate cluster labels: {label_texts}"
        )

    def test_confidence_in_valid_range(self, openai_results: dict) -> None:
        for lbl in openai_results["llm_labels"]:
            assert 0.0 <= lbl.confidence <= 1.0, (
                f"Cluster {lbl.cluster}: confidence {lbl.confidence} "
                "outside [0, 1]"
            )

    def test_label_matches_dominant_topic(self, openai_results: dict) -> None:
        """Each label or short_description contains ≥1 topic-appropriate keyword."""
        dominant_map = openai_results["dominant_topic"]
        failures: list[str] = []
        for lbl in openai_results["llm_labels"]:
            dom_id = dominant_map[lbl.cluster]
            topic_name = GOLDEN_20NG_CLASSES[dom_id]
            keywords = TOPIC_KEYWORDS[topic_name]
            haystack = f"{lbl.label} {lbl.short_description} {lbl.evidence_summary}"
            if not _label_contains_any_keyword(haystack, keywords):
                failures.append(
                    f"cluster {lbl.cluster} (dominant={topic_name}): "
                    f"label={lbl.label!r} description={lbl.short_description!r} "
                    f"contains none of {keywords[:6]}"
                )
        assert not failures, (
            f"OpenAI labels did not match dominant-topic keywords: {failures}"
        )


# ---------------------------------------------------------------------------
# Cross-provider consistency
# ---------------------------------------------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(
    not (_has_anthropic_key() and _has_openai_key()),
    reason="Both ANTHROPIC_API_KEY and OPENAI_API_KEY required for cross-provider test",
)
class TestCrossProviderConsistency:
    """Anthropic and OpenAI must agree on the dominant topic per cluster.

    They may phrase labels differently ("Space Exploration" vs "NASA
    Missions"), but both labels should match the same topic-keyword set.
    """

    def test_both_providers_agree_on_dominant_topic(
        self,
        text_snap: DatasetSnapshot,
        kmeans_labels_and_dominant: dict,
    ) -> None:
        assert text_snap.texts is not None
        _, anthropic_labels = label_clusters_combined(
            text_snap.texts,
            kmeans_labels_and_dominant["labels"],
            provider="anthropic",
            model=ANTHROPIC_MODEL,
            n_sample_texts=5,
        )
        _, openai_labels = label_clusters_combined(
            text_snap.texts,
            kmeans_labels_and_dominant["labels"],
            provider="openai",
            model=OPENAI_MODEL,
            n_sample_texts=5,
        )
        dominant_map = kmeans_labels_and_dominant["dominant_topic"]

        disagreements: list[str] = []
        for a, o in zip(
            sorted(anthropic_labels, key=lambda x: x.cluster),
            sorted(openai_labels, key=lambda x: x.cluster),
        ):
            assert a.cluster == o.cluster
            dom_id = dominant_map[a.cluster]
            topic_name = GOLDEN_20NG_CLASSES[dom_id]
            keywords = TOPIC_KEYWORDS[topic_name]
            a_haystack = f"{a.label} {a.short_description}"
            o_haystack = f"{o.label} {o.short_description}"
            a_match = _label_contains_any_keyword(a_haystack, keywords)
            o_match = _label_contains_any_keyword(o_haystack, keywords)
            if a_match != o_match:
                disagreements.append(
                    f"cluster {a.cluster} ({topic_name}): "
                    f"anthropic_matches={a_match} (label={a.label!r}); "
                    f"openai_matches={o_match} (label={o.label!r})"
                )
        assert not disagreements, (
            f"Providers disagree on dominant-topic match: {disagreements}"
        )
