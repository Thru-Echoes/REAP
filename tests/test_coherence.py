"""Unit tests for the high-level coherence module (src/reap/coherence.py).

Verifies that all three pre-registered coherence metrics (UMass, NPMI,
c_v) produce finite, in-range values on a tiny synthetic corpus, and
that the convenience entry point :func:`reap.coherence.compute_all_coherence`
returns a well-formed :class:`CoherenceResult`.

Skipped cleanly when gensim is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gensim")


# ---------------------------------------------------------------------------
# Synthetic corpus with two clearly distinct topics
# ---------------------------------------------------------------------------

_FOREST_DOCS = [
    "forest tree planting management woodland canopy timber",
    "forest tree biodiversity conservation ecosystem woodland",
    "forest tree management sustainable timber canopy",
    "tree planting reforestation forest restoration woodland",
    "forest biodiversity ecosystem canopy tree planting",
    "forest management woodland timber tree restoration",
]

_CLIMATE_DOCS = [
    "carbon emission reduction climate policy renewable energy",
    "green energy climate carbon sequestration emission",
    "climate policy carbon tax emission trading renewable",
    "renewable energy solar wind green carbon transition",
    "climate carbon emission reduction policy renewable",
    "energy carbon climate policy emission green",
]

_TEXTS: list[str] = _FOREST_DOCS + _CLIMATE_DOCS
_LABELS: np.ndarray = np.array([0] * len(_FOREST_DOCS) + [1] * len(_CLIMATE_DOCS))


def test_compute_coherence_each_metric_is_finite_and_in_range() -> None:
    """All three metrics must produce finite per-cluster scores in their declared range."""
    from reap.coherence import ALL_COHERENCE_METRICS, compute_coherence

    # Smaller top_n keeps gensim happy with the tiny corpus.
    top_n = 5

    for metric in ALL_COHERENCE_METRICS:
        scores = compute_coherence(_LABELS, _TEXTS, metric=metric, top_n=top_n)
        assert len(scores) == 2, (
            f"Expected 2 per-cluster scores for metric={metric!r}, got {len(scores)}"
        )
        for cid, s in enumerate(scores):
            assert np.isfinite(s), (
                f"{metric}: cluster {cid} produced non-finite score {s}"
            )

        # Per-metric range checks (manuscript §7).
        if metric == "u_mass":
            # UMass is bounded above by 0 by construction (log of a probability ratio
            # capped by smoothing); negative or zero is the expected regime.
            for s in scores:
                assert s <= 1e-6, f"UMass score {s} unexpectedly > 0"
        elif metric == "c_npmi":
            for s in scores:
                assert -1.0 - 1e-9 <= s <= 1.0 + 1e-9, f"NPMI out of [-1, 1]: {s}"
            # Well-separated synthetic topics should produce positive NPMI.
            assert all(s > 0.0 for s in scores), (
                f"Expected positive NPMI on well-separated synthetic, got {scores}"
            )
        elif metric == "c_v":
            for s in scores:
                # c_v is bounded in [0, 1] by Röder et al. 2015 but gensim can return
                # very small negatives due to floating point; allow a tiny slack.
                assert -1e-6 <= s <= 1.0 + 1e-6, f"c_v out of [0, 1]: {s}"


def test_compute_all_coherence_returns_well_formed_result() -> None:
    """The convenience entry point must populate every field consistently."""
    from reap.coherence import CoherenceResult, compute_all_coherence

    result = compute_all_coherence(_LABELS, _TEXTS, top_n=5)

    assert isinstance(result, CoherenceResult)

    # Two non-noise clusters → length-2 per-cluster lists.
    assert result.cluster_ids == [0, 1]
    assert len(result.umass_per_cluster) == 2
    assert len(result.npmi_per_cluster) == 2
    assert len(result.cv_per_cluster) == 2

    # Aggregate means must match nanmean of the per-cluster vectors.
    np.testing.assert_allclose(
        result.umass_mean, np.nanmean(result.umass_per_cluster), atol=1e-12
    )
    np.testing.assert_allclose(
        result.npmi_mean, np.nanmean(result.npmi_per_cluster), atol=1e-12
    )
    np.testing.assert_allclose(
        result.cv_mean, np.nanmean(result.cv_per_cluster), atol=1e-12
    )

    # Finite-count accounting must match per-cluster arrays.
    assert result.n_clusters_scored["u_mass"] == int(
        np.sum(np.isfinite(result.umass_per_cluster))
    )
    assert result.n_clusters_scored["c_npmi"] == int(
        np.sum(np.isfinite(result.npmi_per_cluster))
    )
    assert result.n_clusters_scored["c_v"] == int(
        np.sum(np.isfinite(result.cv_per_cluster))
    )

    # On this well-separated synthetic, NPMI mean should be clearly positive.
    assert result.npmi_mean > 0.0, (
        f"Expected positive aggregate NPMI on well-separated synthetic, got {result.npmi_mean}"
    )

    # Top-n is round-tripped exactly.
    assert result.top_n == 5
