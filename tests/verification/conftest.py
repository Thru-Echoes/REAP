"""Pytest configuration for the verification suite.

Several verification modules validate *research-output artifacts* — model
state dicts, consensus embeddings/labels, and published metric CSVs under
``results/``. Most of that material is not tracked in this package
repository: the committed copies live in the sibling ``REAP-research``
repository (see ``docs/reap_research_split_plan_2026-07.md``). The modules
find each artifact through ``tests/_research_paths.get_artifact_path``, which
searches REAP-research first and this repository's ``results/`` second, with
``REAP_RESULTS_ROOT`` as a single explicit override.

Skip rule: the artifact-dependent modules listed below run whenever this
machine has research artifacts (``REAP_RESULTS_ROOT`` is set, or a
REAP-research checkout resolves through ``REAP_RESEARCH_ROOT`` or the
``../REAP-research`` sibling). There, a missing artifact is a test failure,
not a skip. On a code-only checkout — notably package CI — they are skipped
with a visible reason; the package's own code tests and the self-contained
synthetic ladder rungs (rung0/1/2) still run.

Side effects: reads environment variables and the filesystem to decide
whether research artifacts are available.
"""

from __future__ import annotations

import pytest
from _research_paths import get_results_roots, has_research_artifacts

# Verification modules whose tests load artifacts under results/. Kept explicit
# (not pattern-matched) so a new self-contained verification module is never
# skipped by accident.
_ARTIFACT_DEPENDENT_MODULES = frozenset(
    {
        "test_ai_art_oos_demo_smoke",
        "test_ai_art_cross_source_smoke",
        "test_ari_rung3_twentynewsgroups",
        "test_ari_rung4_real_corpora",
        "test_projection_head_oos_provenance",
        "test_silhouette_rung3_twentynewsgroups",
        "test_silhouette_rung4_real_corpora",
        "test_trustworthiness_rung3_real_corpora",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip artifact-dependent verification modules on a code-only checkout.

    Side effect: adds a skip marker, in place, to each collected item that
    belongs to an artifact-dependent module when no research artifacts are available.
    """
    if has_research_artifacts():
        return
    skip_marker = pytest.mark.skip(
        reason=(
            "research-output artifacts not available: no REAP-research checkout "
            "(set REAP_RESEARCH_ROOT or check out ../REAP-research) and no "
            f"REAP_RESULTS_ROOT; searched {[str(root) for root in get_results_roots()]}"
        )
    )
    for item in items:
        item_path = getattr(item, "path", None)
        stem = item_path.stem if item_path is not None else None
        if stem in _ARTIFACT_DEPENDENT_MODULES:
            item.add_marker(skip_marker)
