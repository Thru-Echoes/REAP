"""Pytest configuration for the verification suite.

Several verification modules validate *research-output artifacts* — model
state dicts, consensus embeddings/labels, and published metric CSVs under
``results/``. Per the repository policy (see the root ``.gitignore``), that
material is gitignored here and is being separated into the sibling
``REAP-research`` repo, so it is absent in the code-only package CI.

The artifact-dependent modules reference those outputs through a hardcoded
absolute results root (the author's research-machine path). That path is
absent off that machine — notably in the code-only package CI, and also note
that the binary artifacts themselves, e.g. ``head_state_dict.pt`` and the
``*.npy``/``*.csv`` outputs, are not tracked in git. So when that results root
is not present, the artifact-dependent modules listed below are skipped — the
package's own code tests and the self-contained synthetic ladder rungs
(rung0/1/2) still run. On the research machine (or the sibling REAP-research
repo) the artifacts are present and these tests run normally. Override the
results root with the ``REAP_RESULTS_ROOT`` environment variable.

This mirrors the hardcoded path the artifact modules use; it is transitional
until that research material moves to the sibling repo and these modules move
with it (see the root ``.gitignore``).

Side effects: reads the ``REAP_RESULTS_ROOT`` env var and the filesystem to
decide whether the artifact root exists.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Must match the hardcoded ``REPO_ROOT / "results"`` used by the artifact
# modules; absent off the research machine (e.g. in CI), which is the skip signal.
_DEFAULT_RESULTS_ROOT = "/Users/echoes/Documents/Berkeley/Research/REAP/results"
_RESULTS_ROOT = Path(os.environ.get("REAP_RESULTS_ROOT", _DEFAULT_RESULTS_ROOT))

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
    """Skip artifact-dependent verification modules when ``results/`` is absent.

    Side effect: adds a skip marker, in place, to each collected item that
    belongs to an artifact-dependent module when the artifact root is missing.
    """
    if _RESULTS_ROOT.exists():
        return
    skip_marker = pytest.mark.skip(
        reason=(
            f"research-output artifacts not present at {_RESULTS_ROOT} — they live in "
            "the sibling REAP-research repo and are absent in code-only CI"
        )
    )
    for item in items:
        item_path = getattr(item, "path", None)
        stem = item_path.stem if item_path is not None else None
        if stem in _ARTIFACT_DEPENDENT_MODULES:
            item.add_marker(skip_marker)
