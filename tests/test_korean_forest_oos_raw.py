"""Tests for the raw Korean-forest pledge embeddings loader.

Unit tests run everywhere. The build test needs the Korean-forest source
project (``REAP_KOREAN_FOREST_SOURCE_PATH`` or the sibling checkout) and the
cached pledge snapshot, and skips with a visible reason when either is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reap.datasets import (
    build_korean_forest_oos_raw_snapshot,
    load_korean_forest_oos,
    load_korean_forest_oos_raw,
)
from reap.datasets.korean_forest_oos_raw import _parse_vector

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_korean_forest_source_root() -> Path:
    """``REAP_KOREAN_FOREST_SOURCE_PATH`` if set, else the sibling checkout."""
    explicit = os.environ.get("REAP_KOREAN_FOREST_SOURCE_PATH")
    if explicit:
        return Path(explicit).expanduser()
    return REPO_ROOT.parent / "green-narrative" / "hye_in"


def test_parse_vector_reads_a_printed_list() -> None:
    values = [0.5, -1.25e-3, 2.0] + [0.0] * 381
    cell = "[" + ", ".join(repr(v) for v in values) + "]"
    np.testing.assert_array_equal(_parse_vector(cell, "x"), np.asarray(values))


def test_parse_vector_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 384"):
        _parse_vector("[0.1, 0.2]", "row 0")


def test_load_without_a_built_snapshot_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_korean_forest_oos_raw(cache_dir=tmp_path)


def test_build_from_source_is_row_aligned_and_unit_length(tmp_path: Path) -> None:
    source = _get_korean_forest_source_root()
    if not (source / "data_may7_2026" / "projections").is_dir():
        pytest.skip(f"Korean-forest source project not present at {source}")
    try:
        projected = load_korean_forest_oos()
    except FileNotFoundError:
        pytest.skip("cached korean_forest_oos snapshot not built on this machine")
    # The builder checks row order against the default-cache pledge snapshot,
    # so build into a temporary cache that holds a copy of it.
    import shutil

    from reap.datasets._cache import get_snapshot_dir
    from reap.datasets.korean_forest_oos import DATASET_NAME, DATASET_VERSION

    shutil.copytree(
        get_snapshot_dir(DATASET_NAME, DATASET_VERSION),
        get_snapshot_dir(DATASET_NAME, DATASET_VERSION, cache_dir=tmp_path),
    )
    build_korean_forest_oos_raw_snapshot(source, cache_dir=tmp_path)
    raw = load_korean_forest_oos_raw(cache_dir=tmp_path)
    assert raw.embeddings.shape == (1662, 384)
    np.testing.assert_allclose(np.linalg.norm(raw.embeddings, axis=1), 1.0, atol=1e-4)
    assert raw.texts == projected.texts
