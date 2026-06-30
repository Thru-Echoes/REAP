"""Unit tests for the 20-Newsgroups manuscript validation loader pair.

Covers:

* `load_twenty_newsgroups_reference` — 800 docs, 8 classes (the same
  curated 8-class subset as `load_golden_text`, scaled up to 100
  docs/class).
* `load_twenty_newsgroups_oos` — 600 docs, 12 held-out classes with
  labels offset by `LABEL_OFFSET=100` to make the two label spaces
  unambiguously disjoint.

All tests use ``pytest.importorskip("sentence_transformers")`` so they
skip cleanly on environments without the optional ``text-fixtures``
extra installed. First run downloads MiniLM (~90 MB) and 20NG
(~14 MB); subsequent runs read from
``~/.cache/reap/datasets/20ng_{reference,oos}_*.npz``.
"""

from __future__ import annotations

import numpy as np
import pytest

from reap.datasets._schema import DatasetSnapshot

pytest.importorskip(
    "sentence_transformers",
    reason=(
        "sentence-transformers is required for the manuscript-validation 20NG "
        "loaders. Install with: pip install 'reap-topics[text-fixtures]'"
    ),
)

from reap.datasets import (  # noqa: E402  — must come after importorskip
    OOS_20NG_CLASSES,
    REFERENCE_20NG_CLASSES,
    load_twenty_newsgroups_oos,
    load_twenty_newsgroups_reference,
)
from reap.datasets.twenty_newsgroups_oos import (  # noqa: E402
    LABEL_OFFSET,
)


@pytest.fixture(scope="module")
def reference_snap() -> DatasetSnapshot:
    """Module-scoped reference snapshot; embedding is cached on first call."""
    return load_twenty_newsgroups_reference()


@pytest.fixture(scope="module")
def oos_snap() -> DatasetSnapshot:
    """Module-scoped OOS snapshot; embedding is cached on first call."""
    return load_twenty_newsgroups_oos()


def test_reference_returns_800_docs_8_classes(reference_snap: DatasetSnapshot) -> None:
    """Reference snapshot has 800 docs, 384-d, 8 classes labeled 0..7."""
    assert reference_snap.embeddings.shape == (800, 384)
    assert np.isfinite(reference_snap.embeddings).all()
    assert reference_snap.texts is not None
    assert len(reference_snap.texts) == 800

    assert reference_snap.labels is not None
    assert reference_snap.labels.shape == (800,)
    assert reference_snap.labels.dtype == np.int64
    unique_labels = set(reference_snap.labels.tolist())
    assert unique_labels == set(range(len(REFERENCE_20NG_CLASSES)))
    assert len(unique_labels) == 8

    # 100 docs per class — exact balance is part of the pre-registered design.
    counts = np.bincount(reference_snap.labels)
    np.testing.assert_array_equal(counts, np.full(8, 100, dtype=counts.dtype))

    assert reference_snap.metadata.name == "twenty_newsgroups_reference"
    assert reference_snap.metadata.n_samples == 800
    assert reference_snap.metadata.embedding_dim == 384


def test_oos_returns_600_docs_12_classes(
    reference_snap: DatasetSnapshot, oos_snap: DatasetSnapshot
) -> None:
    """OOS snapshot has 600 docs, 384-d, 12 classes labeled 100..111, disjoint from reference."""
    assert oos_snap.embeddings.shape == (600, 384)
    assert np.isfinite(oos_snap.embeddings).all()
    assert oos_snap.texts is not None
    assert len(oos_snap.texts) == 600

    assert oos_snap.labels is not None
    assert oos_snap.labels.shape == (600,)
    assert oos_snap.labels.dtype == np.int64

    expected_oos_labels = set(
        range(LABEL_OFFSET, LABEL_OFFSET + len(OOS_20NG_CLASSES))
    )
    unique_oos = set(oos_snap.labels.tolist())
    assert unique_oos == expected_oos_labels
    assert len(unique_oos) == 12

    counts = np.bincount(
        oos_snap.labels - LABEL_OFFSET, minlength=len(OOS_20NG_CLASSES)
    )
    np.testing.assert_array_equal(counts, np.full(12, 50, dtype=counts.dtype))

    assert reference_snap.labels is not None
    ref_labels = set(reference_snap.labels.tolist())
    assert ref_labels.isdisjoint(unique_oos), (
        f"reference labels {sorted(ref_labels)} overlap OOS labels "
        f"{sorted(unique_oos)}"
    )

    assert oos_snap.metadata.name == "twenty_newsgroups_oos"
    assert oos_snap.metadata.n_samples == 600
    assert oos_snap.metadata.embedding_dim == 384


def test_reference_and_oos_use_same_embedding_model(
    reference_snap: DatasetSnapshot, oos_snap: DatasetSnapshot
) -> None:
    """Both snapshots embed with the same MiniLM model + preprocessing."""
    assert (
        reference_snap.metadata.embedding_model
        == oos_snap.metadata.embedding_model
    )
    assert "MiniLM-L6-v2" in reference_snap.metadata.embedding_model
    assert (
        reference_snap.metadata.preprocessing_version
        == oos_snap.metadata.preprocessing_version
    )
    assert reference_snap.metadata.embedding_dim == oos_snap.metadata.embedding_dim


def test_oos_classes_do_not_overlap_reference_classes() -> None:
    """The reference and OOS class-name lists are disjoint and cover 20 classes."""
    ref = set(REFERENCE_20NG_CLASSES)
    oos = set(OOS_20NG_CLASSES)
    assert ref.isdisjoint(oos), (
        f"reference and OOS class lists overlap on {sorted(ref & oos)}"
    )
    assert len(ref) == 8
    assert len(oos) == 12
    assert len(ref | oos) == 20
