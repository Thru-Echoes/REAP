"""REAP dataset API.

Every loader returns a `DatasetSnapshot` — a Pydantic-validated bundle
of embeddings, optional texts, optional expert labels, and full
provenance metadata (SHA256, license, citation, embedding model +
version, preprocessing version).

The contract is documented in `.claude/rules/open-source-package.md`
(Dataset API Contract section) and pre-registered via the pipeline
stage table in `manuscript/evaluation_protocol.md` §4.

Public surface
--------------
- `DatasetSnapshot`, `DatasetMetadata` — the data contract.
- `load_synthetic_blobs`, `load_golden_blobs` — deterministic synthetic
  loaders usable anywhere a real snapshot is expected.
- `load_ai_art`, `load_korean_forest`, `load_corp_sustainability`,
  `load_us_presidential` — stubs for the four paper datasets. Raise
  `NotImplementedError` with a clear migration hint until the real
  snapshot manifests land.

No side effects at import time.
"""

from __future__ import annotations

from reap.datasets._schema import DatasetMetadata, DatasetSnapshot
from reap.datasets.synthetic import load_golden_blobs, load_synthetic_blobs
from reap.datasets.twenty_newsgroups import (
    GOLDEN_20NG_CLASSES,
    OVERLAP_PAIRS,
    load_golden_text,
)

__all__ = [
    "DatasetSnapshot",
    "DatasetMetadata",
    "load_synthetic_blobs",
    "load_golden_blobs",
    "load_golden_text",
    "GOLDEN_20NG_CLASSES",
    "OVERLAP_PAIRS",
    "load_ai_art",
    "load_korean_forest",
    "load_corp_sustainability",
    "load_us_presidential",
]


_STUB_MESSAGE = (
    "{name} loader is a stub: the real snapshot manifest "
    "(manuscript/datasets/manifest.json#{name}) has not been produced yet. "
    "For tests or demos, use `reap.datasets.load_synthetic_blobs()`; for a "
    "local copy, call `DatasetSnapshot(embeddings=..., metadata=...)` "
    "directly with your own data and provenance."
)


def load_ai_art() -> DatasetSnapshot:
    """Load the AI-art discourse dataset (1742 chunks, e5-large-v2, 1024-d).

    STUB. Replace with the real snapshot loader once
    `manuscript/datasets/manifest.json#ai_art_v1` is committed.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="ai_art_v1"))


def load_korean_forest() -> DatasetSnapshot:
    """Load the Korean forest policy dataset (905 chunks, MiniLM, 384-d).

    STUB. Replace with the real snapshot loader once
    `manuscript/datasets/manifest.json#korean_forest_v1` is committed.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="korean_forest_v1"))


def load_corp_sustainability() -> DatasetSnapshot:
    """Load the corporate sustainability dataset (1012 reports).

    STUB. Data extraction in progress; loader will wake up once the
    snapshot manifest is committed.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="corp_sustainability_v1"))


def load_us_presidential() -> DatasetSnapshot:
    """Load the US presidential language dataset.

    STUB. Infrastructure ready; data extraction pending.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="us_presidential_v1"))
