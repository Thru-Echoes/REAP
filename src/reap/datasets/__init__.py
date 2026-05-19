"""REAP dataset API.

Every loader returns a `DatasetSnapshot` — a Pydantic-validated bundle
of embeddings, optional texts, optional expert labels, and full
provenance metadata (SHA256, license, citation, embedding model +
version, preprocessing version).

The contract is documented in ``.claude/rules/open-source-package.md``
(Dataset API Contract section) and pre-registered via the pipeline
stage table in ``manuscript/evaluation_protocol.md`` §4.

Public surface
--------------
Schema
    `DatasetSnapshot`, `DatasetMetadata`.
Synthetic
    `load_synthetic_blobs`, `load_golden_blobs`, `load_golden_text`.
Cache
    `load_snapshot`, `write_snapshot`, `get_default_cache_dir`,
    `compute_payload_sha256`.
Real datasets
    `load_ai_art`, `load_korean_forest`. Both read from the REAP
    cache (``~/.cache/reap/datasets/...``). Populate the cache with
    ``python scripts/build_datasets.py`` pointed at a sibling source
    project.
Stubs
    `load_corp_sustainability`, `load_us_presidential`. Raise
    `NotImplementedError` until their source data lands.

No side effects at import time.
"""

from __future__ import annotations

from reap.datasets._cache import (
    compute_payload_sha256,
    get_default_cache_dir,
    load_snapshot,
    write_snapshot,
)
from reap.datasets._schema import DatasetMetadata, DatasetSnapshot
from reap.datasets.ai_art import (
    build_ai_art_snapshot,
    load_ai_art,
)
from reap.datasets.korean_forest import (
    build_korean_forest_snapshot,
    load_korean_forest,
)
from reap.datasets.korean_forest_oos import (
    KoreanForestOOSSnapshot,
    build_korean_forest_oos_snapshot,
    load_korean_forest_oos,
)
from reap.datasets.synthetic import load_golden_blobs, load_synthetic_blobs
from reap.datasets.twenty_newsgroups import (
    GOLDEN_20NG_CLASSES,
    OVERLAP_PAIRS,
    load_golden_text,
)
from reap.datasets.twenty_newsgroups_oos import (
    OOS_20NG_CLASSES,
    load_twenty_newsgroups_oos,
)
from reap.datasets.twenty_newsgroups_reference import (
    REFERENCE_20NG_CLASSES,
    load_twenty_newsgroups_reference,
)

__all__ = [
    # Schema
    "DatasetSnapshot",
    "DatasetMetadata",
    # Cache helpers (power-user surface)
    "load_snapshot",
    "write_snapshot",
    "get_default_cache_dir",
    "compute_payload_sha256",
    # Synthetic
    "load_synthetic_blobs",
    "load_golden_blobs",
    "load_golden_text",
    "GOLDEN_20NG_CLASSES",
    "OVERLAP_PAIRS",
    # Real datasets
    "load_ai_art",
    "build_ai_art_snapshot",
    "load_korean_forest",
    "build_korean_forest_snapshot",
    "load_korean_forest_oos",
    "build_korean_forest_oos_snapshot",
    "KoreanForestOOSSnapshot",
    # 20-Newsgroups manuscript validation pair (in-distribution + OOS)
    "load_twenty_newsgroups_reference",
    "load_twenty_newsgroups_oos",
    "REFERENCE_20NG_CLASSES",
    "OOS_20NG_CLASSES",
    # Stubs (kept in the surface so imports don't break once they land)
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


def load_corp_sustainability() -> DatasetSnapshot:
    """Load the corporate sustainability dataset (1012 reports).

    STUB. Data extraction in progress; loader will wake up once the
    snapshot manifest is committed.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="corp_sustainability"))


def load_us_presidential() -> DatasetSnapshot:
    """Load the US presidential language dataset.

    STUB. Infrastructure ready; data extraction pending.
    """
    raise NotImplementedError(_STUB_MESSAGE.format(name="us_presidential"))
