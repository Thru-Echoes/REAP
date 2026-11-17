"""Korean forest out-of-sample pledges as raw MiniLM embeddings (1,662 × 384).

The pledge snapshot ``load_korean_forest_oos()`` holds the pledges *after* an
earlier projection head mapped them into an 18-d space. The corpus-level
hold-out evaluation (protocol §18.c K1) needs the pledges *before* any
projection, in the same raw embedding space as the reference corpus
(``load_korean_forest()``), so every method can place them itself.

The source project stores those raw vectors: each pledge row in
``<source>/data_may7_2026/projections/{Lee,Park,Moon}_projection.xlsx`` has an
``embeddings`` column holding the 384-d, unit-length
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` embedding the
source pipeline computed. This module reads those vectors as stored (no
re-embedding), in the same row order as ``load_korean_forest_oos()``, and
refuses to build if that order or any shape check fails.

Provenance notes, recorded in the snapshot metadata:

- Re-encoding each pledge's ``text_sentence`` with the same model reproduces
  the stored vector to float precision for 1,661 of 1,662 rows; the remaining
  row (an OCR-garbled Moon pledge) has cosine 0.988 to its re-encoding.
- The reference corpus was embedded from its preprocessed
  ``section_text_lexical`` column, the pledges from their raw text, as the
  source pipeline did. Both are unit-normalized MiniLM-L12 outputs.

Exports: ``DATASET_NAME``, ``DATASET_VERSION``, ``load_korean_forest_oos_raw``,
``build_korean_forest_oos_raw_snapshot``.

Example
-------
>>> from reap.datasets import load_korean_forest_oos_raw  # doctest: +SKIP
>>> pledges = load_korean_forest_oos_raw()  # doctest: +SKIP
>>> pledges.embeddings.shape  # doctest: +SKIP
(1662, 384)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from reap.datasets._cache import compute_payload_sha256, load_snapshot, write_snapshot
from reap.datasets._schema import DatasetMetadata, DatasetSnapshot
from reap.datasets.korean_forest_oos import (
    EXPECTED_N_SAMPLES,
    EXPECTED_PER_PRESIDENT,
    load_korean_forest_oos,
)

logger = logging.getLogger(__name__)

DATASET_NAME = "korean_forest_oos_raw"
DATASET_VERSION = "2026-05-09"
EXPECTED_EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_PRESIDENT_ORDER = ("Lee", "Park", "Moon")  # same file order as load_korean_forest_oos
_PARK_TEXT_COLUMN_TYPO = "ㅎ"
_UNIT_NORM_TOLERANCE = 1e-4  # float32 text round-trip of unit vectors
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_korean_forest_oos_raw(cache_dir: Path | None = None) -> DatasetSnapshot:
    """Load the raw pledge embeddings from the REAP cache.

    Side effects: reads the cached snapshot and verifies its SHA-256.

    Raises
    ------
    FileNotFoundError : If the snapshot has not been built; build it with
        ``build_korean_forest_oos_raw_snapshot(<source>)`` or
        ``scripts/build_datasets.py``.
    """
    return load_snapshot(DATASET_NAME, DATASET_VERSION, cache_dir=cache_dir)


def _parse_vector(cell: object, where: str) -> np.ndarray:
    """Parse one stored ``embeddings`` cell (a printed list of floats).

    Raises
    ------
    ValueError : If the cell does not hold exactly 384 numbers.
    """
    values = np.asarray([float(x) for x in _NUMBER.findall(str(cell))], dtype=np.float64)
    if values.shape != (EXPECTED_EMBEDDING_DIM,):
        raise ValueError(f"{where}: expected {EXPECTED_EMBEDDING_DIM} numbers, got {values.size}")
    return values


def build_korean_forest_oos_raw_snapshot(
    source_path: Path, cache_dir: Path | None = None
) -> Path:
    """Build the raw-pledge snapshot from the source project's stored embeddings.

    Parameters
    ----------
    source_path : Root of the ``green-narrative/hye_in`` source project.
    cache_dir : Optional cache-root override.

    Returns
    -------
    Path of the written snapshot directory.

    Raises
    ------
    FileNotFoundError : If a source workbook is missing.
    ValueError : On a wrong row count, a malformed or non-unit vector, or a row
        order that differs from ``load_korean_forest_oos()``'s texts.

    Side effects: reads three XLSX files and the cached pledge snapshot; writes
    the snapshot into the cache.
    """
    import pandas as pd

    source_path = Path(source_path).expanduser().resolve()
    projection_dir = source_path / "data_may7_2026" / "projections"
    vectors: list[np.ndarray] = []
    texts: list[str] = []
    for president in _PRESIDENT_ORDER:
        workbook = projection_dir / f"{president}_projection.xlsx"
        if not workbook.is_file():
            raise FileNotFoundError(f"Korean forest pledge source missing: {workbook}")
        frame = pd.read_excel(workbook)
        if _PARK_TEXT_COLUMN_TYPO in frame.columns and "text_sentence" not in frame.columns:
            frame = frame.rename(columns={_PARK_TEXT_COLUMN_TYPO: "text_sentence"})
        for column in ("embeddings", "text_sentence"):
            if column not in frame.columns:
                raise ValueError(f"{workbook.name} is missing column {column!r}")
        if len(frame) != EXPECTED_PER_PRESIDENT[president]:
            raise ValueError(
                f"{workbook.name} has {len(frame)} rows, expected "
                f"{EXPECTED_PER_PRESIDENT[president]}"
            )
        for row, cell in enumerate(frame["embeddings"].tolist()):
            vectors.append(_parse_vector(cell, f"{workbook.name} row {row}"))
        texts.extend(str(t) for t in frame["text_sentence"].tolist())

    embeddings = np.stack(vectors)
    if embeddings.shape != (EXPECTED_N_SAMPLES, EXPECTED_EMBEDDING_DIM):
        raise ValueError(f"unexpected pledge embedding shape {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.max(np.abs(norms - 1.0)) > _UNIT_NORM_TOLERANCE:
        raise ValueError("stored pledge embeddings are not unit length")
    projected = load_korean_forest_oos(cache_dir=cache_dir)
    if projected.texts is None or list(projected.texts) != texts:
        raise ValueError(
            "pledge row order differs from load_korean_forest_oos(); refusing to build"
        )

    embeddings32 = embeddings.astype(np.float32)
    notes = json.dumps(
        {
            "description": (
                "Raw MiniLM-L12 embeddings of the 1,662 out-of-sample pledge sentences, "
                "as stored by the source pipeline, row-aligned with "
                "load_korean_forest_oos()."
            ),
            "source_column": "embeddings (Lee, Park, Moon projection workbooks)",
            "text_column_embedded_by_source": "text_sentence (raw pledge text)",
            "reference_corpus_note": (
                "load_korean_forest() embeds section_text_lexical (preprocessed) text; "
                "both are unit-normalized MiniLM-L12 outputs"
            ),
            "reencoding_check": (
                "1,661 of 1,662 rows reproduce to float precision when text_sentence "
                "is re-encoded with the same model; one OCR-garbled Moon row has "
                "cosine 0.988"
            ),
            "row_order": list(_PRESIDENT_ORDER),
            "build_source_project": source_path.name,
        },
        ensure_ascii=False,
    )
    metadata = DatasetMetadata(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        n_samples=EXPECTED_N_SAMPLES,
        embedding_dim=EXPECTED_EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        preprocessing_version="hye_in-data_may7_2026-stored-embeddings",
        sha256=compute_payload_sha256(embeddings32, None, texts),
        license="public-domain",
        citation=None,
        source_url=None,
        notes=notes,
    )
    snapshot = DatasetSnapshot(embeddings=embeddings32, texts=texts, labels=None, metadata=metadata)
    path = write_snapshot(snapshot, cache_dir=cache_dir)
    logger.info("Built %s %s at %s", DATASET_NAME, DATASET_VERSION, path)
    return path
