"""AI-art discourse loader (1,736 chunks, e5-large-v2, 1024-d).

The underlying corpus is the cleaned public-discourse chunks from the
sibling research project ``When-Algorithms-Meet-Artists``, produced by
their ``text_processing.py`` pipeline (1,736 chunks in
``data/public_discourse_clean_chunks.csv``). The pre-computed
embeddings live at
``figures/prefix_grid_search/prefix_embeddings_public.npy`` and were
produced with ``intfloat/e5-large-v2`` under the ``query: `` prefix
convention recommended by the e5 authors for clustering / retrieval.

Labels
------
The corpus has **no per-chunk expert labels**. The only label arrays
committed to the sibling project (``figures/config_comparison/*/labels_pub.npy``)
are KMeans outputs from downstream pipeline runs, not external ground
truth — they are REAP-adjacent artifacts, not expert attributions.
`DatasetSnapshot.labels` is therefore set to ``None``.

Usage
-----
>>> from reap.datasets import load_ai_art
>>> data = load_ai_art()                   # reads cache
>>> data.embeddings.shape                   # (1736, 1024)
>>> data.labels is None                     # True
>>> len(data.texts)                         # 1736

Building the cache (one-time)::

    python scripts/build_datasets.py \
        --dataset ai_art \
        --source-path ~/Documents/Berkeley/Research/When-Algorithms-Meet-Artists
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reap.datasets._cache import (
    compute_payload_sha256,
    get_default_cache_dir,
    load_snapshot,
    write_snapshot,
)
from reap.datasets._schema import DatasetMetadata, DatasetSnapshot

DATASET_NAME = "ai_art"
DATASET_VERSION = "2026-04-14"
EXPECTED_N_SAMPLES = 1736
EXPECTED_EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "intfloat/e5-large-v2 (prefix='query: ')"

_SOURCE_CHUNKS_RELPATH = Path("data/public_discourse_clean_chunks.csv")
_SOURCE_EMBEDDINGS_RELPATH = Path(
    "figures/prefix_grid_search/prefix_embeddings_public.npy"
)
_CHUNK_TEXT_COLUMN = "chunk_text"

_CITATION_NEEDED_NOTE = (
    "Citation pending — a formal BibTeX entry must be added at publication "
    "time, attributing the When-Algorithms-Meet-Artists corpus authors."
)


def load_ai_art(cache_dir: Path | None = None) -> DatasetSnapshot:
    """Load the AI-art discourse snapshot from the REAP cache.

    Reads ``<cache_dir>/ai_art/<DATASET_VERSION>/``. Raises
    `FileNotFoundError` with a clear builder-invocation hint if the
    cache is empty.

    Side effects: reads from the filesystem only.
    """
    return load_snapshot(DATASET_NAME, DATASET_VERSION, cache_dir=cache_dir)


def build_ai_art_snapshot(
    source_path: Path,
    cache_dir: Path | None = None,
) -> Path:
    """Materialize the AI-art snapshot from the sibling WAMA project.

    Reads two files under ``source_path``:

    - ``data/public_discourse_clean_chunks.csv`` — one row per chunk
      with a ``chunk_text`` column and 1,736 rows.
    - ``figures/prefix_grid_search/prefix_embeddings_public.npy`` —
      ``(1736, 1024)`` float32 e5-large-v2 embeddings, positionally
      aligned with the CSV rows (per WAMA's ``load_or_embed`` contract).

    Parameters
    ----------
    source_path
        Root of the sibling ``When-Algorithms-Meet-Artists`` project.
    cache_dir
        Optional override for the cache root.

    Returns
    -------
    Path to the materialized snapshot directory.

    Raises
    ------
    FileNotFoundError
        A required source file is missing.
    ValueError
        Row counts disagree between texts and embeddings, or the
        embedding dimensionality is not 1024.

    Side effects: reads source files and writes four files into the cache.
    """
    source_path = Path(source_path).expanduser().resolve()
    chunks_path = source_path / _SOURCE_CHUNKS_RELPATH
    emb_path = source_path / _SOURCE_EMBEDDINGS_RELPATH

    if not chunks_path.is_file():
        raise FileNotFoundError(f"AI-art chunks CSV not found: {chunks_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(
            f"AI-art embeddings NPY not found: {emb_path}. "
            "Expected prefix_embeddings_public.npy under figures/prefix_grid_search/."
        )

    texts = _read_chunk_texts(chunks_path)
    embeddings = np.load(emb_path, allow_pickle=False)

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"row count mismatch: texts={len(texts)} vs "
            f"embeddings={embeddings.shape[0]}. The WAMA embedding npy must be "
            f"the one generated from the clean_chunks CSV."
        )
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"unexpected embedding dim {embeddings.shape[1]}, "
            f"expected {EXPECTED_EMBEDDING_DIM}"
        )
    if embeddings.shape[0] != EXPECTED_N_SAMPLES:
        raise ValueError(
            f"unexpected n_samples {embeddings.shape[0]}, "
            f"expected {EXPECTED_N_SAMPLES}"
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    sha = compute_payload_sha256(embeddings, None, texts)
    notes = json.dumps(
        {
            "description": (
                "Public discourse about AI-generated art, cleaned into "
                "~1.7k text chunks (250 words each) for embedding."
            ),
            "expert_labels": (
                "None — per-chunk expert attributions are not available. "
                "Cluster-level human-verified topic labels exist upstream but "
                "are not per-sample ground truth."
            ),
            "license_note": (
                "Publicly-available corpus; treated as public-domain for "
                "research use."
            ),
            "citation_status": _CITATION_NEEDED_NOTE,
            "sibling_project_license": "GPL-3.0 (pipeline code only, not corpus)",
            "build_source_project": str(source_path),
        },
        ensure_ascii=False,
    )
    meta = DatasetMetadata(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        n_samples=embeddings.shape[0],
        embedding_dim=embeddings.shape[1],
        embedding_model=EMBEDDING_MODEL,
        preprocessing_version="wama-clean_chunks-e5_large_v2-query_prefix",
        sha256=sha,
        license="public-domain",
        citation=None,
        source_url=None,
        notes=notes,
    )
    snapshot = DatasetSnapshot(
        embeddings=embeddings,
        texts=texts,
        labels=None,
        metadata=meta,
    )

    target_cache = cache_dir if cache_dir is not None else get_default_cache_dir()
    return write_snapshot(snapshot, cache_dir=target_cache)


def _read_chunk_texts(path: Path) -> list[str]:
    """Load ``chunk_text`` column from the WAMA clean-chunks CSV.

    Side effects: reads the CSV from disk.
    """
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        if _CHUNK_TEXT_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{path} missing required column '{_CHUNK_TEXT_COLUMN}'; "
                f"found {reader.fieldnames}"
            )
        return [row[_CHUNK_TEXT_COLUMN] for row in reader]
