"""Korean forest policy loader (905 chunks, multilingual MiniLM, 384-d).

The underlying corpus is a set of Korean forest-policy strategy sentences
drawn from three presidential administrations (Moon, Park, Lee), embedded
with ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` in
the sibling research project
``green-narrative/hye_in``.

Expert taxonomy
---------------
Each sentence carries a ``strategy`` field — a human-authored policy
strategy descriptor attached by the corpus authors. There are 20 unique
strategy strings; we expose them as integer-coded expert labels via
`DatasetSnapshot.labels`. The string-to-integer mapping is recorded in
``metadata.notes`` and is part of the snapshot identity (it enters the
SHA256 through the labels array).

Usage
-----
>>> from reap.datasets import load_korean_forest
>>> data = load_korean_forest()            # reads cache
>>> data.embeddings.shape                   # (905, 384)
>>> data.labels                             # (905,) int strategy codes

Building the cache (one-time)::

    python scripts/build_datasets.py \
        --dataset korean_forest \
        --source-path ~/Documents/Berkeley/Research/green-narrative/hye_in
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

DATASET_NAME = "korean_forest"
DATASET_VERSION = "2026-04-14"
EXPECTED_N_SAMPLES = 905
EXPECTED_EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_SOURCE_TEXT_RELPATH = Path("data/og_data/Sections_of_Strategy_Sentences6.csv")
_SOURCE_EMBEDDINGS_RELPATH = Path(
    "data/og_data/Sections_of_Strategy_Sentences_embeddings6.csv"
)

_CITATION_NEEDED_NOTE = (
    "Citation pending — a formal BibTeX entry must be added at publication "
    "time, attributing Hye In Kim's Korean forest-policy corpus."
)


def load_korean_forest(cache_dir: Path | None = None) -> DatasetSnapshot:
    """Load the Korean forest policy snapshot from the REAP cache.

    Reads ``<cache_dir>/korean_forest/<DATASET_VERSION>/``. Raises
    `FileNotFoundError` with a clear builder-invocation hint if the
    cache is empty.

    Side effects: reads from the filesystem only.
    """
    return load_snapshot(DATASET_NAME, DATASET_VERSION, cache_dir=cache_dir)


def build_korean_forest_snapshot(
    source_path: Path,
    cache_dir: Path | None = None,
) -> Path:
    """Materialize the Korean forest snapshot from the sibling hye_in project.

    Reads two CSV files under ``source_path``:

    - ``data/og_data/Sections_of_Strategy_Sentences6.csv`` — one row per
      sentence with fields ``sent_id, president, strategy, text, ...``.
    - ``data/og_data/Sections_of_Strategy_Sentences_embeddings6.csv`` —
      one row per sentence with 384 float columns (MiniLM output).

    The two files are aligned positionally. We convert ``strategy`` to
    integer codes (sorted-unique mapping), keep the raw text, and write
    the snapshot to the cache.

    Parameters
    ----------
    source_path
        Root of the sibling ``green-narrative/hye_in`` project.
    cache_dir
        Optional override for the cache root.

    Returns
    -------
    Path to the materialized snapshot directory.

    Raises
    ------
    FileNotFoundError
        A required source CSV is missing.
    ValueError
        Row counts disagree between text and embedding CSVs, or the
        embedding dimensionality is not 384.

    Side effects: reads source CSVs and writes four files into the cache.
    """
    source_path = Path(source_path).expanduser().resolve()
    text_path = source_path / _SOURCE_TEXT_RELPATH
    emb_path = source_path / _SOURCE_EMBEDDINGS_RELPATH

    if not text_path.is_file():
        raise FileNotFoundError(f"Korean forest text CSV not found: {text_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"Korean forest embeddings CSV not found: {emb_path}")

    texts, strategies, presidents = _read_strategy_csv(text_path)
    embeddings = _read_embedding_csv(emb_path)

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"row count mismatch: texts={len(texts)} vs embeddings={embeddings.shape[0]}"
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

    unique_strategies = sorted(set(strategies))
    strategy_to_code = {s: i for i, s in enumerate(unique_strategies)}
    labels = np.asarray(
        [strategy_to_code[s] for s in strategies], dtype=np.int64
    )

    sha = compute_payload_sha256(embeddings, labels, texts)
    notes = json.dumps(
        {
            "description": (
                "Korean forest policy strategy sentences across three "
                "presidential administrations (Moon, Park, Lee)."
            ),
            "expert_label_source": (
                "`strategy` column — human-authored policy strategy descriptor "
                "attached by the corpus authors."
            ),
            "n_classes": len(unique_strategies),
            "strategy_label_map": strategy_to_code,
            "president_distribution": _value_counts(presidents),
            "license_note": (
                "Publicly-available corpus (Republic of Korea forest-policy "
                "documents); treated as public-domain for research use."
            ),
            "citation_status": _CITATION_NEEDED_NOTE,
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
        preprocessing_version="hye_in-og_data-sentences6",
        sha256=sha,
        license="public-domain",
        citation=None,
        source_url=None,
        notes=notes,
    )
    snapshot = DatasetSnapshot(
        embeddings=embeddings.astype(np.float32, copy=False),
        texts=texts,
        labels=labels,
        metadata=meta,
    )
    # The float32 cast above changes the SHA input; recompute so cache writes
    # the matching digest.
    sha_final = compute_payload_sha256(snapshot.embeddings, snapshot.labels, snapshot.texts)
    meta_final = meta.model_copy(update={"sha256": sha_final})
    snapshot = DatasetSnapshot(
        embeddings=snapshot.embeddings,
        texts=snapshot.texts,
        labels=snapshot.labels,
        metadata=meta_final,
    )

    target_cache = cache_dir if cache_dir is not None else get_default_cache_dir()
    return write_snapshot(snapshot, cache_dir=target_cache)


def _read_strategy_csv(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (texts, strategies, presidents) aligned row-by-row.

    Tolerant to the UTF-8 BOM on the first column name. Side effects:
    reads the CSV from disk.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        required = {"president", "strategy", "text"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} missing required columns: {sorted(missing)}"
            )
        texts: list[str] = []
        strategies: list[str] = []
        presidents: list[str] = []
        for row in reader:
            texts.append(row["text"])
            strategies.append(row["strategy"])
            presidents.append(row["president"])
    return texts, strategies, presidents


def _read_embedding_csv(path: Path) -> np.ndarray:
    """Load a (n, 384) float matrix from a headered CSV; no pandas.

    Side effects: reads the CSV from disk.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        rows: list[list[float]] = []
        for row in reader:
            if len(row) != n_cols:
                raise ValueError(
                    f"{path}: ragged row with {len(row)} cols, expected {n_cols}"
                )
            rows.append([float(x) for x in row])
    return np.asarray(rows, dtype=np.float64)


def _value_counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
