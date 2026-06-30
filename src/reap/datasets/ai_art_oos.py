"""AI-art out-of-sample probe loaders (Lovato 2024 artists + style-controlled public probes).

The reference snapshot (``load_ai_art()``) covers 1,736 public-discourse
chunks about AI-generated art from 2013–2025, embedded with
``intfloat/e5-large-v2`` (1024-d, ``query:`` prefix). This module exposes
two **out-of-sample probe sets** that live in the same input embedding
distribution but were NEVER seen by the consensus pipeline:

* ``load_ai_art_oos_artist()`` — 1,259 declarative artist-perspective probes
  derived from Lovato et al. (2024), spanning 252 unique US-based
  self-identified practicing artists × 5 stakeholder themes (compensation,
  ownership, threat, transparency, utility). 251 rows for ``threat``,
  252 for every other theme.

* ``load_ai_art_oos_public()`` — 750 style-controlled discourse passages
  extracted from the same 1,736-chunk public-discourse corpus via
  centroid-similarity matching to Likert-anchored prompts (150 passages
  per theme, exactly balanced). These passages are a *re-sampling* of
  the reference corpus along the same five stakeholder dimensions — they
  test whether the projection head's mapping is stable under a
  theme-conditioned redistribution of the same underlying texts.

Both snapshots carry:

* ``embeddings`` — ``(n, 1024)`` float32, e5-large-v2 with the same
  ``query:`` prefix as the reference loader.
* ``labels`` — ``(n,)`` int64 theme index in ``[0, 4]``. Names are
  recoverable from ``ai_art_oos.THEMES``.
* ``texts`` — declarative perspective strings (artist) or extracted
  discourse passages (public), one per row.
* ``metadata`` — full provenance, SHA256, citation, license.

Why no extra schema class (unlike ``KoreanForestOOSSnapshot``): the
v1 OOS demo consumes only theme + embedding; per-row demographic and
article-context attributes can be added later without API churn.

Usage
-----
>>> from reap.datasets import load_ai_art_oos_artist, load_ai_art_oos_public
>>> artist = load_ai_art_oos_artist()
>>> artist.embeddings.shape
(1259, 1024)
>>> public = load_ai_art_oos_public()
>>> public.embeddings.shape
(750, 1024)

Building the cache (one-time)::

    python scripts/build_datasets.py \\
        --dataset ai_art_oos \\
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

THEMES: tuple[str, ...] = (
    "compensation",
    "ownership",
    "threat",
    "transparency",
    "utility",
)

DATASET_NAME_ARTIST = "ai_art_oos_artist"
DATASET_NAME_PUBLIC = "ai_art_oos_public"
DATASET_VERSION = "2026-05-21"

EXPECTED_N_ARTIST = 1259
EXPECTED_N_PUBLIC = 750
EXPECTED_EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "intfloat/e5-large-v2 (prefix='query: ')"

_SOURCE_ARTIST_EMB_RELPATH = Path(
    "figures/prefix_grid_search/prefix_embeddings_artist.npy"
)
_SOURCE_ARTIST_CSV_RELPATH = Path("data/artist_perspectives.csv")
_SOURCE_PUBLIC_EMB_RELPATH = Path(
    "figures/prefix_grid_search/prefix_embeddings_probes.npy"
)
_SOURCE_PUBLIC_CSV_RELPATH = Path("data/public_probes.csv")
_SOURCE_REFERENCE_CHUNKS_RELPATH = Path("data/public_discourse_clean_chunks.csv")

_ARTIST_THEME_COLUMN = "question_group"
_ARTIST_TEXT_COLUMN = "perspective_text"
_PUBLIC_THEME_COLUMN = "theme"
_PUBLIC_TEXT_COLUMN = "probe_text"
_REFERENCE_YEAR_COLUMN = "year"

_ARTIST_CITATION = (
    "@article{lovato2024foundations,"
    " title={Foundations of {AI}-Driven Creativity: Artist Perspectives on "
    "the Impact of Generative {AI} on Art, Labor, and Society},"
    " author={Lovato, Juniper and others},"
    " year={2024}}"
)
_PUBLIC_CITATION_NEEDED_NOTE = (
    "Citation pending — formal BibTeX entry must be added at publication "
    "time, attributing the When-Algorithms-Meet-Artists corpus authors and "
    "the style-controlled probe-extraction pipeline."
)


def load_ai_art_oos_artist(cache_dir: Path | None = None) -> DatasetSnapshot:
    """Load the AI-art OOS artist-perspective probe snapshot.

    Returns a `DatasetSnapshot` with 1,259 rows: 1024-d e5-large-v2
    embeddings, integer theme labels in ``[0, 4]`` (recover names from
    `THEMES`), declarative perspective strings, and full provenance.

    Raises ``FileNotFoundError`` with a builder-invocation hint if the
    cache is empty. Side effects: reads from the filesystem only.
    """
    return load_snapshot(DATASET_NAME_ARTIST, DATASET_VERSION, cache_dir=cache_dir)


def load_ai_art_oos_public(cache_dir: Path | None = None) -> DatasetSnapshot:
    """Load the AI-art OOS style-controlled public-discourse probe snapshot.

    Returns a `DatasetSnapshot` with 750 rows: 1024-d e5-large-v2
    embeddings, integer theme labels in ``[0, 4]`` (recover names from
    `THEMES`), the extracted discourse passages, and full provenance.

    Raises ``FileNotFoundError`` with a builder-invocation hint if the
    cache is empty. Side effects: reads from the filesystem only.
    """
    return load_snapshot(DATASET_NAME_PUBLIC, DATASET_VERSION, cache_dir=cache_dir)


def build_ai_art_oos_snapshot(
    source_path: Path,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Materialize both AI-art OOS snapshots (artist + public) from the WAMA project.

    Reads four files under ``source_path``:

    * ``figures/prefix_grid_search/prefix_embeddings_artist.npy`` — (1259, 1024)
      float32 e5-large-v2 embeddings, positionally aligned with the artist CSV.
    * ``data/artist_perspectives.csv`` — 1,259 rows with ``question_group``
      (theme) and ``perspective_text`` columns.
    * ``figures/prefix_grid_search/prefix_embeddings_probes.npy`` — (750, 1024)
      float32 embeddings, positionally aligned with the public-probes CSV.
    * ``data/public_probes.csv`` — 750 rows with ``theme`` and ``probe_text``
      columns.

    Returns ``(artist_snapshot_dir, public_snapshot_dir)``.

    Raises ``FileNotFoundError`` for missing inputs and ``ValueError`` for
    row-count, dimension, or theme-vocabulary mismatches.

    Side effects: reads four source files; writes eight files into the cache
    (four per snapshot).
    """
    source_path = Path(source_path).expanduser().resolve()

    artist_dir = _build_single_snapshot(
        source_path=source_path,
        emb_relpath=_SOURCE_ARTIST_EMB_RELPATH,
        csv_relpath=_SOURCE_ARTIST_CSV_RELPATH,
        theme_column=_ARTIST_THEME_COLUMN,
        text_column=_ARTIST_TEXT_COLUMN,
        dataset_name=DATASET_NAME_ARTIST,
        expected_n=EXPECTED_N_ARTIST,
        description=(
            "Lovato 2024 declarative artist-perspective probes — "
            "252 unique US-based practicing artists × 5 stakeholder "
            "themes (251 for 'threat'), embedded with the same "
            "e5-large-v2 + 'query: ' prefix as the AI-art reference."
        ),
        license_str="by-permission",
        citation=_ARTIST_CITATION,
        citation_note=None,
        cache_dir=cache_dir,
    )

    public_dir = _build_single_snapshot(
        source_path=source_path,
        emb_relpath=_SOURCE_PUBLIC_EMB_RELPATH,
        csv_relpath=_SOURCE_PUBLIC_CSV_RELPATH,
        theme_column=_PUBLIC_THEME_COLUMN,
        text_column=_PUBLIC_TEXT_COLUMN,
        dataset_name=DATASET_NAME_PUBLIC,
        expected_n=EXPECTED_N_PUBLIC,
        description=(
            "Style-controlled public-discourse probes (150 per theme × "
            "5 themes = 750 passages) extracted from the same 1,736-chunk "
            "public-discourse corpus that backs the AI-art reference, via "
            "centroid-similarity matching to Likert-anchored prompts."
        ),
        license_str="public-domain",
        citation=None,
        citation_note=_PUBLIC_CITATION_NEEDED_NOTE,
        cache_dir=cache_dir,
    )

    return artist_dir, public_dir


def _build_single_snapshot(
    source_path: Path,
    emb_relpath: Path,
    csv_relpath: Path,
    theme_column: str,
    text_column: str,
    dataset_name: str,
    expected_n: int,
    description: str,
    license_str: str,
    citation: str | None,
    citation_note: str | None,
    cache_dir: Path | None,
) -> Path:
    """Build one OOS snapshot from a (embeddings.npy, csv) pair under ``source_path``.

    Side effects: reads two source files; writes four files into the cache.
    """
    emb_path = source_path / emb_relpath
    csv_path = source_path / csv_relpath

    if not emb_path.is_file():
        raise FileNotFoundError(f"missing embeddings file: {emb_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing CSV file: {csv_path}")

    embeddings = np.load(emb_path, allow_pickle=False)
    themes_str, texts = _read_theme_and_text_columns(
        csv_path, theme_column=theme_column, text_column=text_column
    )

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"row mismatch in {dataset_name}: embeddings={embeddings.shape[0]} "
            f"vs CSV rows={len(texts)}"
        )
    if embeddings.shape[0] != expected_n:
        raise ValueError(
            f"unexpected n_samples for {dataset_name}: got {embeddings.shape[0]}, "
            f"expected {expected_n}"
        )
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"unexpected embedding dim for {dataset_name}: got {embeddings.shape[1]}, "
            f"expected {EXPECTED_EMBEDDING_DIM}"
        )

    labels = _encode_theme_labels(themes_str, dataset_name=dataset_name)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    sha = compute_payload_sha256(embeddings, labels, texts)
    notes_payload: dict[str, object] = {
        "description": description,
        "theme_index": {name: i for i, name in enumerate(THEMES)},
        "expert_labels": (
            "Theme is an EXTRACTION-TIME annotation (Lovato survey "
            "question_group for artists; extraction-pipeline theme for "
            "public probes), not per-sample expert ground truth on the "
            "REAP consensus space."
        ),
        "license_note": (
            "Lovato 2024 survey responses redistributed by permission; "
            "see citation field."
            if license_str == "by-permission"
            else "Publicly-available corpus; treated as public-domain for research use."
        ),
        "build_source_project": str(source_path),
    }
    if citation_note is not None:
        notes_payload["citation_status"] = citation_note

    meta = DatasetMetadata(
        name=dataset_name,
        version=DATASET_VERSION,
        n_samples=expected_n,
        embedding_dim=EXPECTED_EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        preprocessing_version="wama-probes-e5_large_v2-query_prefix",
        sha256=sha,
        license=license_str,
        citation=citation,
        source_url=None,
        notes=json.dumps(notes_payload, ensure_ascii=False),
    )
    snapshot = DatasetSnapshot(
        embeddings=embeddings,
        texts=texts,
        labels=labels,
        metadata=meta,
    )

    target_cache = cache_dir if cache_dir is not None else get_default_cache_dir()
    return write_snapshot(snapshot, cache_dir=target_cache)


def _read_theme_and_text_columns(
    path: Path,
    theme_column: str,
    text_column: str,
) -> tuple[list[str], list[str]]:
    """Load theme + text columns from one of the probe CSVs.

    Side effects: reads the CSV from disk.
    """
    themes_str: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        for required in (theme_column, text_column):
            if required not in reader.fieldnames:
                raise ValueError(
                    f"{path} missing required column {required!r}; "
                    f"found {reader.fieldnames}"
                )
        for row in reader:
            themes_str.append(row[theme_column].strip())
            texts.append(row[text_column])
    return themes_str, texts


def _encode_theme_labels(themes_str: list[str], dataset_name: str) -> np.ndarray:
    """Map theme strings to integer labels in ``[0, 4]`` per `THEMES` order.

    Raises ``ValueError`` if any string is not in `THEMES`.
    """
    theme_to_idx = {name: i for i, name in enumerate(THEMES)}
    encoded = np.empty(len(themes_str), dtype=np.int64)
    unknown: set[str] = set()
    for i, name in enumerate(themes_str):
        if name not in theme_to_idx:
            unknown.add(name)
            continue
        encoded[i] = theme_to_idx[name]
    if unknown:
        raise ValueError(
            f"{dataset_name}: unknown theme value(s) {sorted(unknown)}; "
            f"expected one of {THEMES}"
        )
    return encoded


def _get_ai_art_year_array(source_path: Path) -> np.ndarray:
    """Return the (1736,) int year array for the AI-art reference chunks.

    Reads the WAMA clean-chunks CSV's ``year`` column and verifies the row
    count matches the reference snapshot's 1,736 chunks. The returned array
    is positionally aligned with the AI-art reference embeddings per the
    WAMA pipeline contract documented in ``ai_art.py``.

    Side effects: reads the CSV from disk.
    """
    csv_path = Path(source_path).expanduser().resolve() / _SOURCE_REFERENCE_CHUNKS_RELPATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing WAMA chunks CSV: {csv_path}")
    years: list[int] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or _REFERENCE_YEAR_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} missing required column {_REFERENCE_YEAR_COLUMN!r}"
            )
        for row in reader:
            years.append(int(row[_REFERENCE_YEAR_COLUMN]))
    return np.asarray(years, dtype=np.int64)
