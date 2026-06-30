"""Korean forest policy OOS pledges loader (1,662 chunks).

Out-of-sample district-level political pledge sentences from three South
Korean presidential election cycles (Lee = 480, Park = 633, Moon = 549),
projected by REAP's projection head into the same 18-d consensus space
used by the reference Korean forest planning corpus
(`load_korean_forest()`). This snapshot powers the §4.7 OOS conformal
filter validation in the manuscript and the §16 pre-registered Korean-
forest filter replication.

Provenance
----------
The pledges originate in the sibling research project
``green-narrative/hye_in/data_may7_2026/projections/{Lee,Park,Moon}_projection.xlsx``,
each row carrying the 18-d projection-head output (columns ``D1..D18``),
the cluster assignment (``cluster_k`` 1-indexed, normalized to 0-indexed
on load), the pledge text, and admin metadata. The Park file's text
column was renamed by the source authors to a single Korean letter
``ㅎ`` (typo); this loader maps it back to ``text_sentence`` on import.

What the snapshot exposes
-------------------------
- ``embeddings`` — ``(1662, 18)`` float64 projection-head coordinates,
  the canonical input to ``reap.filter`` for this dataset.
- ``cluster_labels`` — ``(1662,)`` int64 cluster_k (0-indexed).
- ``presidents`` — ``(1662,)`` object array of subgroup labels in
  ``{"Lee", "Park", "Moon"}``. Used by ``calibrate_filter_per_subgroup``.
- ``texts`` — pledge sentences (1,662 strings).
- ``metadata`` — `DatasetMetadata` with SHA256 of the binary payload,
  embedding-model name, citation/license stubs, and notes.

Usage
-----
>>> from reap.datasets import load_korean_forest_oos
>>> oos = load_korean_forest_oos()
>>> oos.embeddings.shape
(1662, 18)
>>> oos.cluster_labels.shape
(1662,)
>>> set(oos.presidents.tolist())
{'Lee', 'Park', 'Moon'}

Building the cache (one-time)::

    python scripts/build_datasets.py \\
        --dataset korean_forest_oos \\
        --source-path ~/Documents/Berkeley/Research/green-narrative/hye_in
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from reap.datasets._cache import get_default_cache_dir, get_snapshot_dir
from reap.datasets._schema import DatasetMetadata

DATASET_NAME = "korean_forest_oos"
DATASET_VERSION = "2026-05-09"
EXPECTED_N_SAMPLES = 1662
EXPECTED_PROJECTION_DIM = 18
EXPECTED_K = 8
EXPECTED_PRESIDENTS = ("Lee", "Moon", "Park")
EXPECTED_PER_PRESIDENT = {"Lee": 480, "Park": 633, "Moon": 549}

EMBEDDING_MODEL = (
    "reap-projection-head/v1 "
    "(input: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)"
)

_DIM_COLUMNS = [f"D{i}" for i in range(1, EXPECTED_PROJECTION_DIM + 1)]
_PARK_TEXT_COLUMN_TYPO = "ㅎ"

_CITATION_NEEDED_NOTE = (
    "Citation pending — a formal BibTeX entry must be added at publication "
    "time, attributing the South Korean presidential pledge corpus authors."
)


class KoreanForestOOSSnapshot(BaseModel):
    """Immutable bundle of OOS-projected pledge coords + subgroup labels.

    Distinct from `DatasetSnapshot` because OOS data carries a presidential
    subgroup label needed by the per-subgroup filter variant. The
    `embeddings` field stores the **18-d projection-head output** (not raw
    MiniLM 384-d embeddings) — the actual input to ``reap.filter``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    embeddings: np.ndarray = Field(
        ...,
        description=(
            "(n_samples, 18) float64 projection-head output coordinates."
        ),
    )
    cluster_labels: np.ndarray = Field(
        ...,
        description="(n_samples,) int cluster assignments (0-indexed).",
    )
    presidents: np.ndarray = Field(
        ...,
        description=(
            "(n_samples,) object array of subgroup labels in {Lee, Park, Moon}."
        ),
    )
    texts: list[str] = Field(
        ...,
        description="Pledge sentences, one per row.",
    )
    metadata: DatasetMetadata

    @field_validator("embeddings")
    @classmethod
    def _embeddings_must_be_2d_finite(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"embeddings must be 2-d, got shape {v.shape}")
        if v.shape[1] != EXPECTED_PROJECTION_DIM:
            raise ValueError(
                f"embeddings dim {v.shape[1]} != expected "
                f"{EXPECTED_PROJECTION_DIM}"
            )
        if not np.isfinite(v).all():
            raise ValueError(
                "embeddings contain NaN or Inf — clean data before wrapping"
            )
        return v

    @field_validator("cluster_labels")
    @classmethod
    def _cluster_labels_must_be_1d_int(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError(f"cluster_labels must be 1-d, got shape {v.shape}")
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(
                f"cluster_labels must be integer-typed, got dtype {v.dtype}"
            )
        return v

    @field_validator("presidents")
    @classmethod
    def _presidents_must_be_1d(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError(f"presidents must be 1-d, got shape {v.shape}")
        return v

    def model_post_init(self, _context: object) -> None:
        del _context
        n = self.metadata.n_samples
        if self.embeddings.shape[0] != n:
            raise ValueError(
                f"embeddings length {self.embeddings.shape[0]} != n_samples {n}"
            )
        if self.cluster_labels.shape != (n,):
            raise ValueError(
                f"cluster_labels shape {self.cluster_labels.shape} != ({n},)"
            )
        if self.presidents.shape != (n,):
            raise ValueError(
                f"presidents shape {self.presidents.shape} != ({n},)"
            )
        if len(self.texts) != n:
            raise ValueError(
                f"texts length {len(self.texts)} != n_samples {n}"
            )


def load_korean_forest_oos(cache_dir: Path | None = None) -> KoreanForestOOSSnapshot:
    """Load the Korean forest OOS pledge snapshot from the REAP cache.

    Reads ``<cache_dir>/korean_forest_oos/<DATASET_VERSION>/`` and verifies
    the SHA256 against the metadata. Raises ``FileNotFoundError`` with a
    builder-invocation hint if the cache is empty.

    Side effects: reads from the filesystem only.
    """
    snap_dir = get_snapshot_dir(DATASET_NAME, DATASET_VERSION, cache_dir=cache_dir)
    if not snap_dir.is_dir():
        raise FileNotFoundError(
            f"REAP snapshot not found: {snap_dir}. "
            f"Run `python scripts/build_datasets.py --dataset {DATASET_NAME}` "
            "to materialize it from a local source project."
        )
    meta_path = snap_dir / "metadata.json"
    embeddings_path = snap_dir / "embeddings.npy"
    cluster_labels_path = snap_dir / "cluster_labels.npy"
    presidents_path = snap_dir / "presidents.npy"
    texts_path = snap_dir / "texts.json"
    for p in (
        meta_path,
        embeddings_path,
        cluster_labels_path,
        presidents_path,
        texts_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file in {snap_dir}: {p.name}")

    meta = DatasetMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False)
    cluster_labels = np.load(cluster_labels_path, allow_pickle=False)
    presidents = np.load(presidents_path, allow_pickle=True)
    texts = json.loads(texts_path.read_text(encoding="utf-8"))

    observed = _payload_sha256(embeddings, cluster_labels, presidents, texts)
    if observed != meta.sha256:
        raise ValueError(
            f"SHA256 mismatch for {DATASET_NAME}@{DATASET_VERSION}: "
            f"metadata says {meta.sha256}, payload hashes to {observed}"
        )
    return KoreanForestOOSSnapshot(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        presidents=presidents,
        texts=texts,
        metadata=meta,
    )


def build_korean_forest_oos_snapshot(
    source_path: Path,
    cache_dir: Path | None = None,
) -> Path:
    """Materialize the OOS snapshot from the sibling green-narrative project.

    Reads three XLSX files under ``<source_path>/data_may7_2026/projections/``:
    ``Lee_projection.xlsx``, ``Park_projection.xlsx``, ``Moon_projection.xlsx``.
    Concatenates them, normalises the cluster_k column to 0-indexed,
    handles the Park ``ㅎ`` text-column typo, computes the SHA256, and
    writes the snapshot.

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
        A required source XLSX is missing.
    ImportError
        ``openpyxl`` not installed (required for ``.xlsx`` reading).
    ValueError
        Per-president row counts disagree with `EXPECTED_PER_PRESIDENT`,
        the projection dimensionality is wrong, or required columns are
        missing.

    Side effects: reads three XLSX files; writes five files into the cache.
    """
    try:
        import openpyxl as _openpyxl  # noqa: F401  # verifies engine is importable
        import pandas as pd

        del _openpyxl  # only needed to fail-fast at import time
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "build_korean_forest_oos_snapshot requires pandas and openpyxl. "
            "Install with `pip install pandas openpyxl`."
        ) from exc

    source_path = Path(source_path).expanduser().resolve()
    proj_dir = source_path / "data_may7_2026" / "projections"
    files = {p: proj_dir / f"{p}_projection.xlsx" for p in ("Lee", "Park", "Moon")}
    for p, f in files.items():
        if not f.is_file():
            raise FileNotFoundError(
                f"Korean forest OOS source missing for {p}: {f}"
            )

    embeddings_rows: list[np.ndarray] = []
    cluster_labels_rows: list[np.ndarray] = []
    presidents_rows: list[np.ndarray] = []
    texts: list[str] = []

    for president, f in files.items():
        df = pd.read_excel(f)
        # Park file has its text column renamed to the single-letter typo.
        if _PARK_TEXT_COLUMN_TYPO in df.columns and "text_sentence" not in df.columns:
            df = df.rename(columns={_PARK_TEXT_COLUMN_TYPO: "text_sentence"})
        for col in (*_DIM_COLUMNS, "cluster_k", "text_sentence"):
            if col not in df.columns:
                raise ValueError(
                    f"{f.name} missing required column {col!r}"
                )
        n_p = len(df)
        if n_p != EXPECTED_PER_PRESIDENT[president]:
            raise ValueError(
                f"{president} row count {n_p} != expected "
                f"{EXPECTED_PER_PRESIDENT[president]}"
            )
        coords = df[_DIM_COLUMNS].to_numpy(dtype=np.float64)
        # cluster_k in the source xlsx is 1-indexed; normalise to 0-indexed.
        labels = df["cluster_k"].astype(int).to_numpy() - 1
        if labels.min() < 0 or labels.max() >= EXPECTED_K:
            raise ValueError(
                f"{f.name} has cluster_k out of range after normalization: "
                f"[{int(labels.min())}, {int(labels.max())}], expected "
                f"[0, {EXPECTED_K - 1}]"
            )
        embeddings_rows.append(coords)
        cluster_labels_rows.append(labels.astype(np.int64))
        presidents_rows.append(np.array([president] * n_p, dtype=object))
        texts.extend(str(t) for t in df["text_sentence"].tolist())

    embeddings = np.concatenate(embeddings_rows, axis=0)
    cluster_labels = np.concatenate(cluster_labels_rows, axis=0)
    presidents = np.concatenate(presidents_rows, axis=0)
    n_total = embeddings.shape[0]
    if n_total != EXPECTED_N_SAMPLES:
        raise ValueError(
            f"total OOS row count {n_total} != expected {EXPECTED_N_SAMPLES}"
        )

    sha = _payload_sha256(embeddings, cluster_labels, presidents, texts)
    notes = json.dumps(
        {
            "description": (
                "Out-of-sample district-level political pledge sentences "
                "from three South Korean presidential election cycles (Lee, "
                "Park, Moon), projected via REAP's projection head into the "
                "18-d consensus space defined by the reference forest "
                "planning corpus. Used to validate the OOS conformal filter "
                "(manuscript §3.6, §4.7)."
            ),
            "subgroup_field": "presidents",
            "expected_per_president": EXPECTED_PER_PRESIDENT,
            "license_note": (
                "Publicly-available presidential pledge corpus; treated as "
                "public-domain for research use."
            ),
            "citation_status": _CITATION_NEEDED_NOTE,
            "build_source_project": str(source_path),
            "data_quality_flags": [
                "Park source xlsx has its text column literally renamed to "
                "the Korean letter 'ㅎ' (a typo); loader maps it back to "
                "'text_sentence'.",
                "Park and Moon files contain duplicated rows in some "
                "(cluster, president) cells (visible in cluster 8). Affects "
                "centroid estimates marginally; reported in §4.7.3.",
                "Moon file has OCR artifacts (random Latin character "
                "fragments) consistent with PDF-based extraction.",
            ],
        },
        ensure_ascii=False,
    )
    meta = DatasetMetadata(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        n_samples=n_total,
        embedding_dim=EXPECTED_PROJECTION_DIM,
        embedding_model=EMBEDDING_MODEL,
        preprocessing_version="presidential-pledges-2026-05-07-projected-18d",
        sha256=sha,
        license="public-domain",
        citation=None,
        source_url=None,
        notes=notes,
    )

    target_cache = cache_dir if cache_dir is not None else get_default_cache_dir()
    snap_dir = target_cache / DATASET_NAME / DATASET_VERSION
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.save(snap_dir / "embeddings.npy", embeddings, allow_pickle=False)
    np.save(snap_dir / "cluster_labels.npy", cluster_labels, allow_pickle=False)
    np.save(snap_dir / "presidents.npy", presidents, allow_pickle=True)
    (snap_dir / "texts.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    (snap_dir / "metadata.json").write_text(
        meta.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return snap_dir


def _payload_sha256(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    presidents: np.ndarray,
    texts: list[str],
) -> str:
    """Deterministic SHA256 of the OOS payload.

    Hash ordering: embeddings bytes → cluster_labels bytes → presidents
    JSON bytes → texts JSON bytes. Two snapshots with the same SHA have
    identical embeddings, cluster_labels, presidents, and texts.

    Side effects: none.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(embeddings).tobytes())
    h.update(np.ascontiguousarray(cluster_labels).tobytes())
    h.update(json.dumps(presidents.tolist(), ensure_ascii=False).encode("utf-8"))
    h.update(json.dumps(texts, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()
