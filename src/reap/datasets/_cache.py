"""Cache-backed snapshot storage for REAP dataset loaders.

Every materialized `DatasetSnapshot` lives on disk under
`~/.cache/reap/datasets/<name>/<version>/` as four files:

- ``embeddings.npy`` — float32/64 `(n_samples, embedding_dim)` array.
- ``labels.npy`` — int `(n_samples,)` array; absent when the dataset
  has no expert labels.
- ``texts.json`` — JSON array of source strings; absent when the
  dataset has no texts attached.
- ``metadata.json`` — serialized `DatasetMetadata` (Pydantic v2
  `model_dump_json` output).

The cache directory is resolved from, in order:
1. Explicit `cache_dir` argument passed to a helper.
2. ``REAP_CACHE_DIR`` environment variable.
3. ``~/.cache/reap/datasets``.

Separating cache reads (`load_snapshot`) from cache writes
(`write_snapshot`) keeps the runtime loaders purely declarative — no
filesystem side effects beyond reading. Builder scripts explicitly
invoke `write_snapshot` when materializing a new cache entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from reap.datasets._schema import DatasetMetadata, DatasetSnapshot


def get_default_cache_dir() -> Path:
    """Resolve the default REAP dataset cache directory.

    Returns the path without creating it. Side effects: none.
    """
    env = os.environ.get("REAP_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "reap" / "datasets"


def get_snapshot_dir(name: str, version: str, cache_dir: Path | None = None) -> Path:
    """Return the directory a snapshot lives in (no filesystem check).

    Side effects: none.
    """
    root = cache_dir if cache_dir is not None else get_default_cache_dir()
    return root / name / version


def compute_payload_sha256(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    texts: list[str] | None,
) -> str:
    """SHA256 of the snapshot's binary payload, computed deterministically.

    The hash covers, in order: embeddings bytes → labels bytes (or the
    literal ``b"no-labels"`` sentinel) → texts JSON bytes (or the literal
    ``b"no-texts"`` sentinel). Two snapshots with the same SHA256 have
    identical embeddings, labels, and texts.

    Side effects: none.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(embeddings).tobytes())
    if labels is None:
        h.update(b"no-labels")
    else:
        h.update(np.ascontiguousarray(labels).tobytes())
    if texts is None:
        h.update(b"no-texts")
    else:
        h.update(json.dumps(texts, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def write_snapshot(snapshot: DatasetSnapshot, cache_dir: Path | None = None) -> Path:
    """Materialize a snapshot into the cache.

    Writes ``embeddings.npy``, optional ``labels.npy``, optional
    ``texts.json``, and ``metadata.json`` under
    ``<cache_dir>/<name>/<version>/``. Creates parent directories as
    needed. Overwrites any existing files at the same path.

    Side effects: creates directories and writes four files.

    Returns
    -------
    Path to the snapshot directory.
    """
    meta = snapshot.metadata
    snap_dir = get_snapshot_dir(meta.name, meta.version, cache_dir=cache_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    np.save(snap_dir / "embeddings.npy", snapshot.embeddings, allow_pickle=False)

    if snapshot.labels is not None:
        np.save(snap_dir / "labels.npy", snapshot.labels, allow_pickle=False)
    else:
        _unlink_if_exists(snap_dir / "labels.npy")

    if snapshot.texts is not None:
        (snap_dir / "texts.json").write_text(
            json.dumps(snapshot.texts, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
    else:
        _unlink_if_exists(snap_dir / "texts.json")

    (snap_dir / "metadata.json").write_text(
        meta.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return snap_dir


def load_snapshot(
    name: str,
    version: str,
    cache_dir: Path | None = None,
    verify_sha256: bool = True,
) -> DatasetSnapshot:
    """Load a snapshot from the cache, validating metadata and integrity.

    Parameters
    ----------
    name, version
        Identify the snapshot directory under ``<cache_dir>/<name>/<version>/``.
    cache_dir
        Optional override; defaults to `get_default_cache_dir()`.
    verify_sha256
        If True (default), recompute the payload SHA256 and compare
        against ``metadata.sha256``. A mismatch raises ``ValueError``.

    Returns
    -------
    A validated `DatasetSnapshot`.

    Raises
    ------
    FileNotFoundError
        The snapshot directory or a required file is missing.
    ValueError
        Integrity check fails or metadata is inconsistent with payloads.

    Side effects: reads from the filesystem only.
    """
    snap_dir = get_snapshot_dir(name, version, cache_dir=cache_dir)
    if not snap_dir.is_dir():
        raise FileNotFoundError(
            f"REAP snapshot not found: {snap_dir}. "
            f"Run `python scripts/build_datasets.py --dataset {name}` to "
            f"materialize it from a local source project."
        )

    meta_path = snap_dir / "metadata.json"
    embeddings_path = snap_dir / "embeddings.npy"
    labels_path = snap_dir / "labels.npy"
    texts_path = snap_dir / "texts.json"

    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json missing in {snap_dir}")
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"embeddings.npy missing in {snap_dir}")

    meta = DatasetMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False) if labels_path.is_file() else None
    texts = (
        json.loads(texts_path.read_text(encoding="utf-8"))
        if texts_path.is_file()
        else None
    )

    if verify_sha256:
        observed = compute_payload_sha256(embeddings, labels, texts)
        if observed != meta.sha256:
            raise ValueError(
                f"SHA256 mismatch for snapshot {name}@{version}: "
                f"metadata says {meta.sha256}, payload hashes to {observed}. "
                f"The cache may be corrupted or out of date."
            )

    return DatasetSnapshot(
        embeddings=embeddings,
        labels=labels,
        texts=texts,
        metadata=meta,
    )


def _unlink_if_exists(path: Path) -> None:
    """Remove a file if present; ignore missing. Side effects: file I/O."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
