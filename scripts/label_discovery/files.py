"""Atomic writers and fail-closed readers for label-discovery artifacts.

Every artifact is written to a uniquely named temporary file in its target
directory and then renamed into place, so a crash never leaves a half-written
file that could be mistaken for a real one.

Exports: ``get_file_sha256``, ``set_json``, ``set_jsonl``, ``get_jsonl``,
``set_text``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def get_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Side effects: reads ``path`` in 1 MiB chunks.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through a temporary file and a rename.

    Side effects: creates parent directories; writes ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def set_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically. Side effects: writes ``path``."""
    _set_bytes_atomic(path, text.encode("utf-8"))


def set_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented, key-sorted JSON atomically.

    Side effects: writes ``path``.
    """
    set_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def set_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write one JSON object per line, gzip-compressed when ``path`` ends in ``.gz``.

    Gzip output uses a fixed modification time of zero so identical rows give
    identical bytes (and so identical SHA-256) on every run.
    Output: the number of rows written. Side effects: writes ``path``.
    """
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    payload = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    _set_bytes_atomic(path, payload)
    return len(lines)


def get_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (gzip when the name ends in ``.gz``).

    Fails closed: a missing file or a malformed line raises.
    Side effects: reads ``path``.

    Raises
    ------
    FileNotFoundError : If ``path`` does not exist.
    ValueError : If a line is not a JSON object.
    """
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows
