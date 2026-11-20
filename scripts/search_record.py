"""Parameter-search records: the settings a REAP map may use, tied to its exact input.

REAP has no default parameters. A map built for a result takes every setting
(UMAP dimensions, neighbours, minimum distance, the K rule) from a parameter
search run on the same data. A search record states those settings together
with the SHA-256 of the input the search ran on, and scripts accept a record
only when that hash matches the data they are about to map, so settings
cannot carry over between corpora, reference variants or encoders.

A script may run without a record only under an explicit code-check flag; it
then stamps its outputs as a code check (``CODE_CHECK.txt`` plus a flag in its
record file), and nothing may be read from them.

Exports: ``SEARCH_RECORD_SCHEMA_VERSION``, ``CODE_CHECK_NOTE``,
``SearchProvenance``, ``SearchRecord``, ``get_input_sha256``,
``read_search_record``, ``get_checked_record``, ``write_code_check_marker``.

Example
-------
>>> record = get_checked_record(Path("search/presidential_spoken_e5.json"),  # doctest: +SKIP
...                             input_sha256=get_input_sha256(X), n_rows=len(X), method="reap")
>>> record.n_neighbors  # doctest: +SKIP
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StringConstraints

SEARCH_RECORD_SCHEMA_VERSION = "1.0"
CODE_CHECK_NOTE = (
    "CODE CHECK: these outputs come from a run without a recorded parameter search. "
    "They test that the code works and are not results: do not report, interpret, "
    "plot or use them to choose a direction.\n"
)

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SearchProvenance(BaseModel):
    """Where the settings in a search record came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    script: NonEmptyStr = Field(..., description="Script that ran the search.")
    commit: NonEmptyStr = Field(..., description="Git commit of the code that ran the search.")
    run_id: NonEmptyStr = Field(..., description="Identifier of the search run.")
    grid: dict[str, JsonValue] = Field(..., description="Every value the search tried.")
    selection_rule: NonEmptyStr = Field(
        ..., description="How the chosen setting was picked from the grid, in plain words."
    )
    seeds: list[int] = Field(..., min_length=1, description="Seeds the search used.")
    finished_at: AwareDatetime = Field(..., description="When the search finished.")


class SearchRecord(BaseModel):
    """The searched settings for one method on one exact input. Strict and frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = Field(
        default=SEARCH_RECORD_SCHEMA_VERSION, description="Version of this record format."
    )
    dataset: NonEmptyStr = Field(..., description="Dataset or corpus id.")
    variant: str | None = Field(default=None, description="Reference variant, if any.")
    encoder: NonEmptyStr = Field(..., description="Embedding model the input was made with.")
    method: NonEmptyStr = Field(..., description="Method the settings are for, e.g. 'reap'.")
    input_sha256: str = Field(
        ..., pattern=_SHA256_PATTERN,
        description="get_input_sha256 of the exact input rows the search ran on.",
    )
    n_rows: Annotated[int, Field(ge=1)] = Field(..., description="Rows in that input.")
    n_components: Annotated[int, Field(ge=1)] = Field(..., description="UMAP dimensions.")
    n_neighbors: Annotated[int, Field(ge=2)] = Field(..., description="UMAP neighbourhood size.")
    min_dist: Annotated[float, Field(ge=0.0)] = Field(..., description="UMAP minimum distance.")
    k_rule: NonEmptyStr = Field(..., description="The K rule the search fixed, in plain words.")
    k_range: list[int] = Field(..., min_length=1, description="Candidate K values of that rule.")
    k: int | None = Field(
        default=None, description="K itself, when the search fixed it rather than a rule."
    )
    search: SearchProvenance = Field(..., description="Where the settings came from.")


def get_input_sha256(X: np.ndarray) -> str:
    """SHA-256 of an input matrix as float32, C-ordered: the hash records are checked against."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()


def read_search_record(path: Path) -> SearchRecord:
    """Read and validate a search record. Raises on any missing, unknown or mistyped field."""
    return SearchRecord.model_validate_json(path.read_text(encoding="utf-8"))


def get_checked_record(path: Path, *, input_sha256: str, n_rows: int, method: str) -> SearchRecord:
    """A search record that belongs to this exact input and method, or an error.

    Raises ``ValueError`` when the record was made on different input rows,
    a different number of rows, or for another method.
    """
    record = read_search_record(path)
    if record.method != method:
        raise ValueError(f"{path}: search record is for method {record.method!r}, not {method!r}")
    if record.n_rows != n_rows or record.input_sha256 != input_sha256:
        raise ValueError(
            f"{path}: search record was made on a different input "
            f"({record.n_rows} rows, sha256 {record.input_sha256[:12]}...) than the "
            f"{n_rows} rows being mapped (sha256 {input_sha256[:12]}...); settings do not "
            "carry over, run a search on this input"
        )
    return record


def write_code_check_marker(out_dir: Path) -> None:
    """Stamp an output folder as a code check. Side effects: writes ``CODE_CHECK.txt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CODE_CHECK.txt").write_text(CODE_CHECK_NOTE, encoding="utf-8")
