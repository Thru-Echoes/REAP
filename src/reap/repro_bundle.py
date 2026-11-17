"""Versioned, checked record file (``bundle.json``) for every result REAP produces.

A result is only as trustworthy as the record of how it was made. Each result
directory carries one ``bundle.json`` that says which code produced it (the
commit, and whether the working tree had uncommitted changes), on which data,
with which seeds and settings, under which protocol version, in which
evaluation mode, with which exact metric calculations, in which numerical
environment, and whether the run finished. Several scripts write these files
and one aggregation step reads them all back, so the format is a strict,
frozen data model rather than a loose dictionary: an unknown, missing, or
mistyped field fails on write *and* on read.

The numerical environment matters because the numbers themselves move with
library versions and with the low-level linear-algebra (BLAS) library numpy
was built against; ``get_environment_snapshot`` records both.

Exports: ``REPRO_BUNDLE_SCHEMA_VERSION``, ``NUMBER_MOVING_PACKAGES``,
``EnvironmentSnapshot``, ``CodeState``, ``ReproBundle``,
``get_environment_snapshot``, ``get_code_state``, ``validate_protocol_version``,
``write_repro_bundle``, ``read_repro_bundle``.

Example
-------
>>> from datetime import datetime, timezone
>>> from pathlib import Path
>>> from reap.repro_bundle import (
...     ReproBundle, get_code_state, get_environment_snapshot, write_repro_bundle,
... )
>>> bundle = ReproBundle(  # doctest: +SKIP
...     protocol_version="1.6",
...     code=get_code_state(Path(".")),
...     script="scripts/run_holdout.py",
...     run_id="ai_art/T1",
...     dataset="ai_art", dataset_version="2026-04-13", dataset_sha256="ab" * 32,
...     seed_set="D", seeds=[11, 22], seed_manifest="seed_manifest.json",
...     config={"n_neighbors": 53},
...     evaluation_mode="holdout",
...     metric_recipe_ids=["trustworthiness"],
...     environment=get_environment_snapshot(),
...     run_status="completed",
...     started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
... )
>>> write_repro_bundle(bundle, Path("results/ai_art/T1/bundle.json"))  # doctest: +SKIP
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

logger = logging.getLogger(__name__)

REPRO_BUNDLE_SCHEMA_VERSION = "1.0"

# Libraries whose versions can change a reported number. Recorded in every
# bundle; a package that is not installed is recorded as "not-installed".
NUMBER_MOVING_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "pynndescent",
    "numba",
    "torch",
)

NOT_INSTALLED = "not-installed"

_PROTOCOL_VERSION_PATTERN = re.compile(r"^\d+\.\d+$")
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

NonEmptyStr = Annotated[str, Field(min_length=1)]


class EnvironmentSnapshot(BaseModel):
    """The numerical environment a run executed in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: NonEmptyStr = Field(..., description="Python version, e.g. '3.12.13'.")
    platform: NonEmptyStr = Field(
        ...,
        description="Full platform string (OS, OS version, architecture), e.g. "
        "'macOS-26.5-arm64-arm-64bit'. The OS version matters when the BLAS "
        "library ships with the OS (Apple Accelerate).",
    )
    packages: dict[str, NonEmptyStr] = Field(
        ...,
        description="Installed version of every package in NUMBER_MOVING_PACKAGES, "
        "or 'not-installed'.",
    )
    blas: NonEmptyStr = Field(
        ...,
        description="BLAS library numpy was built against, as 'name version' from "
        "numpy's build configuration (version may be 'unknown').",
    )
    lapack: NonEmptyStr = Field(
        ..., description="LAPACK library numpy was built against, same format as blas."
    )

    @model_validator(mode="after")
    def _check_all_packages_recorded(self) -> EnvironmentSnapshot:
        """Fail if any number-moving package is missing from ``packages``."""
        missing = set(NUMBER_MOVING_PACKAGES) - set(self.packages)
        if missing:
            raise ValueError(f"environment snapshot is missing packages: {sorted(missing)}")
        return self


class CodeState(BaseModel):
    """Which code produced a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str = Field(
        ..., pattern=_COMMIT_PATTERN, description="Full 40-character git commit SHA of HEAD."
    )
    dirty: bool = Field(
        ...,
        description="True if tracked files had uncommitted changes when the run "
        "started; such a run cannot be reproduced from the commit alone.",
    )


class ReproBundle(BaseModel):
    """The record file stored as ``bundle.json`` next to every result.

    Strict and frozen: unknown fields, missing fields, wrong types, an empty
    seed list, or an inconsistent run status raise ``ValidationError``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = Field(
        REPRO_BUNDLE_SCHEMA_VERSION, description="Version of this record-file format."
    )
    protocol_version: str = Field(
        ...,
        pattern=r"^\d+\.\d+$",
        description="Evaluation-protocol version the run follows, e.g. '1.6'.",
    )
    code: CodeState = Field(..., description="Commit and working-tree state of the code.")
    script: NonEmptyStr = Field(
        ..., description="Repository-relative path of the script that produced the run."
    )
    run_id: NonEmptyStr = Field(
        ...,
        description="Identifier of the run within its plan, e.g. a run-matrix cell "
        "id such as 'ai_art/T1'.",
    )
    dataset: NonEmptyStr = Field(..., description="Dataset id, e.g. 'ai_art'.")
    dataset_version: NonEmptyStr = Field(..., description="Dataset snapshot version.")
    dataset_sha256: str = Field(
        ...,
        pattern=_SHA256_PATTERN,
        description="SHA-256 of the dataset snapshot's embeddings as loaded.",
    )
    encoder: str | None = Field(
        None, description="Embedding model id, when the run chose among encoders."
    )
    seed_set: NonEmptyStr = Field(..., description="Seed-manifest set name, e.g. 'D'.")
    seeds: list[int] = Field(
        ..., min_length=1, description="Every seed the run actually used, in order."
    )
    seed_manifest: NonEmptyStr = Field(
        ..., description="Repository-relative path of the seed manifest the seeds came from."
    )
    config: dict[str, JsonValue] = Field(
        ..., description="Every setting the result depends on (UMAP, head, split, K rule)."
    )
    evaluation_mode: Literal["cv", "final_fit", "holdout"] = Field(
        ...,
        description="How the reported numbers were obtained: cross-validation, "
        "fit and evaluated on the same data, or evaluated on held-out data "
        "(protocol v1.5 clause b).",
    )
    metric_recipe_ids: list[NonEmptyStr] = Field(
        ...,
        description="Registered recipe id of every metric the run reports "
        "(protocol v1.5 clause k).",
    )
    environment: EnvironmentSnapshot = Field(..., description="Numerical environment.")
    run_status: Literal["completed", "failed"] = Field(
        ..., description="Whether the run finished; a failed run still leaves a record."
    )
    failure_reason: str | None = Field(
        None, description="Why the run failed; required when run_status is 'failed'."
    )
    started_at: AwareDatetime = Field(..., description="Run start, timezone-aware.")
    finished_at: AwareDatetime = Field(..., description="Run end, timezone-aware.")

    @model_validator(mode="after")
    def _check_consistency(self) -> ReproBundle:
        """Enforce status/reason pairing, time order, and unique recipe ids."""
        if self.run_status == "failed" and not self.failure_reason:
            raise ValueError("a failed run must record failure_reason")
        if self.run_status == "completed" and self.failure_reason is not None:
            raise ValueError("a completed run must not carry failure_reason")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is earlier than started_at")
        if len(set(self.metric_recipe_ids)) != len(self.metric_recipe_ids):
            raise ValueError("metric_recipe_ids contains duplicates")
        return self


def _get_numpy_build_library(kind: Literal["blas", "lapack"]) -> str:
    """Return ``'name version'`` of the BLAS or LAPACK library numpy was built with.

    Raises
    ------
    RuntimeError : If numpy's build configuration does not name the library.
    """
    config = np.show_config(mode="dicts")
    entry = config["Build Dependencies"][kind]  # type: ignore[index]
    name = entry.get("name")
    if not name:
        raise RuntimeError(f"numpy build configuration does not name its {kind} library")
    return f"{name} {entry.get('version', 'unknown')}"


def get_environment_snapshot() -> EnvironmentSnapshot:
    """Capture the current numerical environment.

    Output: an ``EnvironmentSnapshot`` with the Python version, full platform
    string, every number-moving package version, and numpy's BLAS/LAPACK.
    Side effects: reads installed-package metadata.
    """
    packages: dict[str, str] = {}
    for name in NUMBER_MOVING_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = NOT_INSTALLED
    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        blas=_get_numpy_build_library("blas"),
        lapack=_get_numpy_build_library("lapack"),
    )


def _run_git(repo_root: Path, *args: str) -> str:
    """Run one git command in ``repo_root`` and return its stripped stdout.

    Side effects: runs a git subprocess.

    Raises
    ------
    RuntimeError : If git fails, e.g. because ``repo_root`` is not a repository.
    """
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def get_code_state(repo_root: Path) -> CodeState:
    """Return the commit and working-tree state of the repository at ``repo_root``.

    ``dirty`` is True when tracked files differ from the commit; untracked files
    (such as outputs being written) do not count.
    Side effects: runs git subprocesses.

    Raises
    ------
    RuntimeError : If ``repo_root`` is not inside a git repository — a result
        without a commit cannot be traced to its code.
    """
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    status = _run_git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return CodeState(commit=commit, dirty=bool(status))


def _parse_protocol_version(version: str) -> tuple[int, int]:
    """Parse ``'major.minor'`` into integers so '1.10' sorts after '1.9'.

    Raises
    ------
    ValueError : If ``version`` is not of the form 'major.minor'.
    """
    if not _PROTOCOL_VERSION_PATTERN.match(version):
        raise ValueError(f"protocol version {version!r} is not of the form 'major.minor'")
    major, minor = version.split(".")
    return int(major), int(minor)


def validate_protocol_version(protocol_version: str, minimum: str) -> None:
    """Refuse to proceed under a protocol older than ``minimum``.

    Result-generating drivers call this at startup, and the aggregation step
    calls it on every bundle it reads, so no result can come from, or be
    combined under, a protocol that predates the amendments it must follow.

    Raises
    ------
    ValueError : If either version is malformed, or ``protocol_version`` is
        older than ``minimum``.
    """
    if _parse_protocol_version(protocol_version) < _parse_protocol_version(minimum):
        raise ValueError(
            f"protocol version {protocol_version} predates the required minimum {minimum}"
        )


def write_repro_bundle(bundle: ReproBundle, path: Path) -> None:
    """Write ``bundle`` to ``path`` as JSON, atomically.

    The JSON goes to a uniquely named temporary file in the same directory and
    is then renamed into place, so a crash never leaves a half-written
    ``bundle.json`` that could be mistaken for a real one, and two writers
    never share a temporary file.
    Side effects: creates parent directories; writes ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(bundle.model_dump_json(indent=2))
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    logger.info("Wrote record file %s (run %s, %s)", path, bundle.run_id, bundle.run_status)


def read_repro_bundle(path: Path) -> ReproBundle:
    """Read and fully validate a ``bundle.json``.

    Fails closed: a missing file, invalid JSON, an unknown or missing field, or
    any inconsistency raises instead of returning a partial record.
    Side effects: reads ``path``.

    Raises
    ------
    FileNotFoundError : If ``path`` does not exist.
    pydantic.ValidationError : If the content does not match the schema.
    """
    return ReproBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))
