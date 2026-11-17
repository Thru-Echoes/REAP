"""The ``bundle.json`` record file for every label-discovery step.

Discovery steps read text and call language models; they do not fit an
embedding run's record (no embedding snapshot, no seed-manifest set, no
evaluation mode). This record keeps the same contract: which code produced
the artifact (commit and working-tree state), every input file with its
SHA-256, every setting, every model requested and every model version the API
reported, the seeds, the software environment, every output file with its
SHA-256, and whether the step finished. It is strict and frozen: an unknown,
missing, or mistyped field fails on write and on read.

Every discovery artifact is marked exploratory: it may not be quoted as a
paper number until the human review step promotes it.

Exports: ``RECORD_SCHEMA``, ``FileDigest``, ``CodeState``, ``ModelUse``,
``DiscoveryRecord``, ``get_code_state``, ``get_software_versions``,
``get_file_digest``, ``set_record``, ``get_record``.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from label_discovery.files import get_file_sha256, set_text

RECORD_SCHEMA = "label-discovery-record/1.0"

_SHA256 = r"^[0-9a-f]{64}$"
_COMMIT = r"^[0-9a-f]{40}$"

# Libraries whose versions can change a discovery artifact.
_TRACKED_PACKAGES = ("openai", "pydantic", "numpy", "pymupdf")


class FileDigest(BaseModel):
    """One input or output file and its content hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(..., min_length=1, description="What the file is to this step.")
    path: str = Field(
        ...,
        min_length=1,
        description="Path relative to the research root (outputs, research inputs) or a "
        "labelled source root such as 'white_house_root:' (raw sources).",
    )
    sha256: str = Field(..., pattern=_SHA256, description="SHA-256 of the file bytes.")
    n_rows: int | None = Field(default=None, ge=0, description="Row count for tabular files.")


class CodeState(BaseModel):
    """Commit and working-tree state of the REAP checkout that ran the step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str = Field(..., pattern=_COMMIT, description="Full 40-character commit hash.")
    dirty: bool = Field(..., description="True when tracked files had uncommitted changes.")


class ModelUse(BaseModel):
    """One language model a step used and the versions the API reported back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai", "anthropic-inline"] = Field(
        ...,
        description="'openai' for API calls; 'anthropic-inline' for coding done by a Claude "
        "model inside an interactive session, with no API call.",
    )
    requested_model: str = Field(..., min_length=1, description="Model id requested.")
    reported_models: list[str] = Field(
        ..., description="Distinct model ids the API reported in its responses."
    )
    settings: dict[str, JsonValue] = Field(
        ..., description="Every request setting (token budget, reasoning effort, format)."
    )
    n_calls: int = Field(..., ge=0, description="Number of calls made.")
    prompt_tokens: int = Field(..., ge=0, description="Total prompt tokens reported.")
    completion_tokens: int = Field(..., ge=0, description="Total completion tokens reported.")
    reasoning_tokens: int = Field(..., ge=0, description="Total hidden reasoning tokens reported.")


class DiscoveryRecord(BaseModel):
    """The record file stored as ``bundle.json`` next to every discovery artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: Literal["label-discovery-record/1.0"] = Field(
        default=RECORD_SCHEMA, description="Version of this record-file format."
    )
    step: Literal["D0", "D1", "D2", "D3", "D4"] = Field(..., description="Discovery step.")
    run_id: str = Field(..., min_length=1, description="Identifier of this artifact's run.")
    citability: Literal["exploratory"] = Field(
        default="exploratory",
        description="Discovery artifacts are exploratory until the human review promotes them.",
    )
    script: str = Field(..., min_length=1, description="REAP-relative path of the driver.")
    code: CodeState = Field(..., description="REAP code state.")
    inputs: list[FileDigest] = Field(..., description="Every file the step read.")
    outputs: list[FileDigest] = Field(..., min_length=1, description="Every file the step wrote.")
    config: dict[str, JsonValue] = Field(..., description="Every setting the artifact depends on.")
    seeds: list[int] = Field(..., description="Every random seed used; empty when none.")
    models: list[ModelUse] = Field(..., description="Language models used; empty when none.")
    python_version: str = Field(..., min_length=1, description="Python version.")
    platform: str = Field(..., min_length=1, description="Full platform string.")
    packages: dict[str, str] = Field(..., description="Versions of the tracked libraries.")
    run_status: Literal["completed", "failed"] = Field(..., description="Whether the step finished.")
    failure_reason: str | None = Field(default=None, description="Required when the step failed.")
    started_at: AwareDatetime = Field(..., description="Step start, timezone-aware.")
    finished_at: AwareDatetime = Field(..., description="Step end, timezone-aware.")

    @model_validator(mode="after")
    def _check_consistency(self) -> DiscoveryRecord:
        """Enforce status/reason pairing and time order."""
        if self.run_status == "failed" and not self.failure_reason:
            raise ValueError("a failed step must record failure_reason")
        if self.run_status == "completed" and self.failure_reason is not None:
            raise ValueError("a completed step must not carry failure_reason")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is earlier than started_at")
        return self


def get_code_state(repo_root: Path) -> CodeState:
    """Return the commit and dirty flag of the git checkout at ``repo_root``.

    Side effects: runs ``git rev-parse`` and ``git status``.

    Raises
    ------
    RuntimeError : If ``repo_root`` is not a git checkout.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"cannot read git state of {repo_root}: {exc}") from exc
    return CodeState(commit=commit, dirty=bool(status.strip()))


def get_software_versions() -> tuple[str, str, dict[str, str]]:
    """Return the Python version, platform string, and tracked package versions.

    A tracked package that is not installed is recorded as ``not-installed``.
    Side effects: reads installed-package metadata.
    """
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return platform.python_version(), platform.platform(), packages


def get_file_digest(role: str, path: Path, label: str, n_rows: int | None = None) -> FileDigest:
    """Hash ``path`` and describe it under the display path ``label``.

    Side effects: reads ``path``.
    """
    return FileDigest(role=role, path=label, sha256=get_file_sha256(path), n_rows=n_rows)


def set_record(record: DiscoveryRecord, path: Path) -> None:
    """Write the record atomically as ``bundle.json``. Side effects: writes ``path``."""
    set_text(path, record.model_dump_json(indent=2) + "\n")


def get_record(path: Path) -> DiscoveryRecord:
    """Read and fully validate a record file; fails closed on any defect.

    Side effects: reads ``path``.
    """
    return DiscoveryRecord.model_validate_json(path.read_text(encoding="utf-8"))
