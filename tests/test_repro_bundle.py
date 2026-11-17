"""Tests for the versioned record file (``reap.repro_bundle``).

Covers: write/read round trip; fail-closed reading (missing field, unknown
field, empty seeds, malformed commit); run-status consistency; atomic writing;
the environment snapshot on the real installed stack; the code state of a real
git repository; and the protocol-version gate.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from reap.repro_bundle import (
    NOT_INSTALLED,
    NUMBER_MOVING_PACKAGES,
    CodeState,
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    read_repro_bundle,
    validate_protocol_version,
    write_repro_bundle,
)

START = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _get_bundle_fields(**overrides: object) -> dict:
    """Return the fields of a valid completed bundle, with optional overrides."""
    fields: dict = {
        "protocol_version": "1.6",
        "code": CodeState(commit="a" * 40, dirty=False),
        "script": "scripts/run_example.py",
        "run_id": "ai_art/T1",
        "dataset": "ai_art",
        "dataset_version": "2026-04-13",
        "dataset_sha256": "b" * 64,
        "seed_set": "D",
        "seeds": [11, 22, 33],
        "seed_manifest": "manuscript/seeds/seed_manifest.json",
        "config": {"n_neighbors": 53, "min_dist": 0.01, "hidden": [128, 64]},
        "evaluation_mode": "holdout",
        "metric_recipe_ids": ["trustworthiness", "distance_correlation"],
        "environment": get_environment_snapshot(),
        "run_status": "completed",
        "started_at": START,
        "finished_at": START + timedelta(minutes=5),
    }
    fields.update(overrides)
    return fields


class TestRoundTrip:
    def test_write_then_read_returns_an_equal_bundle(self, tmp_path: Path) -> None:
        bundle = ReproBundle(**_get_bundle_fields())
        path = tmp_path / "nested" / "bundle.json"
        write_repro_bundle(bundle, path)
        assert read_repro_bundle(path) == bundle

    def test_written_file_carries_the_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bundle.json"
        write_repro_bundle(ReproBundle(**_get_bundle_fields()), path)
        assert json.loads(path.read_text())["schema_version"] == "1.0"

    def test_failed_run_round_trips_with_its_reason(self, tmp_path: Path) -> None:
        bundle = ReproBundle(
            **_get_bundle_fields(run_status="failed", failure_reason="head diverged")
        )
        path = tmp_path / "bundle.json"
        write_repro_bundle(bundle, path)
        assert read_repro_bundle(path).failure_reason == "head diverged"


class TestFailClosedReading:
    def _get_written_json(self, tmp_path: Path) -> tuple[Path, dict]:
        path = tmp_path / "bundle.json"
        write_repro_bundle(ReproBundle(**_get_bundle_fields()), path)
        return path, json.loads(path.read_text())

    @pytest.mark.parametrize(
        "field",
        [
            "protocol_version",
            "code",
            "script",
            "run_id",
            "dataset",
            "dataset_sha256",
            "seeds",
            "config",
            "evaluation_mode",
            "metric_recipe_ids",
            "environment",
            "run_status",
            "started_at",
        ],
    )
    def test_missing_required_field_raises(self, tmp_path: Path, field: str) -> None:
        path, data = self._get_written_json(tmp_path)
        del data[field]
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            read_repro_bundle(path)

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        path, data = self._get_written_json(tmp_path)
        data["notes"] = "extra"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            read_repro_bundle(path)

    def test_missing_package_in_environment_raises(self, tmp_path: Path) -> None:
        path, data = self._get_written_json(tmp_path)
        del data["environment"]["packages"]["umap-learn"]
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError, match="missing packages"):
            read_repro_bundle(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_repro_bundle(tmp_path / "absent.json")


class TestValidation:
    def test_empty_seed_list_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReproBundle(**_get_bundle_fields(seeds=[]))

    def test_short_commit_raises(self) -> None:
        with pytest.raises(ValidationError):
            CodeState(commit="abc123", dirty=False)

    def test_failed_run_without_reason_raises(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason"):
            ReproBundle(**_get_bundle_fields(run_status="failed"))

    def test_completed_run_with_reason_raises(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason"):
            ReproBundle(**_get_bundle_fields(failure_reason="leftover"))

    def test_finish_before_start_raises(self) -> None:
        with pytest.raises(ValidationError, match="earlier"):
            ReproBundle(**_get_bundle_fields(finished_at=START - timedelta(seconds=1)))

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReproBundle(**_get_bundle_fields(started_at=datetime(2026, 11, 16, 12, 0)))

    def test_unregistered_evaluation_mode_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReproBundle(**_get_bundle_fields(evaluation_mode="oos"))

    def test_duplicate_recipe_ids_raise(self) -> None:
        with pytest.raises(ValidationError, match="duplicates"):
            ReproBundle(**_get_bundle_fields(metric_recipe_ids=["trustworthiness"] * 2))

    def test_bundle_is_frozen(self) -> None:
        bundle = ReproBundle(**_get_bundle_fields())
        with pytest.raises(ValidationError):
            bundle.run_status = "failed"  # type: ignore[misc]


class TestAtomicWrite:
    def test_no_temporary_file_is_left_behind(self, tmp_path: Path) -> None:
        write_repro_bundle(ReproBundle(**_get_bundle_fields()), tmp_path / "bundle.json")
        assert [p.name for p in tmp_path.iterdir()] == ["bundle.json"]

    def test_overwrite_replaces_the_previous_record(self, tmp_path: Path) -> None:
        path = tmp_path / "bundle.json"
        write_repro_bundle(ReproBundle(**_get_bundle_fields()), path)
        write_repro_bundle(ReproBundle(**_get_bundle_fields(run_id="ai_art/T2")), path)
        assert read_repro_bundle(path).run_id == "ai_art/T2"


class TestEnvironmentSnapshot:
    def test_records_every_number_moving_package(self) -> None:
        snapshot = get_environment_snapshot()
        assert set(snapshot.packages) == set(NUMBER_MOVING_PACKAGES)

    def test_installed_core_libraries_have_real_versions(self) -> None:
        packages = get_environment_snapshot().packages
        for name in ("numpy", "scipy", "scikit-learn", "umap-learn"):
            assert packages[name] != NOT_INSTALLED
            assert packages[name][0].isdigit()

    def test_records_blas_and_platform(self) -> None:
        snapshot = get_environment_snapshot()
        assert snapshot.blas and snapshot.lapack
        assert snapshot.platform and snapshot.python_version[0] == "3"


def _git(repo: Path, *args: str) -> None:
    """Run a git command in a test repository (side effect: mutates ``repo``)."""
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a one-commit git repository in ``tmp_path``."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


class TestCodeState:
    def test_clean_repository(self, git_repo: Path) -> None:
        state = get_code_state(git_repo)
        assert len(state.commit) == 40
        assert state.dirty is False

    def test_modified_tracked_file_marks_dirty(self, git_repo: Path) -> None:
        (git_repo / "tracked.txt").write_text("v2\n")
        assert get_code_state(git_repo).dirty is True

    def test_untracked_output_does_not_mark_dirty(self, git_repo: Path) -> None:
        (git_repo / "new_result.csv").write_text("x\n")
        assert get_code_state(git_repo).dirty is False

    def test_non_repository_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="git"):
            get_code_state(tmp_path)


class TestProtocolVersionGate:
    @pytest.mark.parametrize("version", ["1.6", "1.7", "1.10", "2.0"])
    def test_current_or_newer_passes(self, version: str) -> None:
        validate_protocol_version(version, minimum="1.6")

    @pytest.mark.parametrize("version", ["1.5", "1.4", "0.9"])
    def test_older_raises(self, version: str) -> None:
        with pytest.raises(ValueError, match="predates"):
            validate_protocol_version(version, minimum="1.6")

    @pytest.mark.parametrize("version", ["1", "1.6.0", "v1.6", ""])
    def test_malformed_raises(self, version: str) -> None:
        with pytest.raises(ValueError, match="major.minor"):
            validate_protocol_version(version, minimum="1.6")


def test_public_names_are_exported_from_the_package() -> None:
    """The record-file API is reachable as ``reap.<name>`` and listed in ``__all__``."""
    import reap

    for name in (
        "ReproBundle",
        "EnvironmentSnapshot",
        "CodeState",
        "get_environment_snapshot",
        "get_code_state",
        "validate_protocol_version",
        "write_repro_bundle",
        "read_repro_bundle",
    ):
        assert name in reap.__all__
        assert hasattr(reap, name)
