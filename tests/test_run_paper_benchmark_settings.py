"""The paper benchmark takes its settings only from a parameter-search record.

REAP has no default parameters: ``scripts/run_paper_benchmark.py`` must refuse
to run a dataset without a search record made on exactly that dataset's
embeddings, unless it is an explicit code check, and must read the UMAP
settings and K from the record rather than from its own table. These tests
stop before any map is built, so they are fast.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_paper_benchmark as bench  # noqa: E402  # pyright: ignore[reportMissingImports]
import search_record as sr  # noqa: E402  # pyright: ignore[reportMissingImports]


def _write_record(path: Path, X: np.ndarray, **overrides: object) -> Path:
    """A search record for exactly the rows X, with settings unlike the script's table."""
    fields: dict[str, object] = {
        "dataset": "synthetic", "encoder": "synthetic", "method": "reap",
        "input_sha256": sr.get_input_sha256(X), "n_rows": len(X), "n_components": 4,
        "n_neighbors": 10, "min_dist": 0.05, "k_rule": "silhouette over k_range",
        "k_range": [3, 4, 5], "k": None,
        "search": sr.SearchProvenance(
            script="test", commit="0" * 40, run_id="search/synthetic", grid={"n_neighbors": [10]},
            selection_rule="the only grid point", seeds=[1],
            finished_at=datetime.datetime.now(datetime.timezone.utc)),
    }
    fields.update(overrides)
    path.write_text(sr.SearchRecord.model_validate(fields).model_dump_json())
    return path


@pytest.fixture(scope="module")
def synthetic_X() -> np.ndarray:
    """The benchmark's synthetic dataset embeddings, as its loader returns them."""
    return np.asarray(bench.DATASET_CONFIGS["synthetic"].loader().embeddings)


def test_refuses_to_run_without_a_search_record() -> None:
    """No record and no code check is an error: REAP has no defaults."""
    with pytest.raises(ValueError, match="no default parameters"):
        bench.main(["--dataset", "synthetic", "--skip-bertopic"])


def test_reads_settings_from_the_record(synthetic_X: np.ndarray, tmp_path: Path) -> None:
    """With a matching record the config holds the record's settings, not the table's."""
    path = _write_record(tmp_path / "rec.json", synthetic_X)
    cfg, k_fixed, source = bench.get_resolved_config("synthetic", path, code_check=False)
    assert (cfg.n_components, cfg.n_neighbors, cfg.min_dist) == (4, 10, 0.05)
    assert cfg.k_range == [3, 4, 5]
    assert k_fixed is None
    assert source["code_check"] is False
    assert source["search_run_id"] == "search/synthetic"


def test_fixed_k_in_the_record_is_passed_on(synthetic_X: np.ndarray, tmp_path: Path) -> None:
    """A K fixed by the search is returned for the benchmark's matched-K mode."""
    path = _write_record(tmp_path / "rec.json", synthetic_X, k=4)
    _, k_fixed, _ = bench.get_resolved_config("synthetic", path, code_check=False)
    assert k_fixed == 4


def test_refuses_a_record_made_on_other_input(synthetic_X: np.ndarray, tmp_path: Path) -> None:
    """Settings searched on other rows do not carry over."""
    path = _write_record(tmp_path / "rec.json", synthetic_X[:-1])
    with pytest.raises(ValueError, match="different input"):
        bench.get_resolved_config("synthetic", path, code_check=False)


def test_code_check_is_labelled_and_exclusive(synthetic_X: np.ndarray, tmp_path: Path) -> None:
    """A code check keeps the table's settings and says so; it cannot be combined with a record."""
    cfg, _, source = bench.get_resolved_config("synthetic", None, code_check=True)
    assert cfg == bench.DATASET_CONFIGS["synthetic"]
    assert source["code_check"] is True
    path = _write_record(tmp_path / "rec.json", synthetic_X)
    with pytest.raises(ValueError, match="not both"):
        bench.get_resolved_config("synthetic", path, code_check=True)


def test_refuses_records_for_datasets_not_being_run(tmp_path: Path) -> None:
    """A record for a dataset that is not requested is an error, not silently ignored."""
    with pytest.raises(ValueError, match="not being run"):
        bench.main(["--dataset", "synthetic", "--code-check",
                    "--search-record", f"ai_art={tmp_path / 'x.json'}"])


def test_code_check_marker_text() -> None:
    """The code-check note says plainly that the outputs are not results."""
    assert "not results" in sr.CODE_CHECK_NOTE
