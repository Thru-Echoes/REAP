"""Tests for the shared REAP-research root locator and the results write guard.

The locator (``scripts/research_root.py``) is the single way result-writing
scripts resolve where number-bearing artifacts go. It must fail CLOSED: a
result-writing script invoked without any resolvable research root raises
rather than writing locally, and a candidate that was explicitly provided
(argument or environment variable) but does not exist raises rather than
silently falling through to the next candidate — a typo must never
quietly redirect results.

The write guard compares this repo's gitignored ``results/`` tree against the
Stage A import manifest (path -> SHA-256): a new or changed number-bearing
file beyond that baseline means some writer bypassed the locator contract.
The live guard run needs the research machine (real ``results/`` tree +
sibling checkout) and skips with a visible reason elsewhere; the guard
LOGIC is fully tested here on synthetic trees, so CI still covers it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from research_root import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    find_unaccounted_results_files,
    get_research_root,
    get_stage_a_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Locator: three-way resolution, fail-closed
# ---------------------------------------------------------------------------


def test_explicit_path_wins(tmp_path, monkeypatch):
    """An explicit existing directory is returned, resolved, over the env var."""
    explicit = tmp_path / "research-a"
    other = tmp_path / "research-b"
    explicit.mkdir()
    other.mkdir()
    monkeypatch.setenv("REAP_RESEARCH_ROOT", str(other))
    assert get_research_root(str(explicit)) == explicit.resolve()


def test_explicit_invalid_raises_not_falls_through(tmp_path, monkeypatch):
    """An explicit path that is not a directory raises — no silent fallback.

    The pre-extraction locator fell through to the env var / sibling here,
    so a typo in --research-root could quietly redirect results. Fail-closed
    means a provided-but-invalid candidate is an error naming that candidate.
    """
    valid_env = tmp_path / "env-root"
    valid_env.mkdir()
    monkeypatch.setenv("REAP_RESEARCH_ROOT", str(valid_env))
    missing = tmp_path / "typo"
    with pytest.raises(RuntimeError, match="typo"):
        get_research_root(str(missing))


def test_env_var_used_when_no_explicit(tmp_path, monkeypatch):
    """REAP_RESEARCH_ROOT resolves when no explicit path is given."""
    env_root = tmp_path / "env-root"
    env_root.mkdir()
    monkeypatch.setenv("REAP_RESEARCH_ROOT", str(env_root))
    assert get_research_root() == env_root.resolve()


def test_env_var_invalid_raises_not_falls_through(tmp_path, monkeypatch):
    """A set-but-invalid REAP_RESEARCH_ROOT raises rather than using the sibling."""
    monkeypatch.setenv("REAP_RESEARCH_ROOT", str(tmp_path / "gone"))
    reap_root = tmp_path / "REAP"
    sibling = tmp_path / "REAP-research"
    reap_root.mkdir()
    sibling.mkdir()
    with pytest.raises(RuntimeError, match="REAP_RESEARCH_ROOT"):
        get_research_root(reap_root=reap_root)


def test_sibling_fallback(tmp_path, monkeypatch):
    """With no explicit path and no env var, the ../REAP-research sibling wins."""
    monkeypatch.delenv("REAP_RESEARCH_ROOT", raising=False)
    reap_root = tmp_path / "REAP"
    sibling = tmp_path / "REAP-research"
    reap_root.mkdir()
    sibling.mkdir()
    assert get_research_root(reap_root=reap_root) == sibling.resolve()


def test_nothing_resolves_raises_with_actionable_message(tmp_path, monkeypatch):
    """No candidate at all → RuntimeError naming the env var and the flag."""
    monkeypatch.delenv("REAP_RESEARCH_ROOT", raising=False)
    reap_root = tmp_path / "REAP"
    reap_root.mkdir()
    with pytest.raises(RuntimeError, match="REAP_RESEARCH_ROOT"):
        get_research_root(reap_root=reap_root)


# ---------------------------------------------------------------------------
# Guard logic on synthetic trees (runs everywhere, including CI)
# ---------------------------------------------------------------------------


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


@pytest.fixture()
def synthetic_repo(tmp_path):
    """A tiny fake REAP tree with a results/ dir and a baseline map."""
    reap = tmp_path / "REAP"
    known = b"seed,ari\n0,0.5\n"
    _write(reap / "results" / "corpus" / "known.csv", known)
    baseline = {"results/corpus/known.csv": _sha256(known)}
    return reap, baseline


def test_guard_passes_when_results_match_baseline(synthetic_repo):
    """Files present in the baseline with matching hashes are accounted for."""
    reap, baseline = synthetic_repo
    assert find_unaccounted_results_files(reap, baseline, []) == []


def test_guard_flags_new_number_file(synthetic_repo):
    """A number-bearing file absent from the baseline is a violation."""
    reap, baseline = synthetic_repo
    _write(reap / "results" / "corpus" / "sneaky.json", b'{"ari": 0.9}')
    violations = find_unaccounted_results_files(reap, baseline, [])
    assert len(violations) == 1
    assert "sneaky.json" in violations[0]


def test_guard_flags_changed_file_by_hash(synthetic_repo):
    """A baseline file whose content changed is flagged, not just new paths."""
    reap, baseline = synthetic_repo
    _write(reap / "results" / "corpus" / "known.csv", b"seed,ari\n0,0.9\n")
    violations = find_unaccounted_results_files(reap, baseline, [])
    assert len(violations) == 1
    assert "known.csv" in violations[0]
    assert "changed" in violations[0]


def test_guard_honors_exclusion_patterns(synthetic_repo):
    """Manifest-excluded paths (e.g. recomputable matrices) are not violations."""
    reap, baseline = synthetic_repo
    _write(reap / "results" / "corpus" / "consensus_distance_matrix.npy", b"\x93NUMPY")
    violations = find_unaccounted_results_files(
        reap, baseline, ["results/**/consensus_distance_matrix.npy"]
    )
    assert violations == []


def test_guard_ignores_non_number_files(synthetic_repo):
    """Only number-bearing suffixes are guarded; logs/readmes are not."""
    reap, baseline = synthetic_repo
    _write(reap / "results" / "corpus" / "notes.txt", b"scratch")
    _write(reap / "results" / ".DS_Store", b"\x00")
    assert find_unaccounted_results_files(reap, baseline, []) == []


# ---------------------------------------------------------------------------
# Live guard on the research machine (skips visibly elsewhere)
# ---------------------------------------------------------------------------


def test_no_new_ignored_number_files_beyond_stage_a_baseline():
    """Working-machine guard: results/ holds nothing beyond the Stage A import.

    A violation means a writer produced a number-bearing artifact into this
    repo's gitignored results/ tree instead of the REAP-research home —
    bypassing the locator contract this change establishes.
    """
    try:
        research_root = get_research_root(reap_root=REPO_ROOT)
    except RuntimeError:
        pytest.skip("REAP-research root not resolvable (CI / fresh checkout) — live guard n/a")
    results_dir = REPO_ROOT / "results"
    if not results_dir.is_dir():
        pytest.skip("no local results/ tree (code-only checkout) — live guard n/a")

    baseline = get_stage_a_baseline(research_root)
    manifest = json.loads(
        (research_root / "migration" / "stage_a_import_manifest.json").read_text()
    )
    excluded = [entry.split(" (")[0].rstrip("/") for entry in manifest["excluded"]]
    violations = find_unaccounted_results_files(REPO_ROOT, baseline, excluded)
    assert violations == [], (
        "number-bearing files under results/ beyond the Stage A baseline "
        "(a writer bypassed the research-root contract):\n" + "\n".join(violations)
    )
