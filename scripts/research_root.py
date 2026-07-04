"""Shared locator for the REAP-research artifact root (fail-closed).

Every result-writing script resolves its output home through
``get_research_root()`` so that new number-bearing artifacts land in the
committed REAP-research sibling repository, never as untracked files under
this repo's gitignored ``results/`` tree. The Stage A import manifest is the
baseline for the write guard: any gitignored number-bearing file under this
repo's ``results/`` that is not in that baseline (or whose content changed)
means a writer bypassed the contract.

Lives under ``scripts/`` deliberately — ``src/reap`` stays research-agnostic
and must not know about the research repository layout.

Resolution order (kept identical to the metrics-catalog builder that first
carried this logic): explicit argument -> ``REAP_RESEARCH_ROOT`` environment
variable -> the ``../REAP-research`` sibling directory -> raise. One
deliberate strengthening over that original: a candidate that was PROVIDED
but does not exist raises immediately instead of silently falling through —
a typo in ``--research-root`` or the env var must never quietly redirect
results to a different tree. An empty-string value counts as not provided.

Exports: ``get_research_root``, ``get_stage_a_baseline``,
``find_unaccounted_results_files``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Suffixes treated as number-bearing artifacts for the write guard — the
# kinds the Stage A import covered (tables, records, arrays, model weights).
NUMBER_BEARING_SUFFIXES = frozenset({".csv", ".json", ".npy", ".npz", ".pt"})

_MANIFEST_RELPATH = Path("migration") / "stage_a_import_manifest.json"


def get_research_root(explicit: str | None = None, *, reap_root: Path | None = None) -> Path:
    """Resolve the REAP-research artifact root, fail-closed.

    Inputs: ``explicit`` (a caller-supplied path, e.g. from a
    ``--research-root`` CLI flag) and ``reap_root`` (this repository's root;
    defaults to the parent of ``scripts/`` — injectable for tests).
    Output: the resolved, absolute research-root path.
    Side effects: reads the ``REAP_RESEARCH_ROOT`` environment variable and
    the filesystem.

    Raises
    ------
    RuntimeError : If a provided candidate (explicit or env var) is not an
        existing directory, or if no candidate resolves at all. Result
        writers must treat this as fatal — never fall back to writing
        inside this repository.
    """
    base = reap_root if reap_root is not None else Path(__file__).resolve().parents[1]

    provided_candidates = (
        ("--research-root", explicit or None),
        ("REAP_RESEARCH_ROOT", os.environ.get("REAP_RESEARCH_ROOT") or None),
    )
    for label, candidate in provided_candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.is_dir():
            return path.resolve()
        raise RuntimeError(
            f"{label} points to {candidate!r}, which is not an existing "
            "directory. Refusing to fall back to another location: fix the "
            "path so results cannot be silently redirected."
        )

    sibling = base.parent / "REAP-research"
    if sibling.is_dir():
        return sibling.resolve()
    raise RuntimeError(
        "Cannot resolve the REAP-research artifact root: pass "
        "--research-root, set REAP_RESEARCH_ROOT, or check out the "
        "../REAP-research sibling. Refusing to continue without a "
        "committed artifact home."
    )


def get_stage_a_baseline(research_root: Path) -> dict[str, str]:
    """Load the Stage A import manifest as a ``path -> sha256`` baseline map.

    Inputs: the resolved research root. Output: one entry per imported file,
    keyed by the REAP-relative path recorded in the manifest.
    Side effects: reads the manifest file.

    Raises
    ------
    RuntimeError : If the manifest is missing — a resolvable research root
        without its migration manifest is an inconsistent state the guard
        must not paper over.
    """
    manifest_path = research_root / _MANIFEST_RELPATH
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Stage A import manifest not found at {manifest_path} — cannot "
            "establish the write-guard baseline."
        )
    manifest = json.loads(manifest_path.read_text())
    return {entry["path"]: entry["sha256"] for entry in manifest["files"]}


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked_results_files(reap_root: Path) -> set[str]:
    """Relative paths of git-tracked files under ``results/``.

    Tracked files are outside the guard's scope (they have a committed home
    already). Returns the empty set when ``reap_root`` is not a git
    repository — synthetic test trees and exported tarballs then simply
    guard every number-bearing file, the stricter direction.
    Side effects: runs ``git ls-files``.
    """
    try:
        output = subprocess.run(
            ["git", "-C", str(reap_root), "ls-files", "results"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def find_unaccounted_results_files(
    reap_root: Path, baseline: dict[str, str], excluded_patterns: list[str]
) -> list[str]:
    """List number-bearing files under ``results/`` not accounted for.

    A file is accounted for when it is git-tracked, matches an exclusion
    pattern from the manifest, or appears in the baseline with an identical
    SHA-256. Everything else is a violation: either NEW (a writer bypassed
    the research-root contract) or CHANGED (a writer altered an imported
    artifact in place).

    Inputs: the REAP repo root, the ``path -> sha256`` baseline, and the
    manifest's exclusion patterns (``fnmatch`` style, e.g.
    ``results/**/consensus_distance_matrix.npy``). Output: sorted,
    human-readable violation strings; empty when clean.
    Side effects: reads the filesystem and runs ``git ls-files``.
    """
    results_dir = reap_root / "results"
    if not results_dir.is_dir():
        return []

    tracked = _git_tracked_results_files(reap_root)
    violations: list[str] = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in NUMBER_BEARING_SUFFIXES:
            continue
        rel = path.relative_to(reap_root).as_posix()
        if rel in tracked:
            continue
        if any(fnmatch.fnmatch(rel, pattern) for pattern in excluded_patterns):
            continue
        if rel not in baseline:
            violations.append(f"{rel} (new — not in the Stage A baseline)")
        elif _sha256_file(path) != baseline[rel]:
            violations.append(f"{rel} (changed since the Stage A import)")
    return violations
