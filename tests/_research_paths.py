"""Where the tests find research artifacts and source projects, with no machine paths.

Some tests check committed research outputs (result arrays, metric CSVs, saved
projection-head records) or compare the dataset cache against the source
projects it was built from. None of that material ships with the package, so
these tests need to know where it lives on the current machine. This module is
the one place that decides, using only environment variables and the standard
side-by-side checkout layout, never an absolute path:

- **Research artifacts.** Imported, previously untracked outputs live in the
  REAP-research sibling repository under the same ``results/...`` paths they
  had here; outputs that were already tracked still live in this repository's
  ``results/``. ``get_artifact_path`` looks in REAP-research first, then here.
  ``REAP_RESULTS_ROOT`` overrides both with a single explicit root.
- **Source projects.** The AI-art and Korean-forest source projects resolve from
  ``REAP_AI_ART_SOURCE_PATH`` / ``REAP_KOREAN_FOREST_SOURCE_PATH``, else the
  sibling checkouts ``../When-Algorithms-Meet-Artists`` and
  ``../green-narrative/hye_in``.

Exports: ``REPO_ROOT``, ``get_research_root``, ``has_research_artifacts``,
``get_results_roots``, ``get_artifact_path``, ``get_ai_art_source_root``,
``get_korean_forest_source_root``.

Side effects: every function reads environment variables and the filesystem;
none writes anything.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_env_path(name: str) -> Path | None:
    """Return the path in environment variable ``name``, or None if unset or empty.

    Input: the variable name. Output: the expanded path, unchecked.
    """
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def get_research_root() -> Path | None:
    """Return the REAP-research checkout, or None when this machine has none.

    Resolution: ``REAP_RESEARCH_ROOT`` if set, else the ``../REAP-research``
    sibling. Output: the directory if it exists, else None.

    Raises
    ------
    RuntimeError : If ``REAP_RESEARCH_ROOT`` is set but is not a directory — a
        mistyped override must not quietly turn the artifact tests into skips.
    """
    explicit = _get_env_path("REAP_RESEARCH_ROOT")
    if explicit is not None:
        if not explicit.is_dir():
            raise RuntimeError(f"REAP_RESEARCH_ROOT={explicit} is not a directory")
        return explicit
    sibling = REPO_ROOT.parent / "REAP-research"
    return sibling if sibling.is_dir() else None


def has_research_artifacts() -> bool:
    """Report whether this machine is set up to run the artifact-dependent tests.

    True when ``REAP_RESULTS_ROOT`` is set or a REAP-research checkout resolves.
    On such a machine a missing artifact is a test failure, not a skip; on a
    code-only checkout (CI) the artifact tests skip with a visible reason.
    """
    return _get_env_path("REAP_RESULTS_ROOT") is not None or get_research_root() is not None


def get_results_roots() -> tuple[Path, ...]:
    """Return the ``results/`` directories to search, in priority order.

    ``REAP_RESULTS_ROOT`` alone if set; otherwise REAP-research's ``results/``
    (when that checkout exists) followed by this repository's ``results/``.
    """
    explicit = _get_env_path("REAP_RESULTS_ROOT")
    if explicit is not None:
        return (explicit,)
    research_root = get_research_root()
    roots = [] if research_root is None else [research_root / "results"]
    roots.append(REPO_ROOT / "results")
    return tuple(roots)


def get_artifact_path(*parts: str) -> Path:
    """Return where the artifact at ``results/<parts...>`` lives on this machine.

    Input: path components below ``results/``. Output: the first existing match
    across ``get_results_roots()``; if none exists, the candidate in the first
    root, so the caller's open/exists check fails with a meaningful path.

    Resolve files, not directories: one results directory can hold files from
    both roots (imported artifacts in REAP-research, tracked ones here), so a
    resolved directory would hide the files that live in the other root.
    """
    relative = Path(*parts)
    roots = get_results_roots()
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return roots[0] / relative


def get_ai_art_source_root() -> Path:
    """Return the AI-art source project (When-Algorithms-Meet-Artists) location.

    ``REAP_AI_ART_SOURCE_PATH`` if set, else the sibling checkout. Unchecked:
    callers skip or fail when it does not exist.
    """
    explicit = _get_env_path("REAP_AI_ART_SOURCE_PATH")
    return explicit if explicit is not None else REPO_ROOT.parent / "When-Algorithms-Meet-Artists"


def get_korean_forest_source_root() -> Path:
    """Return the Korean-forest source project (green-narrative/hye_in) location.

    ``REAP_KOREAN_FOREST_SOURCE_PATH`` if set, else the sibling checkout.
    Unchecked: callers skip or fail when it does not exist.
    """
    explicit = _get_env_path("REAP_KOREAN_FOREST_SOURCE_PATH")
    if explicit is not None:
        return explicit
    return REPO_ROOT.parent / "green-narrative" / "hye_in"
