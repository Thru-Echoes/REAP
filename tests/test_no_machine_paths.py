"""Guard: no machine-specific absolute paths in the package, scripts, tests, or examples.

Research data and sibling projects are located through environment variables
and the side-by-side checkout layout (``tests/_research_paths.py``,
``scripts/research_root.py``), never through one person's home directory.
A hardcoded home-directory path silently breaks when a checkout moves: the
artifact-dependent verification tests then skip with "not present" instead of
running, which is how they once stopped running on every machine. This test
fails on any new such path.

Side effects: reads every Python file under ``src/``, ``scripts/``, ``tests/``
and ``examples/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("src", "scripts", "tests", "examples")

# Home-directory roots on macOS, Linux and Windows. Built from parts so this
# file does not match itself.
_HOME_ROOTS = ("/" + "Users/", "/" + "home/", "C:" + r"\\" + "Users" + r"\\")
MACHINE_PATH_PATTERN = re.compile(
    "(" + "|".join(re.escape(root) for root in _HOME_ROOTS) + r")[A-Za-z0-9._-]+"
)


def _get_scanned_files() -> list[Path]:
    """Return every Python file under the scanned directories, sorted."""
    return sorted(
        path for directory in SCANNED_DIRS for path in (REPO_ROOT / directory).rglob("*.py")
    )


def test_scan_covers_files() -> None:
    """The scan finds files, so an empty or moved tree cannot pass vacuously."""
    files = _get_scanned_files()
    assert any(path.parts[-3:-1] == ("src", "reap") for path in files)
    assert len(files) > 50


def test_pattern_catches_home_directory_paths() -> None:
    """The pattern flags home-directory paths and leaves relative and cache paths alone."""
    assert MACHINE_PATH_PATTERN.search('Path("' + "/" + 'Users/alice/research/REAP")')
    assert MACHINE_PATH_PATTERN.search("/" + "home/alice/data")
    assert not MACHINE_PATH_PATTERN.search('Path.home() / ".cache/reap/datasets"')
    assert not MACHINE_PATH_PATTERN.search("--source-path ../When-Algorithms-Meet-Artists")


@pytest.mark.parametrize("path", _get_scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_machine_specific_paths(path: Path) -> None:
    """No scanned file names a specific user's home directory."""
    hits = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if MACHINE_PATH_PATTERN.search(line)
    ]
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} hardcodes a machine-specific path; resolve it "
        "through an environment variable or the sibling-checkout layout instead:\n"
        + "\n".join(hits)
    )
