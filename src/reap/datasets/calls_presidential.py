"""Earnings-call and presidential climate and energy passages (tertiary case study).

The snapshots are built outside this package, by the research repository's
call pilot, from a frozen passage input: keyword-matched passages of the
president's own speech in White House documents and of management and
analyst turns of S&P 500 earnings calls, kept by a relevance filter. The
same passages are embedded twice, once per encoder; each encoding is its
own snapshot, so the two can be compared on identical inputs. This module
only reads them from the REAP cache by name and version; it holds no paths
or data of its own.

There are no expert labels: ``labels`` is None.

Exports: ``DATASET_NAMES``, ``DATASET_VERSION``, ``SMOKE_VERSIONS``,
``load_calls_presidential``.
"""

from __future__ import annotations

from pathlib import Path

from reap.datasets._cache import load_snapshot
from reap.datasets._schema import DatasetSnapshot

# Snapshot name per encoder.
DATASET_NAMES = {
    "e5": "calls_presidential_e5",
    "openai": "calls_presidential_openai",
}
# The frozen input of the first full runs.
DATASET_VERSION = "2026-11-20"
# Smaller samples of the same input, used to measure time and memory first.
SMOKE_VERSIONS = ("smoke-1000", "smoke-2000")


def load_calls_presidential(
    encoder: str, version: str = DATASET_VERSION, cache_dir: Path | None = None
) -> DatasetSnapshot:
    """Load one encoding of the calls-and-presidential passages from the REAP cache.

    Parameters
    ----------
    encoder
        ``"e5"`` or ``"openai"`` (``DATASET_NAMES``).
    version
        Snapshot version: ``DATASET_VERSION`` or one of ``SMOKE_VERSIONS``.
    cache_dir
        Optional override of the REAP cache directory.

    Raises
    ------
    KeyError : If ``encoder`` is not a known encoder.
    FileNotFoundError, ValueError : From ``load_snapshot`` when the snapshot is
        missing or fails its integrity check.

    Side effects: reads from the filesystem only.
    """
    return load_snapshot(DATASET_NAMES[encoder], version, cache_dir=cache_dir)
