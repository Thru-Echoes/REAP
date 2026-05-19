"""Materialize REAP dataset snapshots from sibling research projects.

Reads raw embeddings + texts + taxonomy from a locally-checked-out
sibling project and writes a cached `DatasetSnapshot` into the REAP
cache directory (``~/.cache/reap/datasets/<name>/<version>/`` by
default, overridable via ``REAP_CACHE_DIR`` or ``--cache-dir``).

Examples
--------
Build the Korean forest snapshot from the hye_in project::

    python scripts/build_datasets.py \\
        --dataset korean_forest \\
        --source-path ~/Documents/Berkeley/Research/green-narrative/hye_in

Build the AI-art snapshot::

    python scripts/build_datasets.py \\
        --dataset ai_art \\
        --source-path ~/Documents/Berkeley/Research/When-Algorithms-Meet-Artists

Build both in one go (requires both ``--ai-art-source`` and
``--korean-forest-source``)::

    python scripts/build_datasets.py --all \\
        --ai-art-source ~/Documents/Berkeley/Research/When-Algorithms-Meet-Artists \\
        --korean-forest-source ~/Documents/Berkeley/Research/green-narrative/hye_in

Side effects
------------
Writes files under ``<cache_dir>/<name>/<version>/``. Nothing else.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reap.datasets.ai_art import build_ai_art_snapshot
from reap.datasets.korean_forest import build_korean_forest_snapshot

logger = logging.getLogger(__name__)

_BUILDERS = {
    "ai_art": build_ai_art_snapshot,
    "korean_forest": build_korean_forest_snapshot,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Side effects: builds snapshot files and logs progress to stderr.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    builds: list[tuple[str, Path]] = []
    if args.all:
        if not args.ai_art_source or not args.korean_forest_source:
            parser.error(
                "--all requires both --ai-art-source and --korean-forest-source"
            )
        builds.append(("ai_art", Path(args.ai_art_source)))
        builds.append(("korean_forest", Path(args.korean_forest_source)))
    else:
        if not args.dataset or not args.source_path:
            parser.error(
                "either --all or both --dataset and --source-path must be provided"
            )
        builds.append((args.dataset, Path(args.source_path)))

    for name, source in builds:
        builder = _BUILDERS.get(name)
        if builder is None:
            logger.error("unknown dataset %r (known: %s)", name, sorted(_BUILDERS))
            return 2
        logger.info("building %s from %s", name, source)
        snap_dir = builder(source, cache_dir=cache_dir)
        logger.info("wrote %s snapshot to %s", name, snap_dir)

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize REAP dataset snapshots from local sibling projects.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_BUILDERS),
        help="Name of the dataset to build (ignored when --all is set).",
    )
    parser.add_argument(
        "--source-path",
        help="Path to the sibling source project (ignored when --all is set).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Override the REAP cache root (defaults to ~/.cache/reap/datasets).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build both ai_art and korean_forest (requires --ai-art-source "
        "and --korean-forest-source).",
    )
    parser.add_argument("--ai-art-source", help="Used with --all.")
    parser.add_argument("--korean-forest-source", help="Used with --all.")
    return parser


if __name__ == "__main__":  # pragma: no cover — CLI entry point
    sys.exit(main())
