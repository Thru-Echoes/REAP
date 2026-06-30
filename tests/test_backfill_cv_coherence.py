"""E2E tests for ``scripts/backfill_cv_coherence.py``.

These tests build a tiny in-memory results/ tree with a known synthetic
labeling + texts, run the backfill's :func:`backfill_cell` function, and
verify the three required invariants:

  1. ``topic_coherence_cv`` ends up populated in both ``topic_metrics.json``
     and ``combined_set_<X>/all_methods.csv`` for the method we ran.
  2. UMass + NPMI + diversity + exclusivity in ``topic_metrics.json`` are
     byte-identical pre- vs post-backfill.
  3. Other methods' rows in ``all_methods.csv`` (notably ``bertopic`` whose
     c_v is already populated) are byte-identical pre- vs post-backfill.

Plus:

  4. ``--dry-run`` mode is a true no-op on disk.
  5. The c_v value the backfill writes matches what
     :func:`reap.benchmarks._compute_topic_fields` returns on the same
     (texts, labels, top_n) inputs — the script must use the exact harness
     code path so its values are directly comparable to BERTopic's c_v.

Each test is self-contained and runs in <5 seconds.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# scripts/ isn't on sys.path by default; add it so we can import the backfill module.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Skip the whole module if gensim isn't installed (c_v requires it).
pytest.importorskip("gensim")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_corpus(
    n_clusters: int = 3, docs_per_cluster: int = 20, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """Build a tiny labeled text corpus where cluster id is encoded into the text.

    Each cluster's docs share several distinctive tokens (``"clusterX"``,
    ``"topicX"``, ``"wordX"``), with a small amount of cross-cluster noise.
    This gives well-separated topics so c_v ends up well-defined.
    """
    rng = np.random.default_rng(seed)
    labels: list[int] = []
    texts: list[str] = []
    common_filler = "the and of in to a"
    for c in range(n_clusters):
        for _ in range(docs_per_cluster):
            n_repeats = rng.integers(2, 5)
            topic_tokens = " ".join(
                [f"cluster{c}", f"topic{c}", f"word{c}"] * int(n_repeats)
            )
            # Sprinkle a single token from a *different* cluster as light noise
            # so c_v isn't trivially saturating at 1.0.
            other = (c + 1) % n_clusters
            noise = f"word{other}" if rng.random() < 0.25 else ""
            texts.append(f"{topic_tokens} {noise} {common_filler}".strip())
            labels.append(c)
    return np.asarray(labels, dtype=np.int64), texts


@pytest.fixture
def synthetic_cell(tmp_path: Path) -> dict[str, Any]:
    """Build a minimal results/ tree with one (dataset, method, set) cell ready to backfill.

    Pre-state:
      - ``topic_metrics.json`` exists with UMass/NPMI/diversity/exclusivity
        set but ``topic_coherence_cv`` is None (the situation on disk for
        the 4 non-bertopic methods).
      - ``all_methods.csv`` exists with a row for our method (cv cell empty)
        and an unrelated ``bertopic`` row (cv cell already populated — should
        survive untouched).
    """
    labels, texts = _make_synthetic_corpus(n_clusters=3, docs_per_cluster=20, seed=0)

    dataset_name = "synthetic_test"
    method = "reap"
    set_name = "A"
    dataset_root = tmp_path / dataset_name
    cell_dir = dataset_root / method / f"set_{set_name}"
    cell_dir.mkdir(parents=True)
    combined_dir = dataset_root / f"combined_set_{set_name}"
    combined_dir.mkdir()

    np.save(cell_dir / "consensus_labels.npy", labels)

    pre_topic_metrics = {
        "topic_coherence_umass": -2.123456,
        "topic_coherence_npmi": -0.413579,
        "topic_coherence_cv": None,
        "topic_diversity": 0.85,
        "topic_exclusivity": 0.531,
    }
    (cell_dir / "topic_metrics.json").write_text(
        json.dumps(pre_topic_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    # Combined CSV with the method we're backfilling (cv empty) and a
    # bertopic row whose cv is already populated and must NOT be touched.
    fieldnames = [
        "method",
        "silhouette",
        "trustworthiness",
        "topic_coherence_umass",
        "topic_coherence_npmi",
        "topic_coherence_cv",
        "topic_diversity",
    ]
    rows = [
        {
            "method": "reap",
            "silhouette": "0.500000",
            "trustworthiness": "0.900000",
            "topic_coherence_umass": "-2.123456",
            "topic_coherence_npmi": "-0.413579",
            "topic_coherence_cv": "",  # empty -> must end up populated
            "topic_diversity": "0.850000",
        },
        {
            "method": "bertopic",
            "silhouette": "0.400000",
            "trustworthiness": "0.910000",
            "topic_coherence_umass": "-1.800000",
            "topic_coherence_npmi": "-0.420000",
            "topic_coherence_cv": "0.420000",  # pre-populated; must be preserved
            "topic_diversity": "0.800000",
        },
    ]
    with (combined_dir / "all_methods.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "tmp_path": tmp_path,
        "dataset_root": dataset_root,
        "method": method,
        "set_name": set_name,
        "labels": labels,
        "texts": texts,
        "pre_topic_metrics": pre_topic_metrics,
        "cell_dir": cell_dir,
        "combined_csv": combined_dir / "all_methods.csv",
    }


# ---------------------------------------------------------------------------
# Invariant 1 — c_v gets populated
# ---------------------------------------------------------------------------


def test_backfill_writes_c_v_for_method(synthetic_cell: dict[str, Any]) -> None:
    """backfill_cell writes a finite, non-None c_v into the cell."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    result = backfill_cell(
        dataset_root=synthetic_cell["dataset_root"],
        method=synthetic_cell["method"],
        set_name=synthetic_cell["set_name"],
        texts=synthetic_cell["texts"],
        dry_run=False,
    )

    assert result["status"] == "written"
    assert result["old_cv"] is None
    assert isinstance(result["new_cv"], float)
    assert np.isfinite(result["new_cv"])
    # c_v lives in roughly [0, 1] for sensible corpora.
    assert -0.05 <= result["new_cv"] <= 1.05, (
        f"c_v out of expected range [0, 1]: {result['new_cv']}"
    )

    # File on disk reflects the update.
    post = json.loads((synthetic_cell["cell_dir"] / "topic_metrics.json").read_text())
    assert post["topic_coherence_cv"] == result["new_cv"]


# ---------------------------------------------------------------------------
# Invariant 2 — UMass/NPMI/diversity/exclusivity preserved byte-identical
# ---------------------------------------------------------------------------


def test_backfill_preserves_other_topic_metric_fields(
    synthetic_cell: dict[str, Any],
) -> None:
    """All other topic_metrics.json fields are identical pre- vs post-backfill."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    backfill_cell(
        dataset_root=synthetic_cell["dataset_root"],
        method=synthetic_cell["method"],
        set_name=synthetic_cell["set_name"],
        texts=synthetic_cell["texts"],
        dry_run=False,
    )

    post = json.loads((synthetic_cell["cell_dir"] / "topic_metrics.json").read_text())
    pre = synthetic_cell["pre_topic_metrics"]
    for key in [
        "topic_coherence_umass",
        "topic_coherence_npmi",
        "topic_diversity",
        "topic_exclusivity",
    ]:
        assert post[key] == pre[key], (
            f"{key} drifted: pre={pre[key]!r} post={post[key]!r}"
        )
    # And c_v is now set.
    assert post["topic_coherence_cv"] is not None


# ---------------------------------------------------------------------------
# Invariant 3 — other rows in all_methods.csv preserved
# ---------------------------------------------------------------------------


def test_backfill_preserves_other_methods_in_csv(
    synthetic_cell: dict[str, Any],
) -> None:
    """The bertopic row (cv already set) is byte-identical post-backfill."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    csv_path = synthetic_cell["combined_csv"]

    with csv_path.open(encoding="utf-8") as f:
        pre_rows = {r["method"]: dict(r) for r in csv.DictReader(f)}

    backfill_cell(
        dataset_root=synthetic_cell["dataset_root"],
        method=synthetic_cell["method"],
        set_name=synthetic_cell["set_name"],
        texts=synthetic_cell["texts"],
        dry_run=False,
    )

    with csv_path.open(encoding="utf-8") as f:
        post_rows = {r["method"]: dict(r) for r in csv.DictReader(f)}

    # bertopic row must be byte-identical.
    assert post_rows["bertopic"] == pre_rows["bertopic"], (
        f"bertopic row drifted: pre={pre_rows['bertopic']} post={post_rows['bertopic']}"
    )

    # reap row must differ ONLY in topic_coherence_cv.
    pre_reap = pre_rows["reap"]
    post_reap = post_rows["reap"]
    for key in pre_reap:
        if key == "topic_coherence_cv":
            assert pre_reap[key] == ""
            assert post_reap[key] != ""
            assert float(post_reap[key]) == float(post_reap[key])  # not NaN
        else:
            assert post_reap[key] == pre_reap[key], (
                f"reap row {key} drifted: pre={pre_reap[key]!r} post={post_reap[key]!r}"
            )


# ---------------------------------------------------------------------------
# Invariant 4 — dry-run is a true no-op
# ---------------------------------------------------------------------------


def test_backfill_dry_run_does_not_modify_files(
    synthetic_cell: dict[str, Any],
) -> None:
    """dry_run=True: the script computes c_v but writes nothing to disk."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    cell_dir = synthetic_cell["cell_dir"]
    csv_path = synthetic_cell["combined_csv"]
    pre_tm = (cell_dir / "topic_metrics.json").read_text()
    pre_csv = csv_path.read_text()

    result = backfill_cell(
        dataset_root=synthetic_cell["dataset_root"],
        method=synthetic_cell["method"],
        set_name=synthetic_cell["set_name"],
        texts=synthetic_cell["texts"],
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert isinstance(result["new_cv"], float)
    assert result["old_cv"] is None
    assert (cell_dir / "topic_metrics.json").read_text() == pre_tm
    assert csv_path.read_text() == pre_csv


# ---------------------------------------------------------------------------
# Invariant 5 — c_v matches reap.benchmarks._compute_topic_fields exactly
# ---------------------------------------------------------------------------


def test_backfill_c_v_matches_harness_code_path(
    synthetic_cell: dict[str, Any],
) -> None:
    """The value written must equal what _compute_topic_fields returns on the same inputs.

    This is the safety net that guarantees the backfill's c_v values are
    methodologically interchangeable with BERTopic's c_v (which was computed
    through the same private function during the overnight sweep).
    """
    from backfill_cv_coherence import TOP_N, backfill_cell  # type: ignore[import-not-found]

    from reap.benchmarks import _compute_topic_fields

    direct = _compute_topic_fields(
        synthetic_cell["texts"], synthetic_cell["labels"], top_n=TOP_N
    )
    expected_cv = direct["topic_coherence_cv"]

    result = backfill_cell(
        dataset_root=synthetic_cell["dataset_root"],
        method=synthetic_cell["method"],
        set_name=synthetic_cell["set_name"],
        texts=synthetic_cell["texts"],
        dry_run=True,
    )
    assert result["new_cv"] == expected_cv, (
        f"backfill returned {result['new_cv']!r} but _compute_topic_fields returned "
        f"{expected_cv!r} — code paths have diverged"
    )


# ---------------------------------------------------------------------------
# Invariant 6 — skipped status on missing artifacts
# ---------------------------------------------------------------------------


def test_backfill_skips_missing_consensus_labels(tmp_path: Path) -> None:
    """If consensus_labels.npy is missing, status is 'skipped' with a clear reason."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    # Build only the cell dir (no consensus_labels.npy)
    dataset_root = tmp_path / "fake"
    (dataset_root / "reap" / "set_A").mkdir(parents=True)
    (dataset_root / "combined_set_A").mkdir(parents=True)

    result = backfill_cell(
        dataset_root=dataset_root,
        method="reap",
        set_name="A",
        texts=["doc1", "doc2"],
        dry_run=False,
    )
    assert result["status"] == "skipped"
    assert "consensus_labels" in result["reason"]


# ---------------------------------------------------------------------------
# Invariant 7 — skipped status when texts and labels disagree on length
# ---------------------------------------------------------------------------


def test_backfill_skips_on_length_mismatch(tmp_path: Path) -> None:
    """If labels.shape[0] != len(texts) the cell is skipped with a clear reason.

    A misalignment between cached consensus_labels.npy and the live dataset
    snapshot would silently produce garbage c_v values; the script must
    refuse rather than proceed.
    """
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    dataset_root = tmp_path / "fake"
    cell_dir = dataset_root / "reap" / "set_A"
    cell_dir.mkdir(parents=True)

    # 5 labels, 3 texts — explicit mismatch.
    np.save(cell_dir / "consensus_labels.npy", np.array([0, 1, 2, 0, 1], dtype=np.int64))

    result = backfill_cell(
        dataset_root=dataset_root,
        method="reap",
        set_name="A",
        texts=["doc1", "doc2", "doc3"],
        dry_run=False,
    )
    assert result["status"] == "skipped"
    assert "labels.shape[0]" in result["reason"]


# ---------------------------------------------------------------------------
# Invariant 8 — RuntimeError when method missing from combined CSV
# ---------------------------------------------------------------------------


def test_backfill_raises_when_method_row_missing_from_csv(
    synthetic_cell: dict[str, Any],
) -> None:
    """If the target method's row is missing from combined_set_X/all_methods.csv,
    the script raises RuntimeError instead of silently corrupting the CSV."""
    from backfill_cv_coherence import backfill_cell  # type: ignore[import-not-found]

    # Rewrite the CSV without the reap row so the lookup fails.
    csv_path = synthetic_cell["combined_csv"]
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["method"] != "reap"]
        fieldnames = list(reader.fieldnames or [])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with pytest.raises(RuntimeError, match="not found"):
        backfill_cell(
            dataset_root=synthetic_cell["dataset_root"],
            method=synthetic_cell["method"],
            set_name=synthetic_cell["set_name"],
            texts=synthetic_cell["texts"],
            dry_run=False,
        )
