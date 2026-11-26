"""Tests for the final-step diagnosis experiment (scripts/experiments/).

The diagnosis is only meaningful if its seed-42 run *is* the benchmark's final
step, so the end-to-end test builds a small averaged matrix, runs the
benchmark's own final step on it, saves the result in the benchmark's layout,
and checks that the diagnosis reproduces it, writes every output, and stops
when the saved labels do not match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

import final_step  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_final_step_diagnosis as diag  # noqa: E402  # pyright: ignore[reportMissingImports]
import run_paper_benchmark  # noqa: E402  # pyright: ignore[reportMissingImports]

from reap.benchmarks import _cluster_embedding  # noqa: E402
from reap.clustering import find_best_k  # noqa: E402
from reap.consensus import (  # noqa: E402
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.repro_bundle import read_repro_bundle  # noqa: E402

# The benchmark's own synthetic-dataset settings, so the config guard passes.
SYNTH = run_paper_benchmark.DATASET_CONFIGS["synthetic"]


def test_arm_seed_pairs_cover_each_arm_once_per_seed() -> None:
    """Each arm gets one run per final seed; only the varied seed changes."""
    pairs = diag.get_arm_seed_pairs([42, 0, 1])
    assert [a for a, _, _ in pairs] == ["joint"] * 3 + ["map_only"] * 3 + ["cluster_only"] * 3
    assert pairs[3:6] == [("map_only", 42, 42), ("map_only", 0, 42), ("map_only", 1, 42)]
    assert pairs[6:9] == [("cluster_only", 42, 42), ("cluster_only", 42, 0),
                          ("cluster_only", 42, 1)]


def test_best_and_second_breaks_ties_toward_smaller_k() -> None:
    """Ties resolve as find_best_k resolves them (first K in insertion order)."""
    assert final_step.get_best_and_second({5: 0.4, 6: 0.4, 7: 0.1}) == (5, 0.4, 6, 0.4)


def test_cross_ari_pair_counts() -> None:
    """Matched gives n pairs, unmatched gives n * (n - 1)."""
    runs = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([1, 1, 0, 0])]
    assert len(final_step.get_cross_ari(runs, runs, matched=True)) == 3
    assert len(final_step.get_cross_ari(runs, runs, matched=False)) == 6
    assert final_step.get_cross_ari(runs, runs, matched=True)[0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def saved_benchmark(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved REAP benchmark result for sets A and B of the synthetic dataset.

    Built with the benchmark's own final-step functions and default seed 42,
    in the benchmark's on-disk layout (reap/set_X/ with the averaged matrix,
    consensus labels and a bundle carrying the config).
    """
    root = tmp_path_factory.mktemp("saved")
    X = SYNTH.loader().embeddings
    for name, seeds in (("A", [1, 2, 3]), ("B", [4, 5, 6])):
        maps = get_multi_seed_embeddings(
            X, seeds, SYNTH.n_components, SYNTH.n_neighbors, SYNTH.min_dist
        )
        D = get_consensus_distance_matrix(maps)
        emb, _ = get_consensus_embedding(D, SYNTH.n_components, SYNTH.n_neighbors, SYNTH.min_dist)
        k = find_best_k(emb, SYNTH.k_range).best_k
        labels = _cluster_embedding(emb, k)
        set_dir = root / "reap" / f"set_{name}"
        set_dir.mkdir(parents=True)
        np.save(set_dir / "consensus_distance_matrix.npy", D)
        np.save(set_dir / "consensus_labels.npy", labels)
        (set_dir / "bundle.json").write_text(json.dumps({"config": {
            "n_components": SYNTH.n_components, "n_neighbors": SYNTH.n_neighbors,
            "min_dist": SYNTH.min_dist, "k_range": SYNTH.k_range,
        }}))
    return root


def test_diagnosis_end_to_end(saved_benchmark: Path, tmp_path: Path) -> None:
    """The diagnosis reproduces the saved labels and writes complete outputs."""
    out = tmp_path / "out"
    rc = diag.main([
        "--dataset", "synthetic", "--saved-root", str(saved_benchmark),
        "--sets", "A", "B", "--cross-sets", "A", "B", "--final-seeds", "42", "0", "1",
        "--research-root", str(tmp_path), "--output-dir", str(out),
    ])
    assert rc == 0
    for s in ("A", "B"):
        summary = json.loads((out / f"set_{s}" / "set_summary.json").read_text())
        assert summary["reproduces_benchmark_labels_exactly"] is True
        for arm in diag.ARMS:
            assert summary["arms"][arm]["ari_chosen_k"]["n_pairs"] == 3
            assert summary["arms"][arm]["ari_to_benchmark_chosen_k"]["n_pairs"] == 2
        labels = np.load(out / f"set_{s}" / "labels_chosen_k.npy")
        assert labels.shape == (9, 400)
        saved = np.load(saved_benchmark / "reap" / f"set_{s}" / "consensus_labels.npy")
        assert np.array_equal(labels[0], saved)
    cross = json.loads((out / "cross_set_summary.json").read_text())
    saved_a = np.load(saved_benchmark / "reap" / "set_A" / "consensus_labels.npy")
    saved_b = np.load(saved_benchmark / "reap" / "set_B" / "consensus_labels.npy")
    from sklearn.metrics import adjusted_rand_score

    assert cross["A-B"]["benchmark_seed42"] == pytest.approx(
        adjusted_rand_score(saved_a, saved_b)
    )
    assert cross["A-B"]["matched_final_seeds"]["n_pairs"] == 3
    assert cross["A-B"]["unmatched_final_seeds"]["n_pairs"] == 6
    bundle = read_repro_bundle(out / "bundle_A+B.json")
    assert bundle.run_status == "completed"
    assert bundle.config["final_seeds"] == [42, 0, 1]


def test_diagnosis_stops_when_saved_labels_do_not_reproduce(
    saved_benchmark: Path, tmp_path: Path
) -> None:
    """Tampered saved labels make the run raise and leave a failure record."""
    tampered = tmp_path / "tampered"
    set_dir = tampered / "reap" / "set_A"
    set_dir.mkdir(parents=True)
    src = saved_benchmark / "reap" / "set_A"
    for name in ("consensus_distance_matrix.npy", "bundle.json"):
        (set_dir / name).write_bytes((src / name).read_bytes())
    labels = np.load(src / "consensus_labels.npy")
    np.save(set_dir / "consensus_labels.npy", np.roll(labels, 7))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="did not reproduce"):
        diag.main([
            "--dataset", "synthetic", "--saved-root", str(tampered), "--sets", "A",
            "--final-seeds", "42", "0", "--research-root", str(tmp_path),
            "--output-dir", str(out),
        ])
    assert (out / "set_A" / "reproduction_failure.json").exists()
    assert not (out / "set_A" / "set_summary.json").exists()
