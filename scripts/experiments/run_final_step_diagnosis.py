"""Hold REAP's averaged distance matrix fixed and rerun only its final step.

Question: REAP's averaged distance matrices barely differ between seed sets
(0.5% to 2.1% relative Frobenius distance), yet its consensus clusters repeat
less across seed sets than the baselines' on some corpora. If the final step
(one UMAP map of the averaged matrix, K by silhouette, k-means) is the source,
rerunning only that step on one fixed matrix with different seeds should
already produce disagreement of the same size as the between-set spread.

For each seed set, the averaged matrix is either read from the benchmark's
saved ``consensus_distance_matrix.npy`` or rebuilt from the set's 30 seeds
with the package's own functions. Three arms are run on it, each over the
final-step seeds (default 42 and 0 to 8, where 42 is the benchmark's seed):

- ``joint``: UMAP, K search and k-means all use seed s (what a different
  final seed would do to the benchmark);
- ``map_only``: UMAP seed s, K search and k-means seed 42;
- ``cluster_only``: UMAP seed 42, K search and k-means seed s.

Each run's labels are kept at the K it chose and at the reference K (the K
the seed-42 run chose), which separates the K rule from the map. The seed-42
run must reproduce the benchmark's saved consensus labels exactly; if it does
not, the script stops, because the reconstruction would not be the benchmark.

Outputs under ``<research root>/results/final_step_repeatability/diagnosis/
<dataset>/``: per set ``runs.csv``, ``k_scores.csv``, ``labels_chosen_k.npy``,
``labels_reference_k.npy`` and ``set_summary.json``; with ``--cross-sets``,
``cross_set.csv`` and ``cross_set_summary.json``; and one ``bundle.json``
per invocation.

Usage::

    python scripts/experiments/run_final_step_diagnosis.py \\
        --dataset calls_presidential_e5 --dataset-version 2026-11-20 \\
        --saved-root results/corporate_presidential/reap_runs/full/calls_presidential_e5 \\
        --sets A B C --cross-sets A B C

    python scripts/experiments/run_final_step_diagnosis.py \\
        --dataset twenty_newsgroups_reference \\
        --saved-root results/twenty_newsgroups_reference --sets A --recompute

``--saved-root`` is relative to the research root unless absolute.

Side effects: reads dataset snapshots, the seed manifest and saved benchmark
arrays; writes CSV, JSON and ``.npy`` files under the output directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import JsonValue
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_step import (  # noqa: E402
    BENCHMARK_FINAL_SEED,
    FinalStepConfig,
    FinalStepRun,
    get_ari_summary,
    get_best_and_second,
    get_cross_ari,
    get_file_sha256,
    get_final_embedding,
    get_k_search,
    get_kmeans_labels,
    get_pairwise_ari,
)
from research_root import get_research_root  # noqa: E402

from reap.consensus import (  # noqa: E402
    get_consensus_distance_matrix,
    get_multi_seed_embeddings,
)
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    write_repro_bundle,
)

logger = logging.getLogger(__name__)

DEFAULT_FINAL_SEEDS = [BENCHMARK_FINAL_SEED, 0, 1, 2, 3, 4, 5, 6, 7, 8]
ARMS = ("joint", "map_only", "cluster_only")
SEED_MANIFEST_RELPATH = "manuscript/seeds/seed_manifest.json"
PROTOCOL_VERSION = "1.6"


def get_arm_seed_pairs(final_seeds: list[int]) -> list[tuple[str, int, int]]:
    """List (arm, map seed, cluster seed) for every run, in output order."""
    pairs: list[tuple[str, int, int]] = []
    for s in final_seeds:
        pairs.append(("joint", s, s))
    for s in final_seeds:
        pairs.append(("map_only", s, BENCHMARK_FINAL_SEED))
    for s in final_seeds:
        pairs.append(("cluster_only", BENCHMARK_FINAL_SEED, s))
    return pairs


def get_seed_set(manifest_path: Path, seed_set: str) -> list[int]:
    """Read one seed set from the seed manifest, as the benchmark reads it.

    Side effects: reads the file.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [int(s) for s in manifest["sets"][seed_set]["seeds"]]


def get_peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (macOS reports bytes, Linux KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def get_dataset(dataset: str, dataset_version: str | None) -> Any:
    """Load a dataset snapshot by name.

    The benchmark's own loaders are used for the datasets it configures;
    any other name is read from the REAP cache with ``load_snapshot`` and
    needs ``dataset_version``. Side effects: reads the dataset cache.
    """
    from run_paper_benchmark import DATASET_CONFIGS

    if dataset in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset].loader()
    if dataset_version is None:
        raise ValueError(f"--dataset-version is required for {dataset!r}")
    from reap.datasets._cache import load_snapshot

    return load_snapshot(dataset, dataset_version)


def get_config(dataset: str, saved_set_dir: Path) -> FinalStepConfig:
    """Final-step settings: the benchmark's saved bundle first, else its config table.

    When both exist they must agree, so a run never silently uses settings
    other than those that produced the saved result.
    """
    from_bundle: FinalStepConfig | None = None
    bundle_path = saved_set_dir / "bundle.json"
    if bundle_path.exists():
        cfg = json.loads(bundle_path.read_text(encoding="utf-8"))["config"]
        from_bundle = FinalStepConfig(
            n_components=cfg["n_components"],
            n_neighbors=cfg["n_neighbors"],
            min_dist=cfg["min_dist"],
            k_range=list(cfg["k_range"]),
        )
    from run_paper_benchmark import DATASET_CONFIGS

    from_table: FinalStepConfig | None = None
    if dataset in DATASET_CONFIGS:
        c = DATASET_CONFIGS[dataset]
        from_table = FinalStepConfig(
            n_components=c.n_components,
            n_neighbors=c.n_neighbors,
            min_dist=c.min_dist,
            k_range=list(c.k_range),
        )
    if from_bundle is not None and from_table is not None and from_bundle != from_table:
        raise ValueError(f"saved bundle and benchmark config disagree for {dataset!r}")
    chosen = from_bundle or from_table
    if chosen is None:
        raise ValueError(f"no final-step settings found for {dataset!r}")
    return chosen


def get_distance_matrix(
    *,
    saved_set_dir: Path,
    recompute: bool,
    X: np.ndarray | None,
    seeds: list[int],
    config: FinalStepConfig,
    out_set_dir: Path,
) -> tuple[np.ndarray, str]:
    """Return the averaged distance matrix and a description of where it came from.

    With ``recompute``, the 30 per-seed UMAP maps are rebuilt (cosine input
    metric, the benchmark's default) and averaged; the matrix is saved to
    ``out_set_dir`` so later experiments reuse it. Otherwise the benchmark's
    saved matrix is read; a missing file raises.
    Side effects: reads or writes ``consensus_distance_matrix.npy``.
    """
    if not recompute:
        path = saved_set_dir / "consensus_distance_matrix.npy"
        if not path.exists():
            raise FileNotFoundError(f"saved averaged matrix not found: {path}")
        return np.load(path), f"saved:{path.name}:sha256={get_file_sha256(path)}"
    if X is None:
        raise ValueError("recomputing the averaged matrix needs the embeddings")
    maps = get_multi_seed_embeddings(
        X, seeds, config.n_components, config.n_neighbors, config.min_dist, metric="cosine"
    )
    D = get_consensus_distance_matrix(maps)
    out_set_dir.mkdir(parents=True, exist_ok=True)
    path = out_set_dir / "consensus_distance_matrix.npy"
    np.save(path, D)
    return D, f"recomputed:{path.name}:sha256={get_file_sha256(path)}"


def run_set(
    *,
    D: np.ndarray,
    config: FinalStepConfig,
    final_seeds: list[int],
    reference_labels: np.ndarray,
    out_set_dir: Path,
    source: str,
) -> dict[str, Any]:
    """Run all three arms on one fixed matrix and write the per-set outputs.

    Raises ``RuntimeError`` when the seed-42 joint run does not reproduce the
    saved benchmark labels exactly. Side effects: writes files under
    ``out_set_dir``.
    """
    if BENCHMARK_FINAL_SEED not in final_seeds:
        raise ValueError("the final seeds must include the benchmark seed 42")
    t0 = time.monotonic()
    maps = {s: get_final_embedding(D, config, s) for s in final_seeds}
    map_seconds = time.monotonic() - t0

    searches: dict[tuple[int, int], dict[int, float]] = {}
    plan = get_arm_seed_pairs(final_seeds)
    for _, u, c in plan:
        if (u, c) not in searches:
            searches[(u, c)] = get_k_search(maps[u], config.k_range, c)

    ref_scores = searches[(BENCHMARK_FINAL_SEED, BENCHMARK_FINAL_SEED)]
    reference_k = get_best_and_second(ref_scores)[0]

    runs: list[FinalStepRun] = []
    chosen: list[np.ndarray] = []
    at_ref_k: list[np.ndarray] = []
    for arm, u, c in plan:
        best_k, best_sil, second_k, second_sil = get_best_and_second(searches[(u, c)])
        runs.append(FinalStepRun(
            arm=arm, map_seed=u, cluster_seed=c, k=best_k, best_silhouette=best_sil,
            second_silhouette=second_sil, second_k=second_k,
        ))
        chosen.append(get_kmeans_labels(maps[u], best_k, c))
        at_ref_k.append(get_kmeans_labels(maps[u], reference_k, c))

    seed42 = chosen[0]
    exact = bool(np.array_equal(seed42, reference_labels))
    if not exact:
        out_set_dir.mkdir(parents=True, exist_ok=True)
        (out_set_dir / "reproduction_failure.json").write_text(json.dumps({
            "source": source,
            "reference_k_rerun": int(reference_k),
            "saved_k": int(len(np.unique(reference_labels))),
            "ari_rerun_vs_saved": _ari(seed42, reference_labels),
        }, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            "the seed-42 final step did not reproduce the saved consensus labels; "
            "the reconstruction is not the benchmark, so no diagnosis is written"
        )

    out_set_dir.mkdir(parents=True, exist_ok=True)
    with (out_set_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FinalStepRun.model_fields))
        writer.writeheader()
        for run in runs:
            writer.writerow(run.model_dump())
    with (out_set_dir / "k_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["map_seed", "cluster_seed", "k", "silhouette"])
        for (u, c), scores in searches.items():
            for k, sil in scores.items():
                writer.writerow([u, c, k, repr(sil)])
    np.save(out_set_dir / "labels_chosen_k.npy", np.stack(chosen))
    np.save(out_set_dir / "labels_reference_k.npy", np.stack(at_ref_k))

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        idx = [i for i, (a, _, _) in enumerate(plan) if a == arm]
        ks = [runs[i].k for i in idx]
        per_arm[arm] = {
            "k_counts": {str(k): ks.count(k) for k in sorted(set(ks))},
            "ari_chosen_k": get_ari_summary(get_pairwise_ari([chosen[i] for i in idx])),
            "ari_reference_k": get_ari_summary(get_pairwise_ari([at_ref_k[i] for i in idx])),
            "ari_to_benchmark_chosen_k": get_ari_summary(
                [
                    _ari(chosen[i], reference_labels)
                    for i in idx
                    if (runs[i].map_seed, runs[i].cluster_seed)
                    != (BENCHMARK_FINAL_SEED, BENCHMARK_FINAL_SEED)
                ]
            ),
        }
    summary = {
        "source": source,
        "n_samples": int(D.shape[0]),
        "final_seeds": final_seeds,
        "reference_k": int(reference_k),
        "reproduces_benchmark_labels_exactly": exact,
        "map_seconds": map_seconds,
        "total_seconds": time.monotonic() - t0,
        "arms": per_arm,
    }
    (out_set_dir / "set_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings."""
    return float(adjusted_rand_score(a, b))


def run_cross_sets(
    *, out_dir: Path, saved_root: Path, sets: list[str], final_seeds: list[int]
) -> dict[str, Any]:
    """Cross-set agreement of the joint arm, from the per-set outputs on disk.

    Every named set must already have its outputs; a missing one raises.
    Side effects: writes ``cross_set.csv`` and ``cross_set_summary.json``.
    """
    n = len(final_seeds)
    joint: dict[str, list[np.ndarray]] = {}
    benchmark: dict[str, np.ndarray] = {}
    for s in sets:
        path = out_dir / f"set_{s}" / "labels_chosen_k.npy"
        if not path.exists():
            raise FileNotFoundError(f"set {s} has no diagnosis outputs: {path}")
        joint[s] = list(np.load(path)[:n])
        benchmark[s] = np.load(saved_root / "reap" / f"set_{s}" / "consensus_labels.npy")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for a, b in itertools.combinations(sets, 2):
        matched = get_cross_ari(joint[a], joint[b], matched=True)
        unmatched = get_cross_ari(joint[a], joint[b], matched=False)
        bench = _ari(benchmark[a], benchmark[b])
        summary[f"{a}-{b}"] = {
            "benchmark_seed42": bench,
            "matched_final_seeds": get_ari_summary(matched),
            "unmatched_final_seeds": get_ari_summary(unmatched),
        }
        for i, v in enumerate(matched):
            rows.append({"set_a": a, "set_b": b, "pairing": "matched",
                         "seed_a": final_seeds[i], "seed_b": final_seeds[i], "ari": repr(v)})
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append({"set_a": a, "set_b": b, "pairing": "unmatched",
                                 "seed_a": final_seeds[i], "seed_b": final_seeds[j],
                                 "ari": repr(unmatched[k])})
                    k += 1
    with (out_dir / "cross_set.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["set_a", "set_b", "pairing", "seed_a",
                                               "seed_b", "ari"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "cross_set_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _get_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--saved-root", required=True,
                   help="Benchmark output folder of the dataset (holds reap/set_X/).")
    p.add_argument("--sets", nargs="*", default=[])
    p.add_argument("--cross-sets", nargs="*", default=[])
    p.add_argument("--recompute", action="store_true",
                   help="Rebuild the averaged matrix from the set's 30 seeds.")
    p.add_argument("--final-seeds", nargs="+", type=int, default=DEFAULT_FINAL_SEEDS)
    p.add_argument("--research-root", default=None)
    p.add_argument("--output-dir", default=None,
                   help="Override of the output folder (tests use this).")
    p.add_argument("--seed-manifest", default=str(REPO_ROOT / SEED_MANIFEST_RELPATH))
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the diagnosis; return 0 on success. Raises on any missing input."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _get_parser().parse_args(argv)
    started = datetime.datetime.now(datetime.timezone.utc)
    research_root = get_research_root(args.research_root)
    saved_root = Path(args.saved_root)
    if not saved_root.is_absolute():
        saved_root = research_root / saved_root
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else research_root / "results" / "final_step_repeatability" / "diagnosis" / args.dataset
    )
    snapshot = get_dataset(args.dataset, args.dataset_version)
    used_seeds: list[int] = list(args.final_seeds)
    sources: dict[str, JsonValue] = {}
    config: FinalStepConfig | None = None
    for s in args.sets:
        saved_set_dir = saved_root / "reap" / f"set_{s}"
        config = get_config(args.dataset, saved_set_dir)
        set_seeds = get_seed_set(Path(args.seed_manifest), s)
        if args.recompute:
            used_seeds += set_seeds
        D, source = get_distance_matrix(
            saved_set_dir=saved_set_dir, recompute=args.recompute,
            X=snapshot.embeddings, seeds=set_seeds, config=config,
            out_set_dir=out_dir / f"set_{s}",
        )
        ref_path = saved_set_dir / "consensus_labels.npy"
        if not ref_path.exists():
            raise FileNotFoundError(f"saved consensus labels not found: {ref_path}")
        summary = run_set(
            D=D, config=config, final_seeds=list(args.final_seeds),
            reference_labels=np.load(ref_path), out_set_dir=out_dir / f"set_{s}",
            source=source,
        )
        sources[s] = source
        logger.info("set %s: reference K %d, %.0f s", s, summary["reference_k"],
                    summary["total_seconds"])
    if args.cross_sets:
        run_cross_sets(out_dir=out_dir, saved_root=saved_root, sets=list(args.cross_sets),
                       final_seeds=list(args.final_seeds))
        if config is None:
            config = get_config(args.dataset, saved_root / "reap" / f"set_{args.cross_sets[0]}")
    if config is None:
        raise ValueError("nothing to do: give --sets and/or --cross-sets")
    peak_rss_mb = get_peak_rss_mb()
    bundle = ReproBundle(
        protocol_version=PROTOCOL_VERSION,
        code=get_code_state(REPO_ROOT),
        script="scripts/experiments/run_final_step_diagnosis.py",
        run_id=f"final_step_diagnosis/{args.dataset}/"
               f"{'+'.join(args.sets) or 'cross'}",
        dataset=args.dataset,
        dataset_version=snapshot.metadata.version,
        dataset_sha256=snapshot.metadata.sha256,
        seed_set="+".join(args.sets or args.cross_sets),
        seeds=used_seeds,
        seed_manifest=SEED_MANIFEST_RELPATH,
        config={
            "final_step": config.model_dump(mode="json"),
            "final_seeds": list(args.final_seeds),
            "arms": list(ARMS),
            "matrix_sources": sources,
            "cross_sets": list(args.cross_sets),
            "peak_rss_mb": peak_rss_mb,
        },
        evaluation_mode="final_fit",
        metric_recipe_ids=["ari.between_final_step_runs", "cluster_count.consensus"],
        environment=get_environment_snapshot(),
        run_status="completed",
        started_at=started,
        finished_at=datetime.datetime.now(datetime.timezone.utc),
    )
    name = "bundle.json" if not args.sets else f"bundle_{'+'.join(args.sets)}.json"
    write_repro_bundle(bundle, out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
