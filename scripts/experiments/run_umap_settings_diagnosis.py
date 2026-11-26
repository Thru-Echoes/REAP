"""Does REAP's repeatability across seed sets depend on the UMAP settings?

20 Newsgroups and calls-presidential ran on default UMAP settings (15
neighbours, minimum distance 0.1), and there REAP repeats less across seed
sets than the baselines; AI-art and Korean forest ran on searched settings,
and there REAP repeats most. Those corpora also differ in how well separated
they are, so only varying the settings on the same corpus separates the two.

For one dataset, one cell of a fixed grid (n_neighbors in {15, 30, 50} x
min_dist in {0.0, 0.1}; the number of dimensions and the K range stay at the
dataset's benchmark values), and one seed set, the benchmark's own harness
(``reap.benchmarks.run_benchmark_with_artifacts``) runs REAP and the three
baselines that share its 30 per-seed maps. The settings apply to every
per-seed map and to REAP's final map alike. REAP's final step is also rerun on
its fixed averaged matrix under ten final seeds (the diagnosis's joint arm).
No setting is adopted from this: it is a diagnosis.

``--combine`` reads every cell and set (all must exist) and reports, per
setting and method, the mean between-set ARI under both registered K rules
(per-set silhouette K as the benchmark runs it, and the protocol's K frozen
from REAP's Set A map), the fixed-K sweep (K in {5, 7, 10, 15, 20}, k-means
seed 0 with ten starts as in the existing fixed-K check), consensus
silhouette and trustworthiness, and REAP's within-matrix agreement.

Outputs under ``<research root>/results/final_step_repeatability/
umap_settings/<dataset label>/``.

Usage::

    python scripts/experiments/run_umap_settings_diagnosis.py \\
        --dataset twenty_newsgroups_reference \\
        --saved-root results/twenty_newsgroups_reference --n-neighbors 15 --min-dist 0.1 --set A
    python scripts/experiments/run_umap_settings_diagnosis.py \\
        --dataset <name> [--dataset-version v] \\
        --combine

Side effects: reads dataset snapshots and the seed manifest; writes CSV,
JSON and ``.npy`` files; CPU-heavy (30 UMAP runs per job).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import logging
import sys
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
    get_ari_summary,
    get_best_and_second,
    get_final_embedding,
    get_k_search,
    get_kmeans_labels,
    get_pairwise_ari,
)
from final_step_candidates import get_kmeans_restart_labels  # noqa: E402
from research_root import get_research_root  # noqa: E402
from run_final_step_diagnosis import (  # noqa: E402
    DEFAULT_FINAL_SEEDS,
    PROTOCOL_VERSION,
    SEED_MANIFEST_RELPATH,
    get_config,
    get_dataset,
    get_peak_rss_mb,
    get_seed_set,
)
from run_final_step_pilot import get_timed  # noqa: E402

from reap.benchmarks import run_benchmark_with_artifacts  # noqa: E402
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    write_repro_bundle,
)

logger = logging.getLogger(__name__)

N_NEIGHBORS_GRID = (15, 30, 50)
MIN_DIST_GRID = (0.0, 0.1)
METHODS = ("reap", "best_of_n", "naive_average", "procrustes")
SETS = ("A", "B", "C")
FIXED_KS = (5, 7, 10, 15, 20)
FIXED_K_SEED = 0
FIXED_K_RESTARTS = 10


def get_cell_name(n_neighbors: int, min_dist: float) -> str:
    """Folder name of one grid cell, e.g. ``nn15_md0.1``."""
    return f"nn{n_neighbors}_md{min_dist}"


def get_within_matrix_agreement(
    D: np.ndarray, config: FinalStepConfig, final_seeds: list[int]
) -> dict[str, Any]:
    """REAP's final step rerun on a fixed averaged matrix under each final seed.

    Each run seeds the final UMAP, the K search and k-means with the same
    seed (the diagnosis's joint arm). Returns the pairwise-ARI summary and
    the K each run chose.
    """
    labels: list[np.ndarray] = []
    ks: list[int] = []
    for s in final_seeds:
        emb = get_final_embedding(D, config, s)
        k = get_best_and_second(get_k_search(emb, config.k_range, s))[0]
        ks.append(k)
        labels.append(get_kmeans_labels(emb, k, s))
    return {"ari": get_ari_summary(get_pairwise_ari(labels)), "k_per_seed": ks,
            "final_seeds": final_seeds}


def run_job(
    *,
    dataset: str,
    snapshot: Any,
    config: FinalStepConfig,
    seeds: list[int],
    out_set_dir: Path,
    saved_reap_labels: np.ndarray | None,
) -> dict[str, Any]:
    """One grid cell and one seed set: harness, within-matrix rerun, outputs.

    Raises ``RuntimeError`` when ``saved_reap_labels`` is given and the
    harness's REAP labels differ from it. Side effects: writes files.
    """
    X = np.asarray(snapshot.embeddings)
    expert = snapshot.labels
    (result, artifacts), harness_time = get_timed(lambda: run_benchmark_with_artifacts(
        X=X, dataset_name=dataset, seeds=seeds, n_components=config.n_components,
        n_neighbors=config.n_neighbors, min_dist=config.min_dist, k_range=config.k_range,
        methods=list(METHODS), ground_truth_labels=expert, texts=None,
    ))
    reap = artifacts.per_method["reap"]
    if saved_reap_labels is not None and not np.array_equal(
        reap["consensus_labels"], saved_reap_labels
    ):
        raise RuntimeError("the harness's REAP labels differ from the saved benchmark labels")
    out_set_dir.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        np.save(out_set_dir / f"labels_{m}.npy", artifacts.per_method[m]["consensus_labels"])
        np.save(out_set_dir / f"embedding_{m}.npy",
                artifacts.per_method[m]["consensus_embedding"])
    within, within_time = get_timed(lambda: get_within_matrix_agreement(
        np.asarray(reap["consensus_distance_matrix"]), config, list(DEFAULT_FINAL_SEEDS)))
    record = {
        "dataset": dataset,
        "n_samples": int(X.shape[0]),
        "config": config.model_dump(mode="json"),
        "seeds": seeds,
        "reproduces_saved_reap_labels": (True if saved_reap_labels is not None else None),
        "methods": {
            m.method: {
                "k": m.best_k,
                "silhouette": m.silhouette,
                "trustworthiness": m.trustworthiness,
                "ari_expert": (float(adjusted_rand_score(
                    expert, artifacts.per_method[m.method]["consensus_labels"]))
                    if expert is not None else None),
            }
            for m in result.methods
        },
        "reap_within_matrix": within,
        "timings": {"harness": harness_time.model_dump(),
                    "within_matrix": within_time.model_dump()},
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (out_set_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n",
                                             encoding="utf-8")
    return record


def _mean_between_set(labels: dict[str, np.ndarray]) -> float:
    """Mean ARI over the three set pairs."""
    return float(np.mean([adjusted_rand_score(labels[a], labels[b])
                          for a, b in itertools.combinations(SETS, 2)]))


def run_combine(out_dir: Path) -> list[dict[str, Any]]:
    """Per setting and method: both K rules, fixed-K sweep, quality, REAP's within-matrix ARI.

    Raises ``FileNotFoundError`` when any grid cell or set is missing.
    Side effects: writes ``settings_summary.csv``, ``fixed_k.csv`` and
    ``settings_summary.md``.
    """
    rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for nn, md in itertools.product(N_NEIGHBORS_GRID, MIN_DIST_GRID):
        cell = out_dir / get_cell_name(nn, md)
        records = {}
        for s in SETS:
            path = cell / f"set_{s}" / "record.json"
            if not path.exists():
                raise FileNotFoundError(f"missing settings-diagnosis output: {path}")
            records[s] = json.loads(path.read_text(encoding="utf-8"))
        protocol_k = int(records["A"]["methods"]["reap"]["k"])
        within = [records[s]["reap_within_matrix"]["ari"]["mean"] for s in SETS]
        for m in METHODS:
            per_set = {s: np.load(cell / f"set_{s}" / f"labels_{m}.npy") for s in SETS}
            maps = {s: np.load(cell / f"set_{s}" / f"embedding_{m}.npy") for s in SETS}
            frozen = {s: get_kmeans_labels(maps[s], protocol_k, BENCHMARK_FINAL_SEED)
                      for s in SETS}
            rows.append({
                "n_neighbors": nn, "min_dist": md, "method": m,
                "k_per_set": "/".join(str(records[s]["methods"][m]["k"]) for s in SETS),
                "between_set_ari_per_set_k": _mean_between_set(per_set),
                "protocol_k": protocol_k,
                "between_set_ari_protocol_k": _mean_between_set(frozen),
                "silhouette_mean": float(np.mean(
                    [records[s]["methods"][m]["silhouette"] for s in SETS])),
                "trustworthiness_mean": float(np.mean(
                    [records[s]["methods"][m]["trustworthiness"] for s in SETS])),
                "ari_expert_mean": (float(np.mean(
                    [records[s]["methods"][m]["ari_expert"] for s in SETS]))
                    if records["A"]["methods"][m]["ari_expert"] is not None else None),
                "reap_within_matrix_ari_mean": float(np.mean(within)) if m == "reap" else None,
            })
            for k in FIXED_KS:
                fixed = {s: get_kmeans_restart_labels(maps[s], k, FIXED_K_SEED, FIXED_K_RESTARTS)
                         for s in SETS}
                fixed_rows.append({"n_neighbors": nn, "min_dist": md, "method": m, "k": k,
                                   "between_set_ari": repr(_mean_between_set(fixed))})
    with (out_dir / "settings_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in rows])
    with (out_dir / "fixed_k.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0]))
        writer.writeheader()
        writer.writerows(fixed_rows)
    (out_dir / "settings_summary.md").write_text(get_markdown(rows), encoding="utf-8")
    return rows


def get_markdown(rows: list[dict[str, Any]]) -> str:
    """One Markdown table, three decimals; full precision stays in the CSV."""
    def f3(v: Any) -> str:
        return "" if v is None else f"{v:.3f}"

    lines = [
        "| n_neighbors | min_dist | Method | K (A/B/C) | Between-set ARI, per-set K "
        "| Protocol K | Between-set ARI, protocol K | Silhouette | Trustworthiness "
        "| ARI with expert labels | REAP within-matrix ARI |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_neighbors']} | {r['min_dist']} | {r['method']} | {r['k_per_set']} | "
            f"{f3(r['between_set_ari_per_set_k'])} | {r['protocol_k']} | "
            f"{f3(r['between_set_ari_protocol_k'])} | {f3(r['silhouette_mean'])} | "
            f"{f3(r['trustworthiness_mean'])} | {f3(r['ari_expert_mean'])} | "
            f"{f3(r['reap_within_matrix_ari_mean'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run one job or combine a dataset's grid; return 0. Raises on missing input."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="UMAP-settings diagnosis of between-set ARI.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--n-neighbors", type=int, choices=N_NEIGHBORS_GRID)
    p.add_argument("--min-dist", type=float, choices=MIN_DIST_GRID)
    p.add_argument("--set", choices=SETS)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--saved-root", default=None,
                   help="Saved benchmark folder: settings come from it, and at its own "
                        "setting REAP's labels must reproduce its saved ones.")
    p.add_argument("--config-from", default=None,
                   help="A saved benchmark bundle.json to take the settings from.")
    p.add_argument("--research-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed-manifest", default=str(REPO_ROOT / SEED_MANIFEST_RELPATH))
    args = p.parse_args(argv)
    root = get_research_root(args.research_root)
    label = args.dataset + (f"_{args.dataset_version}" if args.dataset_version else "")
    out_dir = (Path(args.output_dir) if args.output_dir
               else root / "results" / "final_step_repeatability" / "umap_settings" / label)
    if args.combine:
        run_combine(out_dir)
        return 0
    if args.n_neighbors is None or args.min_dist is None or args.set is None:
        raise ValueError("a job needs --n-neighbors, --min-dist and --set")

    def _resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else root / path

    if args.saved_root:
        base = get_config(args.dataset, _resolve(args.saved_root) / "reap" / f"set_{args.set}")
    elif args.config_from:
        base = get_config(args.dataset, _resolve(args.config_from).parent)
    else:
        raise ValueError("give --saved-root or --config-from so the settings are traceable")
    config = FinalStepConfig(n_components=base.n_components, n_neighbors=args.n_neighbors,
                             min_dist=args.min_dist, k_range=base.k_range)
    saved = None
    if args.saved_root and (base.n_neighbors, base.min_dist) == (args.n_neighbors, args.min_dist):
        saved = np.load(_resolve(args.saved_root) / "reap" / f"set_{args.set}"
                        / "consensus_labels.npy")
    started = datetime.datetime.now(datetime.timezone.utc)
    snapshot = get_dataset(args.dataset, args.dataset_version)
    seeds = get_seed_set(Path(args.seed_manifest), args.set)
    out_set_dir = out_dir / get_cell_name(args.n_neighbors, args.min_dist) / f"set_{args.set}"
    record = run_job(dataset=args.dataset, snapshot=snapshot, config=config, seeds=seeds,
                     out_set_dir=out_set_dir, saved_reap_labels=saved)
    config_record: dict[str, JsonValue] = {
        "umap_and_k": config.model_dump(mode="json"),
        "grid": {"n_neighbors": list(N_NEIGHBORS_GRID), "min_dist": list(MIN_DIST_GRID)},
        "methods": list(METHODS),
        "final_seeds": list(DEFAULT_FINAL_SEEDS),
        "peak_rss_mb": record["peak_rss_mb"],
    }
    write_repro_bundle(ReproBundle(
        protocol_version=PROTOCOL_VERSION,
        code=get_code_state(REPO_ROOT),
        script="scripts/experiments/run_umap_settings_diagnosis.py",
        run_id=f"umap_settings/{label}/{get_cell_name(args.n_neighbors, args.min_dist)}/"
               f"{args.set}",
        dataset=args.dataset,
        dataset_version=snapshot.metadata.version,
        dataset_sha256=snapshot.metadata.sha256,
        seed_set=args.set,
        seeds=seeds + list(DEFAULT_FINAL_SEEDS),
        seed_manifest=SEED_MANIFEST_RELPATH,
        config=config_record,
        evaluation_mode="final_fit",
        metric_recipe_ids=["ari.between_set", "ari.between_final_step_runs",
                           "silhouette.consensus", "trustworthiness.consensus"],
        environment=get_environment_snapshot(),
        run_status="completed",
        started_at=started,
        finished_at=datetime.datetime.now(datetime.timezone.utc),
    ), out_set_dir / "bundle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
