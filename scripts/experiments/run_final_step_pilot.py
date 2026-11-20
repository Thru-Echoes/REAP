"""Pilot of the final-step candidates at smoke size: feasibility, not a verdict.

Registered in the final-step amendment draft (``docs/proposals/``). For one
dataset and one seed set, the benchmark's own harness
(``reap.benchmarks.run_benchmark_with_artifacts``) runs REAP and the three
baselines that share its 30 per-seed maps (best of N, naive average,
Procrustes). Every candidate in ``final_step_candidates`` is then run on
REAP's averaged distance matrix, with REAP's seed-42 final map as the display
map, and candidate d also re-clusters each baseline's consensus map. Labels
at fixed K (5, 7, 10, 15, 20) are kept for the fixed-K check. Each
candidate's wall time and peak traced memory are recorded.

``--combine`` then reads the per-set outputs for sets A, B and C (all must
exist) and writes between-set agreement, the amendment's criteria evaluated
descriptively, a document-bootstrap interval for each candidate's margin over
the best baseline, and the fixed-K table. The pilot drops no candidate on
these numbers; only infeasible time or memory does (amendment, clause f).

Outputs under ``<research root>/results/final_step_repeatability/pilot/
<dataset>/``.

Usage::

    python scripts/experiments/run_final_step_pilot.py \\
        --dataset twenty_newsgroups_reference --set A \\
        --saved-root results/twenty_newsgroups_reference
    python scripts/experiments/run_final_step_pilot.py \\
        --dataset calls_presidential_e5 --dataset-version smoke-1000 --set A \\
        --config-from <smoke run folder>/reap/set_A/bundle.json
    python scripts/experiments/run_final_step_pilot.py --dataset <name> --combine

Side effects: reads dataset snapshots and the seed manifest; writes CSV,
JSON and ``.npy`` files under the output folder; CPU-heavy (30 UMAP runs per
set plus the candidates).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import json
import logging
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_step import (  # noqa: E402
    BENCHMARK_FINAL_SEED,
    FinalStepConfig,
    get_file_sha256,
    get_final_embedding,
    get_kmeans_labels,
)
from final_step_candidates import (  # noqa: E402
    get_average_linkage_labels,
    get_coassociation_distance,
    get_kmeans_restart_labels,
    get_kmedoids_build_order,
    get_kmedoids_labels,
    run_candidate_a1,
    run_candidate_a2,
    run_candidate_b,
    run_candidate_c,
    run_candidate_d,
)
from research_root import get_research_root  # noqa: E402
from run_final_step_diagnosis import (  # noqa: E402
    PROTOCOL_VERSION,
    SEED_MANIFEST_RELPATH,
    get_config,
    get_dataset,
    get_peak_rss_mb,
    get_seed_set,
)

from reap.benchmarks import run_benchmark_with_artifacts  # noqa: E402
from reap.evaluation import compute_continuity, compute_trustworthiness  # noqa: E402
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    write_repro_bundle,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PILOT_BASELINES = ("best_of_n", "naive_average", "procrustes")
CANDIDATES = ("a1", "a2", "b", "c", "d")
FINAL_SEEDS = [BENCHMARK_FINAL_SEED, 0, 1, 2, 3, 4, 5, 6, 7, 8]
# The K values of the existing fixed-K check, clustered the same way
# (k-means, random_state=0, ten starts) so the two tables are comparable.
FIXED_KS = [5, 7, 10, 15, 20]
FIXED_K_SEED = 0
FIXED_K_RESTARTS = 10
# Amendment margins (clause d): how much lower than REAP's current default a
# candidate's consensus silhouette and trustworthiness may be.
SILHOUETTE_MARGIN = 0.05
TRUSTWORTHINESS_MARGIN = 0.01
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261120
SETS = ("A", "B", "C")


class TimedResult(BaseModel):
    """Wall time and peak traced memory of one step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seconds: float = Field(..., description="Wall-clock seconds.")
    peak_traced_mb: float = Field(..., description="Peak memory traced by tracemalloc, MB.")


def get_timed(fn: Callable[[], T]) -> tuple[T, TimedResult]:
    """Run ``fn`` and return its value with its wall time and peak traced memory."""
    tracemalloc.start()
    t0 = time.monotonic()
    try:
        value = fn()
    finally:
        seconds = time.monotonic() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return value, TimedResult(seconds=seconds, peak_traced_mb=peak / (1024 * 1024))


def get_fixed_k_coassociation(
    maps: dict[int, np.ndarray], k: int
) -> np.ndarray:
    """Candidate b at a fixed K: k-means at K on every map, co-association cut at K."""
    runs = [get_kmeans_labels(emb, k, seed) for seed, emb in maps.items()]
    return get_average_linkage_labels(get_coassociation_distance(runs), [k])[k]


def run_one_set(
    *,
    dataset: str,
    snapshot: Any,
    config: FinalStepConfig,
    seeds: list[int],
    out_set_dir: Path,
    saved_labels: np.ndarray | None,
    saved_set_root: Path | None = None,
) -> dict[str, Any]:
    """Benchmark harness plus every candidate for one seed set; writes its outputs.

    Raises ``RuntimeError`` if REAP's seed-42 map from the harness differs
    from the one rebuilt here, or if ``saved_labels`` is given and the
    harness's REAP labels do not equal it. When ``saved_set_root`` (the saved
    benchmark folder) is given, whether each baseline's rerun labels equal its
    saved ones is recorded, not enforced: older baseline results may predate
    the current harness.
    """
    X = np.asarray(snapshot.embeddings)
    expert = snapshot.labels
    set_name = out_set_dir.name
    (result, artifacts), bench_time = get_timed(lambda: run_benchmark_with_artifacts(
        X=X, dataset_name=dataset, seeds=seeds, n_components=config.n_components,
        n_neighbors=config.n_neighbors, min_dist=config.min_dist, k_range=config.k_range,
        methods=["reap", *PILOT_BASELINES], ground_truth_labels=expert, texts=None,
    ))
    reap = artifacts.per_method["reap"]
    D = np.asarray(reap["consensus_distance_matrix"])
    display = np.asarray(reap["consensus_embedding"])
    if saved_labels is not None and not np.array_equal(reap["consensus_labels"], saved_labels):
        raise RuntimeError("the harness's REAP labels differ from the saved benchmark labels")

    labels_dir = out_set_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_set_dir / "consensus_distance_matrix.npy", D)
    for m in ("reap", *PILOT_BASELINES):
        np.save(labels_dir / f"{m}.npy", artifacts.per_method[m]["consensus_labels"])
        np.save(out_set_dir / f"embedding_{m}.npy", artifacts.per_method[m]["consensus_embedding"])

    k_range = config.k_range
    maps, maps_time = get_timed(lambda: {s: get_final_embedding(D, config, s) for s in FINAL_SEEDS})
    if not np.array_equal(maps[BENCHMARK_FINAL_SEED], display):
        raise RuntimeError("rebuilt seed-42 final map differs from the harness's REAP map")

    runs: dict[str, Any] = {}
    timings: dict[str, TimedResult] = {"benchmark_harness": bench_time, "final_maps": maps_time}
    runs["a1"], timings["a1"] = get_timed(lambda: run_candidate_a1(D, k_range))
    runs["a2"], timings["a2"] = get_timed(lambda: run_candidate_a2(D, k_range))
    runs["b"], timings["b"] = get_timed(lambda: run_candidate_b(maps, k_range))
    runs["c"], timings["c"] = get_timed(
        lambda: run_candidate_c(maps, k_range, BENCHMARK_FINAL_SEED))
    runs["d"], timings["d"] = get_timed(
        lambda: run_candidate_d(display, k_range, BENCHMARK_FINAL_SEED))
    for name in CANDIDATES:
        np.save(labels_dir / f"{name}.npy", runs[name].labels)
    for m in PILOT_BASELINES:
        emb = np.asarray(artifacts.per_method[m]["consensus_embedding"])
        restarted = run_candidate_d(emb, k_range, BENCHMARK_FINAL_SEED)
        np.save(labels_dir / f"{m}_d.npy", restarted.labels)

    n_nn = min(15, X.shape[0] - 1)
    trust = compute_trustworthiness(X, display, n_neighbors=n_nn, metric="cosine")
    continuity = compute_continuity(X, display, n_neighbors=n_nn, metric="cosine")
    rows: list[dict[str, Any]] = []
    entries: list[tuple[str, int, np.ndarray]] = [
        ("reap", int(len(np.unique(reap["consensus_labels"]))), reap["consensus_labels"])
    ] + [(name, runs[name].k, runs[name].labels) for name in CANDIDATES]
    for name, k, labels in entries:
        rows.append({
            "candidate": name,
            "k": k,
            "silhouette_map": float(silhouette_score(display, labels, metric="euclidean")),
            "silhouette_matrix": float(silhouette_score(D, labels, metric="precomputed")),
            "trustworthiness_map": trust,
            "continuity_map": continuity,
            "ari_expert": (float(adjusted_rand_score(expert, labels))
                           if expert is not None else None),
            "seconds": timings[name].seconds if name in timings else None,
            "peak_traced_mb": timings[name].peak_traced_mb if name in timings else None,
        })
    with (out_set_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_set_dir / "k_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "k_rule", "k", "score"])
        for name in CANDIDATES:
            for k, score in runs[name].k_scores.items():
                writer.writerow([name, runs[name].k_rule, k, repr(score)])

    fixed_dir = out_set_dir / "fixed_k"
    fixed_dir.mkdir(exist_ok=True)
    order = get_kmedoids_build_order(D, max(FIXED_KS))
    linkage_cuts = get_average_linkage_labels(D, FIXED_KS)
    for k in FIXED_KS:
        for m in ("reap", *PILOT_BASELINES):
            emb = np.asarray(artifacts.per_method[m]["consensus_embedding"])
            np.save(fixed_dir / f"{m}_k{k}.npy",
                    get_kmeans_restart_labels(emb, k, FIXED_K_SEED, FIXED_K_RESTARTS))
        np.save(fixed_dir / f"a1_k{k}.npy", linkage_cuts[k])
        np.save(fixed_dir / f"a2_k{k}.npy", get_kmedoids_labels(D, order[:k])[0])
        np.save(fixed_dir / f"b_k{k}.npy", get_fixed_k_coassociation(maps, k))

    record = {
        "dataset": dataset,
        "n_samples": int(X.shape[0]),
        "seeds": seeds,
        "final_seeds": FINAL_SEEDS,
        "config": config.model_dump(mode="json"),
        "distance_matrix_sha256": get_file_sha256(out_set_dir / "consensus_distance_matrix.npy"),
        "checked_against_saved_labels": saved_labels is not None,
        "baselines_match_saved_labels": {
            m: bool(np.array_equal(
                artifacts.per_method[m]["consensus_labels"],
                np.load(saved_set_root / m / set_name / "consensus_labels.npy")))
            for m in PILOT_BASELINES
        } if saved_set_root is not None else None,
        "timings": {k: v.model_dump() for k, v in timings.items()},
        "harness_metrics": {
            m.method: {"k": m.best_k, "silhouette": m.silhouette,
                       "trustworthiness": m.trustworthiness}
            for m in result.methods
        },
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (out_set_dir / "set_record.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    return record


def get_label_sources() -> dict[str, tuple[str, str]]:
    """Every label file compared between sets: name -> (harness, role)."""
    sources: dict[str, tuple[str, str]] = {"reap": ("default", "reap_default")}
    for m in PILOT_BASELINES:
        sources[m] = ("default", "baseline")
        sources[f"{m}_d"] = ("restarts10", "baseline")
    for c in ("a1", "a2", "b", "c"):
        sources[c] = ("default", "candidate")
    sources["d"] = ("restarts10", "candidate")
    return sources


def _get_bootstrap_margin(
    cand: dict[str, np.ndarray], base: dict[str, np.ndarray], n: int
) -> dict[str, float]:
    """Paired document bootstrap of the mean between-set ARI difference.

    Each resample draws N documents with replacement (one generator, fixed
    seed), recomputes the three pairwise ARIs for both label sources on the
    same documents, and takes the difference of their means. Returns the
    point estimate and the 2.5% and 97.5% percentiles.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairs = list(itertools.combinations(SETS, 2))

    def margin(idx: np.ndarray) -> float:
        c = np.mean([adjusted_rand_score(cand[a][idx], cand[b][idx]) for a, b in pairs])
        r = np.mean([adjusted_rand_score(base[a][idx], base[b][idx]) for a, b in pairs])
        return float(c - r)

    point = margin(np.arange(n))
    draws = [margin(rng.integers(0, n, size=n)) for _ in range(BOOTSTRAP_RESAMPLES)]
    return {"estimate": point, "lower_95": float(np.percentile(draws, 2.5)),
            "upper_95": float(np.percentile(draws, 97.5)), "resamples": BOOTSTRAP_RESAMPLES}


def run_combine(out_dir: Path) -> dict[str, Any]:
    """Between-set agreement, criteria and fixed-K table from sets A, B and C.

    Raises ``FileNotFoundError`` when any set's outputs are missing.
    Side effects: writes ``between_set.csv``, ``fixed_k.csv`` and
    ``pilot_summary.json``.
    """
    for s in SETS:
        if not (out_dir / f"set_{s}" / "set_record.json").exists():
            raise FileNotFoundError(f"set {s} has no pilot outputs in {out_dir}")
    sources = get_label_sources()
    labels = {name: {s: np.load(out_dir / f"set_{s}" / "labels" / f"{name}.npy") for s in SETS}
              for name in sources}
    n = int(next(iter(labels["reap"].values())).shape[0])
    rows: list[dict[str, Any]] = []
    means: dict[str, float] = {}
    for name, (harness, role) in sources.items():
        aris = []
        for a, b in itertools.combinations(SETS, 2):
            ari = float(adjusted_rand_score(labels[name][a], labels[name][b]))
            ami = float(adjusted_mutual_info_score(labels[name][a], labels[name][b]))
            aris.append(ari)
            rows.append({"source": name, "harness": harness, "role": role, "set_a": a,
                         "set_b": b, "ari": repr(ari), "ami": repr(ami)})
        means[name] = float(np.mean(aris))
    with (out_dir / "between_set.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_set = {s: {r["candidate"]: r for r in csv.DictReader(
        (out_dir / f"set_{s}" / "candidates.csv").open(encoding="utf-8"))} for s in SETS}
    criteria: dict[str, Any] = {}
    for cand in CANDIDATES:
        harness = sources[cand][0]
        baselines = {m: means[m] for m, (h, role) in sources.items()
                     if role == "baseline" and h == harness}
        best = max(baselines, key=lambda m: baselines[m])
        sil_ok = all(float(per_set[s][cand]["silhouette_map"])
                     >= float(per_set[s]["reap"]["silhouette_map"]) - SILHOUETTE_MARGIN
                     for s in SETS)
        trust_ok = all(float(per_set[s][cand]["trustworthiness_map"])
                       >= float(per_set[s]["reap"]["trustworthiness_map"]) - TRUSTWORTHINESS_MARGIN
                       for s in SETS)
        criteria[cand] = {
            "between_set_ari_mean": means[cand],
            "best_baseline": best,
            "best_baseline_ari_mean": baselines[best],
            "meets_ari_criterion": means[cand] >= baselines[best],
            "bootstrap_margin_over_best_baseline": _get_bootstrap_margin(
                labels[cand], labels[best], n),
            "meets_silhouette_margin": sil_ok,
            "meets_trustworthiness_margin": trust_ok,
            "k_per_set": {s: int(per_set[s][cand]["k"]) for s in SETS},
        }
    fixed_rows: list[dict[str, Any]] = []
    for k in FIXED_KS:
        for name in ("reap", *PILOT_BASELINES, "a1", "a2", "b"):
            runs = {s: np.load(out_dir / f"set_{s}" / "fixed_k" / f"{name}_k{k}.npy")
                    for s in SETS}
            ari = float(np.mean([adjusted_rand_score(runs[a], runs[b])
                                 for a, b in itertools.combinations(SETS, 2)]))
            fixed_rows.append({"k": k, "source": name, "between_set_ari_mean": repr(ari)})
    with (out_dir / "fixed_k.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "source", "between_set_ari_mean"])
        writer.writeheader()
        writer.writerows(fixed_rows)
    records = {s: json.loads((out_dir / f"set_{s}" / "set_record.json").read_text())
               for s in SETS}
    summary = {
        "between_set_ari_mean": means,
        "criteria": criteria,
        "reap_default_k_per_set": {s: int(per_set[s]["reap"]["k"]) for s in SETS},
        "timings_per_set": {s: records[s]["timings"] for s in SETS},
        "peak_rss_mb_per_set": {s: records[s]["peak_rss_mb"] for s in SETS},
        "margins": {"silhouette": SILHOUETTE_MARGIN, "trustworthiness": TRUSTWORTHINESS_MARGIN},
    }
    (out_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run one set or combine three; return 0. Raises on any missing input."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Pilot of the final-step candidates.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--set", default=None, choices=SETS)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--saved-root", default=None,
                   help="Benchmark output folder; REAP's labels must reproduce its saved ones.")
    p.add_argument("--config-from", default=None,
                   help="A saved benchmark bundle.json whose config the pilot must use.")
    p.add_argument("--research-root", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed-manifest", default=str(REPO_ROOT / SEED_MANIFEST_RELPATH))
    args = p.parse_args(argv)
    if (args.set is None) == (not args.combine):
        raise ValueError("give exactly one of --set or --combine")
    started = datetime.datetime.now(datetime.timezone.utc)
    root = get_research_root(args.research_root)
    label = args.dataset + (f"_{args.dataset_version}" if args.dataset_version else "")
    out_dir = (Path(args.output_dir) if args.output_dir
               else root / "results" / "final_step_repeatability" / "pilot" / label)
    if args.combine:
        run_combine(out_dir)
        return 0

    def _resolve(path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else root / path

    if args.config_from:
        config = get_config(args.dataset, _resolve(args.config_from).parent)
    elif args.saved_root:
        config = get_config(args.dataset, _resolve(args.saved_root) / "reap" / f"set_{args.set}")
    else:
        raise ValueError("give --saved-root or --config-from so the settings are traceable")
    saved_labels = None
    if args.saved_root:
        saved_labels = np.load(
            _resolve(args.saved_root) / "reap" / f"set_{args.set}" / "consensus_labels.npy")
    snapshot = get_dataset(args.dataset, args.dataset_version)
    seeds = get_seed_set(Path(args.seed_manifest), args.set)
    record = run_one_set(dataset=args.dataset, snapshot=snapshot, config=config, seeds=seeds,
                         out_set_dir=out_dir / f"set_{args.set}", saved_labels=saved_labels,
                         saved_set_root=_resolve(args.saved_root) if args.saved_root else None)
    config_record: dict[str, JsonValue] = {
        "final_step": config.model_dump(mode="json"),
        "final_seeds": list(FINAL_SEEDS),
        "candidates": list(CANDIDATES),
        "baselines": list(PILOT_BASELINES),
        "fixed_ks": list(FIXED_KS),
        "peak_rss_mb": record["peak_rss_mb"],
    }
    write_repro_bundle(ReproBundle(
        protocol_version=PROTOCOL_VERSION,
        code=get_code_state(REPO_ROOT),
        script="scripts/experiments/run_final_step_pilot.py",
        run_id=f"final_step_pilot/{label}/{args.set}",
        dataset=args.dataset,
        dataset_version=snapshot.metadata.version,
        dataset_sha256=snapshot.metadata.sha256,
        seed_set=args.set,
        seeds=seeds + list(FINAL_SEEDS),
        seed_manifest=SEED_MANIFEST_RELPATH,
        config=config_record,
        evaluation_mode="final_fit",
        metric_recipe_ids=["ari.between_set", "silhouette.consensus", "trustworthiness.consensus"],
        environment=get_environment_snapshot(),
        run_status="completed",
        started_at=started,
        finished_at=datetime.datetime.now(datetime.timezone.utc),
    ), out_dir / f"set_{args.set}" / "bundle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
