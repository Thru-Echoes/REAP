"""Pilot of the final-step candidates at smoke size: feasibility, not a verdict.

Registered in ``docs/proposals/2026-11-protocol-v1.8-amendment.md`` (clauses
a to h). For one dataset and one seed set (``--set``), the benchmark's own
harness (``reap.benchmarks.run_benchmark_with_artifacts``) runs REAP and the
three baselines that share its 30 per-seed maps (best of N, naive average,
Procrustes). REAP's final map is rebuilt from its averaged matrix under each
of the ten final seeds (42, 0 to 8), and every candidate in
``final_step_candidates`` is run under rule S (per-set silhouette K), with the
final seed entering each one as clause (c) states. Each baseline is clustered
under every final seed too, with one k-means start (the harness) and with
ten (candidate d's harness). Each candidate's wall time and peak traced
memory are recorded.

``--combine`` reads sets A, B and C (all must exist), adds rule F (K frozen
at REAP's Set A K under rule S at seed 42) for every source, and writes
between-set agreement per rule and final seed, the amendment's criteria
evaluated descriptively, a paired document-bootstrap interval for each
candidate's margin over the best baseline, and the fixed-K sweep. The pilot
drops no candidate on these numbers (clause g).

Outputs under ``<research root>/results/final_step_repeatability/pilot/
<dataset label>/``.

Usage::

    python scripts/experiments/run_final_step_pilot.py \\
        --dataset twenty_newsgroups_reference --set A \\
        --saved-root results/twenty_newsgroups_reference
    python scripts/experiments/run_final_step_pilot.py \\
        --dataset calls_presidential_e5 --dataset-version smoke-1000 --set A \\
        --config-from <smoke run folder>/reap/set_A/bundle.json
    python scripts/experiments/run_final_step_pilot.py --dataset <name> \\
        [--dataset-version v] --combine

Side effects: reads dataset snapshots and the seed manifest; writes CSV,
JSON and ``.npy``/``.npz`` files; CPU-heavy (30 UMAP runs per set plus ten
final maps and the candidates).
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
from sklearn.metrics import adjusted_rand_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_step import (  # noqa: E402
    BENCHMARK_FINAL_SEED,
    FinalStepConfig,
    get_best_and_second,
    get_final_embedding,
    get_k_search,
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
# The K values and clustering of the existing fixed-K check (k-means,
# random_state=0, ten starts), so the two tables are comparable.
FIXED_KS = [5, 7, 10, 15, 20]
FIXED_K_SEED = 0
FIXED_K_RESTARTS = 10
# Amendment clause (e): how far below REAP's current final step a candidate's
# consensus silhouette and trustworthiness may fall. Judgment values.
SILHOUETTE_MARGIN = 0.05
TRUSTWORTHINESS_MARGIN = 0.01
# Amendment clause (h).
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261120
SETS = ("A", "B", "C")
RULES = ("S", "F")


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


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Save named arrays to one ``.npz`` file. Side effects: writes ``path``."""
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


def get_label_sources() -> dict[str, str]:
    """Every labeling compared between sets: name -> clustering harness.

    ``single`` is one k-means start (the benchmark's harness); ``ten`` is ten
    starts (candidate d's harness). Candidates a1, a2, b and c are compared
    against the baselines under ``single``.
    """
    sources = {"reap": "single", "a1": "single", "a2": "single", "b": "single",
               "c": "single", "d": "ten"}
    for m in PILOT_BASELINES:
        sources[m] = "single"
        sources[f"{m}_d"] = "ten"
    return sources


def _get_silhouette_k_labels(embedding: np.ndarray, k_range: list[int], seed: int,
                             ten_starts: bool) -> tuple[int, np.ndarray]:
    """Rule S on one map: K by silhouette, then k-means at K (one or ten starts)."""
    if ten_starts:
        result = run_candidate_d(embedding, k_range, seed)
        return result.k, result.labels
    k = get_best_and_second(get_k_search(embedding, k_range, seed))[0]
    return k, get_kmeans_labels(embedding, k, seed)


def get_rule_s_labels(
    *, D: np.ndarray, maps: dict[int, np.ndarray], baseline_maps: dict[str, np.ndarray],
    k_range: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, TimedResult]]:
    """Rule-S labels of every source under every final seed, shape (10, N) each.

    Returns (labels, K per seed, timings of the candidates).
    """
    labels: dict[str, list[np.ndarray]] = {name: [] for name in get_label_sources()}
    ks: dict[str, list[int]] = {name: [] for name in get_label_sources()}
    timings: dict[str, TimedResult] = {}
    a1, timings["a1"] = get_timed(lambda: run_candidate_a1(D, k_range))
    a2, timings["a2"] = get_timed(lambda: run_candidate_a2(D, k_range))
    b, timings["b"] = get_timed(lambda: run_candidate_b(maps, k_range))
    c, timings["c"] = get_timed(lambda: run_candidate_c(maps, k_range, BENCHMARK_FINAL_SEED))
    _, timings["d"] = get_timed(lambda: run_candidate_d(maps[BENCHMARK_FINAL_SEED], k_range,
                                                        BENCHMARK_FINAL_SEED))
    for s in FINAL_SEEDS:
        for name, result in (("a1", a1), ("a2", a2), ("b", b)):
            labels[name].append(result.labels)
            ks[name].append(result.k)
        labels["c"].append(get_kmeans_labels(maps[s], c.k, s))
        ks["c"].append(c.k)
        for name, emb, ten in [("reap", maps[s], False), ("d", maps[s], True)] + [
            (m, baseline_maps[m], False) for m in PILOT_BASELINES
        ] + [(f"{m}_d", baseline_maps[m], True) for m in PILOT_BASELINES]:
            k, lab = _get_silhouette_k_labels(emb, k_range, s, ten)
            labels[name].append(lab)
            ks[name].append(k)
    return {n: np.stack(v) for n, v in labels.items()}, ks, timings


def get_rule_f_labels(
    *, D: np.ndarray, maps: dict[int, np.ndarray], baseline_maps: dict[str, np.ndarray],
    k: int,
) -> dict[str, np.ndarray]:
    """Rule-F labels (K frozen at ``k``) of every source except c, shape (10, N) each."""
    order = get_kmedoids_build_order(D, k)
    a1 = get_average_linkage_labels(D, [k])[k]
    a2 = get_kmedoids_labels(D, order)[0]
    b = get_average_linkage_labels(get_coassociation_distance(
        [get_kmeans_labels(maps[s], k, s) for s in FINAL_SEEDS]), [k])[k]
    out: dict[str, list[np.ndarray]] = {"reap": [], "a1": [], "a2": [], "b": [], "d": []}
    for m in PILOT_BASELINES:
        out[m] = []
        out[f"{m}_d"] = []
    for s in FINAL_SEEDS:
        out["reap"].append(get_kmeans_labels(maps[s], k, s))
        out["d"].append(get_kmeans_restart_labels(maps[s], k, s))
        out["a1"].append(a1)
        out["a2"].append(a2)
        out["b"].append(b)
        for m in PILOT_BASELINES:
            out[m].append(get_kmeans_labels(baseline_maps[m], k, s))
            out[f"{m}_d"].append(get_kmeans_restart_labels(baseline_maps[m], k, s))
    return {n: np.stack(v) for n, v in out.items()}


def run_one_set(
    *, dataset: str, snapshot: Any, config: FinalStepConfig, seeds: list[int],
    out_set_dir: Path, saved_labels: np.ndarray | None,
) -> dict[str, Any]:
    """Harness, ten final maps and rule-S labels for one seed set; writes outputs.

    Raises ``RuntimeError`` if the rebuilt seed-42 final map differs from the
    harness's REAP map, or if ``saved_labels`` is given and the harness's
    REAP labels differ from it.
    """
    X = np.asarray(snapshot.embeddings)
    (result, artifacts), harness_time = get_timed(lambda: run_benchmark_with_artifacts(
        X=X, dataset_name=dataset, seeds=seeds, n_components=config.n_components,
        n_neighbors=config.n_neighbors, min_dist=config.min_dist, k_range=config.k_range,
        methods=["reap", *PILOT_BASELINES], ground_truth_labels=snapshot.labels, texts=None,
    ))
    reap = artifacts.per_method["reap"]
    D = np.asarray(reap["consensus_distance_matrix"])
    if saved_labels is not None and not np.array_equal(reap["consensus_labels"], saved_labels):
        raise RuntimeError("the harness's REAP labels differ from the saved benchmark labels")
    maps, maps_time = get_timed(lambda: {s: get_final_embedding(D, config, s)
                                         for s in FINAL_SEEDS})
    if not np.array_equal(maps[BENCHMARK_FINAL_SEED], np.asarray(reap["consensus_embedding"])):
        raise RuntimeError("rebuilt seed-42 final map differs from the harness's REAP map")
    if not np.array_equal(get_kmeans_labels(maps[BENCHMARK_FINAL_SEED], int(
            len(np.unique(reap["consensus_labels"]))), BENCHMARK_FINAL_SEED),
            reap["consensus_labels"]):
        raise RuntimeError("seed-42 rule-S labels differ from the harness's REAP labels")
    baseline_maps = {m: np.asarray(artifacts.per_method[m]["consensus_embedding"])
                     for m in PILOT_BASELINES}
    labels, ks, timings = get_rule_s_labels(D=D, maps=maps, baseline_maps=baseline_maps,
                                            k_range=config.k_range)
    out_set_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_set_dir / "consensus_distance_matrix.npy", D)
    _save_arrays(out_set_dir / "final_maps.npz", {str(s): maps[s] for s in FINAL_SEEDS})
    _save_arrays(out_set_dir / "baseline_maps.npz", baseline_maps)
    _save_arrays(out_set_dir / "labels_rule_S.npz", labels)
    record = {
        "dataset": dataset, "n_samples": int(X.shape[0]), "seeds": seeds,
        "final_seeds": FINAL_SEEDS, "config": config.model_dump(mode="json"),
        "checked_against_saved_labels": saved_labels is not None,
        "k_rule_S_per_seed": ks,
        "quality_rule_S_seed42": get_quality(X, D, maps[BENCHMARK_FINAL_SEED],
                                             {n: labels[n][0] for n in ("reap", *CANDIDATES)},
                                             snapshot.labels),
        "timings": {"benchmark_harness": harness_time.model_dump(),
                    "final_maps": maps_time.model_dump(),
                    **{k: v.model_dump() for k, v in timings.items()}},
        "harness_metrics": {m.method: {"k": m.best_k, "silhouette": m.silhouette,
                                       "trustworthiness": m.trustworthiness}
                            for m in result.methods},
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (out_set_dir / "set_record.json").write_text(json.dumps(record, indent=2) + "\n",
                                                 encoding="utf-8")
    return record


def get_quality(X: np.ndarray, D: np.ndarray, display: np.ndarray,
                labels: dict[str, np.ndarray], expert: np.ndarray | None) -> dict[str, Any]:
    """Consensus silhouette (display map and D), trustworthiness, continuity, expert ARI.

    Trustworthiness and continuity belong to the display map and are the same
    for every labeling; they are reported per labeling for the criteria.
    """
    n_nn = min(15, X.shape[0] - 1)
    trust = compute_trustworthiness(X, display, n_neighbors=n_nn, metric="cosine")
    continuity = compute_continuity(X, display, n_neighbors=n_nn, metric="cosine")
    return {name: {
        "k": int(len(np.unique(lab))),
        "silhouette_map": float(silhouette_score(display, lab, metric="euclidean")),
        "silhouette_matrix": float(silhouette_score(D, lab, metric="precomputed")),
        "trustworthiness_map": trust, "continuity_map": continuity,
        "ari_expert": float(adjusted_rand_score(expert, lab)) if expert is not None else None,
    } for name, lab in labels.items()}


def _get_between_set(runs: dict[str, np.ndarray], idx: np.ndarray | None = None) -> list[float]:
    """Per final seed, the mean ARI over the three set pairs (optionally on documents idx)."""
    values = []
    for i in range(len(FINAL_SEEDS)):
        aris = []
        for a, b in itertools.combinations(SETS, 2):
            la, lb = runs[a][i], runs[b][i]
            if idx is not None:
                la, lb = la[idx], lb[idx]
            aris.append(adjusted_rand_score(la, lb))
        values.append(float(np.mean(aris)))
    return values


def _get_bootstrap_margin(cand: dict[str, np.ndarray], base: dict[str, np.ndarray],
                          n: int) -> dict[str, float]:
    """Paired document bootstrap of the seed-averaged between-set ARI margin (clause h)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def margin(idx: np.ndarray | None) -> float:
        return float(np.mean(_get_between_set(cand, idx)) - np.mean(_get_between_set(base, idx)))

    draws = [margin(rng.integers(0, n, size=n)) for _ in range(BOOTSTRAP_RESAMPLES)]
    return {"estimate": margin(None), "lower_95": float(np.percentile(draws, 2.5)),
            "upper_95": float(np.percentile(draws, 97.5)), "resamples": BOOTSTRAP_RESAMPLES}


def _load_set(out_dir: Path, s: str) -> dict[str, Any]:
    """Read one set's saved outputs; raises ``FileNotFoundError`` when missing."""
    d = out_dir / f"set_{s}"
    if not (d / "set_record.json").exists():
        raise FileNotFoundError(f"set {s} has no pilot outputs in {out_dir}")
    final_maps = np.load(d / "final_maps.npz")
    return {
        "record": json.loads((d / "set_record.json").read_text(encoding="utf-8")),
        "D": np.load(d / "consensus_distance_matrix.npy"),
        "maps": {s_: np.asarray(final_maps[str(s_)]) for s_ in FINAL_SEEDS},
        "baseline_maps": dict(np.load(d / "baseline_maps.npz")),
        "S": dict(np.load(d / "labels_rule_S.npz")),
    }


def get_criteria(between: dict[str, dict[str, list[float]]], labels: dict[str, dict[str, Any]],
                 quality: dict[str, dict[str, Any]], n: int) -> dict[str, Any]:
    """Clause (e) criteria per candidate and K rule, evaluated descriptively."""
    sources = get_label_sources()
    out: dict[str, Any] = {}
    for rule in RULES:
        for cand in CANDIDATES:
            if cand not in between[rule]:
                continue
            harness = sources[cand]
            base = {m: float(np.mean(between[rule][m])) for m, h in sources.items()
                    if h == harness and m not in CANDIDATES and m != "reap"}
            best = max(base, key=lambda m: base[m])
            mean_c = float(np.mean(between[rule][cand]))
            q = quality[rule]
            out[f"{cand}/{rule}"] = {
                "between_set_ari_mean_over_seeds": mean_c,
                "between_set_ari_min_over_seeds": min(between[rule][cand]),
                "best_baseline": best, "best_baseline_ari": base[best],
                "meets_repeatability": mean_c >= base[best],
                "bootstrap_margin": _get_bootstrap_margin(labels[rule][cand],
                                                          labels[rule][best], n),
                "meets_silhouette_margin": all(
                    q[s][cand]["silhouette_map"]
                    >= q[s]["reap"]["silhouette_map"] - SILHOUETTE_MARGIN for s in SETS),
                "meets_trustworthiness_margin": all(
                    q[s][cand]["trustworthiness_map"]
                    >= q[s]["reap"]["trustworthiness_map"] - TRUSTWORTHINESS_MARGIN
                    for s in SETS),
            }
    return out


def run_combine(out_dir: Path, expert: np.ndarray | None, X: np.ndarray) -> dict[str, Any]:
    """Both K rules, between-set agreement, criteria and the fixed-K sweep.

    Side effects: writes ``between_set.csv``, ``fixed_k.csv`` and
    ``pilot_summary.json``.
    """
    sets = {s: _load_set(out_dir, s) for s in SETS}
    k_frozen = int(sets["A"]["record"]["k_rule_S_per_seed"]["reap"][0])
    labels: dict[str, dict[str, Any]] = {"S": {}, "F": {}}
    for s in SETS:
        f_labels = get_rule_f_labels(D=sets[s]["D"], maps=sets[s]["maps"],
                                     baseline_maps=sets[s]["baseline_maps"], k=k_frozen)
        for name in get_label_sources():
            labels["S"].setdefault(name, {})[s] = sets[s]["S"][name]
            if name in f_labels:
                labels["F"].setdefault(name, {})[s] = f_labels[name]
    between = {rule: {name: _get_between_set(runs) for name, runs in labels[rule].items()}
               for rule in RULES}
    quality: dict[str, dict[str, Any]] = {
        "S": {s: sets[s]["record"]["quality_rule_S_seed42"] for s in SETS},
        "F": {s: get_quality(X, sets[s]["D"], sets[s]["maps"][BENCHMARK_FINAL_SEED],
                             {n: labels["F"][n][s][0] for n in ("reap", "a1", "a2", "b", "d")},
                             expert) for s in SETS},
    }
    n = int(X.shape[0])
    with (out_dir / "between_set.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "source", "final_seed", "between_set_ari"])
        for rule in RULES:
            for name, values in between[rule].items():
                for seed, v in zip(FINAL_SEEDS, values):
                    writer.writerow([rule, name, seed, repr(v)])
    fixed_rows = get_fixed_k_rows(sets)
    with (out_dir / "fixed_k.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "source", "between_set_ari"])
        writer.writeheader()
        writer.writerows(fixed_rows)
    summary = {
        "frozen_k": k_frozen,
        "between_set": {rule: {name: {"mean_over_seeds": float(np.mean(v)), "min": min(v),
                                      "max": max(v), "seed42": v[0]}
                               for name, v in between[rule].items()} for rule in RULES},
        "criteria": get_criteria(between, labels, quality, n),
        "quality": quality,
        "k_rule_S_per_seed": {s: sets[s]["record"]["k_rule_S_per_seed"] for s in SETS},
        "timings_per_set": {s: sets[s]["record"]["timings"] for s in SETS},
        "peak_rss_mb_per_set": {s: sets[s]["record"]["peak_rss_mb"] for s in SETS},
        "margins": {"silhouette": SILHOUETTE_MARGIN, "trustworthiness": TRUSTWORTHINESS_MARGIN},
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    return summary


def get_fixed_k_rows(sets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed-K sweep: REAP's seed-42 map, the baselines' maps, a1, a2 and b at each K."""
    rows: list[dict[str, Any]] = []
    for k in FIXED_KS:
        runs: dict[str, dict[str, np.ndarray]] = {}
        for s, data in sets.items():
            maps = {"reap": data["maps"][BENCHMARK_FINAL_SEED], **data["baseline_maps"]}
            for name, emb in maps.items():
                runs.setdefault(name, {})[s] = get_kmeans_restart_labels(
                    emb, k, FIXED_K_SEED, FIXED_K_RESTARTS)
            runs.setdefault("a1", {})[s] = get_average_linkage_labels(data["D"], [k])[k]
            runs.setdefault("a2", {})[s] = get_kmedoids_labels(
                data["D"], get_kmedoids_build_order(data["D"], k))[0]
            runs.setdefault("b", {})[s] = get_average_linkage_labels(get_coassociation_distance(
                [get_kmeans_labels(data["maps"][x], k, x) for x in FINAL_SEEDS]), [k])[k]
        for name, by_set in runs.items():
            ari = float(np.mean([adjusted_rand_score(by_set[a], by_set[b])
                                 for a, b in itertools.combinations(SETS, 2)]))
            rows.append({"k": k, "source": name, "between_set_ari": repr(ari)})
    return rows


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
    snapshot = get_dataset(args.dataset, args.dataset_version)
    if args.combine:
        run_combine(out_dir, snapshot.labels, np.asarray(snapshot.embeddings))
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
    seeds = get_seed_set(Path(args.seed_manifest), args.set)
    record = run_one_set(dataset=args.dataset, snapshot=snapshot, config=config, seeds=seeds,
                         out_set_dir=out_dir / f"set_{args.set}", saved_labels=saved_labels)
    config_record: dict[str, JsonValue] = {
        "final_step": config.model_dump(mode="json"), "final_seeds": list(FINAL_SEEDS),
        "candidates": list(CANDIDATES), "baselines": list(PILOT_BASELINES),
        "fixed_ks": list(FIXED_KS), "peak_rss_mb": record["peak_rss_mb"],
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
