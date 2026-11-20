"""Fit a REAP map on a reference corpus and project probe corpora into it.

The general form of REAP's out-of-sample use: build the consensus map on the
reference documents only, train the projection head on that map, then place
other document sets ("probes") into it with the head, without letting them
touch the map, its K, or the head's training. It works on any embeddings:
the reference and each probe are read from an ``.npz`` file (one 1-d id
array and one 2-d vector array) or a folder holding ``emb.npy`` and
``ids.txt``; a passage table (``.csv`` or ``.jsonl`` with an ``id`` column)
selects the reference rows with ``--where column=value[,value...]``.

One run is one reference variant and one seed set:

1. the per-seed UMAP maps of the seed set, their averaged distance matrix,
   and REAP's final map at seed 42 (the package's own functions);
2. K: chosen by silhouette on this run's seed-42 map (``--k auto``; on Set A
   this is protocol §9's rule), or read from another run's record
   (``--k-from``), so sets B and C use Set A's K;
3. consensus labels (k-means at K, seed 42) and their centroids;
4. the final step repeated under each final seed (map, k-means at the same
   K, and the K silhouette would pick), with the pairwise ARI between
   repeats, because REAP's labels move with the final seed;
5. the projection head trained on the reference (5-fold cross-validation
   plus a final fit), with the settings of ``run_projection_head_oos.py``;
6. every probe projected by the head, then assigned to a reference cluster
   by nearest centroid and by a vote of its 15 nearest reference passages in
   the consensus space; the same again after shifting the probe vectors to
   the reference mean (a sensitivity check);
7. a 2-D layout of the consensus space for plots only (UMAP fit on the
   reference coordinates; probes placed with its ``transform``);
8. for every probe, its distance to the nearest reference passage and
   whether REAP's conformal filter keeps it, both strictly (inside its
   reference cluster) and with the package's default location correction.
   Probes far outside the reference are extrapolated by the head, so their
   cluster assignment is not meaningful on its own.

Outputs in ``--output-dir``: ``reference.npz``, ``centroids.npy``,
``probe_<name>.npz``, ``head.pt`` (head weights), ``head_cv.json``, the
long tables ``reference_labels.csv``, ``probe_assignments.csv``,
``clusters.csv``, ``layout_2d.csv`` and ``probe_placement.csv``,
``run_record.json`` and ``bundle.json``.

Usage::

    python scripts/run_reference_probes.py --label presidential_ps_e5 \\
        --reference <embeddings> --reference-table <passages.csv> --where corner=PS \\
        --probe calls=<embeddings> --probe reports=<embeddings> \\
        --seed-set A --output-dir <folder>
    python scripts/run_reference_probes.py ... --seed-set B --k-from <set A folder>/run_record.json

Side effects: reads the input files and the seed manifest; writes the output
folder; CPU-heavy (one UMAP run per seed plus the final maps, and the head).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reap.clustering import find_best_k  # noqa: E402
from reap.consensus import (  # noqa: E402
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    write_repro_bundle,
)

logger = logging.getLogger(__name__)

SEED_MANIFEST_RELPATH = "manuscript/seeds/seed_manifest.json"
PROTOCOL_VERSION = "1.6"
# REAP's final step uses seed 42 for the final map and k-means (the default
# random_state of get_consensus_embedding and the benchmark's clustering).
FINAL_SEED = 42
DEFAULT_FINAL_SEEDS = [FINAL_SEED, 0, 1, 2, 3, 4, 5, 6, 7, 8]
# Neighbours in the k-NN vote; matches the trustworthiness neighbourhood the
# benchmark uses (15).
KNN_VOTE_K = 15
# Projection-head settings of scripts/run_projection_head_oos.py.
HEAD_SETTINGS: dict[str, Any] = {
    "hidden_layers": [128, 64], "dropout": 0.3, "alpha": 0.7, "n_folds": 5,
    "batch_size": 64, "max_epochs": 500, "patience": 50, "lr": 1e-3,
    "weight_decay": 1e-4, "device": "cpu",
}
HEAD_SEED = 42
# Conformal filter significance level: the package default (protocol §16.a).
FILTER_ALPHA = 0.01
# Rows count as unit-length when every norm is within this of 1 (float32 storage).
UNIT_NORM_TOLERANCE = 1e-3


def get_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (ids, vectors) from an ``.npz`` file or an ``emb.npy`` + ``ids.txt`` folder.

    An ``.npz`` must hold exactly one 1-d array (the ids) and one 2-d array
    (the vectors). Raises ``ValueError`` on anything ambiguous or misaligned.
    Side effects: reads files.
    """
    if path.is_dir():
        vectors = np.load(path / "emb.npy")
        ids = np.array((path / "ids.txt").read_text(encoding="utf-8").split("\n"))
        ids = ids[ids != ""]
    else:
        with np.load(path, allow_pickle=False) as data:
            one_d = [k for k in data.files if data[k].ndim == 1]
            two_d = [k for k in data.files if data[k].ndim == 2]
            if len(one_d) != 1 or len(two_d) != 1:
                raise ValueError(f"{path}: expected one id array and one vector array, "
                                 f"found {data.files}")
            ids, vectors = data[one_d[0]], data[two_d[0]]
    ids = np.asarray(ids).astype(str)
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"{path}: {len(ids)} ids for {vectors.shape[0]} vectors")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return ids, np.asarray(vectors, dtype=np.float32)


def get_table_ids(path: Path, where: list[str]) -> list[str]:
    """Ids of the table rows that match every ``column=value[,value...]`` filter.

    Values are compared as text. Raises ``ValueError`` for a filter column the
    table lacks. Side effects: reads the file.
    """
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    filters = []
    for item in where:
        column, _, values = item.partition("=")
        if not values:
            raise ValueError(f"expected column=value, got {item!r}")
        if rows and column not in rows[0]:
            raise ValueError(f"{path} has no column {column!r}")
        filters.append((column, set(values.split(","))))
    return [str(r["id"]) for r in rows if all(str(r[c]) in v for c, v in filters)]


def get_reference(emb_path: Path, table: Path | None,
                  where: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reference ids and vectors, restricted to the table rows the filters keep.

    Raises ``ValueError`` when a selected id has no vector, or nothing is kept.
    """
    ids, vectors = get_embeddings(emb_path)
    if table is None:
        if where:
            raise ValueError("--where needs --reference-table")
        return ids, vectors
    keep = get_table_ids(table, where)
    index = {i: n for n, i in enumerate(ids)}
    missing = [i for i in keep if i not in index]
    if missing:
        raise ValueError(f"{len(missing)} selected reference ids have no vector, e.g. "
                         f"{missing[:3]}")
    if not keep:
        raise ValueError("the reference filters keep no rows")
    rows = np.array([index[i] for i in keep])
    return ids[rows], vectors[rows]


def set_head_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch for the projection head (global state, on purpose).

    ``train_projection_head(seed=None)`` defers to these, as the OOS script does.
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_centroids(Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean consensus coordinates of each cluster, rows ordered by label 0..K-1."""
    return np.stack([Y[labels == k].mean(axis=0) for k in range(int(labels.max()) + 1)])


def get_assignments(Y_ref: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                    Y_probe: np.ndarray) -> dict[str, np.ndarray]:
    """Nearest-centroid and k-NN-vote cluster assignments of projected probes.

    The vote takes the ``KNN_VOTE_K`` nearest reference passages in the
    consensus space; ties go to the smaller cluster label. Also returns the
    winning label's share of the vote.
    """
    d2 = ((Y_probe[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    nn = NearestNeighbors(n_neighbors=min(KNN_VOTE_K, len(Y_ref))).fit(Y_ref)
    idx = nn.kneighbors(Y_probe, return_distance=False)
    votes = np.zeros((len(Y_probe), centroids.shape[0]), dtype=np.int64)
    for j in range(idx.shape[1]):
        np.add.at(votes, (np.arange(len(Y_probe)), labels[idx[:, j]]), 1)
    knn = np.argmax(votes, axis=1)
    share = votes[np.arange(len(Y_probe)), knn] / idx.shape[1]
    return {"nearest_centroid": nearest.astype(np.int64), "knn_vote": knn.astype(np.int64),
            "knn_vote_share": share.astype(np.float64)}


def get_mean_shifted(X_probe: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Probe vectors moved to the reference mean; re-normalised if the reference is unit-length."""
    shifted = X_probe - X_probe.mean(axis=0) + X_ref.mean(axis=0)
    if np.all(np.abs(np.linalg.norm(X_ref, axis=1) - 1.0) < UNIT_NORM_TOLERANCE):
        shifted = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    return shifted.astype(np.float32)


def get_final_step_repeats(D: np.ndarray, k: int, k_range: list[int], settings: dict[str, Any],
                           final_seeds: list[int]) -> dict[str, Any]:
    """REAP's final step under each final seed: labels at K, and the K silhouette picks."""
    labels, ks = [], []
    for s in final_seeds:
        emb, _ = get_consensus_embedding(D, settings["n_components"], settings["n_neighbors"],
                                         settings["min_dist"], random_state=s)
        labels.append(KMeans(n_clusters=k, random_state=s, n_init="auto").fit_predict(emb))
        ks.append(int(find_best_k(emb, k_range, random_state=s).best_k))
    aris = [float(adjusted_rand_score(a, b)) for a, b in itertools.combinations(labels, 2)]
    return {"labels": np.stack(labels).astype(np.int64), "silhouette_k_per_seed": ks,
            "pairwise_ari": {"mean": float(np.mean(aris)), "min": float(min(aris)),
                             "max": float(max(aris)), "n_pairs": len(aris)}}


def get_placement(Y_ref: np.ndarray, labels: np.ndarray, Y_probe: np.ndarray,
                  probe_labels: np.ndarray) -> dict[str, np.ndarray]:
    """How far each probe lands from the reference, and whether the conformal filter keeps it.

    ``nearest_reference_distance`` is the distance to the closest reference
    passage in the consensus space. ``accepted_strict`` uses the filter
    without location correction (is the probe inside its reference cluster?);
    ``accepted_pooled`` uses the package default, which re-centres each
    cluster on its probes and so tolerates a shift between corpora.
    """
    from reap.filter import apply as apply_filter
    from reap.filter import calibrate

    distance = NearestNeighbors(n_neighbors=1).fit(Y_ref).kneighbors(Y_probe)[0][:, 0]
    out = {"nearest_reference_distance": distance}
    for key, method in (("accepted_strict", "mahalanobis_no_correction"),
                        ("accepted_pooled", "mahalanobis_pooled")):
        calibration = calibrate(Y_ref, labels, Y_probe, probe_labels, alpha=FILTER_ALPHA,
                                method=method)
        out[key] = apply_filter(Y_probe, probe_labels, calibration)
    return out


def get_probe_rows(name: str, p_ids: np.ndarray, Y_ref: np.ndarray, repeat_labels: np.ndarray,
                   Yp: np.ndarray, Yp_shift: np.ndarray, variant: str, seed_set: str,
                   final_seeds: list[int]) -> list[tuple[Any, ...]]:
    """Long-format probe assignments for every final seed, rule and shift.

    Probe positions come from the head trained on the seed-42 map. Under final
    seed s only the reference labels change (the repeat's k-means labels), so
    the centroids and the k-NN vote use those labels in the same seed-42 space.
    """
    rows: list[tuple[Any, ...]] = []
    for s, lab in zip(final_seeds, repeat_labels):
        cents = get_centroids(Y_ref, lab)
        for shifted, coords in ((False, Yp), (True, Yp_shift)):
            a = get_assignments(Y_ref, lab, cents, coords)
            for rule in ("nearest_centroid", "knn_vote"):
                rows += [(i, name, variant, seed_set, s, rule, shifted, int(c))
                         for i, c in zip(p_ids, a[rule])]
    return rows


def _write_csv(path: Path, header: list[str], rows: list[tuple[Any, ...]]) -> None:
    """Write rows under a header. Side effects: writes ``path``."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_long_tables(out: Path, ids: np.ndarray, repeat_labels: np.ndarray,
                      layout_ref: np.ndarray, probe_rows: list[tuple[Any, ...]],
                      layout_rows: list[tuple[Any, ...]], variant: str, seed_set: str,
                      final_seeds: list[int]) -> None:
    """The long-format CSV tables the downstream analyses read.

    ``reference_labels.csv``, ``probe_assignments.csv``, ``clusters.csv`` and
    ``layout_2d.csv`` (display only). Side effects: writes four files.
    """
    ref_rows = [(i, variant, seed_set, s, int(c)) for s, lab in zip(final_seeds, repeat_labels)
                for i, c in zip(ids, lab)]
    _write_csv(out / "reference_labels.csv",
               ["id", "variant", "seed_set", "final_seed", "cluster"], ref_rows)
    _write_csv(out / "probe_assignments.csv",
               ["id", "probe_set", "variant", "seed_set", "final_seed", "rule", "shifted",
                "cluster"], probe_rows)
    cluster_rows = [(variant, seed_set, s, int(c), int(n)) for s, lab in
                    zip(final_seeds, repeat_labels) for c, n in enumerate(np.bincount(lab))]
    _write_csv(out / "clusters.csv",
               ["variant", "seed_set", "final_seed", "cluster", "n_reference"], cluster_rows)
    ref_layout = [(i, "reference", "", float(x), float(y)) for i, (x, y) in zip(ids, layout_ref)]
    _write_csv(out / "layout_2d.csv", ["id", "role", "probe_set", "x", "y"],
               ref_layout + layout_rows)


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Save named arrays to one ``.npz`` file. Side effects: writes ``path``."""
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]


def _sha256(arr: np.ndarray) -> str:
    """SHA-256 of an array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """One reference variant and seed set; writes every output. Returns the run record."""
    from reap.projection import train_projection_head

    started = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.monotonic()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids, X = get_reference(Path(args.reference),
                           Path(args.reference_table) if args.reference_table else None,
                           args.where)
    manifest = json.loads((REPO_ROOT / SEED_MANIFEST_RELPATH).read_text(encoding="utf-8"))
    seeds = [int(s) for s in manifest["sets"][args.seed_set]["seeds"]]
    if args.n_seeds is not None:
        seeds = seeds[: args.n_seeds]
    settings = {"n_components": args.n_components, "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist}
    k_range = list(range(args.k_min, args.k_max + 1))
    logger.info("reference %s: %d rows, %d seeds", args.label, len(ids), len(seeds))
    maps = get_multi_seed_embeddings(X, seeds, **settings, metric="cosine")
    D = get_consensus_distance_matrix(maps)
    Y, _ = get_consensus_embedding(D, **settings, random_state=FINAL_SEED)
    if args.k_from:
        k = int(json.loads(Path(args.k_from).read_text(encoding="utf-8"))["k"])
        k_rule = f"frozen from {Path(args.k_from).parent.name}"
    else:
        k = int(find_best_k(Y, k_range, random_state=FINAL_SEED).best_k)
        k_rule = "silhouette on this run's seed-42 map"
    labels = np.asarray(KMeans(n_clusters=k, random_state=FINAL_SEED,
                               n_init="auto").fit_predict(Y), dtype=np.int64)
    centroids = get_centroids(Y, labels)
    repeats = get_final_step_repeats(D, k, k_range, settings, list(args.final_seeds))
    t_map = time.monotonic() - t0

    set_head_seeds(HEAD_SEED)
    head = train_projection_head(X=X, Y=Y.astype(np.float32), labels=labels, seed=None,
                                 **HEAD_SETTINGS)
    model = head["model"]
    t_head = time.monotonic() - t0 - t_map

    import umap

    layout = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                       random_state=FINAL_SEED).fit(Y)
    _save_arrays(out / "reference.npz", {
        "ids": ids, "consensus": Y, "labels": labels, "repeat_labels": repeats["labels"],
        "layout_2d": np.asarray(layout.embedding_)})
    np.save(out / "centroids.npy", centroids)
    probes: dict[str, Any] = {}
    probe_rows: list[tuple[Any, ...]] = []
    layout_rows: list[tuple[Any, ...]] = []
    placement_rows: list[tuple[Any, ...]] = []
    for item in args.probe:
        name, _, path = item.partition("=")
        p_ids, Xp = get_embeddings(Path(path))
        if Xp.shape[1] != X.shape[1]:
            raise ValueError(f"probe {name}: dimension {Xp.shape[1]} != reference {X.shape[1]}")
        Yp = np.asarray(model.forward(Xp), dtype=np.float64)
        Yp_shift = np.asarray(model.forward(get_mean_shifted(Xp, X)), dtype=np.float64)
        a = get_assignments(Y, labels, centroids, Yp)
        a_shift = get_assignments(Y, labels, centroids, Yp_shift)
        layout_p = np.asarray(layout.transform(Yp))
        _save_arrays(out / f"probe_{name}.npz", {
            "ids": p_ids, "consensus": Yp, "layout_2d": layout_p,
            "consensus_mean_shifted": Yp_shift,
            **{f"assign_{k_}": v for k_, v in a.items()},
            **{f"assign_{k_}_mean_shifted": v for k_, v in a_shift.items()}})
        probe_rows += get_probe_rows(name, p_ids, Y, repeats["labels"], Yp, Yp_shift,
                                     args.label, args.seed_set, list(args.final_seeds))
        layout_rows += [(i, "probe", name, float(x), float(y)) for i, (x, y)
                        in zip(p_ids, layout_p)]
        placement = get_placement(Y, labels, Yp, a["knn_vote"])
        placement_rows += [(i, name, args.label, args.seed_set, float(d), bool(s), bool(p))
                           for i, d, s, p in zip(p_ids, placement["nearest_reference_distance"],
                                                 placement["accepted_strict"],
                                                 placement["accepted_pooled"])]
        probes[name] = {
            "n": int(len(p_ids)), "input_sha256": _sha256(Xp),
            "agreement_centroid_vs_knn": float(np.mean(a["nearest_centroid"] == a["knn_vote"])),
            "nearest_reference_distance_median": float(
                np.median(placement["nearest_reference_distance"])),
            "accepted_strict_share": float(np.mean(placement["accepted_strict"])),
            "accepted_pooled_share": float(np.mean(placement["accepted_pooled"])),
        }
    write_long_tables(out, ids, repeats["labels"], np.asarray(layout.embedding_), probe_rows,
                      layout_rows, args.label, args.seed_set, list(args.final_seeds))
    _write_csv(out / "probe_placement.csv",
               ["id", "probe_set", "variant", "seed_set", "nearest_reference_distance",
                "accepted_strict", "accepted_pooled"], placement_rows)
    model.save(str(out / "head.pt"))
    (out / "head_cv.json").write_text(json.dumps(
        {"cv_metrics": head["cv_metrics"], "final_metrics": head["final_metrics"],
         "settings": HEAD_SETTINGS, "seed": HEAD_SEED}, indent=2, default=float) + "\n",
        encoding="utf-8")
    record = {
        "label": args.label, "seed_set": args.seed_set, "n_reference": int(len(ids)),
        "reference_input": str(args.reference), "reference_where": args.where,
        "reference_sha256": _sha256(X), "settings": settings, "k_range": [args.k_min, args.k_max],
        "k": k, "k_rule": k_rule, "cluster_sizes": np.bincount(labels).tolist(),
        "final_step_repeats": {"final_seeds": list(args.final_seeds),
                               "silhouette_k_per_seed": repeats["silhouette_k_per_seed"],
                               "pairwise_ari_at_k": repeats["pairwise_ari"]},
        "knn_vote_k": KNN_VOTE_K, "probes": probes,
        "reference_nearest_neighbour_distance_median": float(np.median(
            NearestNeighbors(n_neighbors=2).fit(Y).kneighbors(Y)[0][:, 1])),
        "filter": {"alpha": FILTER_ALPHA, "labels": "knn_vote at final seed 42",
                   "strict": "mahalanobis_no_correction", "pooled": "mahalanobis_pooled"},
        "seconds": {"map_and_repeats": t_map, "head": t_head,
                    "total": time.monotonic() - t0},
    }
    (out / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_repro_bundle(ReproBundle(
        protocol_version=PROTOCOL_VERSION, code=get_code_state(REPO_ROOT),
        script="scripts/run_reference_probes.py", run_id=f"reference_probes/{args.label}/"
        f"{args.seed_set}", dataset=args.label, dataset_version=args.dataset_version,
        dataset_sha256=_sha256(X), encoder=args.encoder, seed_set=args.seed_set,
        seeds=seeds + list(args.final_seeds) + [HEAD_SEED], seed_manifest=SEED_MANIFEST_RELPATH,
        config={"umap": settings, "k_range": [args.k_min, args.k_max], "k": k,
                "k_rule": k_rule, "where": list(args.where), "head": HEAD_SETTINGS,
                "probes": [p.partition("=")[0] for p in args.probe],
                "knn_vote_k": KNN_VOTE_K, "n_seeds": len(seeds)},
        evaluation_mode="final_fit",
        metric_recipe_ids=["ari.between_final_step_runs", "projection_head.cv"],
        environment=get_environment_snapshot(), run_status="completed",
        started_at=started, finished_at=datetime.datetime.now(datetime.timezone.utc),
    ), out / "bundle.json")
    return record


def _get_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    p = argparse.ArgumentParser(description="REAP map on a reference, probes projected in.")
    p.add_argument("--label", required=True, help="Name of this reference variant.")
    p.add_argument("--dataset-version", default="unversioned")
    p.add_argument("--encoder", default=None)
    p.add_argument("--reference", required=True, help="Reference embeddings (.npz or folder).")
    p.add_argument("--reference-table", default=None)
    p.add_argument("--where", action="append", default=[],
                   help="column=value[,value...] filter on the reference table (repeatable).")
    p.add_argument("--probe", action="append", default=[], help="name=embeddings (repeatable).")
    p.add_argument("--seed-set", default="A", choices=["A", "B", "C"])
    p.add_argument("--n-seeds", type=int, default=None,
                   help="Use only the first n seeds of the set (smoke runs and tests).")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=30)
    p.add_argument("--k-from", default=None, help="run_record.json whose K this run uses.")
    p.add_argument("--final-seeds", nargs="+", type=int, default=DEFAULT_FINAL_SEEDS)
    p.add_argument("--output-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the driver; return 0. Raises on any missing or inconsistent input."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(_get_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
