"""Held-out check of projection heads on a saved REAP reference map, with a calibrated inside test.

For one saved run of ``run_reference_probes.py`` (its ``reference.npz``:
ids, consensus coordinates, cluster labels) and the reference embeddings,
each head variant in the settings file is checked the same way:

1. Every reference passage is placed by a head trained without it (shuffled
   k-fold, not stratified by cluster, so the folds do not change with K).
2. For each held-out passage the script records how far it lands from its
   true position and from the nearest other reference position. It also
   records its embedding-space (cosine) distance to the nearest training
   passage, and whether it keeps its cluster: the nearest centroid of the
   training fold's clusters is its true label.
3. The inside cutoff is the split-conformal (1 - alpha) rank of the held-out
   nearest-reference distances (``reap.placement.get_conformal_cutoff``).
4. A head trained on every reference passage places each probe, and a probe
   counts as inside when its nearest-reference distance is within the cutoff.

Every setting comes from ``--head-settings`` (JSON); nothing has a default.
Example settings file::

    {"n_folds": 5, "fold_seed": 1, "alpha": 0.01,
     "heads": {"ridge": {"alpha_grid": [0.01, 0.1, 1.0, 10.0]},
               "knn": {"k_grid": [1, 3, 5, 10], "weights": "distance",
                       "metric": "cosine", "n_inner_folds": 5, "random_state": 1}}}

Heads: ``reap`` ({"settings": {...}, "seed": int}), ``scaled_mlp``
(``reap.placement.fit_scaled_mlp_head`` keywords), ``ridge`` and ``knn``.

Outputs in ``--output-dir``: ``heldout_<head>.csv`` (one row per reference
passage), ``probe_placement_<head>.csv`` (one row per probe passage),
``placements_<head>.npz`` (the placed coordinates: ``reference_ids`` and
``heldout`` for the reference, ``probe_ids__<name>`` and ``probe__<name>``
for each probe set) and ``summary.json`` (per-head summary, settings, input
hashes, code commit).

Side effects: reads the inputs, trains the heads, writes the output folder
(refuses to overwrite an existing ``summary.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reference_probes import get_embeddings  # noqa: E402

from reap.placement import (  # noqa: E402
    HeadFit,
    fit_knn_head,
    fit_reap_head,
    fit_ridge_head,
    fit_scaled_mlp_head,
    get_conformal_cutoff,
    get_heldout_predictions,
    get_nearest_reference_distances,
)

HEAD_KINDS = ("reap", "scaled_mlp", "ridge", "knn")


def get_fit(kind: str, spec: dict[str, Any], X: np.ndarray, Y: np.ndarray,
            labels: np.ndarray) -> Any:
    """A function from training row indices to a fitted head of ``kind``. Side effects: none."""
    if kind == "reap":
        return lambda tr: fit_reap_head(X[tr], Y[tr], labels[tr], settings=spec["settings"],
                                        seed=spec["seed"])
    if kind == "scaled_mlp":
        return lambda tr: fit_scaled_mlp_head(X[tr], Y[tr], **spec)
    if kind == "ridge":
        return lambda tr: fit_ridge_head(X[tr], Y[tr], **spec)
    if kind == "knn":
        return lambda tr: fit_knn_head(X[tr], Y[tr], **spec)
    raise ValueError(f"unknown head {kind!r}; choose from {HEAD_KINDS}")


def get_nearest_training_distance(X: np.ndarray, fold_of_row: np.ndarray) -> np.ndarray:
    """Cosine distance from each row to its nearest row in the other folds. Side effects: none."""
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    dist = 1.0 - unit @ unit.T
    dist[fold_of_row[:, None] == fold_of_row[None, :]] = np.inf
    return dist.min(axis=1)


def get_kept_labels(Y: np.ndarray, Y_hat: np.ndarray, labels: np.ndarray,
                    fold_of_row: np.ndarray) -> np.ndarray:
    """Each held-out row's nearest training-fold centroid label. Side effects: none."""
    predicted = np.empty(len(Y), dtype=np.int64)
    for f in np.unique(fold_of_row):
        train, test = fold_of_row != f, fold_of_row == f
        names = np.unique(labels[train])
        centroids = np.stack([Y[train][labels[train] == c].mean(axis=0) for c in names])
        d = np.linalg.norm(Y_hat[test][:, None, :] - centroids[None, :, :], axis=2)
        predicted[test] = names[d.argmin(axis=1)]
    return predicted


def get_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes. Side effects: reads the file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_code_commit() -> str:
    """The checkout's commit, with ``+dirty`` when tracked files changed. Side effects: runs git."""
    here = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True,
                           check=True).stdout.strip()
    return commit + ("+dirty" if dirty else "")


def get_head_summary(kind: str, fit: Any, X: np.ndarray, Y: np.ndarray, labels: np.ndarray,
                     ids: np.ndarray, sources: np.ndarray, probes: dict[str, tuple[np.ndarray,
                     np.ndarray]], cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Run the held-out check and probe placement for one head; write its two tables.

    Side effects: trains heads, writes ``heldout_<kind>.csv`` and ``probe_placement_<kind>.csv``.
    """
    Y_hat, fold_of_row, fold_fits = get_heldout_predictions(
        X, fit, n_folds=cfg["n_folds"], random_state=cfg["fold_seed"])
    miss = np.linalg.norm(Y_hat - Y, axis=1)
    nn = get_nearest_reference_distances(Y, Y_hat, own_rows=np.arange(len(Y)))
    e5_nn = get_nearest_training_distance(X, fold_of_row)
    predicted = get_kept_labels(Y, Y_hat, labels, fold_of_row)
    cutoff = get_conformal_cutoff(nn, alpha=cfg["alpha"])
    pd.DataFrame({"id": ids, "source": sources, "label": labels, "fold": fold_of_row,
                  "miss": miss, "nearest_reference_distance": nn,
                  "nearest_training_cosine_distance": e5_nn, "predicted_label": predicted,
                  "kept_cluster": predicted == labels}).to_csv(
        out / f"heldout_{kind}.csv", index=False)
    full: HeadFit = fit(np.arange(len(X)))
    probe_rows, probe_summary = [], {}
    coordinates: dict[str, np.ndarray] = {"reference_ids": ids, "heldout": Y_hat}
    for name, (p_ids, P) in probes.items():
        P_hat = full.predict(P)
        coordinates[f"probe_ids__{name}"], coordinates[f"probe__{name}"] = p_ids, P_hat
        p_nn = get_nearest_reference_distances(Y, P_hat, own_rows=None)
        probe_rows.append(pd.DataFrame({"id": p_ids, "probe_set": name,
                                        "nearest_reference_distance": p_nn,
                                        "inside_calibrated": p_nn <= cutoff}))
        probe_summary[name] = {
            "n": int(len(p_ids)), "inside_share": float(np.mean(p_nn <= cutoff)),
            "nearest_reference_distance_median": float(np.median(p_nn)),
            "ratio_to_heldout_median": float(np.median(p_nn) / np.median(nn))}
    if probe_rows:
        pd.concat(probe_rows).to_csv(out / f"probe_placement_{kind}.csv", index=False)
    np.savez(out / f"placements_{kind}.npz", **coordinates)  # pyright: ignore[reportArgumentType]
    by_cluster = pd.DataFrame({"label": labels, "miss": miss, "kept": predicted == labels}
                              ).groupby("label").agg(n=("miss", "size"),
                                                     miss_median=("miss", "median"),
                                                     kept_share=("kept", "mean"))
    rho = spearmanr(miss, e5_nn)
    return {
        "chosen_per_fold": [f.chosen for f in fold_fits], "chosen_full": full.chosen,
        "heldout": {
            "miss_median": float(np.median(miss)), "miss_mean": float(np.mean(miss)),
            "nearest_reference_distance_median": float(np.median(nn)),
            "inside_cutoff": cutoff, "inside_share": float(np.mean(nn <= cutoff)),
            "kept_cluster_share": float(np.mean(predicted == labels)),
            "kept_cluster_ari": float(adjusted_rand_score(labels, predicted)),
            "spearman_miss_vs_nearest_training_distance": float(rho.statistic),  # pyright: ignore[reportAttributeAccessIssue]
            "miss_median_by_source": {str(s): float(np.median(miss[sources == s]))
                                      for s in np.unique(sources)},
            "kept_share_by_source": {str(s): float(np.mean((predicted == labels)[sources == s]))
                                     for s in np.unique(sources)},
            "by_cluster": {str(k): {c: float(v) for c, v in row.items()}
                           for k, row in by_cluster.iterrows()},
        },
        "probes": probe_summary,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every head, write the outputs. Side effects: see module docstring."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="Reference embeddings (.npz with ids and vectors).")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="A run_reference_probes.py output folder (holds reference.npz).")
    parser.add_argument("--probe", action="append", default=[], metavar="NAME=PATH",
                        help="Probe embeddings to place (repeatable).")
    parser.add_argument("--head-settings", type=Path, required=True)
    parser.add_argument("--source-separator", required=True,
                        help="A passage's source is the part of its id before this text.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = json.loads(args.head_settings.read_text(encoding="utf-8"))
    missing = {"n_folds", "fold_seed", "alpha", "heads"} - set(cfg)
    if missing:
        parser.error(f"--head-settings lacks {sorted(missing)}")
    unknown = set(cfg["heads"]) - set(HEAD_KINDS)
    if unknown:
        parser.error(f"unknown heads {sorted(unknown)}; choose from {HEAD_KINDS}")
    if (args.output_dir / "summary.json").exists():
        parser.error(f"{args.output_dir}/summary.json exists; choose a new --output-dir")

    ids, X = get_embeddings(args.reference)
    with np.load(args.run_dir / "reference.npz", allow_pickle=False) as run:
        run_ids, Y, labels = run["ids"].astype(str), run["consensus"], run["labels"]
    if not np.array_equal(ids, run_ids):
        parser.error("--reference ids do not match the run's reference.npz ids in order")
    Y = np.asarray(Y, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    sources = np.array([i.split(args.source_separator)[0] for i in ids])
    probes = {}
    for spec in args.probe:
        name, _, path = spec.partition("=")
        probes[name] = get_embeddings(Path(path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heads = {kind: get_head_summary(kind, get_fit(kind, spec, X, Y, labels), X, Y, labels, ids,
                                    sources, probes, cfg, args.output_dir)
             for kind, spec in cfg["heads"].items()}
    summary = {
        "run_dir": str(args.run_dir), "n_reference": int(len(ids)), "k": int(len(set(labels))),
        "settings": cfg, "code_commit": get_code_commit(),
        "input_sha256": {"reference": get_sha256(args.reference),
                         "run_reference": get_sha256(args.run_dir / "reference.npz"),
                         "head_settings": get_sha256(args.head_settings),
                         **{f"probe:{s.partition('=')[0]}": get_sha256(Path(s.partition('=')[2]))
                            for s in args.probe}},
        "heads": heads,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
