"""Run one in-scope hold-out cell of the protocol and write its results to REAP-research.

A "cell" is one row of ``docs/run_matrix_v1_6.json`` — one corpus, one hold-out
strategy, one encoder (protocol §18, v1.6 amendment). For each replicate of the
cell this script:

1. builds the split exactly as v1.6 clause (e) fixes it;
2. builds the in-set consensus from the 30 Set D seeds and chooses K once
   (silhouette maximum over 5…30 — the §9 composite rule);
3. builds the adapted-ARI reference partition from a consensus rerun on
   in-set ∪ held-out;
4. runs the hold-out harness — REAP MLP head, linear head and parametric UMAP
   trained once per Set D seed, the identity baseline once (v1.6 clause b);
5. computes the pre-designated paired subsample intervals on run 1 and the
   leakage audit;
6. writes every artifact plus a ``bundle.json`` record file.

Gates (fail closed, before any compute): the cell must be ``in_scope``; the
matrix's protocol version must be ≥ 1.6; the code must be a commit already on
``origin/main`` (so the amendment is ratified and the code reviewed) with no
uncommitted tracked changes; and the output directory must not already hold a
completed result — results are never silently overwritten. A run that crashes
still writes a ``bundle.json`` with ``run_status = "failed"`` and the reason.

``--preflight`` checks the gates that do not need merged code, builds the
splits, and logs sizes, fold-floor checks, groups that fall on both sides, and
the leakage audit, without training anything or writing anything.

Usage::

    python scripts/run_holdout_cell.py --cell ai_art/T1
    python scripts/run_holdout_cell.py --cell ai_art/T5 --preflight

Source projects (for split variables): ``REAP_AI_ART_SOURCE_PATH`` /
``REAP_KOREAN_FOREST_SOURCE_PATH``, else the sibling checkouts. Results go to
the research root (``--research-root``, ``REAP_RESEARCH_ROOT``, or the
``../REAP-research`` sibling) under
``results/temporal_holdout/<corpus>/<strategy>[/<encoder>][/replicate_<k>]/``.

Side effects: reads datasets and source files, trains models, writes result
files under the research root, logs progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_root import get_research_root  # noqa: E402

from reap import find_best_k, run_consensus_pipeline, run_kmeans  # noqa: E402
from reap.baselines import get_identity_projection  # noqa: E402
from reap.holdout import (  # noqa: E402
    ALL_METHODS,
    HOLDOUT_METRIC_IDS,
    HeadTrainingSettings,
    ParametricUMAPSettings,
    get_paired_subsample_intervals,
    run_holdout_comparison,
)
from reap.repro_bundle import (  # noqa: E402
    ReproBundle,
    get_code_state,
    get_environment_snapshot,
    read_repro_bundle,
    validate_protocol_version,
    write_repro_bundle,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_MATRIX_PATH = REPO_ROOT / "docs" / "run_matrix_v1_6.json"
SEED_MANIFEST_RELPATH = "manuscript/seeds/seed_manifest.json"
MINIMUM_PROTOCOL_VERSION = "1.6"
SCRIPT_RELPATH = "scripts/run_holdout_cell.py"

# Canonical UMAP settings per corpus (hold-out proposal §2.1 step 2).
UMAP_SETTINGS = {
    "ai_art": {"n_components": 5, "n_neighbors": 53, "min_dist": 0.01},
    "korean_forest": {"n_components": 18, "n_neighbors": 19, "min_dist": 0.005},
}
K_RANGE = range(5, 31)  # protocol §9 / v1.6 clause (e)
KMEANS_SEED = 42  # v1.6 clause (e): K choice and reference partition
N_HELD_OUT_FOLDS = 5  # 1 of 5 folds ≈ 20% held out (§18.b/c, v1.6 clause e)
LEAKAGE_SIMILARITY_THRESHOLD = 0.95  # v1.5 clause (f)
ENCODER_DIRS = {
    "intfloat/e5-large-v2": None,
    "paraphrase-multilingual-MiniLM-L12-v2": "minilm",
}

# AI-art T1–T4 held-out years (protocol §18.b).
AI_ART_YEAR_STRATEGIES = {
    "T1": {2023, 2024, 2025},
    "T2": {2013, 2015, 2016, 2017},
    "T3": {2025},
    "T4": {2019},
}
# Korean-forest K2-T1..T3 held-out president (protocol §18.c).
KF_PRESIDENT_STRATEGIES = {"K2-T1": "Moon", "K2-T2": "Lee", "K2-T3": "Park"}


class SplitVariables(BaseModel):
    """Per-row variables a corpus's splits are built from, row-aligned with its snapshot."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    strata: np.ndarray = Field(..., description="(n,) year (AI-art) or president (Korean forest).")
    groups: np.ndarray = Field(
        ...,
        description="(n,) grouping unit for random splits: source document (AI-art) or "
        "normalized text (Korean forest).",
    )


class Split(BaseModel):
    """One replicate's in-set and held-out row indices."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    replicate: int | None = Field(..., description="Replicate number, None if not replicated.")
    master_seed: int | None = Field(..., description="Split seed for random strategies.")
    in_idx: np.ndarray = Field(..., description="In-set row indices (sorted).")
    out_idx: np.ndarray = Field(..., description="Held-out row indices (sorted).")


def get_cell(matrix: dict, cell_id: str) -> dict:
    """Return one cell of the run matrix after the scope and protocol gates.

    Raises
    ------
    ValueError : If the cell is unknown or not in scope, or the matrix's protocol
        version is below ``MINIMUM_PROTOCOL_VERSION``.
    """
    validate_protocol_version(matrix["protocol_version"], MINIMUM_PROTOCOL_VERSION)
    cells = {cell["cell_id"]: cell for cell in matrix["cells"]}
    if cell_id not in cells:
        raise ValueError(f"unknown cell {cell_id!r}")
    cell = cells[cell_id]
    if cell["status"] != "in_scope":
        raise ValueError(f"cell {cell_id} is {cell['status']}: {cell.get('deferral_reason')}")
    return cell


def get_set_d_seeds(manifest_path: Path) -> list[int]:
    """Return the 30 Set D seeds from the seed manifest (raises if not exactly 30)."""
    seeds = json.loads(manifest_path.read_text())["sets"]["D"]["seeds"]
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise ValueError("seed manifest Set D must hold 30 distinct seeds")
    return [int(s) for s in seeds]


def _get_source_root(env_var: str, sibling: Path) -> Path:
    """Return a source project's root from ``env_var`` or the sibling checkout.

    Raises
    ------
    FileNotFoundError : If neither exists.
    """
    explicit = os.environ.get(env_var)
    root = Path(explicit).expanduser() if explicit else sibling
    if not root.is_dir():
        raise FileNotFoundError(f"source project not found at {root}; set {env_var}")
    return root


def get_ai_art_split_variables(source_root: Path, snapshot_texts: list[str]) -> SplitVariables:
    """Read AI-art years and source-document ids, checked row by row against the snapshot.

    Raises
    ------
    ValueError : If any row's chunk text differs from the snapshot's text.
    """
    path = source_root / "data" / "public_discourse_clean_chunks.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [r["chunk_text"] for r in rows] != snapshot_texts:
        raise ValueError(f"{path} rows do not match the AI-art snapshot texts")
    return SplitVariables(
        strata=np.asarray([int(r["year"]) for r in rows]),
        groups=np.asarray([int(r["line_number"]) for r in rows]),
    )


def get_korean_forest_split_variables(
    source_root: Path, snapshot_texts: list[str]
) -> SplitVariables:
    """Read Korean-forest presidents and normalized-text groups, checked against the snapshot.

    Raises
    ------
    ValueError : If any row's text differs from the snapshot's text.
    """
    path = source_root / "data" / "og_data" / "Sections_of_Strategy_Sentences6.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [r["text"] for r in rows] != snapshot_texts:
        raise ValueError(f"{path} rows do not match the Korean-forest snapshot texts")
    normalized = [r["section_text_norm"] for r in rows]
    group_ids = {text: index for index, text in enumerate(dict.fromkeys(normalized))}
    return SplitVariables(
        strata=np.asarray([r["president"] for r in rows]),
        groups=np.asarray([group_ids[text] for text in normalized]),
    )


def _get_first_fold(
    variables: SplitVariables, master_seed: int, stratified: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_idx, out_idx) from the first group-aware fold (v1.6 clause e)."""
    n = variables.strata.shape[0]
    rows = np.zeros((n, 1))
    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=N_HELD_OUT_FOLDS, shuffle=True, random_state=master_seed
        )
        in_idx, out_idx = next(splitter.split(rows, variables.strata, variables.groups))
    else:
        shuffler = GroupShuffleSplit(
            n_splits=1, test_size=1.0 / N_HELD_OUT_FOLDS, random_state=master_seed
        )
        in_idx, out_idx = next(shuffler.split(rows, groups=variables.groups))
    return np.sort(in_idx), np.sort(out_idx)


def get_splits(cell: dict, variables: SplitVariables) -> list[Split]:
    """Build every replicate's split for a within-corpus cell (v1.6 clause e).

    Pure function. K1 (corpus hold-out) has no within-corpus split; see main.

    Raises
    ------
    ValueError : For an unknown strategy.
    """
    strategy = cell["strategy"]
    n = variables.strata.shape[0]
    if strategy in AI_ART_YEAR_STRATEGIES or strategy in KF_PRESIDENT_STRATEGIES:
        if strategy in AI_ART_YEAR_STRATEGIES:
            held = np.isin(variables.strata, list(AI_ART_YEAR_STRATEGIES[strategy]))
        else:
            held = variables.strata == KF_PRESIDENT_STRATEGIES[strategy]
        return [
            Split(
                replicate=None,
                master_seed=None,
                in_idx=np.flatnonzero(~held),
                out_idx=np.flatnonzero(held),
            )
        ]
    if strategy in ("T5", "K2-T4", "K2-T5"):
        stratified = strategy != "K2-T5"
        seeds = cell["master_seeds"]
        splits = []
        for replicate, master_seed in enumerate(seeds, start=1):
            in_idx, out_idx = _get_first_fold(variables, master_seed, stratified)
            splits.append(
                Split(
                    replicate=replicate if len(seeds) > 1 else None,
                    master_seed=master_seed,
                    in_idx=in_idx,
                    out_idx=out_idx,
                )
            )
        return splits
    raise ValueError(f"no within-corpus split rule for strategy {strategy!r} (n={n})")


def get_leakage_audit(X_in: np.ndarray, X_out: np.ndarray) -> dict[str, float | int]:
    """Summarize each held-out row's highest cosine similarity to any in-set row.

    Pure function (v1.5 clause f). Returns the distribution's quantiles and the
    count and share above ``LEAKAGE_SIMILARITY_THRESHOLD``.
    """
    dummy_map = np.zeros((X_in.shape[0], 1))
    similarity = get_identity_projection(X_in, dummy_map, X_out).nearest_reference_similarity
    above = int(np.sum(similarity > LEAKAGE_SIMILARITY_THRESHOLD))
    quantiles = np.percentile(similarity, [0, 25, 50, 75, 95, 100])
    return {
        "n_held_out": int(similarity.size),
        "threshold": LEAKAGE_SIMILARITY_THRESHOLD,
        "n_above_threshold": above,
        "share_above_threshold": above / similarity.size,
        **{f"q{q}": float(v) for q, v in zip((0, 25, 50, 75, 95, 100), quantiles)},
    }


def get_output_dir(research_root: Path, cell: dict, replicate: int | None) -> Path:
    """Return the result directory for one replicate of one cell (protocol §18.f)."""
    path = research_root / "results" / "temporal_holdout" / cell["corpus"] / cell["strategy"]
    encoder_dir = ENCODER_DIRS.get(cell["encoder"])
    if encoder_dir:
        path = path / encoder_dir
    return path / f"replicate_{replicate}" if replicate is not None else path


def _write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write via ``write(tmp_path)`` to a unique temp file, then rename into place.

    Side effects: creates the parent directory and writes ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as indented JSON, atomically."""
    text = json.dumps(payload, indent=2) + "\n"

    def _save(tmp: Path) -> None:
        tmp.write_text(text, encoding="utf-8")

    _write_atomic(path, _save)


def _write_npy(path: Path, array: np.ndarray) -> None:
    """Write ``array`` as .npy, atomically."""

    def _save(tmp: Path) -> None:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)

    _write_atomic(path, _save)


def validate_output_dir_is_free(output_dir: Path) -> None:
    """Refuse to overwrite a completed result (fails closed).

    Raises
    ------
    FileExistsError : If ``output_dir/bundle.json`` records a completed run.
    """
    bundle_path = output_dir / "bundle.json"
    if bundle_path.exists() and read_repro_bundle(bundle_path).run_status == "completed":
        raise FileExistsError(
            f"{output_dir} already holds a completed result; move it aside deliberately "
            "before re-running"
        )


def validate_code_is_merged(repo_root: Path) -> None:
    """Refuse to produce results from code that is not on origin/main or is modified.

    Side effects: runs git.

    Raises
    ------
    RuntimeError : If tracked files have uncommitted changes, or HEAD is not an
        ancestor of ``origin/main`` (unmerged code; the protocol amendment it
        relies on would not be ratified).
    """
    if get_code_state(repo_root).dirty:
        raise RuntimeError("tracked files have uncommitted changes; commit or stash them first")
    merged = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", "HEAD", "origin/main"],
        check=False,
    )
    if merged.returncode != 0:
        raise RuntimeError("HEAD is not on origin/main; results come only from merged code")


def run_replicate(
    *,
    cell: dict,
    split: Split | None,
    X_in: np.ndarray,
    X_out: np.ndarray,
    set_d_seeds: list[int],
    output_dir: Path,
    dataset_meta: dict[str, str],
    n_resamples: int,
    groups_out: np.ndarray | None,
    head_settings: HeadTrainingSettings,
    umap_epochs: int,
    code_root: Path,
    methods: tuple[str, ...] = ALL_METHODS,
) -> ReproBundle:
    """Run one replicate end to end and write its artifacts and bundle.

    ``methods`` defaults to all four registered methods; tests pass a subset.
    Side effects: trains models; writes files under ``output_dir``. On any error
    a failed bundle is written before the error propagates.
    """
    started = datetime.now(timezone.utc)
    umap = UMAP_SETTINGS[cell["corpus"]]
    config: dict = {
        "cell_id": cell["cell_id"],
        "strategy": cell["strategy"],
        "replicate": None if split is None else split.replicate,
        "split_master_seed": None if split is None else split.master_seed,
        "split_rule": "protocol v1.6 clause (e)",
        "umap": umap,
        "k_range": [K_RANGE.start, K_RANGE.stop - 1],
        "kmeans_seed": KMEANS_SEED,
        "head_settings": head_settings.model_dump(),
        "parametric_umap_epochs": umap_epochs,
        "n_resamples": n_resamples,
        "subsample_fraction": 0.8,
        "subsample_seed": set_d_seeds[0],
        "n_in": int(X_in.shape[0]),
        "n_out": int(X_out.shape[0]),
        "methods": list(methods),
        **{f"dataset_{k}": v for k, v in dataset_meta.items() if k.startswith("held_out_")},
    }
    base_fields = {
        "protocol_version": MINIMUM_PROTOCOL_VERSION,
        "code": get_code_state(code_root),
        "script": SCRIPT_RELPATH,
        "run_id": cell["cell_id"] + ("" if split is None or split.replicate is None
                                     else f"/replicate_{split.replicate}"),
        "dataset": cell["corpus"],
        "dataset_version": dataset_meta["version"],
        "dataset_sha256": dataset_meta["sha256"],
        "encoder": cell["encoder"],
        "seed_set": "D",
        "seeds": set_d_seeds,
        "seed_manifest": SEED_MANIFEST_RELPATH,
        "config": config,
        "evaluation_mode": "holdout",
        "metric_recipe_ids": list(HOLDOUT_METRIC_IDS),
        "environment": get_environment_snapshot(),
        "started_at": started,
    }
    try:
        if split is not None:
            _write_npy(output_dir / "retained_indices.npy", split.in_idx)
            _write_npy(output_dir / "held_out_indices.npy", split.out_idx)
        Y_in, _ = run_consensus_pipeline(X_in, seeds=set_d_seeds, **umap)
        k_holdout = find_best_k(Y_in, K_RANGE, random_state=KMEANS_SEED).best_k
        labels_in, _ = run_kmeans(Y_in, k_holdout, random_state=KMEANS_SEED)
        X_all = np.concatenate([X_in, X_out])
        Y_all, _ = run_consensus_pipeline(X_all, seeds=set_d_seeds, **umap)
        labels_all, _ = run_kmeans(Y_all, k_holdout, random_state=KMEANS_SEED)
        reference_out = labels_all[X_in.shape[0]:]
        _write_npy(output_dir / "consensus_embedding.npy", Y_in)
        _write_npy(output_dir / "consensus_labels.npy", labels_in)
        _write_npy(output_dir / "reference_held_out_labels.npy", reference_out)

        comparison, projections = run_holdout_comparison(
            X_in, Y_in, labels_in, X_out, reference_out,
            training_seeds=set_d_seeds, run_id=base_fields["run_id"],
            methods=methods,  # type: ignore[arg-type]
            head_settings=head_settings,
            umap_settings=ParametricUMAPSettings(**umap, n_epochs=umap_epochs),
        )
        _write_json(output_dir / "comparison_metrics.json", comparison.model_dump(mode="json"))
        for method, projection in projections.items():
            _write_npy(output_dir / f"run1_{method}_held_out_projection.npy", projection.Y_out)
        intervals = get_paired_subsample_intervals(
            X_out, projections, labels_in, reference_out, comparison.k_holdout,
            seed=set_d_seeds[0], n_resamples=n_resamples, groups=groups_out,
        )
        _write_json(
            output_dir / "subsample_intervals.json", [i.model_dump(mode="json") for i in intervals]
        )
        _write_json(output_dir / "leakage_audit.json", get_leakage_audit(X_in, X_out))
        config["k_holdout"] = comparison.k_holdout
    except BaseException as error:
        failed = ReproBundle(
            **base_fields, run_status="failed", failure_reason=f"{type(error).__name__}: {error}",
            finished_at=datetime.now(timezone.utc),
        )
        write_repro_bundle(failed, output_dir / "bundle.json")
        raise
    bundle = ReproBundle(**base_fields, run_status="completed",
                         finished_at=datetime.now(timezone.utc))
    write_repro_bundle(bundle, output_dir / "bundle.json")
    return bundle


def _load_cell_data(cell: dict) -> dict:
    """Load the cell's embeddings, split variables, and dataset metadata.

    Side effects: reads cached snapshots and source files.
    """
    from reap.datasets import load_ai_art, load_korean_forest, load_korean_forest_oos_raw

    if cell["corpus"] == "ai_art":
        snapshot = load_ai_art()
        source = _get_source_root(
            "REAP_AI_ART_SOURCE_PATH", REPO_ROOT.parent / "When-Algorithms-Meet-Artists"
        )
        variables = get_ai_art_split_variables(source, list(snapshot.texts or []))
    else:
        snapshot = load_korean_forest()
        source = _get_source_root(
            "REAP_KOREAN_FOREST_SOURCE_PATH", REPO_ROOT.parent / "green-narrative" / "hye_in"
        )
        variables = get_korean_forest_split_variables(source, list(snapshot.texts or []))
    data = {
        "X": np.asarray(snapshot.embeddings, dtype=np.float64),
        "variables": variables,
        "meta": {"version": snapshot.metadata.version, "sha256": snapshot.metadata.sha256},
    }
    if cell["strategy"] == "K1":
        pledges = load_korean_forest_oos_raw()
        data["X_external"] = np.asarray(pledges.embeddings, dtype=np.float64)
        data["meta"]["held_out_dataset"] = pledges.metadata.name
        data["meta"]["held_out_version"] = pledges.metadata.version
        data["meta"]["held_out_sha256"] = pledges.metadata.sha256
    return data


def _print_preflight(cell: dict, plans: list[tuple[Split | None, np.ndarray, np.ndarray]],
                     variables: SplitVariables) -> None:
    """Log the split sizes, fold-floor checks, group checks and leakage audit."""
    n_neighbors = 15
    for split, X_in, X_out in plans:
        label = "single split" if split is None or split.replicate is None else (
            f"replicate {split.replicate}")
        logger.info("%s %s: n_in=%d n_out=%d", cell["cell_id"], label, len(X_in), len(X_out))
        if len(X_out) < 2 * n_neighbors + 2:
            logger.warning("held-out set below the trustworthiness floor (n < 32)")
        if split is not None:
            shared = set(variables.groups[split.in_idx]) & set(variables.groups[split.out_idx])
            logger.info("groups on both sides: %d", len(shared))
        logger.info("leakage audit: %s", json.dumps(get_leakage_audit(X_in, X_out)))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code.

    Side effects: see the module docstring.
    """
    parser = argparse.ArgumentParser(description="Run one in-scope hold-out cell.")
    parser.add_argument("--cell", required=True, help="Run-matrix cell id, e.g. ai_art/T1.")
    parser.add_argument("--research-root", default=None, help="Override the research root.")
    parser.add_argument("--n-resamples", type=int, default=2000, help="Subsample draws.")
    parser.add_argument("--preflight", action="store_true", help="Check and report only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    matrix = json.loads(RUN_MATRIX_PATH.read_text())
    cell = get_cell(matrix, args.cell)
    seeds = get_set_d_seeds(REPO_ROOT / SEED_MANIFEST_RELPATH)
    data = _load_cell_data(cell)
    X, variables = data["X"], data["variables"]

    if cell["strategy"] == "K1":
        plans: list[tuple[Split | None, np.ndarray, np.ndarray]] = [
            (None, X, data["X_external"])
        ]
    else:
        plans = [(s, X[s.in_idx], X[s.out_idx]) for s in get_splits(cell, variables)]

    if args.preflight:
        _print_preflight(cell, plans, variables)
        return 0

    validate_code_is_merged(REPO_ROOT)
    research_root = get_research_root(args.research_root)
    output_dirs = [
        get_output_dir(research_root, cell, None if s is None else s.replicate)
        for s, _, _ in plans
    ]
    for output_dir in output_dirs:
        validate_output_dir_is_free(output_dir)
    for (split, X_in, X_out), output_dir in zip(plans, output_dirs):
        started = time.time()
        groups_out = None
        if split is not None and split.master_seed is not None:
            groups_out = variables.groups[split.out_idx]
        run_replicate(
            cell=cell, split=split, X_in=X_in, X_out=X_out, set_d_seeds=seeds,
            output_dir=output_dir, dataset_meta=data["meta"], n_resamples=args.n_resamples,
            groups_out=groups_out, head_settings=HeadTrainingSettings(), umap_epochs=50,
            code_root=REPO_ROOT,
        )
        logger.info("%s done in %.0f min -> %s", cell["cell_id"], (time.time() - started) / 60,
                    output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
