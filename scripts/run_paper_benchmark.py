"""Run the protocol-compliant REAP benchmark sweep for the paper.

Writes the full output layout prescribed by
``manuscript/evaluation_protocol.md`` §7 and §12 to ``--output-dir``
(default ``results/``)::

    results/<dataset>/<method>/set_<X>/
        per_seed.csv                # 30 rows x (seed, Cell A, Cell B, Cell C,
                                    #            trust, continuity, db, ch, k)
        s2s_ari.csv, s2s_ami.csv, s2s_nmi.csv    # 30x30 matrices
        s2c.csv                     # 30 rows: seed, ari, ami, nmi
        consensus.json              # Cell D + all consensus singletons + K + runtime + peak_mb
        external_validity.json      # only if ground_truth_labels provided
        topic_metrics.json          # only if texts provided
        cluster_persistence.json    # consensus methods only
        consensus_embedding.npy     # consensus coords (N, d)   -- consensus methods only
        consensus_labels.npy        # (N,) int                  -- consensus methods only
        consensus_distance_matrix.npy                          -- REAP only
        seed_labels.npy             # (n_seeds, N) int
        bundle.json                 # provenance per §12

    results/<dataset>/combined_set_<X>/
        all_methods.csv             # one row per method, headline summary

    results/<dataset>/
        between_set_reliability.csv # pairwise {ARI, AMI, NMI, Frobenius} across A/B/C
        pairwise_tests.csv          # paired Wilcoxon + Cohen's d + Holm/BH per metric

After all three seed sets complete for a dataset, the between-set
reliability and paired-test tables are computed from the per-set artifacts
and written to the dataset-level directory.

Usage
-----
Korean forest, all three seed sets::

    python scripts/run_paper_benchmark.py --dataset korean_forest

AI-art, all three seed sets::

    python scripts/run_paper_benchmark.py --dataset ai_art

Both datasets::

    python scripts/run_paper_benchmark.py --dataset korean_forest --dataset ai_art

Quick smoke (first 5 seeds of Set A only, no BERTopic)::

    python scripts/run_paper_benchmark.py --dataset korean_forest --quick --skip-bertopic

Synthetic blobs (no real data needed)::

    python scripts/run_paper_benchmark.py --dataset synthetic --skip-bertopic

Side effects
------------
- Reads from ``~/.cache/reap/datasets/`` (real datasets).
- Writes result CSVs + JSONs + .npy arrays + bundle.json under
  ``--output-dir``.
- CPU-bound: ~15-25 min per dataset x seed set on M-series Mac at 30 seeds.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reap.benchmarks import (  # noqa: E402
    ALL_METHODS,
    BenchmarkArtifacts,
    BenchmarkResult,
    MethodResult,
    run_benchmark_with_artifacts,
)
from reap.datasets import (  # noqa: E402
    DatasetSnapshot,
    load_ai_art,
    load_korean_forest,
    load_twenty_newsgroups_reference,
)
from reap.statistics import (  # noqa: E402
    compare_family,
    compute_seed_bootstrap_ci,
)

logger = logging.getLogger(__name__)

SEED_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent
    / "manuscript"
    / "seeds"
    / "seed_manifest.json"
)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Fixed UMAP + K params per dataset (selected upstream, frozen before benchmark)."""

    display_name: str
    n_components: int
    n_neighbors: int
    min_dist: float
    k_range: list[int]
    loader: Any


def _load_synthetic() -> DatasetSnapshot:
    """Synthetic blobs for quick pipeline testing (no real data needed)."""
    from sklearn.datasets import make_blobs

    from reap.datasets._schema import DatasetMetadata, DatasetSnapshot

    X, y = make_blobs(
        n_samples=400, n_features=384, centers=8,
        cluster_std=2.5, center_box=(-3.0, 3.0), random_state=42,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    # Fake metadata for a synthetic fixture
    meta = DatasetMetadata(
        name="synthetic_blobs_v1",
        version="1.0.0",
        n_samples=int(X.shape[0]),
        embedding_dim=int(X.shape[1]),
        embedding_model="synthetic/make_blobs",
        preprocessing_version="v1",
        sha256="0" * 64,
        license="public-domain",
        citation=None,
        source_url=None,
        notes="Synthetic blobs for smoke testing the benchmark harness.",
    )
    return DatasetSnapshot(embeddings=X, texts=None, labels=y, metadata=meta)


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "korean_forest": DatasetConfig(
        display_name="Korean forest policy",
        n_components=18,
        n_neighbors=19,
        min_dist=0.005,
        k_range=list(range(5, 31)),
        loader=load_korean_forest,
    ),
    "ai_art": DatasetConfig(
        display_name="AI-art discourse",
        n_components=5,
        n_neighbors=53,
        min_dist=0.01,
        k_range=list(range(5, 31)),
        loader=load_ai_art,
    ),
    "twenty_newsgroups_reference": DatasetConfig(
        display_name="20 Newsgroups (8-class reference)",
        n_components=5,
        n_neighbors=15,
        min_dist=0.1,
        k_range=list(range(5, 21)),
        loader=load_twenty_newsgroups_reference,
    ),
    "synthetic": DatasetConfig(
        display_name="Synthetic blobs",
        n_components=5,
        n_neighbors=15,
        min_dist=0.1,
        k_range=list(range(3, 12)),
        loader=_load_synthetic,
    ),
}

SEED_SET_NAMES = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Git / environment helpers
# ---------------------------------------------------------------------------


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_seed_sets(manifest_path: Path) -> dict[str, list[int]]:
    """Load Sets A/B/C from the pre-registered manifest."""
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return {name: manifest["sets"][name]["seeds"] for name in SEED_SET_NAMES}


# ---------------------------------------------------------------------------
# Per-method CSV / JSON emission
# ---------------------------------------------------------------------------


def _write_per_seed_csv(m: MethodResult, path: Path, seeds: list[int]) -> None:
    """Per-seed table with Cell A, Cell B (if any), Cell C (if any), trust, continuity, DB, CH, K."""
    n = len(m.per_seed_silhouette)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "seed_index", "seed_value",
            "silhouette_A_self",
            "silhouette_B_seed_labels_on_consensus",
            "silhouette_C_consensus_labels_on_seed",
            "trustworthiness", "continuity",
            "davies_bouldin", "calinski_harabasz", "k",
        ]
        w.writerow(header)
        for i in range(n):
            row = [
                i, seeds[i] if i < len(seeds) else "",
                _fmt(m.per_seed_silhouette[i] if i < len(m.per_seed_silhouette) else None),
                _fmt(m.cell_b_silhouette[i] if m.cell_b_silhouette else None),
                _fmt(m.cell_c_silhouette[i] if m.cell_c_silhouette else None),
                _fmt(m.per_seed_trustworthiness[i] if i < len(m.per_seed_trustworthiness) else None),
                _fmt(m.per_seed_continuity[i] if i < len(m.per_seed_continuity) else None),
                _fmt(m.per_seed_davies_bouldin[i] if i < len(m.per_seed_davies_bouldin) else None),
                _fmt(m.per_seed_calinski_harabasz[i] if i < len(m.per_seed_calinski_harabasz) else None),
                m.per_seed_k[i] if i < len(m.per_seed_k) else "",
            ]
            w.writerow(row)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (v != v):  # NaN
        return ""
    return f"{float(v):.6f}"


def _write_matrix_csv(matrix: list[list[float]] | None, path: Path) -> None:
    if matrix is None:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in matrix:
            w.writerow([f"{float(v):.6f}" for v in row])


def _write_s2c_csv(m: MethodResult, path: Path, seeds: list[int]) -> None:
    if m.s2c_ari_list is None:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed_index", "seed_value", "ari", "ami", "nmi"])
        for i, (ari, ami, nmi) in enumerate(
            zip(m.s2c_ari_list, m.s2c_ami_list or [], m.s2c_nmi_list or [])
        ):
            w.writerow([
                i, seeds[i] if i < len(seeds) else "",
                _fmt(ari), _fmt(ami), _fmt(nmi),
            ])


def _write_consensus_json(m: MethodResult, path: Path) -> None:
    data = {
        "method": m.method,
        "description": m.description,
        "n_seeds": m.n_seeds,
        "runtime_seconds": m.runtime_seconds,
        "peak_memory_mb": m.peak_memory_mb,
        "consensus": {
            "k": m.consensus_k,
            "silhouette_cell_D": m.consensus_silhouette,
            "trustworthiness": m.consensus_trustworthiness,
            "continuity": m.consensus_continuity,
            "davies_bouldin": m.consensus_davies_bouldin,
            "calinski_harabasz": m.consensus_calinski_harabasz,
        },
        "headline_summary": {
            "silhouette": m.silhouette,
            "silhouette_std": m.silhouette_std,
            "trustworthiness": m.trustworthiness,
            "trustworthiness_std": m.trustworthiness_std,
            "best_k": m.best_k,
            "best_k_std": m.best_k_std,
        },
        "s2s_summary": {
            "ari_mean": m.seed_to_seed_ari_mean,
            "ari_std": m.seed_to_seed_ari_std,
            "ami_mean": m.s2s_ami_mean,
            "ami_std": m.s2s_ami_std,
            "nmi_mean": m.s2s_nmi_mean,
            "nmi_std": m.s2s_nmi_std,
        },
        "s2c_summary": {
            "ari_mean": m.seed_to_consensus_ari_mean,
            "ari_std": m.s2c_ari_std,
            "ami_mean": m.s2c_ami_mean,
            "ami_std": m.s2c_ami_std,
            "nmi_mean": m.s2c_nmi_mean,
            "nmi_std": m.s2c_nmi_std,
        },
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _write_external_validity_json(m: MethodResult, path: Path) -> None:
    if m.ext_ari is None:
        return
    data = {
        "ari": m.ext_ari,
        "ami": m.ext_ami,
        "nmi": m.ext_nmi,
        "homogeneity": m.ext_homogeneity,
        "completeness": m.ext_completeness,
        "v_measure": m.ext_v_measure,
        "fowlkes_mallows": m.ext_fowlkes_mallows,
        "mean_purity": m.ext_mean_purity,
        "purity_per_cluster": m.ext_purity_per_cluster,
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _write_topic_metrics_json(m: MethodResult, path: Path) -> None:
    if m.topic_diversity is None:
        return
    data = {
        "topic_coherence_umass": m.topic_coherence_umass,
        "topic_coherence_npmi": m.topic_coherence_npmi,
        "topic_coherence_cv": m.topic_coherence_cv,
        "topic_diversity": m.topic_diversity,
        "topic_exclusivity": m.topic_exclusivity,
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _write_persistence_json(m: MethodResult, path: Path) -> None:
    if m.cluster_persistence_mean is None:
        return
    data = {
        "mean": m.cluster_persistence_mean,
        "min": m.cluster_persistence_min,
        "median": m.cluster_persistence_median,
        "per_cluster": m.cluster_persistence_per_cluster,
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def _write_bundle_json(
    path: Path,
    dataset_name: str,
    method: str,
    seed_set_name: str,
    seeds: list[int],
    runtime_seconds: float,
    peak_memory_mb: float | None,
    config: DatasetConfig,
    started_at: str,
) -> None:
    bundle = {
        "git_commit": _get_git_commit(),
        "script": "scripts/run_paper_benchmark.py",
        "method": method,
        "dataset": dataset_name,
        "dataset_display_name": config.display_name,
        "dataset_manifest_entry": f"manuscript/datasets/manifest.json#{dataset_name}",
        "seed_set": seed_set_name,
        "seeds": seeds,
        "seed_manifest": str(SEED_MANIFEST_PATH.name),
        "config": {
            "n_components": config.n_components,
            "n_neighbors": config.n_neighbors,
            "min_dist": config.min_dist,
            "k_range": config.k_range,
        },
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "started_at": started_at,
        "runtime_seconds": round(runtime_seconds, 3),
        "peak_memory_mb": (
            round(peak_memory_mb, 3) if peak_memory_mb is not None else None
        ),
    }
    path.write_text(json.dumps(bundle, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")


def _write_all_methods_summary(
    result: BenchmarkResult, path: Path, decimals: int = 6,
) -> None:
    fields = [
        "method", "description", "n_seeds", "runtime_seconds", "peak_memory_mb",
        "silhouette", "silhouette_std",
        "trustworthiness", "trustworthiness_std",
        "continuity_mean", "continuity_std",
        "davies_bouldin_mean", "davies_bouldin_std",
        "calinski_harabasz_mean", "calinski_harabasz_std",
        "best_k", "best_k_std",
        "consensus_silhouette", "consensus_trustworthiness", "consensus_continuity",
        "consensus_davies_bouldin", "consensus_calinski_harabasz", "consensus_k",
        "s2s_ari_mean", "s2s_ari_std",
        "s2s_ami_mean", "s2s_ami_std",
        "s2s_nmi_mean", "s2s_nmi_std",
        "s2c_ari_mean", "s2c_ari_std",
        "s2c_ami_mean", "s2c_ami_std",
        "s2c_nmi_mean", "s2c_nmi_std",
        "ext_ari", "ext_ami", "ext_nmi",
        "ext_homogeneity", "ext_completeness", "ext_v_measure",
        "ext_fowlkes_mallows", "ext_mean_purity",
        "topic_coherence_umass", "topic_coherence_npmi", "topic_coherence_cv",
        "topic_diversity", "topic_exclusivity",
        "cluster_persistence_mean", "cluster_persistence_min",
        "cluster_persistence_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in result.methods:
            cont_arr = np.asarray(m.per_seed_continuity)
            db_arr = np.asarray(m.per_seed_davies_bouldin)
            ch_arr = np.asarray(m.per_seed_calinski_harabasz)
            row: dict[str, Any] = {
                "method": m.method,
                "description": m.description,
                "n_seeds": m.n_seeds,
                "runtime_seconds": round(m.runtime_seconds, decimals),
                "peak_memory_mb": (
                    round(m.peak_memory_mb, decimals)
                    if m.peak_memory_mb is not None else ""
                ),
                "silhouette": _round(m.silhouette, decimals),
                "silhouette_std": _round(m.silhouette_std, decimals),
                "trustworthiness": _round(m.trustworthiness, decimals),
                "trustworthiness_std": _round(m.trustworthiness_std, decimals),
                "continuity_mean": _round(float(cont_arr.mean()) if cont_arr.size else None, decimals),
                "continuity_std": _round(float(cont_arr.std()) if cont_arr.size else None, decimals),
                "davies_bouldin_mean": _round(float(db_arr.mean()) if db_arr.size else None, decimals),
                "davies_bouldin_std": _round(float(db_arr.std()) if db_arr.size else None, decimals),
                "calinski_harabasz_mean": _round(float(ch_arr.mean()) if ch_arr.size else None, decimals),
                "calinski_harabasz_std": _round(float(ch_arr.std()) if ch_arr.size else None, decimals),
                "best_k": m.best_k,
                "best_k_std": _round(m.best_k_std, decimals),
                "consensus_silhouette": _round(m.consensus_silhouette, decimals),
                "consensus_trustworthiness": _round(m.consensus_trustworthiness, decimals),
                "consensus_continuity": _round(m.consensus_continuity, decimals),
                "consensus_davies_bouldin": _round(m.consensus_davies_bouldin, decimals),
                "consensus_calinski_harabasz": _round(m.consensus_calinski_harabasz, decimals),
                "consensus_k": m.consensus_k if m.consensus_k is not None else "",
                "s2s_ari_mean": _round(m.seed_to_seed_ari_mean, decimals),
                "s2s_ari_std": _round(m.seed_to_seed_ari_std, decimals),
                "s2s_ami_mean": _round(m.s2s_ami_mean, decimals),
                "s2s_ami_std": _round(m.s2s_ami_std, decimals),
                "s2s_nmi_mean": _round(m.s2s_nmi_mean, decimals),
                "s2s_nmi_std": _round(m.s2s_nmi_std, decimals),
                "s2c_ari_mean": _round(m.seed_to_consensus_ari_mean, decimals),
                "s2c_ari_std": _round(m.s2c_ari_std, decimals),
                "s2c_ami_mean": _round(m.s2c_ami_mean, decimals),
                "s2c_ami_std": _round(m.s2c_ami_std, decimals),
                "s2c_nmi_mean": _round(m.s2c_nmi_mean, decimals),
                "s2c_nmi_std": _round(m.s2c_nmi_std, decimals),
                "ext_ari": _round(m.ext_ari, decimals),
                "ext_ami": _round(m.ext_ami, decimals),
                "ext_nmi": _round(m.ext_nmi, decimals),
                "ext_homogeneity": _round(m.ext_homogeneity, decimals),
                "ext_completeness": _round(m.ext_completeness, decimals),
                "ext_v_measure": _round(m.ext_v_measure, decimals),
                "ext_fowlkes_mallows": _round(m.ext_fowlkes_mallows, decimals),
                "ext_mean_purity": _round(m.ext_mean_purity, decimals),
                "topic_coherence_umass": _round(m.topic_coherence_umass, decimals),
                "topic_coherence_npmi": _round(m.topic_coherence_npmi, decimals),
                "topic_coherence_cv": _round(m.topic_coherence_cv, decimals),
                "topic_diversity": _round(m.topic_diversity, decimals),
                "topic_exclusivity": _round(m.topic_exclusivity, decimals),
                "cluster_persistence_mean": _round(m.cluster_persistence_mean, decimals),
                "cluster_persistence_min": _round(m.cluster_persistence_min, decimals),
                "cluster_persistence_median": _round(m.cluster_persistence_median, decimals),
            }
            w.writerow(row)


def _round(v: Any, decimals: int) -> Any:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, float) and (v != v):
        return ""
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# Main per-(dataset, seed-set) runner
# ---------------------------------------------------------------------------


def _persist_method_artifacts(
    method: MethodResult,
    artifacts: BenchmarkArtifacts,
    method_dir: Path,
    seeds: list[int],
    dataset_name: str,
    seed_set_name: str,
    config: DatasetConfig,
    started_at: str,
) -> None:
    """Write every artifact for one (dataset, method, seed_set) cell."""
    method_dir.mkdir(parents=True, exist_ok=True)
    _write_per_seed_csv(method, method_dir / "per_seed.csv", seeds)
    _write_matrix_csv(method.s2s_ari_matrix, method_dir / "s2s_ari.csv")
    _write_matrix_csv(method.s2s_ami_matrix, method_dir / "s2s_ami.csv")
    _write_matrix_csv(method.s2s_nmi_matrix, method_dir / "s2s_nmi.csv")
    _write_s2c_csv(method, method_dir / "s2c.csv", seeds)
    _write_consensus_json(method, method_dir / "consensus.json")
    _write_external_validity_json(method, method_dir / "external_validity.json")
    _write_topic_metrics_json(method, method_dir / "topic_metrics.json")
    _write_persistence_json(method, method_dir / "cluster_persistence.json")

    # Numpy artifacts
    arts = artifacts.per_method.get(method.method, {})
    if "consensus_embedding" in arts:
        np.save(method_dir / "consensus_embedding.npy", arts["consensus_embedding"])
    if "consensus_labels" in arts:
        np.save(method_dir / "consensus_labels.npy", arts["consensus_labels"])
    if "consensus_distance_matrix" in arts:
        np.save(
            method_dir / "consensus_distance_matrix.npy",
            arts["consensus_distance_matrix"],
        )
    if "seed_labels" in arts:
        seed_labels = np.stack(arts["seed_labels"])
        np.save(method_dir / "seed_labels.npy", seed_labels)

    _write_bundle_json(
        method_dir / "bundle.json",
        dataset_name=dataset_name,
        method=method.method,
        seed_set_name=seed_set_name,
        seeds=seeds,
        runtime_seconds=method.runtime_seconds,
        peak_memory_mb=method.peak_memory_mb,
        config=config,
        started_at=started_at,
    )


def run_one_dataset_seed_set(
    dataset_name: str,
    seed_set_name: str,
    seeds: list[int],
    output_dir: Path,
    methods: list[str],
) -> BenchmarkResult:
    """Run the full benchmark for one dataset x seed set and persist all artifacts."""
    cfg = DATASET_CONFIGS[dataset_name]
    logger.info(
        "=== %s / Set %s (%d seeds, methods=%s) ===",
        cfg.display_name, seed_set_name, len(seeds), methods,
    )

    snap: DatasetSnapshot = cfg.loader()
    X = snap.embeddings
    texts = snap.texts
    labels = snap.labels

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result, artifacts = run_benchmark_with_artifacts(
        X=X,
        dataset_name=cfg.display_name,
        seeds=seeds,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        k_range=cfg.k_range,
        methods=methods,
        ground_truth_labels=labels,
        texts=texts,
    )

    # Emit per method
    dataset_root = output_dir / dataset_name
    for m in result.methods:
        method_dir = dataset_root / m.method / f"set_{seed_set_name}"
        _persist_method_artifacts(
            m, artifacts, method_dir, seeds, dataset_name, seed_set_name,
            cfg, started_at,
        )

    # Combined summary CSV for this seed set
    combined_dir = dataset_root / f"combined_set_{seed_set_name}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    _write_all_methods_summary(result, combined_dir / "all_methods.csv")

    logger.info(
        "%s / Set %s done. Results in %s/",
        cfg.display_name, seed_set_name, dataset_root,
    )
    return result


# ---------------------------------------------------------------------------
# Post-hoc: between-set reliability and paired tests
# ---------------------------------------------------------------------------


def _load_method_set(
    dataset_root: Path, method: str, seed_set_name: str,
) -> dict[str, Any] | None:
    """Load per-set artifacts for (method, seed_set) if present."""
    method_dir = dataset_root / method / f"set_{seed_set_name}"
    if not method_dir.exists():
        return None
    d: dict[str, Any] = {"dir": method_dir}
    for fname, key in [
        ("consensus_labels.npy", "consensus_labels"),
        ("consensus_embedding.npy", "consensus_embedding"),
        ("consensus_distance_matrix.npy", "consensus_distance_matrix"),
        ("seed_labels.npy", "seed_labels"),
    ]:
        fpath = method_dir / fname
        if fpath.exists():
            d[key] = np.load(fpath)
    # per_seed.csv for Cell C distribution
    per_seed = method_dir / "per_seed.csv"
    if per_seed.exists():
        d["per_seed_rows"] = _read_csv(per_seed)
    return d


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_between_set_reliability(
    dataset_root: Path,
    methods: list[str],
    seed_sets: list[str],
) -> None:
    """Pairwise consensus-agreement and (REAP) Frobenius distance across seed sets.

    For each method, computes ARI / AMI / NMI between consensus labels produced
    by disjoint seed sets. For REAP, additionally computes the relative
    Frobenius distance between consensus distance matrices
    ``||D_a - D_b||_F / ||D_a||_F``.

    Writes ``<dataset_root>/between_set_reliability.csv``.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    rows: list[dict[str, Any]] = []
    for method in methods:
        per_set: dict[str, dict[str, Any]] = {}
        for ss in seed_sets:
            data = _load_method_set(dataset_root, method, ss)
            if data is not None and "consensus_labels" in data:
                per_set[ss] = data
        if len(per_set) < 2:
            logger.info(
                "between_set_reliability: skipping %s (only %d sets available)",
                method, len(per_set),
            )
            continue
        names = sorted(per_set)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                la = per_set[a]["consensus_labels"]
                lb = per_set[b]["consensus_labels"]
                ari = float(adjusted_rand_score(la, lb))
                ami = float(adjusted_mutual_info_score(la, lb))
                nmi = float(normalized_mutual_info_score(la, lb))
                frob = None
                if (
                    method == "reap"
                    and "consensus_distance_matrix" in per_set[a]
                    and "consensus_distance_matrix" in per_set[b]
                ):
                    da = per_set[a]["consensus_distance_matrix"]
                    db = per_set[b]["consensus_distance_matrix"]
                    num = float(np.linalg.norm(da - db))
                    den = float(np.linalg.norm(da))
                    frob = num / den if den > 0 else 0.0
                rows.append({
                    "method": method,
                    "set_a": a, "set_b": b,
                    "ari": _round(ari, 6),
                    "ami": _round(ami, 6),
                    "nmi": _round(nmi, 6),
                    "frobenius_relative": _round(frob, 6) if frob is not None else "",
                })

    out = dataset_root / "between_set_reliability.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "set_a", "set_b", "ari", "ami", "nmi", "frobenius_relative"],
        )
        w.writeheader()
        w.writerows(rows)
    logger.info("between_set_reliability -> %s (%d rows)", out, len(rows))


def compute_pairwise_tests(
    dataset_root: Path,
    methods: list[str],
    seed_sets: list[str],
    reference: str = "reap",
) -> None:
    """Paired Wilcoxon + Cohen's d + Holm/BH for REAP vs each baseline.

    For each seed set and each metric (silhouette_A, silhouette_C, trust,
    continuity, DB, CH, s2c_ari), pull the 30-length per-seed vectors from
    ``per_seed.csv`` and ``s2c.csv`` for each method, then pit REAP's vector
    against each baseline's via :func:`reap.statistics.compare_family`.

    Writes ``<dataset_root>/pairwise_tests.csv`` with one row per
    ``(seed_set, metric, baseline)``.
    """
    # Metric -> (column in per_seed.csv, direction)
    metric_sources: dict[str, tuple[str, str, str]] = {
        # (source, column, direction)
        "silhouette_A": ("per_seed", "silhouette_A_self", "higher_is_better"),
        "silhouette_C": ("per_seed", "silhouette_C_consensus_labels_on_seed", "higher_is_better"),
        "trustworthiness": ("per_seed", "trustworthiness", "higher_is_better"),
        "continuity": ("per_seed", "continuity", "higher_is_better"),
        "davies_bouldin": ("per_seed", "davies_bouldin", "lower_is_better"),
        "calinski_harabasz": ("per_seed", "calinski_harabasz", "higher_is_better"),
        "s2c_ari": ("s2c", "ari", "higher_is_better"),
        "s2c_ami": ("s2c", "ami", "higher_is_better"),
    }

    out_rows: list[dict[str, Any]] = []
    for ss in seed_sets:
        # Collect per-method vectors
        method_vectors: dict[str, dict[str, np.ndarray]] = {}
        for method in methods:
            method_dir = dataset_root / method / f"set_{ss}"
            ps = method_dir / "per_seed.csv"
            s2c = method_dir / "s2c.csv"
            if not ps.exists():
                continue
            ps_rows = _read_csv(ps)
            s2c_rows = _read_csv(s2c) if s2c.exists() else []
            mv: dict[str, np.ndarray] = {}
            for metric, (source, col, _dir) in metric_sources.items():
                rows = ps_rows if source == "per_seed" else s2c_rows
                values: list[float] = []
                for r in rows:
                    v = r.get(col, "")
                    if v == "":
                        values.append(float("nan"))
                    else:
                        try:
                            values.append(float(v))
                        except ValueError:
                            values.append(float("nan"))
                arr = np.asarray(values, dtype=np.float64)
                # Drop NaNs
                arr = arr[~np.isnan(arr)]
                mv[metric] = arr
            method_vectors[method] = mv

        if reference not in method_vectors:
            logger.warning(
                "pairwise_tests: reference %r missing for set %s, skipping",
                reference, ss,
            )
            continue

        for metric, (_source, _col, direction) in metric_sources.items():
            ref_arr = method_vectors[reference].get(metric)
            if ref_arr is None or ref_arr.size == 0:
                continue
            baselines: dict[str, np.ndarray] = {}
            for m, mv in method_vectors.items():
                if m == reference:
                    continue
                arr = mv.get(metric)
                if arr is not None and arr.size == ref_arr.size:
                    baselines[m] = arr
            if not baselines:
                continue
            family = compare_family(
                reference_values=ref_arr,
                baseline_values=baselines,
                reference_name=reference,
                metric_name=metric,
                direction=direction,  # type: ignore[arg-type]
            )
            for cmp_res, p_holm, p_bh in zip(
                family.comparisons, family.p_holm, family.p_bh
            ):
                # Add a bootstrap CI of the reference for reference context
                ci = compute_seed_bootstrap_ci(ref_arr, n_bootstrap=2000)
                out_rows.append({
                    "seed_set": ss,
                    "metric": metric,
                    "direction": direction,
                    "reference": reference,
                    "comparator": cmp_res.comparator,
                    "n": cmp_res.n,
                    "ref_mean": _round(ci.mean, 6),
                    "ref_ci_lower": _round(ci.ci_lower, 6),
                    "ref_ci_upper": _round(ci.ci_upper, 6),
                    "mean_diff": _round(cmp_res.mean_diff, 6),
                    "median_diff": _round(cmp_res.median_diff, 6),
                    "wilcoxon_W": _round(cmp_res.wilcoxon_W, 6),
                    "wilcoxon_p_raw": _round(cmp_res.wilcoxon_p, 6),
                    "wilcoxon_p_holm": _round(p_holm, 6),
                    "wilcoxon_p_bh": _round(p_bh, 6),
                    "cohens_d": _round(cmp_res.cohens_d, 4),
                    "cliffs_delta": _round(cmp_res.cliffs_delta, 4),
                })

    out = dataset_root / "pairwise_tests.csv"
    fields = [
        "seed_set", "metric", "direction", "reference", "comparator", "n",
        "ref_mean", "ref_ci_lower", "ref_ci_upper",
        "mean_diff", "median_diff",
        "wilcoxon_W", "wilcoxon_p_raw", "wilcoxon_p_holm", "wilcoxon_p_bh",
        "cohens_d", "cliffs_delta",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    logger.info("pairwise_tests -> %s (%d rows)", out, len(out_rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Protocol-compliant REAP benchmark sweep for the paper.",
    )
    parser.add_argument(
        "--dataset", action="append", choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset(s) to benchmark. Pass multiple times for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output root directory (default 'results/').",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke mode: first 5 seeds of each requested set.",
    )
    parser.add_argument(
        "--seed-sets", default="A,B,C",
        help="Comma-separated seed-set names (default 'A,B,C').",
    )
    parser.add_argument(
        "--skip-bertopic", action="store_true",
        help="Skip the BERTopic baseline (heavy install).",
    )
    parser.add_argument(
        "--skip-reliability", action="store_true",
        help="Skip between-set reliability and pairwise tests after the sweep.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    methods = list(ALL_METHODS)
    if args.skip_bertopic and "bertopic" in methods:
        methods.remove("bertopic")

    # Synthetic dataset: no real seed manifest needed
    synthetic_seeds = {
        "A": [42, 137, 7, 11, 2024],
        "B": [100, 101, 102, 103, 104],
        "C": [200, 201, 202, 203, 204],
    }
    try:
        manifest_sets = _load_seed_sets(SEED_MANIFEST_PATH)
    except Exception as exc:
        logger.warning("could not load seed manifest (%s); using synthetic fallback", exc)
        manifest_sets = synthetic_seeds

    output_dir = Path(args.output_dir)
    requested_sets = [s.strip() for s in args.seed_sets.split(",") if s.strip()]

    for dataset_name in args.dataset:
        dataset_root = output_dir / dataset_name
        # Choose seed source: real manifest unless synthetic (for which we use the fallback)
        seed_source = synthetic_seeds if dataset_name == "synthetic" else manifest_sets
        for set_name in requested_sets:
            if set_name not in seed_source:
                logger.error(
                    "unknown seed set %r (available: %s)", set_name, list(seed_source),
                )
                return 2
            seeds = seed_source[set_name]
            if args.quick:
                seeds = seeds[:5]
                logger.info("--quick: using first %d seeds of Set %s", len(seeds), set_name)
            run_one_dataset_seed_set(
                dataset_name, set_name, seeds, output_dir, methods,
            )

        if not args.skip_reliability:
            compute_between_set_reliability(dataset_root, methods, requested_sets)
            compute_pairwise_tests(
                dataset_root, methods, requested_sets, reference="reap",
            )

    logger.info("all benchmarks complete; results in %s/", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
