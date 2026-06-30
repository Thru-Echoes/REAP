"""Project AI-art OOS probes through REAP's projection head — empirical showcase.

For each consensus set (A, B, C) of the AI-art reference (1,736 chunks, 5-d
UMAP; pipeline targeted k=20 but effective K is 20/19/18 because empty
clusters are dropped, see TRACE evt_002), this script:

1. Trains a final projection head on the full reference with deterministic
   seed=42. The trained head MAP : R^1024 -> R^5.
2. Projects 1,259 Lovato 2024 artist-perspective probes and 750 style-
   controlled public probes through the head.
3. Quantifies cross-source patterns:
   - Cluster assignment (probes go to their nearest reference-cluster
     centroid in 5-d Euclidean; reference clusters are the pipeline's own
     consensus labels, not a fresh KMeans fit).
   - Occupancy histogram + KL(probe || reference).
   - k-NN purity (k=15) in 5-d cosine space.
   - Centroid cosine (5-d).
   - Theme x cluster contingency (chi^2 + Cramer's V).
   - Source-separability silhouette (binary ref-vs-probe label).
4. Reports per-reference-cluster year distribution (temporal alignment).
5. Compares probe coordinates across sets via orthogonal Procrustes
   (cross-set consistency check).

Outputs land under ``results/projection_head/ai_art/oos_demo/``:

    cross_set_consistency.json
    set_<S>/
        head_state_dict.pt
        head_config.json
        projection_artist.npy   (1259, 5)
        projection_public.npy   (750,  5)
        assignment_artist.npy   (1259,)
        assignment_public.npy   (750,)
        comparison_metrics.json
        temporal_alignment.json
        showcase_figure.png     (set A only)

Reproducibility: deterministic seeds for python, numpy, torch (CPU + CUDA),
``torch.use_deterministic_algorithms``, ``CUBLAS_WORKSPACE_CONFIG``. Git
commit and per-input SHA256s recorded in every output JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
WAMA_SOURCE_PATH = Path.home() / "Documents/Berkeley/Research/When-Algorithms-Meet-Artists"

logger = logging.getLogger(__name__)

SETS = ("A", "B", "C")
KNN_K = 15
# Note: the AI-art consensus pipeline produces a per-set K (sets A/B/C → 20/19/18),
# so all downstream array sizes derive from labels_ref rather than a hardcoded
# constant. This is a real discovery from the consensus run, not a config bug.


def _set_deterministic_seeds(seed: int) -> None:
    """Pin every relevant RNG (python, numpy, torch CPU+CUDA, hash seed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _sha256_of_array(arr: np.ndarray) -> str:
    """Hex SHA256 of the raw bytes of an array."""
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "<unknown>"


def _train_final_head(
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    labels_ref: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train one final projection head on the full reference.

    Returns (final_head, config). Uses ``train_projection_head`` with
    ``n_folds=2`` (we only need the final-fit ``model``; CV metrics for AI-art
    are already saved in ``results/projection_head/ai_art/set_<S>/oos_metrics.json``).

    Side effects: trains a torch model.
    """
    from reap.projection import train_projection_head

    _set_deterministic_seeds(seed)
    result = train_projection_head(
        X=X_ref,
        Y=Y_ref,
        labels=labels_ref,
        hidden_layers=[128, 64],
        dropout=0.3,
        alpha=0.7,
        n_folds=2,
        batch_size=64,
        max_epochs=500,
        patience=50,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
    )
    return result["model"], result["config"]


def _kl_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 1.0) -> float:
    """KL(p || q) with Laplace smoothing.

    Both inputs are count arrays of the same length. Adds ``alpha`` to every
    bin before normalizing.
    """
    p_smoothed = p.astype(np.float64) + alpha
    q_smoothed = q.astype(np.float64) + alpha
    p_norm = p_smoothed / p_smoothed.sum()
    q_norm = q_smoothed / q_smoothed.sum()
    return float(np.sum(p_norm * (np.log(p_norm) - np.log(q_norm))))


def _knn_purity(
    Y_probe: np.ndarray,
    Y_ref: np.ndarray,
    probe_assign: np.ndarray,
    ref_assign: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """For each probe, fraction of its ``k`` nearest reference neighbors that
    share its assigned cluster. Returns (mean, std). Uses cosine distance.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Y_ref)
    _, idx = nn.kneighbors(Y_probe)
    purities = np.array(
        [
            float(np.mean(ref_assign[neigh] == probe_assign[i]))
            for i, neigh in enumerate(idx)
        ]
    )
    return float(purities.mean()), float(purities.std())


def _compute_cluster_centroids(
    Y_ref: np.ndarray, ref_assign: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Return (n_clusters, d) centroids — mean of Y_ref grouped by cluster label."""
    centroids = np.zeros((n_clusters, Y_ref.shape[1]), dtype=np.float64)
    for c in range(n_clusters):
        mask = ref_assign == c
        if mask.any():
            centroids[c] = Y_ref[mask].mean(axis=0)
    return centroids


def _assign_probes_by_nearest_centroid(
    Y_probe: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Assign each probe to its nearest centroid (Euclidean) — matches KMeans.predict."""
    # (n_probe, 1, d) - (1, n_clusters, d) -> (n_probe, n_clusters)
    dists = np.linalg.norm(Y_probe[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


def _median_centroid_cosine(
    Y_probe: np.ndarray,
    centroids: np.ndarray,
    probe_assign: np.ndarray,
) -> float:
    """Median cosine between each probe and its assigned reference cluster centroid (5-d)."""

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    cosines = np.array(
        [_cos(Y_probe[i], centroids[probe_assign[i]]) for i in range(len(Y_probe))]
    )
    return float(np.median(cosines))


def _theme_cluster_contingency(
    theme_labels: np.ndarray,
    cluster_assign: np.ndarray,
    n_clusters: int,
    n_themes: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Build the (n_themes x n_clusters) contingency table + chi-square + Cramer's V."""
    from scipy.stats import chi2_contingency

    table = np.zeros((n_themes, n_clusters), dtype=np.int64)
    for t in range(n_themes):
        for c in range(n_clusters):
            table[t, c] = int(((theme_labels == t) & (cluster_assign == c)).sum())

    # Drop any all-zero columns to keep chi2_contingency happy.
    nonzero_cols = table.sum(axis=0) > 0
    sub = table[:, nonzero_cols]
    if sub.size == 0 or sub.sum() == 0:
        return table, 0.0, 0.0

    result = chi2_contingency(sub)
    chi2_raw: Any = result[0]  # scipy returns a namedtuple; runtime is a float-like scalar
    chi2_value = float(chi2_raw)
    n = int(sub.sum())
    r, c = sub.shape
    cramers_v = float(np.sqrt(chi2_value / (n * max(min(r - 1, c - 1), 1))))
    return table, chi2_value, cramers_v


def _source_separability_silhouette(Y_ref: np.ndarray, Y_probe: np.ndarray) -> float:
    """Silhouette on {ref ∪ probe} with binary source labels (0=ref, 1=probe)."""
    from sklearn.metrics import silhouette_score

    Y_concat = np.vstack([Y_ref, Y_probe])
    src_labels = np.concatenate(
        [np.zeros(len(Y_ref), dtype=np.int64), np.ones(len(Y_probe), dtype=np.int64)]
    )
    return float(silhouette_score(Y_concat, src_labels, metric="euclidean"))


def _compute_probe_source_metrics(
    Y_probe: np.ndarray,
    probe_theme_labels: np.ndarray,
    Y_ref: np.ndarray,
    ref_cluster_assign: np.ndarray,
    ref_occupancy: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    probe_source_name: str,
    probe_sha: str,
) -> dict[str, Any]:
    """Compute all per-source metrics in the comparison_metrics.json schema."""
    probe_assign = _assign_probes_by_nearest_centroid(Y_probe, centroids)
    occupancy_hist = np.bincount(probe_assign, minlength=n_clusters)

    kl = _kl_divergence(occupancy_hist, ref_occupancy)
    knn_mean, knn_std = _knn_purity(
        Y_probe, Y_ref, probe_assign, ref_cluster_assign, KNN_K
    )
    centroid_cos = _median_centroid_cosine(Y_probe, centroids, probe_assign)
    contingency, chi2, cramers_v = _theme_cluster_contingency(
        probe_theme_labels, probe_assign, n_clusters=n_clusters, n_themes=5
    )
    sep_sil = _source_separability_silhouette(Y_ref, Y_probe)

    return {
        "n": int(Y_probe.shape[0]),
        "input_dim": 1024,
        "output_dim": int(Y_probe.shape[1]),
        "probe_input_sha256": probe_sha,
        "probe_assignment": probe_assign.tolist(),  # for later debugging
        "occupancy_hist": occupancy_hist.astype(int).tolist(),
        "kl_to_reference": kl,
        "knn_purity_mean": knn_mean,
        "knn_purity_std": knn_std,
        "centroid_cosine_median": centroid_cos,
        "theme_cluster_contingency": contingency.astype(int).tolist(),
        "chi2_statistic": chi2,
        "cramers_v": cramers_v,
        "source_separability_silhouette": sep_sil,
        "probe_source_name": probe_source_name,
    }


def _compute_temporal_alignment(
    years: np.ndarray, ref_cluster_assign: np.ndarray, n_clusters: int
) -> dict[str, Any]:
    """Per-reference-cluster year distribution + temporal ordering."""
    per_cluster: list[dict[str, Any]] = []
    mean_years: list[float] = []
    for c in range(n_clusters):
        mask = ref_cluster_assign == c
        ys = years[mask]
        if len(ys) == 0:
            per_cluster.append(
                {"cluster": c, "n": 0, "mean_year": None, "q25": None, "q75": None, "histogram": {}}
            )
            mean_years.append(float("nan"))
            continue
        hist = {int(y): int(c_y) for y, c_y in zip(*np.unique(ys, return_counts=True))}
        per_cluster.append(
            {
                "cluster": c,
                "n": int(mask.sum()),
                "mean_year": float(np.mean(ys)),
                "median_year": float(np.median(ys)),
                "q25": float(np.percentile(ys, 25)),
                "q75": float(np.percentile(ys, 75)),
                "histogram": hist,
            }
        )
        mean_years.append(float(np.mean(ys)))

    # Ordering: clusters sorted by mean year (NaN clusters go last).
    finite_mask = np.isfinite(mean_years)
    finite_idx = np.where(finite_mask)[0]
    sorted_finite = finite_idx[np.argsort(np.array(mean_years)[finite_idx])]
    nan_idx = np.where(~finite_mask)[0]
    ordering = sorted_finite.tolist() + nan_idx.tolist()

    return {
        "per_cluster": per_cluster,
        "ordering_by_mean_year": [int(c) for c in ordering],
    }


def _procrustes_disparity(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Orthogonal Procrustes alignment of two (n, d) matrices, returns disparity.

    Uses scipy.spatial.procrustes which standardizes both inputs and reports
    the residual sum of squares (``disparity``) after the best similarity
    transform. Lower is more aligned (0 = identical up to rotation/scale).
    """
    from scipy.spatial import procrustes

    _, _, disparity = procrustes(A, B)
    return {"procrustes_disparity": float(disparity)}


def _save_head_state_dict(head: Any, out_path: Path) -> None:
    """Persist ProjectionHead._model.state_dict() to disk via torch.save."""
    head.save(str(out_path))


def _run_for_set(
    set_letter: str,
    X_ref: np.ndarray,
    artist_probes: Any,
    public_probes: Any,
    years: np.ndarray,
    seed: int,
    out_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    """End-to-end for one consensus set. Returns the (set-specific) bundle.

    The reference cluster assignment uses the pre-existing pipeline labels
    (`consensus_labels.npy`) directly, NOT a fresh KMeans fit. Probes are
    assigned to their nearest reference-cluster centroid in 5-d Euclidean
    space. The effective K varies per set because the consensus pipeline
    drops empty clusters (see TRACE annotation evt_002).
    """
    consensus_emb_path = (
        RESULTS_ROOT / "ai_art" / "reap" / f"set_{set_letter}" / "consensus_embedding.npy"
    )
    consensus_labels_path = (
        RESULTS_ROOT / "ai_art" / "reap" / f"set_{set_letter}" / "consensus_labels.npy"
    )
    Y_ref = np.load(consensus_emb_path).astype(np.float32)
    labels_ref = np.load(consensus_labels_path).astype(np.int64)

    n_clusters = int(labels_ref.max()) + 1
    expected_labels = np.arange(n_clusters)
    if not np.array_equal(np.sort(np.unique(labels_ref)), expected_labels):
        raise RuntimeError(
            f"set {set_letter}: consensus_labels are not contiguous 0..{n_clusters - 1}; "
            f"got unique values {sorted(np.unique(labels_ref).tolist())}"
        )

    logger.info(
        "[set_%s] training final head on X=%s, Y=%s, labels=%s, n_clusters=%d",
        set_letter,
        X_ref.shape,
        Y_ref.shape,
        labels_ref.shape,
        n_clusters,
    )
    head, head_config = _train_final_head(X_ref, Y_ref, labels_ref, seed)

    # Reference cluster assignment = pipeline's consensus labels (verbatim).
    # Centroids in 5-d Euclidean space → probe assignment by nearest centroid.
    ref_cluster_assign = labels_ref
    ref_occupancy = np.bincount(ref_cluster_assign, minlength=n_clusters)
    centroids = _compute_cluster_centroids(Y_ref, ref_cluster_assign, n_clusters)

    set_dir = out_root / f"set_{set_letter}"
    set_dir.mkdir(parents=True, exist_ok=True)
    head_path = set_dir / "head_state_dict.pt"
    _save_head_state_dict(head, head_path)

    with open(set_dir / "head_config.json", "w") as fp:
        json.dump(head_config, fp, indent=2)

    # Project both probe sources.
    Y_artist = head.forward(np.ascontiguousarray(artist_probes.embeddings, dtype=np.float32))
    Y_public = head.forward(np.ascontiguousarray(public_probes.embeddings, dtype=np.float32))
    np.save(set_dir / "projection_artist.npy", Y_artist)
    np.save(set_dir / "projection_public.npy", Y_public)

    # Compute per-source metrics.
    artist_metrics = _compute_probe_source_metrics(
        Y_probe=Y_artist,
        probe_theme_labels=artist_probes.labels,
        Y_ref=Y_ref,
        ref_cluster_assign=ref_cluster_assign,
        ref_occupancy=ref_occupancy,
        centroids=centroids,
        n_clusters=n_clusters,
        probe_source_name="artist",
        probe_sha=_sha256_of_array(artist_probes.embeddings),
    )
    public_metrics = _compute_probe_source_metrics(
        Y_probe=Y_public,
        probe_theme_labels=public_probes.labels,
        Y_ref=Y_ref,
        ref_cluster_assign=ref_cluster_assign,
        ref_occupancy=ref_occupancy,
        centroids=centroids,
        n_clusters=n_clusters,
        probe_source_name="public",
        probe_sha=_sha256_of_array(public_probes.embeddings),
    )

    artist_assign_arr = np.array(artist_metrics["probe_assignment"], dtype=np.int64)
    public_assign_arr = np.array(public_metrics["probe_assignment"], dtype=np.int64)
    np.save(set_dir / "assignment_artist.npy", artist_assign_arr)
    np.save(set_dir / "assignment_public.npy", public_assign_arr)
    # Drop the verbose probe_assignment array from JSON (it's already in the npy).
    del artist_metrics["probe_assignment"]
    del public_metrics["probe_assignment"]

    temporal = _compute_temporal_alignment(years, ref_cluster_assign, n_clusters)
    with open(set_dir / "temporal_alignment.json", "w") as fp:
        json.dump(temporal, fp, indent=2)

    comparison_payload = {
        "corpus": "ai_art",
        "set": set_letter,
        "git_commit": git_commit,
        "seed": seed,
        "n_ref": int(X_ref.shape[0]),
        "n_ref_clusters": n_clusters,
        "ref_input_sha256": _sha256_of_array(X_ref),
        "ref_target_sha256": _sha256_of_array(Y_ref),
        "head_config": head_config,
        "ref_occupancy": ref_occupancy.astype(int).tolist(),
        "probes": {
            "artist": artist_metrics,
            "public": public_metrics,
        },
    }
    with open(set_dir / "comparison_metrics.json", "w") as fp:
        json.dump(comparison_payload, fp, indent=2)

    logger.info(
        "[set_%s] artist: KL=%.4f knn_purity=%.4f cramers_v=%.4f sep_sil=%+.4f",
        set_letter,
        artist_metrics["kl_to_reference"],
        artist_metrics["knn_purity_mean"],
        artist_metrics["cramers_v"],
        artist_metrics["source_separability_silhouette"],
    )
    logger.info(
        "[set_%s] public: KL=%.4f knn_purity=%.4f cramers_v=%.4f sep_sil=%+.4f",
        set_letter,
        public_metrics["kl_to_reference"],
        public_metrics["knn_purity_mean"],
        public_metrics["cramers_v"],
        public_metrics["source_separability_silhouette"],
    )

    return {
        "set": set_letter,
        "Y_artist": Y_artist,
        "Y_public": Y_public,
        "Y_ref": Y_ref,
        "ref_cluster_assign": ref_cluster_assign,
    }


def _make_showcase_figure(
    set_letter: str,
    Y_ref: np.ndarray,
    ref_cluster_assign: np.ndarray,
    Y_artist: np.ndarray,
    Y_public: np.ndarray,
    out_path: Path,
) -> None:
    """2-D scatter (consensus dims 0,1) of reference + probes. Single seed, no UMAP."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    scat_ref = ax.scatter(
        Y_ref[:, 0],
        Y_ref[:, 1],
        c=ref_cluster_assign,
        cmap="tab20",
        s=10,
        alpha=0.55,
        linewidths=0,
        label="reference (1736 chunks)",
    )
    ax.scatter(
        Y_artist[:, 0],
        Y_artist[:, 1],
        marker="x",
        c="black",
        s=18,
        alpha=0.55,
        linewidths=0.6,
        label="Lovato 2024 artist probes (1259)",
    )
    ax.scatter(
        Y_public[:, 0],
        Y_public[:, 1],
        marker="+",
        c="red",
        s=22,
        alpha=0.6,
        linewidths=0.7,
        label="style-controlled public probes (750)",
    )
    ax.set_xlabel("consensus dim 0")
    ax.set_ylabel("consensus dim 1")
    ax.set_title(
        f"AI-art OOS demo (set {set_letter}): probes projected into reference space"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.colorbar(scat_ref, ax=ax, label="reference consensus cluster", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI args yet
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from reap.datasets import (
        load_ai_art,
        load_ai_art_oos_artist,
        load_ai_art_oos_public,
    )
    from reap.datasets.ai_art_oos import _get_ai_art_year_array

    seed = 42
    _set_deterministic_seeds(seed)
    git_commit = _current_git_commit()

    ref = load_ai_art()
    X_ref = np.ascontiguousarray(ref.embeddings, dtype=np.float32)
    if X_ref.shape != (1736, 1024):
        raise RuntimeError(f"unexpected AI-art reference shape: {X_ref.shape}")

    artist_probes = load_ai_art_oos_artist()
    public_probes = load_ai_art_oos_public()
    years = _get_ai_art_year_array(WAMA_SOURCE_PATH)
    if len(years) != 1736:
        raise RuntimeError(
            f"reference year array length {len(years)} != 1736 (WAMA chunks)"
        )

    out_root = RESULTS_ROOT / "projection_head" / "ai_art" / "oos_demo"
    out_root.mkdir(parents=True, exist_ok=True)

    set_bundles: dict[str, dict[str, Any]] = {}
    for set_letter in SETS:
        bundle = _run_for_set(
            set_letter=set_letter,
            X_ref=X_ref,
            artist_probes=artist_probes,
            public_probes=public_probes,
            years=years,
            seed=seed,
            out_root=out_root,
            git_commit=git_commit,
        )
        set_bundles[set_letter] = bundle

    # Cross-set Procrustes consistency on probe coordinates.
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    cross_set: dict[str, Any] = {
        "git_commit": git_commit,
        "seed": seed,
        "pairs": {},
    }
    for a, b in pairs:
        pair_key = f"{a}_vs_{b}"
        cross_set["pairs"][pair_key] = {
            "artist": _procrustes_disparity(
                set_bundles[a]["Y_artist"], set_bundles[b]["Y_artist"]
            ),
            "public": _procrustes_disparity(
                set_bundles[a]["Y_public"], set_bundles[b]["Y_public"]
            ),
        }
        logger.info(
            "Procrustes %s artist=%.4e public=%.4e",
            pair_key,
            cross_set["pairs"][pair_key]["artist"]["procrustes_disparity"],
            cross_set["pairs"][pair_key]["public"]["procrustes_disparity"],
        )
    with open(out_root / "cross_set_consistency.json", "w") as fp:
        json.dump(cross_set, fp, indent=2)

    # Generate showcase figure from set A (canonical view).
    fig_path = out_root / "set_A" / "showcase_figure.png"
    _make_showcase_figure(
        set_letter="A",
        Y_ref=set_bundles["A"]["Y_ref"],
        ref_cluster_assign=set_bundles["A"]["ref_cluster_assign"],
        Y_artist=set_bundles["A"]["Y_artist"],
        Y_public=set_bundles["A"]["Y_public"],
        out_path=fig_path,
    )
    logger.info("wrote showcase figure to %s", fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
