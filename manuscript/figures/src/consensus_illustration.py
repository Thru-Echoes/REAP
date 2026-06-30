"""Synthetic illustration: distance-matrix consensus vs Procrustes averaging.

Renders a 2-row figure on three well-separated 2-D Gaussian blobs.

    Row 1 (4 panels): four simulated UMAP runs of the same canonical
        embedding, each with (a) a different random rotation+reflection
        (the orthogonal ambiguity Procrustes is designed to remove) and
        (b) per-cluster local rotations (the non-rigid distortion that
        Procrustes cannot remove because it is not a global rigid
        motion).

    Row 2 (3 panels + 1 stats panel):
        - Naive coordinate average (collapses because rotations cancel).
        - Procrustes coordinate average (recovers global layout but
          blurs cluster shape because per-cluster local rotations
          differ across runs).
        - REAP distance-matrix consensus (recovers structure from the
          averaged pairwise-distance matrix via classical MDS; the
          relational geometry is intact because pairwise distances are
          invariant to the per-run orthogonal ambiguity by construction
          — §3.2 of the manuscript).
        - Stats panel: per-pair distance variance across runs vs
          per-coordinate variance across runs (after Procrustes
          alignment). Distance variance is much lower because pairwise
          distances are exactly invariant to the orthogonal-ambiguity
          component; coordinate variance retains the full non-rigid
          residual that Procrustes cannot remove. This is the
          mechanistic claim the figure makes.

Inputs
------
None (purely synthetic). Deterministic via `np.random.seed(RANDOM_SEED)`.

Outputs
-------
manuscript/figures/consensus_illustration.png
manuscript/figures/consensus_illustration.pdf

Why this figure
---------------
Geometric blob fixtures cannot exhibit REAP's *empirical* advantage on
real text (that is what `metrics_bars_*` show against real CSVs). This
figure illustrates the *mechanism*: coordinate averaging is blind to
the orthogonal ambiguity of UMAP outputs, while distance-matrix
consensus is invariant to it by construction (§3.2). The figure is a
methodological illustration, not an empirical claim — the caption
should label it as such.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

FIG_DIR = Path(__file__).resolve().parents[1]
OUT_PNG = FIG_DIR / "consensus_illustration.png"
OUT_PDF = FIG_DIR / "consensus_illustration.pdf"

NUM_RUNS = 4
NUM_POINTS_PER_CLUSTER = 60
NUM_CLUSTERS = 3
CLUSTER_STD = 0.45
RANDOM_SEED = 0
# Per-cluster local-rotation magnitude (radians). Each cluster's points
# are rotated around the cluster centroid by an angle uniformly drawn
# from [-PER_CLUSTER_ROT_MAX, +PER_CLUSTER_ROT_MAX] independently per
# run. Calibrated so Procrustes still recovers the bulk of the structure
# but leaves a non-trivial residual; distance-matrix averaging recovers
# slightly closer to the canonical layout.
PER_CLUSTER_ROT_MAX = 0.55


def get_canonical_embedding() -> tuple[np.ndarray, np.ndarray]:
    """Build the canonical 2-D layout used as the noiseless target.

    Returns
    -------
    coords : (N, 2) float array
    labels : (N,) int array of cluster assignments
    """
    rng = np.random.default_rng(RANDOM_SEED)
    centres = np.array([[-2.0, -1.5], [2.0, -1.5], [0.0, 2.0]])
    coords = np.vstack([
        centres[c] + CLUSTER_STD * rng.standard_normal((NUM_POINTS_PER_CLUSTER, 2))
        for c in range(NUM_CLUSTERS)
    ])
    labels = np.concatenate([
        np.full(NUM_POINTS_PER_CLUSTER, c, dtype=int)
        for c in range(NUM_CLUSTERS)
    ])
    return coords, labels


def get_random_orthogonal_2d(rng: np.random.Generator) -> np.ndarray:
    """Return a random 2x2 orthogonal matrix (rotation + optional reflection)."""
    theta = rng.uniform(0.0, 2.0 * np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    if rng.random() < 0.5:
        R = R @ np.array([[1.0, 0.0], [0.0, -1.0]])
    return R


def get_simulated_umap_runs(
    canonical: np.ndarray,
    labels: np.ndarray,
) -> list[np.ndarray]:
    """Simulate per-run UMAP variability with non-rigid per-cluster distortion.

    Real UMAP runs vary across seeds in two distinct ways:
      1. A global rotation/reflection of the embedding (Procrustes is
         designed to remove this).
      2. A *non-rigid* per-cluster distortion: each cluster is locally
         rotated by a small random angle around its own centroid, mimicking
         the way different seeds resolve the same local neighbourhood
         differently. Procrustes alignment cannot remove these because
         they are not part of a global rigid motion. Averaging coordinates
         after Procrustes blurs the cluster shapes (a different per-run
         local rotation contributes to each averaged coordinate).
         Averaging *distance matrices*, by contrast, preserves the
         per-pair distance distribution and re-embeds via classical MDS
         into the closest-to-canonical layout.

    This is the methodological asymmetry §3.1 / §3.2 of the manuscript
    references; the figure visualises it. The simulation is *not* an
    empirical claim about UMAP magnitudes — those are in the bar-chart
    figures against real CSVs.
    """
    rng = np.random.default_rng(RANDOM_SEED + 7)
    centres = np.stack([
        canonical[labels == c].mean(axis=0) for c in range(NUM_CLUSTERS)
    ])
    runs: list[np.ndarray] = []
    for _ in range(NUM_RUNS):
        # Per-cluster local rotations (in canonical coords).
        distorted = canonical.copy()
        for c in range(NUM_CLUSTERS):
            theta_local = rng.uniform(-PER_CLUSTER_ROT_MAX, PER_CLUSTER_ROT_MAX)
            R_local = np.array([
                [np.cos(theta_local), -np.sin(theta_local)],
                [np.sin(theta_local),  np.cos(theta_local)],
            ])
            mask = labels == c
            centred = canonical[mask] - centres[c]
            distorted[mask] = centred @ R_local + centres[c]
        # Global random rotation + reflection.
        Q = get_random_orthogonal_2d(rng)
        t = rng.normal(scale=0.2, size=2)
        runs.append(distorted @ Q + t)
    return runs


def get_procrustes_consensus(runs: list[np.ndarray]) -> np.ndarray:
    """Orthogonal Procrustes alignment to runs[0], then coordinate average."""
    reference = runs[0]
    aligned = [reference.copy()]
    for emb in runs[1:]:
        # Centre both reference and emb (Procrustes invariance to translation)
        ref_c = reference - reference.mean(axis=0)
        emb_c = emb - emb.mean(axis=0)
        u, _, vt = np.linalg.svd(emb_c.T @ ref_c, full_matrices=False)
        Q = u @ vt
        aligned.append(emb_c @ Q + reference.mean(axis=0))
    return np.mean(np.stack(aligned, axis=0), axis=0)


def get_distance_consensus_2d(runs: list[np.ndarray]) -> np.ndarray:
    """Average pairwise-distance matrices, then classical-MDS to 2-D.

    Notes
    -----
    Classical MDS rather than UMAP because (a) this is a 2-D synthetic and
    UMAP at N=180 is overkill, (b) MDS gives a deterministic, principled
    re-embedding of the averaged distance matrix that the reader can
    interpret as "the structure encoded in the averaged distances". The
    real REAP pipeline uses UMAP with `metric="precomputed"`; the
    illustration is for *why* averaging distances works, not for matching
    UMAP's exact output.
    """
    n = runs[0].shape[0]
    d_sum = np.zeros((n, n), dtype=float)
    for emb in runs:
        diff = emb[:, None, :] - emb[None, :, :]
        d_sum += np.sqrt((diff ** 2).sum(axis=-1))
    d_mean = d_sum / len(runs)
    # Classical MDS: double-centre then take top-2 eigenvectors of -0.5 * J D^2 J
    sq = d_mean ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ sq @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    # Top 2 eigenvalues (eigh returns ascending).
    top_idx = np.argsort(eigvals)[::-1][:2]
    coords = eigvecs[:, top_idx] * np.sqrt(np.clip(eigvals[top_idx], 0.0, None))
    return coords


def get_normalised_distance_vector(coords: np.ndarray) -> np.ndarray:
    """Flattened pairwise distance vector, normalised by RMS norm.

    Returns
    -------
    1-D array of length N*(N-1)/2.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    vec = squareform(d, checks=False)
    rms = float(np.sqrt(np.mean(vec ** 2)))
    return vec / max(rms, 1e-12)


def get_per_pair_distance_std_across_runs(runs: list[np.ndarray]) -> np.ndarray:
    """Standard deviation of each pair's distance across runs.

    Output values normalised by the across-run mean of the corresponding
    pair so the histogram is dimensionless.
    """
    dist_vectors = np.stack([
        get_normalised_distance_vector(emb) for emb in runs
    ])  # (n_runs, n_pairs)
    return dist_vectors.std(axis=0)


def get_per_coord_std_after_procrustes(runs: list[np.ndarray]) -> np.ndarray:
    """Per-coordinate standard deviation across runs after Procrustes alignment.

    Aligns every run to the first run via orthogonal Procrustes, then
    computes per-(point, axis) standard deviation across runs. Returns
    the magnitude (Euclidean norm of the per-point std vector) for each
    point. Normalised by the overall RMS of the aligned coordinates so
    the histogram is on the same dimensionless scale as the distance
    std.
    """
    reference = runs[0]
    aligned = [reference.copy()]
    for emb in runs[1:]:
        ref_c = reference - reference.mean(axis=0)
        emb_c = emb - emb.mean(axis=0)
        u, _, vt = np.linalg.svd(emb_c.T @ ref_c, full_matrices=False)
        Q = u @ vt
        aligned.append(emb_c @ Q + reference.mean(axis=0))
    stacked = np.stack(aligned, axis=0)  # (n_runs, n, 2)
    std_per_point = stacked.std(axis=0)  # (n, 2)
    magnitude = np.sqrt((std_per_point ** 2).sum(axis=1))  # (n,)
    rms = float(np.sqrt(np.mean(stacked ** 2)))
    return magnitude / max(rms, 1e-12)


def align_for_plot(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Procrustes-align a 2-D point cloud to a reference for visual comparison.

    This is *cosmetic only* — it does not affect any quantitative statistic
    reported in the figure.
    """
    ref_c = reference - reference.mean(axis=0)
    coords_c = coords - coords.mean(axis=0)
    # Scale to match the reference's RMS norm (so PCA-rotated MDS output
    # is comparable in marker size to the canonical layout).
    scale_ratio = np.sqrt((ref_c ** 2).sum()) / max(
        np.sqrt((coords_c ** 2).sum()), 1e-12
    )
    coords_c = coords_c * scale_ratio
    u, _, vt = np.linalg.svd(coords_c.T @ ref_c, full_matrices=False)
    Q = u @ vt
    return coords_c @ Q + reference.mean(axis=0)


def plot_cluster_scatter(
    ax: Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    show_axes: bool = False,
) -> None:
    """Coloured scatter, one colour per cluster."""
    colours = ["#1f4e79", "#806000", "#2e6c2e"]
    for c in range(NUM_CLUSTERS):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=14, c=colours[c], alpha=0.7,
                   label=f"cluster {c + 1}")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888")


def main() -> None:
    np.random.seed(RANDOM_SEED)

    canonical, labels = get_canonical_embedding()
    runs = get_simulated_umap_runs(canonical, labels)

    naive_avg = np.mean(np.stack(runs, axis=0), axis=0)
    procrustes_avg = get_procrustes_consensus(runs)
    reap_mds = get_distance_consensus_2d(runs)

    # PCA-rotate the MDS output so the figure orientation is canonical-like.
    reap_aligned = align_for_plot(reap_mds, canonical)
    procrustes_aligned = align_for_plot(procrustes_avg, canonical)
    naive_aligned = align_for_plot(naive_avg, canonical)

    dist_std = get_per_pair_distance_std_across_runs(runs)
    coord_std = get_per_coord_std_after_procrustes(runs)
    logger.info(
        "Across-run variability (normalised): "
        "pairwise-distance std median=%.4f mean=%.4f | "
        "Procrustes-aligned coord std median=%.4f mean=%.4f",
        float(np.median(dist_std)), float(np.mean(dist_std)),
        float(np.median(coord_std)), float(np.mean(coord_std)),
    )

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.30)

    # Row 1: four simulated UMAP runs
    for k in range(NUM_RUNS):
        ax = fig.add_subplot(gs[0, k])
        plot_cluster_scatter(
            ax, runs[k], labels,
            title=f"Simulated UMAP run {k + 1}",
        )
        if k == 0:
            ax.legend(fontsize=8, loc="best", frameon=False)

    # Row 2: naive, procrustes, REAP, stats panel
    ax_naive = fig.add_subplot(gs[1, 0])
    plot_cluster_scatter(ax_naive, naive_aligned, labels,
                         title="Naive coordinate average")

    ax_proc = fig.add_subplot(gs[1, 1])
    plot_cluster_scatter(ax_proc, procrustes_aligned, labels,
                         title="Procrustes coordinate average")

    ax_reap = fig.add_subplot(gs[1, 2])
    plot_cluster_scatter(ax_reap, reap_aligned, labels,
                         title="REAP distance-matrix consensus")

    ax_stats = fig.add_subplot(gs[1, 3])
    bins = np.linspace(0.0, max(dist_std.max(), coord_std.max()) * 1.05, 40)
    ax_stats.hist(dist_std, bins=bins, alpha=0.6,
                  label="pairwise-distance std",
                  color="#2ca02c")
    ax_stats.hist(coord_std, bins=bins, alpha=0.6,
                  label="coordinate std (post-Procrustes)",
                  color="#1f77b4")
    ax_stats.set_xlabel("Across-run variability (normalised)")
    ax_stats.set_ylabel("Count")
    ax_stats.set_title("Where the seed noise lives")
    ax_stats.legend(fontsize=8, frameon=False, loc="upper right")
    for spine in ("top", "right"):
        ax_stats.spines[spine].set_visible(False)

    suptitle = (
        "Methodological illustration: distance averaging is invariant to "
        "UMAP's per-run rotation/reflection ambiguity; coordinate "
        "averaging is not"
    )
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s and %s", OUT_PNG, OUT_PDF)

    # Side-effect-only: report whether the eigenvalue trick produced a
    # usable 2-D embedding (sanity guard for future maintainers).
    if np.linalg.matrix_rank(reap_mds) < 2:
        raise RuntimeError(
            "Distance-matrix consensus degenerated to rank <2; "
            "check synthetic-run construction."
        )


if __name__ == "__main__":
    main()
