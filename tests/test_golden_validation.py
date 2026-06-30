"""Golden validation tests — Tier 1 (invariants) and Tier 2 (ranges).

These tests enforce the scientific claims of the REAP evaluation protocol
(`manuscript/evaluation_protocol.md` §5-§6). A failure here means either
(a) a code regression has broken a pre-registered claim, or (b) a
platform/library upgrade has shifted UMAP/MiniLM outputs and a range
needs justified widening via a documented protocol correction (§13).

Fixture hierarchy
-----------------
* **Tier 1 (invariants)** runs on `reap.datasets.load_golden_blobs()` —
  a deterministic low-intrinsic-dim-projected blob fixture. Fast, no
  embedding dependency, no text model. Invariants hold regardless of
  fixture content.
* **Tier 2 (ranges + comparative claims)** runs on
  `reap.datasets.load_golden_text()` — a curated 8-class subset of
  20newsgroups embedded with `sentence-transformers/all-MiniLM-L6-v2`.
  Embeddings are cached in `~/.cache/reap/datasets/` after first run,
  so repeated test runs complete in seconds.

Tier 3 (tolerance snapshots against a stored reference run) wires in
once an `environment-lock.yml`-pinned reference run is captured.

Scope notes
-----------
* Seeds: the first 10 seeds of Set A (legacy seeds from the Korean
  forest work). The full 30-seed × 3-set sweep lives in a slower CI
  workflow that reuses the same metric schema.
* UMAP params: nn=15, md=0.1 — conservative defaults. Tier-2 tests
  confirm the fixture exhibits realistic UMAP instability at these
  params (s2s ARI well below 1.0).
* Failure messages carry the metric name, observed value, pre-registered
  range, and a pointer to §13 of the protocol.

Headline comparative claim on the golden fixture
-------------------------------------------------
On this 20newsgroups subset, REAP produces a **more internally coherent
consensus embedding** than Procrustes (silhouette +0.10 to +0.15
consistently). REAP ARI vs ground truth is moderate because the
fixture's two overlap pairs (politics.guns/mideast,
ibm.hardware/mac.hardware) are honestly merged by the consensus. The
paper's ARI-beats-Procrustes claim is tested on real datasets with
different overlap structure (Korean, AI-art) — not on the golden
fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from reap.clustering import run_kmeans
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
    get_procrustes_consensus,
)
from reap.datasets import DatasetSnapshot, load_golden_blobs, load_golden_text
from reap.evaluation import compute_silhouette, compute_trustworthiness

SEED_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "manuscript" / "seeds" / "seed_manifest.json"
)

GOLDEN_K = 8
GOLDEN_N_COMPONENTS = 8
GOLDEN_N_NEIGHBORS = 15
GOLDEN_MIN_DIST = 0.1
GOLDEN_N_SEEDS = 10  # first 10 of Set A — full 30-seed sweep is in slow CI

# Pre-registered Tier-2 ranges on the 20ng golden text fixture.
# Calibrated against the reference run (2026-04-13, MiniLM-L6-v2 v5.4.0).
RANGE_TEXT_SINGLE_SEED_SILHOUETTE: tuple[float, float] = (0.40, 0.70)
RANGE_TEXT_S2S_ARI_MEAN: tuple[float, float] = (0.70, 1.00)
RANGE_TEXT_SEED_VS_TRUTH_ARI: tuple[float, float] = (0.45, 0.75)
RANGE_TEXT_REAP_ARI_VS_TRUTH: tuple[float, float] = (0.40, 0.80)
RANGE_TEXT_REAP_SILHOUETTE: tuple[float, float] = (0.55, 0.85)
RANGE_TEXT_REAP_TRUSTWORTHINESS: tuple[float, float] = (0.85, 0.99)
RANGE_TEXT_PROC_ARI_VS_TRUTH: tuple[float, float] = (0.45, 0.80)

# Headline comparative threshold: REAP silhouette minus Procrustes silhouette.
# Observed +0.149 on the reference run; a positive margin of ≥0.03 is the
# pre-registered floor.
REAP_VS_PROC_SILHOUETTE_MIN_MARGIN: float = 0.03


def _assert_in_range(
    value: float, low_high: tuple[float, float], label: str
) -> None:
    """Fail-loud range check with full diagnostic context."""
    low, high = low_high
    if not (low <= value <= high):
        pytest.fail(
            f"{label}: observed {value:.4f} outside pre-registered range "
            f"[{low:.4f}, {high:.4f}]. Either (a) a code regression broke a "
            "REAP claim, or (b) a platform/library shift needs a protocol "
            "correction. See manuscript/evaluation_protocol.md §13."
        )


def _load_set_a_seeds(n: int) -> list[int]:
    """First `n` seeds from Set A of the canonical seed manifest."""
    manifest = json.loads(SEED_MANIFEST_PATH.read_text())
    seeds = manifest["sets"]["A"]["seeds"][:n]
    assert len(seeds) == n, f"seed manifest Set A has fewer than {n} seeds"
    return [int(s) for s in seeds]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def set_a_seeds() -> list[int]:
    """First GOLDEN_N_SEEDS seeds from Set A — shared across Tier 1/2 tests."""
    return _load_set_a_seeds(GOLDEN_N_SEEDS)


@pytest.fixture(scope="module")
def blob_snap() -> DatasetSnapshot:
    """Tier-1 fixture: the deterministic projected-blobs snapshot."""
    return load_golden_blobs()


@pytest.fixture(scope="module")
def blob_umap_embs(
    blob_snap: DatasetSnapshot, set_a_seeds: list[int]
) -> list[np.ndarray]:
    """Multi-seed UMAP on the blob fixture — computed once for Tier 1."""
    return get_multi_seed_embeddings(
        blob_snap.embeddings,
        set_a_seeds,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )


@pytest.fixture(scope="module")
def blob_reap(blob_umap_embs: list[np.ndarray]) -> dict:
    """REAP consensus on the blob fixture — Tier 1 invariant target."""
    D = get_consensus_distance_matrix(blob_umap_embs)
    embedding, reducer = get_consensus_embedding(
        D,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    return {"D": D, "embedding": embedding, "reducer": reducer}


@pytest.fixture(scope="module")
def text_snap() -> DatasetSnapshot:
    """Tier-2 fixture: the 8-class 20newsgroups text snapshot."""
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence-transformers is required for Tier-2 text fixture. "
        "Install with: pip install 'reap-topics[text-fixtures]'",
    )
    return load_golden_text()


@pytest.fixture(scope="module")
def text_umap_embs(
    text_snap: DatasetSnapshot, set_a_seeds: list[int]
) -> list[np.ndarray]:
    """Multi-seed UMAP on the text fixture — shared across Tier 2 tests."""
    return get_multi_seed_embeddings(
        text_snap.embeddings,
        set_a_seeds,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )


@pytest.fixture(scope="module")
def text_reap(text_umap_embs: list[np.ndarray]) -> dict:
    """REAP consensus outputs for the text fixture."""
    D = get_consensus_distance_matrix(text_umap_embs)
    embedding, _ = get_consensus_embedding(
        D,
        n_components=GOLDEN_N_COMPONENTS,
        n_neighbors=GOLDEN_N_NEIGHBORS,
        min_dist=GOLDEN_MIN_DIST,
    )
    labels, _ = run_kmeans(embedding, k=GOLDEN_K)
    return {"D": D, "embedding": embedding, "labels": labels}


@pytest.fixture(scope="module")
def text_procrustes(text_umap_embs: list[np.ndarray]) -> dict:
    """Procrustes consensus outputs for the text fixture — Tier-2 comparator."""
    embedding = get_procrustes_consensus(text_umap_embs)
    labels, _ = run_kmeans(embedding, k=GOLDEN_K)
    return {"embedding": embedding, "labels": labels}


# ---------------------------------------------------------------------------
# Tier 1 — mathematical invariants (exact; no tolerance)
# ---------------------------------------------------------------------------


class TestTier1MathInvariants:
    """Properties that MUST hold by construction. Exact checks, no tolerance."""

    def test_consensus_matrix_shape(
        self, blob_reap: dict, blob_snap: DatasetSnapshot
    ) -> None:
        D = blob_reap["D"]
        n = blob_snap.embeddings.shape[0]
        assert D.shape == (n, n), (
            f"Consensus D shape {D.shape} != expected ({n}, {n})"
        )

    def test_consensus_matrix_symmetric(self, blob_reap: dict) -> None:
        D = blob_reap["D"]
        asymmetry = float(np.max(np.abs(D - D.T)))
        assert asymmetry < 1e-10, (
            f"Consensus D is not symmetric: max |D - D.T| = {asymmetry:.3e}"
        )

    def test_consensus_matrix_zero_diagonal(self, blob_reap: dict) -> None:
        D = blob_reap["D"]
        diag_max = float(np.max(np.abs(np.diag(D))))
        assert diag_max == 0.0, (
            f"Consensus D diagonal is not zero: max |diag| = {diag_max:.3e}"
        )

    def test_consensus_matrix_nonnegative(self, blob_reap: dict) -> None:
        D = blob_reap["D"]
        min_val = float(D.min())
        assert min_val >= 0.0, (
            f"Consensus D has negative entry: min = {min_val:.3e}"
        )

    def test_consensus_matrix_triangle_inequality(self, blob_reap: dict) -> None:
        """Convex combination of Euclidean distances preserves metric properties.

        Exhaustive O(N³) check is infeasible; draw a reproducible sample
        of 3000 triples and require zero violations beyond 1e-9.
        """
        D = blob_reap["D"]
        n = D.shape[0]
        rng = np.random.default_rng(20260413)
        tol = 1e-9
        n_triples = 3000
        violations: list[tuple[int, int, int, float]] = []
        for _ in range(n_triples):
            idx = rng.integers(0, n, size=3)
            i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
            if len({i, j, k}) != 3:
                continue
            gap = float(D[i, k] - (D[i, j] + D[j, k]))
            if gap > tol:
                violations.append((i, j, k, gap))
        assert not violations, (
            f"{len(violations)}/{n_triples} triangle-inequality violations in "
            f"consensus distance matrix (first 3: {violations[:3]}). "
            "Convex combination of metrics must preserve metric properties."
        )

    def test_per_seed_embeddings_finite_and_shaped(
        self, blob_umap_embs: list[np.ndarray], blob_snap: DatasetSnapshot
    ) -> None:
        n = blob_snap.embeddings.shape[0]
        for i, emb in enumerate(blob_umap_embs):
            assert emb.shape == (n, GOLDEN_N_COMPONENTS), (
                f"Seed {i}: shape {emb.shape} != ({n}, {GOLDEN_N_COMPONENTS})"
            )
            assert np.isfinite(emb).all(), (
                f"Seed {i}: embedding has non-finite values"
            )

    def test_consensus_embedding_finite_and_shaped(
        self, blob_reap: dict, blob_snap: DatasetSnapshot
    ) -> None:
        emb = blob_reap["embedding"]
        n = blob_snap.embeddings.shape[0]
        assert emb.shape == (n, GOLDEN_N_COMPONENTS), (
            f"Consensus embedding shape {emb.shape} != ({n}, {GOLDEN_N_COMPONENTS})"
        )
        assert np.isfinite(emb).all(), (
            "Consensus embedding has NaN/Inf — UMAP failed to converge cleanly"
        )

    def test_kmeans_labels_match_requested_k(self, blob_reap: dict) -> None:
        labels, _ = run_kmeans(blob_reap["embedding"], k=GOLDEN_K)
        observed_k = len(np.unique(labels))
        assert observed_k == GOLDEN_K, (
            f"KMeans returned {observed_k} distinct labels; requested K={GOLDEN_K}"
        )


# ---------------------------------------------------------------------------
# Tier 2 — pre-registered statistical ranges on the 20ng text fixture
# ---------------------------------------------------------------------------


@pytest.mark.reference_platform
class TestTier2TextFixtureRanges:
    """Metric ranges for the 20newsgroups golden text fixture (§6 v1.2).

    Marked ``reference_platform``: these pre-registered absolute metric ranges
    (§13) are calibrated on the Linux reference environment and asserted there
    (including the dedicated golden-validation job). They are skipped on macOS,
    where sentence-transformers/UMAP float behavior moves the metrics below the
    pre-registered floors; the package's code paths are still exercised on macOS
    by the rest of the suite.
    """

    def test_single_seed_silhouette_in_range(
        self, text_umap_embs: list[np.ndarray]
    ) -> None:
        """Each individual UMAP seed produces recognizable clusters on real text."""
        for i, emb in enumerate(text_umap_embs):
            labels, _ = run_kmeans(emb, k=GOLDEN_K)
            sil = compute_silhouette(emb, labels)
            _assert_in_range(
                sil,
                RANGE_TEXT_SINGLE_SEED_SILHOUETTE,
                f"text: single-seed silhouette (seed index {i})",
            )

    def test_s2s_ari_demonstrates_instability(
        self, text_umap_embs: list[np.ndarray]
    ) -> None:
        """Pairwise seed-to-seed ARI quantifies the instability REAP addresses.

        Mean must be below 1.0 — if it's 1.0, there's no instability to fix.
        """
        labels_list = [run_kmeans(e, k=GOLDEN_K)[0] for e in text_umap_embs]
        aris = [
            float(adjusted_rand_score(labels_list[i], labels_list[j]))
            for i in range(len(labels_list))
            for j in range(i + 1, len(labels_list))
        ]
        assert aris, "no ARI pairs computed — need ≥2 seeds"
        mean_ari = float(np.mean(aris))
        _assert_in_range(
            mean_ari, RANGE_TEXT_S2S_ARI_MEAN, "text: mean pairwise s2s ARI"
        )
        assert mean_ari < 1.0, (
            f"text: s2s ARI = {mean_ari:.4f} exactly 1.0 — fixture is too easy "
            "to demonstrate REAP's consensus value. Harden the fixture."
        )

    def test_single_seed_ari_vs_truth_in_range(
        self,
        text_snap: DatasetSnapshot,
        text_umap_embs: list[np.ndarray],
    ) -> None:
        """Individual seeds recover the 8-class ground truth moderately well."""
        truth = text_snap.labels
        assert truth is not None
        per_seed_ari = [
            float(adjusted_rand_score(truth, run_kmeans(e, k=GOLDEN_K)[0]))
            for e in text_umap_embs
        ]
        mean_ari = float(np.mean(per_seed_ari))
        _assert_in_range(
            mean_ari,
            RANGE_TEXT_SEED_VS_TRUTH_ARI,
            "text: mean single-seed ARI vs ground truth",
        )

    def test_reap_consensus_ari_vs_truth_in_range(
        self, text_snap: DatasetSnapshot, text_reap: dict
    ) -> None:
        """REAP consensus ARI vs ground-truth K=8 labels.

        Moderate (not perfect) because the fixture contains two
        genuinely overlapping class pairs that the consensus honestly
        merges. This is a feature of honest consensus, not a failure.
        """
        truth = text_snap.labels
        assert truth is not None
        ari = float(adjusted_rand_score(truth, text_reap["labels"]))
        _assert_in_range(
            ari,
            RANGE_TEXT_REAP_ARI_VS_TRUTH,
            "text: REAP consensus ARI vs ground truth (K=8)",
        )

    def test_reap_consensus_silhouette_in_range(self, text_reap: dict) -> None:
        sil = compute_silhouette(text_reap["embedding"], text_reap["labels"])
        _assert_in_range(
            sil, RANGE_TEXT_REAP_SILHOUETTE, "text: REAP consensus silhouette"
        )

    def test_reap_consensus_trustworthiness_in_range(
        self, text_snap: DatasetSnapshot, text_reap: dict
    ) -> None:
        tw = compute_trustworthiness(
            text_snap.embeddings,
            text_reap["embedding"],
            n_neighbors=GOLDEN_N_NEIGHBORS,
        )
        _assert_in_range(
            tw,
            RANGE_TEXT_REAP_TRUSTWORTHINESS,
            "text: REAP consensus trustworthiness",
        )

    def test_procrustes_ari_vs_truth_in_range(
        self, text_snap: DatasetSnapshot, text_procrustes: dict
    ) -> None:
        truth = text_snap.labels
        assert truth is not None
        ari = float(adjusted_rand_score(truth, text_procrustes["labels"]))
        _assert_in_range(
            ari,
            RANGE_TEXT_PROC_ARI_VS_TRUTH,
            "text: Procrustes consensus ARI vs ground truth",
        )


# ---------------------------------------------------------------------------
# Tier 2b — headline comparative claim on the golden text fixture
# ---------------------------------------------------------------------------


class TestReapBeatsProcrustesOnTextSilhouette:
    """REAP produces a more internally coherent consensus embedding than Procrustes.

    This is the comparative claim that holds on the golden fixture. The
    ARI-vs-ground-truth comparison depends on whether the ground truth
    aligns with the honest consensus geometry (on this fixture, overlap
    pairs are merged by REAP, so ARI-vs-K=8-truth is lower than
    Procrustes); that comparison is tested on the real Korean and
    AI-art fixtures where overlap patterns differ.
    """

    def test_reap_silhouette_exceeds_procrustes_silhouette(
        self, text_reap: dict, text_procrustes: dict
    ) -> None:
        reap_sil = float(
            compute_silhouette(text_reap["embedding"], text_reap["labels"])
        )
        proc_sil = float(
            compute_silhouette(
                text_procrustes["embedding"], text_procrustes["labels"]
            )
        )
        margin = reap_sil - proc_sil
        assert margin >= REAP_VS_PROC_SILHOUETTE_MIN_MARGIN, (
            f"HEADLINE CLAIM FAILED on golden text fixture: "
            f"REAP silhouette = {reap_sil:.4f} does not exceed Procrustes "
            f"silhouette = {proc_sil:.4f} by the required margin "
            f"{REAP_VS_PROC_SILHOUETTE_MIN_MARGIN:+.2f}. "
            f"Observed margin: {margin:+.4f}. "
            "This is REAP's core empirical claim on the golden fixture — "
            "investigate before widening the margin threshold."
        )


# ---------------------------------------------------------------------------
# Dataset API contract
# ---------------------------------------------------------------------------


class TestDatasetSnapshotContract:
    """DatasetSnapshot behaves as specified in open-source-package.md."""

    def test_blob_snapshot_shape_and_types(self) -> None:
        snap = load_golden_blobs()
        assert snap.embeddings.shape == (400, 384), (
            f"Blob golden shape {snap.embeddings.shape} != (400, 384)"
        )
        assert snap.embeddings.dtype == np.float64
        assert snap.labels is not None
        assert snap.labels.shape == (400,)
        assert snap.metadata.n_samples == 400
        assert snap.metadata.embedding_dim == 384
        assert len(snap.metadata.sha256) == 64

    def test_blob_snapshot_finite_and_normalized(self) -> None:
        snap = load_golden_blobs()
        assert np.isfinite(snap.embeddings).all()
        norms = np.linalg.norm(snap.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), (
            "Blob fixture must be L2-normalized"
        )

    def test_blob_snapshot_sha256_is_stable(self) -> None:
        """Two independent loads must produce byte-identical SHA256."""
        a = load_golden_blobs()
        b = load_golden_blobs()
        assert a.metadata.sha256 == b.metadata.sha256, (
            f"Blob golden SHA256 drifted between loads: "
            f"{a.metadata.sha256} vs {b.metadata.sha256}"
        )

    def test_blob_snapshot_has_eight_classes(self) -> None:
        snap = load_golden_blobs()
        assert snap.labels is not None
        k = len(np.unique(snap.labels))
        assert k == GOLDEN_K, f"Blob fixture has {k} classes; expected {GOLDEN_K}"

    def test_text_snapshot_shape_and_classes(
        self, text_snap: DatasetSnapshot
    ) -> None:
        assert text_snap.embeddings.shape == (400, 384), (
            f"Text golden shape {text_snap.embeddings.shape} != (400, 384)"
        )
        assert text_snap.texts is not None
        assert len(text_snap.texts) == 400
        assert text_snap.labels is not None
        assert len(np.unique(text_snap.labels)) == GOLDEN_K

    def test_text_snapshot_sha256_stable(self) -> None:
        """Embedding cache must produce byte-identical output across loads."""
        pytest.importorskip("sentence_transformers")
        a = load_golden_text()
        b = load_golden_text()
        assert a.metadata.sha256 == b.metadata.sha256, (
            f"Text golden SHA256 drifted: {a.metadata.sha256[:16]}... vs "
            f"{b.metadata.sha256[:16]}..."
        )

    def test_text_snapshot_normalized(self, text_snap: DatasetSnapshot) -> None:
        """MiniLM with normalize_embeddings=True must produce unit-norm vectors."""
        norms = np.linalg.norm(text_snap.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), (
            f"Text embeddings not unit-norm: min={norms.min():.4f} max={norms.max():.4f}"
        )

    def test_named_dataset_stubs_raise_not_implemented(self) -> None:
        """Paper-dataset loaders without snapshots raise clear migration errors.

        ``ai_art`` and ``korean_forest`` now load from the REAP cache
        (``src/reap/datasets/ai_art.py``, ``korean_forest.py``); if the
        cache is populated they succeed, if it is empty they raise
        ``FileNotFoundError`` pointing at ``scripts/build_datasets.py``.
        ``corp_sustainability`` and ``us_presidential`` remain stubs
        until their source data lands.
        """
        import reap.datasets as ds

        for fn in (ds.load_corp_sustainability, ds.load_us_presidential):
            with pytest.raises(NotImplementedError, match="stub"):
                fn()

        for fn in (ds.load_ai_art, ds.load_korean_forest):
            try:
                snap = fn()
            except FileNotFoundError as exc:
                assert "build_datasets.py" in str(exc), (
                    f"cache-miss error should point at the builder: {exc}"
                )
            else:
                assert snap.metadata.n_samples >= 1
