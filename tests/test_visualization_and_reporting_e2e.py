"""End-to-end tests for `reap.visualization` and `reap.reporting`.

These tests complement the existing unit-style tests in
`test_visualization.py` and `test_reporting.py` (which use small random
arrays and hand-crafted Pydantic fixtures) by running each module
against *real* REAP pipeline outputs — the 20newsgroups golden
consensus for visualization, and a real `run_benchmark` sweep for
reporting.

Philosophy: unit tests verify the module's contract in isolation; E2E
tests verify that the module composes correctly with the rest of the
pipeline. Both are needed.

No mocking of UMAP, sklearn, torch, numpy, or Pydantic validators.
"""

from __future__ import annotations

import csv as csvmod
import json
from pathlib import Path

import numpy as np
import pytest

from reap.benchmarks import BenchmarkResult, run_benchmark
from reap.consensus import (
    get_consensus_distance_matrix,
    get_consensus_embedding,
    get_multi_seed_embeddings,
)
from reap.datasets import DatasetSnapshot, load_golden_text, load_synthetic_blobs
from reap.reporting import (
    benchmark_to_csv,
    benchmark_to_latex,
    benchmark_to_markdown,
    cross_dataset_latex,
    cross_dataset_table,
)
from reap.visualization import (
    fit_pca_visualization,
    load_visualization_reducer,
    save_visualization_reducer,
    transform_to_2d,
)

SEED_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "manuscript"
    / "seeds"
    / "seed_manifest.json"
)


def _load_set_a_seeds(n: int) -> list[int]:
    return [
        int(s)
        for s in json.loads(SEED_MANIFEST_PATH.read_text())["sets"]["A"]["seeds"][:n]
    ]


# ---------------------------------------------------------------------------
# Visualization E2E on real 20ng REAP consensus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def text_snap_for_viz() -> DatasetSnapshot:
    """Load the 20ng golden text fixture (sentence-transformers required)."""
    pytest.importorskip(
        "sentence_transformers",
        reason="text fixture needs sentence-transformers",
    )
    return load_golden_text()


@pytest.fixture(scope="module")
def real_consensus_embedding(text_snap_for_viz: DatasetSnapshot) -> np.ndarray:
    """REAP consensus on the 20ng golden fixture — the real input viz consumes."""
    seeds = _load_set_a_seeds(5)
    embs = get_multi_seed_embeddings(
        text_snap_for_viz.embeddings,
        seeds,
        n_components=8,
        n_neighbors=15,
        min_dist=0.1,
    )
    D = get_consensus_distance_matrix(embs)
    Y, _ = get_consensus_embedding(D, n_components=8, n_neighbors=15, min_dist=0.1)
    return Y


class TestVisualizationE2EOnRealConsensus:
    """Visualization module consumes real REAP consensus output without
    mocking any upstream step."""

    def test_pca_fits_on_real_consensus(
        self, real_consensus_embedding: np.ndarray
    ) -> None:
        pca, metadata = fit_pca_visualization(real_consensus_embedding)
        assert pca.n_components == 2
        assert "pc1_variance_ratio" in metadata
        assert "pc2_variance_ratio" in metadata
        total = metadata["total_variance_ratio"]
        assert 0.0 < total <= 1.0, (
            f"Total variance ratio {total} out of (0, 1]; PCA did not fit "
            "properly on real consensus data"
        )

    def test_pca_captures_meaningful_variance_on_consensus(
        self, real_consensus_embedding: np.ndarray
    ) -> None:
        """REAP consensus has strong linear structure per the visualization
        module docstring — first 2 PCs should explain ≥30% of variance on
        real 20ng consensus output.

        This is a calibrated floor: observed runs return 40–70%.
        """
        _, metadata = fit_pca_visualization(real_consensus_embedding)
        total = metadata["total_variance_ratio"]
        assert total >= 0.30, (
            f"PC1+PC2 variance {total:.4f} below 0.30 floor on real REAP "
            "consensus output; either consensus lost linear structure or "
            "PCA is misapplied."
        )

    def test_transform_produces_expected_2d_shape(
        self, real_consensus_embedding: np.ndarray
    ) -> None:
        pca, _ = fit_pca_visualization(real_consensus_embedding)
        Y2d = transform_to_2d(pca, real_consensus_embedding)
        n = real_consensus_embedding.shape[0]
        assert Y2d.shape == (n, 2)
        assert np.isfinite(Y2d).all(), (
            "2D visualization has NaN/Inf — PCA fit produced degenerate output"
        )

    def test_additive_projection_on_real_data(
        self, real_consensus_embedding: np.ndarray
    ) -> None:
        """Reference points must not move when new data is added.

        Additivity is the key property justifying PCA-over-UMAP for
        visualization.
        """
        pca, _ = fit_pca_visualization(real_consensus_embedding)
        half = real_consensus_embedding.shape[0] // 2
        y_half = transform_to_2d(pca, real_consensus_embedding[:half])
        y_full = transform_to_2d(pca, real_consensus_embedding)
        np.testing.assert_array_almost_equal(y_half, y_full[:half], decimal=10)

    def test_save_load_roundtrip_on_real_data(
        self,
        real_consensus_embedding: np.ndarray,
        tmp_path: Path,
    ) -> None:
        pca, _ = fit_pca_visualization(real_consensus_embedding)
        before = transform_to_2d(pca, real_consensus_embedding)
        pkl_path = tmp_path / "viz_reducer.pkl"
        save_visualization_reducer(pca, pkl_path)
        assert pkl_path.exists(), "save_visualization_reducer did not write a file"
        loaded = load_visualization_reducer(pkl_path)
        after = transform_to_2d(loaded, real_consensus_embedding)
        np.testing.assert_array_almost_equal(before, after, decimal=10)

    def test_transform_is_deterministic_on_real_data(
        self, real_consensus_embedding: np.ndarray
    ) -> None:
        pca, _ = fit_pca_visualization(real_consensus_embedding)
        r1 = transform_to_2d(pca, real_consensus_embedding)
        r2 = transform_to_2d(pca, real_consensus_embedding)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Reporting E2E — feed a real `run_benchmark` result through each formatter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_benchmark_result() -> BenchmarkResult:
    """Run run_benchmark on a small synthetic-blob fixture for the reporting E2E.

    We use blobs rather than 20ng here so the benchmark completes in ~20s
    even with 4 methods × 3 seeds. The reporting module is agnostic to
    dataset size; what matters is that the BenchmarkResult it formats comes
    from a real pipeline run, not a hand-crafted fixture.
    """
    snap = load_synthetic_blobs(
        n_samples=200,
        n_features=64,
        centers=4,
        cluster_std=2.0,
        random_state=42,
        l2_normalize=True,
    )
    return run_benchmark(
        X=snap.embeddings,
        dataset_name="reporting_e2e_blobs",
        seeds=[1, 2, 3],
        n_components=4,
        n_neighbors=10,
        min_dist=0.1,
        k_range=[3, 4, 5, 6],
        metric="euclidean",
        silhouette_metric="euclidean",
        methods=["single_seed", "naive_average", "procrustes", "reap"],
    )


class TestReportingE2EOnRealBenchmark:
    """Feed real BenchmarkResult through every formatter and validate output."""

    def test_benchmark_has_expected_methods(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        names = [m.method for m in real_benchmark_result.methods]
        assert {"single_seed", "naive_average", "procrustes", "reap"} <= set(names), (
            f"run_benchmark missing expected methods; got {names}"
        )

    def test_csv_roundtrip_preserves_real_values(
        self,
        real_benchmark_result: BenchmarkResult,
        tmp_path: Path,
    ) -> None:
        """CSV written from a real BenchmarkResult must round-trip to the
        same values within the formatter's stated precision.

        `benchmark_to_csv` writes float values at 3-decimal precision, so
        the tolerance here is 5e-4 (half the last-digit step). This
        catches silent precision bugs while allowing the intentional
        rounding for human-readable CSV output.
        """
        csv_path = tmp_path / "real_benchmark.csv"
        benchmark_to_csv(real_benchmark_result, csv_path)
        assert csv_path.exists()
        with csv_path.open() as f:
            rows = list(csvmod.DictReader(f))
        assert len(rows) == len(real_benchmark_result.methods)
        method_to_row = {r["method"]: r for r in rows}
        tol = 5e-4
        for m in real_benchmark_result.methods:
            r = method_to_row[m.method]
            assert abs(float(r["trustworthiness"]) - m.trustworthiness) < tol, (
                f"CSV trustworthiness for {m.method} did not round-trip "
                f"within {tol}: {r['trustworthiness']} vs {m.trustworthiness}"
            )
            assert abs(float(r["silhouette"]) - m.silhouette) < tol, (
                f"CSV silhouette for {m.method} did not round-trip within {tol}"
            )
            assert int(r["best_k"]) == m.best_k, (
                f"CSV best_k for {m.method} did not round-trip"
            )

    def test_markdown_contains_all_real_methods_and_best_values(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        md = benchmark_to_markdown(real_benchmark_result, bold_best=True)
        assert real_benchmark_result.dataset_name in md
        for m in real_benchmark_result.methods:
            assert m.method in md, (
                f"Markdown output missing method {m.method!r}"
            )
        # The bolded cell may include `± std` suffix when std > 0, so the
        # prefix assertion `**{value}` is the correct invariant rather
        # than a full exact match.
        best_tw = max(m.trustworthiness for m in real_benchmark_result.methods)
        best_tw_rounded = f"{best_tw:.3f}"
        assert f"**{best_tw_rounded}" in md, (
            f"Markdown output did not bold the best trustworthiness "
            f"{best_tw_rounded}; got:\n{md}"
        )

    def test_latex_contains_all_real_methods(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        tex = benchmark_to_latex(real_benchmark_result, bold_best=True)
        assert "\\toprule" in tex and "\\bottomrule" in tex
        for m in real_benchmark_result.methods:
            method_escaped = m.method.replace("_", "\\_")
            assert method_escaped in tex, (
                f"LaTeX output missing escaped method {method_escaped!r}"
            )

    def test_cross_dataset_table_on_real_result(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        md = cross_dataset_table([real_benchmark_result])
        assert real_benchmark_result.dataset_name in md
        pipe_lines = [ln for ln in md.strip().split("\n") if ln.startswith("|")]
        # header + separator + 1 data row = 3
        assert len(pipe_lines) == 3, (
            f"cross_dataset_table row count wrong on real result: {len(pipe_lines)}"
        )

    def test_cross_dataset_latex_on_real_result(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        tex = cross_dataset_latex([real_benchmark_result])
        assert "\\toprule" in tex and "\\bottomrule" in tex
        # Dataset names with underscores are LaTeX-escaped; check both the
        # raw form (for names without special chars) and the escaped form.
        escaped = real_benchmark_result.dataset_name.replace("_", "\\_")
        assert escaped in tex or real_benchmark_result.dataset_name in tex, (
            f"Neither {escaped!r} nor {real_benchmark_result.dataset_name!r} "
            f"found in LaTeX output"
        )


class TestReportingReapMetricsAreLoadBearing:
    """The real REAP-row values in the report must match the source object.

    Reporting is the final step before numbers land in the paper. A
    formatting bug that silently shows the wrong number is the kind of
    provenance failure the publication-standards.md rule exists to
    prevent — this test enforces that directly.
    """

    def test_reap_trustworthiness_appears_in_markdown(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        reap = next(
            m for m in real_benchmark_result.methods if m.method == "reap"
        )
        md = benchmark_to_markdown(real_benchmark_result)
        tw_str = f"{reap.trustworthiness:.3f}"
        assert tw_str in md, (
            f"REAP trustworthiness {tw_str} not present in markdown output"
        )

    def test_reap_silhouette_appears_in_latex(
        self, real_benchmark_result: BenchmarkResult
    ) -> None:
        reap = next(
            m for m in real_benchmark_result.methods if m.method == "reap"
        )
        tex = benchmark_to_latex(real_benchmark_result)
        sil_str = f"{reap.silhouette:.3f}"
        assert sil_str in tex, (
            f"REAP silhouette {sil_str} not present in LaTeX output"
        )

    def test_reap_best_k_appears_in_csv(
        self,
        real_benchmark_result: BenchmarkResult,
        tmp_path: Path,
    ) -> None:
        reap = next(
            m for m in real_benchmark_result.methods if m.method == "reap"
        )
        csv_path = tmp_path / "reap_bestk.csv"
        benchmark_to_csv(real_benchmark_result, csv_path)
        with csv_path.open() as f:
            rows = {r["method"]: r for r in csvmod.DictReader(f)}
        assert int(rows["reap"]["best_k"]) == reap.best_k
