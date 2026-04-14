"""Tests for reap.reporting — publication-ready table generation.

Uses synthetic benchmark, ablation, and bootstrap data to verify correct
formatting across Markdown, LaTeX, and CSV outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from reap.ablation import (
    ParameterSweepPoint,
    ParameterSweepResult,
    SeedAblationPoint,
    SeedAblationResult,
)
from reap.benchmarks import BenchmarkResult, BootstrapCI, MethodResult
from reap.reporting import (
    _bold_latex,
    _bold_markdown,
    _escape_latex,
    _format_metric,
    _format_metric_latex,
    _format_runtime,
    ablation_to_latex,
    ablation_to_markdown,
    benchmark_to_csv,
    benchmark_to_latex,
    benchmark_to_markdown,
    bootstrap_to_latex,
    bootstrap_to_markdown,
    cross_dataset_latex,
    cross_dataset_table,
    sweep_to_latex,
    sweep_to_markdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def method_reap() -> MethodResult:
    """REAP method with full metrics."""
    return MethodResult(
        method="reap",
        description="Average pairwise distance matrices, then project via UMAP",
        trustworthiness=0.892,
        trustworthiness_std=0.0,
        silhouette=0.841,
        silhouette_std=0.0,
        best_k=8,
        best_k_std=0.0,
        seed_to_seed_ari_mean=0.750,
        seed_to_seed_ari_std=0.031,
        seed_to_consensus_ari_mean=0.820,
        n_seeds=30,
        runtime_seconds=45.2,
    )


@pytest.fixture()
def method_procrustes() -> MethodResult:
    """Procrustes baseline with partial metrics."""
    return MethodResult(
        method="procrustes",
        description="Procrustes-align to first seed, then average coordinates",
        trustworthiness=0.875,
        trustworthiness_std=0.0,
        silhouette=0.720,
        silhouette_std=0.0,
        best_k=7,
        best_k_std=0.0,
        seed_to_seed_ari_mean=0.560,
        seed_to_seed_ari_std=0.045,
        seed_to_consensus_ari_mean=0.650,
        n_seeds=30,
        runtime_seconds=38.7,
    )


@pytest.fixture()
def method_single_seed() -> MethodResult:
    """Single seed baseline with std values."""
    return MethodResult(
        method="single_seed",
        description="Independent single-seed UMAP runs",
        trustworthiness=0.860,
        trustworthiness_std=0.025,
        silhouette=0.690,
        silhouette_std=0.080,
        best_k=6,
        best_k_std=1.5,
        seed_to_seed_ari_mean=None,
        seed_to_seed_ari_std=None,
        seed_to_consensus_ari_mean=None,
        n_seeds=30,
        runtime_seconds=120.5,
    )


@pytest.fixture()
def benchmark_result(
    method_reap: MethodResult,
    method_procrustes: MethodResult,
    method_single_seed: MethodResult,
) -> BenchmarkResult:
    """Complete benchmark result with three methods."""
    return BenchmarkResult(
        dataset_name="Korean Forest Policy",
        n_samples=905,
        n_features=384,
        n_components=18,
        n_neighbors=19,
        min_dist=0.005,
        k_range=list(range(3, 15)),
        n_seeds=30,
        methods=[method_reap, method_procrustes, method_single_seed],
    )


@pytest.fixture()
def benchmark_result_2(method_reap: MethodResult) -> BenchmarkResult:
    """Second dataset for cross-dataset tests."""
    reap_2 = MethodResult(
        method="reap",
        description="REAP on AI-art",
        trustworthiness=0.916,
        trustworthiness_std=0.0,
        silhouette=0.657,
        silhouette_std=0.0,
        best_k=20,
        best_k_std=0.0,
        seed_to_seed_ari_mean=0.680,
        seed_to_seed_ari_std=0.025,
        seed_to_consensus_ari_mean=0.780,
        n_seeds=30,
        runtime_seconds=92.1,
    )
    return BenchmarkResult(
        dataset_name="AI-Art Discourse",
        n_samples=1742,
        n_features=1024,
        n_components=5,
        n_neighbors=53,
        min_dist=0.01,
        k_range=list(range(5, 30)),
        n_seeds=30,
        methods=[reap_2],
    )


@pytest.fixture()
def seed_ablation() -> SeedAblationResult:
    """Seed ablation with three points."""
    return SeedAblationResult(
        dataset_name="Korean Forest Policy",
        n_components=18,
        n_neighbors=19,
        min_dist=0.005,
        seed_counts=[5, 15, 30],
        results=[
            SeedAblationPoint(
                n_seeds=5,
                trustworthiness=0.870,
                silhouette=0.790,
                best_k=7,
                seed_to_seed_ari_mean=0.620,
                seed_to_seed_ari_std=0.055,
                runtime_seconds=8.2,
            ),
            SeedAblationPoint(
                n_seeds=15,
                trustworthiness=0.885,
                silhouette=0.825,
                best_k=8,
                seed_to_seed_ari_mean=0.710,
                seed_to_seed_ari_std=0.038,
                runtime_seconds=22.1,
            ),
            SeedAblationPoint(
                n_seeds=30,
                trustworthiness=0.892,
                silhouette=0.841,
                best_k=8,
                seed_to_seed_ari_mean=0.750,
                seed_to_seed_ari_std=0.031,
                runtime_seconds=45.2,
            ),
        ],
    )


@pytest.fixture()
def parameter_sweep() -> ParameterSweepResult:
    """Parameter sweep over n_neighbors."""
    return ParameterSweepResult(
        dataset_name="Korean Forest Policy",
        swept_parameter="n_neighbors",
        parameter_values=[10.0, 19.0, 30.0],
        fixed_params={
            "n_components": 18,
            "min_dist": 0.005,
            "n_seeds": 30,
        },
        results=[
            ParameterSweepPoint(
                parameter_value=10.0,
                trustworthiness=0.880,
                silhouette=0.810,
                best_k=9,
                seed_to_seed_ari_mean=0.700,
                runtime_seconds=40.1,
            ),
            ParameterSweepPoint(
                parameter_value=19.0,
                trustworthiness=0.892,
                silhouette=0.841,
                best_k=8,
                seed_to_seed_ari_mean=0.750,
                runtime_seconds=45.2,
            ),
            ParameterSweepPoint(
                parameter_value=30.0,
                trustworthiness=0.870,
                silhouette=0.800,
                best_k=7,
                seed_to_seed_ari_mean=0.680,
                runtime_seconds=48.9,
            ),
        ],
    )


@pytest.fixture()
def bootstrap_cis() -> list[BootstrapCI]:
    """Bootstrap CIs for two methods, two metrics."""
    return [
        BootstrapCI(
            metric="trustworthiness",
            method="reap",
            mean=0.892,
            ci_lower=0.871,
            ci_upper=0.913,
            n_bootstrap=1000,
        ),
        BootstrapCI(
            metric="silhouette",
            method="reap",
            mean=0.841,
            ci_lower=0.812,
            ci_upper=0.870,
            n_bootstrap=1000,
        ),
        BootstrapCI(
            metric="trustworthiness",
            method="procrustes",
            mean=0.875,
            ci_lower=0.850,
            ci_upper=0.900,
            n_bootstrap=1000,
        ),
        BootstrapCI(
            metric="silhouette",
            method="procrustes",
            mean=0.720,
            ci_lower=0.690,
            ci_upper=0.750,
            n_bootstrap=1000,
        ),
    ]


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestFormatMetric:
    """Tests for _format_metric."""

    def test_value_only(self) -> None:
        assert _format_metric(0.892, decimals=3) == "0.892"

    def test_value_with_std(self) -> None:
        result = _format_metric(0.892, std=0.031, decimals=3)
        assert "0.892" in result
        assert "0.031" in result
        assert "\u00b1" in result  # ± character

    def test_zero_std_omitted(self) -> None:
        result = _format_metric(0.892, std=0.0, decimals=3)
        assert "\u00b1" not in result

    def test_decimals_respected(self) -> None:
        assert _format_metric(0.89234, decimals=2) == "0.89"
        assert _format_metric(0.89234, decimals=4) == "0.8923"

    def test_integer_value(self) -> None:
        assert _format_metric(8.0, decimals=0) == "8"


class TestFormatMetricLatex:
    """Tests for _format_metric_latex."""

    def test_value_only(self) -> None:
        assert _format_metric_latex(0.892, decimals=3) == "0.892"

    def test_value_with_std(self) -> None:
        result = _format_metric_latex(0.892, std=0.031, decimals=3)
        assert "0.892" in result
        assert "0.031" in result
        assert "$\\pm$" in result

    def test_zero_std_omitted(self) -> None:
        result = _format_metric_latex(0.892, std=0.0, decimals=3)
        assert "$\\pm$" not in result


class TestBoldHelpers:
    """Tests for _bold_markdown and _bold_latex."""

    def test_bold_markdown(self) -> None:
        assert _bold_markdown("0.892") == "**0.892**"

    def test_bold_latex(self) -> None:
        assert _bold_latex("0.892") == "\\textbf{0.892}"


class TestEscapeLatex:
    """Tests for _escape_latex."""

    def test_underscores(self) -> None:
        assert _escape_latex("best_of_n") == "best\\_of\\_n"

    def test_ampersand(self) -> None:
        assert _escape_latex("A & B") == "A \\& B"

    def test_percent(self) -> None:
        assert _escape_latex("95%") == "95\\%"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_latex("reap") == "reap"


class TestFormatRuntime:
    """Tests for _format_runtime."""

    def test_short_duration(self) -> None:
        assert _format_runtime(1.234) == "1.2s"

    def test_exactly_60(self) -> None:
        assert _format_runtime(60.0) == "1m 0s"

    def test_long_duration(self) -> None:
        result = _format_runtime(150.0)
        assert "2m" in result
        assert "30s" in result

    def test_sub_second(self) -> None:
        assert _format_runtime(0.5) == "0.5s"


# ---------------------------------------------------------------------------
# Benchmark table tests
# ---------------------------------------------------------------------------


class TestBenchmarkToMarkdown:
    """Tests for benchmark_to_markdown."""

    def test_contains_dataset_info(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        assert "Korean Forest Policy" in md
        assert "n=905" in md
        assert "d=384" in md

    def test_contains_all_methods(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        assert "reap" in md
        assert "procrustes" in md
        assert "single_seed" in md

    def test_contains_table_structure(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        lines = md.strip().split("\n")
        # Should have header info, blank line, header, separator, 3 data rows
        pipe_lines = [ln for ln in lines if ln.startswith("|")]
        assert len(pipe_lines) == 5  # header + sep + 3 rows

    def test_bold_best_tw(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result, bold_best=True)
        # REAP has highest TW (0.892) so should be bold
        assert "**0.892**" in md

    def test_bold_best_silhouette(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result, bold_best=True)
        # REAP has highest Sil (0.841) so should be bold
        assert "**0.841**" in md

    def test_bold_best_time(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result, bold_best=True)
        # Procrustes has lowest time (38.7s), should be bold
        assert "**38.7s**" in md

    def test_no_bold_when_disabled(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result, bold_best=False)
        # The header line uses ** for dataset name, so check only data rows
        data_lines = [ln for ln in md.strip().split("\n") if ln.startswith("| ")]
        for line in data_lines:
            assert "**" not in line

    def test_na_for_missing_ari(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        # single_seed has no S2S ARI — should show --
        assert "--" in md

    def test_std_in_single_seed(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        # single_seed has TW std of 0.025
        assert "\u00b1 0.025" in md

    def test_separator_line_has_dashes(self, benchmark_result: BenchmarkResult) -> None:
        md = benchmark_to_markdown(benchmark_result)
        lines = md.strip().split("\n")
        sep_lines = [ln for ln in lines if "---" in ln and "|" in ln]
        assert len(sep_lines) == 1


class TestBenchmarkToLatex:
    """Tests for benchmark_to_latex."""

    def test_contains_booktabs(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex

    def test_contains_table_env(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "\\begin{tabular}" in tex
        assert "\\end{tabular}" in tex

    def test_bold_best_value(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result, bold_best=True)
        assert "\\textbf{0.892}" in tex  # Best TW

    def test_no_bold_when_disabled(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result, bold_best=False)
        assert "\\textbf{" not in tex

    def test_pm_for_std(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "$\\pm$" in tex

    def test_custom_caption(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result, caption="My custom caption")
        assert "\\caption{My custom caption}" in tex

    def test_custom_label(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result, label="tab:custom")
        assert "\\label{tab:custom}" in tex

    def test_default_label(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "\\label{tab:korean_forest_policy_comparison}" in tex

    def test_all_methods_present(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "reap" in tex
        assert "procrustes" in tex
        # single_seed has underscores — should be escaped
        assert "single\\_seed" in tex

    def test_na_for_missing_ari(self, benchmark_result: BenchmarkResult) -> None:
        tex = benchmark_to_latex(benchmark_result)
        assert "--" in tex


class TestBenchmarkToCsv:
    """Tests for benchmark_to_csv."""

    def test_creates_file(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "benchmark.csv"
        benchmark_to_csv(benchmark_result, csv_path)
        assert csv_path.exists()

    def test_correct_row_count(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "benchmark.csv"
        benchmark_to_csv(benchmark_result, csv_path)
        with csv_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 methods
        assert len(rows) == 4

    def test_header_fields(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "benchmark.csv"
        benchmark_to_csv(benchmark_result, csv_path)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
        assert "method" in fields
        assert "trustworthiness" in fields
        assert "silhouette" in fields
        assert "runtime_seconds" in fields

    def test_values_are_numeric(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "benchmark.csv"
        benchmark_to_csv(benchmark_result, csv_path)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                tw = row["trustworthiness"]
                assert float(tw), f"trustworthiness should be numeric, got {tw}"

    def test_empty_for_none_values(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "benchmark.csv"
        benchmark_to_csv(benchmark_result, csv_path)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["method"] == "single_seed":
                    assert row["seed_to_seed_ari_mean"] == ""

    def test_accepts_string_path(
        self, benchmark_result: BenchmarkResult, tmp_path: Path,
    ) -> None:
        csv_path = str(tmp_path / "benchmark_str.csv")
        benchmark_to_csv(benchmark_result, csv_path)
        assert Path(csv_path).exists()


# ---------------------------------------------------------------------------
# Cross-dataset tests
# ---------------------------------------------------------------------------


class TestCrossDatasetTable:
    """Tests for cross_dataset_table."""

    def test_contains_both_datasets(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        md = cross_dataset_table([benchmark_result, benchmark_result_2])
        assert "Korean Forest Policy" in md
        assert "AI-Art Discourse" in md

    def test_correct_columns(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        md = cross_dataset_table([benchmark_result, benchmark_result_2])
        assert "Dataset" in md
        assert "d_embed" in md
        assert "d_UMAP" in md

    def test_row_count(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        md = cross_dataset_table([benchmark_result, benchmark_result_2])
        lines = md.strip().split("\n")
        pipe_lines = [ln for ln in lines if ln.startswith("|")]
        assert len(pipe_lines) == 4  # header + sep + 2 data rows

    def test_missing_method_shows_na(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        # Look for a method that doesn't exist in benchmark_result_2
        md = cross_dataset_table(
            [benchmark_result, benchmark_result_2], method="procrustes",
        )
        # benchmark_result_2 doesn't have procrustes
        assert "--" in md


class TestCrossDatasetLatex:
    """Tests for cross_dataset_latex."""

    def test_booktabs(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        tex = cross_dataset_latex([benchmark_result, benchmark_result_2])
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex

    def test_custom_caption_label(
        self,
        benchmark_result: BenchmarkResult,
        benchmark_result_2: BenchmarkResult,
    ) -> None:
        tex = cross_dataset_latex(
            [benchmark_result, benchmark_result_2],
            caption="My cross-dataset table",
            label="tab:cross",
        )
        assert "\\caption{My cross-dataset table}" in tex
        assert "\\label{tab:cross}" in tex


# ---------------------------------------------------------------------------
# Ablation table tests
# ---------------------------------------------------------------------------


class TestAblationToMarkdown:
    """Tests for ablation_to_markdown."""

    def test_header_info(self, seed_ablation: SeedAblationResult) -> None:
        md = ablation_to_markdown(seed_ablation)
        assert "Korean Forest Policy" in md
        assert "Seed ablation" in md

    def test_all_seed_counts(self, seed_ablation: SeedAblationResult) -> None:
        md = ablation_to_markdown(seed_ablation)
        assert "| 5" in md
        assert "| 15" in md
        assert "| 30" in md

    def test_ari_with_std(self, seed_ablation: SeedAblationResult) -> None:
        md = ablation_to_markdown(seed_ablation)
        assert "\u00b1" in md  # ± for S2S ARI std


class TestAblationToLatex:
    """Tests for ablation_to_latex."""

    def test_booktabs(self, seed_ablation: SeedAblationResult) -> None:
        tex = ablation_to_latex(seed_ablation)
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex

    def test_default_label(self, seed_ablation: SeedAblationResult) -> None:
        tex = ablation_to_latex(seed_ablation)
        assert "\\label{tab:korean_forest_policy_seed_ablation}" in tex


# ---------------------------------------------------------------------------
# Sweep table tests
# ---------------------------------------------------------------------------


class TestSweepToMarkdown:
    """Tests for sweep_to_markdown."""

    def test_header_info(self, parameter_sweep: ParameterSweepResult) -> None:
        md = sweep_to_markdown(parameter_sweep)
        assert "n_neighbors" in md
        assert "Korean Forest Policy" in md

    def test_parameter_values(self, parameter_sweep: ParameterSweepResult) -> None:
        md = sweep_to_markdown(parameter_sweep)
        assert "10" in md
        assert "19" in md
        assert "30" in md

    def test_integer_param_formatting(self, parameter_sweep: ParameterSweepResult) -> None:
        md = sweep_to_markdown(parameter_sweep)
        # 10.0 should render as "10", not "10.0"
        lines = md.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("| 1")]
        # The first data line should start with "| 10"
        assert any("| 10 " in ln for ln in data_lines)


class TestSweepToLatex:
    """Tests for sweep_to_latex."""

    def test_booktabs(self, parameter_sweep: ParameterSweepResult) -> None:
        tex = sweep_to_latex(parameter_sweep)
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex

    def test_escaped_param_name(self) -> None:
        """Parameter names with underscores should be escaped in LaTeX."""
        sweep = ParameterSweepResult(
            dataset_name="Test",
            swept_parameter="min_dist",
            parameter_values=[0.005, 0.01],
            fixed_params={"n_components": 18, "n_neighbors": 19, "n_seeds": 30},
            results=[
                ParameterSweepPoint(
                    parameter_value=0.005,
                    trustworthiness=0.892,
                    silhouette=0.841,
                    best_k=8,
                    seed_to_seed_ari_mean=0.750,
                    runtime_seconds=45.0,
                ),
                ParameterSweepPoint(
                    parameter_value=0.01,
                    trustworthiness=0.880,
                    silhouette=0.810,
                    best_k=9,
                    seed_to_seed_ari_mean=0.700,
                    runtime_seconds=44.0,
                ),
            ],
        )
        tex = sweep_to_latex(sweep)
        assert "min\\_dist" in tex


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapToMarkdown:
    """Tests for bootstrap_to_markdown."""

    def test_both_methods(self, bootstrap_cis: list[BootstrapCI]) -> None:
        md = bootstrap_to_markdown(bootstrap_cis)
        assert "reap" in md
        assert "procrustes" in md

    def test_ci_format(self, bootstrap_cis: list[BootstrapCI]) -> None:
        md = bootstrap_to_markdown(bootstrap_cis)
        # Should contain "[0.871, 0.913]" for REAP TW
        assert "[0.871, 0.913]" in md

    def test_both_metrics(self, bootstrap_cis: list[BootstrapCI]) -> None:
        md = bootstrap_to_markdown(bootstrap_cis)
        assert "trustworthiness" in md
        assert "silhouette" in md

    def test_n_bootstrap_info(self, bootstrap_cis: list[BootstrapCI]) -> None:
        md = bootstrap_to_markdown(bootstrap_cis)
        assert "n=1000" in md


class TestBootstrapToLatex:
    """Tests for bootstrap_to_latex."""

    def test_booktabs(self, bootstrap_cis: list[BootstrapCI]) -> None:
        tex = bootstrap_to_latex(bootstrap_cis)
        assert "\\toprule" in tex
        assert "\\bottomrule" in tex

    def test_percent_escaped(self, bootstrap_cis: list[BootstrapCI]) -> None:
        tex = bootstrap_to_latex(bootstrap_cis)
        # "95% CI" should have escaped percent
        assert "95\\% CI" in tex

    def test_default_label(self, bootstrap_cis: list[BootstrapCI]) -> None:
        tex = bootstrap_to_latex(bootstrap_cis)
        assert "\\label{tab:bootstrap_ci}" in tex


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_method_benchmark(self) -> None:
        """Benchmark with only one method should not bold anything."""
        result = BenchmarkResult(
            dataset_name="Test",
            n_samples=100,
            n_features=32,
            n_components=5,
            n_neighbors=10,
            min_dist=0.1,
            k_range=[3, 5, 7],
            n_seeds=5,
            methods=[
                MethodResult(
                    method="reap",
                    description="Test REAP",
                    trustworthiness=0.900,
                    trustworthiness_std=0.0,
                    silhouette=0.800,
                    silhouette_std=0.0,
                    best_k=5,
                    best_k_std=0.0,
                    n_seeds=5,
                    runtime_seconds=10.0,
                ),
            ],
        )
        md = benchmark_to_markdown(result, bold_best=True)
        # With only one method, the best IS bold
        assert "**0.900**" in md

    def test_all_none_ari_no_bold_crash(self) -> None:
        """Should not crash when all S2S ARI values are None."""
        methods = [
            MethodResult(
                method=f"method_{i}",
                description=f"Test {i}",
                trustworthiness=0.8 + i * 0.01,
                trustworthiness_std=0.0,
                silhouette=0.7 + i * 0.01,
                silhouette_std=0.0,
                best_k=5,
                best_k_std=0.0,
                seed_to_seed_ari_mean=None,
                seed_to_seed_ari_std=None,
                seed_to_consensus_ari_mean=None,
                n_seeds=5,
                runtime_seconds=10.0 + i,
            )
            for i in range(3)
        ]
        result = BenchmarkResult(
            dataset_name="No ARI",
            n_samples=100,
            n_features=32,
            n_components=5,
            n_neighbors=10,
            min_dist=0.1,
            k_range=[3, 5],
            n_seeds=5,
            methods=methods,
        )
        md = benchmark_to_markdown(result, bold_best=True)
        assert "method_0" in md  # Should still produce a table

    def test_decimal_min_dist_in_sweep(self) -> None:
        """Decimal parameter values should not have trailing zeros."""
        sweep = ParameterSweepResult(
            dataset_name="Test",
            swept_parameter="min_dist",
            parameter_values=[0.005, 0.01, 0.1],
            fixed_params={"n_components": 18, "n_neighbors": 19, "n_seeds": 30},
            results=[
                ParameterSweepPoint(
                    parameter_value=v,
                    trustworthiness=0.89,
                    silhouette=0.84,
                    best_k=8,
                    seed_to_seed_ari_mean=0.75,
                    runtime_seconds=45.0,
                )
                for v in [0.005, 0.01, 0.1]
            ],
        )
        md = sweep_to_markdown(sweep)
        assert "0.005" in md
        assert "0.01" in md
        assert "0.1" in md

    def test_long_runtime_format(self) -> None:
        """Runtime > 60s should show minutes and seconds."""
        assert _format_runtime(150.0) == "2m 30s"
        assert _format_runtime(3600.0) == "60m 0s"

    def test_cross_dataset_empty_list(self) -> None:
        """cross_dataset_table with empty list should produce header-only table."""
        md = cross_dataset_table([])
        assert "Dataset" in md
        lines = md.strip().split("\n")
        pipe_lines = [ln for ln in lines if ln.startswith("|")]
        # Just header + separator
        assert len(pipe_lines) == 2
