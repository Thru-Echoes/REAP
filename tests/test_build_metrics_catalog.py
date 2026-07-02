"""E2E test for scripts/build_metrics_catalog.py on a tiny synthetic artifact tree."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_metrics_catalog import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    build_catalog,
    get_research_root,
    main,
)

from reap.metrics_catalog import get_catalog_records, read_metrics_catalog  # noqa: E402

ALL_METHODS_HEADER = (
    "method,description,n_seeds,runtime_seconds,peak_memory_mb,"
    "silhouette,silhouette_std,trustworthiness,trustworthiness_std,"
    "continuity_mean,continuity_std,davies_bouldin_mean,davies_bouldin_std,"
    "calinski_harabasz_mean,calinski_harabasz_std,best_k,best_k_std,"
    "consensus_silhouette,consensus_trustworthiness,consensus_continuity,"
    "consensus_davies_bouldin,consensus_calinski_harabasz,consensus_k,"
    "s2s_ari_mean,s2s_ari_std,s2s_ami_mean,s2s_ami_std,s2s_nmi_mean,s2s_nmi_std,"
    "s2c_ari_mean,s2c_ari_std,s2c_ami_mean,s2c_ami_std,s2c_nmi_mean,s2c_nmi_std,"
    "ext_ari,ext_ami,ext_nmi,ext_homogeneity,ext_completeness,ext_v_measure,"
    "ext_fowlkes_mallows,ext_mean_purity,topic_coherence_umass,topic_coherence_npmi,"
    "topic_coherence_cv,topic_diversity,topic_exclusivity,"
    "cluster_persistence_mean,cluster_persistence_min,cluster_persistence_median"
).split(",")


def _write_tree(research: Path, reap: Path) -> None:
    """Write one synthetic corpus with the three benchmark artifact families."""
    corpus = research / "results" / "synthcorp"
    for seed_set in ("A", "B", "C"):
        d = corpus / f"combined_set_{seed_set}"
        d.mkdir(parents=True)
        with open(d / "all_methods.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ALL_METHODS_HEADER)
            w.writeheader()
            w.writerow({
                "method": "reap", "n_seeds": "30",
                "consensus_silhouette": "0.5", "consensus_trustworthiness": "0.9",
                "consensus_k": "8", "s2c_ari_mean": "0.6", "s2c_ari_std": "0.05",
                "ext_ari": "0.25",
            })
            w.writerow({
                "method": "single_seed", "n_seeds": "30",
                "silhouette": "0.4", "silhouette_std": "0.01", "ext_ari": "0.2",
            })
        bundle_dir = reap / "results" / "synthcorp" / f"combined_set_{seed_set}"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "bundle.json").write_text(
            json.dumps({"git_commit": "f" * 40, "seeds": [1, 2, 3]}))
    with open(corpus / "between_set_reliability.csv", "w") as fh:
        fh.write("method,set_a,set_b,ari,ami,nmi,frobenius_relative\n")
        fh.write("reap,A,B,0.8,0.85,0.9,0.1\n")
    with open(corpus / "pairwise_tests.csv", "w") as fh:
        fh.write(
            "seed_set,metric,direction,reference,comparator,n,ref_mean,"
            "ref_ci_lower,ref_ci_upper,mean_diff,median_diff,wilcoxon_W,"
            "wilcoxon_p_raw,wilcoxon_p_holm,wilcoxon_p_bh,cohens_d,cliffs_delta\n")
        fh.write("A,s2c_ari,higher_is_better,reap,single_seed,30,0.6,0.58,0.62,"
                 "0.1,0.09,10,0.001,0.002,0.0015,1.2,0.8\n")


class TestBuildCatalog:
    def test_end_to_end_build(self, tmp_path: Path):
        research, reap = tmp_path / "research", tmp_path / "reap"
        _write_tree(research, reap)
        catalog = build_catalog(research, reap)
        # single_seed's headline silhouette and ext_ari land under the
        # per-seed recipes, never the consensus ones.
        ss = get_catalog_records(catalog, method="single_seed", seed_set="A")
        assert {r.recipe_id for r in ss} == {
            "silhouette.per_seed_mean", "ari.per_seed_vs_expert_mean"}
        reap_ext = get_catalog_records(
            catalog, method="reap", seed_set="A", recipe_id="ari.consensus_vs_expert")
        assert reap_ext[0].value == 0.25
        assert reap_ext[0].code_commit == "f" * 40
        assert reap_ext[0].provenance_status == "bundle-backed"
        assert len(catalog.comparisons) == 1
        assert catalog.comparisons[0].recipe_id == "ari.seed_to_consensus"
        rel = get_catalog_records(catalog, recipe_id="ari.between_set")
        assert rel[0].source_detail is not None and "sets A-B" in rel[0].source_detail

    def test_missing_artifact_fails_closed(self, tmp_path: Path):
        research, reap = tmp_path / "research", tmp_path / "reap"
        _write_tree(research, reap)
        (research / "results/synthcorp/pairwise_tests.csv").unlink()
        with pytest.raises(RuntimeError, match="Expected artifact missing"):
            build_catalog(research, reap)

    def test_empty_tree_fails_closed(self, tmp_path: Path):
        (tmp_path / "results").mkdir()
        with pytest.raises(RuntimeError, match="No corpus"):
            build_catalog(tmp_path, tmp_path)

    def test_main_writes_valid_catalog(self, tmp_path: Path):
        research, reap = tmp_path / "research", tmp_path / "reap"
        _write_tree(research, reap)
        out = tmp_path / "catalog.json"
        code = main(["--research-root", str(research), "--reap-root", str(reap),
                     "--output", str(out)])
        assert code == 0
        assert len(read_metrics_catalog(out).records) > 0

    def test_locator_fails_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import build_metrics_catalog as bmc  # pyright: ignore[reportMissingImports]

        monkeypatch.delenv("REAP_RESEARCH_ROOT", raising=False)
        # Point the sibling-directory fallback at an empty location too, so
        # the real ../REAP-research checkout on a dev machine cannot satisfy it.
        monkeypatch.setattr(bmc, "REAP_ROOT_DEFAULT", tmp_path / "no-repo-here")
        with pytest.raises(RuntimeError, match="REAP_RESEARCH_ROOT"):
            get_research_root(str(tmp_path / "definitely-not-there"))
