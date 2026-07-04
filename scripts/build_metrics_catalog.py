"""Build the populated metrics catalog from the on-disk result artifacts.

Reads the benchmark artifact families (per-corpus ``all_methods.csv``,
``pairwise_tests.csv``, ``between_set_reliability.csv``) from the
REAP-research artifact tree, plus the projection-head ``oos_metrics.json``
files tracked in REAP, and assembles one validated
:class:`reap.metrics_catalog.MetricsCatalog` binding every value to a named
recipe and its provenance. Writes the catalog atomically to
``<research-root>/results/metrics_catalog/metrics_catalog.json``.

Fail-closed behavior (this feeds paper numbers): a discovered corpus with a
missing expected artifact, an unknown metric name in ``pairwise_tests.csv``,
or an unparseable file stops the build with an error — it never quietly
catalogs "whatever is there". Blank cells are the one allowed absence (a
metric genuinely not applicable to a method/corpus, e.g. expert-label
metrics on an unlabeled corpus).

Side effects: writes the catalog file (and its parent directory).

Usage:
    python scripts/build_metrics_catalog.py [--research-root PATH]
        [--reap-root PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from reap.metrics_catalog import (
    ComparisonRecord,
    MetricRecipe,
    MetricRecord,
    MetricsCatalog,
    write_metrics_catalog,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_root import get_research_root  # noqa: E402  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

REAP_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

SEED_SETS = ("A", "B", "C")


def _f(cell: str | None) -> float | None:
    """Parse a CSV cell to float; blank/missing means not-applicable."""
    if cell is None or cell.strip() == "":
        return None
    return float(cell)


# --- Recipe registry -------------------------------------------------------
# One recipe per calculation. Definitions record the variant, pairing,
# cluster-count rule, and the implementation defaults that move the number.

_CONSENSUS_NOTE = (
    "computed on the final consensus embedding with its consensus KMeans "
    "labels (K chosen once by silhouette search, KMeans random_state=42)"
)
_PER_SEED_NOTE = (
    "computed per seed on that seed's own UMAP embedding (own KMeans labels "
    "where labels are needed), then mean/std across the 30-seed set"
)


def build_recipe_registry() -> list[MetricRecipe]:
    """Return every recipe the builder can emit records against."""
    recipes: list[MetricRecipe] = []

    def add(recipe_id: str, name: str, definition: str, implementation: str | None,
            parameters: dict[str, str | int | float | bool | None] | None = None,
            kind: str = "measured", verification: str | None = None) -> None:
        recipes.append(MetricRecipe(
            recipe_id=recipe_id, name=name,
            kind=kind,  # type: ignore[arg-type]
            definition=definition, implementation=implementation,
            parameters=parameters or {}, verification=verification,
        ))

    for scope, note in (("consensus", _CONSENSUS_NOTE), ("per_seed_mean", _PER_SEED_NOTE)):
        add(f"silhouette.{scope}", f"Silhouette ({scope})",
            f"sklearn silhouette_score, Euclidean, {note}.",
            "reap.evaluation.compute_silhouette", {"metric": "euclidean"})
        add(f"trustworthiness.{scope}", f"Trustworthiness ({scope})",
            f"sklearn.manifold.trustworthiness of the low-d embedding vs the original "
            f"high-d space, {note}.",
            "reap.evaluation.compute_trustworthiness",
            {"n_neighbors": 15, "metric": "cosine"})
        add(f"continuity.{scope}", f"Continuity ({scope})",
            f"Continuity (trustworthiness with the roles of the two spaces swapped), {note}.",
            "reap.evaluation.compute_continuity", {"n_neighbors": 15, "metric": "cosine"})
        add(f"davies_bouldin.{scope}", f"Davies-Bouldin index ({scope}; lower is better)",
            f"sklearn davies_bouldin_score, {note}.",
            "reap.evaluation.compute_davies_bouldin")
        add(f"calinski_harabasz.{scope}", f"Calinski-Harabasz index ({scope})",
            f"sklearn calinski_harabasz_score, {note}.",
            "reap.evaluation.compute_calinski_harabasz")

    add("cluster_count.consensus", "Selected K (consensus)",
        "Number of clusters chosen once on the consensus embedding by silhouette search.",
        "reap.clustering.find_best_k")
    add("cluster_count.per_seed_best_k", "Selected K (per-seed mean)",
        "Per-seed silhouette-selected K, mean/std across seeds.",
        "reap.clustering.find_best_k")

    for pairing, pair_def in (
        ("seed_to_seed", "between every pair of per-seed KMeans labelings "
                         "(upper triangle of the 30x30 agreement matrix, mean/std with ddof=0)"),
        ("seed_to_consensus", "between each seed's KMeans labeling and the consensus labeling "
                              "(mean/std over the 30 seeds)"),
    ):
        for m, impl in (("ari", "adjusted_rand_score"), ("ami", "adjusted_mutual_info_score"),
                        ("nmi", "normalized_mutual_info_score")):
            add(f"{m}.{pairing}", f"{m.upper()} ({pairing.replace('_', '-')})",
                f"sklearn {impl} {pair_def}.",
                f"sklearn.metrics.{impl}",
                verification="ARI variants pinned by the tests/verification ladder (rung 4)")

    ext = {
        "ari": "adjusted_rand_score", "ami": "adjusted_mutual_info_score",
        "nmi": "normalized_mutual_info_score", "homogeneity": "homogeneity_score",
        "completeness": "completeness_score", "v_measure": "v_measure_score",
        "fowlkes_mallows": "fowlkes_mallows_score",
        "mean_purity": "(cluster-majority purity, mean over clusters)",
    }
    for m, impl in ext.items():
        impl_path = None if impl.startswith("(") else f"sklearn.metrics.{impl}"
        add(f"{m}.consensus_vs_expert", f"{m} (consensus vs expert labels)",
            "Agreement between the consensus KMeans labels (K chosen once by silhouette) "
            "and the corpus's full expert/ground-truth label set, all rows. NOTE: this is "
            "one of several ARI-family pairings in this project - never compare it against "
            "numbers computed under a different pairing or K.",
            impl_path,
            verification="independent re-derivation for korean_forest set A "
                         "(docs/verification/ari_ladder_log.md, Adjudication)")
        add(f"{m}.per_seed_vs_expert_mean", f"{m} (per-seed vs expert labels, mean)",
            "Mean over the 30 seeds of the agreement between each seed's own KMeans "
            "labels and the expert/ground-truth label set - the single_seed method's "
            "external-validity recipe, numerically different from the consensus pairing "
            "under the same column name.",
            impl_path)

    for m, definition in (
        ("topic_coherence.umass", "UMass topic coherence over c-TF-IDF top terms"),
        ("topic_coherence.npmi", "NPMI topic coherence over c-TF-IDF top terms"),
        ("topic_coherence.cv", "C_v topic coherence over c-TF-IDF top terms"),
        ("topic_diversity", "Fraction of unique words across cluster top-term lists"),
        ("topic_exclusivity", "Mean exclusivity of cluster top terms"),
    ):
        add(m, m.replace(".", " "), definition + ".", "reap.coherence")

    for stat in ("mean", "min", "median"):
        add(f"cluster_persistence.jaccard.{stat}", f"Cluster persistence ({stat})",
            f"Per-cluster Jaccard persistence across per-seed clusterings "
            f"(match threshold 0.5), {stat} over clusters.",
            "reap.evaluation.compute_cluster_persistence", {"threshold": 0.5})

    for m in ("ari", "ami", "nmi"):
        add(f"{m}.between_set", f"{m.upper()} (between seed-set consensus labelings)",
            f"sklearn {m} between the consensus labelings produced from two disjoint "
            "pre-registered 30-seed sets of the same method - a replication/reliability "
            "measure, not an accuracy measure.", f"sklearn.metrics.{m}")
    add("frobenius.between_set_relative", "Relative Frobenius distance (between sets)",
        "||D_a - D_b||_F / ||D_a||_F between consensus distance matrices of two "
        "disjoint seed sets (REAP method only).", "numpy.linalg.norm")

    for mode, mode_note, mode_id in (
        ("cv", "held-out fold of a 5-fold StratifiedKFold (shuffle, random_state=42), "
               "mean/std over folds", "projection_cv"),
        ("final-fit", "fit on all points and evaluated on the same points - in-sample "
                      "and optimistic; prefer the CV numbers for any inferential claim",
         "projection_final_fit"),
    ):
        add(f"mse.{mode_id}", f"Projection-head MSE ({mode})",
            f"Mean squared error between true and predicted consensus coordinates, {mode_note}.",
            "reap.projection.train_projection_head")
        add(f"trustworthiness.{mode_id}", f"Projection-head trustworthiness ({mode})",
            f"Trustworthiness of projected points vs the original high-d space, "
            f"n_neighbors=min(15, n-1), {mode_note}.",
            "reap.evaluation.compute_trustworthiness", {"metric": "cosine"})
        add(f"silhouette.{mode_id}", f"Projection-head silhouette ({mode})",
            f"Silhouette of projected points under consensus labels, {mode_note}.",
            "reap.evaluation.compute_silhouette")
        add(f"ari.{mode_id}", f"Projection-head ARI ({mode})",
            f"adjusted_rand_score between consensus labels and KMeans (K = number of "
            f"unique labels, fixed random_state=42) on projected points, {mode_note}.",
            "sklearn.metrics.adjusted_rand_score")
        add(f"distance_correlation.{mode_id}", f"Projection-head distance correlation ({mode})",
            f"Pearson correlation between condensed Euclidean pairwise-distance vectors "
            f"of true vs predicted coordinates, {mode_note}. This is NOT Szekely's "
            f"distance correlation despite the name, and it is numerically distinct from "
            f"the full-matrix training-loss variant.",
            "reap.evaluation.compute_distance_correlation", kind="adapted")

    return recipes


# Column -> (recipe_id, std_column) for all_methods.csv scalar records.
ALL_METHODS_MAP: dict[str, tuple[str, str | None]] = {
    "consensus_silhouette": ("silhouette.consensus", None),
    "consensus_trustworthiness": ("trustworthiness.consensus", None),
    "consensus_continuity": ("continuity.consensus", None),
    "consensus_davies_bouldin": ("davies_bouldin.consensus", None),
    "consensus_calinski_harabasz": ("calinski_harabasz.consensus", None),
    "consensus_k": ("cluster_count.consensus", None),
    "continuity_mean": ("continuity.per_seed_mean", "continuity_std"),
    "davies_bouldin_mean": ("davies_bouldin.per_seed_mean", "davies_bouldin_std"),
    "calinski_harabasz_mean": ("calinski_harabasz.per_seed_mean", "calinski_harabasz_std"),
    "best_k": ("cluster_count.per_seed_best_k", "best_k_std"),
    "s2s_ari_mean": ("ari.seed_to_seed", "s2s_ari_std"),
    "s2s_ami_mean": ("ami.seed_to_seed", "s2s_ami_std"),
    "s2s_nmi_mean": ("nmi.seed_to_seed", "s2s_nmi_std"),
    "s2c_ari_mean": ("ari.seed_to_consensus", "s2c_ari_std"),
    "s2c_ami_mean": ("ami.seed_to_consensus", "s2c_ami_std"),
    "s2c_nmi_mean": ("nmi.seed_to_consensus", "s2c_nmi_std"),
    "ext_ari": ("ari.consensus_vs_expert", None),
    "ext_ami": ("ami.consensus_vs_expert", None),
    "ext_nmi": ("nmi.consensus_vs_expert", None),
    "ext_homogeneity": ("homogeneity.consensus_vs_expert", None),
    "ext_completeness": ("completeness.consensus_vs_expert", None),
    "ext_v_measure": ("v_measure.consensus_vs_expert", None),
    "ext_fowlkes_mallows": ("fowlkes_mallows.consensus_vs_expert", None),
    "ext_mean_purity": ("mean_purity.consensus_vs_expert", None),
    "topic_coherence_umass": ("topic_coherence.umass", None),
    "topic_coherence_npmi": ("topic_coherence.npmi", None),
    "topic_coherence_cv": ("topic_coherence.cv", None),
    "topic_diversity": ("topic_diversity", None),
    "topic_exclusivity": ("topic_exclusivity", None),
    "cluster_persistence_mean": ("cluster_persistence.jaccard.mean", None),
    "cluster_persistence_min": ("cluster_persistence.jaccard.min", None),
    "cluster_persistence_median": ("cluster_persistence.jaccard.median", None),
}

# pairwise_tests.csv metric names (set suffix stripped) -> recipe ids.
PAIRWISE_MAP: dict[str, str] = {
    "silhouette": "silhouette.per_seed_mean",
    "trustworthiness": "trustworthiness.per_seed_mean",
    "continuity": "continuity.per_seed_mean",
    "davies_bouldin": "davies_bouldin.per_seed_mean",
    "calinski_harabasz": "calinski_harabasz.per_seed_mean",
    "s2c_ari": "ari.seed_to_consensus",
    "s2c_ami": "ami.seed_to_consensus",
    "s2c_nmi": "nmi.seed_to_consensus",
}

# (dataset, method, seed_set, recipe_id) -> ladder citation for values whose
# implementation AND value were independently re-derived.
VERIFIED_INDEPENDENT: dict[tuple[str, str, str, str], str] = {
    ("korean_forest", "reap", "A", "ari.consensus_vs_expert"):
        "docs/verification/ari_ladder_log.md Adjudication: from-scratch Hubert-Arabie "
        "ARI reproduces 0.122634 to 1e-7",
    ("korean_forest", "reap", "A", "trustworthiness.consensus"):
        "docs/verification/trustworthiness_ladder_log.md: independent re-derivation",
    ("korean_forest", "reap", "B", "trustworthiness.consensus"):
        "docs/verification/trustworthiness_ladder_log.md: independent re-derivation",
    ("korean_forest", "reap", "C", "trustworthiness.consensus"):
        "docs/verification/trustworthiness_ladder_log.md: independent re-derivation",
    ("korean_forest", "reap", "A", "silhouette.consensus"):
        "docs/verification/silhouette_ladder_log.md: independent re-derivation",
    ("korean_forest", "reap", "A", "ari.seed_to_seed"):
        "docs/verification/ari_ladder_log.md rung-4 table: re-derivation delta < 1e-6",
    ("korean_forest", "reap", "A", "ari.seed_to_consensus"):
        "docs/verification/ari_ladder_log.md rung-4 table: re-derivation delta 3.11e-07",
}


def _read_bundle(path: Path) -> tuple[str | None, list[int] | None]:
    """Read (git_commit, seeds) from a legacy bundle.json if present."""
    if not path.is_file():
        return None, None
    d = json.loads(path.read_text())
    seeds = d.get("seeds")
    return d.get("git_commit"), list(seeds) if seeds else None


def build_benchmark_records(
    corpus: str, research_root: Path, reap_root: Path,
) -> list[MetricRecord]:
    """Records from all_methods.csv (x3 sets) + between_set_reliability.csv."""
    records: list[MetricRecord] = []
    for seed_set in SEED_SETS:
        csv_path = (research_root / "results" / corpus
                    / f"combined_set_{seed_set}" / "all_methods.csv")
        if not csv_path.is_file():
            raise RuntimeError(f"Expected artifact missing: {csv_path}")
        commit, seeds = _read_bundle(
            reap_root / "results" / corpus / f"combined_set_{seed_set}" / "bundle.json")
        status = "bundle-backed" if commit else "legacy-artifact"
        src = f"REAP-research:results/{corpus}/combined_set_{seed_set}/all_methods.csv"
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                method = row["method"]
                n_seeds = int(float(row["n_seeds"])) if row.get("n_seeds") else None
                col_map = dict(ALL_METHODS_MAP)
                # The headline silhouette/trustworthiness columns are a known
                # semantic trap: per-seed mean +/- std for single_seed, but the
                # consensus value (std 0.0) for consensus methods - where they
                # duplicate the consensus_* columns and are skipped.
                if method == "single_seed":
                    col_map["silhouette"] = ("silhouette.per_seed_mean", "silhouette_std")
                    col_map["trustworthiness"] = (
                        "trustworthiness.per_seed_mean", "trustworthiness_std")
                    # single_seed's external validity is a different recipe:
                    # per-seed-vs-expert mean, not consensus-vs-expert.
                    for m in ("ari", "ami", "nmi", "homogeneity", "completeness",
                              "v_measure", "fowlkes_mallows", "mean_purity"):
                        col_map[f"ext_{m}"] = (f"{m}.per_seed_vs_expert_mean", None)
                for col, (recipe_id, std_col) in col_map.items():
                    value = _f(row.get(col))
                    if value is None:
                        continue
                    per_seed = col not in ("consensus_k",) and (
                        std_col is not None or col.startswith(("s2s_", "s2c_")))
                    caveats: list[str] = []
                    if col.startswith("s2s_"):
                        caveats.append(
                            "Property of the shared per-seed clustering stage, identical "
                            "across methods within a seed set - never phrase as a "
                            "method-level claim.")
                    records.append(MetricRecord(
                        dataset=corpus, method=method, seed_set=seed_set,
                        seeds=seeds, evaluation_mode="in_sample",
                        recipe_id=recipe_id, value=value,
                        value_std=_f(row.get(std_col)) if std_col else None,
                        n=n_seeds if per_seed else 1,
                        source_artifact=src,
                        source_detail=f"row method={method}, column {col}",
                        produced_by="scripts/run_paper_benchmark.py",
                        code_commit=commit,
                        provenance_status=status,  # type: ignore[arg-type]
                        caveats=caveats,
                    ))
    rel_path = research_root / "results" / corpus / "between_set_reliability.csv"
    if not rel_path.is_file():
        raise RuntimeError(f"Expected artifact missing: {rel_path}")
    with open(rel_path, newline="") as fh:
        for row in csv.DictReader(fh):
            for col, recipe_id in (("ari", "ari.between_set"), ("ami", "ami.between_set"),
                                   ("nmi", "nmi.between_set"),
                                   ("frobenius_relative", "frobenius.between_set_relative")):
                value = _f(row.get(col))
                if value is None:
                    continue
                records.append(MetricRecord(
                    dataset=corpus, method=row["method"], seed_set=None,
                    evaluation_mode="in_sample", recipe_id=recipe_id, value=value,
                    source_artifact=f"REAP-research:results/{corpus}/between_set_reliability.csv",
                    source_detail=(f"row method={row['method']}, sets "
                                   f"{row['set_a']}-{row['set_b']}, column {col}"),
                    produced_by="scripts/run_paper_benchmark.py",
                    provenance_status="legacy-artifact",
                ))
    return records


def build_comparison_records(corpus: str, research_root: Path) -> list[ComparisonRecord]:
    """ComparisonRecords from pairwise_tests.csv; unknown metric names raise."""
    path = research_root / "results" / corpus / "pairwise_tests.csv"
    if not path.is_file():
        raise RuntimeError(f"Expected artifact missing: {path}")
    out: list[ComparisonRecord] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            base = row["metric"]
            for suffix in ("_A", "_B", "_C"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            if base not in PAIRWISE_MAP:
                raise RuntimeError(
                    f"Unknown metric {row['metric']!r} in {path}; extend PAIRWISE_MAP "
                    "rather than skipping (fail closed).")
            out.append(ComparisonRecord(
                dataset=corpus, seed_set=row["seed_set"], evaluation_mode="in_sample",
                recipe_id=PAIRWISE_MAP[base],
                reference_method=row["reference"], comparator_method=row["comparator"],
                direction=row["direction"],  # type: ignore[arg-type]
                n=int(float(row["n"])) if row.get("n") else None,
                reference_mean=_f(row.get("ref_mean")),
                mean_diff=_f(row.get("mean_diff")), median_diff=_f(row.get("median_diff")),
                test_name="wilcoxon-signed-rank", statistic=_f(row.get("wilcoxon_W")),
                p_raw=_f(row.get("wilcoxon_p_raw")), p_holm=_f(row.get("wilcoxon_p_holm")),
                p_bh=_f(row.get("wilcoxon_p_bh")),
                cohens_d=_f(row.get("cohens_d")), cliffs_delta=_f(row.get("cliffs_delta")),
                ci_low=_f(row.get("ref_ci_lower")), ci_high=_f(row.get("ref_ci_upper")),
                ci_method="bootstrap (reference-mean CI)",
                source_artifact=f"REAP-research:results/{corpus}/pairwise_tests.csv",
                source_detail=(f"row metric={row['metric']}, {row['reference']} vs "
                               f"{row['comparator']}, set {row['seed_set']}"),
                provenance_status="legacy-artifact",
                caveats=["Paired on matched per-seed vectors; Holm family = per "
                         "(seed set, metric) as computed by the harness, pending "
                         "the pre-registered run-matrix family definition."],
            ))
    return out


def build_projection_records(reap_root: Path) -> list[MetricRecord]:
    """Records from every results/projection_head/*/set_*/oos_metrics.json."""
    out: list[MetricRecord] = []
    for path in sorted(reap_root.glob("results/projection_head/*/set_*/oos_metrics.json")):
        d = json.loads(path.read_text())
        corpus = d.get("corpus", path.parts[-3])
        seed_set = d.get("set", path.parts[-2].split("_")[-1])
        rel = path.relative_to(reap_root)
        common = dict(
            dataset=corpus, method="projection_head_mlp", seed_set=seed_set,
            source_artifact=f"REAP:{rel}", produced_by="scripts/run_projection_head_oos.py",
            code_commit=d.get("git_commit"), provenance_status="bundle-backed",
        )
        for metric, block in d.get("cv_summary", {}).items():
            recipe = f"{metric}.projection_cv"
            out.append(MetricRecord(
                **common, evaluation_mode="cv", recipe_id=recipe,
                value=block.get("mean"), value_std=block.get("std"),
                n=len(d.get("per_fold", [])) or None,
                source_detail=f"cv_summary.{metric}",
            ))
        for metric, value in d.get("final_metrics", {}).items():
            out.append(MetricRecord(
                **common, evaluation_mode="in_sample",
                recipe_id=f"{metric}.projection_final_fit", value=value,
                source_detail=f"final_metrics.{metric}",
                caveats=["Final-fit: evaluated on the training points; optimistic. "
                         "Prefer the CV numbers for inferential claims."],
            ))
    return out


def apply_verified_overrides(records: list[MetricRecord]) -> list[MetricRecord]:
    """Upgrade provenance to verified-independent where a ladder re-derived the value."""
    out: list[MetricRecord] = []
    for r in records:
        key = (r.dataset, r.method, r.seed_set or "", r.recipe_id)
        citation = VERIFIED_INDEPENDENT.get(key)
        if citation:
            out.append(r.model_copy(update={
                "provenance_status": "verified-independent",
                "caveats": [*r.caveats, f"Independently re-derived: {citation}"],
            }))
        else:
            out.append(r)
    return out


def build_catalog(research_root: Path, reap_root: Path) -> MetricsCatalog:
    """Assemble the full catalog from both artifact trees. Pure aside from reads."""
    corpora = sorted(
        p.parent.parent.name
        for p in research_root.glob("results/*/combined_set_A/all_methods.csv")
    )
    if not corpora:
        raise RuntimeError(
            f"No corpus with combined_set_A/all_methods.csv under {research_root}/results "
            "- refusing to build an empty catalog.")
    records: list[MetricRecord] = []
    comparisons: list[ComparisonRecord] = []
    for corpus in corpora:
        records.extend(build_benchmark_records(corpus, research_root, reap_root))
        comparisons.extend(build_comparison_records(corpus, research_root))
    records.extend(build_projection_records(reap_root))
    records = apply_verified_overrides(records)

    def head(root: Path) -> str:
        try:
            return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return MetricsCatalog(
        built_by="scripts/build_metrics_catalog.py",
        built_from={"REAP": head(reap_root), "REAP-research": head(research_root)},
        description=(
            f"Benchmark, reliability, paired-comparison, and projection-head metrics for "
            f"{', '.join(corpora)}. Not yet cataloged: cross-source A1/A2/A3, OOS-demo "
            f"probe metrics, common-space silhouette, 20NG OOS-bridge (extend the builder)."
        ),
        recipes=build_recipe_registry(),
        records=records,
        comparisons=comparisons,
    )


def main(argv: list[str] | None = None) -> int:
    """Build and write the catalog; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", default=None,
                        help="REAP-research root (default: env REAP_RESEARCH_ROOT, "
                             "then ../REAP-research)")
    parser.add_argument("--reap-root", default=str(REAP_ROOT_DEFAULT),
                        help="REAP repo root (default: this script's repo)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <research-root>/results/"
                             "metrics_catalog/metrics_catalog.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    research_root = get_research_root(args.research_root)
    reap_root = Path(args.reap_root).resolve()
    catalog = build_catalog(research_root, reap_root)
    output = Path(args.output) if args.output else (
        research_root / "results" / "metrics_catalog" / "metrics_catalog.json")
    write_metrics_catalog(catalog, output)
    logger.info("Catalog: %d recipes, %d records, %d comparisons",
                len(catalog.recipes), len(catalog.records), len(catalog.comparisons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
