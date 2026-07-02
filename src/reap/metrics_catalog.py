"""Typed catalog for every metric, statistic, and paired comparison REAP produces.

The problem this solves: benchmark and analysis runs emit dozens of metrics per
(dataset, method, seed set) cell, scattered across CSVs and JSONs with nothing
recording *exactly which calculation* each number is ("ARI" alone names several
different quantities in this project). A catalog entry binds every value to a
named recipe (the exact calculation), its provenance (which artifact, which
code commit), and its evaluation mode — so any number can be looked up, quoted
with its definition, and traced back to a committed file.

Three layers:

- ``MetricRecipe`` — the exact definition of one metric variant (what is
  computed, by which implementation, with which parameters, and how it was
  verified). One recipe per *calculation*, not per metric name.
- ``MetricRecord`` — one recorded value of one recipe on one
  (dataset, method, seed set, evaluation mode) cell, with dispersion and
  provenance.
- ``ComparisonRecord`` — one paired method-vs-method comparison of one recipe
  (test statistic, corrected p-values, effect sizes), matching the project's
  report-everything framing: comparisons are recorded, never gated on.

``MetricsCatalog`` holds all three and fails closed at validation time:
duplicate recipe ids, dangling recipe references, or unknown fields raise
instead of loading. Files are written atomically (temp file, then rename).

Two numbers may only be compared when they share a ``recipe_id`` — that rule
is what the recipe layer exists to make mechanical.

Exports: ``MetricRecipe``, ``MetricRecord``, ``ComparisonRecord``,
``MetricsCatalog``, ``get_catalog_records``, ``get_catalog_recipe``,
``write_metrics_catalog``, ``read_metrics_catalog``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

METRICS_CATALOG_SCHEMA_VERSION = "1.0"

ParamValue = str | int | float | bool | None

EvaluationMode = Literal["in_sample", "cv", "oos"]

ProvenanceStatus = Literal[
    # independently re-derived (e.g. a verification-ladder value)
    "verified-independent",
    # a sibling record file documents commit/seeds/config
    "bundle-backed",
    # committed artifact predating the versioned record-file format
    "legacy-artifact",
    # exists only in prose; NOT citable as a result
    "prose-only",
    # a number produced outside this project's harness
    "external-reference",
]


class MetricRecipe(BaseModel):
    """The exact definition of one metric variant.

    One recipe per *calculation*: if two numbers differ in label pairing,
    cluster count, subset, distance metric, or aggregation, they get two
    recipes — even under the same everyday name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_id: str = Field(
        ...,
        description=(
            "Canonical id, dot-separated from general to specific, e.g. "
            "'ari.seed_to_consensus' or 'trustworthiness.cosine.k12'. Unique "
            "within a catalog; the join key for records and comparisons."
        ),
    )
    name: str = Field(
        ...,
        description=(
            "Human-readable metric name, e.g. "
            "'Adjusted Rand Index (seed vs consensus)'."
        ),
    )
    kind: Literal["measured", "adapted", "external-reference"] = Field(
        ...,
        description=(
            "'measured' = standard quantity computed directly here; 'adapted' "
            "= a project-defined variant (must not be compared to outside "
            "numbers under the standard name); 'external-reference' = a "
            "number produced outside this harness."
        ),
    )
    definition: str = Field(
        ...,
        description=(
            "The exact calculation in plain language: variant, label pairing, "
            "cluster-count rule, distance metric, subset, and aggregation. "
            "Precise enough that a reader could re-implement it."
        ),
    )
    implementation: str | None = Field(
        default=None,
        description=(
            "Dotted path of the producing implementation, e.g. "
            "'reap.evaluation.compute_trustworthiness'; None for external "
            "references."
        ),
    )
    parameters: dict[str, ParamValue] = Field(
        default_factory=dict,
        description=(
            "Parameters that move the number (k, metric=, random_state, "
            "normalization), including implementation defaults relied upon."
        ),
    )
    verification: str | None = Field(
        default=None,
        description=(
            "How the implementation is checked: test path or "
            "verification-ladder rung; None = unverified (a fact worth "
            "recording, not hiding)."
        ),
    )
    tolerance_note: str | None = Field(
        default=None,
        description=(
            "If any check allows numerical wiggle: how much and why. "
            "A tolerance without a reason is a bug."
        ),
    )


class MetricRecord(BaseModel):
    """One recorded value of one recipe on one experiment cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(
        ...,
        description=(
            "Dataset id, e.g. 'korean_forest', 'ai_art', "
            "'twenty_newsgroups_reference'."
        ),
    )
    method: str = Field(
        ...,
        description=(
            "Method/pipeline that produced the embedding or labels, e.g. "
            "'reap', 'procrustes', 'best_of_n', 'projection_head_mlp'."
        ),
    )
    encoder: str | None = Field(
        default=None,
        description=(
            "Embedding model when a dataset has several; None when "
            "unambiguous."
        ),
    )
    seed_set: str | None = Field(
        default=None,
        description=(
            "Pre-registered seed-set label ('A', 'B', 'C', 'D') or None when "
            "seeds are not the unit of replication."
        ),
    )
    seeds: list[int] | None = Field(
        default=None,
        description="The exact seeds consumed, when known.",
    )
    evaluation_mode: EvaluationMode = Field(
        ...,
        description=(
            "'in_sample' (fit and evaluated on the same data), 'cv' "
            "(cross-validated), or 'oos' (held-out / out-of-sample)."
        ),
    )
    recipe_id: str = Field(
        ...,
        description=(
            "Which MetricRecipe this value instantiates. Must resolve within "
            "the catalog."
        ),
    )
    value: float | None = Field(
        default=None,
        description=(
            "The scalar value (full precision, never pre-rounded). None when "
            "only per-seed values exist."
        ),
    )
    value_std: float | None = Field(
        default=None,
        description=(
            "Standard deviation across seeds/folds, when reported by the "
            "source artifact."
        ),
    )
    ci_low: float | None = Field(
        default=None, description="Lower confidence bound, if computed."
    )
    ci_high: float | None = Field(
        default=None, description="Upper confidence bound, if computed."
    )
    ci_method: str | None = Field(
        default=None,
        description=(
            "How the CI was computed (e.g. 'bootstrap-10k', 't'), when "
            "bounds are present."
        ),
    )
    n: int | None = Field(
        default=None,
        description=(
            "Replication count behind value/std/CI (seeds, folds, or items — "
            "the recipe says which)."
        ),
    )
    per_seed_values: list[float] | None = Field(
        default=None,
        description=(
            "Per-seed (or per-fold) values when the source artifact carries "
            "them."
        ),
    )
    source_artifact: str = Field(
        ...,
        description=(
            "Repo-qualified path of the committed artifact this value was "
            "read from, e.g. 'REAP-research:results/korean_forest/"
            "combined_set_A/all_methods.csv'."
        ),
    )
    source_detail: str | None = Field(
        default=None,
        description=(
            "Row/column/key inside the artifact, e.g. "
            "'row method=reap, column ext_ari'."
        ),
    )
    produced_by: str | None = Field(
        default=None,
        description="Script or function that wrote the source artifact, when known.",
    )
    code_commit: str | None = Field(
        default=None,
        description="Code commit that produced the artifact, when recorded.",
    )
    provenance_status: ProvenanceStatus = Field(
        ...,
        description=(
            "How strongly this value is backed; anything below "
            "'bundle-backed' needs work before publication."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Known interpretation limits (e.g. 'per-probe test; 1,259 probes "
            "from 252 artists — treat p-values as inflated')."
        ),
    )


class ComparisonRecord(BaseModel):
    """One paired method-vs-method comparison of one recipe.

    Both sides MUST share the recipe: comparing two different calculations is
    exactly the failure mode the catalog exists to prevent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(..., description="Dataset id.")
    encoder: str | None = Field(
        default=None,
        description="Embedding model, when a dataset has several.",
    )
    seed_set: str | None = Field(
        default=None,
        description=(
            "Seed-set label pairing the two sides, when seeds are the unit "
            "of replication."
        ),
    )
    evaluation_mode: EvaluationMode = Field(
        ..., description="Evaluation mode shared by both sides."
    )
    recipe_id: str = Field(
        ...,
        description=(
            "The single recipe both sides were computed under. Must resolve "
            "within the catalog."
        ),
    )
    reference_method: str = Field(
        ..., description="The method being compared (typically 'reap')."
    )
    comparator_method: str = Field(
        ..., description="The baseline being compared against."
    )
    direction: Literal["higher_is_better", "lower_is_better"] = Field(
        ...,
        description="Which way improvement points for this recipe.",
    )
    n: int | None = Field(
        default=None, description="Number of matched pairs behind the test."
    )
    reference_mean: float | None = Field(
        default=None,
        description="Mean of the reference method's paired values.",
    )
    mean_diff: float | None = Field(
        default=None,
        description="Mean paired difference (reference minus comparator).",
    )
    median_diff: float | None = Field(
        default=None, description="Median paired difference."
    )
    test_name: str | None = Field(
        default=None,
        description="The paired test used, e.g. 'wilcoxon-signed-rank'.",
    )
    statistic: float | None = Field(
        default=None, description="The test statistic, when reported."
    )
    p_raw: float | None = Field(
        default=None, description="Uncorrected p-value."
    )
    p_holm: float | None = Field(
        default=None,
        description=(
            "Holm-corrected p-value (family defined by the pre-registered "
            "run matrix)."
        ),
    )
    p_bh: float | None = Field(
        default=None,
        description="Benjamini-Hochberg p-value, reported as sensitivity.",
    )
    cohens_d: float | None = Field(
        default=None, description="Paired Cohen's d effect size."
    )
    cliffs_delta: float | None = Field(
        default=None,
        description="Cliff's delta effect size (nonparametric).",
    )
    ci_low: float | None = Field(
        default=None,
        description="CI lower bound on the paired difference, if computed.",
    )
    ci_high: float | None = Field(
        default=None,
        description="CI upper bound on the paired difference, if computed.",
    )
    ci_method: str | None = Field(
        default=None,
        description="How the CI was computed, when bounds are present.",
    )
    source_artifact: str = Field(
        ...,
        description=(
            "Repo-qualified path of the committed artifact this comparison "
            "was read from."
        ),
    )
    source_detail: str | None = Field(
        default=None, description="Row/key inside the artifact."
    )
    code_commit: str | None = Field(
        default=None,
        description="Code commit that produced the artifact, when recorded.",
    )
    provenance_status: ProvenanceStatus = Field(
        ..., description="How strongly this comparison is backed."
    )
    caveats: list[str] = Field(
        default_factory=list, description="Known interpretation limits."
    )


class MetricsCatalog(BaseModel):
    """Every recipe, record, and comparison in one validated, versioned file.

    Validation fails closed: duplicate recipe ids, a record or comparison
    referencing an unknown recipe, or unknown fields anywhere raise a
    validation error instead of producing a partially-usable catalog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(
        default=METRICS_CATALOG_SCHEMA_VERSION,
        description=(
            "Catalog schema version; bump on any breaking field change."
        ),
    )
    built_by: str = Field(
        ..., description="Script or tool that assembled the catalog."
    )
    built_from: dict[str, str] = Field(
        ...,
        description=(
            "Provenance of the assembly, e.g. "
            "{'REAP': '<commit sha>', 'REAP-research': '<commit sha>'}."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "One-line scope note, e.g. which corpora/artifact families this "
            "catalog covers."
        ),
    )
    recipes: list[MetricRecipe] = Field(
        ...,
        description="Every metric recipe referenced by records/comparisons.",
    )
    records: list[MetricRecord] = Field(
        default_factory=list, description="All recorded metric values."
    )
    comparisons: list[ComparisonRecord] = Field(
        default_factory=list, description="All paired comparisons."
    )

    @model_validator(mode="after")
    def _validate_references(self) -> MetricsCatalog:
        """Fail closed on duplicate recipe ids or dangling recipe references."""
        seen: set[str] = set()
        for recipe in self.recipes:
            if recipe.recipe_id in seen:
                raise ValueError(
                    f"Duplicate recipe_id in catalog: {recipe.recipe_id!r}"
                )
            seen.add(recipe.recipe_id)
        for record in self.records:
            if record.recipe_id not in seen:
                raise ValueError(
                    f"MetricRecord references unknown recipe_id "
                    f"{record.recipe_id!r} (dataset={record.dataset!r}, "
                    f"method={record.method!r}, "
                    f"source={record.source_artifact!r})"
                )
        for comparison in self.comparisons:
            if comparison.recipe_id not in seen:
                raise ValueError(
                    f"ComparisonRecord references unknown recipe_id "
                    f"{comparison.recipe_id!r} "
                    f"({comparison.reference_method!r} vs "
                    f"{comparison.comparator_method!r} on "
                    f"{comparison.dataset!r})"
                )
        return self


def get_catalog_recipe(catalog: MetricsCatalog, recipe_id: str) -> MetricRecipe:
    """Return the recipe with the given id.

    Pure function. No side effects.

    Parameters
    ----------
    catalog : The catalog to search.
    recipe_id : The recipe id to resolve.

    Returns
    -------
    The matching ``MetricRecipe``.

    Raises
    ------
    KeyError : If no recipe has that id (fail closed — a typo'd recipe id
        must never silently return nothing).
    """
    for recipe in catalog.recipes:
        if recipe.recipe_id == recipe_id:
            return recipe
    raise KeyError(f"No recipe with id {recipe_id!r} in catalog")


def get_catalog_records(
    catalog: MetricsCatalog,
    dataset: str | None = None,
    method: str | None = None,
    recipe_id: str | None = None,
    seed_set: str | None = None,
    evaluation_mode: EvaluationMode | None = None,
) -> list[MetricRecord]:
    """Filter catalog records; every argument left as None matches everything.

    Pure function. No side effects.

    Parameters
    ----------
    catalog : The catalog to filter.
    dataset, method, recipe_id, seed_set, evaluation_mode : Optional equality
        filters, combined with AND.

    Returns
    -------
    The matching records, in catalog order.
    """
    out: list[MetricRecord] = []
    for record in catalog.records:
        if dataset is not None and record.dataset != dataset:
            continue
        if method is not None and record.method != method:
            continue
        if recipe_id is not None and record.recipe_id != recipe_id:
            continue
        if seed_set is not None and record.seed_set != seed_set:
            continue
        if evaluation_mode is not None and record.evaluation_mode != evaluation_mode:
            continue
        out.append(record)
    return out


def write_metrics_catalog(catalog: MetricsCatalog, path: Path) -> None:
    """Write a catalog to JSON atomically (temp file, then rename).

    Side effects: creates parent directories; writes ``path``.

    Parameters
    ----------
    catalog : The validated catalog to persist.
    path : Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(catalog.model_dump_json(indent=1))
    tmp.rename(path)
    logger.info(
        "Wrote metrics catalog: %d recipes, %d records, %d comparisons -> %s",
        len(catalog.recipes),
        len(catalog.records),
        len(catalog.comparisons),
        path,
    )


def read_metrics_catalog(path: Path) -> MetricsCatalog:
    """Read and fully validate a catalog from JSON.

    Fails closed: schema violations, unknown fields, duplicate recipe ids, or
    dangling recipe references raise instead of returning a partial catalog.

    Parameters
    ----------
    path : The catalog file to read.

    Returns
    -------
    The validated ``MetricsCatalog``.
    """
    raw = json.loads(Path(path).read_text())
    return MetricsCatalog.model_validate(raw)
