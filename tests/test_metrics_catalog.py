"""Tests for the metrics catalog: schema strictness, fail-closed validation, atomic IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reap.metrics_catalog import (
    METRICS_CATALOG_SCHEMA_VERSION,
    ComparisonRecord,
    MetricRecipe,
    MetricRecord,
    MetricsCatalog,
    get_catalog_recipe,
    get_catalog_records,
    read_metrics_catalog,
    write_metrics_catalog,
)


def _recipe(recipe_id: str = "ari.consensus_vs_expert") -> MetricRecipe:
    return MetricRecipe(
        recipe_id=recipe_id,
        name="Adjusted Rand Index (consensus vs expert labels)",
        kind="measured",
        definition=(
            "sklearn adjusted_rand_score between the consensus KMeans labels "
            "(K chosen once by silhouette) and the full expert label set, all rows."
        ),
        implementation="sklearn.metrics.adjusted_rand_score",
        parameters={"k_rule": "silhouette-once", "subset": "all"},
        verification="tests/verification ARI ladder (independent re-derivation)",
    )


def _record(recipe_id: str = "ari.consensus_vs_expert") -> MetricRecord:
    return MetricRecord(
        dataset="korean_forest",
        method="reap",
        seed_set="A",
        evaluation_mode="in_sample",
        recipe_id=recipe_id,
        value=0.122634,
        n=1,
        source_artifact="REAP-research:results/korean_forest/combined_set_A/all_methods.csv",
        source_detail="row method=reap, column ext_ari",
        provenance_status="verified-independent",
    )


def _comparison(recipe_id: str = "ari.consensus_vs_expert") -> ComparisonRecord:
    return ComparisonRecord(
        dataset="korean_forest",
        seed_set="A",
        evaluation_mode="in_sample",
        recipe_id=recipe_id,
        reference_method="reap",
        comparator_method="procrustes",
        direction="higher_is_better",
        n=30,
        test_name="wilcoxon-signed-rank",
        p_holm=0.04,
        cohens_d=0.6,
        source_artifact="REAP-research:results/korean_forest/pairwise_tests.csv",
        provenance_status="legacy-artifact",
    )


def _catalog(
    recipes: list[MetricRecipe] | None = None,
    records: list[MetricRecord] | None = None,
    comparisons: list[ComparisonRecord] | None = None,
) -> MetricsCatalog:
    return MetricsCatalog(
        built_by="tests/test_metrics_catalog.py",
        built_from={"REAP": "0" * 40},
        recipes=recipes if recipes is not None else [_recipe()],
        records=records if records is not None else [_record()],
        comparisons=comparisons if comparisons is not None else [_comparison()],
    )


class TestSchemaStrictness:
    def test_valid_catalog_builds(self):
        catalog = _catalog()
        assert catalog.schema_version == METRICS_CATALOG_SCHEMA_VERSION
        assert len(catalog.records) == 1

    def test_models_are_frozen(self):
        record = _record()
        with pytest.raises(ValidationError):
            record.value = 0.5  # type: ignore[misc]

    def test_unknown_field_rejected(self):
        payload = _catalog().model_dump()
        payload["surprise"] = "not allowed"
        with pytest.raises(ValidationError):
            MetricsCatalog.model_validate(payload)

    def test_unknown_field_rejected_in_nested_record(self):
        payload = _catalog().model_dump()
        payload["records"][0]["surprise"] = 1
        with pytest.raises(ValidationError):
            MetricsCatalog.model_validate(payload)

    def test_missing_required_field_rejected(self):
        payload = _catalog().model_dump()
        del payload["records"][0]["source_artifact"]
        with pytest.raises(ValidationError):
            MetricsCatalog.model_validate(payload)

    def test_bad_provenance_status_rejected(self):
        payload = _catalog().model_dump()
        payload["records"][0]["provenance_status"] = "vibes"
        with pytest.raises(ValidationError):
            MetricsCatalog.model_validate(payload)


class TestFailClosedReferences:
    def test_duplicate_recipe_id_raises(self):
        with pytest.raises(ValidationError, match="Duplicate recipe_id"):
            _catalog(recipes=[_recipe(), _recipe()])

    def test_record_with_unknown_recipe_raises(self):
        with pytest.raises(ValidationError, match="unknown recipe_id"):
            _catalog(records=[_record(recipe_id="nope.never")])

    def test_comparison_with_unknown_recipe_raises(self):
        with pytest.raises(ValidationError, match="unknown recipe_id"):
            _catalog(comparisons=[_comparison(recipe_id="nope.never")])


class TestLookups:
    def test_get_recipe_found(self):
        catalog = _catalog()
        assert get_catalog_recipe(catalog, "ari.consensus_vs_expert").kind == "measured"

    def test_get_recipe_missing_raises(self):
        with pytest.raises(KeyError):
            get_catalog_recipe(_catalog(), "nope.never")

    def test_filter_by_dataset_and_method(self):
        catalog = _catalog()
        assert len(get_catalog_records(catalog, dataset="korean_forest", method="reap")) == 1
        assert get_catalog_records(catalog, dataset="korean_forest", method="procrustes") == []

    def test_filter_by_mode_and_seed_set(self):
        catalog = _catalog()
        assert len(get_catalog_records(catalog, evaluation_mode="in_sample", seed_set="A")) == 1
        assert get_catalog_records(catalog, evaluation_mode="oos") == []


class TestRoundTrip:
    def test_write_read_round_trip(self, tmp_path: Path):
        catalog = _catalog()
        out = tmp_path / "catalog.json"
        write_metrics_catalog(catalog, out)
        assert read_metrics_catalog(out) == catalog

    def test_write_is_atomic_no_temp_left_behind(self, tmp_path: Path):
        out = tmp_path / "nested" / "catalog.json"
        write_metrics_catalog(_catalog(), out)
        assert out.exists()
        assert list(out.parent.glob("*.tmp")) == []

    def test_read_rejects_tampered_file(self, tmp_path: Path):
        out = tmp_path / "catalog.json"
        write_metrics_catalog(_catalog(), out)
        payload = json.loads(out.read_text())
        payload["records"][0]["recipe_id"] = "dangling.reference"
        out.write_text(json.dumps(payload))
        with pytest.raises(ValidationError):
            read_metrics_catalog(out)
