"""Tests for label-discovery codebook induction (D3), crosswalk (D4), and coding statistics.

The OpenAI client is replaced by a fake that returns scripted JSON, the one
external service in these steps; validation, statistics, and classification
run for real. The live test at the bottom checks the committed codebooks and
crosswalk in the REAP-research checkout and skips, with a reason, when they
are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from label_discovery.codebooks import (  # noqa: E402
    Codebook,
    get_category_shares,
    get_codebook_errors,
    get_induction_format,
    run_induction,
)
from label_discovery.crosswalk import (  # noqa: E402
    DEDUCTIVE_FRAMES,
    ConceptGroup,
    get_alignment_errors,
    get_group_shares_by_stratum,  # noqa: E402
    get_group_stats,
    get_model_family,
)
from label_discovery.stats import (  # noqa: E402
    get_cohen_kappa,
    get_keyword_rule_check,
    get_pair_agreement,
)

IDS = [f"p{i:02d}" for i in range(24)]


def _codebook_payload(n_categories: int = 8, drop: str | None = None) -> dict[str, Any]:
    names = [f"Cat {k}" for k in range(n_categories)]
    assignments = [{"id": pid, "category": names[i % n_categories]} for i, pid in enumerate(IDS)]
    if drop:
        assignments = [a for a in assignments if a["id"] != drop]
    categories = []
    for k, name in enumerate(names):
        members = [pid for i, pid in enumerate(IDS) if i % n_categories == k]
        while len(members) < 3:
            members.append(members[0])
        categories.append(
            {
                "name": name,
                "definition": "d",
                "include_rule": "i",
                "exclude_rule": "e",
                "example_ids": members[:3],
                "family": "F1" if k < 4 else "F2",
            }
        )
    return {
        "categories": categories,
        "families": [{"name": "F1", "definition": "f"}, {"name": "F2", "definition": "f"}],
        "assignments": assignments,
        "spans_multiple_dimensions": False,
        "dimensions": [{"dimension": "subject", "categories": names}],
        "dimension_explanation": "one",
    }


def _response(content: str) -> Any:
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=1,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
    )
    return SimpleNamespace(
        id="r",
        model="fake",
        system_fingerprint=None,
        usage=usage,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
    )


def test_induction_retries_until_the_codebook_is_valid() -> None:
    replies = iter(
        [
            json.dumps(_codebook_payload(n_categories=4)),
            "not json",
            json.dumps(_codebook_payload(8)),
        ]
    )
    items = [{"id": pid, "labels": ["x"], "care": "c"} for pid in IDS]
    codebook, logs = run_induction(
        items, "t/A", "terra_x", "A", lambda **_: _response(next(replies)), "m", "high"
    )
    assert codebook is not None and len(codebook.categories) == 8
    assert [bool(e["validation_errors"]) for e in logs] == [True, True, False]


def test_a_category_named_other_becomes_the_residual_bin() -> None:
    from label_discovery.codebooks import get_codebook_from_content

    payload = _codebook_payload(8)
    payload["categories"].append(dict(payload["categories"][0], name="other"))
    payload["assignments"][0]["category"] = "OTHER"
    codebook = get_codebook_from_content(json.dumps(payload), "t/A", "terra_x", "A")
    assert codebook.residual_category_dropped and len(codebook.categories) == 8
    assert codebook.assignments["p00"] == "Other"
    errors = get_codebook_errors(codebook, IDS)
    assert errors == ["example p00 of 'Cat 0' is not assigned to it"]


def test_induction_fails_closed_after_all_attempts() -> None:
    items = [{"id": pid, "labels": ["x"], "care": "c"} for pid in IDS]
    codebook, logs = run_induction(
        items, "t/A", "terra_x", "A", lambda **_: _response("not json"), "m", "high"
    )
    assert codebook is None and len(logs) == 3


def test_codebook_validation_catches_each_rule() -> None:
    from label_discovery.codebooks import get_codebook_from_content

    payload = _codebook_payload(n_categories=5)  # 4 members each, so 3 distinct examples
    codebook = get_codebook_from_content(json.dumps(payload), "t/A", "terra_x", "A")
    assert get_codebook_errors(codebook, IDS) == ["5 categories, need 8-15"]
    dropped = get_codebook_from_content(
        json.dumps(_codebook_payload(5, drop="p03")), "t/A", "terra_x", "A"
    )
    assert any("unassigned" in e for e in get_codebook_errors(dropped, IDS))
    assert get_induction_format()["json_schema"]["strict"] is True


def _book(
    codebook_id: str, source_pass: str, half: str, names: list[str], ids: list[str]
) -> Codebook:
    return Codebook(
        codebook_id=codebook_id,
        source_pass=source_pass,
        half=half,
        categories=[
            {
                "name": n,
                "definition": "d",
                "include_rule": "i",
                "exclude_rule": "e",  # type: ignore[list-item]
                "example_ids": ["a", "b", "c"],
                "family": "F",
            }
            for n in names
        ],
        families=[{"name": "F", "definition": "f"}],  # type: ignore[list-item]
        assignments={pid: names[i % len(names)] for i, pid in enumerate(ids)},
        spans_multiple_dimensions=False,
        dimensions={},
        dimension_explanation="",
    )


def test_crosswalk_classifies_stability_and_shares() -> None:
    ids_a, ids_b = ["wh-1", "cs-1"], ["wh-2", "cs-2"]
    sides = {
        "wh-1": "presidential",
        "wh-2": "presidential",
        "cs-1": "corporate",
        "cs-2": "corporate",
    }
    books = [
        _book("terra_straight_1/A", "terra_straight_1", "A", ["Targets", "Prices"], ids_a),
        _book("terra_straight_1/B", "terra_straight_1", "B", ["Targets", "Prices"], ids_b),
        _book("sol_straight/A", "sol_straight", "A", ["Goals", "Other energy"], ids_a),
    ]
    groups = [
        ConceptGroup(
            group_id="G1",
            name="Targets",
            definition="d",
            suggested_family="F",
            members=[
                "terra_straight_1/A::Targets",
                "terra_straight_1/B::Targets",
                "sol_straight/A::Goals",
            ],
            deductive_frames=["targets_accounting"],
        ),
        ConceptGroup(
            group_id="G2",
            name="Prices",
            definition="d",
            suggested_family="F",
            members=["terra_straight_1/A::Prices", "terra_straight_1/B::Prices"],
            deductive_frames=[],
        ),
        ConceptGroup(
            group_id="G3",
            name="Energy",
            definition="d",
            suggested_family="F",
            members=["sol_straight/A::Other energy"],
            deductive_frames=["energy_supply_security"],
        ),
    ]
    assert get_alignment_errors(groups, books) == []
    assert get_alignment_errors(groups[:2], books) == [
        "1 categories in no group, e.g. ['sol_straight/A::Other energy']"
    ]
    kinds = {"terra_straight_1": "straightforward", "sol_straight": "straightforward"}
    stats, deductive_only = get_group_stats(groups, books, sides, kinds)
    by_id = {s.group_id: s for s in stats}
    assert by_id["G1"].stability == "stable" and by_id["G1"].models == ["sol", "terra"]
    assert by_id["G2"].stability == "model-specific" and by_id["G2"].emergent
    assert by_id["G3"].stability == "sample-specific"
    assert by_id["G1"].presidential_share_mean == 1.0 and by_id["G1"].corporate_share_mean == 0.0
    assert set(deductive_only) == set(DEDUCTIVE_FRAMES) - {
        "targets_accounting",
        "energy_supply_security",
    }
    shares = get_category_shares(books[0], sides)
    assert shares["Prices"] == {"presidential": 0.0, "corporate": 1.0}
    assert get_model_family("claude_inline_straight") == "claude"
    with pytest.raises(ValueError):
        get_model_family("unknown_pass")


def test_group_shares_by_stratum() -> None:
    ids = ["wh-1", "wh-2", "cs-1", "cs-2"]
    book = _book("terra_straight_1/A", "terra_straight_1", "A", ["Targets", "Prices"], ids)
    group = ConceptGroup(
        group_id="G1",
        name="Targets",
        definition="d",
        suggested_family="F",
        members=["terra_straight_1/A::Targets"],
        deductive_frames=[],
    )
    strata = {"wh-1": "obama", "wh-2": "obama", "cs-1": "corporate", "cs-2": "corporate"}
    # Assignments alternate Targets, Prices over the ids in order.
    assert get_group_shares_by_stratum([group], [book], strata) == {
        "G1": {"corporate": 0.5, "obama": 0.5}
    }


def test_keyword_rule_check_counts_signatures_and_pool() -> None:
    def row(
        pid: str, side: str, keywords: list[str], admin: str | None = "obama"
    ) -> dict[str, Any]:
        return {
            "keyword_excluded": False,
            "passage": {"passage_id": pid, "side": side, "keywords": keywords, "admin": admin},
        }

    sample = {
        "a": row("a", "presidential", ["energy"]),
        "b": row("b", "presidential", ["climate", "energy"]),
        "c": row("c", "corporate", ["carbon"], None),
    }
    neither = {"stance": "none", "labels": ["x"]}
    codes = [
        {"passage_id": "a", "topic_scope": "neither", **neither},
        {"passage_id": "b", "topic_scope": "climate", **neither},
        {"passage_id": "c", "topic_scope": "neither", **neither},
    ]
    pool = [r["passage"] for r in sample.values()]
    result = get_keyword_rule_check(codes, codes, sample, pool)
    shares = result["off_topic_share_by_signature"]
    assert shares["presidential__energy_only"] == {"n": 1, "off_topic_share": 1.0}
    assert shares["presidential__two_or_more_ids"] == {"n": 1, "off_topic_share": 0.0}
    assert result["relevant_kept_passages_by_stratum"] == {"corporate": 0, "obama": 1}
    assert result["pool_counts_by_stratum"]["obama"] == {"kept": 2, "two_or_more_ids": 1}


def test_kappa_and_pair_agreement() -> None:
    assert get_cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert get_cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert get_cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == -1.0
    a = [
        {
            "passage_id": "p1",
            "stance": "supports",
            "topic_scope": "climate",
            "labels": ["Net zero target"],
        },
        {
            "passage_id": "p2",
            "stance": "none",
            "topic_scope": "neither",
            "labels": ["Campaign rally"],
        },
    ]
    b = [
        {
            "passage_id": "p1",
            "stance": "supports",
            "topic_scope": "both",
            "labels": ["Net zero goal"],
        },
        {
            "passage_id": "p2",
            "stance": "none",
            "topic_scope": "neither",
            "labels": ["Campaign rally"],
        },
    ]
    result = get_pair_agreement(a, b)
    assert (
        result["n"] == 2 and result["stance_agreement"] == 1.0 and result["scope_agreement"] == 0.5
    )
    assert result["label_word_jaccard_mean"] == pytest.approx((0.5 + 1.0) / 2)


def test_live_codebooks_and_crosswalk_are_valid() -> None:
    try:
        from research_root import get_research_root  # pyright: ignore[reportMissingImports]

        root = get_research_root(None) / "results" / "corporate_presidential" / "label_discovery"
    except RuntimeError as exc:
        pytest.skip(f"REAP-research checkout not available: {exc}")
    if not (root / "d4_crosswalk" / "bundle.json").is_file():
        pytest.skip("crosswalk artifacts not present in the REAP-research checkout")
    sample = [
        json.loads(line)
        for line in (root / "d1_discovery_sample" / "sample_passages.jsonl")
        .read_text()
        .splitlines()
    ]
    books = [
        Codebook.model_validate_json(p.read_text())
        for p in sorted((root / "d3_codebooks").glob("*/*/codebook.json"))
    ]
    assert books
    for book in books:
        coded = {
            json.loads(line)["passage_id"]
            for line in (root / "d2_open_coding" / book.source_pass / "codes.jsonl")
            .read_text()
            .splitlines()
        }
        ids = [
            r["passage"]["passage_id"]
            for r in sample
            if r["half"] == book.half and r["passage"]["passage_id"] in coded
        ]
        assert get_codebook_errors(book, ids) == [], book.codebook_id
    alignment = json.loads((root / "d4_crosswalk" / "alignment.json").read_text())
    groups = [ConceptGroup.model_validate(g) for g in alignment["groups"]]
    assert get_alignment_errors(groups, books) == []
