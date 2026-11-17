"""Descriptive statistics of the open-coding passes (D2 summary).

Pure functions over coded passages: per-pass distributions of stance and
climate/energy scope by corpus side, quote faithfulness, agreement between
two passes on the passages both coded (raw agreement and Cohen's kappa for
stance and scope, mean label-word overlap), and the keyword-rule check (how
often keyword-excluded passages are judged really about climate or energy,
and how often kept passages are judged not to be).

Exports: ``get_cohen_kappa``, ``get_label_words``, ``get_pass_summary``,
``get_pair_agreement``, ``get_keyword_rule_check``.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Any

_STOP = frozenset(
    "a an and as at by for from in into of on or the to with its their our us we".split()
)


def get_cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters' labels on the same items; None when undefined. Pure.

    Undefined when there are no items or when chance agreement is 1 (both
    raters used one identical label for everything).

    Raises
    ------
    ValueError : If the two lists differ in length.
    """
    if len(a) != len(b):
        raise ValueError("rater lists differ in length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in set(count_a) | set(count_b)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def get_label_words(labels: list[str]) -> set[str]:
    """Content words of a passage's labels, lowercased, stop words removed. Pure."""
    words = set(re.findall(r"[a-z0-9]+", " ".join(labels).lower()))
    return words - _STOP


def _get_share_table(rows: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Share of each value within each group, from (group, value) pairs. Pure."""
    groups = Counter(g for g, _ in rows)
    pairs = Counter(rows)
    return {
        group: {
            value: round(pairs[(group, value)] / total, 4)
            for (g, value) in sorted(pairs)
            if g == group
        }
        for group, total in sorted(groups.items())
    }


def get_pass_summary(
    codes: list[dict[str, Any]], sample: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Distributions and checks for one pass. Pure.

    ``codes`` are code rows (``passage_id``, ``stance``, ``topic_scope``,
    ``labels``, ``quote_verbatim``); ``sample`` maps passage id to its sample
    row (``passage.side``, ``keyword_excluded``).
    """

    def group_of(pid: str) -> str:
        row = sample[pid]
        return f"{row['passage']['side']}{'_excluded' if row['keyword_excluded'] else '_kept'}"

    excluded = [c for c in codes if sample[c["passage_id"]]["keyword_excluded"]]
    kept = [c for c in codes if not sample[c["passage_id"]]["keyword_excluded"]]
    return {
        "n_codes": len(codes),
        "quote_verbatim_rate": round(mean(c["quote_verbatim"] for c in codes), 4)
        if codes
        else None,
        "mean_labels_per_passage": round(mean(len(c["labels"]) for c in codes), 3)
        if codes
        else None,
        "stance_by_group": _get_share_table(
            [(group_of(c["passage_id"]), c["stance"]) for c in codes]
        ),
        "scope_by_group": _get_share_table(
            [(group_of(c["passage_id"]), c["topic_scope"]) for c in codes]
        ),
        "keyword_check": {
            "excluded_judged_climate_or_energy": round(
                mean(c["topic_scope"] != "neither" for c in excluded), 4
            )
            if excluded
            else None,
            "kept_judged_neither": round(mean(c["topic_scope"] == "neither" for c in kept), 4)
            if kept
            else None,
            "n_excluded": len(excluded),
            "n_kept": len(kept),
        },
    }


def get_pair_agreement(
    codes_a: list[dict[str, Any]], codes_b: list[dict[str, Any]], ids: set[str] | None = None
) -> dict[str, Any]:
    """Agreement between two passes on the passages both coded (optionally only ``ids``). Pure."""
    by_a = {c["passage_id"]: c for c in codes_a}
    by_b = {c["passage_id"]: c for c in codes_b}
    common = sorted(set(by_a) & set(by_b) & (ids if ids is not None else set(by_a)))
    if not common:
        return {"n": 0}
    stance_a = [by_a[p]["stance"] for p in common]
    stance_b = [by_b[p]["stance"] for p in common]
    scope_a = [by_a[p]["topic_scope"] for p in common]
    scope_b = [by_b[p]["topic_scope"] for p in common]
    overlaps = []
    for pid in common:
        words_a, words_b = (
            get_label_words(by_a[pid]["labels"]),
            get_label_words(by_b[pid]["labels"]),
        )
        union = words_a | words_b
        overlaps.append(len(words_a & words_b) / len(union) if union else 1.0)
    kappa_stance = get_cohen_kappa(stance_a, stance_b)
    kappa_scope = get_cohen_kappa(scope_a, scope_b)
    return {
        "n": len(common),
        "stance_agreement": round(mean(x == y for x, y in zip(stance_a, stance_b)), 4),
        "stance_kappa": None if kappa_stance is None else round(kappa_stance, 4),
        "scope_agreement": round(mean(x == y for x, y in zip(scope_a, scope_b)), 4),
        "scope_kappa": None if kappa_scope is None else round(kappa_scope, 4),
        "label_word_jaccard_mean": round(mean(overlaps), 4),
    }


_CLIMATE_KEYWORDS = frozenset(
    {
        "climate",
        "global_warming",
        "greenhouse",
        "carbon",
        "co2",
        "emissions",
        "net_zero",
        "decarbonization",
        "methane",
        "climate_treaties",
        "climate_hazards",
    }
)


def get_keyword_rule_check(
    codes_a: list[dict[str, Any]],
    codes_b: list[dict[str, Any]],
    sample: dict[str, dict[str, Any]],
    pool: list[dict[str, Any]],
) -> dict[str, Any]:
    """How often keyword-kept passages are off-topic, by keyword signature, plus pool sizes. Pure.

    A kept sample passage counts as off-topic when BOTH passes judged its
    scope 'neither'. Signatures: one keyword id versus two or more; the bare
    word 'energy' as the only match; any climate-family id. ``pool`` rows are
    the kept pool, used to count how many passages each stratum would keep
    under a two-or-more-ids rule.
    """
    by_a = {c["passage_id"]: c for c in codes_a}
    by_b = {c["passage_id"]: c for c in codes_b}
    rows: dict[str, list[bool]] = {}
    relevant: dict[str, int] = {}
    for pid, row in sorted(sample.items()):
        if row["keyword_excluded"] or pid not in by_a or pid not in by_b:
            continue
        passage = row["passage"]
        keywords = set(passage["keywords"])
        off_topic = by_a[pid]["topic_scope"] == "neither" and by_b[pid]["topic_scope"] == "neither"
        side = passage["side"]
        signatures = ["one_id" if len(keywords) == 1 else "two_or_more_ids"]
        if keywords == {"energy"}:
            signatures.append("energy_only")
        signatures.append("has_climate_id" if keywords & _CLIMATE_KEYWORDS else "energy_ids_only")
        for signature in signatures:
            rows.setdefault(f"{side}__{signature}", []).append(off_topic)
        stratum = passage["admin"] if side == "presidential" else "corporate"
        relevant[stratum] = relevant.get(stratum, 0) + (not off_topic)
    pool_counts: dict[str, dict[str, int]] = {}
    for row in pool:
        stratum = row["admin"] if row["side"] == "presidential" else "corporate"
        counts = pool_counts.setdefault(stratum, {"kept": 0, "two_or_more_ids": 0})
        counts["kept"] += 1
        counts["two_or_more_ids"] += len(row["keywords"]) >= 2
    return {
        "off_topic_share_by_signature": {
            key: {"n": len(values), "off_topic_share": round(mean(values), 4)}
            for key, values in sorted(rows.items())
        },
        "relevant_kept_passages_by_stratum": dict(sorted(relevant.items())),
        "pool_counts_by_stratum": dict(sorted(pool_counts.items())),
        "rule": "off-topic = both passes judged topic_scope 'neither'",
    }
