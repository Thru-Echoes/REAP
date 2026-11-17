"""D1: draw the stratified discovery sample and split it into halves A and B.

Strata: the four administrations (presidential), the largest corporate
industries that have at least a full quota of kept passages (ranked by the
number of companies with a kept passage, then by kept passages), and one
stratum per side of keyword-excluded passages drawn from documents that also
have a kept passage. When fewer industries than asked for have a full quota,
the remaining industries are pooled into one "other" stratum that takes the
missing quotas, so the corporate side keeps its planned size. Within a stratum the draw is
round-robin over documents (presidential) or companies (corporate), in
random order, so no single document or company can dominate a stratum.

Each stratum is split evenly between halves A and B; when a stratum has an
odd size the extra passage alternates between halves across strata so the
two halves stay the same size. Two subsets are marked inside the sample: a
repeat set (coded twice by the same model to measure repeat-run variation)
and an inline subset (coded by a Claude model inside an interactive session,
which cannot code every passage).

Every passage drawn here is excluded from all later evaluation samples.

Exports: ``SampleConfig``, ``SampledPassage``, ``get_ranked_industries``,
``get_discovery_sample``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from label_discovery.pool import PoolPassage

ADMIN_ORDER = ("obama", "trump_1", "biden", "trump_2")


class SampleConfig(BaseModel):
    """Every setting the discovery sample depends on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=20261117, description="Sampling seed (the date of the draw).")
    presidential_per_admin: int = Field(
        default=50, ge=2, description="Passages per administration."
    )
    corporate_per_industry: int = Field(default=25, ge=2, description="Passages per industry.")
    n_industries: int = Field(default=8, ge=1, description="Number of largest industries sampled.")
    excluded_per_side: int = Field(
        default=25, ge=0, description="Keyword-excluded passages per side."
    )
    repeat_per_admin: int = Field(
        default=6, ge=0, description="Repeat-set passages per administration."
    )
    repeat_per_industry: int = Field(
        default=3, ge=0, description="Repeat-set passages per industry."
    )
    repeat_excluded_per_side: int = Field(
        default=1, ge=0, description="Repeat-set excluded passages per side."
    )
    inline_per_half_admin: int = Field(
        default=7, ge=0, description="Inline subset, per half and administration."
    )
    inline_per_half_industry: int = Field(
        default=3, ge=0, description="Inline subset, per half and industry."
    )
    inline_per_half_excluded: int = Field(
        default=4, ge=0, description="Inline subset, per half and side, of excluded passages."
    )


class SampledPassage(BaseModel):
    """One sampled passage with its stratum, half, and subset flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage: PoolPassage = Field(..., description="The pool passage.")
    stratum: str = Field(..., description="Stratum id, e.g. 'admin:obama' or 'industry:Energy'.")
    half: str = Field(..., pattern=r"^[AB]$", description="Discovery half.")
    keyword_excluded: bool = Field(..., description="True for keyword-excluded passages.")
    in_repeat_set: bool = Field(..., description="Coded twice by the same model.")
    in_inline_subset: bool = Field(..., description="Also coded inline by a Claude model.")


def get_ranked_industries(kept: list[PoolPassage]) -> list[tuple[str, int, int]]:
    """Rank corporate industries by companies, then passages, then name.

    Output: ``(industry, n_companies, n_passages)`` tuples, largest first;
    the 'unknown' industry is never ranked. Pure.
    """
    companies: dict[str, set[str]] = defaultdict(set)
    passages: Counter[str] = Counter()
    for passage in kept:
        if passage.side != "corporate" or not passage.industry or passage.industry == "unknown":
            continue
        companies[passage.industry].add(passage.company_slug or "")
        passages[passage.industry] += 1
    ranked = [(name, len(companies[name]), passages[name]) for name in companies]
    return sorted(ranked, key=lambda row: (-row[1], -row[2], row[0]))


def _get_group_key(passage: PoolPassage) -> str:
    """Spread unit: the document for presidential passages, the company for corporate. Pure."""
    return (
        passage.doc_id
        if passage.side == "presidential"
        else (passage.company_slug or passage.doc_id)
    )


def _get_round_robin(
    candidates: list[PoolPassage], n: int, rng: np.random.Generator, stratum: str
) -> list[PoolPassage]:
    """Draw ``n`` passages round-robin over spread units in random order.

    Side effects: advances ``rng``.

    Raises
    ------
    ValueError : If the stratum has fewer than ``n`` candidates.
    """
    if len(candidates) < n:
        raise ValueError(f"stratum {stratum} has {len(candidates)} candidates, needs {n}")
    groups: dict[str, list[PoolPassage]] = defaultdict(list)
    for passage in sorted(candidates, key=lambda p: p.passage_id):
        groups[_get_group_key(passage)].append(passage)
    keys = sorted(groups)
    order = [keys[i] for i in rng.permutation(len(keys))]
    queues = {key: [groups[key][i] for i in rng.permutation(len(groups[key]))] for key in order}
    drawn: list[PoolPassage] = []
    while len(drawn) < n:
        for key in order:
            if queues[key] and len(drawn) < n:
                drawn.append(queues[key].pop())
    return drawn


def get_discovery_sample(
    kept: list[PoolPassage], neighbours: list[PoolPassage], config: SampleConfig
) -> tuple[list[SampledPassage], list[tuple[str, int, int]]]:
    """Draw the discovery sample.

    Inputs: the kept pool, the excluded neighbours, and the config.
    Output: sampled passages (sorted by half, stratum, id) and the industry
    ranking used. Pure apart from its own seeded generator.

    Raises
    ------
    ValueError : If any stratum is too small for its quota.
    """
    rng = np.random.default_rng(config.seed)
    ranking = get_ranked_industries(kept)
    eligible = [
        name for name, _, n_passages in ranking if n_passages >= config.corporate_per_industry
    ]
    industries = eligible[: config.n_industries]
    n_short = config.n_industries - len(industries)

    strata: list[tuple[str, list[PoolPassage], int, int, int, bool]] = []
    for admin in ADMIN_ORDER:
        pool = [p for p in kept if p.side == "presidential" and p.admin == admin]
        strata.append(
            (
                f"admin:{admin}",
                pool,
                config.presidential_per_admin,
                config.repeat_per_admin,
                config.inline_per_half_admin,
                False,
            )
        )
    for industry in industries:
        pool = [p for p in kept if p.side == "corporate" and p.industry == industry]
        strata.append(
            (
                f"industry:{industry}",
                pool,
                config.corporate_per_industry,
                config.repeat_per_industry,
                config.inline_per_half_industry,
                False,
            )
        )
    if n_short:
        # Fewer industries than asked for have a full quota of passages: the
        # remaining industries are pooled into one stratum that takes the
        # missing quotas, so the corporate side keeps its planned size.
        others = {name for name, _, _ in ranking} - set(industries)
        pool = [p for p in kept if p.side == "corporate" and p.industry in others]
        strata.append(
            (
                "industry:other",
                pool,
                config.corporate_per_industry * n_short,
                config.repeat_per_industry * n_short,
                config.inline_per_half_industry * n_short,
                False,
            )
        )
    for side in ("presidential", "corporate"):
        pool = [p for p in neighbours if p.side == side]
        strata.append(
            (
                f"excluded:{side}",
                pool,
                config.excluded_per_side,
                config.repeat_excluded_per_side,
                config.inline_per_half_excluded,
                True,
            )
        )

    sampled: list[SampledPassage] = []
    odd_index = 0
    for stratum, pool, n, n_repeat, n_inline_half, excluded in strata:
        drawn = _get_round_robin(pool, n, rng, stratum)
        first_half = "A" if odd_index % 2 == 0 else "B"
        if n % 2 == 1:
            odd_index += 1
        second_half = "B" if first_half == "A" else "A"
        halves = [first_half if i % 2 == 0 else second_half for i in range(n)]
        repeat_ids = {drawn[i].passage_id for i in rng.permutation(n)[:n_repeat]}
        inline_ids: set[str] = set()
        for half in ("A", "B"):
            members = [p for p, h in zip(drawn, halves) if h == half]
            if len(members) < n_inline_half:
                raise ValueError(f"stratum {stratum} half {half} too small for the inline subset")
            inline_ids |= {
                members[i].passage_id for i in rng.permutation(len(members))[:n_inline_half]
            }
        for passage, half in zip(drawn, halves):
            sampled.append(
                SampledPassage(
                    passage=passage,
                    stratum=stratum,
                    half=half,
                    keyword_excluded=excluded,
                    in_repeat_set=passage.passage_id in repeat_ids,
                    in_inline_subset=passage.passage_id in inline_ids,
                )
            )
    ids = [s.passage.passage_id for s in sampled]
    if len(set(ids)) != len(ids):
        raise ValueError("a passage was drawn twice")
    return sorted(sampled, key=lambda s: (s.half, s.stratum, s.passage.passage_id)), ranking


def get_sample_row(sampled: SampledPassage) -> dict[str, object]:
    """Flat table row for ``sample.csv``. Pure."""
    p = sampled.passage
    return {
        "passage_id": p.passage_id,
        "side": p.side,
        "stratum": sampled.stratum,
        "half": sampled.half,
        "keyword_excluded": sampled.keyword_excluded,
        "in_repeat_set": sampled.in_repeat_set,
        "in_inline_subset": sampled.in_inline_subset,
        "doc_id": p.doc_id,
        "admin": p.admin or "",
        "year": p.year,
        "doc_type": p.doc_type or "",
        "company_slug": p.company_slug or "",
        "industry": p.industry or "",
        "n_words": p.n_words,
        "text_sha256": p.text_sha256,
    }
