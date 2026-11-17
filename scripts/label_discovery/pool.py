"""D0: build the passage pool for the corporate/presidential case study.

Both sides are cut into passages by one chunking rule and kept when they
match the broad climate-and-energy keyword list. Corporate passages are kept
only when they are verified word for word against the source PDF (the share
of their word 6-grams found in the PDF text is at least the verification
threshold), because some extractor sections are model paraphrases. Exact
duplicate passages (same text hash, same side) keep only their first
occurrence by id, so boilerplate repeated across documents or years cannot
crowd the sample.

Besides the kept pool, the builder keeps the passages the keyword rule
excluded from documents that have at least one kept passage ("excluded
neighbours"); the discovery sample draws from them to check what the keyword
rule misses.

Exports: ``PoolConfig``, ``PoolPassage``, ``PoolResult``, ``get_pool``,
``get_pool_summary``, ``get_pool_rows``, ``get_passage_from_row``.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from label_discovery.passages import (
    ChunkingRule,
    get_keyword_matches,
    get_paragraphs,
    get_passages,
    get_text_sha256,
    get_verbatim_score,
    get_word_count,
)
from label_discovery.sources import (
    CorporateReport,
    WhiteHouseDocument,
    get_pdf_ngrams,
)

Side = Literal["presidential", "corporate"]


class PoolConfig(BaseModel):
    """Every setting the pool depends on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year_min: int = Field(default=2009, description="First White House year read.")
    year_max: int = Field(default=2025, description="Last White House year read.")
    chunking: ChunkingRule = Field(default_factory=ChunkingRule, description="Chunking rule.")
    verbatim_ngram: int = Field(default=6, ge=2, description="Word n-gram size for verification.")
    verbatim_threshold: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Minimum share of a corporate passage's n-grams found in its PDF. A pilot "
        "showed verbatim sections almost all at 0.8 or above and paraphrases near 0; 0.9 keeps "
        "only clearly verbatim text.",
    )


class PoolPassage(BaseModel):
    """One passage with its id, hash, keyword matches, and source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str = Field(..., description="Stable passage id.")
    side: Side = Field(..., description="Corpus side.")
    doc_id: str = Field(..., description="Document id (White House document or corporate report).")
    passage_index: int = Field(..., ge=0, description="Order of the passage within its document.")
    text: str = Field(..., description="Passage text; paragraphs separated by blank lines.")
    text_sha256: str = Field(..., description="SHA-256 of the text.")
    n_words: int = Field(..., ge=1, description="Word count.")
    keywords: list[str] = Field(..., description="Matched keyword ids (empty when excluded).")
    verbatim_score: float | None = Field(
        default=None, description="Corporate only: n-gram share found in the PDF."
    )
    admin: str | None = Field(default=None, description="Presidential: administration id.")
    year: int = Field(..., description="Document or report year.")
    doc_type: str | None = Field(default=None, description="Presidential: document type.")
    doc_date: str | None = Field(default=None, description="Presidential: ISO date.")
    title: str | None = Field(default=None, description="Presidential: document title.")
    company_slug: str | None = Field(default=None, description="Corporate: company id.")
    company_name: str | None = Field(default=None, description="Corporate: company name.")
    industry: str | None = Field(default=None, description="Corporate: industry.")
    section_title: str | None = Field(default=None, description="Corporate: section title.")
    extraction_model: str | None = Field(default=None, description="Corporate: sectioning model.")


class PoolResult(BaseModel):
    """The kept pool, the excluded neighbours, and the build counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kept: list[PoolPassage] = Field(..., description="Keyword-matched passages, sorted by id.")
    excluded_neighbours: list[PoolPassage] = Field(
        ..., description="Keyword-excluded passages from documents with a kept passage."
    )
    counts: dict[str, int] = Field(..., description="Build counts, flat keys.")


def _get_presidential_passages(
    docs: Iterable[WhiteHouseDocument], rule: ChunkingRule, counts: Counter[str]
) -> list[PoolPassage]:
    """Chunk White House documents and tag keyword matches. Pure apart from counting."""
    passages: list[PoolPassage] = []
    for doc in docs:
        counts["presidential_documents"] += 1
        for index, text in enumerate(get_passages(doc.paragraphs, rule)):
            passages.append(
                PoolPassage(
                    passage_id=f"{doc.doc_id}-{index:03d}",
                    side="presidential",
                    doc_id=doc.doc_id,
                    passage_index=index,
                    text=text,
                    text_sha256=get_text_sha256(text),
                    n_words=get_word_count(text),
                    keywords=get_keyword_matches(text),
                    admin=doc.admin,
                    year=doc.year,
                    doc_type=doc.doc_type,
                    doc_date=doc.doc_date.isoformat(),
                    title=doc.title,
                )
            )
    return passages


def _get_corporate_passages(
    reports: Iterable[CorporateReport], config: PoolConfig, counts: Counter[str]
) -> list[PoolPassage]:
    """Chunk corporate sections, verify each passage against its PDF, tag keywords.

    Side effects: reads each report's PDF.
    """
    passages: list[PoolPassage] = []
    for report in reports:
        counts["corporate_reports"] += 1
        grams = get_pdf_ngrams(report.pdf_path, config.verbatim_ngram)
        report_key = hashlib.sha1(report.record_id.encode("utf-8")).hexdigest()[:10]
        for section in report.sections:
            chunks = get_passages(get_paragraphs(section.section_text), config.chunking)
            for index, text in enumerate(chunks):
                counts["corporate_passages_chunked"] += 1
                score = get_verbatim_score(text, grams, config.verbatim_ngram)
                if score < config.verbatim_threshold:
                    counts[f"corporate_unverified__{report.extraction_model}"] += 1
                    continue
                counts[f"corporate_verified__{report.extraction_model}"] += 1
                passages.append(
                    PoolPassage(
                        passage_id=f"cs-{report_key}-{section.section_index:03d}-{index:02d}",
                        side="corporate",
                        doc_id=f"cs-{report_key}",
                        passage_index=section.section_index * 1000 + index,
                        text=text,
                        text_sha256=get_text_sha256(text),
                        n_words=get_word_count(text),
                        keywords=get_keyword_matches(text),
                        verbatim_score=round(score, 4),
                        year=report.year,
                        company_slug=report.company_slug,
                        company_name=report.company_name,
                        industry=report.industry,
                        section_title=section.section_title,
                        extraction_model=report.extraction_model,
                    )
                )
    return passages


def _get_deduplicated(passages: list[PoolPassage], counts: Counter[str]) -> list[PoolPassage]:
    """Keep the first passage (by id) of every (side, text hash) pair. Pure apart from counting.

    Raises
    ------
    ValueError : If two passages share an id.
    """
    ordered = sorted(passages, key=lambda p: p.passage_id)
    ids = [p.passage_id for p in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("passage ids collide; the id scheme is not unique for these sources")
    seen: set[tuple[str, str]] = set()
    unique: list[PoolPassage] = []
    for passage in ordered:
        key = (passage.side, passage.text_sha256)
        if key in seen:
            counts[f"{passage.side}_duplicates_dropped"] += 1
            continue
        seen.add(key)
        unique.append(passage)
    return unique


def get_pool(
    white_house_docs: Iterable[WhiteHouseDocument],
    corporate_reports: Iterable[CorporateReport],
    config: PoolConfig,
) -> PoolResult:
    """Build the kept pool and the excluded neighbours from both sources.

    Inputs: White House documents, corporate reports, and the pool config.
    Output: a ``PoolResult``. Side effects: reads corporate PDFs.
    """
    counts: Counter[str] = Counter()
    passages = _get_presidential_passages(white_house_docs, config.chunking, counts)
    passages += _get_corporate_passages(corporate_reports, config, counts)
    unique = _get_deduplicated(passages, counts)

    kept = [p for p in unique if p.keywords]
    docs_with_kept = {p.doc_id for p in kept}
    neighbours = [p for p in unique if not p.keywords and p.doc_id in docs_with_kept]

    by_side: dict[str, Counter[str]] = defaultdict(Counter)
    for passage in unique:
        by_side[passage.side]["unique"] += 1
        by_side[passage.side]["kept" if passage.keywords else "excluded"] += 1
    for side, side_counts in by_side.items():
        for key, value in side_counts.items():
            counts[f"{side}_passages_{key}"] = value
    counts["excluded_neighbours"] = len(neighbours)
    return PoolResult(kept=kept, excluded_neighbours=neighbours, counts=dict(sorted(counts.items())))


def get_pool_summary(result: PoolResult) -> dict[str, object]:
    """Per-stratum and per-keyword counts of the kept pool, for the summary file. Pure."""
    kept = result.kept
    presidential = [p for p in kept if p.side == "presidential"]
    corporate = [p for p in kept if p.side == "corporate"]
    industry_companies: dict[str, set[str]] = defaultdict(set)
    for passage in corporate:
        industry_companies[passage.industry or "unknown"].add(passage.company_slug or "")
    return {
        "counts": result.counts,
        "presidential_by_admin": dict(sorted(Counter(p.admin or "" for p in presidential).items())),
        "presidential_documents_by_admin": dict(
            sorted(Counter(a for a, _ in {(p.admin or "", p.doc_id) for p in presidential}).items())
        ),
        "presidential_by_year": dict(sorted(Counter(str(p.year) for p in presidential).items())),
        "presidential_by_doc_type": dict(sorted(Counter(p.doc_type or "" for p in presidential).items())),
        "corporate_by_industry": dict(sorted(Counter(p.industry or "" for p in corporate).items())),
        "corporate_companies_by_industry": {k: len(v) for k, v in sorted(industry_companies.items())},
        "corporate_by_year": dict(sorted(Counter(str(p.year) for p in corporate).items())),
        "keyword_hits_presidential": dict(
            sorted(Counter(k for p in presidential for k in p.keywords).items())
        ),
        "keyword_hits_corporate": dict(sorted(Counter(k for p in corporate for k in p.keywords).items())),
    }


def get_pool_rows(passages: Iterable[PoolPassage]) -> list[dict[str, object]]:
    """Serialize passages for JSONL output. Pure."""
    return [p.model_dump(mode="json") for p in passages]


def get_passage_from_row(row: dict[str, object]) -> PoolPassage:
    """Validate one JSONL row back into a passage; fails closed on any defect. Pure."""
    return PoolPassage.model_validate(row)

