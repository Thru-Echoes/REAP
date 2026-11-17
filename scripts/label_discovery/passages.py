"""Pure text functions for the passage pool: paragraphs, passages, keywords, verbatim checks.

One chunking rule serves both sides of the corpus (White House documents and
corporate report sections): a passage is one to three consecutive paragraphs
and at most ``max_words`` words. Very short paragraphs (headings, captions)
are carried into the following paragraph instead of counting as a paragraph
of their own; a paragraph longer than ``max_words`` is split at sentence
boundaries; passages shorter than ``min_words`` are dropped as fragments.

Exports: ``CHUNKING_RULE_ID``, ``ChunkingRule``, ``KEYWORD_LIST_ID``,
``KEYWORD_PATTERNS``, ``get_paragraphs``, ``get_passages``, ``get_word_count``,
``get_keyword_matches``, ``get_text_sha256``, ``get_normalized_words``,
``get_word_ngrams``, ``get_verbatim_score``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field

CHUNKING_RULE_ID = "cp-chunk-v1"
KEYWORD_LIST_ID = "cp-keywords-v1"

_WORD_RE = re.compile(r"\S+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"'”’)]*\s+")
_LINE_HYPHEN_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-z])")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")


class ChunkingRule(BaseModel):
    """Parameters of the one chunking rule applied to both corpus sides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(default=CHUNKING_RULE_ID, description="Version id of the rule.")
    max_words: int = Field(default=250, ge=20, description="Hard word cap per passage.")
    max_paragraphs: int = Field(default=3, ge=1, description="Paragraph cap per passage.")
    min_words: int = Field(
        default=40,
        ge=1,
        description="Passages shorter than this are fragments (titles, signatures) and dropped.",
    )
    heading_max_words: int = Field(
        default=8,
        ge=0,
        description="A paragraph this short is a heading or caption and is joined to the next one.",
    )


# Broad climate AND energy vocabulary. A passage is kept when ANY pattern
# matches (case-insensitive). Energy terms are included on purpose: White
# House documents from the first Trump term rarely use climate words, so a
# climate-only list would empty that administration's stratum. Generic words
# that saturate sustainability reports ("sustainability", "environment") are
# left out, otherwise the corporate filter would keep nearly every passage.
KEYWORD_PATTERNS: dict[str, str] = {
    "climate": r"\bclimate\b",
    "global_warming": r"\bglobal warming\b",
    "greenhouse": r"\bgreenhouse\b",
    "carbon": r"\bcarbon\b",
    "co2": r"\bco2\b|\bco₂\b",
    "emissions": r"\bemissions?\b",
    "net_zero": r"\bnet[- ]zero\b",
    "decarbonization": r"\bdecarboni[sz]\w*",
    "methane": r"\bmethane\b",
    "climate_treaties": (
        r"\bparis (?:climate )?(?:agreement|accord)\b|\bkyoto\b|\bunfccc\b|\bcop ?2\d\b"
    ),
    "climate_hazards": (
        r"\bextreme weather\b|\bsea[- ]levels?\b|\bwildfires?\b|\bdroughts?\b|\bheat ?waves?\b"
    ),
    "energy": r"\benergy\b",
    "renewables": r"\brenewables?\b",
    "solar": r"\bsolar\b",
    "wind_power": (
        r"\b(?:wind (?:power|energy|turbines?|farms?|projects?)|offshore wind|onshore wind)\b"
    ),
    "oil": r"\boil\b|\bpetroleum\b|\bcrude\b|\bopec\b",
    "gas": (
        r"\bnatural gas\b|\blng\b|\bshale\b|\bfrack\w*|\bhydraulic fracturing\b"
        r"|\bgasoline\b|\bgas prices?\b"
    ),
    "coal": r"\bcoal\b",
    "nuclear_power": r"\bnuclear (?:power|energy|plants?|reactors?|fuel)\b",
    "fossil_fuels": r"\bfossil[- ]fuels?\b",
    "pipelines": r"\b(?:oil|gas|keystone(?: xl)?) pipelines?\b",
    "drilling": r"\bdrilling\b",
    "electricity_grid": (
        r"\belectricity\b|\b(?:power|electric|electrical) grids?\b|\bthe grid\b|\bclean power\b"
    ),
    "electric_vehicles": r"\belectric vehicles?\b|\bevs?\b|\bcharging stations?\b",
    "batteries": r"\bbatter(?:y|ies)\b",
    "hydrogen": r"\bhydrogen\b",
    "biofuels": r"\bbiofuels?\b|\bethanol\b",
    "hydro_geothermal": r"\bhydro(?:power|electric\w*)\b|\bgeothermal\b",
}

_COMPILED_KEYWORDS = {
    key: re.compile(pattern, re.IGNORECASE) for key, pattern in KEYWORD_PATTERNS.items()
}


def get_word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in ``text``. Pure."""
    return len(_WORD_RE.findall(text))


def get_paragraphs(text: str) -> list[str]:
    """Split raw document text into clean paragraphs.

    Paragraphs are separated by blank lines. Inside a paragraph, a hyphen at a
    line break before a lowercase letter is joined ("sus-\\ntainable" becomes
    "sustainable"), and all whitespace runs collapse to one space. Unicode is
    NFKC-normalized so typographic ligatures read as letters. Pure.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_HYPHEN_RE.sub(r"\1\2", text)
    paragraphs = []
    for block in _BLANK_LINE_RE.split(text):
        cleaned = " ".join(block.split())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _get_sentence_pieces(paragraph: str, max_words: int) -> list[str]:
    """Split one over-long paragraph into pieces of at most ``max_words`` words.

    Pieces end at sentence boundaries; a single sentence longer than
    ``max_words`` is cut at the word cap. Pure.
    """
    sentences = [s for s in _SENTENCE_END_RE.split(paragraph) if s.strip()]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _get_units(paragraphs: list[str], rule: ChunkingRule) -> list[str]:
    """Merge headings into the next paragraph and split over-long paragraphs. Pure."""
    units: list[str] = []
    carry = ""
    for paragraph in paragraphs:
        merged = f"{carry} {paragraph}".strip() if carry else paragraph
        if get_word_count(merged) <= rule.heading_max_words:
            carry = merged
            continue
        carry = ""
        if get_word_count(merged) > rule.max_words:
            units.extend(_get_sentence_pieces(merged, rule.max_words))
        else:
            units.append(merged)
    if carry:
        units.append(carry)
    return units


def get_passages(paragraphs: list[str], rule: ChunkingRule) -> list[str]:
    """Pack consecutive paragraphs into passages under one chunking rule.

    Greedy and order-preserving: a passage collects up to
    ``rule.max_paragraphs`` units while staying within ``rule.max_words``;
    passages under ``rule.min_words`` are dropped. Pure.
    """
    passages: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in _get_units(paragraphs, rule):
        unit_words = get_word_count(unit)
        if current and (
            len(current) >= rule.max_paragraphs or current_words + unit_words > rule.max_words
        ):
            passages.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(unit)
        current_words += unit_words
    if current:
        passages.append("\n\n".join(current))
    return [p for p in passages if get_word_count(p) >= rule.min_words]


def get_keyword_matches(text: str) -> list[str]:
    """Return the sorted ids of every keyword pattern that matches ``text``. Pure."""
    return sorted(key for key, pattern in _COMPILED_KEYWORDS.items() if pattern.search(text))


def get_text_sha256(text: str) -> str:
    """SHA-256 of the UTF-8 bytes of ``text``. Pure."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_normalized_words(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens after NFKC and line-break de-hyphenation. Pure."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _LINE_HYPHEN_RE.sub(r"\1\2", text)
    return re.findall(r"[a-z0-9]+", text)


def get_word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of word ``n``-grams of ``words``. Pure."""
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def get_verbatim_score(passage: str, source_ngrams: set[tuple[str, ...]], n: int) -> float:
    """Share of a passage's word ``n``-grams that occur in the source text.

    1.0 means every ``n``-word run of the passage appears in the source; a
    paraphrase scores near 0. Robust to line wrapping, column order, and
    ligatures, because both sides are normalized the same way. Pure.

    Raises
    ------
    ValueError : If the passage has fewer than ``n`` words.
    """
    grams = get_word_ngrams(get_normalized_words(passage), n)
    if not grams:
        raise ValueError(f"passage has fewer than {n} words; cannot score verbatim overlap")
    return len(grams & source_ngrams) / len(grams)
