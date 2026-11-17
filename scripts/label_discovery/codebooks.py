"""D3: group one coding pass's free labels into a candidate codebook for one half.

The inducing model sees, for every passage of the half, only the coder's free
labels and its one-sentence "what to care about" summary: never the passage
source, the passage text, or any REAP output. It returns 8 to 15 categories
(name, definition, include rule, exclude rule, three example passages, and a
broader family), assigns every passage to one category or to "Other", and
says whether the categories span more than one dimension (for example
subject matter, framing, stance, or kind of text).

A codebook is validated before it is accepted: the category count is in
range, names are unique, every passage is assigned exactly once to a known
category, examples are passages of the half assigned to their category, and
every family is declared. An invalid response is retried; a step that cannot
produce a valid codebook fails.

Exports: ``INDUCTION_SYSTEM_PROMPT``, ``Category``, ``Family``, ``Codebook``,
``get_induction_prompt``, ``get_induction_format``, ``get_codebook_from_content``,
``get_codebook_errors``, ``run_induction``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

logger = logging.getLogger(__name__)

OTHER = "Other"
MIN_CATEGORIES = 8
MAX_CATEGORIES = 15

INDUCTION_SYSTEM_PROMPT = (
    "You are a qualitative methodologist turning open codes into a codebook for a research "
    "study of how the White House and companies write about climate change and energy. "
    "Work only from the codes you are given. Build categories that a trained human coder "
    "could apply reliably: each with a clear definition, a rule for what belongs, and a rule "
    "that separates it from its nearest neighbour."
)


class Category(BaseModel):
    """One codebook category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Short category name.")
    definition: str = Field(..., min_length=1, description="One or two sentence definition.")
    include_rule: str = Field(..., min_length=1, description="What belongs in the category.")
    exclude_rule: str = Field(
        ..., min_length=1, description="What does not, and where it goes instead."
    )
    example_ids: list[str] = Field(
        ..., min_length=3, max_length=3, description="Three best-fit passages."
    )
    family: str = Field(..., min_length=1, description="Broader family the category belongs to.")


class Family(BaseModel):
    """A broader grouping of categories (the upper level of the hierarchy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Family name.")
    definition: str = Field(..., min_length=1, description="One-sentence definition.")


class Codebook(BaseModel):
    """A candidate codebook induced from one pass's labels on one half."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    codebook_id: str = Field(..., description="'<pass id>/<half>'.")
    source_pass: str = Field(..., description="Coding pass whose labels were grouped.")
    half: str = Field(..., pattern=r"^[AB]$", description="Discovery half.")
    categories: list[Category] = Field(..., description="Categories in the model's order.")
    families: list[Family] = Field(..., description="Families (upper level).")
    assignments: dict[str, str] = Field(..., description="Passage id to category name or 'Other'.")
    spans_multiple_dimensions: bool = Field(..., description="Whether categories mix dimensions.")
    dimensions: dict[str, list[str]] = Field(..., description="Dimension name to its categories.")
    dimension_explanation: str = Field(
        ..., description="Why the categories do or do not mix dimensions."
    )


def get_induction_prompt(items: list[dict[str, Any]]) -> str:
    """User message listing every passage's codes. Pure.

    ``items`` rows carry ``id``, ``labels`` (list of strings) and ``care``.
    """
    lines = [
        f"Below are open codes for {len(items)} passages. Each line gives a passage id, the "
        "coder's free labels (most important first), and a one-sentence summary of what the "
        "passage asks its reader to care about. Some passages are not really about climate "
        "or energy.",
        "",
        f"1. Group the labels into between {MIN_CATEGORIES} and {MAX_CATEGORIES} categories "
        "that together cover the passages. Prefer categories a human coder could tell apart "
        "reliably. For each category give a short name (two to five words), a definition, an "
        "include rule, an exclude rule that separates it from its nearest neighbour, three "
        "example passage ids that fit it best, and the family it belongs to.",
        "2. Group the categories into three to six broader families, each with a one-sentence "
        "definition.",
        "3. Assign every passage to exactly one category, its best fit (weigh the first label "
        f'most), or to "{OTHER}" when no category fits. Every id below must appear exactly once.',
        "4. Say whether your categories describe more than one dimension (for example subject "
        "matter, framing or justification, stance, or kind of text), name the dimensions, and "
        "list which categories belong to each.",
        "",
        "Codes:",
    ]
    for item in items:
        labels = " | ".join(item["labels"])
        lines.append(f"{item['id']} :: {labels} :: {item['care']}")
    return "\n".join(lines)


def get_induction_format() -> dict[str, Any]:
    """Strict JSON-schema response format for an induced codebook. Pure."""
    string = {"type": "string"}
    category = {
        "type": "object",
        "properties": {
            "name": string,
            "definition": string,
            "include_rule": string,
            "exclude_rule": string,
            "example_ids": {"type": "array", "items": string},
            "family": string,
        },
        "required": ["name", "definition", "include_rule", "exclude_rule", "example_ids", "family"],
        "additionalProperties": False,
    }
    family = {
        "type": "object",
        "properties": {"name": string, "definition": string},
        "required": ["name", "definition"],
        "additionalProperties": False,
    }
    assignment = {
        "type": "object",
        "properties": {"id": string, "category": string},
        "required": ["id", "category"],
        "additionalProperties": False,
    }
    dimension = {
        "type": "object",
        "properties": {"dimension": string, "categories": {"type": "array", "items": string}},
        "required": ["dimension", "categories"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "induced_codebook",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "categories": {"type": "array", "items": category},
                    "families": {"type": "array", "items": family},
                    "assignments": {"type": "array", "items": assignment},
                    "spans_multiple_dimensions": {"type": "boolean"},
                    "dimensions": {"type": "array", "items": dimension},
                    "dimension_explanation": string,
                },
                "required": [
                    "categories",
                    "families",
                    "assignments",
                    "spans_multiple_dimensions",
                    "dimensions",
                    "dimension_explanation",
                ],
                "additionalProperties": False,
            },
        },
    }


def get_codebook_from_content(
    content: str, codebook_id: str, source_pass: str, half: str
) -> Codebook:
    """Parse a model response into a ``Codebook`` (not yet validated against the half). Pure.

    Raises
    ------
    ValueError : If the content is not valid JSON of the expected shape.
    """
    data = json.loads(content)
    assignments: dict[str, str] = {}
    for row in data["assignments"]:
        if row["id"] in assignments:
            raise ValueError(f"passage {row['id']} assigned twice")
        assignments[row["id"]] = row["category"].strip()
    return Codebook(
        codebook_id=codebook_id,
        source_pass=source_pass,
        half=half,
        categories=[
            Category.model_validate(c | {"name": c["name"].strip()}) for c in data["categories"]
        ],
        families=[Family.model_validate(f) for f in data["families"]],
        assignments=assignments,
        spans_multiple_dimensions=bool(data["spans_multiple_dimensions"]),
        dimensions={d["dimension"]: list(d["categories"]) for d in data["dimensions"]},
        dimension_explanation=data["dimension_explanation"],
    )


def get_codebook_errors(codebook: Codebook, ids: list[str]) -> list[str]:
    """Every rule the codebook breaks for the half's passage ``ids``; empty when valid. Pure."""
    errors: list[str] = []
    names = [c.name for c in codebook.categories]
    if not MIN_CATEGORIES <= len(names) <= MAX_CATEGORIES:
        errors.append(f"{len(names)} categories, need {MIN_CATEGORIES}-{MAX_CATEGORIES}")
    if len(set(names)) != len(names) or OTHER in names:
        errors.append("category names are not unique or use the reserved name 'Other'")
    missing = sorted(set(ids) - set(codebook.assignments))
    extra = sorted(set(codebook.assignments) - set(ids))
    if missing:
        errors.append(f"{len(missing)} passages unassigned, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown ids assigned, e.g. {extra[:3]}")
    unknown = sorted({c for c in codebook.assignments.values() if c not in names and c != OTHER})
    if unknown:
        errors.append(f"assignments use unknown categories {unknown[:3]}")
    families = {f.name for f in codebook.families}
    for category in codebook.categories:
        if category.family not in families:
            errors.append(f"category {category.name!r} uses undeclared family {category.family!r}")
        if len(set(category.example_ids)) != 3:
            errors.append(f"category {category.name!r} repeats an example")
        for example in category.example_ids:
            if codebook.assignments.get(example) != category.name:
                errors.append(f"example {example} of {category.name!r} is not assigned to it")
    return errors


def get_usage(response: Any) -> dict[str, int]:
    """Token counts from a chat-completions response. Pure."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
    }


def run_induction(
    items: list[dict[str, Any]],
    codebook_id: str,
    source_pass: str,
    half: str,
    create: Callable[..., Any],
    model: str,
    reasoning_effort: str,
    max_attempts: int = 3,
) -> tuple[Codebook | None, list[dict[str, JsonValue]]]:
    """Induce and validate one codebook, retrying invalid responses.

    Output: the codebook (None when every attempt was invalid) and a log
    entry per call. Side effects: network calls through ``create``.
    """
    ids = [item["id"] for item in items]
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": INDUCTION_SYSTEM_PROMPT},
            {"role": "user", "content": get_induction_prompt(items)},
        ],
        "max_completion_tokens": 32000,
        "reasoning_effort": reasoning_effort,
        "response_format": get_induction_format(),
    }
    logs: list[dict[str, JsonValue]] = []
    for attempt in range(1, max_attempts + 1):
        response = create(**request)
        choice = response.choices[0]
        content = choice.message.content or ""
        entry: dict[str, JsonValue] = {
            "codebook_id": codebook_id,
            "attempt": attempt,
            "response_id": getattr(response, "id", None),
            "reported_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": choice.finish_reason,
            "content": content,
            **get_usage(response),
        }
        try:
            codebook = get_codebook_from_content(content, codebook_id, source_pass, half)
            errors = get_codebook_errors(codebook, ids)
        except (ValueError, KeyError, TypeError) as exc:
            codebook, errors = None, [f"{type(exc).__name__}: {exc}"[:300]]
        entry["validation_errors"] = list(errors)
        logs.append(entry)
        if codebook is not None and not errors:
            return codebook, logs
        logger.warning("%s attempt %d invalid: %s", codebook_id, attempt, errors[:3])
    return None, logs


def get_category_shares(codebook: Codebook, sides: dict[str, str]) -> dict[str, dict[str, float]]:
    """Share of each side's passages assigned to each category (including 'Other'). Pure.

    ``sides`` maps passage id to 'presidential' or 'corporate'.
    """
    totals = Counter(sides[pid] for pid in codebook.assignments)
    counts: Counter[tuple[str, str]] = Counter(
        (category, sides[pid]) for pid, category in codebook.assignments.items()
    )
    names = [c.name for c in codebook.categories] + [OTHER]
    return {
        name: {
            side: (counts[(name, side)] / totals[side]) if totals[side] else 0.0
            for side in ("presidential", "corporate")
        }
        for name in names
    }
