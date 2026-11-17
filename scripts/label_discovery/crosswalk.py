"""D4: align all candidate codebooks and cross them with the ten deductive frames.

One alignment call puts every category of every candidate codebook into a
concept group (categories that mean the same thing share a group) and maps
each group to zero or more of the deductive frames fixed in the protocol.
The model sees only category names, definitions, and families: never
passages, sources, or REAP outputs.

Everything after the alignment is computed, not judged: which models, lenses,
and halves produced each group; its class (stable, model-specific, or
sample-specific); whether it is emergent (no deductive frame); which
deductive frames no group reaches (deductive-only); and each group's share of
presidential and corporate passages, averaged over the codebooks that contain
it.

Exports: ``DEDUCTIVE_FRAMES``, ``ConceptGroup``, ``GroupStats``,
``get_alignment_prompt``, ``get_alignment_format``, ``get_groups_from_content``,
``get_alignment_errors``, ``get_model_family``, ``get_group_stats``,
``get_group_shares_by_stratum``.
"""

from __future__ import annotations

import json
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from label_discovery.codebooks import Codebook, get_category_shares

# The ten deductive frames, as named in the case-study protocol. Only names
# are given to the aligner so it does not inherit anyone's gloss.
DEDUCTIVE_FRAMES: dict[str, str] = {
    "economic_opportunity_jobs": "economic opportunity and jobs",
    "energy_supply_security": "energy supply and security",
    "rules_compliance": "rules and compliance",
    "science_risk_health": "science, risk and health",
    "disasters_resilience": "disasters and resilience",
    "international_commitments": "international commitments",
    "technology_innovation": "technology and innovation",
    "fairness_communities": "fairness and communities",
    "nature_stewardship": "nature and stewardship",
    "targets_accounting": "targets and accounting",
}

ALIGNMENT_SYSTEM_PROMPT = (
    "You are a qualitative methodologist comparing candidate codebooks that different coders "
    "built from the same kind of passages (White House and corporate texts on climate and "
    "energy). Judge meaning, not wording: categories that would capture the same passages "
    "belong together even when their names differ."
)


class ConceptGroup(BaseModel):
    """Categories from different codebooks that share one meaning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str = Field(..., description="Short id, e.g. 'G01'.")
    name: str = Field(..., description="Group name.")
    definition: str = Field(..., description="One-sentence definition.")
    suggested_family: str = Field(..., description="Broader family the group would sit in.")
    members: list[str] = Field(
        ..., min_length=1, description="Category refs '<codebook id>::<name>'."
    )
    deductive_frames: list[str] = Field(
        ..., description="Matched deductive frame ids (may be empty)."
    )


class GroupStats(BaseModel):
    """Computed properties of one concept group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str = Field(..., description="Group id.")
    name: str = Field(..., description="Group name.")
    n_codebooks: int = Field(..., ge=1, description="Codebooks containing the group.")
    support: float = Field(..., description="Share of all codebooks containing the group.")
    models: list[str] = Field(..., description="Model families that produced it.")
    lenses: list[str] = Field(..., description="Lens kinds that produced it.")
    halves: list[str] = Field(..., description="Halves it appears in.")
    stability: Literal["stable", "model-specific", "sample-specific"] = Field(
        ...,
        description="stable: both halves and two or more models; model-specific: both halves, "
        "one model; sample-specific: one half only.",
    )
    emergent: bool = Field(..., description="True when no deductive frame matches.")
    deductive_frames: list[str] = Field(..., description="Matched frame ids.")
    presidential_share_mean: float = Field(..., description="Mean share of presidential passages.")
    presidential_share_range: list[float] = Field(..., description="Min and max over codebooks.")
    corporate_share_mean: float = Field(..., description="Mean share of corporate passages.")
    corporate_share_range: list[float] = Field(..., description="Min and max over codebooks.")


def get_model_family(source_pass: str) -> str:
    """Model family of a pass id ('terra', 'sol', or 'claude'). Pure.

    Raises
    ------
    ValueError : If the pass id names no known family.
    """
    for family in ("terra", "sol", "claude"):
        if source_pass.startswith(family):
            return family
    raise ValueError(f"cannot tell the model family of pass {source_pass!r}")


def get_alignment_prompt(codebooks: list[Codebook]) -> str:
    """User message listing every category of every codebook. Pure."""
    frames = "\n".join(f"- {key}: {name}" for key, name in DEDUCTIVE_FRAMES.items())
    lines = [
        f"Below are the categories of {len(codebooks)} candidate codebooks. Each line gives a "
        "category reference ('<codebook id>::<category name>'), its family in that codebook, "
        "and its definition.",
        "",
        "1. Put every category into exactly one concept group. A group holds categories, from "
        "any codebooks, that would capture the same passages. A group may hold several "
        "categories from one codebook only if that codebook split one idea into parts.",
        "2. Give each group an id (G01, G02, ...), a short name, a one-sentence definition, and "
        "the broader family it would sit in.",
        "3. Map each group to the deductive frames below that substantially match its meaning "
        "(zero, one, or more). Use the frame ids exactly.",
        "",
        "Deductive frames:",
        frames,
        "",
        "Categories:",
    ]
    for codebook in codebooks:
        for category in codebook.categories:
            lines.append(
                f"{codebook.codebook_id}::{category.name} [family: {category.family}] "
                f"{category.definition}"
            )
    return "\n".join(lines)


def get_alignment_format() -> dict[str, Any]:
    """Strict JSON-schema response format for the alignment. Pure."""
    string = {"type": "string"}
    group = {
        "type": "object",
        "properties": {
            "group_id": string,
            "name": string,
            "definition": string,
            "suggested_family": string,
            "members": {"type": "array", "items": string},
            "deductive_frames": {
                "type": "array",
                "items": {"type": "string", "enum": list(DEDUCTIVE_FRAMES)},
            },
        },
        "required": [
            "group_id",
            "name",
            "definition",
            "suggested_family",
            "members",
            "deductive_frames",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "codebook_alignment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"groups": {"type": "array", "items": group}},
                "required": ["groups"],
                "additionalProperties": False,
            },
        },
    }


def get_groups_from_content(content: str) -> list[ConceptGroup]:
    """Parse the alignment response. Pure.

    Raises
    ------
    ValueError : If the content is not valid JSON of the expected shape.
    """
    return [ConceptGroup.model_validate(g) for g in json.loads(content)["groups"]]


def get_alignment_errors(groups: list[ConceptGroup], codebooks: list[Codebook]) -> list[str]:
    """Every rule the alignment breaks; empty when valid. Pure."""
    expected = {f"{b.codebook_id}::{c.name}" for b in codebooks for c in b.categories}
    seen: list[str] = [m for g in groups for m in g.members]
    errors: list[str] = []
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    doubled = sorted(m for m in set(seen) if seen.count(m) > 1)
    if missing:
        errors.append(f"{len(missing)} categories in no group, e.g. {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown category refs, e.g. {extra[:3]}")
    if doubled:
        errors.append(f"{len(doubled)} categories in more than one group, e.g. {doubled[:3]}")
    ids = [g.group_id for g in groups]
    if len(set(ids)) != len(ids):
        errors.append("group ids are not unique")
    return errors


def get_group_stats(
    groups: list[ConceptGroup],
    codebooks: list[Codebook],
    sides: dict[str, str],
    lens_kinds: dict[str, str],
) -> tuple[list[GroupStats], list[str]]:
    """Compute every group's provenance, class, and passage shares; list deductive-only frames.

    Inputs: the alignment groups, the codebooks, passage id to side, and pass
    id to lens kind. Output: stats per group (ordered by support, then id)
    and the deductive frame ids no group matches. Pure.
    """
    by_id = {b.codebook_id: b for b in codebooks}
    shares = {b.codebook_id: get_category_shares(b, sides) for b in codebooks}
    stats: list[GroupStats] = []
    for group in groups:
        per_codebook: dict[str, list[str]] = {}
        for member in group.members:
            codebook_id, name = member.split("::", 1)
            per_codebook.setdefault(codebook_id, []).append(name)
        models = sorted({get_model_family(by_id[c].source_pass) for c in per_codebook})
        lenses = sorted({lens_kinds[by_id[c].source_pass] for c in per_codebook})
        halves = sorted({by_id[c].half for c in per_codebook})
        pres = [
            sum(shares[c][n]["presidential"] for n in names) for c, names in per_codebook.items()
        ]
        corp = [sum(shares[c][n]["corporate"] for n in names) for c, names in per_codebook.items()]
        if len(halves) < 2:
            stability: Literal["stable", "model-specific", "sample-specific"] = "sample-specific"
        elif len(models) >= 2:
            stability = "stable"
        else:
            stability = "model-specific"
        stats.append(
            GroupStats(
                group_id=group.group_id,
                name=group.name,
                n_codebooks=len(per_codebook),
                support=round(len(per_codebook) / len(codebooks), 4),
                models=models,
                lenses=lenses,
                halves=halves,
                stability=stability,
                emergent=not group.deductive_frames,
                deductive_frames=sorted(group.deductive_frames),
                presidential_share_mean=round(mean(pres), 4),
                presidential_share_range=[round(min(pres), 4), round(max(pres), 4)],
                corporate_share_mean=round(mean(corp), 4),
                corporate_share_range=[round(min(corp), 4), round(max(corp), 4)],
            )
        )
    reached = {frame for group in groups for frame in group.deductive_frames}
    deductive_only = [frame for frame in DEDUCTIVE_FRAMES if frame not in reached]
    return sorted(stats, key=lambda s: (-s.support, s.group_id)), deductive_only


def get_group_shares_by_stratum(
    groups: list[ConceptGroup], codebooks: list[Codebook], strata: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Mean share of each stratum's passages that each group captures. Pure.

    ``strata`` maps passage id to a stratum label (for example an
    administration, 'corporate', or 'excluded'). For each codebook that
    contains the group, the share is the fraction of that codebook's passages
    in the stratum assigned to the group's member categories; the result is
    the mean over those codebooks.
    """
    by_id = {b.codebook_id: b for b in codebooks}
    result: dict[str, dict[str, float]] = {}
    labels = sorted(set(strata.values()))
    for group in groups:
        members: dict[str, set[str]] = {}
        for member in group.members:
            codebook_id, name = member.split("::", 1)
            members.setdefault(codebook_id, set()).add(name)
        shares: dict[str, list[float]] = {label: [] for label in labels}
        for codebook_id, names in members.items():
            assignments = by_id[codebook_id].assignments
            for label in labels:
                ids = [pid for pid in assignments if strata[pid] == label]
                if ids:
                    hits = sum(assignments[pid] in names for pid in ids)
                    shares[label].append(hits / len(ids))
        result[group.group_id] = {
            label: round(mean(values), 4) for label, values in shares.items() if values
        }
    return result
