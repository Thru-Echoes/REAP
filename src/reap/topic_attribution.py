"""Cross-dataset topic-attribution evaluation via 3-LLM rubric.

Pre-registered in ``manuscript/evaluation_protocol.md`` §17.

The bridge claim REAP wants to make is "REAP's projection head places
semantically-related OOS documents into semantically-coherent reference
clusters." Verifying that requires an INDEPENDENT semantic judgement of
which reference clusters are related to which OOS classes/themes.

This module implements two LLM-judging tasks:

**Step A — Cluster naming.** For each reference cluster of each dataset,
three independent judges (Claude Opus 4.7 via subagent, OpenAI
``gpt-5.4-mini``, OpenAI ``gpt-5.5``) produce a (label, macro_theme,
description, confidence) tuple from the cluster's top c-TF-IDF terms +
representative documents.

**Step B — Relatedness rubric.** For each (OOS class/theme × labeled
reference cluster) pair, the same three judges score relatedness on a
3-level ordinal scale {0=disjoint, 1=marginal, 2=clearly_related}.

A fourth subagent later consolidates the judges' outputs into a
consensus rubric and reports Krippendorff's α per dataset (the
defensibility band is pre-registered at α ≥ 0.5).

This module exposes the pure OpenAI judging surface; the Claude subagent
dispatch is performed by the controlling Claude Code session (not this
module) so the agent infrastructure handles Claude calls. Both judges'
outputs are stored as JSON for reproducibility (§17.g).

Reuses ``src/reap/labeling.py``'s LLM-call primitives where applicable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from reap.labeling import _call_llm, _check_llm_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default judges
# ---------------------------------------------------------------------------

# Pinned in CLAUDE.md memory; do not bump without asking.
OPENAI_MINI_MODEL: str = "gpt-5.4-mini"
# Added 2026-05-14 per user direction as a stronger judge alongside the mini.
OPENAI_STRONG_MODEL: str = "gpt-5.5"

# Claude judge runs as a dispatched subagent (not via the anthropic API);
# we record the model id in outputs so the audit trail is complete.
CLAUDE_SUBAGENT_MODEL_ID: str = "claude-opus-4-7-via-subagent"

# Pre-registered relatedness scale (§17.c).
RELATEDNESS_LABELS: dict[int, str] = {
    0: "disjoint",
    1: "marginal",
    2: "clearly_related",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ClusterLabel(BaseModel):
    """One judge's labeling of one reference cluster (§17.b)."""

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(..., description="Dataset name (e.g., 'twenty_newsgroups_reference')")
    cluster_id: int = Field(..., ge=0, description="REAP cluster id")
    judge_model: str = Field(..., description="LLM model id (e.g., 'gpt-5.4-mini')")
    judge_provider: Literal["openai", "anthropic-subagent"] = Field(
        ..., description="Provider channel for the call",
    )
    label: str = Field(..., description="Short cluster name (≤ 8 words)")
    macro_theme: str = Field(..., description="Higher-level thematic category")
    description: str = Field(..., description="1-2 sentence cluster description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's confidence")
    raw_response: str | None = Field(
        default=None,
        description="The raw model response text, for audit trail (§17.g)",
    )


class RelatednessJudgment(BaseModel):
    """One judge's relatedness score for one (OOS class × reference cluster) pair (§17.c)."""

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(..., description="Dataset name")
    oos_class_or_theme: str = Field(
        ..., description="OOS class name or theme label (e.g., 'talk.religion.misc' or 'compensation')",
    )
    reference_cluster_id: int = Field(..., ge=0, description="REAP cluster id")
    reference_cluster_label: str = Field(
        ..., description="Consensus name of the reference cluster from Step A",
    )
    judge_model: str = Field(..., description="LLM model id")
    judge_provider: Literal["openai", "anthropic-subagent"] = Field(..., description="Provider channel")
    relatedness_score: int = Field(..., ge=0, le=2, description="0=disjoint, 1=marginal, 2=clearly_related")
    rationale: str = Field(..., description="Why the judge picked this score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's confidence")
    raw_response: str | None = Field(default=None, description="Raw model output for audit trail")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_STEP_A_PROMPT_TEMPLATE = """You are an expert topic-modeling annotator. Your task is to name and \
describe a cluster of documents that an unsupervised algorithm has grouped together. \
Read the cluster's top terms and representative documents and return a JSON object.

Dataset: {dataset_description}

Cluster {cluster_id} of {n_clusters} (size = {cluster_size} documents).

Top c-TF-IDF terms (highest TF-IDF, in this cluster vs all others):
{top_terms_block}

Representative documents (5 docs nearest the cluster centroid):
{rep_docs_block}

Return ONLY a single JSON object with these keys:
  "label": a short cluster name (≤ 8 words)
  "macro_theme": a higher-level thematic category (1-3 words)
  "description": a 1-2 sentence description of what the cluster is about
  "confidence": your confidence in the label, 0.0 to 1.0

Do not include any prose outside the JSON. Do not wrap the JSON in markdown fences."""


_STEP_B_PROMPT_TEMPLATE = """You are an expert topic-modeling annotator. Your task is to judge \
how semantically related an out-of-sample (OOS) class is to a labeled reference cluster, \
based on each one's description and sample documents.

Use this 3-level scale:
  0 = disjoint: no semantic overlap; the OOS class is not topically related to this cluster.
  1 = marginal: tangential or partial topical overlap; could plausibly fit in some narrow sense.
  2 = clearly_related: strong semantic affinity; the OOS class topically belongs here.

Dataset: {dataset_description}

Reference cluster {cluster_id}: "{cluster_label}" ({cluster_macro_theme})
Description: {cluster_description}
Sample documents from this cluster:
{cluster_sample_docs}

OOS class / theme: "{oos_class_name}"
{oos_class_description}
Sample documents from this OOS class/theme:
{oos_sample_docs}

Return ONLY a JSON object with these keys:
  "relatedness_score": 0, 1, or 2 (integer)
  "rationale": one short sentence explaining your score
  "confidence": your confidence, 0.0 to 1.0

Do not include any prose outside the JSON. Do not wrap the JSON in markdown fences."""


# ---------------------------------------------------------------------------
# OpenAI judging surface
# ---------------------------------------------------------------------------


def label_cluster_openai(
    *,
    dataset: str,
    dataset_description: str,
    cluster_id: int,
    n_clusters: int,
    cluster_size: int,
    top_terms: list[str],
    rep_docs: list[str],
    model: str,
    max_doc_chars: int = 600,
) -> ClusterLabel:
    """Run Step A (cluster naming) on a single cluster via OpenAI.

    Pure function over the inputs + the OpenAI API. Saves nothing; the
    caller is responsible for persistence (the raw response is on the
    returned object for the audit trail per §17.g).
    """
    _check_llm_provider("openai")
    top_terms_block = "\n".join(f"  - {t}" for t in top_terms[:10])
    rep_docs_block = "\n".join(
        f"  [{i + 1}] {d[:max_doc_chars]}" for i, d in enumerate(rep_docs[:5])
    )
    prompt = _STEP_A_PROMPT_TEMPLATE.format(
        dataset_description=dataset_description,
        cluster_id=cluster_id,
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        top_terms_block=top_terms_block,
        rep_docs_block=rep_docs_block,
    )
    raw = _call_llm(provider="openai", model=model, prompt=prompt)
    parsed = _parse_step_a_response(raw)
    return ClusterLabel(
        dataset=dataset,
        cluster_id=cluster_id,
        judge_model=model,
        judge_provider="openai",
        label=parsed["label"],
        macro_theme=parsed["macro_theme"],
        description=parsed["description"],
        confidence=float(parsed["confidence"]),
        raw_response=raw,
    )


def judge_relatedness_openai(
    *,
    dataset: str,
    dataset_description: str,
    cluster_id: int,
    cluster_label: str,
    cluster_macro_theme: str,
    cluster_description: str,
    cluster_sample_docs: list[str],
    oos_class_name: str,
    oos_class_description: str,
    oos_sample_docs: list[str],
    model: str,
    max_doc_chars: int = 500,
) -> RelatednessJudgment:
    """Run Step B (relatedness rubric) on one (OOS class, reference cluster) pair via OpenAI."""
    _check_llm_provider("openai")
    prompt = _STEP_B_PROMPT_TEMPLATE.format(
        dataset_description=dataset_description,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        cluster_macro_theme=cluster_macro_theme,
        cluster_description=cluster_description,
        cluster_sample_docs="\n".join(
            f"  [{i + 1}] {d[:max_doc_chars]}" for i, d in enumerate(cluster_sample_docs[:3])
        ),
        oos_class_name=oos_class_name,
        oos_class_description=(
            f"Description: {oos_class_description}" if oos_class_description else ""
        ),
        oos_sample_docs="\n".join(
            f"  [{i + 1}] {d[:max_doc_chars]}" for i, d in enumerate(oos_sample_docs[:3])
        ),
    )
    raw = _call_llm(provider="openai", model=model, prompt=prompt)
    parsed = _parse_step_b_response(raw)
    return RelatednessJudgment(
        dataset=dataset,
        oos_class_or_theme=oos_class_name,
        reference_cluster_id=cluster_id,
        reference_cluster_label=cluster_label,
        judge_model=model,
        judge_provider="openai",
        relatedness_score=int(parsed["relatedness_score"]),
        rationale=str(parsed["rationale"]),
        confidence=float(parsed["confidence"]),
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_step_a_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's Step-A JSON response with light tolerance for surrounding whitespace.

    Raises ValueError if required keys are missing or malformed.
    """
    obj = _extract_json_object(raw)
    for key in ("label", "macro_theme", "description", "confidence"):
        if key not in obj:
            raise ValueError(f"Step-A response missing key '{key}': {raw[:200]!r}")
    return obj


def _parse_step_b_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's Step-B JSON response."""
    obj = _extract_json_object(raw)
    for key in ("relatedness_score", "rationale", "confidence"):
        if key not in obj:
            raise ValueError(f"Step-B response missing key '{key}': {raw[:200]!r}")
    score = int(obj["relatedness_score"])
    if score not in (0, 1, 2):
        raise ValueError(f"Step-B relatedness_score must be 0/1/2; got {score}")
    obj["relatedness_score"] = score
    return obj


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Find the first ``{...}`` JSON object in the raw response.

    Some LLMs add prose around the JSON despite instructions; we extract the
    first balanced ``{ ... }`` substring and parse it. Falls back to the full
    string. Strips common ``` markdown fences.
    """
    s = raw.strip()
    if s.startswith("```"):
        # Strip ```json ... ``` or ``` ... ``` fences.
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Find first balanced object.
    start = s.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start : i + 1])
    # If we never closed, try parsing what we have.
    return json.loads(s[start:])


# ---------------------------------------------------------------------------
# .env loading helper — modules importing this module can call once to
# populate the OpenAI API key from a .env file at the repo root.
# ---------------------------------------------------------------------------


def load_dotenv(env_path: Path | None = None) -> None:
    """Load OPENAI_API_KEY and ANTHROPIC_API_KEY from .env into os.environ.

    Side effect: mutates os.environ. Does not overwrite already-set keys.
    """
    p = env_path or Path(".env")
    if not p.exists():
        logger.info("load_dotenv: no .env file at %s; skipping", p)
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Cluster-input helpers (used by Step A runner)
# ---------------------------------------------------------------------------


def get_top_ctfidf_terms_per_cluster(
    texts: list[str], labels: np.ndarray, top_n: int = 10,
) -> dict[int, list[str]]:
    """Compute top-N c-TF-IDF terms per cluster.

    Reuses :func:`reap.labeling.label_clusters_ctfidf` so the term selection is
    consistent with the rest of the package.
    """
    from reap.labeling import label_clusters_ctfidf

    results = label_clusters_ctfidf(texts, labels, top_n=top_n)
    return {int(r.cluster): [t.term for t in r.top_terms] for r in results}


def get_representative_docs_per_cluster(
    texts: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_rep: int = 5,
) -> dict[int, list[str]]:
    """Pick the ``n_rep`` documents nearest each cluster's centroid in embedding space.

    Pure function. Returns ``{cluster_id: [doc_text, ...]}``.
    """
    rep: dict[int, list[str]] = {}
    for c in sorted({int(x) for x in labels.tolist()}):
        mask = labels == c
        if not mask.any():
            continue
        cluster_emb = embeddings[mask]
        cluster_texts = [texts[i] for i in np.where(mask)[0].tolist()]
        centroid = cluster_emb.mean(axis=0)
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        order = np.argsort(dists)
        picked: list[str] = []
        for idx in order:
            t = cluster_texts[int(idx)].strip()
            if t:
                picked.append(t)
            if len(picked) >= n_rep:
                break
        rep[c] = picked
    return rep
