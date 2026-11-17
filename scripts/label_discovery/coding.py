"""D2: open coding of discovery passages by a language model, with no codebook.

Each passage is coded on its own, blind to its source (the coder sees only
the passage text, not whether it is presidential or corporate, nor the
administration or company), so source identity cannot leak into the labels
that later serve as ground truth for cross-source placement.

Every coding pass uses one model and one prompt *lens*. All lenses return the
same core fields: one sentence on what the passage asks the reader to care
about, one to three free labels, the shortest supporting quote, the stance
toward climate action, and whether the passage is really about climate or
energy. Exploratory lenses change how the labels are asked for and add a few
lens-specific fields.

Every request, prompt, setting, raw response, and reported model version is
saved. Calls are appended to a partial log as they finish, so an interrupted
pass resumes without repeating finished calls.

Exports: ``LENSES``, ``PASSES``, ``Lens``, ``PassSpec``, ``PassageCode``,
``get_user_prompt``, ``get_response_format``, ``get_code_from_content``,
``get_quote_is_verbatim``, ``run_coding_pass``.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

logger = logging.getLogger(__name__)

STANCES = ("supports", "opposes", "mixed", "none")
TOPIC_SCOPES = ("climate", "energy", "both", "neither")

_CORE_FIELD_TEXT = "\n".join(
    [
        "Return these fields:",
        "- care: one sentence saying what the passage asks its reader to care about.",
        "- labels: {labels_instruction}",
        "- quote: the shortest exact phrase or sentence from the passage that best supports "
        "your first label, copied character for character.",
        '- stance: the passage\'s stance toward action on climate change: "supports", '
        '"opposes", "mixed", or "none" when it takes no position on climate action.',
        '- topic_scope: whether the passage is really about "climate", "energy", "both", or '
        '"neither". Use "neither" when climate or energy words appear only in passing.',
    ]
)

_SYSTEM_STRAIGHT = (
    "You are a careful qualitative coder in a research study of how the White House and "
    "companies write about climate change and energy. You read one passage at a time and "
    "describe it in your own words. There is no codebook and no list of categories: do not "
    "fit the passage to any framework you know. Stay concrete and faithful to the text."
)
_SYSTEM_NARRATIVE = (
    "You are a discourse analyst in a research study of how the White House and companies "
    "write about climate change and energy. You read one passage at a time for the story it "
    "tells: what problem or opportunity it sees, who is responsible, what should be done, and "
    "who gains or loses. There is no codebook. Be imaginative: unusual, vivid labels are "
    "welcome as long as the passage supports them."
)
_SYSTEM_HIERARCHICAL = (
    "You are a qualitative coder building a classification of how the White House and "
    "companies write about climate change and energy. You read one passage at a time. There "
    "is no codebook yet: you propose where the passage would sit in a classification that "
    "goes from broad areas down to specific angles, in your own words."
)
_SYSTEM_EXPERT = (
    "You are an expert in environmental politics, political communication, and corporate "
    "disclosure, taking part in a research study of how the White House and companies write "
    "about climate change and energy. You read one passage at a time. There is no codebook. "
    "Look past the surface topic: name what is really going on in the passage (the argument, "
    "the justification, the interests served, the rhetorical move, any tension or omission) "
    "in labels a scholar would find illuminating. Creative, unexpected labels are welcome if "
    "the text supports them."
)


class Lens(BaseModel):
    """A prompt lens: system prompt, label instruction, and lens-specific fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lens_id: str = Field(..., description="Lens id.")
    kind: Literal["straightforward", "exploratory"] = Field(..., description="Lens family.")
    system_prompt: str = Field(..., description="System message.")
    labels_instruction: str = Field(..., description="How the labels are asked for.")
    extra_fields: dict[str, str] = Field(
        default_factory=dict, description="Lens-specific string fields and their instructions."
    )


LENSES: dict[str, Lens] = {
    "straightforward": Lens(
        lens_id="straightforward",
        kind="straightforward",
        system_prompt=_SYSTEM_STRAIGHT,
        labels_instruction=(
            "one to three short labels (two to five words each) in your own words, naming what "
            "the passage is about and how it presents it. Put the most important label first."
        ),
    ),
    "narrative": Lens(
        lens_id="narrative",
        kind="exploratory",
        system_prompt=_SYSTEM_NARRATIVE,
        labels_instruction=(
            "one to three short labels (two to six words each) naming the story or framing move "
            "of the passage rather than its subject, for example the promise it makes or the "
            "threat it names. Put the most important label first."
        ),
        extra_fields={
            "problem_or_opportunity": "the problem or opportunity the passage sees, briefly.",
            "responsible_actor": "who the passage casts as responsible or in charge, briefly.",
            "remedy": "what the passage says should be done, in a few words (or 'none').",
            "who_gains_or_loses": "who the passage says gains or loses, briefly (or 'unstated').",
            "time_orientation": "whether the passage looks mainly to the past, present, or future.",
        },
    ),
    "hierarchical": Lens(
        lens_id="hierarchical",
        kind="exploratory",
        system_prompt=_SYSTEM_HIERARCHICAL,
        labels_instruction=(
            "one to three labels, each written as a path of three levels from broad to specific, "
            "separated by ' > ', for example 'broad area > narrower theme > specific angle'. The "
            "broad area should be able to hold many passages from many sources; the specific "
            "angle should fit this passage. Put the most important label first."
        ),
    ),
    "expert_open": Lens(
        lens_id="expert_open",
        kind="exploratory",
        system_prompt=_SYSTEM_EXPERT,
        labels_instruction=(
            "one to three short labels (two to six words each) naming what is really going on "
            "in the passage. Put the most important label first."
        ),
        extra_fields={
            "implied_audience": "who the passage seems written for, in a few words.",
            "what_a_topic_label_misses": "one sentence on what a plain topic label would miss.",
        },
    ),
}


class PassSpec(BaseModel):
    """One coding pass: a model, a lens, and request settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pass_id: str = Field(..., description="Pass id.")
    model: str = Field(..., description="OpenAI model id.")
    lens_id: str = Field(..., description="Lens id (key of LENSES).")
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort, if set.")
    max_completion_tokens: int = Field(default=4000, ge=256, description="Token budget per call.")
    repeat_of: str | None = Field(
        default=None, description="Pass this one repeats with identical prompt and settings."
    )


# The OpenAI defaults rule: gpt-6.1-terra for small to mid tasks, gpt-6.1-sol
# for difficult ones. sol is a reasoning model: an undersized token budget
# returns empty content, so its budget is large.
PASSES: dict[str, PassSpec] = {
    "terra_straight_1": PassSpec(
        pass_id="terra_straight_1", model="gpt-6.1-terra", lens_id="straightforward"
    ),
    "terra_straight_2": PassSpec(
        pass_id="terra_straight_2",
        model="gpt-6.1-terra",
        lens_id="straightforward",
        repeat_of="terra_straight_1",
    ),
    "terra_narrative": PassSpec(
        pass_id="terra_narrative", model="gpt-6.1-terra", lens_id="narrative"
    ),
    "terra_hierarchical": PassSpec(
        pass_id="terra_hierarchical", model="gpt-6.1-terra", lens_id="hierarchical"
    ),
    "sol_straight": PassSpec(
        pass_id="sol_straight",
        model="gpt-6.1-sol",
        lens_id="straightforward",
        reasoning_effort="medium",
        max_completion_tokens=16000,
    ),
    "sol_expert": PassSpec(
        pass_id="sol_expert",
        model="gpt-6.1-sol",
        lens_id="expert_open",
        reasoning_effort="medium",
        max_completion_tokens=16000,
    ),
}


class PassageCode(BaseModel):
    """One passage's code from one pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passage_id: str = Field(..., description="Passage id.")
    pass_id: str = Field(..., description="Pass id.")
    care: str = Field(
        ..., min_length=1, description="What the passage asks the reader to care about."
    )
    labels: list[str] = Field(
        ..., min_length=1, max_length=3, description="Free labels, most important first."
    )
    quote: str = Field(..., min_length=1, description="Shortest supporting quote.")
    quote_verbatim: bool = Field(..., description="True when the quote occurs in the passage.")
    stance: Literal["supports", "opposes", "mixed", "none"] = Field(
        ..., description="Stance on climate action."
    )
    topic_scope: Literal["climate", "energy", "both", "neither"] = Field(
        ..., description="Climate/energy scope."
    )
    extras: dict[str, str] = Field(default_factory=dict, description="Lens-specific fields.")


def get_user_prompt(lens: Lens, text: str) -> str:
    """Build the user message for one passage. Pure."""
    fields = _CORE_FIELD_TEXT.format(labels_instruction=lens.labels_instruction)
    for name, instruction in lens.extra_fields.items():
        fields += f"\n- {name}: {instruction}"
    return f"Code this passage.\n\nPassage:\n<<<\n{text}\n>>>\n\n{fields}"


def get_response_format(lens: Lens) -> dict[str, Any]:
    """Strict JSON-schema response format for a lens. Pure."""
    properties: dict[str, Any] = {
        "care": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "quote": {"type": "string"},
        "stance": {"type": "string", "enum": list(STANCES)},
        "topic_scope": {"type": "string", "enum": list(TOPIC_SCOPES)},
    }
    for name in lens.extra_fields:
        properties[name] = {"type": "string"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"passage_code_{lens.lens_id}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _get_squashed(text: str) -> str:
    """Lowercase text with whitespace collapsed and typographic quotes unified. Pure."""
    table = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    return " ".join(text.translate(table).lower().split())


def get_quote_is_verbatim(quote: str, text: str) -> bool:
    """True when ``quote`` occurs in ``text`` up to case, whitespace, and quote style. Pure."""
    squashed = _get_squashed(quote).strip(" .\"'")
    return bool(squashed) and squashed in _get_squashed(text)


def get_code_from_content(
    content: str, lens: Lens, passage_id: str, pass_id: str, text: str
) -> PassageCode:
    """Parse and validate one model response into a ``PassageCode``. Pure.

    Raises
    ------
    ValueError : If the content is not valid JSON or misses a field.
    """
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    extras = {name: str(data[name]) for name in lens.extra_fields}
    labels = [str(label).strip() for label in data["labels"] if str(label).strip()]
    return PassageCode(
        passage_id=passage_id,
        pass_id=pass_id,
        care=str(data["care"]).strip(),
        labels=labels,
        quote=str(data["quote"]).strip(),
        quote_verbatim=get_quote_is_verbatim(str(data["quote"]), text),
        stance=data["stance"],
        topic_scope=data["topic_scope"],
        extras=extras,
    )


def _get_usage(response: Any) -> dict[str, int]:
    """Token counts from a chat-completions response. Pure."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
    }


def get_request(spec: PassSpec, lens: Lens, text: str) -> dict[str, Any]:
    """The full chat-completions request for one passage. Pure."""
    request: dict[str, Any] = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": lens.system_prompt},
            {"role": "user", "content": get_user_prompt(lens, text)},
        ],
        "max_completion_tokens": spec.max_completion_tokens,
        "response_format": get_response_format(lens),
    }
    if spec.reasoning_effort is not None:
        request["reasoning_effort"] = spec.reasoning_effort
    return request


def _call_with_retries(
    create: Callable[..., Any], request: dict[str, Any], max_attempts: int
) -> tuple[Any, int]:
    """Call ``create(**request)``, retrying API errors with jittered exponential backoff.

    Output: the response and the number of attempts. Side effects: network
    calls through ``create``; sleeps between attempts.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return create(**request), attempt
        except Exception as exc:  # noqa: BLE001 - the SDK raises several error types
            if attempt == max_attempts:
                raise
            delay = min(60.0, 2.0**attempt) + random.random()
            logger.warning("API error (%s); retry %d in %.1fs", type(exc).__name__, attempt, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _code_one(
    create: Callable[..., Any], spec: PassSpec, lens: Lens, passage_id: str, text: str
) -> tuple[PassageCode | None, list[dict[str, JsonValue]]]:
    """Code one passage; returns the code (None on failure) and every call's log entry.

    A response with empty content or invalid JSON is retried up to twice,
    the last time with double the token budget. Side effects: network calls.
    """
    logs: list[dict[str, JsonValue]] = []
    request = get_request(spec, lens, text)
    for content_attempt in range(1, 4):
        if content_attempt == 3:
            request = request | {"max_completion_tokens": spec.max_completion_tokens * 2}
        response, api_attempts = _call_with_retries(create, request, max_attempts=6)
        choice = response.choices[0]
        content = choice.message.content or ""
        entry: dict[str, JsonValue] = {
            "passage_id": passage_id,
            "pass_id": spec.pass_id,
            "content_attempt": content_attempt,
            "api_attempts": api_attempts,
            "response_id": getattr(response, "id", None),
            "reported_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "created": getattr(response, "created", None),
            "finish_reason": choice.finish_reason,
            "max_completion_tokens": request["max_completion_tokens"],
            "content": content,
            **_get_usage(response),
        }
        try:
            code = get_code_from_content(content, lens, passage_id, spec.pass_id, text)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            entry["parse_error"] = f"{type(exc).__name__}: {exc}"[:500]
            logs.append(entry)
            continue
        logs.append(entry)
        return code, logs
    return None, logs


def run_coding_pass(
    passages: list[tuple[str, str]],
    spec: PassSpec,
    create: Callable[..., Any],
    partial_log: Path,
    max_workers: int = 12,
) -> tuple[list[PassageCode], list[dict[str, JsonValue]], list[str]]:
    """Code every ``(passage_id, text)`` pair with one pass.

    Finished calls are appended to ``partial_log`` (one JSON object per line)
    as they complete; passages already coded there are not called again.
    Output: codes (sorted by passage id), every call's log entry, and the ids
    that failed after all retries.
    Side effects: network calls through ``create``; appends to ``partial_log``.
    """
    lens = LENSES[spec.lens_id]
    texts = dict(passages)
    logs: list[dict[str, JsonValue]] = []
    codes: dict[str, PassageCode] = {}
    if partial_log.exists():
        for line in partial_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            logs.append(entry)
            if "parse_error" not in entry and entry["passage_id"] in texts:
                codes[entry["passage_id"]] = get_code_from_content(
                    entry["content"],
                    lens,
                    entry["passage_id"],
                    spec.pass_id,
                    texts[entry["passage_id"]],
                )
    todo = [(pid, text) for pid, text in passages if pid not in codes]
    logger.info("%s: %d passages to code (%d already done)", spec.pass_id, len(todo), len(codes))
    lock = threading.Lock()
    failed: list[str] = []
    partial_log.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(partial_log, "a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=max_workers) as pool,
    ):
        futures = {pool.submit(_code_one, create, spec, lens, pid, text): pid for pid, text in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            pid = futures[future]
            code, entries = future.result()
            with lock:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                logs.extend(entries)
                if code is None:
                    failed.append(pid)
                else:
                    codes[pid] = code
            if done % 50 == 0:
                logger.info("%s: %d/%d", spec.pass_id, done, len(todo))
    return [codes[k] for k in sorted(codes)], logs, sorted(failed)
