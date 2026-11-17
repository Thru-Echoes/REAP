"""Driver for label discovery in the corporate/presidential case study (steps D0-D4).

Each subcommand runs one step and writes its artifacts plus a ``bundle.json``
record file under
``<research root>/results/corporate_presidential/label_discovery/``:

- ``pool``       D0  passage pool from the White House tree and the corporate
                     extractor manifest (sources are read only)
- ``sample``     D1  stratified discovery sample, halves A and B
- ``code``       D2  one open-coding pass by one OpenAI model and prompt lens
- ``ingest-inline`` D2/D3 validate and record coding or codebooks produced by
                     a Claude model inline in an interactive session
- ``induce``     D3  group one pass's free labels into a candidate codebook per half
- ``crosswalk``  D4  align all candidate codebooks and cross them with the
                     deductive frames
- ``tables``     D4  each concept group's share of every stratum (no model call)

The research root resolves fail-closed through ``research_root.py``
(``--research-root``, then ``REAP_RESEARCH_ROOT``, then the sibling checkout).
OpenAI steps read ``OPENAI_API_KEY`` from the environment or from the
research root's ``.env``; the key is never printed or recorded.

Discovery never reads REAP topics, maps, or clusters for this corpus.

Example::

    python scripts/run_label_discovery.py pool \\
        --white-house-root <green-narrative>/notebooks/admin_text \\
        --manifest-db <corp-sus-report-extractor>/data/manifest.db \\
        --extractor-root <corp-sus-report-extractor> \\
        --path-prefix /old/prefix/Research/=<research folder>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_discovery.files import set_json, set_jsonl  # noqa: E402
from label_discovery.passages import KEYWORD_LIST_ID, KEYWORD_PATTERNS  # noqa: E402
from label_discovery.records import (  # noqa: E402
    DiscoveryRecord,
    FileDigest,
    get_code_state,
    get_file_digest,
    get_software_versions,
    set_record,
)
from research_root import get_research_root  # noqa: E402  # pyright: ignore[reportMissingImports]

logger = logging.getLogger("run_label_discovery")

REAP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/run_label_discovery.py"
OUTPUT_SUBDIR = Path("results") / "corporate_presidential" / "label_discovery"


def get_output_root(research_root: Path) -> Path:
    """Return the label-discovery output folder under the research root. Pure."""
    return research_root / OUTPUT_SUBDIR


def get_now() -> datetime:
    """Return the current UTC time. Side effects: reads the clock."""
    return datetime.now(timezone.utc)


def get_listing_digest(role: str, label: str, entries: list[tuple[str, str]]) -> FileDigest:
    """Digest a source tree as the SHA-256 of its sorted ``path<TAB>sha256`` listing. Pure."""
    listing = "\n".join(f"{path}\t{sha}" for path, sha in sorted(entries))
    return FileDigest(
        role=role,
        path=label,
        sha256=hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        n_rows=len(entries),
    )


def set_step_record(
    *,
    step: str,
    run_id: str,
    out_dir: Path,
    research_root: Path,
    inputs: list[FileDigest],
    outputs: list[tuple[str, Path, int | None]],
    config: dict[str, Any],
    seeds: list[int],
    models: list[Any],
    started_at: datetime,
) -> DiscoveryRecord:
    """Hash the outputs and write the step's ``bundle.json``.

    Side effects: reads the output files; writes ``out_dir/bundle.json``.
    """
    python_version, platform_name, packages = get_software_versions()
    record = DiscoveryRecord(
        step=step,  # type: ignore[arg-type]
        run_id=run_id,
        script=SCRIPT,
        code=get_code_state(REAP_ROOT),
        inputs=inputs,
        outputs=[
            get_file_digest(role, path, path.relative_to(research_root).as_posix(), n_rows)
            for role, path, n_rows in outputs
        ],
        config=json.loads(json.dumps(config)),
        seeds=seeds,
        models=models,
        python_version=python_version,
        platform=platform_name,
        packages=packages,
        run_status="completed",
        started_at=started_at,
        finished_at=get_now(),
    )
    set_record(record, out_dir / "bundle.json")
    return record


def _get_prefix_map(values: list[str]) -> dict[str, Path]:
    """Parse ``OLD=NEW`` path-prefix pairs. Pure.

    Raises
    ------
    ValueError : If a value has no ``=``.
    """
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-prefix needs OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        mapping[old] = Path(new).expanduser()
    return mapping


def run_pool(args: argparse.Namespace) -> int:
    """D0: build and write the passage pool. Side effects: reads sources; writes artifacts."""
    from label_discovery.pool import PoolConfig, get_pool, get_pool_rows, get_pool_summary
    from label_discovery.sources import get_corporate_reports, get_white_house_documents

    started = get_now()
    research_root = get_research_root(args.research_root)
    out_dir = get_output_root(research_root) / "d0_passage_pool"
    config = PoolConfig()
    wh_root = Path(args.white_house_root).expanduser()
    manifest_db = Path(args.manifest_db).expanduser()
    extractor_root = Path(args.extractor_root).expanduser()

    logger.info(
        "reading White House documents %d-%d from %s", config.year_min, config.year_max, wh_root
    )
    skipped: Counter[str] = Counter()
    wh_docs = list(get_white_house_documents(wh_root, config.year_min, config.year_max, skipped))
    logger.info("read %d in-scope White House documents", len(wh_docs))
    reports, manifest_sha, left_out = get_corporate_reports(
        manifest_db, extractor_root, _get_prefix_map(args.path_prefix)
    )
    logger.info("read %d corporate reports; verifying passages against PDFs", len(reports))
    result = get_pool(wh_docs, reports, config)

    wh_sources = [
        {
            "doc_id": d.doc_id,
            "source_path": d.source_path,
            "sha256": d.source_sha256,
            "admin": d.admin,
            "year": d.year,
            "doc_type": d.doc_type,
        }
        for d in wh_docs
    ]
    corp_sources = [
        {
            "doc_id": "cs-" + hashlib.sha1(r.record_id.encode("utf-8")).hexdigest()[:10],
            "record_id": r.record_id,
            "company_slug": r.company_slug,
            "industry": r.industry,
            "year": r.year,
            "extraction_model": r.extraction_model,
            "pdf_sha256": r.pdf_sha256,
            "pdf_name": r.pdf_path.name,
        }
        for r in reports
    ]
    pool_path = out_dir / "pool.jsonl.gz"
    neighbours_path = out_dir / "excluded_neighbours.jsonl.gz"
    wh_list_path = out_dir / "white_house_sources.jsonl.gz"
    corp_list_path = out_dir / "corporate_sources.jsonl.gz"
    keywords_path = out_dir / "keywords.json"
    summary_path = out_dir / "pool_summary.json"
    n_kept = set_jsonl(pool_path, get_pool_rows(result.kept))
    n_neigh = set_jsonl(neighbours_path, get_pool_rows(result.excluded_neighbours))
    set_jsonl(wh_list_path, wh_sources)
    set_jsonl(corp_list_path, corp_sources)
    set_json(
        keywords_path,
        {
            "keyword_list_id": KEYWORD_LIST_ID,
            "match_rule": "keep a passage when ANY pattern matches, case-insensitive",
            "patterns": KEYWORD_PATTERNS,
        },
    )
    set_json(
        summary_path,
        get_pool_summary(result)
        | {
            "white_house_skipped": dict(sorted(skipped.items())),
            "corporate_reports_left_out": left_out,
        },
    )

    inputs = [
        get_listing_digest(
            "white_house_tree",
            "white_house_root:",
            [(d.source_path, d.source_sha256) for d in wh_docs],
        ),
        FileDigest(
            role="corporate_manifest_db",
            path="extractor_root:data/manifest.db",
            sha256=manifest_sha,
        ),
        get_listing_digest(
            "corporate_pdfs",
            "extractor_root:pdfs",
            [(r.pdf_path.name, r.pdf_sha256) for r in reports],
        ),
    ]
    set_step_record(
        step="D0",
        run_id="corporate_presidential/label_discovery/d0_passage_pool",
        out_dir=out_dir,
        research_root=research_root,
        inputs=inputs,
        outputs=[
            ("pool", pool_path, n_kept),
            ("excluded_neighbours", neighbours_path, n_neigh),
            ("white_house_source_list", wh_list_path, len(wh_sources)),
            ("corporate_source_list", corp_list_path, len(corp_sources)),
            ("keyword_list", keywords_path, None),
            ("pool_summary", summary_path, None),
        ],
        config=config.model_dump(mode="json") | {"keyword_list_id": KEYWORD_LIST_ID},
        seeds=[],
        models=[],
        started_at=started,
    )
    logger.info("pool: %d kept passages, %d excluded neighbours -> %s", n_kept, n_neigh, out_dir)
    return 0


def get_openai_key(research_root: Path) -> str:
    """Return ``OPENAI_API_KEY`` from the environment or the research root's ``.env``.

    The key is returned to the caller only; it is never logged or recorded.
    Side effects: reads the environment and ``.env``.

    Raises
    ------
    RuntimeError : If no key is found.
    """
    import os

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    env_file = research_root / ".env"
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set and not found in the research root .env")
    return key


def get_model_use(
    provider: str, requested: str, settings: dict[str, Any], logs: list[dict[str, Any]]
) -> Any:
    """Summarize one model's calls for the record file. Pure."""
    from label_discovery.records import ModelUse

    return ModelUse(
        provider=provider,  # type: ignore[arg-type]
        requested_model=requested,
        reported_models=sorted({str(e["reported_model"]) for e in logs if e.get("reported_model")}),
        settings=json.loads(json.dumps(settings)),
        n_calls=len(logs),
        prompt_tokens=sum(int(e.get("prompt_tokens") or 0) for e in logs),
        completion_tokens=sum(int(e.get("completion_tokens") or 0) for e in logs),
        reasoning_tokens=sum(int(e.get("reasoning_tokens") or 0) for e in logs),
    )


def run_sample(args: argparse.Namespace) -> int:
    """D1: draw the discovery sample. Side effects: reads the pool; writes artifacts."""
    import csv
    import io

    from label_discovery.files import get_jsonl, set_text
    from label_discovery.pool import get_passage_from_row
    from label_discovery.sample import SampleConfig, get_discovery_sample, get_sample_row

    started = get_now()
    research_root = get_research_root(args.research_root)
    root = get_output_root(research_root)
    pool_dir = root / "d0_passage_pool"
    out_dir = root / "d1_discovery_sample"
    kept = [get_passage_from_row(r) for r in get_jsonl(pool_dir / "pool.jsonl.gz")]
    neighbours = [
        get_passage_from_row(r) for r in get_jsonl(pool_dir / "excluded_neighbours.jsonl.gz")
    ]
    config = SampleConfig()
    sampled, ranking = get_discovery_sample(kept, neighbours, config)

    rows = [get_sample_row(s) for s in sampled]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table_path = out_dir / "sample.csv"
    passages_path = out_dir / "sample_passages.jsonl"
    ids_path = out_dir / "discovery_ids.txt"
    ranking_path = out_dir / "industry_ranking.json"
    set_text(table_path, buffer.getvalue())
    set_jsonl(passages_path, [s.model_dump(mode="json") for s in sampled])
    set_text(
        ids_path,
        "".join(
            f"{s.passage.passage_id}\n" for s in sorted(sampled, key=lambda s: s.passage.passage_id)
        ),
    )
    sampled_industries = {
        s.stratum.split(":", 1)[1] for s in sampled if s.stratum.startswith("industry:")
    }
    set_json(
        ranking_path,
        [
            {
                "industry": name,
                "companies": companies,
                "passages": n_passages,
                "stratum": name
                if name in sampled_industries
                else "other"
                if "other" in sampled_industries
                else "not sampled",
            }
            for name, companies, n_passages in ranking
        ],
    )
    set_step_record(
        step="D1",
        run_id="corporate_presidential/label_discovery/d1_discovery_sample",
        out_dir=out_dir,
        research_root=research_root,
        inputs=[
            get_file_digest(
                "pool",
                pool_dir / "pool.jsonl.gz",
                (pool_dir / "pool.jsonl.gz").relative_to(research_root).as_posix(),
                len(kept),
            ),
            get_file_digest(
                "excluded_neighbours",
                pool_dir / "excluded_neighbours.jsonl.gz",
                (pool_dir / "excluded_neighbours.jsonl.gz").relative_to(research_root).as_posix(),
                len(neighbours),
            ),
        ],
        outputs=[
            ("sample_table", table_path, len(rows)),
            ("sample_passages", passages_path, len(sampled)),
            ("discovery_ids", ids_path, len(sampled)),
            ("industry_ranking", ranking_path, len(ranking)),
        ],
        config=config.model_dump(mode="json"),
        seeds=[config.seed],
        models=[],
        started_at=started,
    )
    logger.info("sample: %d passages -> %s", len(sampled), out_dir)
    return 0


def get_sample_passages(research_root: Path) -> list[dict[str, Any]]:
    """Read the D1 sample passages. Side effects: reads the file; fails closed if missing."""
    from label_discovery.files import get_jsonl

    return get_jsonl(
        get_output_root(research_root) / "d1_discovery_sample" / "sample_passages.jsonl"
    )


def run_code(args: argparse.Namespace) -> int:
    """D2: run one open-coding pass. Side effects: OpenAI calls; writes artifacts."""
    import openai
    from label_discovery.coding import (
        LENSES,
        PASSES,
        get_response_format,
        get_user_prompt,
        run_coding_pass,
    )
    from label_discovery.files import get_file_sha256

    started = get_now()
    research_root = get_research_root(args.research_root)
    spec = PASSES[args.pass_id]
    lens = LENSES[spec.lens_id]
    sample_rows = get_sample_passages(research_root)
    passages = [(r["passage"]["passage_id"], r["passage"]["text"]) for r in sample_rows]
    if args.limit:
        passages = passages[: args.limit]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else get_output_root(research_root) / "d2_open_coding" / spec.pass_id
    )
    client = openai.OpenAI(api_key=get_openai_key(research_root), max_retries=0, timeout=300)
    partial = out_dir / "calls.partial.jsonl"
    codes, logs, failed = run_coding_pass(
        passages, spec, client.chat.completions.create, partial, args.max_workers
    )

    prompts_path = out_dir / "prompts.json"
    codes_path = out_dir / "codes.jsonl"
    calls_path = out_dir / "calls.jsonl.gz"
    set_json(
        prompts_path,
        {
            "pass": spec.model_dump(mode="json"),
            "lens": lens.model_dump(mode="json"),
            "user_prompt_template": get_user_prompt(lens, "{passage_text}"),
            "response_format": get_response_format(lens),
            "blinding": (
                "the coder sees only the passage text; no source, administration, company, or year"
            ),
        },
    )
    n_codes = set_jsonl(codes_path, [c.model_dump(mode="json") for c in codes])
    n_calls = set_jsonl(
        calls_path,
        sorted(logs, key=lambda e: (str(e["passage_id"]), int(e["content_attempt"] or 0))),
    )
    if args.out_dir:
        logger.info(
            "pilot run: %d codes, %d calls -> %s (no record file)", n_codes, n_calls, out_dir
        )
        partial.unlink(missing_ok=True)
        return 0
    sample_path = get_output_root(research_root) / "d1_discovery_sample" / "sample_passages.jsonl"
    settings = {
        "max_completion_tokens": spec.max_completion_tokens,
        "reasoning_effort": spec.reasoning_effort,
        "response_format": "json_schema strict",
        "temperature": "API default (1); not settable",
    }
    record_kwargs: dict[str, Any] = dict(
        step="D2",
        run_id=f"corporate_presidential/label_discovery/d2_open_coding/{spec.pass_id}",
        out_dir=out_dir,
        research_root=research_root,
        inputs=[
            FileDigest(
                role="sample_passages",
                path=sample_path.relative_to(research_root).as_posix(),
                sha256=get_file_sha256(sample_path),
                n_rows=len(sample_rows),
            )
        ],
        outputs=[
            ("prompts", prompts_path, None),
            ("codes", codes_path, n_codes),
            ("calls", calls_path, n_calls),
        ],
        config={
            "pass": spec.model_dump(mode="json"),
            "lens_id": lens.lens_id,
            "lens_kind": lens.kind,
            "n_passages": len(passages),
            "failed_passage_ids": failed,
        },
        seeds=[],
        models=[get_model_use("openai", spec.model, settings, logs)],
        started_at=started,
    )
    if failed:
        python_version, platform_name, packages = get_software_versions()
        record = DiscoveryRecord(
            step="D2",
            run_id=record_kwargs["run_id"],
            script=SCRIPT,
            code=get_code_state(REAP_ROOT),
            inputs=record_kwargs["inputs"],
            outputs=[
                get_file_digest(r, p, p.relative_to(research_root).as_posix(), n)
                for r, p, n in record_kwargs["outputs"]
            ],
            config=json.loads(json.dumps(record_kwargs["config"])),
            seeds=[],
            models=record_kwargs["models"],
            python_version=python_version,
            platform=platform_name,
            packages=packages,
            run_status="failed",
            failure_reason=f"{len(failed)} passages failed after all retries; rerun to resume",
            started_at=started,
            finished_at=get_now(),
        )
        set_record(record, out_dir / "bundle.json")
        logger.error(
            "%s: %d passages failed; partial log kept for resume", spec.pass_id, len(failed)
        )
        return 1
    set_step_record(**record_kwargs)
    partial.unlink(missing_ok=True)
    logger.info("%s: %d codes -> %s", spec.pass_id, n_codes, out_dir)
    return 0


INLINE_PASS_ID = "claude_inline_straight"
INLINE_MODEL = "claude-opus-5-5"


def get_rel(path: Path, research_root: Path) -> str:
    """Research-root-relative display path. Pure."""
    return path.relative_to(research_root).as_posix()


def get_codes(research_root: Path, pass_id: str) -> list[dict[str, Any]]:
    """Read one completed pass's codes; fails closed when the pass has no completed record.

    Side effects: reads the pass folder.
    """
    from label_discovery.files import get_jsonl
    from label_discovery.records import get_record

    folder = get_output_root(research_root) / "d2_open_coding" / pass_id
    record = get_record(folder / "bundle.json")
    if record.run_status != "completed":
        raise RuntimeError(f"pass {pass_id} did not complete; refusing to use its codes")
    return get_jsonl(folder / "codes.jsonl")


def run_ingest_inline(args: argparse.Namespace) -> int:
    """D2/D3: validate and record coding and codebooks a Claude model produced inline.

    Inline work is done inside an interactive session (no API call), on
    blinded, shuffled passages; this step maps blind ids back, validates every
    code and codebook against the same models the API passes use, and writes
    the record. Side effects: reads the inputs; writes artifacts.
    """
    from label_discovery.codebooks import get_codebook_errors, get_codebook_from_content
    from label_discovery.coding import LENSES, PassageCode, get_quote_is_verbatim, get_user_prompt
    from label_discovery.files import get_file_sha256, get_jsonl
    from label_discovery.records import ModelUse

    started = get_now()
    research_root = get_research_root(args.research_root)
    sample = {r["passage"]["passage_id"]: r for r in get_sample_passages(research_root)}
    blind = json.loads(Path(args.blind_map).read_text(encoding="utf-8"))
    to_pid = {m["blind_id"]: m["passage_id"] for m in blind["mapping"]}
    subset = {pid for pid, r in sample.items() if r["in_inline_subset"]}
    if set(to_pid.values()) != subset:
        raise ValueError("blind map does not cover exactly the inline subset")
    lens = LENSES["straightforward"]
    raw_rows = [row for path in args.codes for row in get_jsonl(Path(path))]
    codes = []
    for row in raw_rows:
        pid = to_pid[row["blind_id"]]
        codes.append(
            PassageCode(
                passage_id=pid,
                pass_id=INLINE_PASS_ID,
                care=row["care"],
                labels=row["labels"],
                quote=row["quote"],
                quote_verbatim=get_quote_is_verbatim(row["quote"], sample[pid]["passage"]["text"]),
                stance=row["stance"],
                topic_scope=row["topic_scope"],
            )
        )
    if sorted(c.passage_id for c in codes) != sorted(subset):
        raise ValueError("inline codes do not cover each inline-subset passage exactly once")
    code_dir = get_output_root(research_root) / "d2_open_coding" / INLINE_PASS_ID
    codes_path, raw_path, map_path, prompts_path = (
        code_dir / "codes.jsonl",
        code_dir / "raw_inline_codes.jsonl",
        code_dir / "blind_map.json",
        code_dir / "prompts.json",
    )
    set_jsonl(
        codes_path, [c.model_dump(mode="json") for c in sorted(codes, key=lambda c: c.passage_id)]
    )
    set_jsonl(raw_path, raw_rows)
    set_json(map_path, blind)
    set_json(
        prompts_path,
        {
            "lens": lens.model_dump(mode="json"),
            "user_prompt_template": get_user_prompt(lens, "{passage_text}"),
            "procedure": (
                "the 120 inline-subset passages were shuffled with the blind-map seed, given "
                "neutral ids, and read in six batches of 20 with no source, administration, "
                "company, or year; codes were written against the straightforward lens "
                "instructions, then mapped back to passage ids by this step"
            ),
        },
    )
    inline_use = ModelUse(
        provider="anthropic-inline",
        requested_model=INLINE_MODEL,
        reported_models=[INLINE_MODEL],
        settings={"lens_id": "straightforward", "interface": "interactive session, no API call"},
        n_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
    )
    sample_path = get_output_root(research_root) / "d1_discovery_sample" / "sample_passages.jsonl"
    sample_digest = FileDigest(
        role="sample_passages",
        path=get_rel(sample_path, research_root),
        sha256=get_file_sha256(sample_path),
        n_rows=len(sample),
    )
    set_step_record(
        step="D2",
        run_id=f"corporate_presidential/label_discovery/d2_open_coding/{INLINE_PASS_ID}",
        out_dir=code_dir,
        research_root=research_root,
        inputs=[sample_digest],
        outputs=[
            ("codes", codes_path, len(codes)),
            ("raw_inline_codes", raw_path, len(raw_rows)),
            ("blind_map", map_path, len(to_pid)),
            ("prompts", prompts_path, None),
        ],
        config={
            "pass_id": INLINE_PASS_ID,
            "lens_id": "straightforward",
            "lens_kind": "straightforward",
            "n_passages": len(codes),
            "blind_seed": blind["seed"],
        },
        seeds=[int(blind["seed"])],
        models=[inline_use],
        started_at=started,
    )
    for half, path in (("A", args.codebook_a), ("B", args.codebook_b)):
        content = Path(path).read_text(encoding="utf-8")
        codebook_id = f"{INLINE_PASS_ID}/{half}~claude"
        codebook = get_codebook_from_content(content, codebook_id, INLINE_PASS_ID, half)
        ids = [c.passage_id for c in codes if sample[c.passage_id]["half"] == half]
        errors = get_codebook_errors(codebook, ids)
        if errors:
            raise ValueError(f"inline codebook {half} is invalid: {errors}")
        out_dir = (
            get_output_root(research_root) / "d3_codebooks" / f"{INLINE_PASS_ID}~claude" / half
        )
        codebook_path = out_dir / "codebook.json"
        set_json(codebook_path, codebook.model_dump(mode="json"))
        set_step_record(
            step="D3",
            run_id=f"corporate_presidential/label_discovery/d3_codebooks/{INLINE_PASS_ID}~claude/{half}",
            out_dir=out_dir,
            research_root=research_root,
            inputs=[
                get_file_digest("codes", codes_path, get_rel(codes_path, research_root), len(codes))
            ],
            outputs=[("codebook", codebook_path, len(codebook.categories))],
            config={
                "codebook_id": codebook_id,
                "source_pass": INLINE_PASS_ID,
                "half": half,
                "inducer": "Claude inline, from its own labels and care sentences",
            },
            seeds=[],
            models=[
                inline_use.model_copy(
                    update={
                        "settings": {
                            "task": "codebook induction",
                            "interface": "interactive session, no API call",
                        }
                    }
                )
            ],
            started_at=started,
        )
    logger.info("inline: %d codes and two codebooks recorded", len(codes))
    return 0


def get_all_pass_ids(research_root: Path) -> list[str]:
    """Pass ids with a completed D2 record, sorted. Side effects: reads records."""
    from label_discovery.records import get_record

    folder = get_output_root(research_root) / "d2_open_coding"
    ids = []
    for bundle in sorted(folder.glob("*/bundle.json")):
        # Folders starting with "_" hold derived outputs (the summary), not passes.
        if bundle.parent.name.startswith("_") or not (bundle.parent / "codes.jsonl").is_file():
            continue
        if get_record(bundle).run_status == "completed":
            ids.append(bundle.parent.name)
    return ids


def run_summarize(args: argparse.Namespace) -> int:
    """D2 summary: distributions, keyword-rule check, repeat-run and cross-model agreement.

    Side effects: reads codes; writes the summary and its record.
    """
    from label_discovery.stats import get_pair_agreement, get_pass_summary

    started = get_now()
    research_root = get_research_root(args.research_root)
    sample = {r["passage"]["passage_id"]: r for r in get_sample_passages(research_root)}
    pass_ids = get_all_pass_ids(research_root)
    codes = {pid: get_codes(research_root, pid) for pid in pass_ids}
    repeat_ids = {pid for pid, r in sample.items() if r["in_repeat_set"]}
    pairs: dict[str, Any] = {}
    if "terra_straight_1" in codes and "terra_straight_2" in codes:
        pairs["repeat_run_terra_straight__repeat_set"] = get_pair_agreement(
            codes["terra_straight_1"], codes["terra_straight_2"], repeat_ids
        )
        pairs["repeat_run_terra_straight__all"] = get_pair_agreement(
            codes["terra_straight_1"], codes["terra_straight_2"]
        )
    for a, b in (
        ("terra_straight_1", "sol_straight"),
        ("terra_straight_1", INLINE_PASS_ID),
        ("sol_straight", INLINE_PASS_ID),
        ("terra_straight_1", "terra_narrative"),
        ("sol_straight", "sol_expert"),
    ):
        if a in codes and b in codes:
            pairs[f"{a}__vs__{b}"] = get_pair_agreement(codes[a], codes[b])
    from label_discovery.files import get_jsonl
    from label_discovery.stats import get_keyword_rule_check

    pool_path = get_output_root(research_root) / "d0_passage_pool" / "pool.jsonl.gz"
    pool = get_jsonl(pool_path)
    keyword_check = (
        get_keyword_rule_check(codes["terra_straight_1"], codes["sol_straight"], sample, pool)
        if "terra_straight_1" in codes and "sol_straight" in codes
        else {}
    )
    out_dir = get_output_root(research_root) / "d2_open_coding" / "_summary"
    summary_path = out_dir / "coding_summary.json"
    set_json(
        summary_path,
        {
            "passes": {pid: get_pass_summary(c, sample) for pid, c in codes.items()},
            "agreement": pairs,
            "keyword_rule_check": keyword_check,
        },
    )
    inputs = [
        get_file_digest("codes", p, get_rel(p, research_root), len(codes[pid]))
        for pid in pass_ids
        for p in [get_output_root(research_root) / "d2_open_coding" / pid / "codes.jsonl"]
    ] + [get_file_digest("pool", pool_path, get_rel(pool_path, research_root), len(pool))]
    set_step_record(
        step="D2",
        run_id="corporate_presidential/label_discovery/d2_open_coding/_summary",
        out_dir=out_dir,
        research_root=research_root,
        inputs=inputs,
        outputs=[("coding_summary", summary_path, None)],
        config={
            "passes": pass_ids,
            "keyword_rule_check_passes": ["terra_straight_1", "sol_straight"],
        },
        seeds=[],
        models=[],
        started_at=started,
    )
    logger.info("summary of %d passes -> %s", len(pass_ids), summary_path)
    return 0


def run_induce(args: argparse.Namespace) -> int:
    """D3: induce one candidate codebook per half from one pass's labels.

    Side effects: OpenAI calls; writes codebooks, logs, and records.
    """
    import openai
    from label_discovery.codebooks import (
        INDUCTION_SYSTEM_PROMPT,
        get_induction_format,
        get_induction_prompt,
        run_induction,
    )

    research_root = get_research_root(args.research_root)
    sample = {r["passage"]["passage_id"]: r for r in get_sample_passages(research_root)}
    codes = get_codes(research_root, args.pass_id)
    codes_path = get_output_root(research_root) / "d2_open_coding" / args.pass_id / "codes.jsonl"
    client = openai.OpenAI(api_key=get_openai_key(research_root), max_retries=4, timeout=900)
    status = 0
    for half in ("A", "B"):
        started = get_now()
        items = [
            {"id": c["passage_id"], "labels": c["labels"], "care": c["care"]}
            for c in codes
            if sample[c["passage_id"]]["half"] == half
        ]
        codebook_id = f"{args.pass_id}/{half}"
        codebook, logs = run_induction(
            items,
            codebook_id,
            args.pass_id,
            half,
            client.chat.completions.create,
            args.model,
            args.reasoning_effort,
        )
        out_dir = get_output_root(research_root) / "d3_codebooks" / args.pass_id / half
        calls_path, prompt_path, codebook_path = (
            out_dir / "calls.jsonl.gz",
            out_dir / "prompt.json",
            out_dir / "codebook.json",
        )
        set_jsonl(calls_path, logs)
        set_json(
            prompt_path,
            {
                "system": INDUCTION_SYSTEM_PROMPT,
                "user": get_induction_prompt(items),
                "response_format": get_induction_format(),
            },
        )
        outputs: list[tuple[str, Path, int | None]] = [
            ("calls", calls_path, len(logs)),
            ("prompt", prompt_path, None),
        ]
        if codebook is not None:
            set_json(codebook_path, codebook.model_dump(mode="json"))
            outputs.append(("codebook", codebook_path, len(codebook.categories)))
        settings = {
            "max_completion_tokens": 32000,
            "reasoning_effort": args.reasoning_effort,
            "response_format": "json_schema strict",
        }
        kwargs: dict[str, Any] = dict(
            step="D3",
            run_id=f"corporate_presidential/label_discovery/d3_codebooks/{codebook_id}",
            out_dir=out_dir,
            research_root=research_root,
            inputs=[
                get_file_digest("codes", codes_path, get_rel(codes_path, research_root), len(codes))
            ],
            outputs=outputs,
            config={
                "codebook_id": codebook_id,
                "source_pass": args.pass_id,
                "half": half,
                "n_passages": len(items),
                "inducer_model": args.model,
            },
            seeds=[],
            models=[get_model_use("openai", args.model, settings, logs)],
            started_at=started,
        )
        if codebook is None:
            python_version, platform_name, packages = get_software_versions()
            record = DiscoveryRecord(
                step="D3",
                run_id=kwargs["run_id"],
                script=SCRIPT,
                code=get_code_state(REAP_ROOT),
                inputs=kwargs["inputs"],
                outputs=[
                    get_file_digest(r, p, get_rel(p, research_root), n) for r, p, n in outputs
                ],
                config=json.loads(json.dumps(kwargs["config"])),
                seeds=[],
                models=kwargs["models"],
                python_version=python_version,
                platform=platform_name,
                packages=packages,
                run_status="failed",
                failure_reason="no valid codebook after all attempts; see validation_errors",
                started_at=started,
                finished_at=get_now(),
            )
            set_record(record, out_dir / "bundle.json")
            status = 1
            logger.error("%s: no valid codebook", codebook_id)
            continue
        set_step_record(**kwargs)
        logger.info("%s: %d categories", codebook_id, len(codebook.categories))
    return status


def get_all_codebooks(research_root: Path) -> list[Any]:
    """Every completed candidate codebook, sorted by id. Side effects: reads records and files."""
    from label_discovery.codebooks import Codebook
    from label_discovery.records import get_record

    books = []
    for bundle in sorted((get_output_root(research_root) / "d3_codebooks").glob("*/*/bundle.json")):
        if get_record(bundle).run_status != "completed":
            continue
        books.append(
            Codebook.model_validate_json(
                (bundle.parent / "codebook.json").read_text(encoding="utf-8")
            )
        )
    return sorted(books, key=lambda b: b.codebook_id)


def run_crosswalk(args: argparse.Namespace) -> int:
    """D4: align every candidate codebook and cross it with the deductive frames.

    Side effects: one OpenAI alignment call (retried when invalid); writes artifacts.
    """
    import csv
    import io

    import openai
    from label_discovery.codebooks import get_usage
    from label_discovery.coding import LENSES, PASSES
    from label_discovery.crosswalk import (
        ALIGNMENT_SYSTEM_PROMPT,
        DEDUCTIVE_FRAMES,
        get_alignment_errors,
        get_alignment_format,
        get_alignment_prompt,
        get_group_stats,
        get_groups_from_content,
    )
    from label_discovery.files import set_text

    started = get_now()
    research_root = get_research_root(args.research_root)
    sample = {r["passage"]["passage_id"]: r for r in get_sample_passages(research_root)}
    sides = {pid: r["passage"]["side"] for pid, r in sample.items()}
    books = get_all_codebooks(research_root)
    lens_kinds = {pid: LENSES[spec.lens_id].kind for pid, spec in PASSES.items()} | {
        INLINE_PASS_ID: "straightforward"
    }
    client = openai.OpenAI(api_key=get_openai_key(research_root), max_retries=4, timeout=1200)
    request: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT},
            {"role": "user", "content": get_alignment_prompt(books)},
        ],
        "max_completion_tokens": 64000,
        "reasoning_effort": args.reasoning_effort,
        "response_format": get_alignment_format(),
    }
    logs: list[dict[str, Any]] = []
    groups = None
    for attempt in range(1, 4):
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        entry: dict[str, Any] = {
            "attempt": attempt,
            "response_id": response.id,
            "reported_model": response.model,
            "system_fingerprint": response.system_fingerprint,
            "finish_reason": response.choices[0].finish_reason,
            "content": content,
            **get_usage(response),
        }
        try:
            candidate = get_groups_from_content(content)
            errors = get_alignment_errors(candidate, books)
        except (ValueError, KeyError, TypeError) as exc:
            candidate, errors = None, [f"{type(exc).__name__}: {exc}"[:300]]
        entry["validation_errors"] = errors
        logs.append(entry)
        if candidate is not None and not errors:
            groups = candidate
            break
        logger.warning("alignment attempt %d invalid: %s", attempt, errors[:3])
    out_dir = get_output_root(research_root) / "d4_crosswalk"
    calls_path = out_dir / "calls.jsonl.gz"
    set_jsonl(calls_path, logs)
    if groups is None:
        logger.error("no valid alignment after all attempts")
        return 1
    stats, deductive_only = get_group_stats(groups, books, sides, lens_kinds)
    groups_path, table_path, frames_path = (
        out_dir / "alignment.json",
        out_dir / "concept_groups.csv",
        out_dir / "deductive_frames.json",
    )
    set_json(
        groups_path,
        {
            "groups": [g.model_dump(mode="json") for g in groups],
            "codebooks": [b.codebook_id for b in books],
        },
    )
    rows = [s.model_dump(mode="json") for s in stats]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: ";".join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()}
        )
    set_text(table_path, buffer.getvalue())
    set_json(
        frames_path,
        {
            "frames": DEDUCTIVE_FRAMES,
            "deductive_only": deductive_only,
            "groups_per_frame": {
                f: [s.group_id for s in stats if f in s.deductive_frames] for f in DEDUCTIVE_FRAMES
            },
        },
    )
    inputs = [
        get_file_digest("codebook", p, get_rel(p, research_root), None)
        for p in sorted((get_output_root(research_root) / "d3_codebooks").glob("*/*/codebook.json"))
    ]
    settings = {
        "max_completion_tokens": 64000,
        "reasoning_effort": args.reasoning_effort,
        "response_format": "json_schema strict",
    }
    set_step_record(
        step="D4",
        run_id="corporate_presidential/label_discovery/d4_crosswalk",
        out_dir=out_dir,
        research_root=research_root,
        inputs=inputs,
        outputs=[
            ("calls", calls_path, len(logs)),
            ("alignment", groups_path, len(groups)),
            ("concept_groups", table_path, len(rows)),
            ("deductive_frames", frames_path, None),
        ],
        config={
            "codebooks": [b.codebook_id for b in books],
            "aligner_model": args.model,
            "stability_rule": (
                "stable = both halves and two or more model families; model-specific = "
                "both halves, one family; sample-specific = one half"
            ),
        },
        seeds=[],
        models=[get_model_use("openai", args.model, settings, logs)],
        started_at=started,
    )
    logger.info(
        "crosswalk: %d groups over %d codebooks; deductive-only: %s",
        len(groups),
        len(books),
        deductive_only,
    )
    return 0


def run_tables(args: argparse.Namespace) -> int:
    """D4 tables: each concept group's share of every stratum, from the stored alignment.

    Pure computation over the committed alignment and codebooks (no model
    call), so the shares can be cited without repeating the alignment.
    Side effects: reads artifacts; writes the table and its record.
    """
    from label_discovery.crosswalk import ConceptGroup, get_group_shares_by_stratum

    started = get_now()
    research_root = get_research_root(args.research_root)
    sample = {r["passage"]["passage_id"]: r for r in get_sample_passages(research_root)}
    strata = {
        pid: "excluded"
        if r["keyword_excluded"]
        else (r["passage"]["admin"] if r["passage"]["side"] == "presidential" else "corporate")
        for pid, r in sample.items()
    }
    root = get_output_root(research_root)
    alignment_path = root / "d4_crosswalk" / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    groups = [ConceptGroup.model_validate(g) for g in alignment["groups"]]
    books = get_all_codebooks(research_root)
    if [b.codebook_id for b in books] != alignment["codebooks"]:
        raise RuntimeError("codebooks on disk differ from the ones the alignment was made from")
    shares = get_group_shares_by_stratum(groups, books, strata)
    out_dir = root / "d4_crosswalk" / "tables"
    table_path = out_dir / "group_shares_by_stratum.json"
    set_json(
        table_path,
        {
            "definition": "mean over the codebooks containing the group of the share of a "
            "stratum's passages assigned to the group's member categories; strata are "
            "administrations for "
            "keyword-kept presidential passages, 'corporate' for keyword-kept corporate passages, "
            "and 'excluded' for keyword-excluded passages",
            "shares": shares,
            "names": {g.group_id: g.name for g in groups},
        },
    )
    set_step_record(
        step="D4",
        run_id="corporate_presidential/label_discovery/d4_crosswalk/tables",
        out_dir=out_dir,
        research_root=research_root,
        inputs=[
            get_file_digest(
                "alignment", alignment_path, get_rel(alignment_path, research_root), len(groups)
            )
        ],
        outputs=[("group_shares_by_stratum", table_path, len(shares))],
        config={"strata": sorted(set(strata.values()))},
        seeds=[],
        models=[],
        started_at=started,
    )
    logger.info("tables -> %s", table_path)
    return 0


def get_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Pure."""
    parser = argparse.ArgumentParser(
        description="Label discovery for the corporate/presidential case study."
    )
    parser.add_argument("--research-root", default=None, help="REAP-research checkout.")
    sub = parser.add_subparsers(dest="command", required=True)

    pool = sub.add_parser("pool", help="D0: build the passage pool.")
    pool.add_argument(
        "--white-house-root", required=True, help="Folder holding <year>/whitehouse/."
    )
    pool.add_argument("--manifest-db", required=True, help="Corporate extractor manifest.db.")
    pool.add_argument("--extractor-root", required=True, help="Corporate extractor project root.")
    pool.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        help="OLD=NEW prefix rewrite for stale PDF paths in the manifest (repeatable).",
    )
    pool.set_defaults(func=run_pool)

    sample = sub.add_parser("sample", help="D1: draw the discovery sample.")
    sample.set_defaults(func=run_sample)

    code = sub.add_parser("code", help="D2: run one open-coding pass.")
    code.add_argument(
        "--pass-id", required=True, help="Pass id from label_discovery.coding.PASSES."
    )
    code.add_argument("--max-workers", type=int, default=12, help="Concurrent API calls.")
    code.add_argument(
        "--limit", type=int, default=0, help="Code only the first N passages (pilot)."
    )
    code.add_argument(
        "--out-dir", default=None, help="Pilot output folder; no record file is written."
    )
    code.set_defaults(func=run_code)

    inline = sub.add_parser(
        "ingest-inline", help="D2/D3: record inline Claude coding and codebooks."
    )
    inline.add_argument(
        "--codes", nargs="+", required=True, help="Inline code JSONL files (blind ids)."
    )
    inline.add_argument("--blind-map", required=True, help="Blind-id map JSON.")
    inline.add_argument("--codebook-a", required=True, help="Inline codebook JSON for half A.")
    inline.add_argument("--codebook-b", required=True, help="Inline codebook JSON for half B.")
    inline.set_defaults(func=run_ingest_inline)

    summary = sub.add_parser("summarize", help="D2: summarize all completed coding passes.")
    summary.set_defaults(func=run_summarize)

    induce = sub.add_parser("induce", help="D3: induce codebooks for halves A and B from one pass.")
    induce.add_argument("--pass-id", required=True, help="Completed coding pass id.")
    induce.add_argument("--model", default="gpt-6.1-sol", help="Inducing model.")
    induce.add_argument("--reasoning-effort", default="high", help="Reasoning effort.")
    induce.set_defaults(func=run_induce)

    cross = sub.add_parser("crosswalk", help="D4: align codebooks and cross with deductive frames.")
    cross.add_argument("--model", default="gpt-6.1-sol", help="Aligning model.")
    cross.add_argument("--reasoning-effort", default="high", help="Reasoning effort.")
    cross.set_defaults(func=run_crosswalk)

    tables = sub.add_parser("tables", help="D4: per-stratum shares of each concept group.")
    tables.set_defaults(func=run_tables)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code. Side effects: per subcommand."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = get_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
