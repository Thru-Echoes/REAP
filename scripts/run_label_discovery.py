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
