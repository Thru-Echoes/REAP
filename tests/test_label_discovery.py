"""Tests for the corporate/presidential label-discovery steps (scripts/label_discovery).

Logic is tested on small synthetic sources built in a temporary folder (a
White House document tree, a SQLite extractor manifest, and real PDFs written
with PyMuPDF), so CI covers it. OpenAI is the one external service and is
replaced by a fake client that returns canned JSON; everything else runs for
real. The live tests at the bottom check the committed artifacts in the
REAP-research checkout and skip, with a reason, where that checkout or its
discovery tree is absent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from label_discovery.coding import (  # noqa: E402
    LENSES,
    PASSES,
    PassSpec,
    get_code_from_content,
    get_quote_is_verbatim,
    get_response_format,
    run_coding_pass,
)
from label_discovery.files import get_jsonl, set_jsonl  # noqa: E402
from label_discovery.passages import (  # noqa: E402
    ChunkingRule,
    get_keyword_matches,
    get_normalized_words,
    get_paragraphs,
    get_passages,
    get_verbatim_score,
    get_word_count,
    get_word_ngrams,
)
from label_discovery.pool import PoolConfig, PoolPassage, get_pool  # noqa: E402
from label_discovery.records import DiscoveryRecord, get_record  # noqa: E402
from label_discovery.sample import (  # noqa: E402
    SampleConfig,
    get_discovery_sample,
    get_ranked_industries,
)
from label_discovery.sources import get_corporate_reports, get_white_house_documents  # noqa: E402

RULE = ChunkingRule()


def _words(n: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Passages: paragraphs, chunking, keywords, verbatim score
# ---------------------------------------------------------------------------


def test_paragraphs_join_wrapped_lines_hyphens_and_ligatures() -> None:
    text = "Our sus-\ntainability plan is\nefﬁcient.\n\n\n  Second   paragraph.  "
    assert get_paragraphs(text) == ["Our sustainability plan is efficient.", "Second paragraph."]


def test_passages_respect_paragraph_and_word_caps() -> None:
    paragraphs = [_words(60, f"p{i}w") for i in range(7)]
    passages = get_passages(paragraphs, RULE)
    assert all(get_word_count(p) <= RULE.max_words for p in passages)
    assert all(p.count("\n\n") + 1 <= RULE.max_paragraphs for p in passages)
    assert sum(get_word_count(p) for p in passages) == 7 * 60  # nothing lost or duplicated


def test_heading_joins_next_paragraph_and_fragments_are_dropped() -> None:
    passages = get_passages(["Clean Energy", _words(50), "Barack Obama"], RULE)
    assert len(passages) == 1
    assert passages[0].startswith("Clean Energy word0")
    assert get_passages(["Too short to keep as a passage at all."], RULE) == []


def test_long_paragraph_splits_at_sentences_under_the_cap() -> None:
    sentence = _words(30) + "."
    passages = get_passages([" ".join([sentence] * 20)], RULE)
    assert len(passages) >= 3
    assert all(get_word_count(p) <= RULE.max_words for p in passages)
    assert all(p.endswith(".") for p in passages)


def test_keywords_cover_climate_and_energy_but_not_generic_words() -> None:
    assert get_keyword_matches("We cut methane emissions and built offshore wind.") == [
        "emissions",
        "methane",
        "wind_power",
    ]
    assert get_keyword_matches("Gas prices and crude oil rose.") == ["gas", "oil"]
    assert get_keyword_matches("Nuclear weapons talks resumed.") == []
    assert get_keyword_matches("Our sustainability and environment report.") == []
    assert "nuclear_power" in get_keyword_matches("a new nuclear power plant")


def test_verbatim_score_separates_copies_from_paraphrases() -> None:
    source = (
        "We aim to cut our Scope 1 and 2 emissions in half by 2030 against a 2019 baseline, " * 3
    )
    grams = get_word_ngrams(get_normalized_words(source), 6)
    assert (
        get_verbatim_score("We aim to cut our Scope 1 and 2\nemissions in half by 2030", grams, 6)
        == 1.0
    )
    assert (
        get_verbatim_score("The firm wants emissions halved before the decade ends", grams, 6)
        == 0.0
    )
    with pytest.raises(ValueError):
        get_verbatim_score("too short", grams, 6)


# ---------------------------------------------------------------------------
# Synthetic sources
# ---------------------------------------------------------------------------

_FOOTER = [
    "Barack Obama, Statement Online by Gerhard Peters and John T. Woolley, The Presidency Project",
    "Twitter Facebook",
]


def _write_wh_doc(
    root: Path, day: str, president: str, body: list[str], doc_type: str = "statements"
) -> None:
    folder = root / day[:4] / "whitehouse"
    folder.mkdir(parents=True, exist_ok=True)
    name = (
        f"{day}_{doc_type}_doc_{hashlib.sha1((day + president + body[0]).encode()).hexdigest()[:6]}"
    )
    (folder / f"{name}.txt").write_text(
        "\n\n".join(["About Search", *body, *_FOOTER]), encoding="utf-8"
    )
    from datetime import datetime

    date_text = datetime.strptime(day, "%Y%m%d").strftime("%B %d, %Y")
    with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["url", "title", "date_text", "year", "president", "doc_type", "agency"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "url": f"https://example.org/{name}",
                "title": f"Title {name}",
                "date_text": date_text,
                "year": day[:4],
                "president": president,
                "doc_type": doc_type,
                "agency": "whitehouse",
            }
        )


def _write_pdf(path: Path, text: str) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    lines = text.split(" ")
    for start in range(0, len(lines), 300):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(36, 36, 576, 806), " ".join(lines[start : start + 300]), fontsize=7
        )
    doc.save(path)
    doc.close()


def _write_manifest(path: Path, reports: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE companies (company_slug TEXT PRIMARY KEY, company_name TEXT, industry TEXT);
        CREATE TABLE report_records (record_id TEXT PRIMARY KEY, company_slug TEXT,
            company_name TEXT, industry TEXT, year INTEGER, local_path TEXT, sha256_hash TEXT);
        CREATE TABLE extraction_results (extraction_id INTEGER PRIMARY KEY, record_id TEXT,
            model_name TEXT, status TEXT);
        CREATE TABLE extraction_sections (section_id INTEGER PRIMARY KEY, extraction_id INTEGER,
            section_index INTEGER, section_title TEXT, section_text TEXT);
        """
    )
    for i, r in enumerate(reports, start=1):
        conn.execute(
            "INSERT INTO companies VALUES (?,?,?)", (r["slug"], r["slug"].title(), r["industry"])
        )
        conn.execute(
            "INSERT INTO report_records VALUES (?,?,?,?,?,?,?)",
            (
                r["record_id"],
                r["slug"],
                r["slug"].title(),
                r["industry"],
                r["year"],
                r["local_path"],
                r["sha"],
            ),
        )
        for j, (model, sections) in enumerate(r["extractions"].items()):
            extraction_id = i * 10 + j
            conn.execute(
                "INSERT INTO extraction_results VALUES (?,?,?,?)",
                (extraction_id, r["record_id"], model, "completed"),
            )
            for k, (title, text) in enumerate(sections):
                conn.execute(
                    "INSERT INTO extraction_sections "
                    "(extraction_id, section_index, section_title, section_text) VALUES (?,?,?,?)",
                    (extraction_id, k, title, text),
                )
    conn.commit()
    conn.close()


CLIMATE_PARA = (
    "Our plan cuts carbon emissions across every plant and invests in solar power for the "
    "communities that host our operations, with targets we report on every single year "
    "from now on. We also retire older boilers early and buy power from new wind farms near "
    "our largest sites."
)
OTHER_PARA = (
    "Our employees volunteered thousands of hours in local schools and food banks, and we expanded "
    "our mentoring program to every office so that new hires find support from their first "
    "day here. Managers now meet each new colleague in the first week and again after the "
    "first three months."
)


@pytest.fixture()
def sources(tmp_path: Path) -> dict[str, Path]:
    wh = tmp_path / "admin_text"
    for year in range(2009, 2026):
        (wh / str(year) / "whitehouse").mkdir(parents=True)
    _write_wh_doc(wh, "20090105", "George W. Bush", [CLIMATE_PARA])
    _write_wh_doc(wh, "20100304", "Barack Obama", [CLIMATE_PARA, OTHER_PARA])
    _write_wh_doc(wh, "20100305", "Barack Obama", [OTHER_PARA + " A different closing line."])
    _write_wh_doc(wh, "20201110", "Joseph R. Biden, Jr.", [CLIMATE_PARA + " Elect."])
    _write_wh_doc(wh, "20220316", "", [CLIMATE_PARA + " Guest."])
    _write_wh_doc(
        wh, "20220410", "Joseph R. Biden, Jr.", [CLIMATE_PARA]
    )  # duplicate text of the Obama passage
    extractor = tmp_path / "extractor"
    (extractor / "data" / "pdfs").mkdir(parents=True)
    report_text = " ".join([CLIMATE_PARA, OTHER_PARA] * 3)
    pdf_a = extractor / "data" / "pdfs" / "acme_2024.pdf"
    _write_pdf(pdf_a, report_text)
    pdf_b = extractor / "data" / "pdfs" / "beta_2024.pdf"
    _write_pdf(pdf_b, report_text)
    sha_a = hashlib.sha256(pdf_a.read_bytes()).hexdigest()
    paraphrase = (
        "The company says it will lower pollution from its factories and buy sunshine-based "
        "electricity for the towns around its sites, publishing progress against goals annually "
        "going forward. It says it will also shut ageing furnaces sooner and source electricity "
        "from fresh turbine parks."
    )
    reports = [
        {
            "record_id": "rec-a",
            "slug": "acme",
            "industry": "Energy",
            "year": 2024,
            "sha": sha_a,
            "local_path": "/old/root/extractor/data/pdfs/acme_2024.pdf",
            "extractions": {
                "gpt-5-nano": [("Climate", paraphrase)],
                "claude-sonnet-4-20250514": [
                    ("Climate", CLIMATE_PARA + "\n\n" + OTHER_PARA),
                    ("Rewrite", paraphrase),
                ],
            },
        },
        {
            "record_id": "rec-b",
            "slug": "beta",
            "industry": "Energy",
            "year": 2024,
            "sha": "0" * 64,
            "local_path": "data/pdfs/beta_2024.pdf",
            "extractions": {"gpt-5-mini": [("Climate", CLIMATE_PARA)]},
        },
    ]
    manifest = extractor / "data" / "manifest.db"
    _write_manifest(manifest, reports)
    return {"wh": wh, "extractor": extractor, "manifest": manifest, "old_prefix": tmp_path}


def test_white_house_reader_keeps_sitting_presidents_and_strips_boilerplate(
    sources: dict[str, Path],
) -> None:
    skipped: Counter[str] = Counter()
    docs = list(get_white_house_documents(sources["wh"], 2009, 2022, skipped))
    assert {d.admin for d in docs} == {"obama", "biden"}
    assert len(docs) == 3
    assert skipped == Counter(
        {
            "white_house_skipped__out_of_scope_president": 1,
            "white_house_skipped__before_term": 1,
            "white_house_skipped__no_president": 1,
        }
    )
    for doc in docs:
        assert doc.paragraphs[0] != "About Search"
        assert not any("Gerhard Peters" in p or p == "Twitter Facebook" for p in doc.paragraphs)


def test_white_house_reader_fails_closed_without_the_citation_footer(tmp_path: Path) -> None:
    _write_wh_doc(tmp_path, "20100304", "Barack Obama", [CLIMATE_PARA])
    text_file = next((tmp_path / "2010" / "whitehouse").glob("*.txt"))
    text_file.write_text("About Search\n\n" + CLIMATE_PARA, encoding="utf-8")
    with pytest.raises(ValueError, match="citation footer"):
        list(get_white_house_documents(tmp_path, 2010, 2010))


def test_corporate_reader_prefers_verbatim_model_and_leaves_out_changed_pdfs(
    sources: dict[str, Path],
) -> None:
    before = hashlib.sha256(sources["manifest"].read_bytes()).hexdigest()
    reports, manifest_sha, left_out = get_corporate_reports(
        sources["manifest"], sources["extractor"], {"/old/root/": sources["old_prefix"]}
    )
    assert manifest_sha == before == hashlib.sha256(sources["manifest"].read_bytes()).hexdigest()
    assert not Path(f"{sources['manifest']}-shm").exists()  # the source was never opened in place
    assert [r.record_id for r in reports] == ["rec-a"]
    assert reports[0].extraction_model == "claude-sonnet-4-20250514"
    assert left_out == [
        {
            "record_id": "rec-b",
            "company_slug": "beta",
            "year": "2024",
            "reason": "pdf_hash_differs_from_manifest",
        }
    ]


def test_pool_verifies_corporate_text_filters_keywords_and_deduplicates(
    sources: dict[str, Path],
) -> None:
    docs = list(get_white_house_documents(sources["wh"], 2009, 2022))
    reports, _, _ = get_corporate_reports(
        sources["manifest"], sources["extractor"], {"/old/root/": sources["old_prefix"]}
    )
    result = get_pool(docs, reports, PoolConfig(chunking=ChunkingRule(max_paragraphs=1)))
    kept_presidential = [p for p in result.kept if p.side == "presidential"]
    assert len(kept_presidential) == 1  # the Biden passage repeats the Obama text exactly
    assert result.counts["presidential_duplicates_dropped"] == 1
    corporate = [p for p in result.kept if p.side == "corporate"]
    assert len(corporate) == 1 and corporate[0].verbatim_score == 1.0
    assert result.counts["corporate_unverified__claude-sonnet-4-20250514"] == 1  # the paraphrase
    kept_docs = {p.doc_id for p in result.kept}
    assert any(p.side == "corporate" for p in result.excluded_neighbours)
    assert all(not p.keywords and p.doc_id in kept_docs for p in result.excluded_neighbours)


# ---------------------------------------------------------------------------
# Discovery sample
# ---------------------------------------------------------------------------


def _synthetic_pool() -> tuple[list[PoolPassage], list[PoolPassage]]:
    def passage(pid: str, side: str, doc: str, **meta: Any) -> PoolPassage:
        text = f"{pid} climate text"
        return PoolPassage(
            passage_id=pid,
            side=side,
            doc_id=doc,
            passage_index=0,
            text=text,  # type: ignore[arg-type]
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            n_words=3,
            keywords=meta.pop("keywords", ["climate"]),
            year=2020,
            **meta,
        )

    kept: list[PoolPassage] = []
    neighbours: list[PoolPassage] = []
    for admin in ("obama", "trump_1", "biden", "trump_2"):
        for d in range(40):
            for k in range(3):
                kept.append(
                    passage(
                        f"wh-{admin}-{d:02d}-{k}",
                        "presidential",
                        f"wh-{admin}-{d:02d}",
                        admin=admin,
                    )
                )
            neighbours.append(
                passage(
                    f"wh-{admin}-{d:02d}-x",
                    "presidential",
                    f"wh-{admin}-{d:02d}",
                    admin=admin,
                    keywords=[],
                )
            )
    for i in range(9):
        for c in range(14 - i):
            for k in range(4):
                kept.append(
                    passage(
                        f"cs-i{i}-c{c}-{k}",
                        "corporate",
                        f"cs-i{i}-c{c}",
                        industry=f"Industry {i}",
                        company_slug=f"co-{i}-{c}",
                    )
                )
            neighbours.append(
                passage(
                    f"cs-i{i}-c{c}-x",
                    "corporate",
                    f"cs-i{i}-c{c}",
                    industry=f"Industry {i}",
                    company_slug=f"co-{i}-{c}",
                    keywords=[],
                )
            )
    return kept, neighbours


def test_sample_strata_halves_subsets_and_spread() -> None:
    kept, neighbours = _synthetic_pool()
    config = SampleConfig()
    sampled, ranking = get_discovery_sample(kept, neighbours, config)
    assert [name for name, _, _ in ranking][:2] == ["Industry 0", "Industry 1"]
    assert len(sampled) == 4 * 50 + 8 * 25 + 2 * 25
    strata = Counter(s.stratum for s in sampled)
    assert strata["admin:biden"] == 50 and strata["industry:Industry 7"] == 25
    assert "industry:Industry 8" not in strata  # only the eight largest
    halves = Counter(s.half for s in sampled)
    assert halves["A"] == halves["B"] == len(sampled) // 2
    for stratum in strata:
        counts = Counter(s.half for s in sampled if s.stratum == stratum)
        assert abs(counts["A"] - counts["B"]) <= 1
    assert sum(s.in_repeat_set for s in sampled) == 50
    assert sum(s.in_inline_subset for s in sampled) == 120
    per_doc = Counter(s.passage.doc_id for s in sampled if s.stratum.startswith("admin:"))
    assert max(per_doc.values()) == 2  # 50 drawn over 40 documents: round-robin spreads them
    again, _ = get_discovery_sample(kept, neighbours, config)
    assert [s.passage.passage_id for s in again] == [s.passage.passage_id for s in sampled]


def test_sample_fails_closed_when_a_stratum_is_too_small() -> None:
    kept, neighbours = _synthetic_pool()
    with pytest.raises(ValueError, match="needs"):
        get_discovery_sample(kept, neighbours, SampleConfig(presidential_per_admin=500))
    assert get_ranked_industries([]) == []


# ---------------------------------------------------------------------------
# Open coding with a fake OpenAI client
# ---------------------------------------------------------------------------


class FakeCompletions:
    """Stands in for ``client.chat.completions.create``; scripted per call."""

    def __init__(self, bad_first: set[str]) -> None:
        self.bad_first = set(bad_first)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **request: Any) -> Any:
        self.calls.append(request)
        text = request["messages"][1]["content"]
        pid = text.split("<<<\n", 1)[1].split(" ", 1)[0]
        if pid in self.bad_first:
            self.bad_first.discard(pid)
            content = "not json"
        else:
            content = json.dumps(
                {
                    "care": "Care.",
                    "labels": ["clean energy jobs"],
                    "quote": f"{pid} talks",
                    "stance": "supports",
                    "topic_scope": "both",
                }
            )
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        return SimpleNamespace(
            id="r",
            model=request["model"],
            system_fingerprint=None,
            created=0,
            usage=usage,
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
            ],
        )


def test_response_format_is_strict_and_lens_specific() -> None:
    fmt = get_response_format(LENSES["narrative"])["json_schema"]
    assert fmt["strict"] is True
    assert set(fmt["schema"]["required"]) >= {
        "care",
        "labels",
        "quote",
        "stance",
        "topic_scope",
        "remedy",
    }
    assert fmt["schema"]["additionalProperties"] is False
    assert {p.lens_id for p in PASSES.values()} <= set(LENSES)


def test_quote_check_and_parsing() -> None:
    text = "We will “build back better” with clean energy."
    assert get_quote_is_verbatim('"build back better"', text)
    assert not get_quote_is_verbatim("build it better", text)
    code = get_code_from_content(
        json.dumps(
            {
                "care": "c",
                "labels": ["a", " "],
                "quote": "clean energy",
                "stance": "none",
                "topic_scope": "energy",
            }
        ),
        LENSES["straightforward"],
        "p1",
        "x",
        text,
    )
    assert code.labels == ["a"] and code.quote_verbatim


def test_coding_pass_retries_bad_json_and_resumes(tmp_path: Path) -> None:
    passages = [(f"p{i}", f"p{i} talks about energy") for i in range(5)]
    spec = PassSpec(pass_id="t", model="fake-model", lens_id="straightforward")
    fake = FakeCompletions(bad_first={"p2"})
    partial = tmp_path / "calls.partial.jsonl"
    codes, logs, failed = run_coding_pass(passages[:3], spec, fake, partial, max_workers=2)
    assert failed == [] and [c.passage_id for c in codes] == ["p0", "p1", "p2"]
    assert len(logs) == 4 and sum("parse_error" in e for e in logs) == 1
    assert all(c.quote_verbatim for c in codes)
    codes, logs, failed = run_coding_pass(passages, spec, fake, partial, max_workers=2)
    assert len(fake.calls) == 6  # only p3 and p4 were new
    assert [c.passage_id for c in codes] == [p for p, _ in passages] and failed == []


# ---------------------------------------------------------------------------
# Record file and the CLI on synthetic sources
# ---------------------------------------------------------------------------


def test_record_rejects_inconsistent_status() -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    base: dict[str, Any] = dict(
        step="D0",
        run_id="r",
        script="s",
        code={"commit": "a" * 40, "dirty": False},
        inputs=[],
        outputs=[{"role": "o", "path": "p", "sha256": "b" * 64}],
        config={},
        seeds=[],
        models=[],
        python_version="3",
        platform="p",
        packages={},
        started_at=now,
        finished_at=now,
    )
    with pytest.raises(ValueError):
        DiscoveryRecord.model_validate(base | {"run_status": "failed"})
    with pytest.raises(ValueError):
        DiscoveryRecord.model_validate(base | {"run_status": "completed", "unknown": 1})
    assert (
        DiscoveryRecord.model_validate(base | {"run_status": "completed"}).citability
        == "exploratory"
    )


def test_cli_pool_and_sample_write_artifacts_with_valid_records(
    sources: dict[str, Path], tmp_path: Path
) -> None:
    import run_label_discovery as cli  # pyright: ignore[reportMissingImports]

    research = tmp_path / "research"
    research.mkdir()
    assert (
        cli.main(
            [
                "--research-root",
                str(research),
                "pool",
                "--white-house-root",
                str(sources["wh"]),
                "--manifest-db",
                str(sources["manifest"]),
                "--extractor-root",
                str(sources["extractor"]),
                "--path-prefix",
                f"/old/root/={sources['old_prefix']}/",
            ]
        )
        == 0
    )
    pool_dir = (
        research / "results" / "corporate_presidential" / "label_discovery" / "d0_passage_pool"
    )
    record = get_record(pool_dir / "bundle.json")
    assert record.step == "D0" and record.run_status == "completed"
    for output in record.outputs:
        assert hashlib.sha256((research / output.path).read_bytes()).hexdigest() == output.sha256
    summary = json.loads((pool_dir / "pool_summary.json").read_text())
    assert summary["corporate_reports_left_out"][0]["record_id"] == "rec-b"

    kept, neighbours = _synthetic_pool()
    set_jsonl(pool_dir / "pool.jsonl.gz", [p.model_dump(mode="json") for p in kept])
    set_jsonl(
        pool_dir / "excluded_neighbours.jsonl.gz", [p.model_dump(mode="json") for p in neighbours]
    )
    assert cli.main(["--research-root", str(research), "sample"]) == 0
    sample_dir = pool_dir.parent / "d1_discovery_sample"
    rows = get_jsonl(sample_dir / "sample_passages.jsonl")
    assert len(rows) == 450
    ids = (sample_dir / "discovery_ids.txt").read_text().split()
    assert len(ids) == len(set(ids)) == 450
    assert get_record(sample_dir / "bundle.json").seeds == [SampleConfig().seed]


# ---------------------------------------------------------------------------
# Live: the committed discovery artifacts in REAP-research
# ---------------------------------------------------------------------------


def _live_root() -> Path:
    try:
        from research_root import get_research_root  # pyright: ignore[reportMissingImports]

        root = get_research_root(None) / "results" / "corporate_presidential" / "label_discovery"
    except RuntimeError as exc:
        pytest.skip(f"REAP-research checkout not available: {exc}")
    if not (root / "d1_discovery_sample" / "bundle.json").is_file():
        pytest.skip("label-discovery artifacts not present in the REAP-research checkout")
    return root


def test_live_records_match_their_outputs() -> None:
    root = _live_root()
    research = root.parents[2]
    for bundle in sorted(root.rglob("bundle.json")):
        record = get_record(bundle)
        if record.run_status != "completed":
            continue
        for output in record.outputs:
            path = research / output.path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == output.sha256, output.path


def test_live_sample_is_drawn_from_the_pool_with_matching_text() -> None:
    root = _live_root()
    pool = {r["passage_id"]: r for r in get_jsonl(root / "d0_passage_pool" / "pool.jsonl.gz")}
    pool |= {
        r["passage_id"]: r
        for r in get_jsonl(root / "d0_passage_pool" / "excluded_neighbours.jsonl.gz")
    }
    rows = get_jsonl(root / "d1_discovery_sample" / "sample_passages.jsonl")
    assert len(rows) == 450
    for row in rows:
        source = pool[row["passage"]["passage_id"]]
        assert source["text_sha256"] == row["passage"]["text_sha256"]
        assert hashlib.sha256(row["passage"]["text"].encode()).hexdigest() == source["text_sha256"]
        assert bool(source["keywords"]) is not row["keyword_excluded"]
