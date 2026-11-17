"""Read-only readers for the two raw sources of the corporate/presidential case study.

Presidential side: the American Presidency Project document tree, one
``<year>/whitehouse/<name>.txt`` text file per document with a one-row
``<name>.csv`` metadata file beside it. Only White House documents are read
(agency folders are skipped), and each document is assigned to an
administration from its metadata, cross-checked against the inauguration
dates.

Corporate side: the corporate sustainability report extractor's SQLite
manifest. Its section texts were cut out of each report by a language model,
and some are paraphrases rather than report text, so this module also reads
each source PDF so passages can be verified word for word.

Sources are never written. The manifest is copied to a temporary directory
and the copy is opened read-only, so not even SQLite's shared-memory index
file of the source is touched.

Exports: ``ADMINISTRATIONS``, ``WhiteHouseDocument``, ``CorporateSection``,
``CorporateReport``, ``get_white_house_documents``, ``get_corporate_reports``,
``get_pdf_ngrams``.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from label_discovery.files import get_file_sha256
from label_discovery.passages import get_normalized_words, get_paragraphs, get_word_ngrams

AdminId = Literal["obama", "trump_1", "biden", "trump_2"]

# Term windows (first day inclusive, last day inclusive). Inauguration day
# belongs to the incoming administration. None = term still running.
ADMINISTRATIONS: dict[str, tuple[date, date | None]] = {
    "obama": (date(2009, 1, 20), date(2017, 1, 20)),
    "trump_1": (date(2017, 1, 20), date(2021, 1, 20)),
    "biden": (date(2021, 1, 20), date(2025, 1, 20)),
    "trump_2": (date(2025, 1, 20), None),
}

# Presidency Project 'president' values for the four administrations in
# scope. Documents of any other president are out of scope and skipped.
_PRESIDENT_TO_ADMIN: dict[str, str] = {
    "Barack Obama": "obama",
    "Donald J. Trump (1st Term)": "trump_1",
    "Joseph R. Biden, Jr.": "biden",
    "Donald J. Trump (2nd Term)": "trump_2",
}
_OUT_OF_SCOPE_PRESIDENTS = frozenset({"George W. Bush"})

_SITE_HEADER = "About Search"
_CITATION_MARKER = "Online by Gerhard Peters and John T. Woolley"

# Extraction-model preference when a report was sectioned by more than one
# model: the model whose sections were most often verbatim in a pilot check
# comes first. Every passage is still verified against the PDF.
EXTRACTION_MODEL_PREFERENCE = ("claude-sonnet-4-20250514", "gpt-5-mini", "gpt-5-nano")


class WhiteHouseDocument(BaseModel):
    """One White House document, boilerplate removed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(..., description="Stable id: 'wh-' plus 10 hex chars of the path hash.")
    source_path: str = Field(..., description="Path of the text file relative to the tree root.")
    source_sha256: str = Field(..., description="SHA-256 of the raw text file.")
    admin: str = Field(..., description="Administration id.")
    president: str = Field(..., description="President as recorded by the Presidency Project.")
    doc_date: date = Field(..., description="Document date.")
    year: int = Field(..., description="Calendar year of the document.")
    doc_type: str = Field(..., description="Presidency Project document type.")
    title: str = Field(..., description="Document title.")
    url: str = Field(..., description="Presidency Project URL.")
    paragraphs: list[str] = Field(..., description="Body paragraphs without site boilerplate.")


class CorporateSection(BaseModel):
    """One section of a corporate report as cut out by the extractor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    section_index: int = Field(..., ge=0, description="Section order within the extraction.")
    section_title: str = Field(..., description="Section title from the extractor.")
    section_text: str = Field(..., description="Section text from the extractor (unverified).")


class CorporateReport(BaseModel):
    """One corporate report: metadata, its chosen extraction's sections, and its PDF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., description="Extractor manifest record id.")
    company_slug: str = Field(..., description="Company id.")
    company_name: str = Field(..., description="Company name.")
    industry: str = Field(..., description="Industry ('unknown' when the manifest has none).")
    year: int = Field(..., description="Report year as recorded in the manifest.")
    extraction_model: str = Field(..., description="Model whose sections are used.")
    pdf_path: Path = Field(..., description="Resolved path of the source PDF.")
    pdf_sha256: str = Field(..., description="SHA-256 of the PDF, equal to the manifest's hash.")
    sections: list[CorporateSection] = Field(..., description="Sections in order.")


def _get_admin(president: str, doc_date: date, source: str) -> tuple[str | None, str | None]:
    """Map a document to its administration, or give the reason it is out of scope.

    In scope means written by the sitting president during his term. Output:
    ``(admin, None)`` for an in-scope document, ``(None, reason)`` otherwise.
    Reasons: ``out_of_scope_president`` (a president outside the four
    administrations), ``no_president`` (documents by others, such as foreign
    leaders' addresses or governors' statements), ``before_term`` (candidate
    and president-elect material), ``after_term`` (former-president material).
    Pure.

    Raises
    ------
    ValueError : If the president value is not one the module knows.
    """
    if president in _OUT_OF_SCOPE_PRESIDENTS:
        return None, "out_of_scope_president"
    if not president:
        return None, "no_president"
    admin = _PRESIDENT_TO_ADMIN.get(president)
    if admin is None:
        raise ValueError(f"{source}: unknown president {president!r}")
    start, end = ADMINISTRATIONS[admin]
    if doc_date < start:
        return None, "before_term"
    if end is not None and doc_date > end:
        return None, "after_term"
    return admin, None


def _get_body_paragraphs(text: str, source: str) -> list[str]:
    """Strip the Presidency Project header and citation/footer paragraphs.

    Raises
    ------
    ValueError : If the citation paragraph that starts the footer is missing.
    """
    paragraphs = get_paragraphs(text)
    if paragraphs and paragraphs[0] == _SITE_HEADER:
        paragraphs = paragraphs[1:]
    for index, paragraph in enumerate(paragraphs):
        if _CITATION_MARKER in paragraph:
            return paragraphs[:index]
    raise ValueError(f"{source}: citation footer not found; boilerplate cannot be stripped safely")


def get_white_house_documents(
    root: Path, year_min: int, year_max: int, skipped: Counter[str] | None = None
) -> Iterator[WhiteHouseDocument]:
    """Yield every in-scope White House document under ``root`` for the years given.

    Inputs: the tree root (holding ``<year>/whitehouse/``), an inclusive year
    range, and an optional counter. Output: in-scope documents in sorted path
    order (see ``_get_admin`` for scope).
    Side effects: reads the text and metadata files; adds one count per
    skipped document, keyed by reason, to ``skipped`` when it is given.

    Raises
    ------
    FileNotFoundError : If a year folder or a metadata file is missing.
    ValueError : If metadata is malformed or inconsistent with the dates.
    """
    for year in range(year_min, year_max + 1):
        folder = root / str(year) / "whitehouse"
        if not folder.is_dir():
            raise FileNotFoundError(f"missing White House folder for {year}: {folder}")
        for text_path in sorted(folder.glob("*.txt")):
            rel = text_path.relative_to(root).as_posix()
            meta_path = text_path.with_suffix(".csv")
            with open(meta_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise ValueError(f"{meta_path}: expected one metadata row, found {len(rows)}")
            meta = rows[0]
            if meta["agency"] != "whitehouse":
                raise ValueError(f"{meta_path}: agency {meta['agency']!r} in a whitehouse folder")
            doc_date = datetime.strptime(text_path.name[:8], "%Y%m%d").date()
            stated = datetime.strptime(meta["date_text"].strip(), "%B %d, %Y").date()
            if stated != doc_date:
                raise ValueError(f"{rel}: file-name date {doc_date} differs from metadata date {stated}")
            admin, reason = _get_admin(meta["president"].strip(), doc_date, rel)
            if admin is None:
                if skipped is not None:
                    skipped[f"white_house_skipped__{reason}"] += 1
                continue
            raw = text_path.read_text(encoding="utf-8")
            yield WhiteHouseDocument(
                doc_id="wh-" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10],
                source_path=rel,
                source_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                admin=admin,
                president=meta["president"].strip(),
                doc_date=doc_date,
                year=doc_date.year,
                doc_type=meta["doc_type"].strip(),
                title=meta["title"].strip(),
                url=meta["url"].strip(),
                paragraphs=_get_body_paragraphs(raw, rel),
            )


def _get_manifest_copy(manifest_db: Path, workdir: Path) -> Path:
    """Copy the manifest database (and its write-ahead log files) into ``workdir``.

    Side effects: reads the source files; writes copies under ``workdir``.
    """
    for suffix in ("", "-wal", "-shm"):
        source = Path(f"{manifest_db}{suffix}")
        if source.exists():
            shutil.copy2(source, workdir / f"{manifest_db.name}{suffix}")
    return workdir / manifest_db.name


def _get_pdf_path(local_path: str, extractor_root: Path, prefix_map: dict[str, Path]) -> Path:
    """Resolve a manifest ``local_path`` on this machine. Pure (no file access)."""
    for old, new in prefix_map.items():
        if local_path.startswith(old):
            return new / local_path[len(old):]
    candidate = Path(local_path)
    return candidate if candidate.is_absolute() else extractor_root / candidate


def get_corporate_reports(
    manifest_db: Path, extractor_root: Path, prefix_map: dict[str, Path]
) -> tuple[list[CorporateReport], str, list[dict[str, str]]]:
    """Read every report that has extracted sections, choosing one extraction per report.

    Inputs: the manifest database path, the extractor project root (for
    relative PDF paths), and a map from stale absolute path prefixes in the
    manifest to their location on this machine.
    Output: the reports (sorted by record id), the SHA-256 of the manifest
    database file as read, and the reports left out with the reason. A report
    whose PDF hash differs from the manifest's recorded hash is left out: its
    sections were cut from a different file, so they cannot be verified.
    Side effects: copies the database to a temporary directory and reads the
    copy; hashes every PDF.

    Raises
    ------
    FileNotFoundError : If a report's PDF is missing.
    """
    manifest_sha = get_file_sha256(manifest_db)
    with tempfile.TemporaryDirectory(prefix="cp_manifest_") as tmp:
        copy = _get_manifest_copy(manifest_db, Path(tmp))
        if get_file_sha256(copy) != manifest_sha:
            raise ValueError("manifest database changed while it was being copied")
        conn = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        try:
            records = conn.execute(
                """
                SELECT rr.record_id, rr.company_slug, rr.company_name,
                       COALESCE(rr.industry, c.industry), rr.year, rr.local_path, rr.sha256_hash
                FROM report_records rr
                LEFT JOIN companies c USING (company_slug)
                WHERE rr.record_id IN (
                    SELECT er.record_id FROM extraction_results er
                    JOIN extraction_sections es USING (extraction_id)
                    WHERE er.status = 'completed')
                ORDER BY rr.record_id
                """
            ).fetchall()
            reports: list[CorporateReport] = []
            left_out: list[dict[str, str]] = []
            for record_id, slug, name, industry, year, local_path, sha in records:
                extractions = conn.execute(
                    """
                    SELECT er.extraction_id, er.model_name FROM extraction_results er
                    WHERE er.record_id = ? AND er.status = 'completed'
                      AND EXISTS (SELECT 1 FROM extraction_sections es
                                  WHERE es.extraction_id = er.extraction_id)
                    """,
                    (record_id,),
                ).fetchall()
                ranked = sorted(
                    extractions,
                    key=lambda row: (
                        EXTRACTION_MODEL_PREFERENCE.index(row[1])
                        if row[1] in EXTRACTION_MODEL_PREFERENCE
                        else len(EXTRACTION_MODEL_PREFERENCE),
                        row[0],
                    ),
                )
                extraction_id, model = ranked[0]
                sections = [
                    CorporateSection(section_index=index, section_title=title, section_text=text)
                    for index, title, text in conn.execute(
                        "SELECT section_index, section_title, section_text FROM extraction_sections "
                        "WHERE extraction_id = ? ORDER BY section_index",
                        (extraction_id,),
                    )
                ]
                pdf_path = _get_pdf_path(local_path or "", extractor_root, prefix_map)
                if not pdf_path.is_file():
                    raise FileNotFoundError(f"record {record_id}: PDF not found at {pdf_path}")
                pdf_sha = get_file_sha256(pdf_path)
                if sha and pdf_sha != sha:
                    left_out.append({"record_id": record_id, "company_slug": slug, "year": str(year),
                                     "reason": "pdf_hash_differs_from_manifest"})
                    continue
                reports.append(
                    CorporateReport(
                        record_id=record_id,
                        company_slug=slug,
                        company_name=name,
                        industry=industry or "unknown",
                        year=int(year),
                        extraction_model=model,
                        pdf_path=pdf_path,
                        pdf_sha256=pdf_sha,
                        sections=sections,
                    )
                )
        finally:
            conn.close()
    return reports, manifest_sha, left_out


def get_pdf_ngrams(pdf_path: Path, n: int) -> set[tuple[str, ...]]:
    """Return the word ``n``-grams of a PDF's full extracted text.

    Side effects: reads the PDF with PyMuPDF.

    Raises
    ------
    ImportError : If PyMuPDF is not installed.
    """
    import pymupdf  # Imported here so the rest of the package works without it.

    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(str(page.get_text("text")) for page in doc)
    return get_word_ngrams(get_normalized_words(text), n)
