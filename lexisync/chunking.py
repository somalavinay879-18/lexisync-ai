"""Clause-aware chunking.

Legal texts are split where the law itself is split: recitals, articles and
their numbered paragraphs (EU acts), or numbered sections and clauses
(contracts). Each chunk keeps its page numbers and on-page regions so every
answer can cite "GDPR p. 82, Article 83(5)" and highlight that region.
"""
from __future__ import annotations

import re

from .models import Chunk, Document, Unit

MAX_CHARS = 1500  # long paragraphs are split at points (a), (b), ... or sentences
MIN_CHARS = 120  # tiny fragments are merged into the previous chunk of the same clause

ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)\s*$", re.I)
ARTICLE_INLINE = re.compile(r"^Article\s+(\d+[a-z]?)\s+([A-Z].{2,120})$")
CHAPTER = re.compile(r"^(CHAPTER|TITLE|PART|SECTION)\s+([IVXLC\d]+)\b", re.I)
RECITAL = re.compile(r"^\((\d{1,3})\)\s+\S")
PARAGRAPH = re.compile(r"^(\d{1,2})\.\s+\S")
DEFINITION = re.compile(r"^\((\d{1,3})\)\s+\S")
CONTRACT_SECTION = re.compile(r"^(\d{1,2})\.?\s+([A-Z][A-Za-z ,&/'-]{2,80})$")
CONTRACT_CLAUSE = re.compile(r"^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?\s+\S")
POINT = re.compile(r"^\(([a-z]{1,2}|[ivx]{1,4})\)\s")


class _Builder:
    def __init__(self, doc: Document):
        self.doc = doc
        self.chunks: list[Chunk] = []
        self.section = "Preamble"
        self.title = ""
        self.clause = ""
        self.units: list[Unit] = []
        self.expect_title = False
        self.expect_chapter_title = False
        self.in_articles = False
        self.has_contract_sections = False

    def flush(self) -> None:
        if not self.units:
            return
        text = "\n".join(u.text for u in self.units).strip()
        if not text:
            self.units = []
            return
        regions: list[tuple[int, tuple]] = []
        for u in self.units:
            regions.append((u.page, u.bbox))
        prev = self.chunks[-1] if self.chunks else None
        if (
            prev
            and prev.doc_id == self.doc.doc_id
            and prev.section == self.section
            and prev.clause == self.clause
            and prev.kind == "text"
            and (len(text) < MIN_CHARS or len(prev.text) < MIN_CHARS)
            and len(prev.text) + len(text) < MAX_CHARS
        ):
            prev.text += "\n" + text
            prev.page_end = max(prev.page_end, self.units[-1].page)
            prev.regions += regions
        else:
            for part in _split_long(text):
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{self.doc.doc_id}-{len(self.chunks):04d}",
                        doc_id=self.doc.doc_id,
                        doc_name=self.doc.name,
                        section=self.section,
                        clause=self.clause,
                        title=self.title,
                        text=part,
                        page_start=self.units[0].page,
                        page_end=self.units[-1].page,
                        regions=regions,
                    )
                )
        self.units = []

    def start(self, section: str | None = None, clause: str = "", title: str | None = None) -> None:
        self.flush()
        if section is not None:
            self.section = section
            self.title = title or ""
        self.clause = clause

    def add_table(self, unit: Unit) -> None:
        self.flush()
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.doc.doc_id}-{len(self.chunks):04d}",
                doc_id=self.doc.doc_id,
                doc_name=self.doc.name,
                section=self.section,
                clause=self.clause,
                title=self.title,
                text=unit.text,
                page_start=unit.page,
                page_end=unit.page,
                regions=[(unit.page, unit.bbox)],
                kind="table",
            )
        )

    def add_footnote(self, unit: Unit) -> None:
        self.flush()
        self.chunks.append(
            Chunk(
                chunk_id=f"{self.doc.doc_id}-{len(self.chunks):04d}",
                doc_id=self.doc.doc_id,
                doc_name=self.doc.name,
                section=f"Footnote p. {unit.page}",
                clause="",
                title=f"(belongs to {self.section})",
                text=unit.text,
                page_start=unit.page,
                page_end=unit.page,
                regions=[(unit.page, unit.bbox)],
                kind="footnote",
            )
        )


def _split_long(text: str) -> list[str]:
    """Split a very long clause at points (a), (b)... or, failing that, at sentences."""
    if len(text) <= MAX_CHARS:
        return [text]
    lines = text.split("\n")
    pieces = lines if len(lines) > 1 else re.split(r"(?<=[.;])\s+(?=\(?[a-zA-Z0-9])", text)
    out, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > MAX_CHARS:
            out.append(cur.strip())
            cur = ""
        cur += ("\n" if cur else "") + piece
    if cur.strip():
        out.append(cur.strip())
    # a single huge sentence is cut hard as a last resort
    final = []
    for part in out:
        while len(part) > MAX_CHARS * 1.5:
            final.append(part[:MAX_CHARS])
            part = part[MAX_CHARS:]
        final.append(part)
    return final


def chunk_document(doc: Document) -> list[Chunk]:
    b = _Builder(doc)
    for unit in doc.units:
        text = unit.text.strip()
        if unit.kind == "table":
            b.add_table(unit)
            continue
        if unit.kind == "footnote":
            b.add_footnote(unit)
            continue

        # "Article 83" on its own line, with the title on the next line
        m = ARTICLE.match(text)
        if m:
            b.in_articles = True
            b.start(section=f"Article {m.group(1)}")
            b.expect_title = True
            continue
        m = ARTICLE_INLINE.match(text)
        if m and len(text) < 140:
            b.in_articles = True
            b.start(section=f"Article {m.group(1)}", title=m.group(2))
            continue
        if b.expect_chapter_title:
            b.expect_chapter_title = False
            if len(text) < 140 and not PARAGRAPH.match(text):
                continue  # chapter / section title, e.g. "Remedies, liability and penalties"
        if b.expect_title:
            b.expect_title = False
            if len(text) < 140 and not PARAGRAPH.match(text) and not text.endswith("."):
                b.title = text
                continue
        if CHAPTER.match(text) and len(text) < 120:
            b.flush()  # chapter headings only give context; they are not chunks
            b.expect_chapter_title = len(text.split()) <= 3
            continue

        if b.in_articles:
            m = PARAGRAPH.match(text) or DEFINITION.match(text)
            if m:
                article = b.section.split()[-1]
                b.start(clause=f"{article}({m.group(1)})")
            b.units.append(unit)
            continue

        # Preamble of an EU act: recitals "(1) ..."
        m = RECITAL.match(text)
        if m and not b.has_contract_sections:
            b.start(section=f"Recital {m.group(1)}")
            b.units.append(unit)
            continue

        # Contracts: "7. Liability" headings and "7.2 ..." clauses
        m = CONTRACT_SECTION.match(text)
        if m and len(text) < 90:
            b.has_contract_sections = True
            b.start(section=f"Section {m.group(1)}", title=m.group(2).strip())
            continue
        m = CONTRACT_CLAUSE.match(text)
        if m:
            b.has_contract_sections = True
            section = f"Section {m.group(1).split('.')[0]}"
            if section != b.section:
                b.start(section=section, clause=m.group(1))
            else:
                b.start(clause=m.group(1))
            b.units.append(unit)
            continue

        # Unstructured text: keep paragraphs together, cut by size per page
        if b.section == "Preamble" and not b.chunks and unit.page > 1:
            b.start(section=f"Page {unit.page}")
        b.units.append(unit)
        if sum(len(u.text) for u in b.units) > MAX_CHARS * 0.8:
            b.flush()
    b.flush()
    return b.chunks
