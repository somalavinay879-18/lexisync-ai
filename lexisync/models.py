"""Data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Unit:
    """A paragraph-like piece of text from one page, in reading order."""

    doc_id: str
    page: int  # 1-based page number as printed by a PDF viewer
    text: str
    bbox: tuple[float, float, float, float]
    kind: str = "text"  # text | table | footnote | ocr
    font_size: float = 0.0
    bold: bool = False


@dataclass
class PageInfo:
    page: int
    chars: int
    scanned: bool = False
    ocr_used: bool = False
    tables: int = 0
    footnotes: int = 0
    columns: int = 1


@dataclass
class Document:
    doc_id: str
    name: str
    path: str
    pages: list[PageInfo] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    removed_headers: list[str] = field(default_factory=list)

    @property
    def scanned_pages(self) -> list[int]:
        return [p.page for p in self.pages if p.scanned]


@dataclass
class Chunk:
    """The unit that is indexed, retrieved and cited."""

    chunk_id: str
    doc_id: str
    doc_name: str
    section: str  # e.g. "Article 83", "Recital 148", "Section 7", "Page 3"
    clause: str  # e.g. "83(5)", "7.2" or "" when the section has no numbered clauses
    title: str  # e.g. "General conditions for imposing administrative fines"
    text: str
    page_start: int
    page_end: int
    regions: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)
    kind: str = "text"

    @property
    def label(self) -> str:
        """Human-readable citation, e.g. 'GDPR p. 82, Article 83(5)'."""
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        where = self.section
        if self.clause and self.section.startswith("Article"):
            where = f"Article {self.clause}"
        elif self.clause:
            where = f"{self.section}, clause {self.clause}"
        if self.kind == "table":
            where += " (table)"
        return f"{self.doc_name} {pages}, {where}"

    def header(self) -> str:
        parts = [self.doc_name, self.section]
        if self.title:
            parts.append(self.title)
        if self.clause:
            parts.append(f"para {self.clause}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        d = dict(d)
        d["regions"] = [(int(p), tuple(b)) for p, b in d.get("regions", [])]
        return Chunk(**d)
