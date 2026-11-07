"""PDF ingestion: text in reading order, headers/footers removed, footnotes,
tables and scanned pages detected, with OCR as a fallback.
"""
from __future__ import annotations

import logging
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from .models import Document, PageInfo, Unit

log = logging.getLogger(__name__)

HEADER_BAND = 0.08  # top/bottom share of the page searched for running headers/footers
SCANNED_MAX_CHARS = 40  # a page with less text than this and a large image counts as scanned
CLAUSE_START = re.compile(r"^(\(\d{1,3}\)|\d{1,2}\.(\d{1,2})?\s|\(?[a-z]\)\s|Article\s+\d+)")


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def _norm_repeat(text: str) -> str:
    """Normalise a header/footer line so page numbers do not hide repetition."""
    return re.sub(r"\d+", "#", text.strip().lower())


def _block_lines(block: dict) -> list[dict]:
    lines = []
    for line in block.get("lines", []):
        spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
        if not spans:
            continue
        # NFKC turns ligatures such as "ﬁ" into "fi", so "Deﬁnitions" matches section patterns
        text = unicodedata.normalize("NFKC", "".join(s["text"] for s in line["spans"])).strip()
        size = max(s["size"] for s in spans)
        bold = any((s.get("flags", 0) & 16) or "Bold" in s.get("font", "") for s in spans)
        lines.append({"text": text, "bbox": tuple(line["bbox"]), "size": size, "bold": bold})
    return lines


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines, repairing hyphenation at line ends."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = out + " " + line
    return re.sub(r"\s+", " ", out).strip()


def _split_paragraphs(lines: list[dict]) -> list[list[dict]]:
    """Split one PDF block into paragraphs where a new numbered clause starts."""
    paras: list[list[dict]] = []
    for line in lines:
        starts_clause = bool(CLAUSE_START.match(line["text"]))
        prev_ends = paras and paras[-1][-1]["text"].rstrip().endswith((".", ";", ":"))
        if not paras or (starts_clause and prev_ends):
            paras.append([line])
        else:
            paras[-1].append(line)
    return paras


def _reading_order(blocks: list[dict], width: float) -> tuple[list[dict], int]:
    """Order blocks for one or two text columns.

    Full-width blocks act as separators. Between two separators the left column
    is read top to bottom, then the right column.
    """
    if not blocks:
        return [], 1
    mid = width / 2
    # A block belongs to a column only if it stays on one side of the middle.
    # Centred headings ("Article 29") cross the middle and are not column blocks.
    left = [b for b in blocks if b["bbox"][2] <= mid + 2]
    right = [b for b in blocks if b["bbox"][0] >= mid - 2]

    def side_by_side(a: dict, b: dict) -> bool:
        return min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1]) > 2

    # Two columns only if left and right blocks really sit next to each other.
    pairs = sum(any(side_by_side(l, r) for r in right) for l in left)
    two_columns = pairs >= 2
    if not two_columns:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0])), 1

    def column(b: dict) -> str:
        if b in right:
            return "R"
        if b in left:
            return "L"
        return "F"

    full = sorted([b for b in blocks if column(b) == "F"], key=lambda b: b["bbox"][1])
    ordered: list[dict] = []
    top = -1.0
    for sep in full + [None]:
        bottom = sep["bbox"][1] if sep else float("inf")
        band = [b for b in blocks if column(b) != "F" and top <= b["bbox"][1] < bottom]
        ordered += sorted([b for b in band if column(b) == "L"], key=lambda b: b["bbox"][1])
        ordered += sorted([b for b in band if column(b) == "R"], key=lambda b: b["bbox"][1])
        if sep:
            ordered.append(sep)
            top = sep["bbox"][1]
    return ordered, 2


def _image_coverage(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height
    covered = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        covered += r.width * r.height
    return min(covered / area, 1.0) if area else 0.0


def _ocr_page(page: pymupdf.Page, doc_id: str, page_no: int) -> list[Unit]:
    """OCR an image-only page with Tesseract; returns one unit per OCR paragraph."""
    import pytesseract
    from PIL import Image

    zoom = 200 / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # Tesseract's automatic layout mode (psm 3) sometimes drops a clause number at the start of a line
    # ("3. Grant" -> ". Grant"). If that happens, read the page again as one uniform text block (psm 6).
    words = [w for w in data["text"] if w.strip()]
    if any(w in (".", ")") or re.match(r"^\.[A-Z]", w) for w in words[:400]) and sum(
            1 for w in words if w == ".") >= 2:
        data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    # Collect OCR words into lines, then split each OCR paragraph into clauses like digital text,
    # so "3. Grant of Patent License" starts a new unit even if Tesseract merged it into a block.
    lines: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = (data[k][i] / zoom for k in ("left", "top", "width", "height"))
        ln = lines.setdefault(key, {"words": [], "bbox": [x, y, x + w, y + h]})
        ln["words"].append(word)
        b = ln["bbox"]
        ln["bbox"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
    paras: dict[tuple, list[dict]] = {}
    for key in sorted(lines):
        ln = lines[key]
        text = unicodedata.normalize("NFKC", " ".join(ln["words"]))
        paras.setdefault(key[:2], []).append({"text": text, "bbox": tuple(ln["bbox"]), "size": 0, "bold": False})
    units = []
    for para_lines in paras.values():
        for chunk in _split_paragraphs(para_lines):
            text = _join_lines([l["text"] for l in chunk])
            if not text:
                continue
            bbox = (min(l["bbox"][0] for l in chunk), min(l["bbox"][1] for l in chunk),
                    max(l["bbox"][2] for l in chunk), max(l["bbox"][3] for l in chunk))
            units.append(Unit(doc_id, page_no, text, bbox, kind="ocr"))
    return units


def _tables(page: pymupdf.Page, doc_id: str, page_no: int) -> list[Unit]:
    """Extract tables as Markdown. Merged cells (None) inherit the cell to their left."""
    units = []
    try:
        found = page.find_tables()
    except Exception as exc:  # table detection is best effort
        log.debug("table detection failed on page %s: %s", page_no, exc)
        return units
    for table in found.tables:
        rows = [row.cells for row in table.rows]
        if not rows or len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        clean = []
        for row in rows:
            filled, last = [], ""
            for cell in row:
                if cell is None:  # merged cell: repeat the spanning cell's text
                    text = last
                else:
                    text = re.sub(r"\s+", " ", page.get_text("text", clip=pymupdf.Rect(cell))).strip()
                filled.append(text)
                last = text
            clean.append(filled)
        md = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * len(clean[0])]
        md += ["| " + " | ".join(r) + " |" for r in clean[1:]]
        units.append(Unit(doc_id, page_no, "\n".join(md), tuple(table.bbox), kind="table"))
    return units


def load_pdf(path: str | Path, doc_id: str, name: str | None = None, use_ocr: bool = True) -> Document:
    """Parse a PDF into ordered text units plus a per-page structure report."""
    path = Path(path)
    pdf = pymupdf.open(path)
    doc = Document(doc_id=doc_id, name=name or path.stem, path=str(path))

    # Pass 1: collect lines per page and find repeated header/footer lines.
    raw_pages = []
    band_counts: Counter = Counter()
    for page in pdf:
        h = page.rect.height
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        for b in blocks:
            b["lines_"] = _block_lines(b)
        raw_pages.append((page, blocks))
        seen = set()
        for b in blocks:
            for line in b["lines_"]:
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y1 < h * HEADER_BAND or y0 > h * (1 - HEADER_BAND):
                    seen.add(_norm_repeat(line["text"]))
        band_counts.update(seen)
    n_pages = len(raw_pages)
    repeated = {t for t, c in band_counts.items() if n_pages >= 3 and c >= max(2, 0.4 * n_pages)}
    doc.removed_headers = sorted(repeated)

    # Body font size, used to recognise footnotes.
    sizes = [l["size"] for _, blocks in raw_pages for b in blocks for l in b["lines_"]]
    body_size = statistics.median(sizes) if sizes else 10.0

    for idx, (page, blocks) in enumerate(raw_pages):
        page_no = idx + 1
        h, w = page.rect.height, page.rect.width
        info = PageInfo(page=page_no, chars=0)

        tables = _tables(page, doc_id, page_no)
        table_rects = [pymupdf.Rect(t.bbox) for t in tables]
        info.tables = len(tables)

        kept = []
        for b in blocks:
            lines = []
            for line in b["lines_"]:
                y0, y1 = line["bbox"][1], line["bbox"][3]
                in_band = y1 < h * HEADER_BAND or y0 > h * (1 - HEADER_BAND)
                if in_band and (_norm_repeat(line["text"]) in repeated or line["text"].strip().isdigit()):
                    continue
                if any(pymupdf.Rect(line["bbox"]).intersects(r) for r in table_rects):
                    continue
                lines.append(line)
            if lines:
                b["lines_"] = lines
                kept.append(b)

        ordered, info.columns = _reading_order(kept, w)
        units: list[Unit] = []
        for b in ordered:
            for para in _split_paragraphs(b["lines_"]):
                text = _join_lines([l["text"] for l in para])
                if not text:
                    continue
                x0 = min(l["bbox"][0] for l in para)
                y0 = min(l["bbox"][1] for l in para)
                x1 = max(l["bbox"][2] for l in para)
                y1 = max(l["bbox"][3] for l in para)
                size = max(l["size"] for l in para)
                kind = "text"
                if y0 > h * 0.6 and size < body_size * 0.9 and re.match(r"^\(?\d{1,3}\)?\s", text):
                    kind = "footnote"
                    info.footnotes += 1
                units.append(Unit(doc_id, page_no, text, (x0, y0, x1, y1), kind, size, any(l["bold"] for l in para)))
        info.chars = sum(len(u.text) for u in units)

        if info.chars < SCANNED_MAX_CHARS and _image_coverage(page) > 0.5:
            info.scanned = True
            if use_ocr and ocr_available():
                units = _ocr_page(page, doc_id, page_no)
                info.ocr_used = bool(units)
                info.chars = sum(len(u.text) for u in units)
            else:
                log.warning("%s p.%s is scanned and OCR is unavailable; page skipped", name, page_no)

        doc.units += units + tables
        doc.pages.append(info)
    pdf.close()
    return doc


def structure_report(doc: Document) -> dict:
    """Summary shown in the UI after indexing."""
    return {
        "document": doc.name,
        "pages": len(doc.pages),
        "scanned_pages": doc.scanned_pages,
        "ocr_pages": [p.page for p in doc.pages if p.ocr_used],
        "two_column_pages": [p.page for p in doc.pages if p.columns == 2],
        "tables": sum(p.tables for p in doc.pages),
        "footnotes": sum(p.footnotes for p in doc.pages),
        "removed_header_footer_lines": len(doc.removed_headers),
    }
