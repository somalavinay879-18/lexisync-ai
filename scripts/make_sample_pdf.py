"""Create a small synthetic legal PDF used by the tests.

It imitates the Official Journal layout: a running header and footer, recitals,
articles with numbered paragraphs, a two-column page, a footnote, a table and
one scanned (image-only) page. It is test data only, not a real legal text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

W, H = 595, 842
BODY = 10


def _header_footer(page: pymupdf.Page, n: int) -> None:
    page.insert_text((60, 40), "L 999/1   Official Journal of the Sample Union   EN   1.1.2026", fontsize=8)
    page.insert_text((290, 815), f"{n}", fontsize=8)


def _write(page: pymupdf.Page, rect: tuple, text: str, size: float = BODY, bold: bool = False) -> None:
    page.insert_textbox(pymupdf.Rect(*rect), text, fontsize=size, fontname="hebo" if bold else "helv")


def build(path: str | Path) -> Path:
    path = Path(path)
    doc = pymupdf.open()

    # Page 1: title and recitals
    p = doc.new_page(width=W, height=H)
    _header_footer(p, 1)
    _write(p, (60, 70, 540, 110), "SAMPLE REGULATION ON DATA HANDLING", 13, True)
    y = 130
    for text in [
        "(1) The protection of natural persons in relation to the processing of data is a fundamental right.",
        "(2) This Regulation lays down rules on fines, notification duties and the liability of providers.",
    ]:
        _write(p, (60, y, 540, y + 40), text)
        y += 50
    _write(p, (60, y + 10, 540, y + 30), "HAVE ADOPTED THIS REGULATION:", BODY, True)

    # Page 2: Article 7 with numbered paragraphs and a footnote
    p = doc.new_page(width=W, height=H)
    _header_footer(p, 2)
    _write(p, (60, 70, 540, 90), "Article 7", 11, True)
    _write(p, (60, 92, 540, 112), "Liability cap", 11, True)
    _write(p, (60, 120, 540, 170), "1. The total liability of a provider under this Regulation shall be limited as set out in this Article.")
    _write(p, (60, 180, 540, 230), "2. The liability cap shall be EUR 2 000 000 per calendar year for all claims combined.")
    _write(p, (60, 240, 540, 290), "3. The cap in paragraph 2 shall not apply to wilful misconduct.")
    _write(p, (60, 760, 540, 790), "(1) OJ C 123, 1.1.2025, p. 1.", 7)

    # Page 3: two-column page with Article 9 (left) and Article 10 (right)
    p = doc.new_page(width=W, height=H)
    _header_footer(p, 3)
    _write(p, (60, 70, 280, 90), "Article 9", 11, True)
    _write(p, (60, 92, 280, 112), "Indemnities", 11, True)
    _write(p, (60, 120, 280, 200), "1. The provider shall indemnify the customer for third-party claims up to EUR 1 500 000.")
    _write(p, (60, 210, 280, 290), "2. A separate data-breach indemnity of EUR 2 000 000 shall apply.")
    _write(p, (310, 70, 540, 90), "Article 10", 11, True)
    _write(p, (310, 92, 540, 112), "Notification", 11, True)
    _write(p, (310, 120, 540, 200), "1. A breach shall be notified to the authority not later than 72 hours after becoming aware of it.")
    _write(p, (310, 210, 540, 290), "2. Data subjects shall be informed without undue delay where the breach is likely to result in a high risk.")

    # Page 4: Article 11 with a table of fines
    p = doc.new_page(width=W, height=H)
    _header_footer(p, 4)
    _write(p, (60, 70, 540, 90), "Article 11", 11, True)
    _write(p, (60, 92, 540, 112), "Administrative fines", 11, True)
    _write(p, (60, 120, 540, 160), "1. Fines shall be imposed according to the following table.")
    rows = [("Tier", "Maximum fine", "Share of turnover"), ("Lower", "EUR 10 000 000", "2 %"), ("Higher", "EUR 20 000 000", "4 %")]
    x0, y0, cw, rh = 60, 180, 160, 28
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p.insert_textbox(rect + (4, 4, -4, -4), cell, fontsize=9, fontname="helv")

    # Page 5: scanned page (text rendered into an image only)
    src = pymupdf.open()
    sp = src.new_page(width=W, height=H)
    _write(sp, (60, 70, 540, 90), "Article 12", 11, True)
    _write(sp, (60, 92, 540, 112), "Governing law", 11, True)
    _write(sp, (60, 120, 540, 170), "1. This Regulation shall be interpreted in accordance with the law of the Sample Union.")
    pix = sp.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    p = doc.new_page(width=W, height=H)
    p.insert_image(p.rect, pixmap=pix)

    doc.save(path)
    return path


if __name__ == "__main__":
    out = build(sys.argv[1] if len(sys.argv) > 1 else "data/sample/sample_regulation.pdf")
    print(f"Wrote {out}")
