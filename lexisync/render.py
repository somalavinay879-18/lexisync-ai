"""Render a cited page as an image with the cited regions highlighted."""
from __future__ import annotations

import pymupdf

from .models import Chunk


def page_image(pdf_path: str, page_no: int, chunks: list[Chunk], zoom: float = 1.6) -> bytes:
    """PNG of `page_no` (1-based) with a yellow box around each cited region on that page."""
    doc = pymupdf.open(pdf_path)
    page = doc[page_no - 1]
    for chunk in chunks:
        for p, bbox in chunk.regions:
            if p == page_no:
                rect = pymupdf.Rect(bbox) + (-3, -3, 3, 3)
                page.draw_rect(rect, color=(0.95, 0.6, 0.0), fill=(1, 0.9, 0.3), fill_opacity=0.25, width=1.5)
    png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    doc.close()
    return png
