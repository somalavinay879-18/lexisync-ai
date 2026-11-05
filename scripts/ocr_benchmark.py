"""OCR accuracy on a sample (bonus feature 1).

Digital pages give us exact ground truth: render a page to an image (as a
scanner would), OCR it, and compare with the PDF's own text layer.
Character accuracy = 1 - normalised Levenshtein distance.

    python scripts/ocr_benchmark.py data/docs/gdpr.pdf --pages 5
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
from rapidfuzz.distance import Levenshtein


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def benchmark(pdf_path: str, pages: int = 5, dpi: int = 200) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    step = max(1, len(doc) // pages)
    rows = []
    for i in range(0, len(doc), step)[:pages]:
        page = doc[i]
        truth = _clean(page.get_text())
        if len(truth) < 200:
            continue
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        ocr = _clean(pytesseract.image_to_string(img))
        acc = 1 - Levenshtein.normalized_distance(truth, ocr)
        rows.append({"page": i + 1, "chars": len(truth), "char_accuracy": round(acc, 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--out", default="reports/ocr_benchmark.md")
    args = ap.parse_args()
    rows = benchmark(args.pdf, args.pages)
    mean = statistics.mean(r["char_accuracy"] for r in rows) if rows else 0
    lines = ["# OCR benchmark", "", f"Document: {Path(args.pdf).name}, Tesseract, 200 dpi.", "",
             "| page | characters | character accuracy |", "|---|---|---|"]
    lines += [f"| {r['page']} | {r['chars']} | {r['char_accuracy']:.1%} |" for r in rows]
    lines += ["", f"**Mean character accuracy: {mean:.1%}**", "",
              "Fallback: if Tesseract is not installed, scanned pages are flagged in the structure view "
              "and skipped instead of producing wrong text."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
