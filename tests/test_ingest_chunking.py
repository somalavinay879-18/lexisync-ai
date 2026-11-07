import pymupdf

from lexisync.chunking import chunk_document
from lexisync.ingest import load_pdf, ocr_available, structure_report


def test_headers_and_footers_are_removed(sample_pdf):
    doc = load_pdf(sample_pdf, "sample", "Sample")
    assert all("Official Journal" not in u.text for u in doc.units)
    assert doc.removed_headers


def test_two_column_reading_order(sample_pdf):
    doc = load_pdf(sample_pdf, "sample", "Sample")
    page3 = [u.text for u in doc.units if u.page == 3]
    # the whole left column (Article 9) comes before the right column (Article 10)
    assert page3.index("Article 9") < page3.index("Article 10")
    assert any(t.startswith("2. A separate data-breach") for t in page3[: page3.index("Article 10")])
    assert doc.pages[2].columns == 2


def test_scanned_page_is_flagged(sample_pdf):
    doc = load_pdf(sample_pdf, "sample", "Sample")
    assert doc.scanned_pages == [5]
    if ocr_available():
        assert any("Governing law" in u.text for u in doc.units if u.page == 5)


def test_scanned_page_without_ocr_is_skipped(sample_pdf):
    doc = load_pdf(sample_pdf, "sample", "Sample", use_ocr=False)
    assert doc.scanned_pages == [5]
    assert not [u for u in doc.units if u.page == 5]


def test_footnote_and_table_detected(sample_pdf):
    doc = load_pdf(sample_pdf, "sample", "Sample")
    report = structure_report(doc)
    assert report["footnotes"] == 1
    assert report["tables"] == 1
    table = next(u for u in doc.units if u.kind == "table")
    assert "| Higher | EUR 20 000 000 | 4 % |" in table.text


def test_clause_aware_chunks_keep_page_and_clause(sample_pdf):
    chunks = chunk_document(load_pdf(sample_pdf, "sample", "Sample"))
    cap = next(c for c in chunks if "2 000 000 per calendar year" in c.text)
    assert (cap.section, cap.clause, cap.page_start) == ("Article 7", "7(2)", 2)
    assert cap.title == "Liability cap"
    assert cap.label == "Sample p. 2, Article 7(2)"
    assert cap.regions and cap.regions[0][0] == 2


def test_gdpr_layout(gdpr_like_pdf):
    chunks = chunk_document(load_pdf(gdpr_like_pdf, "gdpr", "GDPR"))
    by_clause = {c.clause: c for c in chunks if c.clause}
    assert by_clause["4(11)"].text.startswith("(11) 'consent'")
    assert by_clause["83(4)"].page_start == 2 and "10 000 000 EUR" in by_clause["83(4)"].text
    assert "Articles 8, 11, 25 to 39" in by_clause["83(4)"].text  # point (a) stays with its paragraph
    assert by_clause["83(5)"].page_start == 3 and by_clause["83(5)"].section == "Article 83"
    assert by_clause["83(5)"].title == "General conditions for imposing administrative fines"
    assert by_clause["33(1)"].label == "GDPR p. 4, Article 33(1)"
    # chapter titles are not glued onto the previous article
    assert all("Remedies, liability and penalties" not in c.text for c in chunks)
    assert any(c.section == "Recital 1" for c in chunks)


def _inline_contract(path):
    """A contract whose section headings sit inline with the text ("1. Definitions. ..."), as in
    Apache-style licence agreements."""
    doc = pymupdf.open()
    texts = [
        ["Contributor Licence Agreement", "Thank you for your interest. Please read the terms below."],
        ["1. Definitions. \"You\" means the copyright owner that is making this agreement.",
         "2. Grant of Copyright License. You grant a perpetual, worldwide, royalty-free licence.",
         "3. You represent that you are legally entitled to grant the above licence."],
    ]
    for page in texts:
        p = doc.new_page(width=595, height=842)
        y = 80
        for t in page:
            p.insert_textbox(pymupdf.Rect(60, y, 535, y + 60), t, fontsize=11)
            y += 70
    doc.save(path)
    return path


def test_inline_contract_sections(tmp_path):
    chunks = chunk_document(load_pdf(_inline_contract(tmp_path / "icla.pdf"), "icla", "ICLA"))
    by_section = {c.section: c for c in chunks}
    assert by_section["Section 1"].title == "Definitions"
    assert by_section["Section 2"].title == "Grant of Copyright License"
    assert "Section 3" in by_section                                  # numbered section without heading
    assert by_section["Section 2"].label == "ICLA p. 2, Section 2"


def test_unstructured_pages_get_one_section_per_page(tmp_path):
    doc = pymupdf.open()
    for n in range(3):
        doc.new_page(width=595, height=842).insert_textbox(
            pymupdf.Rect(60, 80, 535, 400), f"Plain prose on page {n + 1} without any numbering at all.", fontsize=11)
    doc.save(tmp_path / "prose.pdf")
    sections = {c.section for c in chunk_document(load_pdf(tmp_path / "prose.pdf", "p", "Prose"))}
    assert {"Page 2", "Page 3"} <= sections


def test_ligatures_are_normalised():
    from lexisync.ingest import _block_lines
    block = {"lines": [{"bbox": (0, 0, 10, 10), "spans": [{"text": "1. Deﬁnitions.", "size": 10, "flags": 0, "font": "x"}]}]}
    assert _block_lines(block)[0]["text"] == "1. Definitions."
