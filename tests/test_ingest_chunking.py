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
