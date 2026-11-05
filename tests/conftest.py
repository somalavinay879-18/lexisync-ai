import os
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["LEXISYNC_OFFLINE"] = "1"  # tests never download models

from make_sample_pdf import build  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "sample.pdf")


def _page(doc, n, blocks):
    p = doc.new_page(width=595, height=842)
    p.insert_text((60, 40), "4.5.2016   EN   Official Journal of the European Union   L 119/" + str(n), fontsize=8)
    y = 70
    for text, size, bold in blocks:
        rect = pymupdf.Rect(60, y, 535, y + 200)
        used = p.insert_textbox(rect, text, fontsize=size, fontname="hebo" if bold else "helv")
        height = 200 - used if used > 0 else 60
        y += height + 8
    return p


@pytest.fixture(scope="session")
def gdpr_like_pdf(tmp_path_factory) -> Path:
    """Four pages laid out like the GDPR in the Official Journal (excerpts)."""
    doc = pymupdf.open()
    _page(doc, 1, [
        ("(1) The protection of natural persons in relation to the processing of personal data is a fundamental right.", 9, False),
        ("HAVE ADOPTED THIS REGULATION:", 9, True),
        ("CHAPTER I", 9, False),
        ("General provisions", 9, True),
        ("Article 4", 9, False),
        ("Definitions", 9, True),
        ("For the purposes of this Regulation:", 9, False),
        ("(1) 'personal data' means any information relating to an identified or identifiable natural person;", 9, False),
        ("(11) 'consent' of the data subject means any freely given, specific, informed and unambiguous indication "
         "of the data subject's wishes by which he or she, by a statement or by a clear affirmative action, signifies "
         "agreement to the processing of personal data relating to him or her;", 9, False),
    ])
    _page(doc, 2, [
        ("CHAPTER VIII", 9, False),
        ("Remedies, liability and penalties", 9, True),
        ("Article 83", 9, False),
        ("General conditions for imposing administrative fines", 9, True),
        ("4. Infringements of the following provisions shall, in accordance with paragraph 2, be subject to "
         "administrative fines up to 10 000 000 EUR, or in the case of an undertaking, up to 2 % of the total "
         "worldwide annual turnover of the preceding financial year, whichever is higher:", 9, False),
        ("(a) the obligations of the controller and the processor pursuant to Articles 8, 11, 25 to 39 and 42 and 43;", 9, False),
    ])
    _page(doc, 3, [
        ("5. Infringements of the following provisions shall, in accordance with paragraph 2, be subject to "
         "administrative fines up to 20 000 000 EUR, or in the case of an undertaking, up to 4 % of the total "
         "worldwide annual turnover of the preceding financial year, whichever is higher:", 9, False),
        ("(a) the basic principles for processing, including conditions for consent, pursuant to Articles 5, 6, 7 and 9;", 9, False),
        ("(b) the data subjects' rights pursuant to Articles 12 to 22;", 9, False),
    ])
    _page(doc, 4, [
        ("Article 33", 9, False),
        ("Notification of a personal data breach to the supervisory authority", 9, True),
        ("1. In the case of a personal data breach, the controller shall without undue delay and, where feasible, "
         "not later than 72 hours after having become aware of it, notify the personal data breach to the "
         "supervisory authority competent in accordance with Article 55.", 9, False),
    ])
    path = tmp_path_factory.mktemp("gdpr") / "gdpr.pdf"
    doc.save(path)
    return path
