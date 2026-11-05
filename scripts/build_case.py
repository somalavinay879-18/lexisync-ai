"""Create (or rebuild) a case from the PDFs in data/docs and index it.

    python scripts/build_case.py --case "GDPR ePrivacy"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexisync.pipeline import Case  # noqa: E402

NAMES = {"gdpr": "GDPR", "eprivacy": "ePrivacy Directive"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="GDPR ePrivacy")
    ap.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    args = ap.parse_args()
    pdfs = sorted(Path(args.docs).glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in data/docs. Run: make docs")
    case = Case(args.case)
    for pdf in pdfs:
        case.add_path(pdf)
    structure = case.build(names=NAMES)
    print(json.dumps([{k: v for k, v in s.items() if k != "sections"} for s in structure], indent=2))
    for s in structure:
        print(f"{s['document']}: {len(s['sections'])} sections, e.g. {s['sections'][:5]} ... {s['sections'][-3:]}")


if __name__ == "__main__":
    main()
