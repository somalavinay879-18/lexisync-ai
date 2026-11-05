"""Download the public legal PDFs used in the evaluation (from EUR-Lex).

If EUR-Lex blocks the automatic download, the script tells you which links to
open in the browser and where to save the files.
"""
from __future__ import annotations

from pathlib import Path

import requests

DOCS = {
    "gdpr.pdf": ("General Data Protection Regulation (EU) 2016/679",
                 "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"),
    "eprivacy.pdf": ("ePrivacy Directive 2002/58/EC (original text)",
                     "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32002L0058"),
}

OUT = Path(__file__).resolve().parent.parent / "data" / "docs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, (title, url) in DOCS.items():
        target = OUT / filename
        if target.exists() and target.read_bytes()[:4] == b"%PDF":
            print(f"ok       {filename} (already there)")
            continue
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 LexiSyncAI student project"})
            if r.ok and r.content[:4] == b"%PDF":
                target.write_bytes(r.content)
                print(f"saved    {filename}  ({len(r.content) // 1024} KB)  {title}")
                continue
        except requests.RequestException:
            pass
        missing.append((filename, title, url))
    if missing:
        print("\nCould not download automatically. Open these links in your browser and save the PDFs as:")
        for filename, title, url in missing:
            print(f"  {OUT / filename}\n    {title}\n    {url}")


if __name__ == "__main__":
    main()
