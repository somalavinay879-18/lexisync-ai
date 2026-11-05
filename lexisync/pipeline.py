"""A case = a folder with 2-5 PDFs, their parsed structure and the search index."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import requests
import yaml

from .answer import Answer, answer_question, closed_book
from .chunking import chunk_document
from .ingest import load_pdf, structure_report
from .retrieve import Index, Retriever

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
CASES_DIR = ROOT / CONFIG.get("cases_dir", "data/cases")


def set_seeds(seed: int = CONFIG.get("seed", 42)) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "case"


_MODELS: dict = {}


def default_encoder():
    if os.getenv("LEXISYNC_OFFLINE"):
        from .embed import HashingEncoder

        return HashingEncoder()
    if "encoder" not in _MODELS:
        from .embed import SentenceEncoder

        _MODELS["encoder"] = SentenceEncoder(CONFIG["embedding_model"])
    return _MODELS["encoder"]


def default_reranker():
    if os.getenv("LEXISYNC_OFFLINE"):
        from .embed import KeywordReranker

        return KeywordReranker()
    if "reranker" not in _MODELS:
        from .embed import CrossReranker

        _MODELS["reranker"] = CrossReranker(CONFIG["reranker_model"])
    return _MODELS["reranker"]


class Case:
    def __init__(self, name: str, root: Path = CASES_DIR):
        self.dir = root / slug(name)
        self.docs_dir = self.dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        meta_name = self.dir / "name.txt"
        if not meta_name.exists():
            meta_name.write_text(name)
        self.name = meta_name.read_text().strip()
        self.index: Index | None = None
        self.retriever: Retriever | None = None
        self.structure: list[dict] = []
        meta = self.dir / "structure.json"
        if meta.exists():
            self.structure = json.loads(meta.read_text())

    @staticmethod
    def list_cases(root: Path = CASES_DIR) -> list[str]:
        return sorted(p.name for p in root.glob("*") if p.is_dir()) if root.exists() else []

    @property
    def pdfs(self) -> list[Path]:
        return sorted(self.docs_dir.glob("*.pdf"))

    def add_file(self, filename: str, data: bytes) -> Path:
        path = self.docs_dir / (slug(Path(filename).stem) + ".pdf")
        path.write_bytes(data)
        return path

    def add_path(self, src: str | Path) -> Path:
        dst = self.docs_dir / (slug(Path(src).stem) + ".pdf")
        shutil.copy(src, dst)
        return dst

    def add_url(self, url: str) -> Path:
        r = requests.get(url, timeout=60, headers={"User-Agent": "LexiSyncAI student project"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError("The URL did not return a PDF file.")
        name = Path(url.split("?")[0]).stem or "document"
        return self.add_file(name, r.content)

    def remove(self, filename: str) -> None:
        (self.docs_dir / filename).unlink(missing_ok=True)

    def build(self, encoder=None, use_ocr: bool = True, names: dict[str, str] | None = None) -> list[dict]:
        """Parse every PDF, chunk it and build the BM25 + dense index."""
        set_seeds()
        names = names or {}
        chunks, structure = [], []
        for pdf in self.pdfs:
            doc_id = pdf.stem
            display = names.get(doc_id) or doc_id.replace("-", " ").upper()[:30]
            doc = load_pdf(pdf, doc_id=doc_id, name=display, use_ocr=use_ocr)
            doc_chunks = chunk_document(doc)
            report = structure_report(doc)
            report["file"] = pdf.name
            report["chunks"] = len(doc_chunks)
            report["sections"] = sorted({c.section for c in doc_chunks if c.kind == "text"},
                                        key=_section_sort_key)
            structure.append(report)
            chunks += doc_chunks
        self.index = Index(chunks, encoder or default_encoder())
        self.index.save(self.dir / "index.pkl")
        (self.dir / "structure.json").write_text(json.dumps(structure, indent=2))
        self.structure = structure
        self.retriever = None
        return structure

    def load(self, encoder=None) -> bool:
        path = self.dir / "index.pkl"
        if not path.exists():
            return False
        self.index = Index.load(path, encoder or default_encoder())
        return True

    def get_retriever(self, reranker=None) -> Retriever:
        if self.index is None and not self.load():
            raise RuntimeError("Index the documents first.")
        if self.retriever is None or reranker is not None:
            self.retriever = Retriever(self.index, reranker or default_reranker(),
                                       candidates=CONFIG.get("rerank_candidates", 30))
        return self.retriever

    def ask(self, question: str, llm, **kwargs) -> Answer:
        retriever = self.get_retriever()
        params = {
            "k": CONFIG.get("top_k", 6),
            "abstain_threshold": CONFIG.get("abstain_threshold", 0.1),
            "max_attempts": CONFIG.get("max_verify_attempts", 2),
        }
        params.update(kwargs)
        docs = [s["document"] for s in self.structure] or None
        if params.pop("no_retrieval", False):
            return closed_book(question, llm)
        return answer_question(question, retriever, llm, doc_names=docs, **params)


def _section_sort_key(s: str) -> tuple:
    m = re.match(r"(\D+?)\s*(\d+)", s)
    order = {"Recital": 0, "Article": 1, "Section": 2, "Page": 3}
    if m:
        return (order.get(m.group(1).strip(), 9), int(m.group(2)))
    return (-1, 0)
