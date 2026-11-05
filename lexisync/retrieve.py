"""Hybrid retrieval: BM25 + dense embeddings, merged with Reciprocal Rank
Fusion, then reranked with a cross-encoder.

Modes (used by the evaluation to compare configurations):
  bm25           keyword search only
  dense          embedding search only
  hybrid         BM25 + dense merged with RRF
  hybrid_rerank  hybrid, then cross-encoder reranking
  hybrid_rerank_mq  hybrid_rerank plus a query planner: a two-part question is
                 split into sub-questions and the best passage for each part is
                 guaranteed a place in the top k (multi-hop questions)
"""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mq")
RRF_K = 60

_STOP = set(
    "a an the of and or to in on at by for with from is are was be been shall may which what who when where how "
    "that this these those it its as under does do not no any all".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words plus legal references kept intact: '7.2', '83(5)', '83'."""
    text = text.lower()
    refs = re.findall(r"\d+(?:\.\d+)*(?:\(\d+\))+|\d+\.\d+(?:\.\d+)*", text)
    words = re.findall(r"[a-zà-ÿ]+|\d+", text)
    expanded = []
    for r in refs:
        expanded.append(r)
        expanded.append(re.split(r"[.(]", r)[0])  # '83(5)' also matches '83'
    return [w for w in words if w not in _STOP] + expanded


def bm25_text(c: Chunk) -> str:
    ref = f"{c.section} {c.clause} article {c.clause}" if c.clause else c.section
    return f"{ref} {c.title} {c.text}"


def dense_text(c: Chunk) -> str:
    return f"{c.header()}\n{c.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    stage_scores: dict = field(default_factory=dict)


class Index:
    """BM25 and dense index over all chunks of one case."""

    def __init__(self, chunks: list[Chunk], encoder, embeddings: np.ndarray | None = None):
        self.chunks = chunks
        self.encoder = encoder
        self.bm25 = BM25Okapi([tokenize(bm25_text(c)) for c in chunks]) if chunks else None
        if embeddings is None:
            embeddings = encoder.encode_passages([dense_text(c) for c in chunks])
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": [c.to_dict() for c in self.chunks], "embeddings": self.embeddings,
                         "encoder": getattr(self.encoder, "name", "")}, f)

    @staticmethod
    def load(path: str | Path, encoder) -> "Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        chunks = [Chunk.from_dict(d) for d in data["chunks"]]
        if data.get("encoder") != getattr(encoder, "name", ""):
            return Index(chunks, encoder)  # different encoder: re-embed
        return Index(chunks, encoder, data["embeddings"])

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if len(self.chunks) == 0:
            return []
        q = self.encoder.encode_query(query)
        sims = self.embeddings @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


_SPLIT = re.compile(r",?\s+and\s+(?=(?:when|what|which|how|where|who|whether|why|does|do|is|are|can|must|on what|under what|in what)\b)|;\s+|\?\s+(?=\S)", re.I)


def plan_subqueries(question: str) -> list[str]:
    """Split a two-part question into sub-questions.

    "When must a breach be notified, and when must data subjects be informed?"
    -> ["When must a breach be notified", "when must data subjects be informed?"]
    A vague part such as "and what is its maximum?" inherits the topic words of
    the first part so that it still finds the right passage.
    """
    parts = [p.strip(" ,") for p in _SPLIT.split(question) if p and len(p.strip()) > 8]
    if len(parts) < 2:
        return []
    topic = " ".join(w for w in re.findall(r"[A-Za-z][\w-]{3,}", parts[0]) if w.lower() not in _STOP)

    def vague(part: str) -> bool:
        words = [w for w in re.findall(r"[a-z]+", part.lower()) if w not in _STOP]
        return len(words) < 4 or bool(re.search(r"\b(its|it|they|them|this|that)\b", part.lower()))

    return [parts[0]] + [f"{p} ({topic})" if vague(p) else p for p in parts[1:]]


def rrf(*rankings: list[tuple[int, float]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum over rankings of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, index: Index, reranker=None, candidates: int = 30):
        self.index = index
        self.reranker = reranker
        self.candidates = candidates
        self.last_timings: dict = {}

    def search(self, query: str, k: int = 6, mode: str = "hybrid_rerank") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid_rerank_mq":
            return self._search_planned(query, k)
        t0 = time.perf_counter()
        n = self.candidates
        chunks = self.index.chunks
        if mode == "bm25":
            ranked = self.index.bm25_search(query, n)
            hits = [Hit(chunks[i], s, {"bm25": s}) for i, s in ranked]
        elif mode == "dense":
            ranked = self.index.dense_search(query, n)
            hits = [Hit(chunks[i], s, {"dense": s}) for i, s in ranked]
        else:
            bm = self.index.bm25_search(query, n)
            de = self.index.dense_search(query, n)
            bm_rank = {i: r for r, (i, _) in enumerate(bm, 1)}
            de_rank = {i: r for r, (i, _) in enumerate(de, 1)}
            fused = rrf(bm, de)[:n]
            hits = [Hit(chunks[i], s, {"rrf": s, "bm25_rank": bm_rank.get(i), "dense_rank": de_rank.get(i)})
                    for i, s in fused]
        t1 = time.perf_counter()
        if mode == "hybrid_rerank" and self.reranker is not None and hits:
            scores = self.reranker.score(query, [dense_text(h.chunk) for h in hits])
            for h, s in zip(hits, scores):
                h.stage_scores["rerank"] = s
                h.score = s
            hits.sort(key=lambda h: -h.score)
        t2 = time.perf_counter()
        self.last_timings = {"retrieve_ms": (t1 - t0) * 1000, "rerank_ms": (t2 - t1) * 1000}
        return hits[:k]

    def _search_planned(self, query: str, k: int) -> list[Hit]:
        """Query planner for multi-hop questions (see plan_subqueries)."""
        t0 = time.perf_counter()
        subs = plan_subqueries(query)
        main = self.search(query, k=self.candidates, mode="hybrid_rerank")
        if not subs:
            self.last_timings["planner_subqueries"] = 0
            return main[:k]
        ranked = list(main)
        seen = {h.chunk.chunk_id for h in ranked}
        best_per_part = []
        for sub in subs:
            part_hits = self.search(sub, k=3, mode="hybrid_rerank")
            for h in part_hits:
                h.stage_scores["subquery"] = sub
            if part_hits:
                best_per_part.append(part_hits[0])
            for h in part_hits:
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    ranked.append(h)
        # top hit of the full question first, then the best hit of every part, then the rest
        out, used = [], set()
        for h in [ranked[0]] + best_per_part + ranked:
            if h.chunk.chunk_id not in used:
                used.add(h.chunk.chunk_id)
                out.append(h)
        self.last_timings = {"retrieve_ms": (time.perf_counter() - t0) * 1000, "rerank_ms": 0.0,
                             "planner_subqueries": len(subs)}
        return out[:k]
