"""Dense encoders and the cross-encoder reranker.

The real models come from Hugging Face and are downloaded on first use.
`HashingEncoder` and `KeywordReranker` are small stand-ins with no download,
used by the tests and in CI.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class SentenceEncoder:
    """Bi-encoder for dense retrieval (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=_device())

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def encode_query(self, text: str) -> np.ndarray:
        prefix = QUERY_PREFIX if "bge" in self.name.lower() else ""
        return self.model.encode([prefix + text], normalize_embeddings=True)[0]


class HashingEncoder:
    """Offline stand-in: hashed bag of words and character trigrams."""

    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        grams = words + [w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class CrossReranker:
    """Cross-encoder reranker (default: cross-encoder/ms-marco-MiniLM-L-6-v2).

    Returns a relevance probability in [0, 1] (sigmoid of the logit), which the
    abstention rule uses as a threshold.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, device=_device())

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = self.model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [1 / (1 + math.exp(-float(x))) for x in np.atleast_1d(logits)]


class KeywordReranker:
    """Offline stand-in: share of query words found in the passage."""

    name = "keyword-overlap"

    def score(self, query: str, texts: list[str]) -> list[float]:
        stop = {"the", "a", "of", "and", "to", "in", "is", "what", "which", "under", "for", "be", "by", "how"}
        q = {w for w in re.findall(r"\w+", query.lower()) if w not in stop}
        out = []
        for t in texts:
            words = set(re.findall(r"\w+", t.lower()))
            out.append(len(q & words) / len(q) if q else 0.0)
        return out
