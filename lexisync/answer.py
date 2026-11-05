"""Evidence-grounded answering.

1. Retrieve passages. If even the best passage is not relevant, abstain
   without calling the LLM (retrieval gate).
2. Ask the LLM to answer only from the numbered passages and to return JSON:
   claims, each with passage ids and a short supporting quote.
3. Verify every claim: the quote must really occur in the cited passage and
   every number in the claim must appear there. If a claim fails, retrieve
   again with a reformulated query and regenerate (max. 2 attempts).
4. Unsupported claims are dropped. No supported claim left -> abstain.
   If only part of a multi-part question is covered, a partial answer is
   returned and the uncovered part is shown ("missing").
5. Confidence badge from claim support and retrieval relevance.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .models import Chunk
from .retrieve import Hit, Retriever

ABSTAIN_MESSAGE = "The documents do not contain enough evidence to answer this question."

ANSWER_SCHEMA = {
    "type": "object",
    # Order matters: the model writes the evidence (claims) first and decides
    # about abstaining last, instead of committing to "abstain" before reading.
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
                "required": ["text", "citations", "quote"],
            },
        },
        "answer": {"type": "string"},
        "missing": {"type": "string"},
        "abstain": {"type": "boolean"},
        "explanation": {"type": "string"},
        "conflicts": {"type": "string"},
    },
    "required": ["claims", "answer", "missing", "abstain", "explanation", "conflicts"],
}

SYSTEM_PROMPT = """You are a careful legal analyst. You answer questions about legal documents
using ONLY the numbered passages you are given. Never use outside knowledge.

Work in this order:
1. "claims": one claim per fact that answers the question. Each claim cites the passage ids it
   comes from (e.g. ["C2"]) and copies a short exact quote (5-25 words) from that passage.
   Only include facts that directly answer the question.
2. "answer": 1-4 plain-language sentences for a lawyer or compliance officer, with [C#] markers.
3. "missing": if the question has several parts and the passages answer only some of them,
   name the parts that are not covered. Otherwise leave it empty.
4. "abstain": true ONLY if the passages answer no part of the question. Then "claims" must be empty.

Rules:
- Many questions have two parts. Answer every part the passages support. A partial answer with
  the uncovered part listed in "missing" is better than abstaining.
- Passages can refer to other paragraphs ("the obligations referred to in paragraphs 1 and 2").
  Use the article title shown after the passage label to understand what they refer to.
- Keep rules and exceptions exactly as written: state the default rule first, then the exception
  or what Member States may change. Never swap numbers, thresholds or conditions.
- Do not guess. If passages disagree (for example two documents give different numbers), say so
  in "conflicts", name both sources and explain which one you rely on.
- "explanation": 1-2 sentences on why the cited evidence supports the answer.

Return only JSON with the keys: claims (text, citations, quote), answer, missing, abstain, explanation, conflicts."""

CLOSED_BOOK_PROMPT = """You are a legal analyst. Answer the question from your own knowledge.
If you are not sure, set "abstain" to true. Return only JSON with the keys:
claims (text, citations, quote), answer, missing, abstain, explanation, conflicts. Use empty citations."""


@dataclass
class Claim:
    text: str
    citations: list[str]
    quote: str
    supported: bool = False
    reason: str = ""


@dataclass
class Answer:
    question: str
    abstained: bool
    answer: str
    claims: list[Claim] = field(default_factory=list)
    cited: list[Chunk] = field(default_factory=list)
    confidence: str = "INSUFFICIENT EVIDENCE"
    explanation: str = ""
    conflicts: str = ""
    searched_docs: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    mode: str = "hybrid_rerank"
    abstain_reason: str = ""
    missing: str = ""  # parts of a multi-part question the documents do not cover


def _norm(text: str) -> str:
    text = text.lower().replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―-]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _numbers(text: str) -> list[str]:
    """Numbers as digit strings: '20 000 000' -> '20000000', '4 %' -> '4'."""
    found = re.findall(r"\d[\d\s.,]*\d|\d", text)
    out = []
    for n in found:
        digits = re.sub(r"[\s.,]", "", n)
        if digits and not (len(digits) == 4 and digits.startswith(("19", "20")) and " " not in n):
            out.append(digits)  # skip bare years such as 2016, they are references, not facts
    return out


def parse_json(text: str) -> dict:
    """Parse the model output; tolerate code fences or text around the JSON."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"abstain": True, "answer": "", "claims": [], "explanation": "Model output was not valid JSON.", "conflicts": ""}


def build_context(hits: list[Hit]) -> tuple[str, dict[str, Chunk]]:
    lookup, parts = {}, []
    for i, h in enumerate(hits, start=1):
        cid = f"C{i}"
        lookup[cid] = h.chunk
        title = f" - {h.chunk.title}" if h.chunk.title else ""
        parts.append(f"[{cid}] {h.chunk.label}{title}\n{h.chunk.text}")
    return "\n\n".join(parts), lookup


def verify_claim(claim: Claim, lookup: dict[str, Chunk]) -> Claim:
    chunks = [lookup[c] for c in claim.citations if c in lookup]
    if not chunks:
        claim.supported, claim.reason = False, "no valid citation"
        return claim
    cited_text = _norm(" ".join(c.text for c in chunks))
    cited_digits = re.sub(r"[\s.,]", "", cited_text)
    quote = _norm(claim.quote)
    quote_ok = bool(quote) and (quote in cited_text or fuzz.partial_ratio(quote, cited_text) >= 85)
    words = [w for w in re.findall(r"[a-z]{4,}", _norm(claim.text))]
    overlap = sum(w in cited_text for w in words) / len(words) if words else 0.0
    missing = [n for n in _numbers(claim.text) if n not in cited_digits]
    if missing:
        claim.supported, claim.reason = False, f"numbers not in cited passage: {', '.join(missing)}"
    elif quote_ok or overlap >= 0.6:
        claim.supported, claim.reason = True, "quote found" if quote_ok else f"word overlap {overlap:.0%}"
    else:
        claim.supported, claim.reason = False, f"quote not found (overlap {overlap:.0%})"
    return claim


def _cite_labels(text: str, lookup: dict[str, Chunk]) -> str:
    """Replace [C1, C3] markers by readable citations."""

    def repl(m: re.Match) -> str:
        ids = [x.strip() for x in m.group(1).split(",")]
        labels = [lookup[i].label for i in ids if i in lookup]
        return f"[{'; '.join(labels)}]" if labels else ""

    return re.sub(r"\[(C\d+(?:\s*,\s*C\d+)*)\]", repl, text).strip()


def _confidence(claims: list[Claim], top_relevance: float | None, partial: bool = False) -> str:
    if not claims:
        return "INSUFFICIENT EVIDENCE"
    share = sum(c.supported for c in claims) / len(claims)
    rel = 1.0 if top_relevance is None else top_relevance
    if share == 1 and rel >= 0.7 and not partial:
        return "HIGH"
    if share == 1 and rel >= 0.3 or share >= 0.75:
        return "MEDIUM"
    return "LOW"


def _generate(llm, question: str, hits: list[Hit], feedback: str = "") -> tuple[dict, dict, dict]:
    context, lookup = build_context(hits)
    user = f"Passages:\n\n{context}\n\nQuestion: {question}"
    if feedback:
        user += f"\n\nYour previous draft had problems: {feedback}\nOnly state what the passages say."
    res = llm.chat(SYSTEM_PROMPT, user, ANSWER_SCHEMA)
    return parse_json(res.text), lookup, {"prompt": res.prompt_tokens, "output": res.output_tokens, "llm_s": res.seconds}


def answer_question(
    question: str,
    retriever: Retriever,
    llm,
    k: int = 6,
    mode: str = "hybrid_rerank",
    abstain_threshold: float = 0.1,
    verify: bool = True,
    max_attempts: int = 2,
    doc_names: list[str] | None = None,
) -> Answer:
    t_start = time.perf_counter()
    hits = retriever.search(question, k=k, mode=mode)
    searched = doc_names or sorted({c.doc_name for c in retriever.index.chunks})
    ans = Answer(question=question, abstained=True, answer=ABSTAIN_MESSAGE, searched_docs=searched,
                 hits=hits, mode=mode, timings=dict(retriever.last_timings))
    top_rel = hits[0].stage_scores.get("rerank") if hits else None

    # Retrieval gate: nothing relevant was found, so do not ask the LLM at all.
    if not hits or (top_rel is not None and top_rel < abstain_threshold):
        ans.abstain_reason = (f"no relevant passage found (best relevance {top_rel:.2f} < {abstain_threshold})"
                              if top_rel is not None else "no passages retrieved")
        ans.timings["total_s"] = time.perf_counter() - t_start
        return ans

    tokens = {"prompt": 0, "output": 0, "llm_s": 0.0}
    feedback, query = "", question
    claims: list[Claim] = []
    data: dict = {}
    lookup: dict[str, Chunk] = {}
    for attempt in range(1, max_attempts + 1):
        data, lookup, usage = _generate(llm, question, hits, feedback)
        for key in tokens:
            tokens[key] += usage[key]
        claims = [
            verify_claim(Claim(str(c.get("text", "")), [str(x).strip("[] ") for x in c.get("citations", [])],
                               str(c.get("quote", ""))), lookup)
            for c in data.get("claims", []) if isinstance(c, dict)
        ]
        failed = [c for c in claims if not c.supported]
        ans.attempts.append({
            "attempt": attempt, "query": query, "abstain": bool(data.get("abstain")),
            "claims": len(claims), "unsupported": [f"{c.text} ({c.reason})" for c in failed],
        })
        if data.get("abstain") or not failed or not verify or attempt == max_attempts:
            break
        # Verifier rejected the draft: reformulate the query and retrieve again.
        query = question + " " + " ".join(c.text for c in failed)
        extra = retriever.search(query, k=k, mode=mode)
        seen = {h.chunk.chunk_id for h in hits}
        hits = (hits + [h for h in extra if h.chunk.chunk_id not in seen])[: k + 3]
        feedback = "; ".join(f"'{c.text}' ({c.reason})" for c in failed)

    ans.hits = hits
    ans.tokens = tokens
    supported = [c for c in claims if c.supported] if verify else claims
    ans.claims = claims
    ans.explanation = str(data.get("explanation", ""))
    ans.conflicts = str(data.get("conflicts", "") or "")
    missing = str(data.get("missing", "") or "").strip()
    if missing.lower() in ("none", "n/a", "nothing", "-"):
        missing = ""
    # A partial answer is kept when the model names the uncovered part and at least
    # one claim passed verification; a plain abstention (or nothing verified) abstains.
    if not supported or (data.get("abstain") and not missing):
        ans.abstain_reason = ("model found the evidence insufficient" if data.get("abstain")
                              else "no claim passed verification")
        ans.timings["total_s"] = time.perf_counter() - t_start
        return ans
    ans.missing = missing

    ans.abstained = False
    if len(supported) == len(claims) and data.get("answer") and not data.get("abstain"):
        ans.answer = _cite_labels(str(data["answer"]), lookup)
    else:  # some claims were dropped: rebuild the answer from the verified claims only
        ans.answer = " ".join(f"{c.text} [{'; '.join(lookup[i].label for i in c.citations if i in lookup)}]"
                              for c in supported)
    seen_ids, cited = set(), []
    for c in supported:
        for cid in c.citations:
            ch = lookup.get(cid)
            if ch and ch.chunk_id not in seen_ids:
                seen_ids.add(ch.chunk_id)
                cited.append(ch)
    ans.cited = cited
    ans.confidence = _confidence(claims if verify else supported, top_rel, partial=bool(missing))
    ans.timings["total_s"] = time.perf_counter() - t_start
    return ans


def closed_book(question: str, llm) -> Answer:
    """Baseline without retrieval, for the mandatory comparison."""
    t0 = time.perf_counter()
    res = llm.chat(CLOSED_BOOK_PROMPT, f"Question: {question}", ANSWER_SCHEMA)
    data = parse_json(res.text)
    abstained = bool(data.get("abstain")) or not data.get("answer")
    return Answer(
        question=question, abstained=abstained,
        answer=ABSTAIN_MESSAGE if abstained else str(data.get("answer", "")),
        confidence="INSUFFICIENT EVIDENCE" if abstained else "UNVERIFIED",
        explanation=str(data.get("explanation", "")), mode="no_retrieval",
        timings={"total_s": time.perf_counter() - t0},
        tokens={"prompt": res.prompt_tokens, "output": res.output_tokens, "llm_s": res.seconds},
    )
