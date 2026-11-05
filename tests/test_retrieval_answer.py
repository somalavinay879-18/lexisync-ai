import pytest

from lexisync.answer import Claim, _numbers, answer_question, closed_book, parse_json, verify_claim
from lexisync.chunking import chunk_document
from lexisync.embed import HashingEncoder, KeywordReranker
from lexisync.ingest import load_pdf
from lexisync.llm import ScriptedLLM
from lexisync.retrieve import Index, Retriever, rrf, tokenize
from eval.run_eval import retrieval_metrics


@pytest.fixture(scope="module")
def retriever(sample_pdf, gdpr_like_pdf):
    chunks = chunk_document(load_pdf(sample_pdf, "sample", "Sample"))
    chunks += chunk_document(load_pdf(gdpr_like_pdf, "gdpr", "GDPR"))
    return Retriever(Index(chunks, HashingEncoder()), KeywordReranker())


def test_tokenizer_keeps_clause_numbers():
    toks = tokenize("Does Section 7.2 exceed Article 83(5)?")
    assert "7.2" in toks and "83(5)" in toks and "83" in toks


def test_rrf_rewards_agreement():
    fused = dict(rrf([(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)]))
    assert fused[2] > fused[1] > fused[3]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_find_the_liability_cap(retriever, mode):
    hits = retriever.search("What is the liability cap per calendar year?", k=3, mode=mode)
    assert any(h.chunk.clause == "7(2)" for h in hits)


def test_bm25_finds_exact_clause_reference(retriever):
    hits = retriever.search("Article 83(5)", k=1, mode="bm25")
    assert hits[0].chunk.clause == "83(5)"


def test_recall_metric():
    class H:
        def __init__(self, chunk):
            self.chunk = chunk

    class C:
        def __init__(self, doc, section, clause):
            self.doc_id, self.section, self.clause = doc, section, clause

    hits = [H(C("gdpr", "Article 1", "1(1)")), H(C("gdpr", "Article 83", "83(5)"))]
    m = retrieval_metrics(hits, [{"doc": "gdpr", "section": "Article 83", "clause": "83(5)"}], strict=True)
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["mrr"] == 0.5


def _cap_answer(retriever, question):
    hits = retriever.search(question, k=6)
    cid = next(f"C{i}" for i, h in enumerate(hits, 1) if h.chunk.clause == "7(2)")
    return cid


def test_grounded_answer_with_citation(retriever):
    q = "What is the liability cap per calendar year?"
    cid = _cap_answer(retriever, q)
    llm = ScriptedLLM([{
        "abstain": False, "answer": f"The cap is EUR 2 000 000 per calendar year [{cid}].",
        "claims": [{"text": "The liability cap is EUR 2 000 000 per calendar year.", "citations": [cid],
                    "quote": "The liability cap shall be EUR 2 000 000 per calendar year"}],
        "explanation": "Article 7(2) states the cap.", "conflicts": "",
    }])
    ans = answer_question(q, retriever, llm)
    assert not ans.abstained
    assert "[Sample p. 2, Article 7(2)]" in ans.answer
    assert ans.cited[0].clause == "7(2)"
    assert ans.confidence in ("HIGH", "MEDIUM")


def test_verifier_rejects_wrong_number_and_retries(retriever):
    q = "What is the liability cap per calendar year?"
    cid = _cap_answer(retriever, q)
    wrong = {"abstain": False, "answer": f"The cap is EUR 5 000 000 [{cid}].",
             "claims": [{"text": "The cap is EUR 5 000 000.", "citations": [cid], "quote": "The liability cap shall be"}],
             "explanation": "", "conflicts": ""}
    llm = ScriptedLLM([wrong, wrong])
    ans = answer_question(q, retriever, llm)
    assert len(ans.attempts) == 2  # verifier rejected the first draft and retried once
    assert "5000000" in ans.attempts[0]["unsupported"][0].replace(" ", "")
    assert ans.abstained  # still unsupported after the retry -> abstain instead of a wrong answer


def test_retrieval_gate_abstains_without_llm(retriever):
    llm = ScriptedLLM([])
    ans = answer_question("What is the corporate tax rate in Germany?", retriever, llm, abstain_threshold=0.5)
    assert ans.abstained and not llm.calls
    assert "Sample" in ans.searched_docs and "GDPR" in ans.searched_docs


def test_model_abstention_is_respected(retriever):
    llm = ScriptedLLM([{"abstain": True, "answer": "", "claims": [], "explanation": "", "conflicts": ""}])
    ans = answer_question("What does Article 33 say about notifying customers by post?", retriever, llm,
                          abstain_threshold=0.0)
    assert ans.abstained and ans.confidence == "INSUFFICIENT EVIDENCE"


def test_numbers_and_json_helpers():
    assert _numbers("up to 20 000 000 EUR or 4 % (2016)") == ["20000000", "4"]
    assert parse_json('```json\n{"abstain": true}\n```')["abstain"] is True
    assert parse_json("not json")["abstain"] is True


def test_verify_claim_without_citation_fails():
    c = verify_claim(Claim("Anything", [], ""), {})
    assert not c.supported and c.reason == "no valid citation"


def test_closed_book_baseline():
    llm = ScriptedLLM([{"abstain": False, "answer": "Up to 4 % of turnover.", "claims": [], "explanation": "", "conflicts": ""}])
    ans = closed_book("Maximum GDPR fine?", llm)
    assert not ans.abstained and ans.confidence == "UNVERIFIED" and not ans.cited


def test_planner_splits_two_part_questions():
    from lexisync.retrieve import plan_subqueries

    subs = plan_subqueries("When must a breach be notified to the authority, and when must data subjects be informed?")
    assert len(subs) == 2 and subs[1].startswith("when must data subjects")
    vague = plan_subqueries("Does the lower or the higher fine tier apply, and what is its maximum?")
    assert "fine" in vague[1]  # vague part inherits the topic
    assert plan_subqueries("What is the liability cap per calendar year?") == []


def test_planner_mode_returns_both_parts(retriever):
    q = "What is the liability cap per calendar year, and when must a breach be notified to the authority?"
    hits = retriever.search(q, k=3, mode="hybrid_rerank_mq")
    clauses = {h.chunk.clause for h in hits}
    assert "7(2)" in clauses and ("10(1)" in clauses or "33(1)" in clauses)


def test_partial_answer_kept_when_one_part_is_missing(retriever):
    q = "What is the liability cap per calendar year, and which court has jurisdiction?"
    hits = retriever.search(q, k=6)
    cid = next(f"C{i}" for i, h in enumerate(hits, 1) if h.chunk.clause == "7(2)")
    llm = ScriptedLLM([{
        "claims": [{"text": "The liability cap is EUR 2 000 000 per calendar year.", "citations": [cid],
                    "quote": "The liability cap shall be EUR 2 000 000 per calendar year"}],
        "answer": f"The cap is EUR 2 000 000 per calendar year [{cid}].",
        "missing": "which court has jurisdiction", "abstain": True, "explanation": "", "conflicts": "",
    }])
    ans = answer_question(q, retriever, llm, abstain_threshold=0.0)
    assert not ans.abstained and ans.missing == "which court has jurisdiction"
    assert ans.confidence in ("MEDIUM", "LOW")  # never HIGH when part of the question is uncovered


def test_plain_abstention_without_missing_part_abstains(retriever):
    llm = ScriptedLLM([{"claims": [], "answer": "", "missing": "", "abstain": True, "explanation": "", "conflicts": ""}])
    ans = answer_question("What is the liability cap per calendar year?", retriever, llm, abstain_threshold=0.0)
    assert ans.abstained
