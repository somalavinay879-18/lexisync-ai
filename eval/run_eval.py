"""Evaluation of LexiSyncAI on held-out questions.

Part 1 - retrieval (no LLM needed):
    recall@1/3/5/10 and MRR for bm25, dense, hybrid and hybrid_rerank,
    at article level and at paragraph level.
Part 2 - end-to-end (needs the LLM):
    answer correctness (automatic key-fact check + column for manual grading),
    citation precision, abstention accuracy, latency and tokens for:
      no_retrieval      LLM alone (mandatory comparison)
      dense_only        dense retrieval, verifier on
      full              hybrid + reranker + query planner + verifier (the system)
      full_no_planner   ablation: query planner switched off
      full_no_verifier  ablation: verifier loop switched off
      full_k4           optimisation: 4 instead of 6 passages in the prompt

Usage:
    python -m eval.run_eval --case "GDPR ePrivacy" --retrieval-only
    python -m eval.run_eval --case "GDPR ePrivacy"
    python -m eval.run_eval --case "GDPR ePrivacy" --configs full no_retrieval
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

from lexisync.answer import Answer
from lexisync.llm import make_llm
from lexisync.pipeline import Case, set_seeds
from lexisync.retrieve import MODES

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 3, 5, 10)

CONFIGS = {
    "no_retrieval": {"no_retrieval": True},
    "dense_only": {"mode": "dense", "verify": True},
    "full": {"mode": "hybrid_rerank_mq", "verify": True},
    "full_no_planner": {"mode": "hybrid_rerank", "verify": True},
    "full_no_verifier": {"mode": "hybrid_rerank_mq", "verify": False},
    "full_k4": {"mode": "hybrid_rerank_mq", "verify": True, "k": 4},
}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _match(chunk, gold: dict, strict: bool) -> bool:
    if chunk.doc_id != gold["doc"] or chunk.section != gold["section"]:
        return False
    return not strict or not gold.get("clause") or chunk.clause == gold["clause"]


def retrieval_metrics(hits, gold: list[dict], strict: bool) -> dict:
    out = {}
    for k in KS:
        top = [h.chunk for h in hits[:k]]
        found = sum(any(_match(c, g, strict) for c in top) for g in gold)
        out[f"recall@{k}"] = found / len(gold)
    rr = 0.0
    for rank, h in enumerate(hits, 1):
        if any(_match(h.chunk, g, strict) for g in gold):
            rr = 1 / rank
            break
    out["mrr"] = rr
    return out


def eval_retrieval(case: Case, questions: list[dict], out_dir: Path) -> list[dict]:
    retriever = case.get_retriever()
    answerable = [q for q in questions if q["gold"]]
    rows, per_question = [], []
    for mode in MODES:
        for level, strict in (("article", False), ("paragraph", True)):
            scores = []
            for q in answerable:
                hits = retriever.search(q["question"], k=max(KS), mode=mode)
                m = retrieval_metrics(hits, q["gold"], strict)
                scores.append(m)
                if level == "paragraph":
                    per_question.append({"mode": mode, "id": q["id"], **{k: round(v, 3) for k, v in m.items()}})
            row = {"mode": mode, "level": level}
            for key in scores[0]:
                row[key] = round(statistics.mean(s[key] for s in scores), 3)
            rows.append(row)

    # Relevance of the best passage for answerable vs unanswerable questions,
    # used to choose the abstention threshold of the retrieval gate.
    gate = []
    for q in questions:
        hits = retriever.search(q["question"], k=1, mode="hybrid_rerank")
        top = hits[0].stage_scores.get("rerank", float("nan")) if hits else 0.0
        gate.append({"id": q["id"], "type": q["type"], "top_relevance": round(top, 4)})

    _write_csv(out_dir / "retrieval.csv", rows)
    _write_csv(out_dir / "retrieval_per_question.csv", per_question)
    _write_csv(out_dir / "abstention_gate.csv", gate)
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower().replace("’", "'"))


def auto_correct(q: dict, ans: Answer) -> bool:
    if q["type"] == "unanswerable":
        return ans.abstained
    if ans.abstained:
        return False
    text = _norm(ans.answer + " " + " ".join(c.text for c in ans.claims))
    # each key fact may list accepted alternatives: "20 000 000|20 million"
    return all(any(_norm(alt) in text for alt in f.split("|")) for f in q["key_facts"])


def citation_precision(q: dict, ans: Answer) -> float | None:
    if ans.abstained or not q["gold"] or not ans.cited:
        return None
    good = sum(any(_match(c, g, strict=False) for g in q["gold"]) for c in ans.cited)
    return good / len(ans.cited)


def eval_end_to_end(case: Case, questions: list[dict], configs: list[str], out_dir: Path,
                    resume: bool = False) -> list[dict]:
    """Answers every question with every configuration.

    answers.csv is written after each question, so an interrupted run loses
    nothing; --resume skips the (config, question) pairs that are already done.
    A failed LLM call is recorded as an error row instead of stopping the run.
    """
    llm = make_llm()
    path = out_dir / "answers.csv"
    rows: list[dict] = []
    if resume and path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if not r.get("error")]
    done = {(r["config"], r["id"]) for r in rows}
    for name in configs:
        params = CONFIGS[name]
        for q in questions:
            if (name, q["id"]) in done:
                continue
            set_seeds()
            t0 = time.perf_counter()
            error = ""
            try:
                ans = case.ask(q["question"], llm, **params)
            except Exception as exc:  # e.g. Ollama timeout: record it and continue
                error = f"{type(exc).__name__}: {exc}"[:200]
                ans = Answer(question=q["question"], abstained=True, answer="", abstain_reason="error")
            seconds = time.perf_counter() - t0
            cp = citation_precision(q, ans)
            rows.append({
                "config": name, "id": q["id"], "type": q["type"], "question": q["question"],
                "abstained": ans.abstained, "auto_correct": auto_correct(q, ans), "manual_correct": "",
                "citation_precision": "" if cp is None else round(cp, 3),
                "confidence": ans.confidence, "attempts": len(ans.attempts),
                "seconds": round(seconds, 2), "llm_seconds": round(ans.tokens.get("llm_s", 0), 2),
                "prompt_tokens": ans.tokens.get("prompt", 0), "output_tokens": ans.tokens.get("output", 0),
                "answer": ans.answer, "citations": "; ".join(c.label for c in ans.cited),
                "abstain_reason": ans.abstain_reason, "missing": ans.missing, "error": error,
            })
            _write_csv(path, rows)
            flag = f" ERROR {error}" if error else ""
            print(f"[{name}] {q['id']} abstained={ans.abstained} correct={rows[-1]['auto_correct']} {seconds:.1f}s{flag}")
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == name and not r.get("error")]
        if not rs:
            continue
        ans = [r for r in rs if r["type"] != "unanswerable"]
        una = [r for r in rs if r["type"] == "unanswerable"]
        manual = [r for r in rs if str(r.get("manual_correct", "")).strip() in ("0", "1")]
        cps = [float(r["citation_precision"]) for r in ans if str(r["citation_precision"]) != ""]
        out.append({
            "config": name,
            "answer_correct_auto": round(sum(_true(r["auto_correct"]) for r in ans) / len(ans), 3) if ans else "",
            "answer_correct_manual": round(sum(r["manual_correct"] == "1" for r in manual) / len(manual), 3) if manual else "",
            "citation_precision": round(statistics.mean(cps), 3) if cps else "",
            "abstention_accuracy": round(sum(_true(r["abstained"]) for r in una) / len(una), 3) if una else "",
            "false_abstention": round(sum(_true(r["abstained"]) for r in ans) / len(ans), 3) if ans else "",
            "mean_seconds": round(statistics.mean(float(r["seconds"]) for r in rs), 2),
            "median_seconds": round(statistics.median(float(r["seconds"]) for r in rs), 2),
            "mean_prompt_tokens": round(statistics.mean(float(r["prompt_tokens"]) for r in rs)),
            "mean_output_tokens": round(statistics.mean(float(r["output_tokens"]) for r in rs)),
            "verifier_retries": sum(int(r["attempts"]) > 1 for r in rs),
        })
    return out


def _true(v) -> bool:
    return str(v).lower() in ("true", "1")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no data_\n"
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, retrieval: list[dict] | None, summary: list[dict] | None, meta: dict) -> None:
    parts = ["# LexiSyncAI evaluation\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')}. "
             f"Case: {meta['case']}. Questions: {meta['n']} ({meta['types']}).\n"]
    if retrieval:
        parts += ["\n## Retrieval: recall@k per configuration\n",
                  "Article level = the right article is retrieved; paragraph level = the exact paragraph.\n\n",
                  _md_table(retrieval)]
    if summary:
        parts += ["\n## End-to-end results\n",
                  "`no_retrieval` vs `full` is the mandatory comparison; `full_no_planner` and `full_no_verifier` are ablations; "
                  "`full_k4` is the latency optimisation (fewer passages in the prompt).\n\n",
                  _md_table(summary)]
    (out_dir / "results.md").write_text("".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="GDPR ePrivacy")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (keeps reports/answers.csv)")
    ap.add_argument("--summary-only", action="store_true", help="re-summarise reports/answers.csv (after manual grading)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions))
    types = ", ".join(f"{t}: {sum(q['type'] == t for q in questions)}" for t in ("factual", "multihop", "unanswerable"))
    meta = {"case": args.case, "n": len(questions), "types": types}

    if args.summary_only:
        with open(out_dir / "answers.csv") as f:
            rows = list(csv.DictReader(f))
        summary = summarise(rows)
        _write_csv(out_dir / "summary.csv", summary)
        retrieval = list(csv.DictReader(open(out_dir / "retrieval.csv"))) if (out_dir / "retrieval.csv").exists() else None
        write_report(out_dir, retrieval, summary, meta)
        print(_md_table(summary))
        return

    set_seeds()
    case = Case(args.case)
    if not case.load():
        raise SystemExit(f"Case '{args.case}' has no index. Run: make index")
    case.structure = case.structure or []
    retrieval = eval_retrieval(case, questions, out_dir)
    print(_md_table(retrieval))
    summary = None
    if not args.retrieval_only:
        rows = eval_end_to_end(case, questions, args.configs, out_dir, resume=args.resume)
        summary = summarise(rows)
        _write_csv(out_dir / "summary.csv", summary)
        print(_md_table(summary))
    write_report(out_dir, retrieval, summary, meta)
    print(f"Reports written to {out_dir}/")


if __name__ == "__main__":
    main()
