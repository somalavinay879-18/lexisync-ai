"""LexiSyncAI - evidence-grounded legal document analyst (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from lexisync.llm import make_llm
from lexisync.pipeline import CONFIG, Case
from lexisync.render import page_image

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
      .lx-sub {color: #8b949e; font-size: 0.95rem; margin: 0.2rem 0 1rem 0;}
      .lx-answer {font-size: 1.08rem; line-height: 1.65;}
      .lx-badge {display: inline-block; color: white; padding: 4px 12px; border-radius: 999px;
                 font-weight: 600; font-size: 0.8rem; letter-spacing: 0.02em;}
      .lx-chip {display: inline-block; padding: 3px 10px; margin: 2px 6px 2px 0; border-radius: 6px;
                border: 1px solid rgba(128,128,128,0.35); font-size: 0.85rem;}
      .lx-muted {color: #8b949e; font-size: 0.85rem;}
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BADGE = {
    "HIGH": ("#1b7f3b", "HIGH CONFIDENCE"),
    "MEDIUM": ("#b26a00", "MEDIUM CONFIDENCE"),
    "LOW": ("#b3261e", "LOW CONFIDENCE"),
    "INSUFFICIENT EVIDENCE": ("#5f6368", "INSUFFICIENT EVIDENCE · abstained"),
    "UNVERIFIED": ("#5f6368", "UNVERIFIED · no retrieval"),
}

EXAMPLES = {
    "Single fact": "Within what time must a controller notify a personal data breach to the supervisory authority?",
    "Two paragraphs": "Under the GDPR, what is the maximum fine a company can receive, and which kinds of "
                      "violations trigger the higher fine?",
    "Not in the documents": "What is the corporate income tax rate in Germany?",
}

CONFIG_LABELS = {
    "no_retrieval": "LLM only (no retrieval)",
    "dense_only": "Dense retrieval",
    "full": "Full system",
    "full_no_planner": "Full − planner",
    "full_no_verifier": "Full − verifier",
    "full_k4": "Full, 4 passages",
}


def badge(level: str) -> str:
    color, text = BADGE.get(level, ("#5f6368", level))
    return f"<span class='lx-badge' style='background:{color}'>{text}</span>"


@st.cache_resource
def get_llm():
    return make_llm()


def set_question(text: str) -> None:
    st.session_state["question"] = text


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚖️ LexiSyncAI")
st.sidebar.caption("Evidence-grounded answers from public legal documents")

cases = Case.list_cases()
# open the first existing case by default, so a page refresh does not jump to "New case"
choice = st.sidebar.selectbox("Case", ["➕ New case"] + cases, index=1 if cases else 0)
if choice == "➕ New case":
    new_name = st.sidebar.text_input("Case name", placeholder="e.g. GDPR compliance review")
    if st.sidebar.button("Create case", disabled=not new_name, type="primary"):
        Case(new_name)
        st.rerun()
    st.title("LexiSyncAI")
    st.markdown("<p class='lx-sub'>Create a case in the sidebar, add 2–5 public legal PDFs, index them and ask "
                "questions. Every answer cites page and clause, and the system abstains when the documents do "
                "not contain the evidence.</p>", unsafe_allow_html=True)
    st.stop()

case = Case(choice)
with st.sidebar.expander(f"Documents ({len(case.pdfs)})", expanded=True):
    for pdf in case.pdfs:
        st.write(f"📄 {pdf.name}")

st.sidebar.markdown("**Settings**")
mode = st.sidebar.selectbox(
    "Retrieval", ["hybrid_rerank_mq", "hybrid_rerank", "hybrid", "dense", "bm25"],
    help="hybrid_rerank = BM25 + dense, merged with RRF, then cross-encoder reranking. "
         "_mq adds the query planner for two-part questions.")
top_k = st.sidebar.slider("Passages given to the LLM", 3, 10, 4,
                          help="4 passages: same accuracy as 6 in the evaluation, about twice as fast")
verify = st.sidebar.checkbox("Verifier loop", value=True,
                             help="Check every claim against its cited passage; retry once if a claim fails")
no_retrieval = st.sidebar.checkbox("No retrieval (baseline)", value=False,
                                   help="Ask the LLM without documents, for comparison")
st.sidebar.caption(f"LLM: {get_llm().name}")

# ---------------------------------------------------------------- header
docs_line = " · ".join(s["document"] for s in case.structure) if case.structure else "not indexed yet"
st.title(case.name)
st.markdown(f"<p class='lx-sub'>{docs_line}</p>", unsafe_allow_html=True)
tab_ask, tab_docs, tab_eval = st.tabs(["💬 Ask", "📄 Documents & structure", "📊 Evaluation"])

# ---------------------------------------------------------------- documents
with tab_docs:
    with st.container(border=True):
        col_up, col_url = st.columns(2)
        with col_up:
            files = st.file_uploader("Upload public legal PDFs (2–5)", type="pdf", accept_multiple_files=True)
            if files and st.button("Add uploaded files"):
                for f in files:
                    case.add_file(f.name, f.getvalue())
                st.success(f"Added {len(files)} file(s). Now index the case.")
                st.rerun()
        with col_url:
            url = st.text_input("…or add a PDF by URL",
                                placeholder="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679")
            if st.button("Download", disabled=not url):
                try:
                    case.add_url(url)
                    st.success("Downloaded.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not download a PDF from that URL: {exc}")

        n = len(case.pdfs)
        if n and not 2 <= n <= 5:
            st.info(f"The case has {n} document(s); the assignment expects 2–5.")
        if st.button("🔎 Index documents", type="primary", disabled=n == 0):
            with st.spinner("Parsing PDFs, chunking by clause and building the BM25 + embedding index…"):
                case.build()
            st.success("Indexed.")

    if case.structure:
        st.subheader("Detected structure")
        for s in case.structure:
            secs = s["sections"]
            arts = [x.split()[-1] for x in secs if x.startswith("Article ")]
            recs = [x for x in secs if x.startswith("Recital ")]
            with st.container(border=True):
                st.markdown(f"**📄 {s['document']}** <span class='lx-muted'>· {s.get('file', '')}</span>",
                            unsafe_allow_html=True)
                cols = st.columns(7)
                for col, (label, value) in zip(cols, [
                    ("Pages", s["pages"]), ("Chunks", s["chunks"]), ("Articles", len(arts)),
                    ("Recitals", len(recs)), ("Footnotes", s["footnotes"]), ("Tables", s["tables"]),
                    ("Scanned pages", len(s["scanned_pages"])),
                ]):
                    col.metric(label, value)
                notes = [f"Removed {s['removed_header_footer_lines']} repeated header/footer line(s)"]
                if s.get("two_column_pages"):
                    pages = s["two_column_pages"]
                    notes.append(f"two-column layout on {len(pages)} page(s): {pages[:15]}"
                                 + (" …" if len(pages) > 15 else ""))
                st.caption(" · ".join(notes))
                if s["scanned_pages"]:
                    ocr = s.get("ocr_pages", [])
                    st.warning(f"Scanned (image-only) pages: {s['scanned_pages']}. "
                               f"OCR used on: {ocr or 'none (Tesseract not installed)'}")
                if arts:
                    with st.expander("Detected articles"):
                        st.write(", ".join(arts))

# ---------------------------------------------------------------- ask
with tab_ask:
    if not case.structure:
        st.info("Index the documents first (tab 📄 Documents & structure).")
    else:
        st.caption("Try an example:")
        ex_cols = st.columns(len(EXAMPLES))
        for col, (label, q) in zip(ex_cols, EXAMPLES.items()):
            col.button(label, on_click=set_question, args=(q,), width="stretch")
        question = st.text_area("Question", key="question", height=90,
                                placeholder="Ask anything about the documents in this case…")
        if st.button("Ask", type="primary", disabled=not question.strip()):
            with st.spinner("Retrieving evidence and writing a cited answer…"):
                try:
                    st.session_state["answer"] = case.ask(question, get_llm(), mode=mode, k=top_k,
                                                          verify=verify, no_retrieval=no_retrieval)
                except Exception as exc:
                    st.error(f"The LLM could not be reached ({exc}). Is Ollama running? Try: ollama serve")
                    st.stop()

        ans = st.session_state.get("answer")
        if ans:
            left, right = st.columns([1, 1], gap="large")
            with left:
                with st.container(border=True):
                    st.markdown(badge(ans.confidence), unsafe_allow_html=True)
                    if ans.abstained:
                        st.markdown(f"<p class='lx-answer'>{ans.answer}</p>", unsafe_allow_html=True)
                        if ans.abstain_reason:
                            st.caption(f"Why: {ans.abstain_reason}")
                        st.caption("Documents searched: " + ", ".join(ans.searched_docs or ["–"]))
                    else:
                        st.markdown(re.sub(r"^(\d+)\.", r"\1\\.", ans.answer))
                        if ans.missing:
                            st.warning(f"**Partial answer, not found in the documents:** {ans.missing}")
                        if ans.explanation:
                            st.caption(f"Why this evidence supports it: {ans.explanation}")
                        if ans.conflicts and ans.conflicts.lower() not in ("", "none", "no conflicts", "n/a"):
                            st.info(f"**Conflicting sources:** {ans.conflicts}")
                        chips = "".join(f"<span class='lx-chip'>{c.label}</span>" for c in ans.cited)
                        st.markdown(f"**Citations**<br>{chips}", unsafe_allow_html=True)

                t, tok = ans.timings, ans.tokens
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total", f"{t.get('total_s', 0):.1f} s")
                m2.metric("Retrieval", f"{t.get('retrieve_ms', 0):.0f} ms")
                m3.metric("Rerank", f"{t.get('rerank_ms', 0):.0f} ms")
                m4.metric("LLM", f"{tok.get('llm_s', 0):.1f} s",
                          help=f"tokens in/out {tok.get('prompt', 0)}/{tok.get('output', 0)}")

                if ans.claims:
                    with st.expander(f"Claim verification ({sum(c.supported for c in ans.claims)}"
                                     f"/{len(ans.claims)} verified)", expanded=False):
                        st.dataframe(pd.DataFrame([{
                            "": "✅" if c.supported else "❌", "Claim": c.text,
                            "Cited": ", ".join(c.citations), "Check": c.reason, "Quote": c.quote,
                        } for c in ans.claims]), hide_index=True, width="stretch")
                if ans.hits:
                    with st.expander(f"Retrieved passages ({len(ans.hits)})"):
                        rows = []
                        for rank, h in enumerate(ans.hits, 1):
                            sc = h.stage_scores
                            rows.append({
                                "#": rank, "Passage": h.chunk.label,
                                "Relevance": sc.get("rerank"), "RRF": sc.get("rrf"),
                                "BM25 rank": sc.get("bm25_rank"), "Dense rank": sc.get("dense_rank"),
                                "Text": h.chunk.text[:220] + ("…" if len(h.chunk.text) > 220 else ""),
                            })
                        df = pd.DataFrame(rows).dropna(axis=1, how="all")
                        st.dataframe(df, hide_index=True, width="stretch", column_config={
                            "Relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1,
                                                                         format="%.2f"),
                            "RRF": st.column_config.NumberColumn(format="%.4f"),
                            "Text": st.column_config.TextColumn(width="large"),
                        })
                if ans.attempts:
                    with st.expander(f"Verifier log ({len(ans.attempts)} attempt(s))"):
                        st.json(ans.attempts)

            with right:
                if ans.cited:
                    pdf_by_doc = {p.stem: str(p) for p in case.pdfs}
                    options = [c.label for c in ans.cited]
                    pick = st.selectbox("Cited page", options)
                    chosen = ans.cited[options.index(pick)]
                    st.image(page_image(pdf_by_doc[chosen.doc_id], chosen.page_start,
                                        [c for c in ans.cited if c.doc_id == chosen.doc_id]),
                             caption=f"{chosen.label}: cited region highlighted", width="stretch")
                    with st.expander("Cited text"):
                        st.write(chosen.text)
                else:
                    with st.container(border=True):
                        st.markdown("**No citations**")
                        st.caption("The system did not find evidence it could cite, so it shows no page "
                                   "instead of guessing.")

# ---------------------------------------------------------------- evaluation
with tab_eval:
    summary_path, retrieval_path, answers_path = (REPORTS / n for n in ("summary.csv", "retrieval.csv", "answers.csv"))
    if not summary_path.exists() and not retrieval_path.exists():
        st.info("No evaluation results yet. Run `make eval-retrieval` and `make eval`.")
    else:
        if summary_path.exists():
            summ = pd.read_csv(summary_path)
            # one "Correct" value per setup: manual grade where available, else the automatic check
            def correct(row):
                m = row.get("answer_correct_manual")
                return (float(m), "manual") if m is not None and not pd.isna(m) else (float(row["answer_correct_auto"]), "auto")

            summ[["correct", "graded"]] = summ.apply(lambda r: pd.Series(correct(r)), axis=1)
            full = summ.set_index("config").to_dict("index")
            base, sys_ = full.get("no_retrieval", {}), full.get("full", {})

            st.subheader("Headline: retrieval vs LLM alone")
            k1, k2, k3, k4 = st.columns(4)
            if sys_ and base:
                k1.metric("Correct answers", f"{sys_['correct']:.0%}",
                          f"{(sys_['correct'] - base['correct']) * 100:+.0f} pts vs LLM only")
                k2.metric("Unanswerable refused", f"{sys_['abstention_accuracy']:.0%}",
                          help=f"LLM only: {base['abstention_accuracy']:.0%}")
                k2.caption(f"LLM only: {base['abstention_accuracy']:.0%}")
            if sys_ and not pd.isna(sys_.get("citation_precision")):
                k3.metric("Citation precision", f"{sys_['citation_precision']:.0%}")
                k3.caption("cited passages from the gold article")
            k4l = full.get("full_k4")
            if k4l:
                k4.metric("Latency, 4 passages", f"{k4l['mean_seconds']:.1f} s")
                k4.caption(f"6 passages: {sys_.get('mean_seconds', float('nan')):.1f} s")
            st.caption("Correct = share of the 18 answerable questions; refused = share of the 4 unanswerable "
                       "questions. Manual grading where available, otherwise the automatic key-fact check.")

            st.subheader("End-to-end comparison")
            short = {"no_retrieval": "LLM only", "dense_only": "Dense", "full": "Full system",
                     "full_no_planner": "− planner", "full_no_verifier": "− verifier", "full_k4": "4 passages"}
            summ["Short"] = summ["config"].map(short).fillna(summ["config"])
            order = list(summ["Short"])
            metrics = {"correct": "Correct", "abstention_accuracy": "Unanswerable refused",
                       "false_abstention": "Wrongly refused"}
            palette = alt.Scale(domain=list(metrics.values()), range=["#4c8bf5", "#2ea043", "#d29922"])
            long = summ.melt(id_vars=["Short"], value_vars=list(metrics), var_name="m", value_name="value").dropna()
            long["Metric"] = long["m"].map(metrics)
            bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Short:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                xOffset=alt.XOffset("Metric:N", sort=list(metrics.values())),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("Metric:N", scale=palette, legend=alt.Legend(orient="top", title=None)),
                tooltip=["Short", "Metric", alt.Tooltip("value:Q", format=".0%")],
            )
            labels = bars.mark_text(dy=-7, fontSize=10, color="#c9d1d9").encode(
                text=alt.Text("value:Q", format=".0%"))
            st.altair_chart((bars + labels).properties(height=330), width="stretch")

            c_lat, c_tab = st.columns([2, 3], gap="large")
            with c_lat:
                lat_base = alt.Chart(summ).encode(
                    y=alt.Y("Short:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12)),
                    x=alt.X("mean_seconds:Q", title="mean seconds per question"))
                lat = lat_base.mark_bar(color="#8b949e", cornerRadiusEnd=3) + lat_base.mark_text(
                    align="left", dx=4, color="#c9d1d9").encode(text=alt.Text("mean_seconds:Q", format=".1f"))
                st.markdown("**Latency**")
                st.altair_chart(lat.properties(height=36 * len(summ)), width="stretch")
            with c_tab:
                fmt = lambda v, f: "–" if v is None or pd.isna(v) else f.format(v)
                table = pd.DataFrame([{
                    "Setup": CONFIG_LABELS.get(r["config"], r["config"]),
                    "Correct": f"{r['correct']:.0%} ({r['graded']})",
                    "Citation prec.": fmt(r.get("citation_precision"), "{:.0%}"),
                    "Refused unanswerable": fmt(r.get("abstention_accuracy"), "{:.0%}"),
                    "Wrongly refused": fmt(r.get("false_abstention"), "{:.0%}"),
                    "Seconds": fmt(r.get("mean_seconds"), "{:.1f}"),
                    "Tokens in": fmt(r.get("mean_prompt_tokens"), "{:.0f}"),
                } for _, r in summ.iterrows()])
                st.markdown("**All setups**")
                st.dataframe(table, hide_index=True, width="stretch")

        if retrieval_path.exists():
            st.subheader("Retrieval: recall@k per configuration")
            ret = pd.read_csv(retrieval_path)
            level = st.radio("Level", ["paragraph", "article"], horizontal=True,
                             help="paragraph = the exact gold paragraph is retrieved; article = any part of the gold article")
            sub = ret[ret["level"] == level]
            ks = [c for c in sub.columns if c.startswith("recall@")]
            rlong = sub.melt(id_vars=["mode"], value_vars=ks, var_name="k", value_name="recall")
            rlong["k"] = rlong["k"].str.replace("recall@", "").astype(int)
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                line = alt.Chart(rlong).mark_line(point=True, strokeWidth=2.5).encode(
                    x=alt.X("k:O", title="k (passages retrieved)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("recall:Q", scale=alt.Scale(domain=[0.3, 1]), axis=alt.Axis(format="%"), title="recall@k"),
                    color=alt.Color("mode:N", legend=alt.Legend(orient="bottom", title=None, columns=3, labelLimit=200)),
                    tooltip=["mode", "k", alt.Tooltip("recall:Q", format=".1%")],
                ).properties(height=300)
                st.altair_chart(line, width="stretch")
            with c2:
                tbl = sub.drop(columns=["level"]).set_index("mode")
                tbl.columns = [c.replace("recall@", "R@").upper() if c != "mrr" else "MRR" for c in tbl.columns]
                st.dataframe(tbl.style.format("{:.2f}").highlight_max(axis=0, color="rgba(46,160,67,0.35)"),
                             width="stretch")
                st.caption("Green = best per column. MRR = 1 / rank of the first correct passage.")

        if answers_path.exists():
            with st.expander("All answers from the evaluation run"):
                answers = pd.read_csv(answers_path)
                cfg = st.selectbox("Setup", list(dict.fromkeys(answers["config"])),
                                   format_func=lambda c: CONFIG_LABELS.get(c, c))
                view = answers[answers["config"] == cfg]
                cols = [c for c in ["id", "type", "question", "answer", "citations", "abstained", "auto_correct",
                                    "manual_correct", "confidence", "seconds"] if c in view]
                st.dataframe(view[cols], hide_index=True, width="stretch", column_config={
                    "question": st.column_config.TextColumn(width="medium"),
                    "answer": st.column_config.TextColumn(width="large"),
                    "seconds": st.column_config.NumberColumn(format="%.1f"),
                })
