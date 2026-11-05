"""LexiSyncAI - evidence-grounded legal document analyst (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import re

import streamlit as st

from lexisync.llm import make_llm
from lexisync.pipeline import CONFIG, Case
from lexisync.render import page_image

st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide")

BADGE = {
    "HIGH": ("#1b7f3b", "HIGH"),
    "MEDIUM": ("#b26a00", "MEDIUM"),
    "LOW": ("#b3261e", "LOW"),
    "INSUFFICIENT EVIDENCE": ("#5f6368", "INSUFFICIENT EVIDENCE (abstained)"),
    "UNVERIFIED": ("#5f6368", "UNVERIFIED (no retrieval)"),
}


def badge(level: str) -> str:
    color, text = BADGE.get(level, ("#5f6368", level))
    return (f"<span style='background:{color};color:white;padding:4px 10px;border-radius:6px;"
            f"font-weight:600;font-size:0.85rem'>{text}</span>")


@st.cache_resource
def get_llm():
    return make_llm()


# ---------------------------------------------------------------- sidebar: cases
st.sidebar.title("⚖️ LexiSyncAI")
st.sidebar.caption("Evidence-grounded answers from public legal documents")

cases = Case.list_cases()
choice = st.sidebar.selectbox("Case", ["➕ New case"] + cases)
if choice == "➕ New case":
    new_name = st.sidebar.text_input("Case name", placeholder="e.g. GDPR compliance review")
    if st.sidebar.button("Create case", disabled=not new_name):
        Case(new_name)
        st.rerun()
    st.title("LexiSyncAI")
    st.write("Create a case in the sidebar, upload 2–5 public legal PDFs, index them and ask questions. "
             "Every answer cites page and clause, and the system abstains when the documents do not contain the evidence.")
    st.stop()

case = Case(choice)
st.sidebar.markdown(f"**Documents ({len(case.pdfs)})**")
for pdf in case.pdfs:
    st.sidebar.write(f"📄 {pdf.name}")

st.sidebar.divider()
st.sidebar.markdown("**Settings**")
mode = st.sidebar.selectbox("Retrieval", ["hybrid_rerank_mq", "hybrid_rerank", "hybrid", "dense", "bm25"],
                            help="hybrid_rerank = BM25 + dense, merged with RRF, then cross-encoder reranking. _mq adds the query planner for two-part questions.")
top_k = st.sidebar.slider("Passages given to the LLM", 3, 10, CONFIG.get("top_k", 6))
verify = st.sidebar.checkbox("Verifier loop", value=True, help="Check every claim against its cited passage; retry once if a claim fails")
no_retrieval = st.sidebar.checkbox("No retrieval (baseline)", value=False)
st.sidebar.caption(f"LLM: {get_llm().name}")

st.title(case.name)
tab_docs, tab_ask = st.tabs(["1 · Documents & structure", "2 · Ask"])

# ---------------------------------------------------------------- documents
with tab_docs:
    col_up, col_url = st.columns(2)
    with col_up:
        files = st.file_uploader("Upload public legal PDFs (2–5)", type="pdf", accept_multiple_files=True)
        if files and st.button("Add uploaded files"):
            for f in files:
                case.add_file(f.name, f.getvalue())
            st.success(f"Added {len(files)} file(s). Now index the case.")
            st.rerun()
    with col_url:
        url = st.text_input("…or add a PDF by URL", placeholder="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679")
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
            with st.expander(f"📄 {s['document']} — {s['pages']} pages, {s['chunks']} chunks", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Pages", s["pages"])
                c2.metric("Tables", s["tables"])
                c3.metric("Footnotes", s["footnotes"])
                c4.metric("Scanned pages", len(s["scanned_pages"]))
                if s["scanned_pages"]:
                    ocr = s.get("ocr_pages", [])
                    st.warning(f"Scanned (image-only) pages: {s['scanned_pages']}. "
                               f"OCR used on: {ocr or 'none (Tesseract not installed)'}")
                if s.get("two_column_pages"):
                    st.caption(f"Two-column layout on pages {s['two_column_pages'][:20]}")
                st.caption(f"Removed {s['removed_header_footer_lines']} repeated header/footer line(s)")
                st.write("**Sections:** " + ", ".join(s["sections"][:200]))

# ---------------------------------------------------------------- ask
with tab_ask:
    if not case.structure:
        st.info("Index the documents first (tab 1).")
        st.stop()
    question = st.text_area("Question", placeholder="Under the GDPR, what is the maximum fine a company can receive, "
                            "and which kinds of violations trigger the higher fine?")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and writing a cited answer…"):
            try:
                st.session_state["answer"] = case.ask(question, get_llm(), mode=mode, k=top_k, verify=verify,
                                                      no_retrieval=no_retrieval)
            except Exception as exc:
                st.error(f"The LLM could not be reached ({exc}). Is Ollama running? Try: ollama serve")
                st.stop()

    ans = st.session_state.get("answer")
    if ans:
        st.markdown(badge(ans.confidence), unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Answer")
            if ans.abstained:
                st.warning(ans.answer)
                if ans.abstain_reason:
                    st.caption(f"Why: {ans.abstain_reason}")
                st.write("**Documents searched:** " + ", ".join(ans.searched_docs or ["–"]))
            else:
                st.markdown(re.sub(r"^(\d+)\.", r"\1\\.", ans.answer))
                if ans.missing:
                    st.warning(f"**Partial answer — not found in the documents:** {ans.missing}")
                if ans.explanation:
                    st.markdown(f"**Why this evidence supports it:** {ans.explanation}")
                if ans.conflicts and ans.conflicts.lower() not in ("", "none", "no conflicts", "n/a"):
                    st.info(f"**Conflicting sources:** {ans.conflicts}")
                st.markdown("**Citations**")
                for ch in ans.cited:
                    st.markdown(f"- {ch.label}" + (f" — *{ch.title}*" if ch.title else ""))
                with st.expander("Claim verification"):
                    for c in ans.claims:
                        st.write(("✅ " if c.supported else "❌ ") + c.text)
                        st.caption(f"{', '.join(c.citations)} · {c.reason} · quote: “{c.quote}”")
            with st.expander("Verifier log (attempts)"):
                st.json(ans.attempts)
            with st.expander("Retrieved passages and scores"):
                for h in ans.hits:
                    st.markdown(f"**{h.chunk.label}** · " + ", ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.stage_scores.items()))
                    st.caption(h.chunk.text[:400])
            t, tok = ans.timings, ans.tokens
            st.caption(f"⏱ total {t.get('total_s', 0):.1f}s · retrieval {t.get('retrieve_ms', 0):.0f} ms · "
                       f"rerank {t.get('rerank_ms', 0):.0f} ms · LLM {tok.get('llm_s', 0):.1f}s · "
                       f"tokens in/out {tok.get('prompt', 0)}/{tok.get('output', 0)}")
        with right:
            st.subheader("Cited evidence")
            if ans.cited:
                pdf_by_doc = {p.stem: str(p) for p in case.pdfs}
                options = [f"{c.label}" for c in ans.cited]
                pick = st.selectbox("Show page", options)
                chosen = ans.cited[options.index(pick)]
                st.image(page_image(pdf_by_doc[chosen.doc_id], chosen.page_start,
                                    [c for c in ans.cited if c.doc_id == chosen.doc_id]),
                         caption=f"{chosen.label} — cited region highlighted", use_container_width=True)
                st.markdown("**Cited text**")
                st.write(chosen.text)
            else:
                st.caption("No citations.")
