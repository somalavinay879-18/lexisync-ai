# Exam preparation: demo script and theory questions

## Demo (5 minutes)

1. **(30 s) Architecture.** Show `docs/architecture.png`: ingestion → clause chunking → hybrid retrieval with
   reranking → grounded generation with verifier → UI.
2. **(1 min) New case.** Create a case, upload the PDF you are given, click *Index*. Show the structure view:
   pages, sections, tables, scanned pages.
3. **(1 min) Prose question.** Ask it and point at the citation, the highlighted region on the page and the
   confidence badge.
4. **(1 min) Multi-hop question.** Show that the answer cites two pages or clauses and open the *Claim
   verification* list.
5. **(1 min) Unanswerable question.** Show the abstention, the reason and the list of searched documents.
6. **(30 s) Evaluation.** Open the *Evaluation* tab: headline numbers, the end-to-end chart and the recall@k chart.

### Checklist before the exam

- Start Ollama and the app, and ask **one warm-up question** (the first answer takes ~40 s while the model loads).
- Set *Passages given to the LLM* to 4 (same accuracy as 6, about twice as fast).
- `brew install tesseract`, so a scanned exam PDF is OCR'd instead of skipped.
- Practise once with a PDF you have never used, ideally a contract (sections 7 / clauses 7.2) and a scanned page.
- Close extra browser tabs of the app; keep the laptop plugged in.

## My results (memorise these)

| | LLM only | Full system | Full, 4 passages |
|---|---|---|---|
| Correct (manual, 18 answerable) | 56% | 78% | 78% |
| Unanswerable refused (4) | 0% | 100% | 100% |
| Citation precision | – | 84% | 85% |
| Wrongly refused | 0% | 11% | 0% |
| Mean latency | 10.9 s | 33.1 s | 17.2 s |

Retrieval, paragraph level: hybrid + rerank R@1 0.72 vs dense 0.53 vs BM25 0.44; MRR 0.92. The cross-encoder gives
the biggest gain. Plain hybrid RRF was slightly worse than dense alone on this question set.

## Questions about my own results

**Why is plain hybrid (RRF) worse than dense alone, if hybrid is supposed to be better?**
On paraphrased questions BM25 ranks keyword-matching but wrong passages high, and RRF averages that noise in.
Hybrid is still the right candidate generator because it finds clause numbers dense misses; the cross-encoder then
fixes the ordering (R@1 0.50 → 0.72).

**What does the query planner do, and did it help?**
It splits a two-part question ("…, and which violations trigger the higher fine?") into sub-queries, searches each,
and keeps the best passage per part. Retrieval improved (R@10 0.89 → 0.92), but end-to-end correctness did not
(0.78 with vs 0.83 without = one question of 18). With six multi-hop questions that difference is noise. I report
it honestly instead of claiming a gain.

**What did you change after the first evaluation run?**
Error analysis of run 1 (`reports_run1/`) showed the model abstaining on two-part questions: the JSON schema made it
decide `abstain` before writing any evidence. I reordered the schema (claims → answer → missing → abstain), allowed
partial answers, and added a rule to keep default rules and exceptions apart (it had swapped 16 and 13 in Art. 8).
The baseline kept its original prompt so the comparison stays fair. Because I tuned on these questions, the numbers
are optimistic; a fresh question set would be the clean test.

**Why both automatic and manual grading?**
The automatic check only looks for key facts such as "72 hours". It rated the LLM-only answers too high (61% vs 56%
manual) because it finds the right number in answers that are vague or partly wrong.

**Why did the verifier not change any answers?**
In the final run no claim failed, so there were no retries. The verifier is a safety net; the tests show it rejecting
a wrong number. Its limit: it checks that a quote and its numbers are in the passage, not legal logic. In M06 a wrong
exception shared enough words with the passage to pass.

**Why is an answer MEDIUM and not HIGH?**
HIGH needs every claim verified and a best relevance of at least 0.7. Partial answers are capped at MEDIUM.

**Why is the 0.1 abstention threshold safe?**
On my questions every answerable question had a best relevance of at least 0.54 and every unanswerable one at most
0.03, so 0.1 separates them with a large margin. On a new document type the threshold should be re-checked.

**Why is it slow, and would a GPU help?**
Retrieval takes milliseconds and reranking ~150 ms; almost all time is the 7B LLM generating on a laptop. Fewer
passages (k = 4) halves the prompt and the time. A smaller model (3B) or a GPU server would be faster.

## Likely theory questions with short answers

**Why BM25 and dense embeddings together?**
BM25 matches exact words, so it is strong on clause numbers ("7.2", "83(5)"), names and rare terms. Dense
embeddings match meaning, so "maximum fine" finds "administrative fines up to". Each fails where the other is
strong, so combining them gives better recall.

**What is Reciprocal Rank Fusion?**
Each document gets the score sum of 1 / (k + rank) over both rankings (k = 60). It uses ranks, not raw scores, so
the very different BM25 and cosine score scales do not have to be normalised. Documents ranked high by both lists win.

**Bi-encoder vs cross-encoder?**
A bi-encoder embeds query and passage separately, which is fast and precomputable, so it is used for search over
all chunks. A cross-encoder reads query and passage together with full attention between them, which is more
accurate but slow, so it is used only to rerank the top 30.

**Why clause-aware chunking instead of fixed-size?**
Fixed windows cut clauses in half and mix two articles, so citations become wrong and retrieval gets noisy. Legal
text already has units (article, paragraph, clause). Chunking along them gives exact citations like Article 83(5).

**How does abstention work?**
Two layers. (1) If the best cross-encoder relevance is below a threshold, the LLM is not called at all. (2) The LLM
can set `abstain`, and any answer whose claims all fail verification becomes an abstention. Measured as abstention
accuracy on unanswerable questions and false-abstention rate on answerable ones.

**How do you prevent hallucinated citations?**
The model may only cite passage ids it was given. Every claim must include a quote, and the verifier checks that the
quote appears in the cited passage and that every number in the claim appears there. Failed claims trigger one retry
with a reformulated query, and then they are dropped.

**What is recall@k?**
The share of the needed gold passages that appear in the top k results. For a two-hop question with one of two
gold passages found, the recall is 0.5.

**What is citation precision?**
Of all passages the answer cites, the share that come from the correct (gold) articles.

**Why a local LLM, and why temperature 0?**
Local means no data leaves the laptop and costs nothing, which matters for legal documents. Temperature 0 and a fixed
seed make answers reproducible for the evaluation.

**How do transformers produce embeddings?** (lecture topic)
Tokens pass through self-attention layers. The embedding is a pooled hidden state (CLS token or mean pooling),
trained contrastively so that a query and its relevant passage end up close in vector space (cosine similarity).

**What is self-attention?**
Each token builds a query, a key and a value vector. Attention weights = softmax(QKᵀ / √d), the output = weights × V,
so every token can use information from every other token. Multi-head attention does this in parallel subspaces.

**How would you improve the system?**
Fine-tune the reranker on legal QA pairs (e.g. CUAD or LEDGAR), add a vision-language model for figures, use an
NLI model as the verifier instead of lexical checks, and run a walk-forward evaluation on more documents.

**Limitations?**
Rule-based clause detection depends on layout. The lexical verifier cannot check legal reasoning. A 7B model can
misread long provisions. The evaluation set is small (22 questions).
