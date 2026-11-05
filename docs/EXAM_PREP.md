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
6. **(30 s) Evaluation.** One slide or table: recall@k (dense vs hybrid vs hybrid+rerank) and no-retrieval vs full.

Tip: before the exam, run the app once with a new PDF so the models are downloaded and warm.

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
