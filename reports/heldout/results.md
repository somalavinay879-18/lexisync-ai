# LexiSyncAI evaluation
Generated 2026-11-07 12:00. Case: Heldout ICLA. Questions: 11 (factual: 5, multihop: 3, unanswerable: 3).

## Retrieval: recall@k per configuration
Article level = the right article is retrieved; paragraph level = the exact paragraph.

| mode | level | recall@1 | recall@3 | recall@5 | recall@10 | mrr |
|---|---|---|---|---|---|---|
| bm25 | article | 0.688 | 0.938 | 0.938 | 0.938 | 0.938 |
| bm25 | paragraph | 0.688 | 0.938 | 0.938 | 0.938 | 0.938 |
| dense | article | 0.812 | 0.938 | 0.938 | 1.0 | 1.0 |
| dense | paragraph | 0.812 | 0.938 | 0.938 | 1.0 | 1.0 |
| hybrid | article | 0.688 | 0.938 | 0.938 | 1.0 | 0.938 |
| hybrid | paragraph | 0.688 | 0.938 | 0.938 | 1.0 | 0.938 |
| hybrid_rerank | article | 0.812 | 0.938 | 1.0 | 1.0 | 1.0 |
| hybrid_rerank | paragraph | 0.812 | 0.938 | 1.0 | 1.0 | 1.0 |
| hybrid_rerank_mq | article | 0.812 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid_rerank_mq | paragraph | 0.812 | 1.0 | 1.0 | 1.0 | 1.0 |

## End-to-end results
`no_retrieval` vs `full` is the mandatory comparison; `full_no_planner` and `full_no_verifier` are ablations; `full_k4` is the latency optimisation (fewer passages in the prompt).

| config | answer_correct_auto | answer_correct_manual | citation_precision | abstention_accuracy | false_abstention | mean_seconds | median_seconds | mean_prompt_tokens | mean_output_tokens | verifier_retries |
|---|---|---|---|---|---|---|---|---|---|---|
| no_retrieval | 0.125 | 0.375 |  | 0.0 | 0.0 | 8.77 | 9.16 | 90 | 185 | 0 |
| full_k4 | 0.75 | 0.875 | 1.0 | 0.667 | 0.125 | 15.94 | 13.1 | 863 | 265 | 0 |
