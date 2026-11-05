# LexiSyncAI evaluation
Generated 2026-11-05 10:39. Case: GDPR ePrivacy. Questions: 22 (factual: 12, multihop: 6, unanswerable: 4).

## Retrieval: recall@k per configuration
Article level = the right article is retrieved; paragraph level = the exact paragraph.

| mode | level | recall@1 | recall@3 | recall@5 | recall@10 | mrr |
|---|---|---|---|---|---|---|
| bm25 | article | 0.667 | 0.806 | 0.833 | 0.944 | 0.839 |
| bm25 | paragraph | 0.444 | 0.639 | 0.694 | 0.806 | 0.637 |
| dense | article | 0.722 | 0.917 | 0.944 | 0.972 | 0.889 |
| dense | paragraph | 0.528 | 0.778 | 0.806 | 0.889 | 0.771 |
| hybrid | article | 0.694 | 0.806 | 0.889 | 0.972 | 0.854 |
| hybrid | paragraph | 0.5 | 0.611 | 0.806 | 0.889 | 0.689 |
| hybrid_rerank | article | 0.861 | 0.917 | 0.917 | 0.944 | 0.972 |
| hybrid_rerank | paragraph | 0.722 | 0.861 | 0.861 | 0.889 | 0.917 |
| hybrid_rerank_mq | article | 0.861 | 0.917 | 0.944 | 0.972 | 0.972 |
| hybrid_rerank_mq | paragraph | 0.722 | 0.861 | 0.889 | 0.917 | 0.907 |

## End-to-end results
`no_retrieval` vs `full` is the mandatory comparison; `full_no_planner` and `full_no_verifier` are ablations; `full_k4` is the latency optimisation (fewer passages in the prompt).

| config | answer_correct_auto | answer_correct_manual | citation_precision | abstention_accuracy | false_abstention | mean_seconds | median_seconds | mean_prompt_tokens | mean_output_tokens | verifier_retries |
|---|---|---|---|---|---|---|---|---|---|---|
| no_retrieval | 0.611 | 0.556 |  | 0.0 | 0.0 | 10.88 | 10.2 | 93 | 207 | 0 |
| dense_only | 0.722 |  | 0.833 | 1.0 | 0.056 | 26.89 | 22.01 | 1139 | 295 | 0 |
| full | 0.778 | 0.778 | 0.844 | 1.0 | 0.111 | 33.14 | 29.76 | 998 | 375 | 0 |
| full_no_planner | 0.833 |  | 0.809 | 1.0 | 0.056 | 29.21 | 26.23 | 1006 | 343 | 0 |
| full_no_verifier | 0.778 |  | 0.844 | 1.0 | 0.111 | 27.12 | 25.09 | 998 | 377 | 0 |
| full_k4 | 0.778 | 0.778 | 0.847 | 1.0 | 0.0 | 17.23 | 15.0 | 764 | 268 | 0 |
