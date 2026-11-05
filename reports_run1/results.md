# LexiSyncAI evaluation
Generated 2026-11-04 18:04. Case: GDPR ePrivacy. Questions: 22 (factual: 12, multihop: 6, unanswerable: 4).

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
| no_retrieval | 0.556 |  |  | 0.75 | 0.111 | 10.79 | 9.84 | 92 | 200 | 0 |
| dense_only | 0.556 |  | 0.956 | 1.0 | 0.167 | 19.79 | 19.79 | 1067 | 230 | 1 |
| full | 0.556 |  | 0.863 | 1.0 | 0.222 | 16.91 | 17.97 | 886 | 195 | 0 |
| full_no_planner | 0.611 |  | 0.881 | 1.0 | 0.222 | 16.28 | 16.32 | 893 | 222 | 0 |
| full_no_verifier | 0.556 |  | 0.863 | 1.0 | 0.222 | 14.87 | 14.79 | 886 | 195 | 0 |
| full_k4 | 0.667 |  | 0.844 | 1.0 | 0.167 | 16.46 | 17.29 | 651 | 216 | 0 |
