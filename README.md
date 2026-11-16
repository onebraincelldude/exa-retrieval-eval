# retrieval-eval
Retrieval quality eval: golden queries with known relevant documents, scored on recall@k
and MRR. Use it to compare retrievers, rankers or embedding changes before shipping.
Thresholds in `cases.json`; any retriever callable via `python run.py my_retriever.py`
(exposing `run(query, k)` returning ranked doc ids). `python run.py` runs a naive keyword
retriever, which the eval blocks.
