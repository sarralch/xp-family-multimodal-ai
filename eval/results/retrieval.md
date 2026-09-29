Known-item retrieval on the project datasets: 1744 records, 285 queries (name 60, synonym 45, cas 60, description 60, product 60); seed 13.

| System | Hit@1 | Hit@3 | MRR@10 | Hit@3 name | Hit@3 synonym | Hit@3 cas | Hit@3 description | Hit@3 product | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1: dense MiniLM-L6 (EN), whole records | 0.46 | 0.55 | 0.52 | 0.93 | 0.82 | 0.05 | 0.18 | 0.85 | 124 ms |
| BM25 | 0.68 | 0.75 | 0.74 | 0.98 | 0.87 | 0.10 | 0.85 | 1.00 | 3 ms |
| Dense multilingual, chunked | 0.32 | 0.36 | 0.36 | 0.22 | 0.44 | 0.03 | 0.15 | 1.00 | 128 ms |
| Hybrid (BM25 + dense, RRF) | 0.68 | 0.81 | 0.75 | 1.00 | 0.96 | 0.53 | 0.60 | 1.00 | 128 ms |
| Hybrid + cross-encoder rerank | 0.68 | 0.85 | 0.77 | 1.00 | 0.82 | 0.85 | 0.70 | 0.88 | 856 ms |
