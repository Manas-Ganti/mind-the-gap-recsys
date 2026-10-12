# Results (generated 2026-10-12T23:02:50.670823+00:00 from real runs)

Seed 42, run date 2026-10-12T23:00:23.621053+00:00, config c9880e4a194c.

## Dataset

MIND-small: 94,057 users, 65,238 items, 230,117 impressions, 347,727 clicks, CTR 0.0405, sparsity 0.999616.

## Split audit

| split | test pairs | eval users | catalog | cold users | cold items | cold positives | cold catalog |
|---|---|---|---|---|---|---|---|
| temporal | 110549 | 20000 | 5369 | 0.025 | 0.615 | 0.490 | 0.609 |
| temporal_val | 46301 | 20110 | 6144 | 0.005 | 0.615 | 0.641 | 0.662 |
| random | 110110 | 20000 | 22771 | 0.005 | 0.181 | 0.011 | 0.590 |

## Gates

| gate | result |
|---|---|
| A | PASS |
| B | PASS |
| C | PASS |
| D | PASS |

## temporal: paired protocols

| method | full Recall@10 | full NDCG@10 | full HR@10 (per positive) | sampled HR@10 | full NDCG@10 (pp) | sampled NDCG@10 |
|---|---|---|---|---|---|---|
| popular | 0.0740 | 0.0351 | 0.0660 | 0.3337 | 0.0265 | 0.2413 |
| itemknn | 0.0045 | 0.0027 | 0.0047 | 0.1523 | 0.0023 | 0.0739 |
| graph | 0.0060 | 0.0035 | 0.0057 | 0.1484 | 0.0028 | 0.0740 |
| two_tower | 0.0081 | 0.0041 | 0.0090 | 0.2342 | 0.0035 | 0.1136 |
| gbdt (end-to-end) | 0.0281 | 0.0161 | - | 0.5359 | - | 0.2899 |
| neural (end-to-end) | 0.0441 | 0.0268 | - | 0.5071 | - | 0.3042 |

Orderings (retrieval, per-positive HR@10): full ['popular', 'two_tower', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'itemknn', 'graph'], Kendall τ = 0.67

Orderings (system): full NDCG@10 ['popular', 'neural', 'gbdt', 'two_tower', 'graph', 'itemknn'] vs sampled HR@10 ['gbdt', 'neural', 'popular', 'two_tower', 'itemknn', 'graph'], Kendall τ = 0.47

### temporal: retrieval ceiling

| pipeline | candidates | retrieval recall@K | ceiling Recall@10 | end-to-end Recall@10 | fraction of ceiling | retrieval loss | ranking loss |
|---|---|---|---|---|---|---|---|
| popular (own order) | 200 | 0.2982 | 0.2982 | 0.0740 | 0.248 | 0.7018 | 0.2242 |
| itemknn (own order) | 200 | 0.0735 | 0.0735 | 0.0045 | 0.061 | 0.9265 | 0.0690 |
| graph (own order) | 200 | 0.0746 | 0.0746 | 0.0060 | 0.081 | 0.9254 | 0.0686 |
| two_tower (own order) | 200 | 0.0978 | 0.0978 | 0.0081 | 0.083 | 0.9022 | 0.0896 |
| union -> gbdt | 619 | 0.4197 | 0.4197 | 0.0281 | 0.067 | 0.5803 | 0.3916 |
| union -> neural | 619 | 0.4197 | 0.4197 | 0.0441 | 0.105 | 0.5803 | 0.3755 |

### temporal: popularity decomposition

| method | HR@10 | IPS HR@10 | share of hits a popularity ranker also gets | HR@10 cold | HR@10 tail | HR@10 torso | HR@10 head |
|---|---|---|---|---|---|---|---|
| popular | 0.0660 | 0.0139 | n/a | 0.0000 | 0.0000 | 0.1577 | 0.2357 |
| itemknn | 0.0047 | 0.0012 | 0.057 | 0.0000 | 0.0023 | 0.0066 | 0.0188 |
| graph | 0.0057 | 0.0014 | 0.079 | 0.0000 | 0.0018 | 0.0104 | 0.0213 |
| two_tower | 0.0090 | 0.0083 | 0.022 | 0.0103 | 0.0021 | 0.0008 | 0.0208 |
| *share of positives* | | | | 0.487 | 0.178 | 0.166 | 0.169 |

## random: paired protocols

| method | full Recall@10 | full NDCG@10 | full HR@10 (per positive) | sampled HR@10 | full NDCG@10 (pp) | sampled NDCG@10 |
|---|---|---|---|---|---|---|
| popular | 0.1300 | 0.0742 | 0.1234 | 0.7368 | 0.0665 | 0.4960 |
| itemknn | 0.0214 | 0.0123 | 0.0221 | 0.1727 | 0.0114 | 0.1099 |
| graph | 0.0408 | 0.0263 | 0.0407 | 0.1849 | 0.0239 | 0.1275 |
| two_tower | 0.1537 | 0.0881 | 0.1479 | 0.7119 | 0.0795 | 0.4948 |

Orderings (retrieval, per-positive HR@10): full ['two_tower', 'popular', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'graph', 'itemknn'], Kendall τ = 0.67

### random: retrieval ceiling

| pipeline | candidates | retrieval recall@K | ceiling Recall@10 | end-to-end Recall@10 | fraction of ceiling | retrieval loss | ranking loss |
|---|---|---|---|---|---|---|---|
| popular (own order) | 200 | 0.5679 | 0.5679 | 0.1300 | 0.229 | 0.4321 | 0.4379 |
| itemknn (own order) | 200 | 0.1097 | 0.1097 | 0.0214 | 0.195 | 0.8903 | 0.0883 |
| graph (own order) | 200 | 0.1256 | 0.1256 | 0.0408 | 0.325 | 0.8744 | 0.0848 |
| two_tower (own order) | 200 | 0.5703 | 0.5703 | 0.1537 | 0.269 | 0.4297 | 0.4166 |

### random: popularity decomposition

| method | HR@10 | IPS HR@10 | share of hits a popularity ranker also gets | HR@10 cold | HR@10 tail | HR@10 torso | HR@10 head |
|---|---|---|---|---|---|---|---|
| popular | 0.1234 | 0.0232 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.3747 |
| itemknn | 0.0221 | 0.0289 | 0.005 | 0.0000 | 0.0385 | 0.0262 | 0.0022 |
| graph | 0.0407 | 0.0332 | 0.151 | 0.0000 | 0.0389 | 0.0390 | 0.0454 |
| two_tower | 0.1479 | 0.0356 | 0.594 | 0.0000 | 0.0028 | 0.0227 | 0.4234 |
| *share of positives* | | | | 0.010 | 0.331 | 0.330 | 0.329 |

## Temporal vs random

| method | temporal NDCG@10 | random NDCG@10 | random / temporal | temporal Recall@10 | random Recall@10 |
|---|---|---|---|---|---|
| popular | 0.0351 | 0.0742 | 2.11x | 0.0740 | 0.1300 |
| itemknn | 0.0027 | 0.0123 | 4.54x | 0.0045 | 0.0214 |
| graph | 0.0035 | 0.0263 | 7.44x | 0.0060 | 0.0408 |
| two_tower | 0.0041 | 0.0881 | 21.34x | 0.0081 | 0.1537 |

## Tuning (temporal_val)

- **popular**: 9 grid points, best {'window_hours': 1.0} -> full-catalog user ndcg@10 = 0.0180
- **itemknn**: 120 grid points, best {'neighbors': 500, 'shrink': 0.0, 'alpha': 0.85} -> full-catalog user ndcg@10 = 0.0064
- **graph**: 48 grid points, best {'neighbors': 300, 'alpha': 0.8, 'beta': 0.2} -> full-catalog user ndcg@10 = 0.0063
- **two_tower**: 4 grid points, best {'temperature': 0.02} -> full-catalog user ndcg@10 = 0.0037 — **WARNING: best on a grid edge: temperature=0.02 (lowest tried)**

## ANN serving

| index | setting | recall | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| exact | - | 1.000 | 0.057 | 0.060 | 0.063 |
| hnsw | ef_search=16 | 0.939 | 0.061 | 0.076 | 0.081 |
| hnsw | ef_search=32 | 0.939 | 0.061 | 0.077 | 0.081 |
| hnsw | ef_search=64 | 0.939 | 0.061 | 0.076 | 0.081 |
| hnsw | ef_search=128 | 0.958 | 0.072 | 0.088 | 0.095 |
| hnsw | ef_search=256 | 0.986 | 0.123 | 0.144 | 0.157 |
| ivf | nprobe=1 | 0.113 | 0.010 | 0.012 | 0.016 |
| ivf | nprobe=4 | 0.373 | 0.015 | 0.018 | 0.020 |
| ivf | nprobe=16 | 0.753 | 0.025 | 0.029 | 0.031 |
| ivf | nprobe=64 | 0.982 | 0.043 | 0.048 | 0.051 |
