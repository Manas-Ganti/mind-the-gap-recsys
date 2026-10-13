# Results (generated 2026-10-13T02:52:38.055451+00:00 from real runs)

Seed 42, run date 2026-10-13T02:46:13.618063+00:00, config 78a29a2f03c2.

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
| two_tower | 0.0044 ± 0.0005 | 0.0025 ± 0.0004 | 0.0043 | 0.2702 ± 0.0268 | 0.0020 | 0.1257 |
| gbdt (end-to-end) | 0.0487 ± 0.0129 | 0.0288 ± 0.0079 | - | 0.5456 ± 0.0202 | - | 0.3110 |
| neural (end-to-end) | 0.0746 ± 0.0090 | 0.0435 ± 0.0083 | - | 0.5519 ± 0.0216 | - | 0.3409 |

Orderings (retrieval, per-positive HR@10): full ['popular', 'graph', 'itemknn', 'two_tower'] vs sampled ['popular', 'two_tower', 'itemknn', 'graph'], Kendall τ = 0.00

Orderings (system): full NDCG@10 ['neural', 'popular', 'gbdt', 'graph', 'itemknn', 'two_tower'] vs sampled HR@10 ['neural', 'gbdt', 'popular', 'two_tower', 'itemknn', 'graph'], Kendall τ = 0.47

### temporal: variation across seeds [42, 43, 44]

Two-tower and the rankers are trained once per seed. Tables above show mean ± sd; the baselines are deterministic, so they have no seed spread.

| model | seed 42: full NDCG@10 / sampled HR@10 | seed 43: full NDCG@10 / sampled HR@10 | seed 44: full NDCG@10 / sampled HR@10 |
|---|---|---|---|
| two_tower | 0.0023 / 0.2608 | 0.0023 / 0.2493 | 0.0030 / 0.3004 |
| gbdt | 0.0200 / 0.5455 | 0.0313 / 0.5658 | 0.0353 / 0.5254 |
| neural | 0.0402 / 0.5282 | 0.0373 / 0.5569 | 0.0529 / 0.5706 |

Method ordering per seed (retrieval, per-positive HR@10, full vs sampled):

- seed 42: full ['popular', 'graph', 'itemknn', 'two_tower'] vs sampled ['popular', 'two_tower', 'itemknn', 'graph'], τ = 0.00; system τ = 0.33
- seed 43: full ['popular', 'graph', 'itemknn', 'two_tower'] vs sampled ['popular', 'two_tower', 'itemknn', 'graph'], τ = 0.00; system τ = 0.33
- seed 44: full ['popular', 'graph', 'two_tower', 'itemknn'] vs sampled ['popular', 'two_tower', 'itemknn', 'graph'], τ = 0.33; system τ = 0.73

### temporal: retrieval ceiling

| pipeline | candidates | retrieval recall@K | ceiling Recall@10 | end-to-end Recall@10 | fraction of ceiling | retrieval loss | ranking loss |
|---|---|---|---|---|---|---|---|
| popular (own order) | 200 | 0.2982 | 0.2982 | 0.0740 | 0.248 | 0.7018 | 0.2242 |
| itemknn (own order) | 200 | 0.0735 | 0.0735 | 0.0045 | 0.061 | 0.9265 | 0.0690 |
| graph (own order) | 200 | 0.0746 | 0.0746 | 0.0060 | 0.081 | 0.9254 | 0.0686 |
| two_tower (own order) | 200 | 0.1119 | 0.1119 | 0.0044 | 0.040 | 0.8881 | 0.1075 |
| union -> gbdt | 620 | 0.4254 | 0.4254 | 0.0487 | 0.114 | 0.5746 | 0.3767 |
| union -> neural | 620 | 0.4254 | 0.4254 | 0.0746 | 0.175 | 0.5746 | 0.3508 |

### temporal: popularity decomposition

| method | HR@10 | IPS HR@10 | share of hits a popularity ranker also gets | HR@10 cold | HR@10 tail | HR@10 torso | HR@10 head |
|---|---|---|---|---|---|---|---|
| popular | 0.0660 | 0.0139 | n/a | 0.0000 | 0.0000 | 0.1577 | 0.2357 |
| itemknn | 0.0047 | 0.0012 | 0.057 | 0.0000 | 0.0023 | 0.0066 | 0.0188 |
| graph | 0.0057 | 0.0014 | 0.079 | 0.0000 | 0.0018 | 0.0104 | 0.0213 |
| two_tower | 0.0043 | 0.0019 | 0.041 | 0.0021 | 0.0005 | 0.0006 | 0.0185 |
| *share of positives* | | | | 0.487 | 0.178 | 0.166 | 0.169 |

## random: paired protocols

| method | full Recall@10 | full NDCG@10 | full HR@10 (per positive) | sampled HR@10 | full NDCG@10 (pp) | sampled NDCG@10 |
|---|---|---|---|---|---|---|
| popular | 0.1300 | 0.0742 | 0.1234 | 0.7368 | 0.0665 | 0.4960 |
| itemknn | 0.0214 | 0.0123 | 0.0221 | 0.1727 | 0.0114 | 0.1099 |
| graph | 0.0408 | 0.0263 | 0.0407 | 0.1849 | 0.0239 | 0.1275 |
| two_tower | 0.1571 ± 0.0124 | 0.0923 ± 0.0103 | 0.1489 | 0.6947 ± 0.0119 | 0.0826 | 0.4772 |

Orderings (retrieval, per-positive HR@10): full ['two_tower', 'popular', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'graph', 'itemknn'], Kendall τ = 0.67

### random: variation across seeds [42, 43, 44]

Two-tower and the rankers are trained once per seed. Tables above show mean ± sd; the baselines are deterministic, so they have no seed spread.

| model | seed 42: full NDCG@10 / sampled HR@10 | seed 43: full NDCG@10 / sampled HR@10 | seed 44: full NDCG@10 / sampled HR@10 |
|---|---|---|---|
| two_tower | 0.0806 / 0.7051 | 0.0997 / 0.6972 | 0.0966 / 0.6817 |

Method ordering per seed (retrieval, per-positive HR@10, full vs sampled):

- seed 42: full ['two_tower', 'popular', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'graph', 'itemknn'], τ = 0.67
- seed 43: full ['two_tower', 'popular', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'graph', 'itemknn'], τ = 0.67
- seed 44: full ['two_tower', 'popular', 'graph', 'itemknn'] vs sampled ['popular', 'two_tower', 'graph', 'itemknn'], τ = 0.67

### random: retrieval ceiling

| pipeline | candidates | retrieval recall@K | ceiling Recall@10 | end-to-end Recall@10 | fraction of ceiling | retrieval loss | ranking loss |
|---|---|---|---|---|---|---|---|
| popular (own order) | 200 | 0.5679 | 0.5679 | 0.1300 | 0.229 | 0.4321 | 0.4379 |
| itemknn (own order) | 200 | 0.1097 | 0.1097 | 0.0214 | 0.195 | 0.8903 | 0.0883 |
| graph (own order) | 200 | 0.1256 | 0.1256 | 0.0408 | 0.325 | 0.8744 | 0.0848 |
| two_tower (own order) | 200 | 0.5430 | 0.5430 | 0.1571 | 0.289 | 0.4570 | 0.3859 |

### random: popularity decomposition

| method | HR@10 | IPS HR@10 | share of hits a popularity ranker also gets | HR@10 cold | HR@10 tail | HR@10 torso | HR@10 head |
|---|---|---|---|---|---|---|---|
| popular | 0.1234 | 0.0232 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.3747 |
| itemknn | 0.0221 | 0.0289 | 0.005 | 0.0000 | 0.0385 | 0.0262 | 0.0022 |
| graph | 0.0407 | 0.0332 | 0.151 | 0.0000 | 0.0389 | 0.0390 | 0.0454 |
| two_tower | 0.1489 | 0.0361 | 0.600 | 0.0025 | 0.0034 | 0.0185 | 0.4301 |
| *share of positives* | | | | 0.010 | 0.331 | 0.330 | 0.329 |

## Temporal vs random

| method | temporal NDCG@10 | random NDCG@10 | random / temporal | temporal Recall@10 | random Recall@10 |
|---|---|---|---|---|---|
| popular | 0.0351 | 0.0742 | 2.11x | 0.0740 | 0.1300 |
| itemknn | 0.0027 | 0.0123 | 4.54x | 0.0045 | 0.0214 |
| graph | 0.0035 | 0.0263 | 7.44x | 0.0060 | 0.0408 |
| two_tower | 0.0025 | 0.0923 | 36.42x | 0.0044 | 0.1571 |

## Tuning (temporal_val)

- **popular**: 9 grid points, best {'window_hours': 1.0} -> full-catalog user ndcg@10 = 0.0180
- **itemknn**: 120 grid points, best {'neighbors': 500, 'shrink': 0.0, 'alpha': 0.85} -> full-catalog user ndcg@10 = 0.0064
- **graph**: 48 grid points, best {'neighbors': 300, 'alpha': 0.8, 'beta': 0.2} -> full-catalog user ndcg@10 = 0.0063
- **two_tower**: 6 grid points, best {'temperature': 0.01} -> full-catalog user ndcg@10 = 0.0039

## ANN serving

| index | setting | recall | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| exact | - | 1.000 | 0.058 | 0.062 | 0.065 |
| hnsw | ef_search=16 | 0.966 | 0.062 | 0.080 | 0.088 |
| hnsw | ef_search=32 | 0.966 | 0.062 | 0.079 | 0.088 |
| hnsw | ef_search=64 | 0.966 | 0.062 | 0.079 | 0.089 |
| hnsw | ef_search=128 | 0.979 | 0.074 | 0.093 | 0.103 |
| hnsw | ef_search=256 | 0.996 | 0.125 | 0.149 | 0.169 |
| ivf | nprobe=1 | 0.141 | 0.011 | 0.012 | 0.014 |
| ivf | nprobe=4 | 0.384 | 0.017 | 0.019 | 0.021 |
| ivf | nprobe=16 | 0.738 | 0.027 | 0.030 | 0.033 |
| ivf | nprobe=64 | 0.984 | 0.045 | 0.049 | 0.054 |
