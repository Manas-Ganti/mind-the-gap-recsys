"""FAISS ANN over two-tower item embeddings: recall against exact search, and single-query
p50/p95/p99 latency. The point is the recall/latency tradeoff, not a service.

Imports faiss only (no torch) and reads the embeddings `recsys train` saved as .npy. Loading
faiss and torch's OpenMP runtimes into one process can crash on macOS.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _latency(index, Q: np.ndarray, k: int) -> dict:
    t = np.empty(len(Q))
    for j in range(len(Q)):
        s = time.perf_counter()
        index.search(Q[j:j + 1], k)
        t[j] = (time.perf_counter() - s) * 1e3
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99))}


def _recall(found: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(f) & set(g)) / k for f, g in zip(found, truth)]))


def benchmark(items: np.ndarray, queries: np.ndarray, *, k: int, hnsw_m: int, ef_search: list[int],
              ivf_nlist: int, nprobe: list[int], n_latency_queries: int, seed: int) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)           # per-query latency, as a serving thread would see it
    items = np.ascontiguousarray(items, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    d = items.shape[1]
    k = min(k, len(items))
    lat_q = queries[np.random.default_rng(seed).permutation(len(queries))[:n_latency_queries]]

    exact = faiss.IndexFlatIP(d)
    exact.add(items)
    _, truth = exact.search(queries, k)
    out = {"n_items": int(len(items)), "n_queries": int(len(queries)), "dim": int(d), "k": int(k),
           "exact": {"recall": 1.0, **_latency(exact, lat_q, k)}, "hnsw": [], "ivf": []}

    hnsw = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    t = time.perf_counter(); hnsw.add(items); build = time.perf_counter() - t
    for ef in ef_search:
        hnsw.hnsw.efSearch = max(ef, k)
        _, found = hnsw.search(queries, k)
        out["hnsw"].append({"ef_search": ef, "recall": _recall(found, truth), "build_s": build,
                            **_latency(hnsw, lat_q, k)})

    nlist = max(1, min(ivf_nlist, len(items) // 39))   # faiss wants >= 39 points per centroid
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t = time.perf_counter(); ivf.train(items); ivf.add(items); build = time.perf_counter() - t
    for p in nprobe:
        ivf.nprobe = min(p, nlist)
        _, found = ivf.search(queries, k)
        out["ivf"].append({"nprobe": p, "nlist": nlist, "recall": _recall(found, truth), "build_s": build,
                           **_latency(ivf, lat_q, k)})
    return out


def run(cfg) -> dict:
    import json

    emb = Path(cfg.paths.models_dir) / "embeddings"
    items, queries = np.load(emb / "items.npy"), np.load(emb / "queries.npy")
    s = cfg.serve
    out = benchmark(items, queries, k=s.k, hnsw_m=s.hnsw_m, ef_search=s.ef_search, ivf_nlist=s.ivf_nlist,
                    nprobe=s.nprobe, n_latency_queries=s.n_latency_queries, seed=cfg.seed)
    out["seed"] = cfg.seed
    out["date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = Path(cfg.paths.results_dir) / "ann.json"
    path.write_text(json.dumps(out, indent=2))
    return out
