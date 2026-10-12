"""Chunked item-item similarity with per-target top-n pruning, shared by item-kNN and RP3beta.

W[:, j] for each target (catalog) item j is computed as left @ right[:, j], normalised by a
method-specific function, the self-similarity is zeroed, and only the n largest sources are
kept. Chunking keeps the dense block at (n_items x chunk) floats.

Several normalisers can share one pass over the (expensive) product, which is what makes a
serious hyperparameter grid affordable: one co-occurrence computation serves every
(alpha, shrink) pair, and smaller `neighbors` values are prefixes of the largest one.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np
import scipy.sparse as sp

Normaliser = Callable[[np.ndarray, np.ndarray], np.ndarray]


def topn_similarity_multi(left: sp.csr_matrix, right: sp.csc_matrix, targets: np.ndarray, n: int,
                          normalisers: dict[Hashable, Normaliser], chunk: int = 512) -> dict[Hashable, sp.csc_matrix]:
    """For each normaliser, W of shape (n_items, len(targets)) with <= n entries per column."""
    n_items = left.shape[0]
    keep = min(n, n_items)
    parts: dict[Hashable, list] = {key: [] for key in normalisers}
    for c0 in range(0, len(targets), chunk):
        J = targets[c0:c0 + chunk]
        raw = np.asarray((left @ right[:, J]).todense(), dtype=np.float32)
        for key, fn in normalisers.items():
            block = fn(raw, J)
            block[J, np.arange(len(J))] = 0.0
            if keep < n_items:
                idx = np.argpartition(-block, keep - 1, axis=0)[:keep]
            else:
                idx = np.tile(np.arange(n_items)[:, None], (1, len(J)))
            v = np.take_along_axis(block, idx, 0)
            nz = v > 0
            cols = np.broadcast_to(np.arange(len(J)), idx.shape) + c0
            parts[key].append((idx[nz], cols[nz], v[nz]))
    out = {}
    for key, ps in parts.items():
        r = np.concatenate([p[0] for p in ps]) if ps else np.zeros(0, int)
        c = np.concatenate([p[1] for p in ps]) if ps else np.zeros(0, int)
        v = np.concatenate([p[2] for p in ps]) if ps else np.zeros(0, np.float32)
        out[key] = sp.csc_matrix((v, (r, c)), shape=(n_items, len(targets)), dtype=np.float32)
    return out


def topn_similarity(left, right, targets, n: int, normalise: Normaliser, chunk: int = 512) -> sp.csr_matrix:
    return topn_similarity_multi(left, right, targets, n, {0: normalise}, chunk)[0].tocsr()


def prune_columns(W: sp.csc_matrix, n: int) -> sp.csr_matrix:
    """Keep the n largest entries of every column."""
    W = W.tocsc()
    W.sort_indices()
    rows, cols, vals = [], [], []
    for j in range(W.shape[1]):
        s, e = W.indptr[j], W.indptr[j + 1]
        if e - s == 0:
            continue
        v, r = W.data[s:e], W.indices[s:e]
        if e - s > n:
            sel = np.argpartition(-v, n - 1)[:n]
            v, r = v[sel], r[sel]
        rows.append(r); vals.append(v); cols.append(np.full(len(r), j))
    if not rows:
        return sp.csr_matrix(W.shape, dtype=np.float32)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=W.shape, dtype=np.float32)
