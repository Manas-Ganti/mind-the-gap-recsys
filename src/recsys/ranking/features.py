"""Candidate features for stage 2. Built only for items stage 1 returned (the union of every
retriever's top-K), and only from information available at the split's reference time:
train clicks, train-side impressions, and the retrievers' own scores.

Per (user, candidate):
  retrieval    score_<m>, in_top_<m>               for every retriever m
  item         log clicks, log impressions, smoothed CTR, age (hours since first shown,
               -1 if never shown before the reference time), cold flag, clicks in the last
               1h / 6h / 24h before the reference time
  user         log #train clicks, log #in-window clicks
  user x item  share of the user's clicks in the item's category and subcategory
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.eval.metrics import ndcg_at_k

RECENT_HOURS = (1, 6, 24)


@dataclass
class CandidateBlock:
    users: np.ndarray        # (U,) user per group
    ptr: np.ndarray          # (U+1,) group offsets
    items: np.ndarray        # (N,) item index per row
    X: np.ndarray            # (N, F) float32
    y: np.ndarray            # (N,) 0/1

    @staticmethod
    def concat(blocks: list["CandidateBlock"]) -> "CandidateBlock":
        sizes = [np.diff(b.ptr) for b in blocks]
        sizes = np.concatenate(sizes) if sizes else np.zeros(0, int)
        return CandidateBlock(
            users=np.concatenate([b.users for b in blocks]),
            ptr=np.concatenate([[0], np.cumsum(sizes)]),
            items=np.concatenate([b.items for b in blocks]),
            X=np.concatenate([b.X for b in blocks]),
            y=np.concatenate([b.y for b in blocks]))

    def with_columns(self, columns: list[int] | None) -> "CandidateBlock":
        """Same rows, only the given feature columns (None = all)."""
        if columns is None:
            return self
        return CandidateBlock(self.users, self.ptr, self.items, np.ascontiguousarray(self.X[:, columns]), self.y)

    def has_positive(self) -> np.ndarray:
        return np.add.reduceat(self.y, self.ptr[:-1]) > 0 if len(self.y) else np.zeros(len(self.users), bool)

    def subset_groups(self, mask: np.ndarray) -> "CandidateBlock":
        idx = np.flatnonzero(mask)
        rows = np.concatenate([np.arange(self.ptr[g], self.ptr[g + 1]) for g in idx]) if len(idx) else np.zeros(0, int)
        sizes = np.diff(self.ptr)[idx]
        return CandidateBlock(self.users[idx], np.concatenate([[0], np.cumsum(sizes)]),
                              self.items[rows], self.X[rows], self.y[rows])


def _share(counts: sp.csr_matrix) -> sp.csr_matrix:
    tot = np.asarray(counts.sum(1)).ravel()
    inv = np.zeros_like(tot, dtype=np.float32)
    inv[tot > 0] = 1.0 / tot[tot > 0]
    return (sp.diags(inv) @ counts).tocsr()


class FeatureBuilder:
    def __init__(self, data, methods: list[str]):
        self.data, self.methods = data, list(methods)
        cat = np.asarray(data.catalog)
        st = data.item_stats
        ref = data.reference_ts
        clicks = st["clicks"].to_numpy(float)[cat]
        impr = st["impressions"].to_numpy(float)[cat]
        first = st["first_seen"].to_numpy(float)[cat]
        age = np.where(np.isfinite(first), (ref - first) / 3600.0, -1.0)
        ctr = (clicks + 1.0) / (impr + 25.0)         # Beta(1, 24) prior ~ MIND's ~4% CTR

        t = data.train
        ts = t["ts"].to_numpy()
        recent = []
        for h in RECENT_HOURS:
            m = ts >= ref - h * 3600.0
            recent.append(np.log1p(np.bincount(t["i"].to_numpy()[m], minlength=data.n_items)[cat]))
        self.item_feats = np.column_stack(
            [np.log1p(clicks), np.log1p(impr), ctr, age, (impr == 0).astype(float), *recent]).astype(np.float32)
        self.item_names = ["item_log_clicks", "item_log_impr", "item_ctr", "item_age_h", "item_cold",
                           *[f"item_recent_{h}h" for h in RECENT_HOURS]]

        n_train = np.bincount(t["u"].to_numpy(), minlength=data.n_users)
        n_win = np.bincount(t["u"].to_numpy()[np.isfinite(ts)], minlength=data.n_users)
        self.user_feats = np.column_stack([np.log1p(n_train), np.log1p(n_win)]).astype(np.float32)
        self.user_names = ["user_log_clicks", "user_log_window_clicks"]

        news = data.news
        cat_code = news["category"].astype("category").cat.codes.to_numpy()
        sub_code = news["subcategory"].astype("category").cat.codes.to_numpy()
        X = data.X
        self.user_cat = _share(X @ sp.csr_matrix((np.ones(data.n_items), (np.arange(data.n_items), cat_code))))
        self.user_sub = _share(X @ sp.csr_matrix((np.ones(data.n_items), (np.arange(data.n_items), sub_code))))
        self.cand_cat, self.cand_sub = cat_code[cat], sub_code[cat]

        self.names = ([f"score_{m}" for m in self.methods] + [f"in_top_{m}" for m in self.methods]
                      + self.item_names + self.user_names + ["user_cat_share", "user_subcat_share"])

    def top_masks(self, B: int, cand_cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        C = len(self.data.catalog)
        in_top = {}
        for m in self.methods:
            mm = np.zeros((B, C), dtype=bool)
            r, c = np.nonzero(cand_cols[m] >= 0)
            mm[r, cand_cols[m][r, c]] = True
            in_top[m] = mm
        return in_top

    def features(self, users: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 scores: dict[str, np.ndarray], in_top: dict[str, np.ndarray]) -> np.ndarray:
        """Feature matrix for (users[rows], catalog[cols]) pairs."""
        uu = users[rows]
        feats = ([scores[m][rows, cols] for m in self.methods]
                 + [in_top[m][rows, cols].astype(np.float32) for m in self.methods])
        return np.column_stack(feats + [self.item_feats[cols], self.user_feats[uu],
                                        np.asarray(self.user_cat[uu, self.cand_cat[cols]]).ravel(),
                                        np.asarray(self.user_sub[uu, self.cand_sub[cols]]).ravel()]).astype(np.float32)

    def build(self, users: np.ndarray, scores: dict[str, np.ndarray], cand_cols: dict[str, np.ndarray],
              with_labels: bool = True) -> CandidateBlock:
        """One group per user: the union of every retriever's top-K candidates."""
        B = len(users)
        in_top = self.top_masks(B, cand_cols)
        union = np.zeros_like(next(iter(in_top.values())))
        for mm in in_top.values():
            union |= mm
        rows, cols = np.nonzero(union)                 # row-major => grouped by user
        sizes = np.bincount(rows, minlength=B)
        X = self.features(users, rows, cols, scores, in_top)
        items = np.asarray(self.data.catalog)[cols]
        uu = users[rows]
        y = (np.asarray(self.data.test_pos[uu, items]).ravel() > 0).astype(np.float32) if with_labels \
            else np.zeros(len(items), np.float32)
        return CandidateBlock(users=users, ptr=np.concatenate([[0], np.cumsum(sizes)]), items=items, X=X, y=y)

    def build_sampled(self, users: np.ndarray, scores: dict[str, np.ndarray], cand_cols: dict[str, np.ndarray],
                      s_rows: np.ndarray, s_pcols: np.ndarray, s_ncols: np.ndarray) -> CandidateBlock:
        """One group per sampled pair: the positive first, then its valid negatives. This is
        what a sampled-negative evaluation of a ranker scores: it bypasses retrieval."""
        in_top = self.top_masks(len(users), cand_cols)
        rows, cols, sizes, y = [], [], [], []
        for r, pc, nc in zip(s_rows, s_pcols, s_ncols):
            nc = nc[nc >= 0]
            rows.append(np.full(1 + len(nc), r)); cols.append(np.concatenate([[pc], nc]))
            sizes.append(1 + len(nc)); y.append(np.concatenate([[1.0], np.zeros(len(nc))]))
        rows = np.concatenate(rows) if rows else np.zeros(0, int)
        cols = np.concatenate(cols) if cols else np.zeros(0, int)
        X = self.features(users, rows, cols, scores, in_top) if len(rows) else \
            np.zeros((0, len(self.names)), np.float32)
        return CandidateBlock(users=users[s_rows], ptr=np.concatenate([[0], np.cumsum(sizes)]).astype(int),
                              items=np.asarray(self.data.catalog)[cols],
                              X=X, y=np.concatenate(y).astype(np.float32) if y else np.zeros(0, np.float32))


def rerank(block: CandidateBlock, scores: np.ndarray, max_len: int | None = None) -> np.ndarray:
    """(U, L) item lists ordered by ranker score within each group, -1 padded."""
    sizes = np.diff(block.ptr)
    L = max_len or (int(sizes.max()) if len(sizes) else 0)
    out = np.full((len(sizes), L), -1, dtype=np.int64)
    for g in range(len(sizes)):
        s, e = block.ptr[g], block.ptr[g + 1]
        order = np.argsort(-scores[s:e], kind="stable")[:L]
        out[g, :len(order)] = block.items[s:e][order]
    return out


def group_ndcg(block: CandidateBlock, scores: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k of each group's candidates ordered by `scores`, over groups with a positive
    among their candidates. Measures ranking only: positives retrieval missed are not counted,
    so it is comparable across rankers on one block, not to the full-catalog numbers."""
    vals = []
    for g in range(len(block.users)):
        a, b = block.ptr[g], block.ptr[g + 1]
        y = block.y[a:b]
        if not y.any():
            continue
        order = np.argsort(-scores[a:b], kind="stable")
        vals.append(ndcg_at_k(list(order), set(np.flatnonzero(y)), k))
    return float(np.mean(vals)) if vals else 0.0
