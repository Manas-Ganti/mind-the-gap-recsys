"""Full-catalog vs sampled-negative evaluation over the same model and the same test set.

One pass per user batch: every method scores the whole catalog once, and from that single
score matrix the harness derives
  * full-catalog ranks   each positive vs every rankable non-positive catalog item
  * sampled ranks        each positive vs its N pre-drawn negatives (shared across methods)
  * top-K candidates     what stage 1 hands to the ranker (see ceiling.py)
so the two protocols can never disagree because of different scoring calls.

Tie-breaking: scores get a seeded jitter of `tie_noise * (1 + |s|)`, i.e. ties are broken
uniformly at random. Without this, a popularity model that gives every fresh item score 0
would be ranked optimistically or pessimistically depending on item index order. The jitter
matrix is shared by all methods within a batch.

Masking: an item is rankable for a user only if it was live on a day the user has a test
click (`data.available`, identical rule in every split), and, when `exclude_train_seen`,
the user did not click it in train. Sampled negatives already obey both rules.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recsys.eval.metrics import mean_metrics, single_positive_metrics, user_metrics_from_positions

Scorer = Callable[[np.ndarray], np.ndarray]   # users -> (len(users), len(catalog)) scores
BatchHook = Callable[..., None]

_LOW = float(np.finfo(np.float32).min) / 2


@dataclass
class MethodRun:
    """Raw per-positive outcomes for one method on one split. Positives are ordered by
    (eval user, item index) identically for every method, so arrays align across methods."""
    name: str
    users: np.ndarray
    user_ptr: np.ndarray
    pos_user: np.ndarray
    pos_item: np.ndarray
    full_rank: np.ndarray        # 1 + #non-positive catalog items scored above the positive
    position: np.ndarray         # 1-based position in the user's full ranked list
    full_n: np.ndarray           # rankable non-positive catalog items that positive competed with
    sampled_user: np.ndarray
    sampled_item: np.ndarray
    sampled_rank: np.ndarray     # 1 + #sampled negatives scored above the positive
    sampled_n: np.ndarray        # negatives actually available for that pair
    candidates: np.ndarray       # (n_users, K) item indices, best first, -1 padded

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items() if k != "name"}, name=self.name)

    @classmethod
    def load(cls, path: Path) -> "MethodRun":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files if k != "name"}
        return cls(name=str(z["name"]), **kw)

    def summary(self, ks: list[int]) -> dict:
        user = user_metrics_from_positions(self.position, self.user_ptr, ks)
        return {
            "n_users": int(len(self.users)),
            "n_positives": int(len(self.pos_item)),
            "full": {
                "user": mean_metrics(user),
                "per_positive": mean_metrics(single_positive_metrics(self.full_rank, ks)),
            },
            "sampled": {
                "per_positive": mean_metrics(single_positive_metrics(self.sampled_rank, ks)),
                "n_pairs": int(len(self.sampled_rank)),
                "mean_negatives": float(self.sampled_n.mean()) if len(self.sampled_n) else 0.0,
            },
        }


def run_protocol(data, scorers: dict[str, Scorer], *, ks: list[int], n_negatives: int,
                 candidates_k: int, user_batch: int, tie_noise: float, seed: int,
                 on_batch: BatchHook | None = None, positive_chunk: int = 256) -> dict[str, MethodRun]:
    """Evaluate every scorer on `data.eval_users` under both protocols in one pass."""
    users_all = np.asarray(data.eval_users)
    C = len(data.catalog)
    K = min(candidates_k, C)
    ccol = data.catalog_col
    X = data.X
    P = data.test_pos
    P.sort_indices()
    neg_u = data.neg_pairs[:, 0]

    acc: dict[str, dict[str, list]] = {m: {k: [] for k in
                                           ("pos_user", "pos_item", "full_rank", "position", "full_n", "sampled_user",
                                            "sampled_item", "sampled_rank", "sampled_n", "candidates")}
                                       for m in scorers}
    n_pos_per_user = []

    for b, start in enumerate(range(0, len(users_all), user_batch)):
        users = users_all[start:start + user_batch]
        B = len(users)

        avail = data.available(users)
        valid = np.ones((B, C), dtype=bool) if avail is None else avail
        if data.exclude_train_seen:
            Xb = X[users]
            rows = np.repeat(np.arange(B), np.diff(Xb.indptr))
            cols = ccol[Xb.indices]
            keep = cols >= 0
            valid[rows[keep], cols[keep]] = False

        Pb = P[users]
        p_rows = np.repeat(np.arange(B), np.diff(Pb.indptr))
        p_items = Pb.indices.astype(np.int64)
        p_cols = ccol[p_items]
        if (p_cols < 0).any():
            raise AssertionError("a test positive is not in the catalog")
        if (np.diff(Pb.indptr) == 0).any():
            raise AssertionError("an eval user has no test positive")
        if not valid[p_rows, p_cols].all():
            raise AssertionError("a test positive is masked: train-seen (leakage) or not live on its day")
        pos_mask = np.zeros((B, C), dtype=bool)
        pos_mask[p_rows, p_cols] = True
        neg_valid = valid & ~pos_mask
        n_pos_per_user.append(np.diff(Pb.indptr))
        full_n = neg_valid.sum(1)[p_rows]

        lo, hi = np.searchsorted(neg_u, users[0], "left"), np.searchsorted(neg_u, users[-1], "right")
        s_users = data.neg_pairs[lo:hi, 0]
        s_rows = np.searchsorted(users, s_users)
        in_batch = (s_rows < B) & (users[np.minimum(s_rows, B - 1)] == s_users)
        s_rows, s_users = s_rows[in_batch], s_users[in_batch]
        s_items = data.neg_pairs[lo:hi, 1][in_batch]
        s_pcols = ccol[s_items]
        negs = data.neg_items[lo:hi][in_batch][:, :n_negatives]
        s_ncols = np.where(negs >= 0, ccol[np.maximum(negs, 0)], -1)
        if (s_pcols < 0).any() or ((negs >= 0) & (s_ncols < 0)).any():
            raise AssertionError("sampled positive/negative outside the catalog")
        s_mask = s_ncols >= 0

        jitter = np.random.default_rng([seed, b]).random((B, C))
        raw_scores: dict[str, np.ndarray] = {}
        cand_cols: dict[str, np.ndarray] = {}

        for name, score_fn in scorers.items():
            S = np.asarray(score_fn(users), dtype=np.float32)
            if S.shape != (B, C):
                raise ValueError(f"{name}: scorer returned {S.shape}, expected {(B, C)}")
            S = np.nan_to_num(S, nan=_LOW, posinf=-_LOW, neginf=_LOW)
            raw_scores[name] = S
            Sp = S.astype(np.float64)
            Sp += jitter * tie_noise * (1.0 + np.abs(Sp))

            full_rank = np.empty(len(p_rows), dtype=np.int64)
            position = np.empty(len(p_rows), dtype=np.int64)
            for c0 in range(0, len(p_rows), positive_chunk):
                r, c = p_rows[c0:c0 + positive_chunk], p_cols[c0:c0 + positive_chunk]
                s = Sp[r, c]
                above = Sp[r] > s[:, None]
                g = (above & neg_valid[r]).sum(1)
                full_rank[c0:c0 + len(r)] = 1 + g
                position[c0:c0 + len(r)] = 1 + g + (above & pos_mask[r]).sum(1)

            s_pos = Sp[s_rows, s_pcols]
            s_neg = Sp[s_rows[:, None], np.maximum(s_ncols, 0)]
            sampled_rank = 1 + ((s_neg > s_pos[:, None]) & s_mask).sum(1)

            M = np.where(valid, Sp, -np.inf)
            top = np.argpartition(-M, K - 1, axis=1)[:, :K] if K < C else np.tile(np.arange(C), (B, 1))
            order = np.argsort(-np.take_along_axis(M, top, 1), axis=1, kind="stable")
            top = np.take_along_axis(top, order, 1)
            top_valid = np.isfinite(np.take_along_axis(M, top, 1))
            cand_cols[name] = np.where(top_valid, top, -1)

            a = acc[name]
            a["pos_user"].append(users[p_rows]); a["pos_item"].append(p_items)
            a["full_rank"].append(full_rank); a["position"].append(position); a["full_n"].append(full_n)
            a["sampled_user"].append(s_users); a["sampled_item"].append(s_items)
            a["sampled_rank"].append(sampled_rank); a["sampled_n"].append(s_mask.sum(1))
            a["candidates"].append(np.where(cand_cols[name] >= 0, data.catalog[np.maximum(cand_cols[name], 0)], -1))

        if on_batch is not None:
            on_batch(users=users, scores=raw_scores, cand_cols=cand_cols, valid=valid,
                     sampled=(s_rows, s_pcols, s_ncols))

    user_ptr = np.concatenate([[0], np.cumsum(np.concatenate(n_pos_per_user))]) if n_pos_per_user else np.zeros(1, int)
    runs = {}
    for name, a in acc.items():
        cat = {k: (np.concatenate(v) if v else np.zeros(0, int)) for k, v in a.items()}
        runs[name] = MethodRun(name=name, users=users_all, user_ptr=user_ptr, **cat)
    return runs


def paired_rows(summaries: dict[str, dict], k: int) -> list[dict]:
    """One row per method: the same metrics under both protocols, side by side."""
    rows = []
    for m, s in summaries.items():
        rows.append({
            "method": m,
            f"full_recall@{k}": s["full"]["user"][f"recall@{k}"],
            f"full_ndcg@{k}": s["full"]["user"][f"ndcg@{k}"],
            f"full_hit@{k}_pp": s["full"]["per_positive"][f"hit@{k}"],
            f"sampled_hit@{k}_pp": s["sampled"]["per_positive"][f"hit@{k}"],
            f"full_ndcg@{k}_pp": s["full"]["per_positive"][f"ndcg@{k}"],
            f"sampled_ndcg@{k}_pp": s["sampled"]["per_positive"][f"ndcg@{k}"],
        })
    return rows


def method_order(summaries: dict[str, dict], protocol: str, metric: str) -> list[str]:
    """Methods sorted best-first under one protocol's per-positive metric."""
    return sorted(summaries, key=lambda m: -summaries[m][protocol]["per_positive"][metric])
