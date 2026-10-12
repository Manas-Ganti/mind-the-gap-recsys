"""Retrieval recall ceiling.

A ranker only reorders the candidates stage 1 hands it; it cannot recover an item retrieval
never returned. So for a candidate set C_u and test positives P_u with h = |C_u ∩ P_u|:

  ceiling Recall@k  = min(h, k) / |P_u|
  ceiling NDCG@k    = DCG of min(h, k) hits in the top slots / IDCG(min(|P_u|, k))
  ceiling HitRate@k = 1[h > 0]
  ceiling MRR       = 1[h > 0]

These are the best scores *any* ranker could achieve on that candidate set. Reporting the
end-to-end number next to its ceiling splits the total loss into
  retrieval loss = 1 - ceiling        (positives never retrieved)
  ranking loss   = ceiling - end_to_end (retrieved but ranked too low)
which says whether to spend effort on stage 1 or stage 2.

End-to-end metrics are computed on the reranked candidate list *only*: positives outside the
candidate set have position = inf. That is how the system runs in production.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from recsys.eval.metrics import _idcg_table, mean_metrics, user_metrics_from_positions


def _positions_in_lists(lists: np.ndarray, users: np.ndarray, test_pos: sp.csr_matrix):
    """For each user's positives, their 1-based position in `lists[row]` (inf if absent).
    Returns (positions, user_ptr, hits_per_user)."""
    P = test_pos[users]
    P.sort_indices()
    positions, counts, hits = [], [], []
    for r in range(len(users)):
        pos_items = P.indices[P.indptr[r]:P.indptr[r + 1]]
        lst = lists[r]
        where = {int(x): j + 1 for j, x in enumerate(lst) if x >= 0}
        p = np.array([where.get(int(i), np.inf) for i in pos_items], dtype=float)
        positions.append(p)
        counts.append(len(p))
        hits.append(int(np.isfinite(p).sum()))
    ptr = np.concatenate([[0], np.cumsum(counts)])
    return (np.concatenate(positions) if positions else np.zeros(0)), ptr, np.asarray(hits)


def ceiling_per_user(candidates: np.ndarray, users: np.ndarray, test_pos: sp.csr_matrix,
                     ks: Sequence[int]) -> dict[str, np.ndarray]:
    """Upper bound on every metric for any reordering of `candidates[row]` (−1 = empty slot)."""
    _, ptr, h = _positions_in_lists(candidates, users, test_pos)
    n = np.diff(ptr)
    if (n <= 0).any():
        raise ValueError("every user needs at least one positive")
    idcg = _idcg_table(max(ks))
    out = {}
    for k in ks:
        best = np.minimum(h, k)
        out[f"recall@{k}"] = best / n
        out[f"hit@{k}"] = (h > 0).astype(float)
        out[f"ndcg@{k}"] = idcg[best] / idcg[np.minimum(n, k)]
    out["mrr"] = (h > 0).astype(float)
    out["retrieval_recall"] = h / n
    return out


def end_to_end_per_user(ranked: np.ndarray, users: np.ndarray, test_pos: sp.csr_matrix,
                        ks: Sequence[int]) -> dict[str, np.ndarray]:
    """Metrics of the final ranked list (stage 2 output over stage 1 candidates)."""
    positions, ptr, _ = _positions_in_lists(ranked, users, test_pos)
    return user_metrics_from_positions(positions, ptr, ks)


def ceiling_report(candidates: np.ndarray, ranked: np.ndarray | None, users: np.ndarray,
                   test_pos: sp.csr_matrix, ks: Sequence[int]) -> dict:
    """Mean ceiling, end-to-end, and the retrieval/ranking loss split for one pipeline.
    `ranked=None` means the retrieval method's own ordering of its candidates."""
    ceil = mean_metrics(ceiling_per_user(candidates, users, test_pos, ks))
    e2e = mean_metrics(end_to_end_per_user(candidates if ranked is None else ranked, users, test_pos, ks))
    split = {}
    for m, v in e2e.items():
        c = ceil[m]
        split[m] = {"end_to_end": v, "ceiling": c, "retrieval_loss": 1.0 - c, "ranking_loss": c - v,
                    "fraction_of_ceiling": (v / c) if c > 0 else float("nan")}
    return {"ceiling": ceil, "end_to_end": e2e, "decomposition": split,
            "candidate_set_size": int((candidates >= 0).sum(1).mean()) if len(candidates) else 0}
