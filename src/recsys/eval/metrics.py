"""Ranking metrics. Each is implemented once, in two equivalent forms:

  * list form (`recall_at_k`, ...): takes a ranked list and a relevant set. Slow, obvious,
    and checked in tests against hand-worked examples and scikit-learn's `ndcg_score`.
  * position form (`user_metrics_from_positions`, `single_positive_metrics`): takes the
    1-based position of every relevant item in the full ranked list. This is what the
    harness uses, because it never has to materialise a 50k-long ranked list per user.
    Tests check it agrees exactly with the list form.

Conventions (binary relevance throughout):
  * positions are 1-based; a relevant item that was never ranked has position = inf.
  * DCG discount is 1/log2(position + 1).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- list form

def recall_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """|top-k ∩ relevant| / |relevant|.

    Assumes every relevant item is equally wanted. Misleads when |relevant| > k: the maximum
    achievable value is then k/|relevant| < 1, so users with many positives drag the mean
    down regardless of model quality. Some papers divide by min(k, |relevant|) instead;
    compare only numbers computed with the same denominator.
    """
    rel = set(relevant)
    if not rel:
        raise ValueError("recall is undefined for a user with no relevant items")
    return sum(1 for x in list(ranked)[:k] if x in rel) / len(rel)


def hit_rate_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """1 if any relevant item is in the top-k else 0.

    Saturates quickly: says nothing about how many relevant items were found or where.
    With one relevant item per query (leave-one-out, sampled protocols) it equals Recall@k.
    """
    rel = set(relevant)
    return float(any(x in rel for x in list(ranked)[:k]))


def ndcg_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """DCG@k / ideal DCG@k with binary gains and 1/log2(pos+1) discount.

    Assumes position matters logarithmically. The ideal ordering is computed over
    min(|relevant|, k) slots, so NDCG@k can reach 1 even when |relevant| > k (unlike Recall@k).
    """
    rel = set(relevant)
    if not rel:
        raise ValueError("NDCG is undefined for a user with no relevant items")
    dcg = sum(1.0 / np.log2(p + 2) for p, x in enumerate(list(ranked)[:k]) if x in rel)
    idcg = sum(1.0 / np.log2(p + 2) for p in range(min(len(rel), k)))
    return dcg / idcg


def reciprocal_rank(ranked: Sequence, relevant: Iterable) -> float:
    """1 / position of the first relevant item (0 if none), with no cutoff.

    Only looks at the first hit, so it ignores every other relevant item. Averaged over users
    it is MRR. Dominated by users whose first hit is at position 1-3.
    """
    rel = set(relevant)
    for p, x in enumerate(ranked, start=1):
        if x in rel:
            return 1.0 / p
    return 0.0


# --------------------------------------------------------------------------- position form

def _discount(positions: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(np.isfinite(positions), 1.0 / np.log2(np.asarray(positions, float) + 1.0), 0.0)


def _idcg_table(max_n: int) -> np.ndarray:
    """idcg[n] = sum_{p=1..n} 1/log2(p+1); idcg[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, max_n + 2)))])


def user_metrics_from_positions(positions: np.ndarray, user_ptr: np.ndarray,
                                ks: Sequence[int]) -> dict[str, np.ndarray]:
    """Per-user Recall@k, NDCG@k, HitRate@k and reciprocal rank.

    `positions[user_ptr[j]:user_ptr[j+1]]` are the 1-based positions of user j's relevant items
    in that user's full ranked list (inf if unranked). Every user must have >= 1 relevant item.
    Returns arrays of length n_users; average them for the reported number.
    """
    positions = np.asarray(positions, dtype=float)
    user_ptr = np.asarray(user_ptr)
    n_rel = np.diff(user_ptr)
    if (n_rel <= 0).any():
        raise ValueError("every user needs at least one relevant item")
    starts = user_ptr[:-1]
    idcg = _idcg_table(max(ks))
    disc = _discount(positions)
    out: dict[str, np.ndarray] = {}
    for k in ks:
        in_k = positions <= k
        hits = np.add.reduceat(in_k.astype(float), starts)
        dcg = np.add.reduceat(np.where(in_k, disc, 0.0), starts)
        out[f"recall@{k}"] = hits / n_rel
        out[f"hit@{k}"] = (hits > 0).astype(float)
        out[f"ndcg@{k}"] = dcg / idcg[np.minimum(n_rel, k)]
    first = np.minimum.reduceat(positions, starts)
    with np.errstate(divide="ignore"):
        out["mrr"] = np.where(np.isfinite(first), 1.0 / first, 0.0)
    return out


def single_positive_metrics(ranks: np.ndarray, ks: Sequence[int]) -> dict[str, np.ndarray]:
    """Per-query HitRate@k, NDCG@k and reciprocal rank when each query has exactly one
    relevant item ranked at `ranks` (1-based) among its negatives.

    This is the form both the sampled-negative protocol and the per-positive full-catalog
    protocol report, so the two are directly comparable: they differ only in which
    negatives each positive competes with.
    """
    ranks = np.asarray(ranks, dtype=float)
    out: dict[str, np.ndarray] = {}
    disc = _discount(ranks)
    for k in ks:
        in_k = ranks <= k
        out[f"hit@{k}"] = in_k.astype(float)
        out[f"ndcg@{k}"] = np.where(in_k, disc, 0.0)
    with np.errstate(divide="ignore"):
        out["mrr"] = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    return out


def mean_metrics(per_unit: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(np.mean(v)) if len(v) else float("nan") for k, v in per_unit.items()}


def bootstrap_ci(values: np.ndarray, n: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling units (users or positives)."""
    values = np.asarray(values, float)
    if len(values) == 0 or n <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        means[b] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Kendall's tau-a between two orderings of the same method names (1 = identical)."""
    if set(order_a) != set(order_b):
        raise ValueError("orderings must contain the same methods")
    pos_b = {m: i for i, m in enumerate(order_b)}
    n, s = len(order_a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if pos_b[order_a[i]] < pos_b[order_a[j]] else -1
    return s / (n * (n - 1) / 2) if n > 1 else 1.0
