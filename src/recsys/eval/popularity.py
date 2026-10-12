"""How much of a metric is item popularity alone?

Three views, all computed from per-positive outcomes so they apply to every method:

1. Overlap decomposition. HitRate@k = (hits a pure popularity ranker also gets)
                                    + (hits beyond popularity).
   A method whose hits are 90% shared with the popularity baseline is mostly re-deriving
   the popularity signal, whatever its headline number.
2. Popularity-stratified hit rate. Positives are bucketed by the item's train clicks:
   cold (0 train clicks) and tertiles of the remaining positives (tail / torso / head).
   Assumes train clicks are a fair proxy for exposure-driven popularity at serving time.
3. IPS-debiased metrics. Each positive is weighted by 1/(pop+1)^gamma (normalised), the
   standard correction for popularity-biased test sets (Yang et al., RecSys 2018). Assumes
   observation propensity ∝ popularity^gamma; gamma is a modelling choice, not measured.
"""
from __future__ import annotations

import numpy as np

from recsys.eval.metrics import single_positive_metrics


BUCKETS = ["cold", "tail", "torso", "head"]


def bucket_thresholds(pos_pop: np.ndarray) -> tuple[float, float]:
    """Tertile cut points of train popularity over warm (pop > 0) test positives."""
    warm = pos_pop[pos_pop > 0]
    if len(warm) == 0:
        return (0.0, 0.0)
    q1, q2 = np.quantile(warm, [1 / 3, 2 / 3])
    return float(q1), float(q2)


def assign_buckets(pop: np.ndarray, q1: float, q2: float) -> np.ndarray:
    """0 = cold (no train clicks), 1 = tail, 2 = torso, 3 = head."""
    pop = np.asarray(pop)
    b = np.zeros(len(pop), dtype=int)
    warm = pop > 0
    b[warm] = 1 + (pop[warm] > q1).astype(int) + (pop[warm] > q2).astype(int)
    return b


def popularity_buckets(pos_pop: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """0 = cold, then tail/torso/head by tertiles of positives with pop > 0."""
    return assign_buckets(pos_pop, *bucket_thresholds(pos_pop)), BUCKETS


def concentration(top_items: np.ndarray, item_pop: np.ndarray, q1: float, q2: float) -> dict:
    """Popularity concentration of what a method *shows*: the share of its top-k slots that
    go to cold / tail / torso / head items, with buckets cut on the test positives so the
    distribution of actual clicks is the reference row. Also the share of slots taken by
    the 1% most-clicked train items."""
    items = top_items[top_items >= 0]
    if len(items) == 0:
        return {b: float("nan") for b in BUCKETS}
    b = assign_buckets(item_pop[items], q1, q2)
    out = {lab: float((b == j).mean()) for j, lab in enumerate(BUCKETS)}
    warm = item_pop[item_pop > 0]
    top1 = np.quantile(warm, 0.99) if len(warm) else np.inf
    out["top1pct_item_share"] = float((item_pop[items] >= top1).mean()) if len(warm) else 0.0
    out["distinct_items_shown"] = int(len(np.unique(items)))
    return out


def ips_weighted(values: np.ndarray, pos_pop: np.ndarray, gamma: float) -> float:
    w = 1.0 / np.power(pos_pop + 1.0, gamma)
    return float((w * values).sum() / w.sum()) if len(values) else float("nan")


def decompose(ranks: np.ndarray, pos_pop: np.ndarray, k: int, gamma: float,
              baseline_ranks: np.ndarray | None = None) -> dict:
    """`ranks` and `baseline_ranks` are full-catalog per-positive ranks aligned positive by
    positive (the protocol guarantees identical ordering across methods)."""
    m = single_positive_metrics(ranks, [k])
    hit, ndcg = m[f"hit@{k}"], m[f"ndcg@{k}"]
    out = {
        f"hit@{k}": float(hit.mean()) if len(hit) else float("nan"),
        f"ips_hit@{k}": ips_weighted(hit, pos_pop, gamma),
        f"ips_ndcg@{k}": ips_weighted(ndcg, pos_pop, gamma),
        "gamma": gamma,
    }
    b, labels = popularity_buckets(pos_pop)
    out["by_bucket"] = {
        lab: {"share_of_positives": float((b == j).mean()) if len(b) else 0.0,
              f"hit@{k}": float(hit[b == j].mean()) if (b == j).any() else float("nan")}
        for j, lab in enumerate(labels)
    }
    if baseline_ranks is not None:
        if len(baseline_ranks) != len(ranks):
            raise ValueError("baseline ranks are not aligned with method ranks")
        base_hit = single_positive_metrics(baseline_ranks, [k])[f"hit@{k}"]
        shared = (hit * base_hit).mean() if len(hit) else 0.0
        out["explained_by_popularity"] = float(shared)
        out["beyond_popularity"] = float(out[f"hit@{k}"] - shared)
        out["share_of_hits_explained"] = float(shared / out[f"hit@{k}"]) if out[f"hit@{k}"] > 0 else float("nan")
    return out
