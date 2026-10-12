"""Every metric checked against an independent reference: hand-worked values and
scikit-learn's ndcg_score; the vectorised position form must equal the list form exactly."""
import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from recsys.eval.metrics import (
    hit_rate_at_k, kendall_tau, ndcg_at_k, recall_at_k, reciprocal_rank, single_positive_metrics,
    user_metrics_from_positions,
)

RANKED = ["a", "b", "c", "d", "e"]
REL = {"b", "e"}


def test_hand_worked_example():
    # relevant at positions 2 and 5
    assert recall_at_k(RANKED, REL, 3) == 0.5
    assert recall_at_k(RANKED, REL, 5) == 1.0
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0
    assert hit_rate_at_k(RANKED, REL, 2) == 1.0
    assert reciprocal_rank(RANKED, REL) == 0.5
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx((1 / math.log2(3)) / idcg)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(6)) / idcg)
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(0.386852807, abs=1e-8)


def test_ndcg_more_relevant_than_k_can_reach_one():
    assert ndcg_at_k(["x", "y"], {"x", "y", "z"}, 2) == pytest.approx(1.0)
    assert recall_at_k(["x", "y"], {"x", "y", "z"}, 2) == pytest.approx(2 / 3)


def test_no_hit():
    assert reciprocal_rank(["a"], {"z"}) == 0.0
    assert ndcg_at_k(["a"], {"z"}, 1) == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_ndcg_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    y = (rng.random(n) < 0.2).astype(int)
    if y.sum() == 0:
        y[int(rng.integers(0, n))] = 1
    s = rng.permutation(n).astype(float)       # distinct scores: sklearn averages over ties
    ranked = list(np.argsort(-s))
    rel = set(np.flatnonzero(y))
    for k in (1, 3, 5, 10, n):
        assert ndcg_at_k(ranked, rel, k) == pytest.approx(ndcg_score([y], [s], k=k), abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_position_form_equals_list_form(seed):
    rng = np.random.default_rng(seed)
    ks = [1, 3, 5, 10]
    positions, ptr, expected = [], [0], {f"{m}@{k}": [] for m in ("recall", "ndcg", "hit") for k in ks}
    expected["mrr"] = []
    for _ in range(15):
        n = int(rng.integers(8, 40))
        ranked = list(rng.permutation(n))
        rel = set(rng.choice(n, int(rng.integers(1, 6)), replace=False).tolist())
        # truncate the list sometimes so some relevant items are unranked (position inf)
        cut = ranked[: int(rng.integers(3, n + 1))]
        pos = [cut.index(r) + 1 if r in cut else np.inf for r in sorted(rel)]
        positions += pos
        ptr.append(ptr[-1] + len(pos))
        for k in ks:
            expected[f"recall@{k}"].append(recall_at_k(cut, rel, k))
            expected[f"ndcg@{k}"].append(ndcg_at_k(cut, rel, k))
            expected[f"hit@{k}"].append(hit_rate_at_k(cut, rel, k))
        expected["mrr"].append(reciprocal_rank(cut, rel))
    got = user_metrics_from_positions(np.array(positions), np.array(ptr), ks)
    for key, vals in expected.items():
        np.testing.assert_allclose(got[key], vals, atol=1e-12, err_msg=key)


def test_single_positive_metrics_hand():
    m = single_positive_metrics(np.array([1, 3, 11, np.inf]), [10])
    np.testing.assert_allclose(m["hit@10"], [1, 1, 0, 0])
    np.testing.assert_allclose(m["ndcg@10"], [1.0, 0.5, 0, 0])
    np.testing.assert_allclose(m["mrr"], [1.0, 1 / 3, 1 / 11, 0])


def test_single_positive_equals_list_form():
    # one relevant item at rank r in a list: single-positive metrics == list-form metrics
    for r in range(1, 15):
        ranked = list(range(20))
        rel = {r - 1}
        m = single_positive_metrics(np.array([r]), [5, 10])
        assert m["hit@5"][0] == hit_rate_at_k(ranked, rel, 5)
        assert m["ndcg@10"][0] == pytest.approx(ndcg_at_k(ranked, rel, 10))
        assert m["mrr"][0] == pytest.approx(reciprocal_rank(ranked, rel))


def test_user_without_relevant_rejected():
    with pytest.raises(ValueError):
        user_metrics_from_positions(np.array([1.0]), np.array([0, 1, 1]), [5])


def test_kendall_tau():
    assert kendall_tau(list("abcd"), list("abcd")) == 1.0
    assert kendall_tau(list("abcd"), list("dcba")) == -1.0
    assert kendall_tau(list("abc"), list("bac")) == pytest.approx(1 / 3)
