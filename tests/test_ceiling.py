"""The ceiling is an upper bound on end-to-end, by construction, and is attained by the
ideal reordering."""
import numpy as np
import pytest
import scipy.sparse as sp

from conftest import make_split
from recsys.eval.ceiling import ceiling_per_user, ceiling_report, end_to_end_per_user
from recsys.eval.protocol import run_protocol

KS = [1, 3, 5, 10]


def _setup(seed, n_users=30, n_items=80, K=15):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for u in range(n_users):
        for i in rng.choice(n_items, int(rng.integers(1, 6)), replace=False):
            rows.append(u); cols.append(i)
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
    cands = np.stack([rng.choice(n_items, K, replace=False) for _ in range(n_users)])
    return rng, P, cands, np.arange(n_users)


@pytest.mark.parametrize("seed", range(10))
def test_any_reordering_is_bounded_by_ceiling(seed):
    rng, P, cands, users = _setup(seed)
    ceil = ceiling_per_user(cands, users, P, KS)
    for _ in range(20):
        perm = np.stack([rng.permutation(c) for c in cands])
        e2e = end_to_end_per_user(perm, users, P, KS)
        for m in e2e:
            assert (e2e[m] <= ceil[m] + 1e-12).all(), m


@pytest.mark.parametrize("seed", range(10))
def test_ideal_reordering_attains_ceiling(seed):
    _, P, cands, users = _setup(seed)
    ideal = []
    for u, c in enumerate(cands):
        pos = set(P[u].indices)
        ideal.append(sorted(c, key=lambda x: x not in pos))
    e2e = end_to_end_per_user(np.array(ideal), users, P, KS)
    ceil = ceiling_per_user(cands, users, P, KS)
    for m in e2e:
        np.testing.assert_allclose(e2e[m], ceil[m], atol=1e-12, err_msg=m)


def test_hand_example():
    P = sp.csr_matrix(([1, 1, 1], ([0, 0, 0], [1, 2, 3])), shape=(1, 10))
    cands = np.array([[9, 8, 1, 2, -1]])      # 2 of 3 positives retrieved
    c = ceiling_per_user(cands, np.array([0]), P, [1, 5])
    assert c["retrieval_recall"][0] == pytest.approx(2 / 3)
    assert c["recall@1"][0] == pytest.approx(1 / 3)
    assert c["recall@5"][0] == pytest.approx(2 / 3)
    assert c["ndcg@1"][0] == 1.0
    rep = ceiling_report(cands, None, np.array([0]), P, [5])
    d = rep["decomposition"]["recall@5"]
    assert d["retrieval_loss"] == pytest.approx(1 / 3)
    assert d["ranking_loss"] == pytest.approx(0.0)      # both hits are within the top 5 already


def test_retrieval_own_order_matches_full_catalog_metrics():
    """For k <= K, a retriever's candidate list *is* the top of its full-catalog ranking."""
    rng = np.random.default_rng(3)
    n_users, n_items = 10, 40
    test = [(u, int(i)) for u in range(n_users) for i in rng.choice(n_items, 3, replace=False)]
    d = make_split([], test, n_users, n_items, negatives={})
    S = rng.normal(size=(n_users, n_items)).astype(np.float32)
    run = run_protocol(d, {"m": lambda u: S[u]}, ks=KS, n_negatives=1, candidates_k=10, user_batch=4,
                       tie_noise=1e-9, seed=0)["m"]
    full = run.summary(KS)["full"]["user"]
    e2e = ceiling_report(run.candidates, None, d.eval_users, d.test_pos, KS)["end_to_end"]
    for k in KS:
        assert e2e[f"recall@{k}"] == pytest.approx(full[f"recall@{k}"])
        assert e2e[f"ndcg@{k}"] == pytest.approx(full[f"ndcg@{k}"])
