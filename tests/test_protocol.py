"""Full-catalog and sampled protocols on tiny hand-built splits."""
import numpy as np
import pytest

from conftest import make_split
from recsys.eval.protocol import run_protocol

KW = dict(ks=[1, 5, 10], candidates_k=5, user_batch=2, tie_noise=1e-9, seed=0)


def _scorer(matrix):
    return lambda users: matrix[users]


def test_sampled_equals_full_when_negatives_are_the_whole_catalog():
    rng = np.random.default_rng(0)
    n_users, n_items = 4, 12
    train = [(0, 0), (1, 1), (2, 2), (3, 3)]
    test = [(0, 5), (0, 6), (1, 7), (2, 8), (3, 9)]
    # negatives = every catalog item that is neither a positive nor train-seen for that user
    pos_by_user = {u: {i for uu, i in test if uu == u} for u in range(n_users)}
    seen = {u: {i for uu, i in train if uu == u} for u in range(n_users)}
    negs = {(u, i): [j for j in range(n_items) if j not in pos_by_user[u] | seen[u]] for u, i in test}
    d = make_split(train, test, n_users, n_items, negatives=negs)
    S = rng.normal(size=(n_users, n_items)).astype(np.float32)
    run = run_protocol(d, {"m": _scorer(S)}, n_negatives=n_items, **KW)["m"]
    sampled = dict(zip(zip(run.sampled_user, run.sampled_item), run.sampled_rank))
    full = dict(zip(zip(run.pos_user, run.pos_item), run.full_rank))
    assert sampled == full


def test_hand_checked_ranks():
    # user 0: positive item 2; scores 0..5 => item 5 best. Item 4 is train-seen (masked).
    S = np.array([[0.0, 1.0, 2.5, 3.0, 9.0, 5.0]], dtype=np.float32)
    d = make_split([(0, 4)], [(0, 2)], 1, 6, negatives={(0, 2): [0, 1, 3]})
    run = run_protocol(d, {"m": _scorer(S)}, n_negatives=3, **KW)["m"]
    # full: above item 2 (2.5) among valid non-positives {0,1,3,5}: items 3, 5 -> rank 3
    assert run.full_rank.tolist() == [3]
    assert run.position.tolist() == [3]
    # sampled: negatives {0,1,3}; only item 3 beats it -> rank 2
    assert run.sampled_rank.tolist() == [2]
    # candidates never include the masked item 4
    assert 4 not in run.candidates[0]
    assert run.candidates[0][:3].tolist() == [5, 3, 2]


def test_position_counts_other_positives():
    S = np.array([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    d = make_split([], [(0, 0), (0, 2)], 1, 4, negatives={(0, 0): [1], (0, 2): [1]})
    run = run_protocol(d, {"m": _scorer(S)}, n_negatives=1, **KW)["m"]
    # item 2: one non-positive (item 1) above it -> full_rank 2; plus positive 0 above -> position 3
    assert run.full_rank.tolist() == [1, 2]
    assert run.position.tolist() == [1, 3]


def test_oracle_scores_rank_first_under_both():
    n_users, n_items = 6, 30
    test = [(u, 10 + u) for u in range(n_users)]
    negs = {(u, i): [j for j in range(n_items) if j != i][:20] for u, i in test}
    d = make_split([], test, n_users, n_items, negatives=negs)
    S = np.zeros((n_users, n_items), np.float32)
    for u, i in test:
        S[u, i] = 1.0
    s = run_protocol(d, {"m": _scorer(S)}, n_negatives=20, **KW)["m"].summary([1, 10])
    assert s["full"]["user"]["ndcg@10"] == 1.0
    assert s["sampled"]["per_positive"]["hit@1"] == 1.0


def test_constant_scores_are_tie_broken_at_random():
    n_users, n_items = 400, 50
    test = [(u, u % n_items) for u in range(n_users)]
    d = make_split([], test, n_users, n_items, negatives={(u, i): [(i + 1) % n_items] for u, i in test})
    S = np.zeros((n_users, n_items), np.float32)
    run = run_protocol(d, {"m": _scorer(S)}, n_negatives=1, **{**KW, "user_batch": 64})["m"]
    # expected HR@10 for a uniformly random rank in 1..50 is 10/50
    assert np.mean(run.full_rank <= 10) == pytest.approx(0.2, abs=0.06)
    # an index-ordered tie-break would always put item 0 first; random should not
    assert 0.3 < np.mean(run.sampled_rank == 1) < 0.7


def test_scorer_shape_checked():
    d = make_split([], [(0, 1)], 1, 3, negatives={(0, 1): [0]})
    with pytest.raises(ValueError):
        run_protocol(d, {"m": lambda u: np.zeros((1, 2))}, n_negatives=1, **KW)


def test_leaky_split_detected():
    # a test positive that is also train-seen must be refused, not silently masked
    d = make_split([(0, 1)], [(0, 1)], 1, 3, negatives={(0, 1): [0]})
    with pytest.raises(AssertionError):
        run_protocol(d, {"m": lambda u: np.zeros((1, 3))}, n_negatives=1, **KW)


def test_items_not_live_on_the_click_day_are_not_ranked():
    import pandas as pd

    # items 0-2 live on day A, 3-5 on day B; user 0 clicked item 1 on day A
    S = np.array([[0.0, 1.0, 0.5, 9.0, 8.0, 7.0]], dtype=np.float32)
    d = make_split([], [(0, 1)], 1, 6, negatives={(0, 1): [0, 2]})
    d.test["day"] = "A"
    d.catalog_days = pd.DataFrame({"day": ["A"] * 3 + ["B"] * 3, "i": range(6)})
    run = run_protocol(d, {"m": _scorer(S)}, n_negatives=2, **KW)["m"]
    assert run.full_rank.tolist() == [1]          # items 3-5 score higher but were not live
    assert set(run.candidates[0][run.candidates[0] >= 0]) <= {0, 1, 2}
