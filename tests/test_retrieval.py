"""Sparse retrievers against brute-force dense reference implementations."""
import numpy as np
import pytest

from conftest import make_split
from recsys.retrieval.graph import RP3betaRetriever
from recsys.retrieval.itemknn import ItemKNNRetriever
from recsys.retrieval.popular import PopularRetriever


def _random_split(seed, n_users=25, n_items=18):
    rng = np.random.default_rng(seed)
    pairs = sorted({(int(u), int(i)) for u, i in zip(rng.integers(0, n_users, 120), rng.integers(0, n_items, 120))})
    return make_split(pairs, [(0, 0)] if (0, 0) not in pairs else [(0, 1)], n_users, n_items,
                      catalog=np.arange(0, n_items, 2)), pairs


def _dense_X(pairs, n_users, n_items):
    X = np.zeros((n_users, n_items))
    for u, i in pairs:
        X[u, i] = 1
    return X


def _topn_cols(W, n):
    out = np.zeros_like(W)
    for j in range(W.shape[1]):
        idx = np.argsort(-W[:, j], kind="stable")[:n]
        out[idx, j] = W[idx, j]
    return np.where(out > 0, out, 0)


@pytest.mark.parametrize("alpha,shrink", [(0.5, 0.0), (0.3, 5.0), (0.8, 1.0)])
def test_itemknn_matches_dense(alpha, shrink):
    d, pairs = _random_split(1)
    X = _dense_X(pairs, d.n_users, d.n_items)
    co = X.T @ X
    deg = X.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(np.outer(deg ** alpha, deg ** (1 - alpha)) + shrink > 0,
                     co / (np.outer(deg ** alpha, deg ** (1 - alpha)) + shrink), 0)
    np.fill_diagonal(W, 0)
    W = W[:, d.catalog]
    for n in (3, 100):
        ref = X @ _topn_cols(W, n)
        r = ItemKNNRetriever(neighbors=n, shrink=shrink, alpha=alpha).fit(d)
        users = np.arange(d.n_users)
        if n >= d.n_items:   # no pruning ambiguity from ties
            np.testing.assert_allclose(r.score(users), ref, rtol=1e-5, atol=1e-6)
        else:                # with ties the kept set may differ; totals of kept weights must match
            np.testing.assert_allclose(np.sort(r.W_.toarray(), 0)[-n:], np.sort(_topn_cols(W, n), 0)[-n:], rtol=1e-5)


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, 0.5)])
def test_rp3beta_matches_dense(alpha, beta):
    d, pairs = _random_split(2)
    X = _dense_X(pairs, d.n_users, d.n_items)
    pui = X / np.maximum(X.sum(1, keepdims=True), 1)
    piu = X.T / np.maximum(X.T.sum(1, keepdims=True), 1)
    W = (piu ** alpha) @ (pui ** alpha)
    W = W / np.maximum(X.sum(0), 1)[None, :] ** beta
    np.fill_diagonal(W, 0)
    W = W[:, d.catalog]
    r = RP3betaRetriever(neighbors=1000, alpha=alpha, beta=beta).fit(d)
    np.testing.assert_allclose(r.score(np.arange(d.n_users)), X @ W, rtol=1e-5, atol=1e-6)


def test_popular_window():
    pairs = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)]
    ts = [np.nan, 990.0, 995.0, 100.0, 999.0]          # reference_ts = 1000
    d = make_split(pairs, [(5, 3)], 6, 4, ts=ts)
    np.testing.assert_array_equal(PopularRetriever(None).fit(d).scores_, [2, 1, 2, 0])
    # last 0.005h = 18s -> only clicks at ts >= 982: (1,0),(2,1),(4,2)
    np.testing.assert_array_equal(PopularRetriever(0.005).fit(d).scores_, [1, 1, 1, 0])
