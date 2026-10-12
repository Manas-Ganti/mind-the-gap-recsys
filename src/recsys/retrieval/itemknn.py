"""Item-item kNN over click co-occurrence, with shrinkage and asymmetric cosine.

    sim(i, j) = |U_i ∩ U_j| / (|U_i|^alpha * |U_j|^(1-alpha) + shrink)
    score(u, j) = sum_{i in clicks(u)} sim(i, j), keeping the top `neighbors` sources per j

alpha = 0.5 is cosine. Shrinkage damps similarities estimated from few co-occurrences.
Tuned seriously (Gate C): a large body of published neural work fails to beat a well-tuned
item-kNN (Dacrema et al., RecSys 2019). If that happens here, it is a result.

Cold items (no train clicks) get an all-zero column and can only be ranked by tie-breaking.
In news, most of the live catalog is such items, which is exactly why this method may lose here.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from recsys.retrieval._sparse import prune_columns, topn_similarity_multi
from recsys.retrieval.base import Retriever


def _normaliser(deg: np.ndarray, alpha: float, shrink: float):
    src = np.power(deg, alpha)[:, None]

    def fn(block: np.ndarray, J: np.ndarray) -> np.ndarray:
        denom = src * np.power(deg[J], 1.0 - alpha)[None, :] + shrink
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom > 0, block / denom, 0.0)
        return out.astype(np.float32)
    return fn


class ItemKNNRetriever(Retriever):
    name = "itemknn"

    def __init__(self, neighbors: int = 100, shrink: float = 10.0, alpha: float = 0.5):
        super().__init__(neighbors=neighbors, shrink=shrink, alpha=alpha)
        self.neighbors, self.shrink, self.alpha = neighbors, shrink, alpha

    def fit(self, data) -> "ItemKNNRetriever":
        return next(fit_grid(data, [self.neighbors], [self.shrink], [self.alpha]))

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray((self.X_[users] @ self.W_).todense(), dtype=np.float32)


def fit_grid(data, neighbors: list[int], shrink: list[float], alpha: list[float]) -> Iterator[ItemKNNRetriever]:
    """Fit every grid point with one co-occurrence pass. Yields retrievers lazily in grid order (one pruned W in memory at a time)."""
    X = data.X.tocsr()
    deg = np.asarray(X.sum(0)).ravel().astype(np.float32)
    norms = {(a, s): _normaliser(deg, a, s) for a, s in itertools.product(alpha, shrink)}
    Ws = topn_similarity_multi(X.T.tocsr(), X.tocsc(), np.asarray(data.catalog), max(neighbors), norms)
    for n, s, a in itertools.product(neighbors, shrink, alpha):
        r = ItemKNNRetriever(neighbors=n, shrink=s, alpha=a)
        r.W_ = prune_columns(Ws[(a, s)], n)
        r.X_ = X
        yield r
