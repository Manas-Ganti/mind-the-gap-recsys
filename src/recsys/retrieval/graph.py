"""RP3beta: a 3-step random walk on the user-item bipartite graph (Paudel et al., 2017).

    P_ui = row-normalised X (user -> item),  P_iu = row-normalised X^T (item -> user)
    W    = (P_iu^alpha @ P_ui^alpha), column j divided by deg(j)^beta, top-n per column
    score(u, j) = X_u @ W[:, j]

This is the "item co-occurrence graph with propagation" option from the spec, chosen over
LightGCN because it is a strong, cheap graph baseline (Dacrema et al. found it competitive
with most neural graph models) and it keeps the four-week budget on the evaluation.
beta > 0 penalises popular targets, which is the knob that trades accuracy for coverage.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

from recsys.retrieval._sparse import prune_columns, topn_similarity_multi
from recsys.retrieval.base import Retriever


def _row_normalise_pow(M: sp.csr_matrix, alpha: float) -> sp.csr_matrix:
    M = M.tocsr().astype(np.float32, copy=True)
    deg = np.asarray(M.sum(1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    M = (sp.diags(inv) @ M).tocsr()
    M.data = np.power(M.data, alpha)
    return M


class RP3betaRetriever(Retriever):
    name = "graph"

    def __init__(self, neighbors: int = 200, alpha: float = 1.0, beta: float = 0.5):
        super().__init__(neighbors=neighbors, alpha=alpha, beta=beta)
        self.neighbors, self.alpha, self.beta = neighbors, alpha, beta

    def fit(self, data) -> "RP3betaRetriever":
        return next(fit_grid(data, [self.neighbors], [self.alpha], [self.beta]))

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray((self.X_[users] @ self.W_).todense(), dtype=np.float32)


def fit_grid(data, neighbors: list[int], alpha: list[float], beta: list[float]) -> Iterator[RP3betaRetriever]:
    """One random-walk product per alpha; beta and neighbors reuse it. Grid order is
    product(neighbors, alpha, beta)."""
    X = data.X.tocsr()
    deg = np.asarray(X.sum(0)).ravel().astype(np.float32)
    targets = np.asarray(data.catalog)
    Ws = {}
    for a in alpha:
        p_ui = _row_normalise_pow(X, a).tocsc()
        p_iu = _row_normalise_pow(X.T.tocsr(), a)
        norms = {}
        for b in beta:
            def fn(block, J, b=b):
                return (block / np.power(np.maximum(deg[J], 1.0), b)[None, :]).astype(np.float32)
            norms[b] = fn
        for b, W in topn_similarity_multi(p_iu, p_ui, targets, max(neighbors), norms).items():
            Ws[(a, b)] = W
    for n, a, b in itertools.product(neighbors, alpha, beta):
        r = RP3betaRetriever(neighbors=n, alpha=a, beta=b)
        r.W_ = prune_columns(Ws[(a, b)], n)
        r.X_ = X
        yield r
