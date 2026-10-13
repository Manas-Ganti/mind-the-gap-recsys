"""A simple deep ranker for comparison: MLP over the same standardised candidate features,
trained with a listwise softmax loss per user (-mean log p(positive) over each user's
candidate list), early stopped on a later day's NDCG@10 of the reranked candidates. With no
validation block it trains for exactly `epochs` (the final refit, epochs fixed by selection)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from recsys.config import deterministic_torch, device_for
from recsys.ranking.features import CandidateBlock, group_ndcg


class _MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _padded(block: CandidateBlock, groups: np.ndarray, X: np.ndarray):
    sizes = np.diff(block.ptr)[groups]
    L = int(sizes.max())
    xb = np.zeros((len(groups), L, X.shape[1]), np.float32)
    yb = np.zeros((len(groups), L), np.float32)
    mb = np.zeros((len(groups), L), bool)
    for r, g in enumerate(groups):
        s, e = block.ptr[g], block.ptr[g + 1]
        xb[r, :e - s], yb[r, :e - s], mb[r, :e - s] = X[s:e], block.y[s:e], True
    return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(mb)


class NeuralRanker:
    name = "neural"
    columns: list[int] | None = None      # feature subset; class default keeps old pickles loadable

    def __init__(self, hidden: list[int], dropout: float, lr: float, epochs: int, batch_users: int,
                 patience: int, device: str, seed: int, threads: int = 16, columns: list[int] | None = None):
        self.hidden, self.dropout, self.lr, self.epochs = hidden, dropout, lr, epochs
        self.batch_users, self.patience, self.device_name, self.seed = batch_users, patience, device, seed
        self.threads, self.columns = threads, columns

    def _std(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def fit(self, train: CandidateBlock, valid: CandidateBlock | None, feature_names: list[str]) -> "NeuralRanker":
        with deterministic_torch(self.threads):
            return self._fit(train.with_columns(self.columns),
                             None if valid is None else valid.with_columns(self.columns), feature_names)

    def _fit(self, train: CandidateBlock, valid: CandidateBlock | None, feature_names: list[str]) -> "NeuralRanker":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(device_for(self.device_name))
        self.mu_ = train.X.mean(0)
        self.sd_ = train.X.std(0) + 1e-6
        Xtr = self._std(train.X)
        has_pos = np.array([train.y[train.ptr[g]:train.ptr[g + 1]].any() for g in range(len(train.users))])
        groups = np.flatnonzero(has_pos)
        self.net_ = _MLP(train.X.shape[1], self.hidden, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, bad = -1.0, None, 0
        self.valid_curve_ = []
        for _ in range(self.epochs):
            self.net_.train()
            groups = rng.permutation(groups)
            for g0 in range(0, len(groups), self.batch_users):
                gb = groups[g0:g0 + self.batch_users]
                x, y, m = (t.to(dev) for t in _padded(train, gb, Xtr))
                logits = self.net_(x).masked_fill(~m, float("-inf"))
                logp = torch.log_softmax(logits, dim=1)
                loss = -(logp.masked_fill(y == 0, 0.0).sum(1) / y.sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if valid is None:
                continue
            score = group_ndcg(valid, self._predict(valid))
            self.valid_curve_.append(score)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.cpu().eval()
        self.best_valid_ndcg_ = best if valid is not None else float("nan")
        self.best_epochs_ = (int(np.argmax(self.valid_curve_)) + 1) if self.valid_curve_ else self.epochs
        return self

    def predict(self, block: CandidateBlock) -> np.ndarray:
        return self._predict(block.with_columns(self.columns))

    @torch.no_grad()
    def _predict(self, block: CandidateBlock, chunk: int = 200_000) -> np.ndarray:
        self.net_.eval()
        dev = next(self.net_.parameters()).device
        X = self._std(block.X)
        out = [self.net_(torch.from_numpy(X[c:c + chunk]).to(dev)).cpu().numpy() for c in range(0, len(X), chunk)]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.net_.cpu()
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "NeuralRanker":
        return pickle.loads(path.read_bytes())
