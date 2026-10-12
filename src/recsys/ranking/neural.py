"""A simple deep ranker for comparison: MLP over the same standardised candidate features,
trained with a listwise softmax loss per user (-mean log p(positive) over each user's
candidate list), early stopped on held-out users' NDCG@10 of the reranked candidates."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from recsys.config import device_for
from recsys.eval.metrics import ndcg_at_k
from recsys.ranking.features import CandidateBlock


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

    def __init__(self, hidden: list[int], dropout: float, lr: float, epochs: int, batch_users: int,
                 patience: int, device: str, seed: int):
        self.hidden, self.dropout, self.lr, self.epochs = hidden, dropout, lr, epochs
        self.batch_users, self.patience, self.device_name, self.seed = batch_users, patience, device, seed

    def _std(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_).astype(np.float32)

    def fit(self, train: CandidateBlock, valid: CandidateBlock, feature_names: list[str]) -> "NeuralRanker":
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
            score = self._valid_ndcg(valid)
            self.valid_curve_.append(score)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.cpu().eval()
        self.best_valid_ndcg_ = best
        return self

    def _valid_ndcg(self, valid: CandidateBlock, k: int = 10) -> float:
        s = self.predict(valid)
        vals = []
        for g in range(len(valid.users)):
            a, b = valid.ptr[g], valid.ptr[g + 1]
            y = valid.y[a:b]
            if not y.any():
                continue
            ranked = np.argsort(-s[a:b], kind="stable")
            vals.append(ndcg_at_k(list(ranked), set(np.flatnonzero(y)), k))
        return float(np.mean(vals)) if vals else 0.0

    @torch.no_grad()
    def predict(self, block: CandidateBlock, chunk: int = 200_000) -> np.ndarray:
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
