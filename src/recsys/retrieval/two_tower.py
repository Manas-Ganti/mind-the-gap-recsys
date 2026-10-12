"""Two-tower retrieval with a text item tower.

Item tower: mean-pooled word embeddings of title and abstract + category and subcategory
            embeddings -> MLP -> L2-normalised vector. It is purely content-based, so it can
            embed items with no clicks. In news, most of the live catalog is such items.
User tower: additive-attention pooling over the item-tower vectors of the user's last
            `history_len` clicks (plus a learned "empty" slot so cold users are defined)
            -> linear -> L2-normalised. No user-ID embedding, so it is fully inductive.
Loss:       in-batch sampled softmax, temperature tau, with logQ correction (Yi et al.,
            RecSys 2019): logit -= log q(item), q = empirical train click frequency.
            Accidental hits (the same item as another row's positive) are masked out.

Training examples are (prefix of the user's chronological train sequence -> next click), so
the model never conditions on a click that came after the one it predicts.

The vocabulary is built from items in the training sequences only.
"""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.config import device_for
from recsys.retrieval.base import Retriever

TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD, OOV = 0, 1


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def build_vocab(texts, max_size: int, min_count: int) -> dict[str, int]:
    c = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, n in sorted(c.items(), key=lambda x: (-x[1], x[0])) if n >= min_count][: max_size - 2]
    return {w: j + 2 for j, w in enumerate(words)}


def encode_text(texts, vocab: dict[str, int], length: int) -> np.ndarray:
    out = np.zeros((len(texts), length), dtype=np.int64)
    for r, t in enumerate(texts):
        ids = [vocab.get(w, OOV) for w in tokenize(t)][:length]
        out[r, :len(ids)] = ids
    return out


class ItemTower(nn.Module):
    def __init__(self, vocab_size: int, n_cat: int, n_sub: int, dim: int):
        super().__init__()
        self.word = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.cat = nn.Embedding(n_cat, dim)
        self.sub = nn.Embedding(n_sub, dim)
        self.mlp = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    @staticmethod
    def _mean(emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD).unsqueeze(-1).float()
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, title, abstract, cat, sub):
        x = torch.cat([self._mean(self.word(title), title), self._mean(self.word(abstract), abstract),
                       self.cat(cat), self.sub(sub)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.empty = nn.Parameter(torch.zeros(dim))
        self.att = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))
        self.out = nn.Linear(dim, dim)

    def forward(self, hist_vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = hist_vecs.shape[0]
        h = torch.cat([self.empty.expand(B, 1, -1), hist_vecs], dim=1)
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        a = self.att(h).squeeze(-1).masked_fill(~m, float("-inf"))
        pooled = (torch.softmax(a, dim=1).unsqueeze(-1) * h).sum(1)
        return F.normalize(self.out(pooled), dim=-1)


class TwoTowerNet(nn.Module):
    def __init__(self, vocab_size: int, n_cat: int, n_sub: int, dim: int):
        super().__init__()
        self.items = ItemTower(vocab_size, n_cat, n_sub, dim)
        self.users = UserTower(dim)


class TwoTowerRetriever(Retriever):
    name = "two_tower"

    def __init__(self, temperature: float = 0.05, *, dim=64, vocab_size=50000, min_count=3,
                 title_len=24, abstract_len=48, history_len=30, lr=1e-3, weight_decay=1e-5,
                 batch_size=1024, epochs=3, max_steps_per_epoch=3000, logq_correction=True,
                 device="auto", seed=42):
        super().__init__(temperature=temperature, dim=dim, epochs=epochs, batch_size=batch_size, lr=lr)
        self.temperature = temperature
        self.dim, self.vocab_size, self.min_count = dim, vocab_size, min_count
        self.title_len, self.abstract_len, self.history_len = title_len, abstract_len, history_len
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.epochs, self.max_steps_per_epoch = epochs, max_steps_per_epoch
        self.logq_correction, self.device_name, self.seed = logq_correction, device, seed

    # ------------------------------------------------------------------ features
    def _item_features(self, data):
        news = data.news
        train_items = np.unique(data.train["i"].to_numpy())
        vocab = build_vocab((news["title"].iloc[train_items] + " " + news["abstract"].iloc[train_items]).tolist(),
                            self.vocab_size, self.min_count)
        cats = {c: j for j, c in enumerate(sorted(news["category"].unique()))}
        subs = {c: j for j, c in enumerate(sorted(news["subcategory"].unique()))}
        self.vocab_, self.cats_, self.subs_ = vocab, cats, subs
        self.title_ = torch.from_numpy(encode_text(news["title"].tolist(), vocab, self.title_len))
        self.abstract_ = torch.from_numpy(encode_text(news["abstract"].tolist(), vocab, self.abstract_len))
        self.cat_ = torch.from_numpy(news["category"].map(cats).to_numpy(np.int64).copy())
        self.sub_ = torch.from_numpy(news["subcategory"].map(subs).to_numpy(np.int64).copy())

    def _encode_items(self, idx: torch.Tensor, dev) -> torch.Tensor:
        idx = idx.cpu()
        return self.net_.items(self.title_[idx].to(dev), self.abstract_[idx].to(dev),
                               self.cat_[idx].to(dev), self.sub_[idx].to(dev))

    def _encode_users(self, hist: np.ndarray, dev) -> torch.Tensor:
        """hist: (B, H) item indices, -1 padded."""
        h = torch.from_numpy(hist)
        mask = h >= 0
        uniq, inv = torch.unique(h.clamp(min=0), return_inverse=True)
        vecs = self._encode_items(uniq, dev)[inv.to(dev)]
        return self.net_.users(vecs, mask.to(dev))

    # ------------------------------------------------------------------ training
    def fit(self, data) -> "TwoTowerRetriever":
        self.started_at = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(device_for(self.device_name))
        self._item_features(data)
        self.net_ = TwoTowerNet(len(self.vocab_) + 2, len(self.cats_), len(self.subs_), self.dim).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        seq_items, seq_ptr = data._sequences
        lens = np.diff(seq_ptr)
        # every (user, p) with p >= 1: predict seq[p] from seq[:p]
        ex_user = np.repeat(np.arange(data.n_users), np.maximum(lens - 1, 0))
        ex_off = np.concatenate([np.arange(1, n) for n in lens if n > 1]) if (lens > 1).any() else np.zeros(0, int)
        if len(ex_user) == 0:
            raise ValueError("no training sequences of length >= 2")
        freq = np.bincount(seq_items, minlength=data.n_items).astype(np.float64)
        logq = torch.from_numpy(np.log(np.maximum(freq, 1.0) / freq.sum())).float().to(dev)

        H, B = self.history_len, self.batch_size
        ar = np.arange(H)
        self.loss_history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(len(ex_user))
            steps = min(len(order) // B, self.max_steps_per_epoch) or 1
            self.net_.train()
            total = 0.0
            for s in range(steps):
                sel = order[s * B:(s + 1) * B]
                u, p = ex_user[sel], ex_off[sel]
                base = seq_ptr[u]
                tgt = seq_items[base + p]
                start = np.maximum(p - H, 0)
                pos = base[:, None] + start[:, None] + ar[None, :]
                valid = (start[:, None] + ar[None, :]) < p[:, None]
                hist = np.where(valid, seq_items[np.minimum(pos, len(seq_items) - 1)], -1)

                uvec = self._encode_users(hist, dev)
                t = torch.from_numpy(tgt)
                ivec = self._encode_items(t, dev)
                logits = uvec @ ivec.T / self.temperature
                t_dev = t.to(dev)
                if self.logq_correction:
                    logits = logits - logq[t_dev][None, :]
                same = t_dev[:, None] == t_dev[None, :]
                same.fill_diagonal_(False)
                logits = logits.masked_fill(same, float("-inf"))
                loss = F.cross_entropy(logits, torch.arange(len(sel), device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            self.loss_history_.append(total / steps)

        self.net_.eval()
        self._prepare_scoring(data)
        self.completed_at = time.time()
        return self

    @torch.no_grad()
    def _prepare_scoring(self, data, chunk: int = 8192) -> None:
        dev = next(self.net_.parameters()).device
        cat = torch.from_numpy(np.asarray(data.catalog))
        vecs = [self._encode_items(cat[c:c + chunk], dev).cpu() for c in range(0, len(cat), chunk)]
        self.item_vecs_ = torch.cat(vecs).numpy().astype(np.float32)
        self.data_ = data
        self.catalog_ = np.asarray(data.catalog).copy()
        self.net_.cpu()

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray) -> np.ndarray:
        hist = self.data_.sequences(users, self.history_len)
        return self._encode_users(hist, torch.device("cpu")).numpy().astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_vectors(users) @ self.item_vecs_.T

    def __getstate__(self):
        s = self.__dict__.copy()
        s.pop("data_", None)        # re-attached by `attach` after loading
        return s

    def attach(self, data) -> "TwoTowerRetriever":
        if not np.array_equal(np.asarray(data.catalog), self.catalog_):
            raise ValueError("two_tower was fitted for a different catalog")
        self.data_ = data
        return self
