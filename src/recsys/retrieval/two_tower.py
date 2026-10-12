"""Two-tower retrieval with a frozen-text item tower (CLAUDE.md decision 1).

Item tower: frozen all-MiniLM-L6-v2 embedding of `title. abstract` (precomputed once, see
            text_embed.py) + learned category and subcategory embeddings -> MLP ->
            L2-normalised vector. Purely content-based, so it embeds items with no clicks;
            in news, most of the live catalog is such items. The language model is never
            fine-tuned: only the projection MLP and category embeddings learn.
User tower: additive-attention pooling over the item-tower vectors of the user's last
            `history_len` clicks (plus a learned "empty" slot so cold users are defined)
            -> linear -> L2-normalised. No user-ID embedding, so it is fully inductive.
Loss:       in-batch sampled softmax, temperature tau, with logQ correction (Yi et al.,
            RecSys 2019): logit -= log q(item), q = empirical train click frequency.
            Accidental hits (the same item as another row's positive) are masked out.

Training examples are (prefix of the user's chronological train sequence -> next click), so
the model never conditions on a click that came after the one it predicts.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recsys.config import deterministic_torch, device_for
from recsys.retrieval.base import Retriever


class ItemTower(nn.Module):
    def __init__(self, text_dim: int, n_cat: int, n_sub: int, dim: int):
        super().__init__()
        self.cat = nn.Embedding(n_cat, dim)
        self.sub = nn.Embedding(n_sub, dim)
        self.mlp = nn.Sequential(nn.Linear(text_dim + 2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, text, cat, sub):
        x = torch.cat([text, self.cat(cat), self.sub(sub)], dim=-1)
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
    def __init__(self, text_dim: int, n_cat: int, n_sub: int, dim: int):
        super().__init__()
        self.items = ItemTower(text_dim, n_cat, n_sub, dim)
        self.users = UserTower(dim)


class TwoTowerRetriever(Retriever):
    name = "two_tower"

    def __init__(self, temperature: float = 0.05, *, embeddings_path: str | Path, dim=64,
                 history_len=30, lr=1e-3, weight_decay=1e-5, batch_size=1024, epochs=3,
                 max_steps_per_epoch=3000, logq_correction=True, device="auto", seed=42):
        super().__init__(temperature=temperature, dim=dim, epochs=epochs, batch_size=batch_size, lr=lr)
        self.temperature = temperature
        self.embeddings_path = str(embeddings_path)
        self.dim, self.history_len = dim, history_len
        self.lr, self.weight_decay, self.batch_size = lr, weight_decay, batch_size
        self.epochs, self.max_steps_per_epoch = epochs, max_steps_per_epoch
        self.logq_correction, self.device_name, self.seed = logq_correction, device, seed

    # ------------------------------------------------------------------ features
    @property
    def text_(self) -> torch.Tensor:
        """Frozen item text embeddings (n_items, text_dim). Loaded lazily and never pickled."""
        t = self.__dict__.get("_text")
        if t is None:
            t = torch.from_numpy(np.load(self.embeddings_path).astype(np.float32))
            self.__dict__["_text"] = t
        return t

    def _item_features(self, data):
        news = data.news
        if self.text_.shape[0] != data.n_items:
            raise ValueError(f"text embeddings cover {self.text_.shape[0]} items, split has {data.n_items}")
        cats = {c: j for j, c in enumerate(sorted(news["category"].unique()))}
        subs = {c: j for j, c in enumerate(sorted(news["subcategory"].unique()))}
        self.cats_, self.subs_ = cats, subs
        self.cat_ = torch.from_numpy(news["category"].map(cats).to_numpy(np.int64).copy())
        self.sub_ = torch.from_numpy(news["subcategory"].map(subs).to_numpy(np.int64).copy())

    def _encode_items(self, idx: torch.Tensor, dev) -> torch.Tensor:
        idx = idx.cpu()
        return self.net_.items(self.text_[idx].to(dev), self.cat_[idx].to(dev), self.sub_[idx].to(dev))

    def _encode_users(self, hist: np.ndarray, dev) -> torch.Tensor:
        """hist: (B, H) item indices, -1 padded."""
        h = torch.from_numpy(hist)
        mask = h >= 0
        uniq, inv = torch.unique(h.clamp(min=0), return_inverse=True)
        vecs = self._encode_items(uniq, dev)[inv.to(dev)]
        return self.net_.users(vecs, mask.to(dev))

    # ------------------------------------------------------------------ training
    def fit(self, data) -> "TwoTowerRetriever":
        with deterministic_torch():
            return self._fit(data)

    def _fit(self, data) -> "TwoTowerRetriever":
        self.started_at = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(device_for(self.device_name))
        self._item_features(data)
        self.net_ = TwoTowerNet(self.text_.shape[1], len(self.cats_), len(self.subs_), self.dim).to(dev)
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
        s.pop("_text", None)        # reloaded from embeddings_path on first use
        return s

    def attach(self, data) -> "TwoTowerRetriever":
        if not np.array_equal(np.asarray(data.catalog), self.catalog_):
            raise ValueError("two_tower was fitted for a different catalog")
        self.data_ = data
        return self
