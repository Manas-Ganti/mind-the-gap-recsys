"""In-memory view of one split, loaded from the Spark-written parquet. Models and the
evaluation harness only ever see a `SplitData`, so nothing downstream can reach the raw logs
and accidentally read past the cutoff."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import Config
from recsys.splits import split_dir


@dataclass
class SplitData:
    name: str
    reference_ts: float            # "now" at serving time: the cutoff (NaN for the random split)
    n_users: int
    n_items: int
    train: pd.DataFrame            # u i ts hist_pos   (ts NaN = pre-window history)
    test: pd.DataFrame             # u i ts is_eval
    catalog: np.ndarray            # sorted item indices rankable at serving time
    eval_users: np.ndarray         # sorted
    neg_pairs: np.ndarray          # (P, 2) [u, pos]
    neg_items: np.ndarray          # (P, N) item indices, -1 padded
    item_stats: pd.DataFrame       # index i: impressions clicks first_seen last_seen
    news: pd.DataFrame             # index i: category subcategory title abstract
    exclude_train_seen: bool = True
    meta: dict = field(default_factory=dict)
    catalog_days: pd.DataFrame | None = None   # day i; None = every catalog item always available

    @cached_property
    def _availability(self):
        if self.catalog_days is None:
            return None
        days = np.sort(self.catalog_days["day"].unique())
        d_idx = {d: j for j, d in enumerate(days)}
        day_items = np.zeros((len(days), len(self.catalog)), dtype=np.float32)
        day_items[self.catalog_days["day"].map(d_idx).to_numpy(), self.catalog_col[self.catalog_days["i"].to_numpy()]] = 1
        t = self.test[self.test["is_eval"]]
        user_day = sp.csr_matrix((np.ones(len(t), np.float32), (t["u"].to_numpy(), t["day"].map(d_idx).to_numpy())),
                                 shape=(self.n_users, len(days)))
        return day_items, user_day

    def available(self, users: np.ndarray) -> np.ndarray | None:
        """(len(users), len(catalog)) bool: items live on the user's evaluation day.
        None means no per-user restriction."""
        av = self._availability
        if av is None:
            return None
        day_items, user_day = av
        return (user_day[users].toarray() > 0).astype(np.float32) @ day_items > 0

    @cached_property
    def X(self) -> sp.csr_matrix:
        """Binary user x item train interactions (history + in-window clicks)."""
        t = self.train
        return sp.csr_matrix((np.ones(len(t), dtype=np.float32), (t["u"].to_numpy(), t["i"].to_numpy())),
                             shape=(self.n_users, self.n_items))

    @cached_property
    def catalog_col(self) -> np.ndarray:
        """item index -> column in the catalog score matrix, -1 if not rankable."""
        col = np.full(self.n_items, -1, dtype=np.int64)
        col[self.catalog] = np.arange(len(self.catalog))
        return col

    @cached_property
    def test_pos(self) -> sp.csr_matrix:
        """Binary user x item test positives, eval users only."""
        t = self.test[self.test["is_eval"]]
        return sp.csr_matrix((np.ones(len(t), dtype=np.float32), (t["u"].to_numpy(), t["i"].to_numpy())),
                             shape=(self.n_users, self.n_items))

    @cached_property
    def _sequences(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-user chronological click sequence: history (by slot) then window clicks (by ts)."""
        t = self.train.assign(_is_win=self.train["ts"].notna(),
                              _key=np.where(self.train["ts"].notna(), self.train["ts"], self.train["hist_pos"]))
        t = t.sort_values(["u", "_is_win", "_key", "i"], kind="mergesort")
        items = t["i"].to_numpy(np.int64)
        ptr = np.zeros(self.n_users + 1, dtype=np.int64)
        np.add.at(ptr, t["u"].to_numpy() + 1, 1)
        return items, np.cumsum(ptr)

    def sequences(self, users: np.ndarray, length: int) -> np.ndarray:
        """(len(users), length) most-recent-last item indices, -1 left-padded."""
        items, ptr = self._sequences
        out = np.full((len(users), length), -1, dtype=np.int64)
        for r, u in enumerate(users):
            s = items[ptr[u]:ptr[u + 1]][-length:]
            if len(s):
                out[r, -len(s):] = s
        return out


def load_index_sizes(cfg: Config) -> tuple[int, int]:
    ing = Path(cfg.paths.processed_dir) / "ingest"
    n_users = len(pd.read_parquet(ing / "users", columns=["u"]))
    n_items = len(pd.read_parquet(ing / "items", columns=["i"]))
    return n_users, n_items


def load_news(cfg: Config, n_items: int) -> pd.DataFrame:
    news = pd.read_parquet(Path(cfg.paths.processed_dir) / "ingest" / "news",
                           columns=["i", "category", "subcategory", "title", "abstract"])
    news = news.set_index("i").reindex(np.arange(n_items))
    return news.fillna({"category": "<unk>", "subcategory": "<unk>", "title": "", "abstract": ""})


def load_split(cfg: Config, name: str) -> SplitData:
    import json

    d = split_dir(cfg, name)
    meta = json.loads((d / "meta.json").read_text())
    n_users, n_items = load_index_sizes(cfg)
    train = pd.read_parquet(d / "train")
    train["ts"] = train["ts"].astype("float64")
    test = pd.read_parquet(d / "test")
    catalog = np.sort(pd.read_parquet(d / "catalog")["i"].to_numpy(np.int64))
    eval_users = np.sort(pd.read_parquet(d / "eval_users")["u"].to_numpy(np.int64))
    negs = pd.read_parquet(d / "negatives").sort_values(["u", "pos"]).reset_index(drop=True)
    n_neg = cfg.eval.n_negatives
    neg_items = np.full((len(negs), n_neg), -1, dtype=np.int64)
    for r, lst in enumerate(negs["negs"]):
        if lst is not None and len(lst):
            neg_items[r, :len(lst)] = lst
    catalog_days = pd.read_parquet(d / "catalog_days")
    item_stats = (pd.read_parquet(d / "item_stats").set_index("i").reindex(np.arange(n_items))
                  .fillna({"impressions": 0, "clicks": 0}))
    return SplitData(
        name=name, reference_ts=float("nan") if meta["reference_ts"] is None else float(meta["reference_ts"]), n_users=n_users, n_items=n_items,
        train=train, test=test, catalog=catalog, eval_users=eval_users,
        neg_pairs=negs[["u", "pos"]].to_numpy(np.int64), neg_items=neg_items,
        item_stats=item_stats, news=load_news(cfg, n_items),
        exclude_train_seen=cfg.split.exclude_train_seen, meta=meta, catalog_days=catalog_days)


def subset_eval_users(data: SplitData, users: np.ndarray) -> SplitData:
    """Same split, evaluated on a subset of its eval users (used for fast tuning)."""
    import copy

    users = np.sort(np.asarray(users))
    d = copy.copy(data)
    d.__dict__.pop("test_pos", None)
    d.__dict__.pop("_availability", None)
    d.eval_users = users
    d.test = data.test.assign(is_eval=data.test["is_eval"] & data.test["u"].isin(users))
    keep = np.isin(data.neg_pairs[:, 0], users)
    d.neg_pairs, d.neg_items = data.neg_pairs[keep], data.neg_items[keep]
    return d
