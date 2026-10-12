from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from recsys.dataset import SplitData


def make_split(train_pairs, test_pairs, n_users, n_items, catalog=None, negatives=None, n_neg=None,
               ts=None, reference_ts=1000.0, exclude_train_seen=True) -> SplitData:
    """Hand-built SplitData. `negatives`: {(u, pos): [items]}."""
    train = pd.DataFrame(train_pairs, columns=["u", "i"]) if len(train_pairs) else pd.DataFrame({"u": [], "i": []})
    train = train.astype({"u": np.int64, "i": np.int64})
    train["ts"] = np.asarray(ts, float) if ts is not None else np.full(len(train), np.nan)
    train["hist_pos"] = np.where(np.isnan(train["ts"]), np.arange(len(train)), np.nan)
    test = pd.DataFrame(test_pairs, columns=["u", "i"]).astype(np.int64)
    test["ts"] = reference_ts + 1
    test["is_eval"] = True
    catalog = np.arange(n_items) if catalog is None else np.sort(np.asarray(catalog))
    negatives = negatives or {}
    pairs = sorted(negatives)
    width = n_neg or (max((len(v) for v in negatives.values()), default=0))
    neg_items = np.full((len(pairs), width), -1, dtype=np.int64)
    for r, key in enumerate(pairs):
        neg_items[r, :len(negatives[key])] = negatives[key]
    stats = pd.DataFrame({"impressions": np.zeros(n_items), "clicks": np.zeros(n_items),
                          "first_seen": np.full(n_items, np.nan), "last_seen": np.full(n_items, np.nan)})
    news = pd.DataFrame({"category": ["c"] * n_items, "subcategory": ["s"] * n_items,
                         "title": [f"item {j}" for j in range(n_items)], "abstract": [""] * n_items})
    return SplitData(name="tiny", reference_ts=reference_ts, n_users=n_users, n_items=n_items, train=train,
                     test=test, catalog=catalog, eval_users=np.sort(test["u"].unique()),
                     neg_pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2), neg_items=neg_items,
                     item_stats=stats, news=news, exclude_train_seen=exclude_train_seen)


def _java_ok() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        return subprocess.run(["java", "-version"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


requires_spark = pytest.mark.skipif(not _java_ok(), reason="no working JVM for pyspark")
