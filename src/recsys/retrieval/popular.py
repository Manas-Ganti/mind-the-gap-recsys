"""Most-popular baseline with a tuned trailing time window.

Not a strawman: in news, freshness dominates, and "most clicked in the last few hours" is a
hard baseline. The window is tuned on temporal_val. Popularity is frozen at the split's
reference time; it is not updated with test-period clicks (see README limitations). On the
random split there is no reference time, so the window is ignored (all-time counts).
"""
from __future__ import annotations

import numpy as np

from recsys.retrieval.base import Retriever


class PopularRetriever(Retriever):
    name = "popular"

    def __init__(self, window_hours: float | None = None):
        super().__init__(window_hours=window_hours)
        self.window_hours = window_hours

    def fit(self, data) -> "PopularRetriever":
        t = data.train
        if self.window_hours is None or not np.isfinite(data.reference_ts):   # random split: no "now"
            mask = np.ones(len(t), dtype=bool)
        else:
            ts = t["ts"].to_numpy()
            mask = ts >= data.reference_ts - self.window_hours * 3600.0   # NaN (history) -> False
        counts = np.bincount(t["i"].to_numpy()[mask], minlength=data.n_items).astype(np.float32)
        self.scores_ = counts[data.catalog]
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.scores_, (len(users), len(self.scores_))).copy()
