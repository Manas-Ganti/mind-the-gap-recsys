"""Tiny synthetic data in the exact MIND file format, for tests and smoke runs only.

Numbers produced from this data are pipeline checks, not results. The smoke config writes
them under smoke/, never results/.

Generative story: items belong to categories and are published on a given day; impressions
show recent items; a user clicks with probability rising with their affinity for the item's
category and the item's latent quality. The pre-window `history` field is fixed per user and
holds items published before the window, as in MIND.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CATS = ["news", "sports", "finance", "lifestyle", "health", "travel"]


def _fmt(t: datetime) -> str:
    h = t.hour % 12 or 12
    return f"{t.month}/{t.day}/{t.year} {h}:{t.minute:02d}:{t.second:02d} {'AM' if t.hour < 12 else 'PM'}"


def make_synthetic_mind(out_dir: Path, *, variant: str = "small", n_users: int = 300, n_items: int = 600,
                        impressions_per_day: int = 400, seed: int = 0,
                        start: datetime = datetime(2019, 11, 9), n_train_days: int = 6) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    vocab = {c: [f"{c}{j}" for j in range(30)] for c in CATS}
    common = [f"w{j}" for j in range(40)]

    cat = rng.integers(0, len(CATS), n_items)
    pub_day = rng.integers(-5, n_train_days + 1, n_items)     # day offset from `start`; <0 = pre-window
    quality = rng.normal(0, 1, n_items)
    ids = [f"N{j + 1}" for j in range(n_items)]
    users = [f"U{j + 1}" for j in range(n_users)]
    affinity = rng.dirichlet(np.ones(len(CATS)) * 0.5, n_users)

    def title(j):
        c = CATS[cat[j]]
        return " ".join(rng.choice(vocab[c], 4).tolist() + rng.choice(common, 3).tolist())

    news_lines = [f"{ids[j]}\t{CATS[cat[j]]}\t{CATS[cat[j]]}_sub{j % 3}\t{title(j)}\t{title(j)} {title(j)}\t"
                  f"https://example.invalid/{ids[j]}\t[]\t[]" for j in range(n_items)]

    old = np.flatnonzero(pub_day < 0)
    history = {}
    for u in range(n_users):
        p = affinity[u][cat[old]] * np.exp(quality[old])
        k = int(rng.integers(0, 8))
        history[u] = rng.choice(old, size=min(k, len(old)), replace=False, p=p / p.sum()).tolist() if k else []

    splits = {"train": range(n_train_days), "dev": range(n_train_days, n_train_days + 1)}
    for split, days in splits.items():
        d = out_dir / f"MIND{variant}_{split}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "news.tsv").write_text("\n".join(news_lines) + "\n")
        lines, imp_id = [], 0
        for day in days:
            live = np.flatnonzero((pub_day <= day) & (pub_day >= day - 2))
            for _ in range(impressions_per_day):
                u = int(rng.integers(0, n_users))
                t = start + timedelta(days=int(day), seconds=int(rng.integers(0, 86400)))
                shown = rng.choice(live, size=min(int(rng.integers(5, 15)), len(live)), replace=False)
                logit = -2.5 + 3.0 * affinity[u][cat[shown]] + 0.8 * quality[shown]
                clicked = rng.random(len(shown)) < 1 / (1 + np.exp(-logit))
                if not clicked.any():
                    clicked[int(np.argmax(logit))] = True    # MIND impressions have >= 1 click
                imp_id += 1
                hist = " ".join(ids[j] for j in history[u])
                toks = " ".join(f"{ids[j]}-{int(c)}" for j, c in zip(shown, clicked))
                lines.append(f"{imp_id}\t{users[u]}\t{_fmt(t)}\t{hist}\t{toks}")
        (d / "behaviors.tsv").write_text("\n".join(lines) + "\n")
    return out_dir
