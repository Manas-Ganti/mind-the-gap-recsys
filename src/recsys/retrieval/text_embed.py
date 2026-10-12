"""Frozen sentence embeddings of every item's text (CLAUDE.md decision 1).

`title. abstract` is encoded once with a pretrained sentence encoder (default
all-MiniLM-L6-v2, 384-d, L2-normalised) and cached at
{processed_dir}/text_emb/<model-slug>.npy, aligned to the global item index. The encoder is
never fine-tuned; only the two-tower projection and user tower are trained.

Leakage note: encoding test-period items is legitimate. An article's text exists the moment
it is published, before any user is shown it, and the encoder sees no clicks. The pretrained
encoder's own corpus is generic web text, not MIND interactions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from recsys.config import Config, device_for


def _slug(name: str) -> str:
    return name.replace("/", "__")


def item_texts(news) -> list[str]:
    title = news["title"].fillna("").astype(str).str.strip()
    abstract = news["abstract"].fillna("").astype(str).str.strip()
    return [f"{t}. {a}".strip(" .") if a else t for t, a in zip(title, abstract)]


def embeddings_path(cfg: Config) -> Path:
    return Path(cfg.paths.processed_dir) / "text_emb" / f"{_slug(cfg.retrieval.two_tower.text_encoder)}.npy"


def ensure_item_embeddings(cfg: Config) -> Path:
    """Compute (or reuse) the cached embedding matrix. Returns its path."""
    from recsys.dataset import load_index_sizes, load_news

    tt = cfg.retrieval.two_tower
    path = embeddings_path(cfg)
    meta_path = path.with_suffix(".json")
    _, n_items = load_index_sizes(cfg)
    news = load_news(cfg, n_items)
    texts = item_texts(news)
    digest = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model") == tt.text_encoder and meta.get("n_items") == n_items and meta.get("text_sha") == digest:
            return path

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    device = device_for(tt.device)
    model = SentenceTransformer(tt.text_encoder, device=device)
    emb = model.encode(texts, batch_size=tt.encode_batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    meta_path.write_text(json.dumps({"model": tt.text_encoder, "n_items": n_items, "dim": int(emb.shape[1]),
                                     "text_sha": digest, "device": device,
                                     "seconds": round(time.time() - t0, 1)}, indent=2))
    return path
