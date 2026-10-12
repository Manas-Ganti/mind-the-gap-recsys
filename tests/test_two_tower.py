"""Two-tower over frozen text embeddings, on a tiny split with a fake embedding file
(no model download)."""
import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import make_split
from recsys.retrieval.text_embed import item_texts
from recsys.retrieval.two_tower import TwoTowerRetriever


def _split_and_embeddings(tmp_path, seed=0, n_users=40, n_items=30):
    rng = np.random.default_rng(seed)
    pairs = sorted({(int(u), int(i)) for u in range(n_users) for i in rng.choice(n_items, 6, replace=False)})
    test = [(u, int(i)) for u in range(5) for i in range(n_items) if (u, i) not in set(pairs)][:5]
    d = make_split(pairs, test, n_users, n_items, ts=np.arange(len(pairs), dtype=float))
    emb = rng.normal(size=(n_items, 12)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    path = tmp_path / "emb.npy"
    np.save(path, emb)
    return d, path


def _model(path, **kw):
    return TwoTowerRetriever(temperature=0.1, embeddings_path=path, dim=8, history_len=4, batch_size=16,
                             epochs=2, max_steps_per_epoch=5, device="cpu", seed=0, **kw)


def test_fit_and_score_shape(tmp_path):
    d, path = _split_and_embeddings(tmp_path)
    r = _model(path).fit(d)
    S = r.score(np.array([0, 1, 2]))
    assert S.shape == (3, len(d.catalog)) and np.isfinite(S).all()
    assert len(r.loss_history_) == 2


def test_deterministic_given_seed(tmp_path):
    d, path = _split_and_embeddings(tmp_path)
    a = _model(path).fit(d).score(np.array([0, 1]))
    b = _model(path).fit(d).score(np.array([0, 1]))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_pickle_drops_embeddings_and_reloads(tmp_path):
    d, path = _split_and_embeddings(tmp_path)
    r = _model(path).fit(d)
    before = r.score(np.array([3, 4]))
    blob = pickle.dumps(r)
    assert b"_text" not in blob or len(blob) < 200_000          # matrix itself is not serialised
    r2 = pickle.loads(blob).attach(d)
    np.testing.assert_allclose(r2.score(np.array([3, 4])), before, rtol=1e-5, atol=1e-6)


def test_embedding_matrix_must_cover_item_index(tmp_path):
    d, _ = _split_and_embeddings(tmp_path)
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((d.n_items - 1, 12), np.float32))
    with pytest.raises(ValueError):
        _model(bad).fit(d)


def test_item_texts_format():
    news = pd.DataFrame({"title": ["A title", "B", ""], "abstract": ["Some abstract.", "", ""]})
    assert item_texts(news) == ["A title. Some abstract", "B", ""]
