"""Stage-2 selection plumbing: declared feature sets, column subsets carried by the rankers,
the within-candidates NDCG used to select, and the fixed-rounds refit (no validation block)."""
import numpy as np
import pytest

from recsys.config import FeatureSetCfg, SplitCfg
from recsys.eval.metrics import ndcg_at_k
from recsys.ranking.features import CandidateBlock, group_ndcg
from recsys.ranking.gbdt import GBDTRanker
from recsys.ranking.neural import NeuralRanker

NAMES = ["score_popular", "score_itemknn", "in_top_popular", "item_age_h", "item_ctr",
         "item_recent_1h", "item_recent_6h", "user_log_clicks", "user_log_window_clicks"]


def _block(n_groups=60, size=12, seed=0):
    """Feature 0 carries the label signal, so any sane ranker beats chance on it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * size, len(NAMES))).astype(np.float32)
    y = np.zeros(n_groups * size, np.float32)
    for g in range(n_groups):
        j = g * size + int(np.argmax(X[g * size:(g + 1) * size, 0]))
        y[j] = 1.0
    ptr = np.arange(0, n_groups * size + 1, size)
    return CandidateBlock(np.arange(n_groups), ptr, np.tile(np.arange(size), n_groups), X, y)


def test_feature_set_prefix_selection():
    assert FeatureSetCfg().select(NAMES) == list(range(len(NAMES)))
    ti = FeatureSetCfg(exclude=["item_age_h", "item_recent_", "user_log_window_clicks"]).select(NAMES)
    assert [NAMES[i] for i in ti] == ["score_popular", "score_itemknn", "in_top_popular", "item_ctr", "user_log_clicks"]
    ro = FeatureSetCfg(include=["score_", "in_top_"]).select(NAMES)
    assert [NAMES[i] for i in ro] == ["score_popular", "score_itemknn", "in_top_popular"]
    with pytest.raises(ValueError):
        FeatureSetCfg(include=["nope_"]).select(NAMES)


def test_ranker_train_cutoff_must_precede_validation_cutoff():
    with pytest.raises(ValueError):
        SplitCfg(temporal_cutoff="2019-11-15", validation_cutoff="2019-11-14", ranker_train_cutoff="2019-11-14")
    SplitCfg(temporal_cutoff="2019-11-15", validation_cutoff="2019-11-14", ranker_train_cutoff="2019-11-13")


def test_with_columns_and_has_positive():
    b = _block(4, 3)
    b.y[:] = 0
    b.y[4] = 1                                      # only group 1 has a positive
    assert b.has_positive().tolist() == [False, True, False, False]
    s = b.with_columns([0, 3])
    assert s.X.shape == (12, 2) and np.array_equal(s.X[:, 1], b.X[:, 3]) and s.y is b.y
    assert b.with_columns(None) is b


def test_group_ndcg_matches_hand_computation_and_skips_groups_without_positive():
    ptr = np.array([0, 3, 6, 8])
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0], np.float32)          # group 2 has no positive
    blk = CandidateBlock(np.arange(3), ptr, np.arange(8), np.zeros((8, 1), np.float32), y)
    scores = np.array([3, 2, 1, 1, 9, 0, 5, 4], float)          # g0: positive ranked 3rd; g1: 1st
    expected = np.mean([ndcg_at_k([0, 1, 2], {2}, 10), 1.0])
    assert group_ndcg(blk, scores) == pytest.approx(expected)
    assert expected == pytest.approx((1 / np.log2(4) + 1) / 2)


def test_neural_fixed_epochs_refit_and_columns():
    tr, va = _block(seed=1), _block(seed=2)
    cols = [0, 4]
    sel = NeuralRanker([16], 0.0, 1e-2, 15, 16, 3, "cpu", 0, 2, columns=cols).fit(tr, va, NAMES)
    assert 1 <= sel.best_epochs_ <= len(sel.valid_curve_)
    ref = NeuralRanker([16], 0.0, 1e-2, sel.best_epochs_, 16, 3, "cpu", 0, 2, columns=cols).fit(va, None, NAMES)
    assert ref.valid_curve_ == [] and ref.best_epochs_ == sel.best_epochs_
    assert np.isnan(ref.best_valid_ndcg_)
    # predict() slices the full block itself: same as a model that only ever saw the subset
    sliced = NeuralRanker([16], 0.0, 1e-2, sel.best_epochs_, 16, 3, "cpu", 0, 2).fit(
        va.with_columns(cols), None, NAMES)
    np.testing.assert_allclose(ref.predict(tr), sliced.predict(tr.with_columns(cols)), rtol=1e-5, atol=1e-6)
    assert group_ndcg(tr, ref.predict(tr)) > 0.5


def test_gbdt_fixed_rounds_refit_and_columns():
    tr, va = _block(seed=1), _block(seed=2)
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "num_leaves": 4,
              "min_data_in_leaf": 5, "learning_rate": 0.1, "verbose": -1}
    sel = GBDTRanker(params, 50, 10, 0, columns=[0, 4]).fit(tr, va, NAMES)
    assert set(sel.feature_importance_) == {"score_popular", "item_ctr"}
    ref = GBDTRanker(params, 7, 10, 0, columns=[0, 4]).fit(va, None, NAMES)
    assert ref.best_iteration_ == 7
    assert group_ndcg(tr, ref.predict(tr)) > 0.5
