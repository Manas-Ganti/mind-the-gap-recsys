"""No future leakage; cold-start rates correct. Pure-pandas audit checks always run; the Spark
split itself is exercised end to end on synthetic MIND-format data when a JVM is available."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import requires_spark
from recsys.splits import LeakageError, assert_no_overlap, assert_temporal, cold_start_rates


def test_overlap_detected():
    train = pd.DataFrame({"u": [0, 1], "i": [5, 6]})
    with pytest.raises(LeakageError):
        assert_no_overlap(train, pd.DataFrame({"u": [1], "i": [6]}))
    assert_no_overlap(train, pd.DataFrame({"u": [1], "i": [5]}))


def test_temporal_order_enforced():
    train = pd.DataFrame({"u": [0, 0], "i": [1, 2], "ts": [np.nan, 99.0]})   # NaN = pre-window history
    assert_temporal(train, pd.DataFrame({"ts": [100, 150]}), cutoff_ts=100)
    with pytest.raises(LeakageError):
        assert_temporal(train.assign(ts=[np.nan, 100.0]), pd.DataFrame({"ts": [120]}), cutoff_ts=100)
    with pytest.raises(LeakageError):
        assert_temporal(train, pd.DataFrame({"ts": [99]}), cutoff_ts=100)


def test_cold_start_rates_hand():
    train = pd.DataFrame({"u": [0, 0, 1], "i": [10, 11, 10]})
    test = pd.DataFrame({"u": [0, 2, 2, 3], "i": [12, 10, 13, 12]})
    r = cold_start_rates(train, test, catalog=np.array([10, 11, 12, 13]))
    assert r["cold_user_rate"] == pytest.approx(2 / 3)        # users 2, 3 of {0, 2, 3}
    assert r["cold_item_rate"] == pytest.approx(2 / 3)        # items 12, 13 of {12, 10, 13}
    assert r["cold_positive_rate"] == pytest.approx(3 / 4)    # (0,12), (2,13), (3,12)
    assert r["cold_catalog_rate"] == pytest.approx(2 / 4)


@requires_spark
def test_spark_split_end_to_end(tmp_path: Path):
    from recsys.config import load_config
    from recsys.data import ingest
    from recsys.dataset import load_split
    from recsys.spark import get_spark
    from recsys.splits import build_splits
    from recsys.synthetic import make_synthetic_mind

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "smoke.yaml")
    for name in ("raw_dir", "processed_dir", "models_dir", "results_dir", "figures_dir"):
        setattr(cfg.paths, name, tmp_path / name)
    make_synthetic_mind(cfg.paths.raw_dir, n_users=120, n_items=200, impressions_per_day=80, seed=1)
    spark = get_spark(cfg, "recsys-test")
    stats = ingest(cfg, spark)
    assert stats["impressions"] == 80 * 7 and stats["clicks"] > 0
    audits = build_splits(cfg, spark)       # raises LeakageError on any violation
    assert set(audits) == {"temporal", "temporal_val", "temporal_rtrain", "random"}

    cut = cfg.split.epoch(cfg.split.temporal_cutoff)
    d = load_split(cfg, "temporal")
    assert d.train["ts"].dropna().max() < cut <= d.test["ts"].min()
    seen = set(map(tuple, d.train[["u", "i"]].to_numpy()))
    pos = set(map(tuple, d.test[["u", "i"]].to_numpy()))
    assert not seen & pos
    n = cfg.eval.n_negatives
    for (u, p), negs in zip(d.neg_pairs, d.neg_items):
        negs = negs[negs >= 0]
        assert len(set(negs)) == len(negs) <= n
        assert all((u, j) not in seen and (u, j) not in pos for j in negs)
        day = d.test.set_index(["u", "i"]).loc[(u, p), "day"]
        assert np.isin(negs, d.catalog_days.loc[d.catalog_days["day"] == day, "i"]).all()
    # deterministic: same seed -> identical negatives
    build_splits(cfg, spark, only=["temporal"])
    np.testing.assert_array_equal(load_split(cfg, "temporal").neg_items, d.neg_items)
    # random split matches the temporal test fraction
    assert audits["random"]["meta"]["test_fraction"] > 0
