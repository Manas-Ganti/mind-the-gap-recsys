"""[2] SPLIT + AUDIT.

Three splits, all written under {processed_dir}/splits/{name}/:

  temporal      train: impressions before `temporal_cutoff`   test: clicks at/after it   (PRIMARY)
  temporal_val  train: before `validation_cutoff`              test: [validation_cutoff, temporal_cutoff)
                used for hyperparameter tuning and ranker training; never touches temporal test
  random        test: a seeded random sample of window clicks, same size fraction as temporal.
                COMPARISON ARTIFACT ONLY. It lets the model see Tuesday while predicting Monday.

Each split directory holds:
  train/        u i ts hist_pos       (ts null = pre-window history click, ordered by hist_pos)
  test/         u i ts day is_eval        (distinct user-item clicks, re-clicks of train items removed)
  catalog/      i                     (union of items rankable at serving time)
  catalog_days/ day i                 (items live on each test day; a click is ranked against
                                       its own day's items, identically in every split)
  eval_users/   u
  negatives/    u pos negs[]          (sampled-negative protocol; shared by every method => paired)
  item_stats/   i impressions clicks first_seen last_seen   (train-side impressions only)
  meta.json

Ingestion, splitting and negative sampling are Spark; the audit reads the written parquet
with pandas because it is a set of assertions over already-small outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Config, SplitCfg

SPLITS = ("temporal", "temporal_val", "random")


def split_dir(cfg: Config, name: str) -> Path:
    return Path(cfg.paths.processed_dir) / "splits" / name


# --------------------------------------------------------------------------- Spark side

class LeakageError(AssertionError):
    pass


def _history(hist, cfg: Config, cutoff_ts: int | None):
    """User history rows allowed into train for a split with this cutoff.

    snapshot: every user's history, after asserting it is a single version per user (if a
    user's history changed across impressions it would contain in-window clicks, possibly
    from the test period, and the snapshot assumption would be false).
    visible_before_cutoff: only history versions first attached to a pre-cutoff impression."""
    from pyspark.sql import functions as F

    if cfg.split.history_policy == "snapshot":
        multi = hist.groupBy("u").agg(F.countDistinct("first_ts").alias("v")).filter("v > 1").count()
        if multi:
            raise LeakageError(f"{multi} users have >1 history version; MIND history is not a fixed "
                               "snapshot here. Use split.history_policy=visible_before_cutoff.")
        return _latest_history(hist, None)
    return _latest_history(hist, cutoff_ts)


def _latest_history(hist, cutoff_ts: int | None):
    """Per user, the latest history version visible strictly before `cutoff_ts`."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    h = hist if cutoff_ts is None else hist.filter(F.col("first_ts") < cutoff_ts)
    w = Window.partitionBy("u")
    h = h.withColumn("_mx", F.max("first_ts").over(w)).filter(F.col("first_ts") == F.col("_mx"))
    return h.groupBy("u", "i").agg(F.min("hist_pos").alias("hist_pos"))


def _build(spark, cfg: Config, name: str, *, train_rows, test_click_rows, all_rows, history,
           reference_ts: int | None, max_eval_users: int, cutoff_ts: int | None = None) -> dict:
    """Shared split construction. `train_rows`: impression rows visible at training time.
    `test_click_rows`: clicked impression rows held out. `all_rows`: every impression row
    (used only to resolve which items were live on each test day)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    out = split_dir(cfg, name)
    seed = cfg.seed

    window_clicks = (train_rows.filter("clicked = 1").groupBy("u", "i")
                     .agg(F.min("ts").alias("ts")).withColumn("hist_pos", F.lit(None).cast("int")))
    hist = history.withColumn("ts", F.lit(None).cast("long")).select("u", "i", "ts", "hist_pos")
    # An item both in history and clicked in-window keeps its history slot (it was seen first).
    train = (hist.unionByName(window_clicks).groupBy("u", "i")
             .agg(F.min("hist_pos").alias("hist_pos"), F.min("ts").alias("_ts"))
             .withColumn("ts", F.when(F.col("hist_pos").isNull(), F.col("_ts")))
             .select("u", "i", "ts", "hist_pos"))
    train.write.mode("overwrite").parquet(str(out / "train"))
    train = spark.read.parquet(str(out / "train"))

    day = F.date_format(F.from_unixtime("ts"), "yyyy-MM-dd") if cfg.split.catalog == "live" else F.lit("all")
    test_all = test_click_rows.groupBy("u", "i").agg(F.min("ts").alias("ts")).withColumn("day", day)
    n_test_raw = test_all.count()
    test = test_all.join(train.select("u", "i"), ["u", "i"], "left_anti")
    n_test = test.count()

    test_users = test.select("u").distinct()
    eval_users = (test_users.withColumn("_h", F.xxhash64("u", F.lit(seed)))
                  .orderBy("_h", "u").limit(max_eval_users).select("u"))
    eval_users.write.mode("overwrite").parquet(str(out / "eval_users"))
    eval_users = spark.read.parquet(str(out / "eval_users"))
    # Each eval user is evaluated on ONE day's request (seeded choice among their test days),
    # ranked against that day's live items. A no-op for a one-day temporal test window; for
    # the random split it keeps the per-user candidate pool the same size as in temporal.
    eval_day = (test.join(eval_users, "u").select("u", "day").distinct()
                .withColumn("_h", F.xxhash64("u", "day", F.lit(seed)))
                .withColumn("_r", F.row_number().over(Window.partitionBy("u").orderBy("_h", "day")))
                .filter("_r = 1").select("u", F.col("day").alias("_eval_day")))
    test = (test.join(eval_day, "u", "left")
            .withColumn("is_eval", F.coalesce(F.col("day") == F.col("_eval_day"), F.lit(False)))
            .drop("_eval_day"))
    test.write.mode("overwrite").parquet(str(out / "test"))
    test = spark.read.parquet(str(out / "test"))

    # Availability is defined identically for every split, so temporal and random are
    # comparable: an item is rankable for a test click iff it was shown on that click's day.
    if cfg.split.catalog == "live":
        test_days = test.select("day").distinct()
        catalog_days = (all_rows.withColumn("day", F.date_format(F.from_unixtime("ts"), "yyyy-MM-dd"))
                        .join(F.broadcast(test_days), "day").select("day", "i").distinct())
    else:
        catalog_days = (spark.read.parquet(str(Path(cfg.paths.processed_dir) / "ingest" / "items"))
                        .select(F.lit("all").alias("day"), "i"))
    catalog_days.write.mode("overwrite").parquet(str(out / "catalog_days"))
    catalog_days = spark.read.parquet(str(out / "catalog_days"))
    catalog = catalog_days.select("i").distinct()
    catalog.write.mode("overwrite").parquet(str(out / "catalog"))
    catalog = spark.read.parquet(str(out / "catalog"))

    item_stats = train_rows.groupBy("i").agg(
        F.count("*").alias("impressions"), F.sum("clicked").alias("clicks"),
        F.min("ts").alias("first_seen"), F.max("ts").alias("last_seen"))
    item_stats.write.mode("overwrite").parquet(str(out / "item_stats"))

    neg_info = sample_negatives(spark, cfg, out, train=train, test=test, catalog_days=catalog_days,
                                eval_users=eval_users)

    meta = {
        "name": name,
        "reference_ts": None if reference_ts is None else int(reference_ts),
        "train_pairs": train.count(),
        "test_pairs_before_reclick_removal": n_test_raw,
        "test_pairs": n_test,
        "reclicks_removed": n_test_raw - n_test,
        "test_users": test_users.count(),
        "eval_users": eval_users.count(),
        "catalog_size": catalog.count(),
        "history_policy": cfg.split.history_policy,
        "catalog_days": catalog_days.select("day").distinct().count(),
        "mean_items_per_day": catalog_days.groupBy("day").count().agg(F.avg("count")).first()[0],
        **neg_info,
    }
    if cutoff_ts is not None:
        # History items first *shown* at/after the cutoff would mean MIND's `history` field
        # carries information from the future.
        first_shown = all_rows.groupBy("i").agg(F.min("ts").alias("first_shown"))
        meta["history_items_first_shown_after_cutoff"] = (
            history.select("i").distinct().join(first_shown, "i")
            .filter(F.col("first_shown") >= cutoff_ts).count())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def sample_negatives(spark, cfg: Config, out: Path, *, train, test, catalog_days, eval_users) -> dict:
    """Uniform negatives for the sampled-negative protocol, drawn once and shared by every
    method so the comparison is paired. A negative for (u, pos) is an item live on the day
    of that click which u did not click in train or test. Draws use xxhash64(u, pos, j, seed)
    so they are deterministic regardless of partitioning (F.rand is not)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    n_neg = cfg.eval.n_negatives
    m = int(math.ceil(n_neg * cfg.eval.negative_oversample))
    cat = catalog_days.withColumn("cidx", F.row_number().over(Window.partitionBy("day").orderBy("i")) - 1)
    sizes = cat.groupBy("day").agg(F.count("*").alias("c"))
    pairs = test.filter("is_eval").select("u", F.col("i").alias("pos"), "day")
    draws = (pairs.join(F.broadcast(sizes), "day")
             .select("u", "pos", "day", "c", F.explode(F.sequence(F.lit(0), F.lit(m - 1))).alias("j"))
             .withColumn("cidx", F.pmod(F.xxhash64("u", "pos", "j", F.lit(cfg.seed)), F.col("c")))
             .join(F.broadcast(cat), ["day", "cidx"]).withColumnRenamed("i", "neg"))
    excluded = (train.select("u", "i").union(test.select("u", "i"))
                .join(eval_users, "u").withColumnRenamed("i", "neg"))
    draws = (draws.join(excluded, ["u", "neg"], "left_anti")
             .groupBy("u", "pos", "neg").agg(F.min("j").alias("j")))
    w = Window.partitionBy("u", "pos").orderBy("j")
    negs = (draws.withColumn("r", F.row_number().over(w)).filter(F.col("r") <= n_neg)
            .groupBy("u", "pos")
            .agg(F.sort_array(F.collect_list(F.struct("j", "neg"))).alias("_s"))
            .select("u", "pos", F.col("_s.neg").alias("negs")))
    # Pairs whose draws were all excluded still need a row (with an empty list) to be counted.
    negs = pairs.select("u", "pos").join(negs, ["u", "pos"], "left")
    negs.write.mode("overwrite").parquet(str(out / "negatives"))
    negs = spark.read.parquet(str(out / "negatives"))
    sizes = negs.select(F.coalesce(F.size("negs"), F.lit(0)).alias("n"))
    s = sizes.agg(F.min("n").alias("mn"), F.avg("n").alias("avg"),
                  F.sum((F.col("n") < n_neg).cast("int")).alias("short")).first()
    return {"n_negatives": n_neg, "negatives_min": int(s["mn"] or 0),
            "negatives_mean": float(s["avg"] or 0.0), "pairs_short_of_negatives": int(s["short"] or 0)}


def build_splits(cfg: Config, spark=None, only: list[str] | None = None) -> dict:
    from pyspark.sql import functions as F

    from recsys.spark import get_spark

    spark = spark or get_spark(cfg, "recsys-split")
    ing = Path(cfg.paths.processed_dir) / "ingest"
    imp = spark.read.parquet(str(ing / "impressions")).cache()
    hist = spark.read.parquet(str(ing / "histories"))
    cut = SplitCfg.epoch(cfg.split.temporal_cutoff)
    vcut = SplitCfg.epoch(cfg.split.validation_cutoff)
    todo = only or list(SPLITS)
    metas = {}

    if "temporal" in todo or "random" in todo:
        metas["temporal"] = _build(
            spark, cfg, "temporal",
            train_rows=imp.filter(F.col("ts") < cut),
            test_click_rows=imp.filter((F.col("ts") >= cut) & (F.col("clicked") == 1)),
            all_rows=imp, history=_history(hist, cfg, cut),
            reference_ts=cut, max_eval_users=cfg.split.max_eval_users, cutoff_ts=cut)

    if "temporal_val" in todo:
        before = imp.filter(F.col("ts") < cut)
        metas["temporal_val"] = _build(
            spark, cfg, "temporal_val",
            train_rows=before.filter(F.col("ts") < vcut),
            test_click_rows=before.filter((F.col("ts") >= vcut) & (F.col("clicked") == 1)),
            all_rows=before, history=_history(hist, cfg, vcut),
            reference_ts=vcut, max_eval_users=cfg.split.max_val_users, cutoff_ts=vcut)

    if "random" in todo:
        clicks = imp.filter("clicked = 1")
        frac = cfg.split.random_test_fraction
        if frac is None:
            n_window = clicks.count()
            n_temporal = imp.filter((F.col("ts") >= cut) & (F.col("clicked") == 1)).count()
            frac = n_temporal / n_window
        bucket = F.pmod(F.xxhash64("impression_id", "i", F.lit(cfg.seed)), F.lit(1_000_000))
        test_rows = clicks.filter(bucket < int(frac * 1_000_000))
        train_rows = imp.join(test_rows.select("impression_id", "i"), ["impression_id", "i"], "left_anti")
        # A random split has no serving time: reference_ts is None, so time-windowed signals
        # (popularity window, recency features) fall back to all train data.
        metas["random"] = _build(
            spark, cfg, "random", train_rows=train_rows, test_click_rows=test_rows,
            all_rows=imp, history=_history(hist, cfg, None),
            reference_ts=None, max_eval_users=cfg.split.max_eval_users)
        metas["random"]["test_fraction"] = frac

    audits = {}
    for name in todo:
        audits[name] = audit_split(cfg, name, cutoff_ts={"temporal": cut, "temporal_val": vcut}.get(name))
    results = Path(cfg.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    prev = results / "split_audit.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    merged.update({n: {"meta": metas.get(n), "audit": audits[n]} for n in todo})
    prev.write_text(json.dumps(merged, indent=2))
    imp.unpersist()
    return merged


# --------------------------------------------------------------------------- audit (pandas)


def assert_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """No test (user, item) pair may appear in train."""
    overlap = train[["u", "i"]].merge(test[["u", "i"]], on=["u", "i"])
    if len(overlap):
        raise LeakageError(f"{len(overlap)} test user-item pairs also appear in train")


def assert_temporal(train: pd.DataFrame, test: pd.DataFrame, cutoff_ts: int) -> None:
    """Every in-window train click strictly precedes the cutoff; every test click is at/after it."""
    tr_ts = train["ts"].dropna()
    if len(tr_ts) and tr_ts.max() >= cutoff_ts:
        raise LeakageError(f"train click at ts={tr_ts.max()} is not before cutoff {cutoff_ts}")
    if len(test) and test["ts"].min() < cutoff_ts:
        raise LeakageError(f"test click at ts={test['ts'].min()} precedes cutoff {cutoff_ts}")


def cold_start_rates(train: pd.DataFrame, test: pd.DataFrame, catalog: np.ndarray) -> dict:
    """Cold = absent from train entirely. Random splits have far lower cold rates than
    temporal ones, which is a large part of why they look better."""
    train_users = train["u"].unique()
    train_items = train["i"].unique()
    test_users = test["u"].unique()
    test_items = test["i"].unique()

    def rate(x: np.ndarray, seen: np.ndarray) -> float:
        return float((~np.isin(x, seen)).mean()) if len(x) else 0.0

    return {
        "cold_user_rate": rate(test_users, train_users),
        "cold_item_rate": rate(test_items, train_items),
        "cold_positive_rate": rate(test["i"].to_numpy(), train_items),
        "cold_catalog_rate": rate(np.asarray(catalog), train_items),
    }


def audit_split(cfg: Config, name: str, cutoff_ts: int | None) -> dict:
    d = split_dir(cfg, name)
    train = pd.read_parquet(d / "train")
    test = pd.read_parquet(d / "test")
    catalog = pd.read_parquet(d / "catalog")["i"].to_numpy()
    assert_no_overlap(train, test)
    if cutoff_ts is not None:
        assert_temporal(train, test, cutoff_ts)
    days = pd.read_parquet(d / "catalog_days")
    live = test[["day", "i"]].merge(days.assign(_ok=True), on=["day", "i"], how="left")["_ok"].fillna(False)
    if not live.all():
        raise LeakageError(f"{int((~live).sum())} test positives were not live on their click day")
    report = {"checks": ["no_train_test_overlap", "positives_live_on_click_day"]
              + (["temporal_order"] if cutoff_ts is not None else []),
              **cold_start_rates(train, test, catalog)}
    return report
