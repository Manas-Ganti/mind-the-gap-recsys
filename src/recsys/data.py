"""[1] INGEST (Spark): parse MIND impression logs into typed, integer-indexed parquet.

MIND layout (per split directory MIND{variant}_{train,dev}/):
  behaviors.tsv  impression_id  user_id  time  history  impressions
                 time        "11/15/2019 8:55:22 AM" (naive; parsed as UTC)
                 history     space-separated news ids clicked *before* this impression
                 impressions space-separated "N123-1" tokens: item shown, 1 = clicked
  news.tsv       news_id category subcategory title abstract url title_entities abstract_entities

Outputs under {processed_dir}/ingest/:
  impressions/   one row per (impression, shown item): impression_id u i ts position clicked
  histories/     one row per (user, history version, slot): u first_ts hist_pos i
  news/          i item_id category subcategory title abstract
  users/, items/ global integer index (sorted ids), shared by every split so models fitted
                 on one split can be applied to another.

About `history`: on MIND it is a fixed per-user snapshot of clicks from before the log
window (one version per user across every impression; verified on MIND-small). With
`split.history_policy: snapshot` the split uses it for every user and asserts the
one-version property, so a dataset where history grows with in-window clicks fails loudly.
"""
from __future__ import annotations

import json
from pathlib import Path

from recsys.config import Config

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = ["item_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
TIME_FORMAT = "M/d/yyyy h:mm:ss a"


def ingest_dir(cfg: Config) -> Path:
    return Path(cfg.paths.processed_dir) / "ingest"


def _read_tsv(spark, path: Path, cols: list[str]):
    schema = ", ".join(f"`{c}` string" for c in cols)
    return (
        spark.read.option("sep", "\t").option("header", "false")
        .option("quote", "")          # MIND titles contain raw quotes; disable quoting
        .option("mode", "FAILFAST")
        .schema(schema).csv(str(path))
    )


def _raw_split_dir(cfg: Config, split: str) -> Path:
    return Path(cfg.paths.raw_dir) / f"MIND{cfg.data.variant}_{split}"


def ingest(cfg: Config, spark=None) -> dict:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    from recsys.spark import get_spark

    spark = spark or get_spark(cfg, "recsys-ingest")
    out = ingest_dir(cfg)

    behaviors, news = None, None
    for split in cfg.data.raw_splits:
        d = _raw_split_dir(cfg, split)
        if not (d / "behaviors.tsv").exists():
            raise FileNotFoundError(f"{d}/behaviors.tsv missing; see data/download.md")
        b = _read_tsv(spark, d / "behaviors.tsv", BEHAVIOR_COLS).withColumn("source_split", F.lit(split))
        n = _read_tsv(spark, d / "news.tsv", NEWS_COLS)
        behaviors = b if behaviors is None else behaviors.unionByName(b)
        news = n if news is None else news.unionByName(n)

    behaviors = (
        behaviors
        .withColumn("ts", F.unix_timestamp(F.to_timestamp("time", TIME_FORMAT)))
        # MIND impression ids restart at 1 in every file
        .withColumn("impression_id", F.concat_ws("-", "source_split", "impression_id"))
    ).cache()
    bad_time = behaviors.filter(F.col("ts").isNull()).count()
    if bad_time:
        raise ValueError(f"{bad_time} behaviors rows have unparseable timestamps ({TIME_FORMAT})")

    imp_raw = (
        behaviors.select("impression_id", "user_id", "ts",
                         F.posexplode(F.split("impressions", " ")).alias("position", "token"))
        .filter(F.length("token") > 0)
        .select("impression_id", "user_id", "ts", "position",
                F.substring_index("token", "-", 1).alias("item_id"),
                F.substring_index("token", "-", -1).cast("int").alias("clicked"))
    )
    bad_labels = imp_raw.filter(~F.col("clicked").isin(0, 1) | F.col("clicked").isNull()).count()
    if bad_labels:
        raise ValueError(f"{bad_labels} impression tokens lack a 0/1 label (was an unlabeled test split ingested?)")

    hist_raw = (
        behaviors.filter(F.col("history").isNotNull() & (F.length("history") > 0))
        .groupBy("user_id", "history").agg(F.min("ts").alias("first_ts"))
        .select("user_id", "first_ts", F.posexplode(F.split("history", " ")).alias("hist_pos", "item_id"))
        .filter(F.length("item_id") > 0)
    )

    news = news.dropDuplicates(["item_id"]).select("item_id", "category", "subcategory", "title", "abstract")

    # Global, deterministic integer indices (sorted ids -> 0..n-1).
    users = (imp_raw.select("user_id").union(hist_raw.select("user_id")).distinct()
             .withColumn("u", F.row_number().over(Window.orderBy("user_id")) - 1))
    items = (news.select("item_id").union(imp_raw.select("item_id")).union(hist_raw.select("item_id"))
             .distinct().withColumn("i", F.row_number().over(Window.orderBy("item_id")) - 1))
    users.write.mode("overwrite").parquet(str(out / "users"))
    items.write.mode("overwrite").parquet(str(out / "items"))
    users = spark.read.parquet(str(out / "users"))
    items = spark.read.parquet(str(out / "items"))

    impressions = (imp_raw.join(users, "user_id").join(F.broadcast(items), "item_id")
                   .select("impression_id", "u", "i", "ts", "position", "clicked"))
    impressions.write.mode("overwrite").parquet(str(out / "impressions"))
    histories = (hist_raw.join(users, "user_id").join(F.broadcast(items), "item_id")
                 .select("u", "first_ts", "hist_pos", "i"))
    histories.write.mode("overwrite").parquet(str(out / "histories"))
    news.join(F.broadcast(items), "item_id").write.mode("overwrite").parquet(str(out / "news"))

    stats = dataset_stats(spark, cfg)
    results = Path(cfg.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    (results / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    behaviors.unpersist()
    return stats


def dataset_stats(spark, cfg: Config) -> dict:
    from pyspark.sql import functions as F

    d = ingest_dir(cfg)
    imp = spark.read.parquet(str(d / "impressions"))
    hist = spark.read.parquet(str(d / "histories"))
    n_users = spark.read.parquet(str(d / "users")).count()
    n_items = spark.read.parquet(str(d / "items")).count()

    agg = imp.agg(
        F.countDistinct("impression_id").alias("impressions"),
        F.count("*").alias("impression_rows"),
        F.sum("clicked").alias("clicks"),
        F.countDistinct("u").alias("users_with_impressions"),
        F.countDistinct("i").alias("items_shown"),
        F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts"),
    ).first().asDict()
    clicked_items = imp.filter("clicked = 1").select("i").distinct().count()
    hist_pairs = hist.select("u", "i").distinct().count()
    click_pairs = imp.filter("clicked = 1").select("u", "i").union(hist.select("u", "i")).distinct().count()
    per_day = {
        r["day"]: {"impressions": r["impressions"], "clicks": r["clicks"]}
        for r in imp.groupBy(F.to_date(F.from_unixtime("ts")).cast("string").alias("day"))
        .agg(F.countDistinct("impression_id").alias("impressions"), F.sum("clicked").alias("clicks"))
        .orderBy("day").collect()
    }
    return {
        "variant": cfg.data.variant,
        "raw_splits": cfg.data.raw_splits,
        "users": n_users,
        "items": n_items,
        **agg,
        "items_clicked": clicked_items,
        "history_user_item_pairs": hist_pairs,
        "positive_user_item_pairs": click_pairs,
        "ctr": agg["clicks"] / agg["impression_rows"],
        "sparsity": 1.0 - click_pairs / (n_users * n_items),
        "per_day": per_day,
    }
