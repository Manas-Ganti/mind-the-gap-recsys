"""SparkSession factory. Session timezone is pinned to UTC so MIND's naive timestamps parse
identically on every machine (otherwise DST in the driver's locale shifts the split cutoff)."""
from __future__ import annotations

from recsys.config import Config


def get_spark(cfg: Config, app: str = "recsys"):
    from pyspark.sql import SparkSession

    b = (
        SparkSession.builder.appName(app)
        .master(cfg.spark.master)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.ui.showConsoleProgress", "false")
    )
    for k, v in cfg.spark.extra.items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
