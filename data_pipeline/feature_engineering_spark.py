"""Optional Spark window implementation for the daily transaction pickle data.

Pandas reads the source pickle shards (Spark has no native pickle reader);
feature windows are then computed with Spark SQL over the transaction table.
This local adapter is useful for learning Spark windows, but production-scale
ingestion should first convert trusted source data to a distributed format.
"""
import glob
import os
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from data_pipeline.feature_engineering import (
    AMOUNT_COLUMN,
    CUSTOMER_COLUMN,
    DATA_DIR,
    FEATURE_COLUMNS,
    FEATURE_PATH,
    LABEL_COLUMN,
    TERMINAL_COLUMN,
    TIMESTAMP_COLUMN,
    load_transactions,
)

SPARK_OUTPUT_DIR = DATA_DIR / "creditcard_features_spark"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("fraud-detection-feature-engineering")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def build_features_spark(data_dir: str | Path = DATA_DIR,
                         out_dir: str | Path = SPARK_OUTPUT_DIR,
                         out_parquet: str | Path = FEATURE_PATH):
    pandas_df = load_transactions(data_dir)
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame(pandas_df)
        df = df.withColumn(
            "_seconds_in_day",
            F.hour(TIMESTAMP_COLUMN) * 3600
            + F.minute(TIMESTAMP_COLUMN) * 60
            + F.second(TIMESTAMP_COLUMN),
        )
        angle = F.lit(2 * 3.141592653589793 / 86400) * F.col("_seconds_in_day")
        df = df.withColumn("HOUR_SIN", F.sin(angle)).withColumn("HOUR_COS", F.cos(angle))

        customer_1h = (
            Window.partitionBy(CUSTOMER_COLUMN)
            .orderBy("TX_TIME_SECONDS")
            .rangeBetween(-3600, -1)
        )
        customer_24h = (
            Window.partitionBy(CUSTOMER_COLUMN)
            .orderBy("TX_TIME_SECONDS")
            .rangeBetween(-86400, -1)
        )
        terminal_1h = (
            Window.partitionBy(TERMINAL_COLUMN)
            .orderBy("TX_TIME_SECONDS")
            .rangeBetween(-3600, -1)
        )
        terminal_24h = (
            Window.partitionBy(TERMINAL_COLUMN)
            .orderBy("TX_TIME_SECONDS")
            .rangeBetween(-86400, -1)
        )
        customer_same_second = (
            Window.partitionBy(CUSTOMER_COLUMN, "TX_TIME_SECONDS")
            .orderBy("TRANSACTION_ID")
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        terminal_same_second = (
            Window.partitionBy(TERMINAL_COLUMN, "TX_TIME_SECONDS")
            .orderBy("TRANSACTION_ID")
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        customer_history = (
            Window.partitionBy(CUSTOMER_COLUMN)
            .orderBy("TX_TIME_SECONDS", "TRANSACTION_ID")
            .rowsBetween(Window.unboundedPreceding, -1)
        )
        customer_terminal_history = (
            Window.partitionBy(CUSTOMER_COLUMN, TERMINAL_COLUMN)
            .orderBy("TX_TIME_SECONDS", "TRANSACTION_ID")
            .rowsBetween(Window.unboundedPreceding, -1)
        )

        df = (
            df.withColumn(
                "IS_NEW_TERMINAL",
                F.coalesce(F.count("*").over(customer_terminal_history) == 0, F.lit(True)),
            )
            .withColumn("_customer_same_second_count", F.count("*").over(customer_same_second))
            .withColumn(
                "_customer_same_second_amount",
                F.coalesce(F.sum(AMOUNT_COLUMN).over(customer_same_second), F.lit(0.0)),
            )
            .withColumn("_terminal_same_second_count", F.count("*").over(terminal_same_second))
            .withColumn(
                "CUSTOMER_TX_COUNT_1H",
                F.count("*").over(customer_1h) + F.col("_customer_same_second_count"),
            )
            .withColumn(
                "CUSTOMER_TX_COUNT_24H",
                F.count("*").over(customer_24h) + F.col("_customer_same_second_count"),
            )
            .withColumn(
                "CUSTOMER_AMOUNT_SUM_1H",
                F.coalesce(F.sum(AMOUNT_COLUMN).over(customer_1h), F.lit(0.0))
                + F.col("_customer_same_second_amount"),
            )
            .withColumn(
                "TERMINAL_TX_COUNT_1H",
                F.count("*").over(terminal_1h) + F.col("_terminal_same_second_count"),
            )
            .withColumn(
                "TERMINAL_TX_COUNT_24H",
                F.count("*").over(terminal_24h) + F.col("_terminal_same_second_count"),
            )
            .withColumn("_customer_mean", F.avg(AMOUNT_COLUMN).over(customer_history))
            .withColumn("_customer_std", F.stddev_samp(AMOUNT_COLUMN).over(customer_history))
            .withColumn(
                "CUSTOMER_AMOUNT_ZSCORE",
                F.when(
                    F.col("_customer_std").isNull() | (F.col("_customer_std") <= 1e-9),
                    F.lit(0.0),
                ).otherwise(
                    (F.col(AMOUNT_COLUMN) - F.col("_customer_mean")) / F.col("_customer_std")
                ),
            )
            .withColumn(
                "CUSTOMER_AMOUNT_RATIO",
                F.when(
                    F.col("_customer_mean").isNull() | (F.col("_customer_mean") <= 1e-9),
                    F.lit(1.0),
                ).otherwise(F.col(AMOUNT_COLUMN) / F.col("_customer_mean")),
            )
        )

        n_total = df.count()
        n_fraud = df.filter(F.col(LABEL_COLUMN) == 1).count()
        print(f"[spark-features] {n_total:,} rows, {n_fraud:,} fraud")
        out_dir = Path(out_dir)
        df.drop(
            "_seconds_in_day",
            "_customer_mean",
            "_customer_std",
            "_customer_same_second_count",
            "_customer_same_second_amount",
            "_terminal_same_second_count",
        ) \
            .orderBy(TIMESTAMP_COLUMN, "TRANSACTION_ID") \
            .coalesce(1).write.mode("overwrite").parquet(str(out_dir))
    finally:
        spark.stop()

    part_files = glob.glob(os.path.join(str(out_dir), "part-*.parquet"))
    if len(part_files) != 1:
        raise RuntimeError(f"Expected one Spark parquet part, found {len(part_files)}")
    result = pd.read_parquet(part_files[0])
    result = result.sort_values([TIMESTAMP_COLUMN, "TRANSACTION_ID"]).reset_index(drop=True)
    if not set(FEATURE_COLUMNS).issubset(result.columns):
        raise ValueError("Spark output does not contain the expected model feature columns")
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_parquet, index=False)
    print(f"[spark-features] wrote {out_parquet} with shape {result.shape}")
    return result


if __name__ == "__main__":
    build_features_spark()
