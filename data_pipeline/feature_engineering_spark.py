"""Optional Spark CSV validation and Parquet materialization path.

The 2023 dataset has no transaction timestamp or card identifier, so this path
does not fabricate time-window or per-card features. It selects and orders the
source columns used by the pandas path.
"""
import os
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).parent.parent
RAW_PATH = ROOT / "data" / "creditcard_2023.csv"
FEATURE_PATH = ROOT / "data" / "creditcard_features.parquet"
FEATURE_PATH_SPARK_DIR = ROOT / "data" / "creditcard_features_spark"
REQUIRED_COLUMNS = ["id", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("fraud-detection-data-preparation")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def build_features_spark(raw_path: str | Path = RAW_PATH,
                         out_dir: str | Path = FEATURE_PATH_SPARK_DIR,
                         out_parquet: str | Path = FEATURE_PATH):
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        print(f"[spark-data] reading {raw_path}")
        df = spark.read.csv(str(raw_path), header=True, inferSchema=True)
        missing = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        df = df.select(*REQUIRED_COLUMNS).orderBy("id")
        n_total = df.count()
        n_fraud = df.filter("Class = 1").count()
        print(f"[spark-data] {n_total:,} rows, fraud rate = {n_fraud / n_total:.4%}")
        df.coalesce(1).write.mode("overwrite").parquet(str(out_dir))
    finally:
        spark.stop()

    import glob
    import pandas as pd

    part_files = glob.glob(os.path.join(str(out_dir), "part-*.parquet"))
    if len(part_files) != 1:
        raise RuntimeError(f"Expected one Spark parquet part, found {len(part_files)}")
    pdf = pd.read_parquet(part_files[0])
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(out_parquet, index=False)
    print(f"[spark-data] wrote {out_parquet} with shape {pdf.shape}")
    return pdf


if __name__ == "__main__":
    build_features_spark()
