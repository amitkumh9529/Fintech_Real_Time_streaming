"""Load daily transaction-simulator pickles and build causal fraud features."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_GLOB = "*.pkl"
FEATURE_PATH = DATA_DIR / "creditcard_features.parquet"
LABEL_COLUMN = "TX_FRAUD"
TIMESTAMP_COLUMN = "TX_DATETIME"
CUSTOMER_COLUMN = "CUSTOMER_ID"
TERMINAL_COLUMN = "TERMINAL_ID"
AMOUNT_COLUMN = "TX_AMOUNT"
REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    TIMESTAMP_COLUMN,
    CUSTOMER_COLUMN,
    TERMINAL_COLUMN,
    AMOUNT_COLUMN,
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    LABEL_COLUMN,
    "TX_FRAUD_SCENARIO",
]
FEATURE_COLUMNS = [
    AMOUNT_COLUMN,
    "HOUR_SIN",
    "HOUR_COS",
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_TX_COUNT_24H",
    "CUSTOMER_AMOUNT_SUM_1H",
    "CUSTOMER_AMOUNT_ZSCORE",
    "CUSTOMER_AMOUNT_RATIO",
    "TERMINAL_TX_COUNT_1H",
    "TERMINAL_TX_COUNT_24H",
]


def load_transactions(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load and validate the daily pandas pickle files in chronological order."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No daily transaction pickle files found in {data_dir}")

    frames = []
    for path in files:
        frame = pd.read_pickle(path)
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame = frame[REQUIRED_COLUMNS].copy()
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise")
        for column in ("TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS",
                       AMOUNT_COLUMN, LABEL_COLUMN, "TX_FRAUD_SCENARIO"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f"{path.name} contains null values in required columns")
        if not set(frame[LABEL_COLUMN].unique()).issubset({0, 1}):
            raise ValueError(f"{path.name} has non-binary {LABEL_COLUMN} values")
        frame[CUSTOMER_COLUMN] = frame[CUSTOMER_COLUMN].astype(str)
        frame[TERMINAL_COLUMN] = frame[TERMINAL_COLUMN].astype(str)
        if frame[TIMESTAMP_COLUMN].dt.date.astype(str).nunique() != 1:
            raise ValueError(f"{path.name} contains rows from multiple calendar days")
        if frame[TIMESTAMP_COLUMN].dt.date.iloc[0].isoformat() != path.stem:
            raise ValueError(f"{path.name} does not match its transaction date")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if df["TRANSACTION_ID"].duplicated().any():
        raise ValueError("TRANSACTION_ID values must be unique across all daily files")
    return df.sort_values([TIMESTAMP_COLUMN, "TRANSACTION_ID"]).reset_index(drop=True)


def _add_entity_history_features(
    df: pd.DataFrame,
    entity_column: str,
    timestamp_ns: np.ndarray,
    amount: np.ndarray,
    prefix: str,
    include_amount_sum: bool,
) -> None:
    """Add prior-only lookback counts and optional prior amount sum in-place."""
    count_1h = np.zeros(len(df), dtype=np.float64)
    count_24h = np.zeros(len(df), dtype=np.float64)
    amount_sum_1h = np.zeros(len(df), dtype=np.float64)
    for _, group in df.groupby(entity_column, sort=False).indices.items():
        positions = np.asarray(group, dtype=np.int64)
        times = timestamp_ns[positions]
        amounts = amount[positions]
        offsets = np.arange(len(positions))
        left_1h = np.searchsorted(times, times - 3_600_000_000_000, side="left")
        left_24h = np.searchsorted(times, times - 86_400_000_000_000, side="left")
        count_1h[positions] = offsets - left_1h
        count_24h[positions] = offsets - left_24h
        if include_amount_sum:
            cumulative_amounts = np.concatenate(([0.0], np.cumsum(amounts)))
            amount_sum_1h[positions] = (
                cumulative_amounts[offsets] - cumulative_amounts[left_1h]
            )

    df[f"{prefix}_TX_COUNT_1H"] = count_1h
    df[f"{prefix}_TX_COUNT_24H"] = count_24h
    if include_amount_sum:
        df[f"{prefix}_AMOUNT_SUM_1H"] = amount_sum_1h


def _add_customer_amount_history(df: pd.DataFrame) -> None:
    """Compute transaction amount statistics from earlier transactions only."""
    amount = df[AMOUNT_COLUMN].astype(np.float64)
    by_customer = df.groupby(CUSTOMER_COLUMN, sort=False)[AMOUNT_COLUMN]
    prior_count = by_customer.cumcount().to_numpy(dtype=np.float64)
    prior_sum = (by_customer.cumsum() - amount).to_numpy(dtype=np.float64)
    prior_square_sum = (
        df.assign(_amount_squared=amount * amount)
        .groupby(CUSTOMER_COLUMN, sort=False)["_amount_squared"]
        .cumsum()
        .to_numpy(dtype=np.float64)
        - np.square(amount.to_numpy(dtype=np.float64))
    )
    safe_count = np.maximum(prior_count, 1.0)
    prior_mean = prior_sum / safe_count
    denominator = np.maximum(prior_count - 1.0, 1.0)
    variance = np.maximum(
        (prior_square_sum - np.square(prior_sum) / safe_count) / denominator,
        0.0,
    )
    prior_std = np.sqrt(variance)
    has_history = prior_count > 0
    has_variation = prior_std > 1e-9

    zscore = np.zeros(len(df), dtype=np.float64)
    zscore[has_history & has_variation] = (
        amount.to_numpy(dtype=np.float64)[has_history & has_variation]
        - prior_mean[has_history & has_variation]
    ) / prior_std[has_history & has_variation]
    ratio = np.ones(len(df), dtype=np.float64)
    valid_mean = has_history & (prior_mean > 1e-9)
    ratio[valid_mean] = (
        amount.to_numpy(dtype=np.float64)[valid_mean] / prior_mean[valid_mean]
    )
    df["CUSTOMER_AMOUNT_ZSCORE"] = zscore
    df["CUSTOMER_AMOUNT_RATIO"] = ratio


def build_features(data_dir: str | Path = DATA_DIR,
                   out_path: str | Path = FEATURE_PATH) -> pd.DataFrame:
    """Load daily files, build leakage-aware features, and write Parquet."""
    df = load_transactions(data_dir)
    amount = df[AMOUNT_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError(f"{AMOUNT_COLUMN} must contain finite non-negative values")

    df["IS_NEW_TERMINAL"] = ~df.duplicated([CUSTOMER_COLUMN, TERMINAL_COLUMN])
    timestamp_ns = df[TIMESTAMP_COLUMN].astype("datetime64[ns]").astype("int64").to_numpy()
    seconds_in_day = (
        df[TIMESTAMP_COLUMN].dt.hour.to_numpy() * 3600
        + df[TIMESTAMP_COLUMN].dt.minute.to_numpy() * 60
        + df[TIMESTAMP_COLUMN].dt.second.to_numpy()
    )
    angle = 2 * np.pi * seconds_in_day / 86400
    df["HOUR_SIN"] = np.sin(angle)
    df["HOUR_COS"] = np.cos(angle)

    _add_entity_history_features(
        df, CUSTOMER_COLUMN, timestamp_ns, amount, "CUSTOMER", include_amount_sum=True
    )
    _add_entity_history_features(
        df, TERMINAL_COLUMN, timestamp_ns, amount, "TERMINAL", include_amount_sum=False
    )
    _add_customer_amount_history(df)

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Engineered model features contain non-finite values")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    fraud_count = int(df[LABEL_COLUMN].sum())
    print(
        f"[feature_engineering] wrote {out_path} with {len(df):,} rows, "
        f"{fraud_count:,} frauds ({fraud_count / len(df):.4%}), "
        f"{len(FEATURE_COLUMNS)} model features"
    )
    return df


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
):
    """Split by whole calendar days into earlier train, validation, and test."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("test_fraction and validation_fraction must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than 1")
    if TIMESTAMP_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Data must contain {TIMESTAMP_COLUMN} and {LABEL_COLUMN}")

    dates = pd.Index(df[TIMESTAMP_COLUMN].dt.normalize().drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("At least three distinct transaction dates are needed for splitting")
    train_end = max(1, int(len(dates) * (1 - test_fraction - validation_fraction)))
    validation_end = max(train_end + 1, int(len(dates) * (1 - test_fraction)))
    if validation_end >= len(dates):
        raise ValueError("Not enough dates for the requested train/validation/test split")
    train_cutoff = dates[train_end]
    test_cutoff = dates[validation_end]
    train = df[df[TIMESTAMP_COLUMN] < train_cutoff].copy()
    validation = df[
        (df[TIMESTAMP_COLUMN] >= train_cutoff)
        & (df[TIMESTAMP_COLUMN] < test_cutoff)
    ].copy()
    test = df[df[TIMESTAMP_COLUMN] >= test_cutoff].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    build_features()
