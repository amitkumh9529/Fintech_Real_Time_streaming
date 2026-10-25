"""Prepare the creditcard_2023.csv dataset for training and evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).parent.parent / "data" / "creditcard_2023.csv"
FEATURE_PATH = Path(__file__).parent.parent / "data" / "creditcard_features.parquet"
LABEL_COLUMN = "Class"
IDENTIFIER_COLUMN = "id"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def build_features(raw_path: str | Path = RAW_PATH,
                   out_path: str | Path = FEATURE_PATH) -> pd.DataFrame:
    """Validate the source schema and persist model inputs as Parquet.

    This dataset has no event timestamp or card identifier, so no temporal or
    entity-history features are fabricated.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    print(f"[feature_engineering] loading {raw_path}")
    df = pd.read_csv(raw_path)

    required = {IDENTIFIER_COLUMN, LABEL_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[IDENTIFIER_COLUMN].isna().any() or df[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError(f"{IDENTIFIER_COLUMN} must be present and unique")
    if not pd.api.types.is_numeric_dtype(df[IDENTIFIER_COLUMN]):
        raise ValueError(f"{IDENTIFIER_COLUMN} must be numeric")
    if df[LABEL_COLUMN].isna().any() or not set(df[LABEL_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must contain only non-null binary labels 0 and 1")
    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Model feature columns must not contain null values")
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Model feature columns must contain only finite values")

    df = df.sort_values(IDENTIFIER_COLUMN).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(
        f"[feature_engineering] wrote {out_path} with shape {df.shape}; "
        f"fraud rate={df[LABEL_COLUMN].mean():.4%}"
    )
    return df


def stratified_split(df: pd.DataFrame, test_fraction: float = 0.2,
                     validation_fraction: float = 0.2,
                     random_state: int = 42):
    """Return deterministic stratified train, validation, and test partitions.

    The dataset's `id` is a row identifier, not a documented timestamp, so a
    stratified random split is used instead of claiming a future-time test.
    """
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("test_fraction and validation_fraction must be between 0 and 1")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than 1")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain the {LABEL_COLUMN!r} label column")
    if IDENTIFIER_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain the {IDENTIFIER_COLUMN!r} identifier column")

    labels = df[LABEL_COLUMN].to_numpy()
    if pd.isna(labels).any() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} must contain only non-null binary labels 0 and 1")
    rng = np.random.default_rng(random_state)
    train_positions = []
    validation_positions = []
    test_positions = []
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        if len(positions) < 3:
            raise ValueError("Each class must contain at least three rows for stratified splitting")
        positions = rng.permutation(positions)
        n_test = max(1, int(round(len(positions) * test_fraction)))
        n_validation = max(1, int(round(len(positions) * validation_fraction)))
        if n_test + n_validation >= len(positions):
            raise ValueError("Not enough examples per class for the requested split fractions")
        test_positions.extend(positions[:n_test])
        validation_positions.extend(positions[n_test:n_test + n_validation])
        train_positions.extend(positions[n_test + n_validation:])

    train = df.iloc[train_positions]
    validation = df.iloc[validation_positions]
    test = df.iloc[test_positions]
    return (
        train.sort_values(IDENTIFIER_COLUMN).reset_index(drop=True),
        validation.sort_values(IDENTIFIER_COLUMN).reset_index(drop=True),
        test.sort_values(IDENTIFIER_COLUMN).reset_index(drop=True),
    )


if __name__ == "__main__":
    build_features()
