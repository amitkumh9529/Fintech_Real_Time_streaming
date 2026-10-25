import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    load_transactions,
    time_based_split,
)


def make_day(day: str, first_id: int) -> pd.DataFrame:
    timestamps = pd.to_datetime([
        f"{day} 00:00:00",
        f"{day} 00:30:00",
        f"{day} 02:00:00",
        f"{day} 03:00:00",
    ])
    return pd.DataFrame({
        "TRANSACTION_ID": range(first_id, first_id + 4),
        "TX_DATETIME": timestamps,
        "CUSTOMER_ID": ["c1", "c1", "c1", "c2"],
        "TERMINAL_ID": ["t1", "t1", "t2", "t1"],
        "TX_AMOUNT": [10.0, 30.0, 20.0, 40.0],
        "TX_TIME_SECONDS": [0, 1800, 7200, 10800],
        "TX_TIME_DAYS": [0, 0, 0, 0],
        "TX_FRAUD": [0, 1, 0, 1],
        "TX_FRAUD_SCENARIO": [0, 1, 0, 1],
    })


class FeatureEngineeringTests(unittest.TestCase):
    def test_loads_daily_pickle_shards_and_validates_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            make_day("2018-04-01", 0).to_pickle(base / "2018-04-01.pkl")
            make_day("2018-04-02", 4).to_pickle(base / "2018-04-02.pkl")

            result = load_transactions(base)

            self.assertEqual(len(result), 8)
            self.assertTrue(result["TX_DATETIME"].is_monotonic_increasing)
            self.assertTrue(result["TRANSACTION_ID"].is_unique)

    def test_features_only_use_prior_transactions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            make_day("2018-04-01", 0).to_pickle(base / "2018-04-01.pkl")
            source = make_day("2018-04-02", 4)
            source["TX_TIME_SECONDS"] += 86_400
            source["TX_TIME_DAYS"] = 1
            source.to_pickle(base / "2018-04-02.pkl")
            output = base / "features.parquet"

            result = build_features(base, output)

            self.assertTrue(output.exists())
            self.assertTrue(set(FEATURE_COLUMNS).issubset(result.columns))
            customer_second_tx = result.loc[result["TRANSACTION_ID"] == 1].iloc[0]
            self.assertEqual(customer_second_tx["CUSTOMER_TX_COUNT_1H"], 1)
            self.assertEqual(customer_second_tx["CUSTOMER_TX_COUNT_24H"], 1)
            self.assertEqual(customer_second_tx["CUSTOMER_AMOUNT_SUM_1H"], 10.0)
            self.assertEqual(customer_second_tx["CUSTOMER_AMOUNT_RATIO"], 3.0)
            self.assertTrue(result.loc[result["TRANSACTION_ID"] == 0, "IS_NEW_TERMINAL"].iloc[0])
            self.assertFalse(result.loc[result["TRANSACTION_ID"] == 1, "IS_NEW_TERMINAL"].iloc[0])
            customer_tx_after_2h = result.loc[result["TRANSACTION_ID"] == 2].iloc[0]
            self.assertEqual(customer_tx_after_2h["CUSTOMER_TX_COUNT_1H"], 0)
            self.assertEqual(customer_tx_after_2h["CUSTOMER_TX_COUNT_24H"], 2)

    def test_time_split_is_ordered_and_keeps_whole_dates(self):
        frame = pd.concat(
            [make_day(f"2018-04-{day:02d}", (day - 1) * 4) for day in range(1, 11)],
            ignore_index=True,
        )

        train, validation, test = time_based_split(frame)

        self.assertEqual(
            train["TX_DATETIME"].dt.normalize().nunique()
            + validation["TX_DATETIME"].dt.normalize().nunique()
            + test["TX_DATETIME"].dt.normalize().nunique(),
            10,
        )
        self.assertLess(train["TX_DATETIME"].max(), validation["TX_DATETIME"].min())
        self.assertLess(validation["TX_DATETIME"].max(), test["TX_DATETIME"].min())
        self.assertGreater(int(train["TX_FRAUD"].sum()), 0)
        self.assertGreater(int(validation["TX_FRAUD"].sum()), 0)
        self.assertGreater(int(test["TX_FRAUD"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
