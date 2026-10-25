import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    stratified_split,
)


class FeatureEngineeringTests(unittest.TestCase):
    def make_dataset(self, rows=100):
        return pd.DataFrame({
            "id": range(rows),
            **{column: [float(i) for i in range(rows)] for column in FEATURE_COLUMNS},
            "Class": [i % 2 for i in range(rows)],
        })

    def test_build_features_validates_and_writes_source_schema(self):
        source = self.make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "features.parquet"
            source.to_csv(raw_path, index=False)

            result = build_features(raw_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(list(result.columns), list(source.columns))
            self.assertEqual(len(pd.read_parquet(output_path)), len(source))

    def test_stratified_split_is_reproducible_disjoint_and_keeps_label_balance(self):
        source = self.make_dataset()
        train, validation, test = stratified_split(source)
        repeated = stratified_split(source)

        self.assertEqual([len(train), len(validation), len(test)], [60, 20, 20])
        self.assertTrue(train["Class"].value_counts(normalize=True).eq(0.5).all())
        self.assertTrue(validation["Class"].value_counts(normalize=True).eq(0.5).all())
        self.assertTrue(test["Class"].value_counts(normalize=True).eq(0.5).all())
        self.assertEqual(set(train["id"]).intersection(validation["id"]), set())
        self.assertEqual(set(train["id"]).intersection(test["id"]), set())
        self.assertEqual(set(validation["id"]).intersection(test["id"]), set())
        for actual, expected in zip((train, validation, test), repeated):
            self.assertListEqual(actual["id"].tolist(), expected["id"].tolist())

    def test_stratified_split_rejects_invalid_fraction_sum(self):
        with self.assertRaises(ValueError):
            stratified_split(self.make_dataset(), test_fraction=0.6, validation_fraction=0.4)


if __name__ == "__main__":
    unittest.main()
