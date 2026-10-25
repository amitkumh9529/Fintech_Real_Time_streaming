import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml_pipeline.mlflow_tracking import log_confusion_matrix, log_feature_importance


class FeatureImportanceModel:
    feature_importances_ = np.array([0.4, 0.6])


class MlflowTrackingTests(unittest.TestCase):
    def test_plot_artifacts_use_temporary_files(self):
        observed_paths = []

        def inspect_artifact(path: str) -> None:
            artifact_path = Path(path)
            self.assertTrue(artifact_path.is_file())
            self.assertEqual(artifact_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            observed_paths.append(artifact_path)

        with patch(
            "ml_pipeline.mlflow_tracking.mlflow.log_artifact",
            side_effect=inspect_artifact,
        ):
            log_confusion_matrix(np.array([[4, 1], [2, 3]]), ["legit", "fraud"])
            log_feature_importance(FeatureImportanceModel(), ["amount", "velocity"])

        self.assertEqual(len(observed_paths), 2)
        self.assertTrue(all(not path.exists() for path in observed_paths))


if __name__ == "__main__":
    unittest.main()
