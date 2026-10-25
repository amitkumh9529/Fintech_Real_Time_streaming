"""Shared MLflow tracking utilities used by train.py / evaluate.py."""
import os
import tempfile
from pathlib import Path

import mlflow

MLRUNS_PATH = Path(__file__).parent.parent / "mlruns"
MLRUNS_PATH.mkdir(exist_ok=True)
MLFLOW_DB_PATH = MLRUNS_PATH / "mlflow.db"
EXPERIMENT_NAME = "fraud-detection"


def init_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.resolve()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_confusion_matrix(cm, labels, artifact_name="confusion_matrix.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / artifact_name
            fig.savefig(out_path)
            mlflow.log_artifact(str(out_path))
    finally:
        plt.close(fig)


def log_feature_importance(model, feature_names, artifact_name="feature_importance.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        return

    order = list(reversed(importances.argsort()[-15:]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([feature_names[i] for i in order], importances[order])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    fig.tight_layout()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / artifact_name
            fig.savefig(out_path)
            mlflow.log_artifact(str(out_path))
    finally:
        plt.close(fig)
