"""
Compares all MLflow runs in the fraud-detection experiment and selects the
champion model by PR-AUC (the correct metric for highly imbalanced fraud
data -- ROC-AUC is misleadingly high even for weak models here).

Registers the champion in the MLflow Model Registry and copies its serialized
artifact to ml_pipeline/models/champion.joblib + champion_meta.json for the
serving layer to load without needing a live MLflow server at inference time.
"""
import json
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(str(Path(__file__).parent))
from mlflow_tracking import init_mlflow, EXPERIMENT_NAME  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"
REGISTRY_MODEL_NAME = "fraud-detection-champion"

MODEL_FILE_MAP = {
    "xgboost": "xgboost.joblib",
    "lightgbm": "lightgbm.joblib",
    "logistic_regression": "logistic_regression.joblib",
}


def select_champion():
    init_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"No MLflow experiment '{EXPERIMENT_NAME}' found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.evaluation_split = 'validation'",
        order_by=["metrics.pr_auc DESC"],
    )
    if not runs:
        raise RuntimeError("No runs found. Run train.py first.")

    print("=== All runs ranked by PR-AUC ===")
    for r in runs:
        model_type = r.data.params.get("model_type", "unknown")
        pr_auc = r.data.metrics.get("pr_auc", float("nan"))
        roc_auc = r.data.metrics.get("roc_auc", float("nan"))
        print(f"  run_id={r.info.run_id[:8]}  model={model_type:20s} pr_auc={pr_auc:.4f}  roc_auc={roc_auc:.4f}")

    best_run = runs[0]
    model_type = best_run.data.params.get("model_type")
    print(f"\nChampion: {model_type} (run_id={best_run.info.run_id}, pr_auc={best_run.data.metrics['pr_auc']:.4f})")

    # Register in MLflow Model Registry
    model_uri = f"runs:/{best_run.info.run_id}/model"
    try:
        mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
        print(f"Registered model '{REGISTRY_MODEL_NAME}' in MLflow Model Registry.")
    except Exception as e:
        print(f"[select_champion] registry step skipped/failed (non-fatal for serving): {e}")

    # Export the exact artifact from the selected run, not a potentially stale
    # per-model file left by a later training invocation.
    artifact_name = MODEL_FILE_MAP[model_type]
    src_file = Path(
        client.download_artifacts(
            best_run.info.run_id,
            f"serving_artifact/{artifact_name}",
        )
    )
    champion_file = MODELS_DIR / "champion.joblib"
    shutil.copy(src_file, champion_file)

    meta = {
        "model_type": model_type,
        "run_id": best_run.info.run_id,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
    }
    with open(MODELS_DIR / "champion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Champion artifact written to {champion_file}")
    print(f"Champion metadata written to {MODELS_DIR / 'champion_meta.json'}")
    return meta


if __name__ == "__main__":
    select_champion()
