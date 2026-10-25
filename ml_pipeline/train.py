"""
Trains 3 candidate fraud models (XGBoost, LightGBM, Logistic Regression
baseline) with light Optuna hyperparameter search, logging every run (params,
metrics, confusion matrix, feature importance) to MLflow.

Usage:
    python ml_pipeline/train.py [--trials 20]
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "data_pipeline"))

from mlflow_tracking import init_mlflow, log_confusion_matrix, log_feature_importance  # noqa: E402
from feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_PATH,
    LABEL_COLUMN,
    build_features,
    time_based_split,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    feature_path = Path(FEATURE_PATH)
    if not feature_path.exists():
        print("[train] feature parquet not found, building features from daily pickle shards...")
        build_features()
    df = pd.read_parquet(feature_path)
    train_df, validation_df, test_df = time_based_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN]
    X_validation, y_validation = validation_df[FEATURE_COLUMNS], validation_df[LABEL_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[LABEL_COLUMN]
    print(f"[train] train={len(X_train):,} (fraud={y_train.sum()}) "
          f"validation={len(X_validation):,} (fraud={y_validation.sum()}) "
          f"test={len(X_test):,} (fraud={y_test.sum()})")
    if min(y_train.sum(), y_validation.sum(), y_test.sum()) == 0:
        raise ValueError("Each chronological partition must contain fraud examples")
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def evaluate_preds(y_true, y_pred, y_proba):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def train_xgboost(X_train, y_train, X_validation, y_validation, n_trials):
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = xgb.XGBClassifier(
            **params, scale_pos_weight=positive_weight,
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_validation)[:, 1]
        return average_precision_score(y_validation, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print(f"[train:xgboost] best PR-AUC={study.best_value:.4f} params={best_params}")

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("dataset_version", "transaction_simulator_2018_v1")
        mlflow.log_param("scale_pos_weight", positive_weight)
        model = xgb.XGBClassifier(
            **best_params, scale_pos_weight=positive_weight,
            eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_validation)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = evaluate_preds(y_validation, preds, proba)
        mlflow.log_metrics(metrics)
        mlflow.log_param("evaluation_split", "validation")
        cm = confusion_matrix(y_validation, preds)
        log_confusion_matrix(cm, ["legit", "fraud"])
        log_feature_importance(model, FEATURE_COLUMNS)
        mlflow.xgboost.log_model(model, "model")

        artifact_path = MODELS_DIR / "xgboost.joblib"
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="serving_artifact")
        print(f"[train:xgboost] metrics={metrics}")
        return "xgboost", metrics, model


def train_lightgbm(X_train, y_train, X_validation, y_validation, n_trials):
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = lgb.LGBMClassifier(
            **params, scale_pos_weight=positive_weight,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_validation)[:, 1]
        return average_precision_score(y_validation, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    print(f"[train:lightgbm] best PR-AUC={study.best_value:.4f} params={best_params}")

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("dataset_version", "transaction_simulator_2018_v1")
        mlflow.log_param("scale_pos_weight", positive_weight)
        model = lgb.LGBMClassifier(
            **best_params, scale_pos_weight=positive_weight,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_validation)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = evaluate_preds(y_validation, preds, proba)
        mlflow.log_metrics(metrics)
        mlflow.log_param("evaluation_split", "validation")
        cm = confusion_matrix(y_validation, preds)
        log_confusion_matrix(cm, ["legit", "fraud"])
        log_feature_importance(model, FEATURE_COLUMNS)
        mlflow.lightgbm.log_model(model, "model")

        artifact_path = MODELS_DIR / "lightgbm.joblib"
        joblib.dump(model, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="serving_artifact")
        print(f"[train:lightgbm] metrics={metrics}")
        return "lightgbm", metrics, model


def train_logistic_regression(X_train, y_train, X_validation, y_validation):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    with mlflow.start_run(run_name="logistic_regression"):
        params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("dataset_version", "transaction_simulator_2018_v1")
        mlflow.log_param("evaluation_split", "validation")

        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_validation_scaled)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = evaluate_preds(y_validation, preds, proba)
        mlflow.log_metrics(metrics)
        cm = confusion_matrix(y_validation, preds)
        log_confusion_matrix(cm, ["legit", "fraud"])
        log_feature_importance(model, FEATURE_COLUMNS)
        mlflow.sklearn.log_model(model, "model")

        artifact_path = MODELS_DIR / "logistic_regression.joblib"
        joblib.dump({"model": model, "scaler": scaler}, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="serving_artifact")
        print(f"[train:logreg] metrics={metrics}")
        return "logistic_regression", metrics, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    init_mlflow()
    X_train, y_train, X_validation, y_validation, _, _ = load_data()

    results = {}
    for name, metrics, _ in [
        train_xgboost(X_train, y_train, X_validation, y_validation, args.trials),
        train_lightgbm(X_train, y_train, X_validation, y_validation, args.trials),
        train_logistic_regression(X_train, y_train, X_validation, y_validation),
    ]:
        results[name] = metrics

    print("\n=== Training complete. Summary (PR-AUC summarizes the precision/recall trade-off) ===")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["pr_auc"]):
        print(f"  {name:20s} pr_auc={m['pr_auc']:.4f}  roc_auc={m['roc_auc']:.4f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
