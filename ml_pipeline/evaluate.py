"""
Runs the champion model against the held-out stratified test set and produces
eval_report.md with:
  - precision / recall / F1 / ROC-AUC / PR-AUC with bootstrap 95% CIs (1000 resamples)
  - sum of Amount values for correctly flagged fraud and false-positive transactions
    (reported in dataset units; currency and real-world loss are not inferred)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "data_pipeline"))
from feature_engineering import FEATURE_PATH, stratified_split  # noqa: E402
from train import FEATURE_COLUMNS, LABEL_COLUMN  # noqa: E402

MODELS_DIR = Path(__file__).parent / "models"
REPORT_PATH = Path(__file__).parent.parent / "eval_report.md"
N_BOOTSTRAP = 1000
RNG_SEED = 42


def load_champion():
    meta = json.loads((MODELS_DIR / "champion_meta.json").read_text())
    model_type = meta["model_type"]
    if model_type == "logistic_regression":
        bundle = joblib.load(MODELS_DIR / "champion.joblib")
        return bundle["model"], bundle["scaler"], meta
    else:
        model = joblib.load(MODELS_DIR / "champion.joblib")
        return model, None, meta


def get_test_set():
    df = pd.read_parquet(FEATURE_PATH)
    _, _, test_df = stratified_split(df)
    return test_df


def bootstrap_ci(y_true, y_pred, y_proba, metric_fns, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    results = {name: [] for name in metric_fns}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if yt.sum() == 0:  # skip degenerate resamples with no positive class
            continue
        for name, fn in metric_fns.items():
            try:
                results[name].append(fn(yt, yp, ypr))
            except ValueError:
                continue

    ci = {}
    for name, vals in results.items():
        vals = np.array(vals)
        ci[name] = {
            "mean": float(vals.mean()),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
        }
    return ci


def main():
    model, scaler, meta = load_champion()
    test_df = get_test_set()
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[LABEL_COLUMN].values
    amounts = test_df["Amount"].values

    X_input = scaler.transform(X_test) if scaler is not None else X_test
    y_proba = model.predict_proba(X_input)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metric_fns = {
        "precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
        "roc_auc": lambda yt, yp, ypr: roc_auc_score(yt, ypr),
        "pr_auc": lambda yt, yp, ypr: average_precision_score(yt, ypr),
    }

    point_metrics = {name: fn(y_test, y_pred, y_proba) for name, fn in metric_fns.items()}
    ci = bootstrap_ci(y_test, y_pred, y_proba, metric_fns)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn_, tp = cm.ravel()

    # Amount totals are descriptive dataset values, not verified losses.
    correctly_flagged_mask = (y_pred == 1) & (y_test == 1)
    fraud_amount_captured = float(amounts[correctly_flagged_mask].sum())

    missed_fraud_mask = (y_pred == 0) & (y_test == 1)
    fraud_amount_missed = float(amounts[missed_fraud_mask].sum())

    false_positive_mask = (y_pred == 1) & (y_test == 0)
    n_false_positives = int(false_positive_mask.sum())
    false_positive_amount_held = float(amounts[false_positive_mask].sum())

    total_fraud_amount = float(amounts[y_test == 1].sum())

    report = build_report_markdown(
        meta=meta,
        point_metrics=point_metrics,
        ci=ci,
        cm=(tn, fp, fn_, tp),
        fraud_amount_captured=fraud_amount_captured,
        fraud_amount_missed=fraud_amount_missed,
        total_fraud_amount=total_fraud_amount,
        n_false_positives=n_false_positives,
        false_positive_amount_held=false_positive_amount_held,
        n_test=len(y_test),
        n_fraud_test=int(y_test.sum()),
    )
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")
    print(report)


def build_report_markdown(meta, point_metrics, ci, cm, fraud_amount_captured, fraud_amount_missed,
                          total_fraud_amount, n_false_positives, false_positive_amount_held,
                          n_test, n_fraud_test):
    tn, fp, fn_, tp = cm
    pct_fraud_amount_captured = (
        fraud_amount_captured / total_fraud_amount * 100 if total_fraud_amount else 0
    )

    lines = []
    lines.append("# Fraud Detection — Champion Model Evaluation Report\n")
    lines.append(f"**Champion model:** `{meta['model_type']}`  (MLflow run `{meta['run_id'][:8]}`)\n")
    lines.append(f"**Test set:** {n_test:,} transactions, {n_fraud_test} fraud "
                 f"({n_fraud_test / n_test:.4%} fraud rate) — held out by deterministic "
                 f"stratified random split (seed {RNG_SEED}); `id` is not treated as time.\n")

    lines.append("## Metrics (point estimate + 95% bootstrap CI, 1000 resamples)\n")
    lines.append("| Metric | Value | 95% CI |")
    lines.append("|---|---|---|")
    for name in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        v = point_metrics[name]
        c = ci[name]
        lines.append(f"| {name.upper()} | {v:.4f} | [{c['ci_low']:.4f}, {c['ci_high']:.4f}] |")
    lines.append("")
    lines.append("> **Note:** PR-AUC summarizes the precision/recall trade-off across thresholds. "
                 "This dataset has a balanced label distribution, so the extreme-imbalance "
                 "rationale from the original dataset does not apply.\n")

    lines.append("## Confusion Matrix (test set, threshold = 0.5)\n")
    lines.append("| | Predicted Legit | Predicted Fraud |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual Legit** | {tn:,} (TN) | {fp:,} (FP) |")
    lines.append(f"| **Actual Fraud** | {fn_:,} (FN) | {tp:,} (TP) |\n")

    lines.append("## Amount Summary (values are in the dataset's units)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total fraud transaction amount | {total_fraud_amount:,.2f} |")
    lines.append(f"| Fraud amount on correctly flagged rows | {fraud_amount_captured:,.2f} |")
    lines.append(f"| Fraud amount on missed rows | {fraud_amount_missed:,.2f} |")
    lines.append(f"| Fraud amount capture rate | {pct_fraud_amount_captured:.1f}% |")
    lines.append(f"| # legitimate transactions incorrectly flagged | {n_false_positives:,} |")
    lines.append(f"| Amount on false-positive rows | {false_positive_amount_held:,.2f} |")
    lines.append("")
    lines.append(
        "These totals describe amounts in the dataset only; they do not establish currency, "
        "prevented losses, customer-friction costs, or net business impact.\n"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
