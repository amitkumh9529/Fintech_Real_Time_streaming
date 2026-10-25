# Fraud Detection — Champion Model Evaluation Report

**Champion model:** `logistic_regression`  (MLflow run `be8abd6a`)

**Test set:** 354,643 transactions, 3177 fraud (0.8958% fraud rate) — held out chronologically after the train and validation periods.

## Metrics (point estimate + 95% bootstrap CI, 1000 resamples)

| Metric | Value | 95% CI |
|---|---|---|
| PRECISION | 0.0198 | [0.0187, 0.0208] |
| RECALL | 0.4599 | [0.4430, 0.4763] |
| F1 | 0.0379 | [0.0359, 0.0398] |
| ROC_AUC | 0.6626 | [0.6519, 0.6742] |
| PR_AUC | 0.3222 | [0.3069, 0.3373] |

> **Note:** PR-AUC summarizes the precision/recall trade-off across thresholds. Fraud is rare in this simulator data, so review precision and recall alongside PR-AUC and the selected operating threshold.

## Confusion Matrix (test set, threshold = 0.5)

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 279,030 (TN) | 72,436 (FP) |
| **Actual Fraud** | 1,716 (FN) | 1,461 (TP) |

## Transaction Amount Summary (values are in dataset units)

| Metric | Value |
|---|---|
| Total fraud transaction amount | 415,732.81 |
| Fraud amount on correctly flagged rows | 346,520.89 |
| Fraud amount on missed rows | 69,211.92 |
| Fraud amount capture rate | 83.4% |
| # legitimate transactions incorrectly flagged | 72,436 |
| Amount on false-positive rows | 7,703,500.34 |

These totals describe amounts in the dataset only; they do not independently establish currency, prevented losses, customer-friction costs, or net business impact.
