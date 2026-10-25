# Fraud Detection ML Pipeline

A learning-oriented fraud scoring system built around dated transaction data:
daily source files, causal customer/terminal features, chronological model
validation, MLflow experiment tracking, FastAPI scoring, replay/Kafka
ingestion, Prometheus metrics, PostgreSQL replay logs, and a React dashboard.

## Current dataset

`data/` contains 183 daily pandas pickle files from `2018-04-01.pkl` through
`2018-09-30.pkl`. Inspection found **1,754,155 transactions** and **14,681
fraudulent transactions** (0.8369%). The daily files share the columns:

| Column | Meaning |
|---|---|
| `TRANSACTION_ID` | Unique transaction row identifier. |
| `TX_DATETIME` | Transaction timestamp; supports chronological splits and replay. |
| `CUSTOMER_ID` | Customer identifier used to group customer-history features. |
| `TERMINAL_ID` | Terminal identifier used to group terminal velocity features. |
| `TX_AMOUNT` | Transaction amount in the source dataset's units; currency is not inferred here. |
| `TX_TIME_SECONDS`, `TX_TIME_DAYS` | Numeric elapsed-time fields supplied with each transaction. |
| `TX_FRAUD` | Binary target label: 0 or 1. |
| `TX_FRAUD_SCENARIO` | Scenario code; retained for analysis but excluded from model inputs to prevent target leakage. |

All inspected shards had the same schema and no nulls in required columns.
There are 4,990 distinct customer IDs and 10,000 terminal IDs. Fraud is rare
enough that PR-AUC, precision, recall, and threshold choice matter; the code
uses class weighting rather than applying SMOTE to this simulator dataset.

## Architecture

```text
183 daily pickle shards
          |
          v
load_transactions: validate, type-normalize, concatenate, timestamp-sort
          |
          v
causal feature engineering (prior customer/terminal history only)
          |
          +--> data/creditcard_features.parquet
                    |
                    +--> chronological whole-day train / validation / test
                             |
                  Optuna tunes on validation only
                             |
                  MLflow records candidate runs/artifacts
                             |
                    champion selection by validation PR-AUC
                             |
                    evaluate untouched later test period
                             |
          FastAPI: validate -> score -> optional rule -> decision
                |                 |                |
            Prometheus         replay logs       React dashboard
                               PostgreSQL
                    ^
               asyncio / Kafka replay
```

The pandas feature pipeline is the default because the source format is
pickle. The optional Spark script first loads the trusted pickle shards with
pandas, then uses Spark windows for feature computation; this is a learning
adapter, not distributed source ingestion.

## Features

Model inputs are `TX_AMOUNT`, `HOUR_SIN`, `HOUR_COS`,
`CUSTOMER_TX_COUNT_1H`, `CUSTOMER_TX_COUNT_24H`,
`CUSTOMER_AMOUNT_SUM_1H`, `CUSTOMER_AMOUNT_ZSCORE`,
`CUSTOMER_AMOUNT_RATIO`, `TERMINAL_TX_COUNT_1H`, and
`TERMINAL_TX_COUNT_24H`.

The two rolling counts, the one-hour amount sum, and the customer amount
history statistics use **prior transactions only**. `TX_FRAUD_SCENARIO`,
customer/terminal identifiers, transaction ID, and raw timestamp are not model
inputs. Customer and terminal identifiers are used only to calculate causal
history signals. `IS_NEW_TERMINAL` is a prior-history context flag for the
rule layer, not a model feature.

Train, validation, and test partitions are chronological and split on whole
calendar days, approximately 60/20/20. Optuna and champion selection use the
validation period; the final evaluation uses the later, untouched test period.

## Components

| Area | Files | Responsibility |
|---|---|---|
| Source loading and features | `data_pipeline/feature_engineering.py` | Validate daily shards, merge them chronologically, calculate prior-only features, write Parquet, and split by date. |
| Optional Spark windows | `data_pipeline/feature_engineering_spark.py` | Apply the same feature concepts using Spark SQL windows after loading source pickles with pandas. |
| Training and tracking | `ml_pipeline/train.py`, `mlflow_tracking.py` | Train Logistic Regression, XGBoost, and LightGBM; tune tree models on validation PR-AUC; log runs and artifacts. |
| Selection and evaluation | `ml_pipeline/select_champion.py`, `evaluate.py` | Select an eligible validation run, export its exact artifact, and evaluate on the chronological test period. |
| Serving | `serving/app/` | Validate requests, load the model, apply decision logic, expose metrics and dashboard endpoints, and persist replay predictions. |
| Ingestion | `data_pipeline/ingest.py`, `kafka_producer.py`, `kafka_consumer.py` | Replay feature rows through an async queue or optional Kafka path. |
| UI and monitoring | `frontend/`, `monitoring/` | React dashboard, Prometheus scrape configuration, and Grafana provisioning. |
| Local orchestration | `docker-compose.yml`, Dockerfiles | Run the API, PostgreSQL, monitoring, frontend, MLflow, optional training, and Kafka services. |

## Setup and run

Prerequisites: Python 3.11, the daily `.pkl` files under `data/`, and Docker
Compose for the full stack. The optional Spark path also needs Java.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python data_pipeline/feature_engineering.py
python ml_pipeline/train.py --trials 20
python ml_pipeline/select_champion.py
python ml_pipeline/evaluate.py
uvicorn serving.app.main:app --reload
```

Optional Spark feature path:

```powershell
python data_pipeline/feature_engineering_spark.py
```

Run the frontend from `frontend/` with `npm install` and `npm run dev`.
Docker Compose exposes the API, dashboard, MLflow, Prometheus, and Grafana on
the ports configured in `docker-compose.yml`.

## API and decisions

| Route | Purpose |
|---|---|
| `POST /predict` | Score the 10 engineered numeric inputs and return probability, decision, and latency. |
| `GET /model/info` | Return the loaded champion identity and its validation metrics. |
| `GET /health` | Report API liveness and whether a model loaded. |
| `GET /metrics` | Expose Prometheus counters and latency/probability histograms. |
| `/simulation/start`, `/simulation/stop`, `/simulation/status` | Replay the later test period and log labeled decisions. |
| `/dashboard/recent-transactions`, `/dashboard/metrics-summary`, `/dashboard/roc-curve`, `/dashboard/drift` | Supply dashboard history, summary metrics, test curves, and offline distribution comparisons. |

The prediction policy maps scores at or above 0.85 to `block`, at or below
0.20 to `allow`, and intermediate values to `review`. The existing large
amount/new-terminal override uses a 10,000 source-unit threshold. The inspected
data's maximum amount is 2,628, so this rule does not activate in the dataset
replay; it is not calibrated by this dataset and should be treated as a
placeholder policy until a risk owner defines an appropriate operational rule.

Direct `/predict` calls update Prometheus metrics but are not currently
persisted to PostgreSQL; the labeled simulator performs database writes.
Drift results compare fixed samples from earlier training and later test
periods; they are not a continuously updated production monitor.

## Evaluation and limitations

`ml_pipeline/evaluate.py` writes `eval_report.md` after the pipeline runs.
It reports precision, recall, F1, ROC-AUC, PR-AUC with bootstrap intervals,
confusion counts, and transaction-amount summaries in dataset units. Amount
totals alone do not establish currency or real-world losses prevented.

The pickle files are executable serialization formats; only load files from a
trusted source. The Spark adapter still loads the source data on the driver
with pandas, so it is not a distributed ingestion solution. A real deployment
would also need secured APIs, reliable feature-store/context integration,
retries and idempotency for streaming, calibrated decision policies, and a
production label-feedback process.
