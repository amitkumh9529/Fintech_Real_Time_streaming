# Fraud Detection ML Pipeline

A learning-oriented fraud scoring system with data preparation, model
experiments, model selection, FastAPI serving, replay/Kafka ingestion,
Prometheus metrics, PostgreSQL replay logs, and a React dashboard.

## Dataset currently used

The expected input is `data/creditcard_2023.csv`, with columns `id`, `V1`–`V28`,
`Amount`, and `Class`. The file currently in this workspace contains 568,630
rows, with 284,315 rows in each class. `id` is a unique source-row identifier;
the pipeline does **not** treat it as a timestamp.

This differs from the older Kaggle credit-card dataset previously described in
this README: that dataset has a `Time` field and a very rare fraud class. The
current dataset does not justify time-based evaluation, synthetic card
identities, rolling per-card velocity features, or SMOTE. Accordingly, this
version trains on `V1`–`V28` and `Amount` only. PR-AUC remains available as a
precision/recall summary, but the severe-imbalance rationale does not apply to
this balanced dataset.

## Architecture

```text
data/creditcard_2023.csv
        |
        +--> feature_engineering.py (pandas schema checks) --+
        |                                                    |
        +--> feature_engineering_spark.py (optional Spark) --+--> Parquet
                                                                 |
                  deterministic stratified 60/20/20 split ------+
                           |                    |
                           v                    v
                 train/Optuna/MLflow       untouched test split
                           |
                    select_champion
                           |
                 champion.joblib + metadata
                           |
             FastAPI: validate -> score -> rule -> decision
                 |             |              |
             /metrics     simulator log    React UI
             Prometheus   PostgreSQL
                           ^
           asyncio/Kafka replay paths call the API
```

The train/validation/test partitions use fixed-seed stratified random sampling.
Optuna and champion selection use validation metrics; `evaluate.py` uses the
separate test partition. Because the file has no documented event-time field,
these results are not a future-period or temporal generalization test.

The optional Spark path validates/selects the same source schema and writes
Parquet. It intentionally does not invent entity or time-window features.
The API's `is_new_device` field is optional operational context for the
deterministic override; it is not inferable from the dataset, and replay paths
set it to false.

## Components

| Area | Files | Purpose |
|---|---|---|
| Data preparation | `data_pipeline/feature_engineering.py`, `feature_engineering_spark.py` | Validate CSV schema; produce the shared Parquet input. |
| Training and tracking | `ml_pipeline/train.py`, `mlflow_tracking.py` | Fit Logistic Regression, XGBoost, and LightGBM; tune tree models against validation PR-AUC; log runs/artifacts. |
| Selection and evaluation | `ml_pipeline/select_champion.py`, `evaluate.py` | Select the best eligible validation run, export its exact model artifact, and report final test metrics and dataset-amount summaries. |
| Serving | `serving/app/` | Pydantic request validation, model inference, policy thresholds, monitoring, persistence, offline drift comparison, and replay simulation. |
| Ingestion | `data_pipeline/ingest.py`, `kafka_producer.py`, `kafka_consumer.py` | Local queue replay or optional Kafka replay; IDs order records but do not supply event-time pacing. |
| UI/monitoring | `frontend/`, `monitoring/` | React dashboard, Prometheus scrape config, and Grafana provisioning. |
| Local stack | `docker-compose.yml`, Dockerfiles | Compose services for API, Postgres, Prometheus, Grafana, frontend, MLflow, and optional training/Kafka profiles. |

## Setup and run

Prerequisites: Python 3.11, Docker Compose for the full stack, and the CSV at
`data/creditcard_2023.csv`. PySpark additionally requires Java; Node.js is
needed for frontend development.

### Local Python workflow

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

The Spark materialization alternative is:

```powershell
python data_pipeline/feature_engineering_spark.py
```

Run the frontend separately with `cd frontend`, `npm install`, and `npm run
dev`. The API defaults to port 8000 in a local run.

### Docker Compose

```powershell
docker compose up --build
docker compose --profile training run trainer
```

The Kafka replay is optional and uses the `kafka-demo` profile. API, dashboard,
MLflow, Prometheus, and Grafana ports are documented in `docker-compose.yml`.

## API and policy

| Route | Purpose |
|---|---|
| `POST /predict` | Score `V1`–`V28` and `Amount`, then return probability, decision, and latency. |
| `GET /model/info` | Return the loaded model identity and validation metrics. |
| `GET /health` | Liveness and model-loaded status. |
| `GET /metrics` | Prometheus-format counters and latency/probability histograms. |
| `POST /simulation/start`, `/simulation/stop`; `GET /simulation/status` | Replay a stratified test sample through inference and store labeled replay results. |
| `GET /dashboard/recent-transactions`, `/dashboard/metrics-summary`, `/dashboard/roc-curve`, `/dashboard/drift` | Dashboard transaction summaries, offline test curves, and train/test feature-distribution checks. |

The current policy maps scores at or above 0.85 to `block`, at or below 0.20
to `allow`, and intermediate values to `review`. A transaction over 10,000
with `is_new_device=true` is escalated by a rule override. That context must
come from a trustworthy caller; neither this dataset nor replay paths provide
it. Direct `/predict` requests update Prometheus metrics but are not persisted
to PostgreSQL; the labeled replay simulator writes the database rows.

The drift endpoint compares fixed, randomly stratified train/test samples. It
is an offline distribution diagnostic, not production temporal drift
detection. Streaming replays use a configured row rate, not event-time
intervals.

## Evaluation and limitations

`ml_pipeline/evaluate.py` writes `eval_report.md` after a model is trained and
selected. It reports precision, recall, F1, ROC-AUC, PR-AUC with bootstrap
intervals, confusion counts, and `Amount` aggregates in dataset units. The
amount column alone does not establish currency or prove prevented losses.
No current model metrics are claimed until this pipeline has been run on the
present CSV.

This project demonstrates components and local workflows; before real
financial use it would need data provenance and temporal validation,
authenticated APIs, reliable feature/context sourcing, database migrations,
stream retry/idempotency policies, calibrated decision thresholds, and
operational drift/label feedback.
