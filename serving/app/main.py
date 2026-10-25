"""
FastAPI serving app for the fraud-detection champion model.

Endpoints:
  POST /predict      - score a transaction, returns fraud_probability/decision/confidence
  GET  /model/info   - which model version is currently serving
  GET  /metrics      - Prometheus-format metrics
  GET  /health       - basic liveness check
"""
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc

from . import monitoring, simulator
from .db import PredictionLog, get_session, init_db
from .drift import check_drift
from .predict import ChampionModel, apply_rule_overrides, decide
from .schemas import ModelInfoResponse, PredictionResponse, TransactionFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-serving")

app = FastAPI(
    title="Fraud Detection Serving API",
    description="Real-time fraud scoring API serving the MLflow-selected champion model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_champion: ChampionModel | None = None
_serving_since: str | None = None


@app.on_event("startup")
def load_model():
    global _champion, _serving_since
    try:
        _champion = ChampionModel()
        _serving_since = datetime.now(timezone.utc).isoformat()
        monitoring.MODEL_INFO_GAUGE.labels(
            model_type=_champion.meta["model_type"], run_id=_champion.meta["run_id"][:8]
        ).set(1)
        logger.info(f"Loaded champion model: {_champion.meta['model_type']}")
    except FileNotFoundError as e:
        logger.warning(f"No champion model available at startup: {e}")
        _champion = None

    try:
        init_db()
    except Exception as e:
        logger.warning(f"DB init failed (dashboard history endpoints will be empty): {e}")


def _get_champion():
    return _champion


@app.post("/simulation/start")
async def start_simulation_endpoint(speed_multiplier: float = 1000.0, max_rows: int = 3000):
    import asyncio
    loop = asyncio.get_running_loop()
    started = simulator.start_simulation(loop, _get_champion, speed_multiplier, max_rows)
    if not started:
        return {"status": "already_running", **simulator.simulation_status()}
    return {"status": "started", **simulator.simulation_status()}


@app.post("/simulation/stop")
def stop_simulation_endpoint():
    simulator.stop_simulation()
    return {"status": "stopped"}


@app.get("/simulation/status")
def simulation_status_endpoint():
    return simulator.simulation_status()


@app.get("/dashboard/recent-transactions")
def recent_transactions(limit: int = 50):
    try:
        session = get_session()
        rows = (
            session.query(PredictionLog)
            .order_by(desc(PredictionLog.id))
            .limit(limit)
            .all()
        )
        session.close()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "transaction_id": r.transaction_id,
                "transaction_datetime": (
                    r.transaction_datetime.isoformat() if r.transaction_datetime else None
                ),
                "amount": r.amount,
                "actual_class": r.actual_class,
                "fraud_probability": r.fraud_probability,
                "decision": r.decision,
                "confidence": r.confidence,
                "rule_override_applied": r.rule_override_applied,
                "latency_ms": r.latency_ms,
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}")


@app.get("/dashboard/metrics-summary")
def metrics_summary(window: int = 500):
    try:
        session = get_session()
        rows = (
            session.query(PredictionLog)
            .order_by(desc(PredictionLog.id))
            .limit(window)
            .all()
        )
        session.close()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}")

    if not rows:
        return {"count": 0}

    tp = sum(1 for r in rows if r.actual_class == 1 and r.decision == "block")
    fp = sum(1 for r in rows if r.actual_class == 0 and r.decision == "block")
    fn = sum(1 for r in rows if r.actual_class == 1 and r.decision != "block")
    tn = sum(1 for r in rows if r.actual_class == 0 and r.decision != "block")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    fraud_caught_amount = sum(
        r.amount for r in rows if r.actual_class == 1 and r.decision == "block"
    )
    fraud_missed_amount = sum(
        r.amount for r in rows if r.actual_class == 1 and r.decision != "block"
    )

    return {
        "count": len(rows),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "fraud_amount_caught": fraud_caught_amount,
        "fraud_amount_missed": fraud_missed_amount,
        "avg_latency_ms": sum(r.latency_ms for r in rows) / len(rows),
    }


@app.get("/dashboard/drift")
def drift_endpoint():
    try:
        return check_drift()
    except Exception as e:
        raise HTTPException(500, f"Drift check failed: {e}")


@app.get("/dashboard/roc-curve")
def roc_curve_endpoint():
    """Computes ROC and PR curve points from the real held-out test set."""
    if _champion is None:
        raise HTTPException(503, "No champion model loaded.")
    from data_pipeline.feature_engineering import FEATURE_PATH, time_based_split
    from .predict import FEATURE_COLUMNS as FC  # noqa
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve

    df = pd.read_parquet(FEATURE_PATH)
    _, _, test_df = time_based_split(df)
    X = test_df[FC]
    y = test_df["TX_FRAUD"].values
    x_in = _champion.scaler.transform(X) if _champion.scaler is not None else X
    proba = _champion.model.predict_proba(x_in)[:, 1]

    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)

    # downsample to ~60 points for a light payload
    def downsample(a, b, n=60):
        if len(a) <= n:
            return list(zip(a.tolist(), b.tolist()))
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return list(zip(a[idx].tolist(), b[idx].tolist()))

    roc_points = [{"fpr": f, "tpr": t} for f, t in downsample(fpr, tpr)]
    pr_points = [{"recall": r, "precision": p} for r, p in downsample(rec, prec)]

    return {"roc": roc_points, "pr": pr_points}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _champion is not None}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    if _champion is None:
        raise HTTPException(503, "No champion model loaded. Run the training pipeline first.")
    return ModelInfoResponse(
        model_type=_champion.meta["model_type"],
        run_id=_champion.meta["run_id"],
        metrics=_champion.meta["metrics"],
        serving_since=_serving_since,
    )


@app.get("/metrics")
def metrics():
    return Response(content=monitoring.metrics_payload(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    if _champion is None:
        raise HTTPException(503, "No champion model loaded. Run the training pipeline first.")

    start = time.perf_counter()
    features = transaction.model_dump()

    ml_probability = _champion.predict_proba(features)
    effective_probability, override_applied, override_reason = apply_rule_overrides(
        ml_probability, features
    )
    decision, confidence = decide(effective_probability)

    latency_ms = (time.perf_counter() - start) * 1000
    monitoring.record_prediction(
        decision=decision,
        probability=effective_probability,
        latency_seconds=latency_ms / 1000,
        rule_override=override_applied,
    )

    return PredictionResponse(
        fraud_probability=effective_probability,
        decision=decision,
        confidence=confidence,
        model_version=_champion.meta["run_id"][:8],
        rule_override_applied=override_applied,
        rule_override_reason=override_reason,
        latency_ms=latency_ms,
    )
