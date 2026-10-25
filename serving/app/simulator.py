"""
Background simulation engine used by the monitoring dashboard.

Replays labeled transactions from the held-out test partition (so the dashboard
can compute replay precision/recall/FPR) through the
champion model + rule layer, logging each prediction to Postgres via db.py.
This stands in for the Kafka/streaming-ingestion layer per project scope
(async in-process loop instead of a broker).
"""
import asyncio
import logging
import time
from pathlib import Path

import pandas as pd

from . import monitoring
from .db import PredictionLog, get_session, init_db
from .predict import FEATURE_COLUMNS, apply_rule_overrides, decide

logger = logging.getLogger("fraud-simulator")

FEATURES_PATH = Path(__file__).parent.parent.parent / "data" / "creditcard_features.parquet"
from data_pipeline.feature_engineering import time_based_split

_state = {
    "running": False,
    "speed_multiplier": 1000.0,
    "max_rows": 3000,
    "task": None,
}


def _load_replay_data(max_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    _, _, test_df = time_based_split(df)
    return test_df.tail(max_rows).reset_index(drop=True)


async def run_simulation(champion_model_getter):
    """Async loop: replays transactions, scores them, writes to DB."""
    init_db()
    df = _load_replay_data(_state["max_rows"])
    logger.info(f"[simulator] replaying {len(df)} transactions "
                f"({(df['TX_FRAUD'] == 1).sum()} fraud) at {_state['speed_multiplier']}x speed")

    session = get_session()
    previous_datetime = None
    for _, row in df.iterrows():
        if not _state["running"]:
            break
        current_datetime = row["TX_DATETIME"]
        if previous_datetime is not None:
            delta = max(0.0, (current_datetime - previous_datetime).total_seconds())
            await asyncio.sleep(min(delta / max(_state["speed_multiplier"], 1.0), 0.3))
        previous_datetime = current_datetime

        champion = champion_model_getter()
        if champion is None:
            await asyncio.sleep(0.5)
            continue

        features = {c: float(row[c]) for c in FEATURE_COLUMNS}
        features["is_new_terminal"] = bool(row["IS_NEW_TERMINAL"])

        start = time.perf_counter()
        ml_proba = champion.predict_proba(features)
        eff_proba, override, _ = apply_rule_overrides(ml_proba, features)
        decision, confidence = decide(eff_proba)
        latency_ms = (time.perf_counter() - start) * 1000

        monitoring.record_prediction(decision, eff_proba, latency_ms / 1000, override)

        try:
            log = PredictionLog(
                transaction_id=int(row["TRANSACTION_ID"]),
                transaction_datetime=current_datetime.to_pydatetime(),
                amount=float(row["TX_AMOUNT"]),
                actual_class=int(row["TX_FRAUD"]),
                fraud_probability=eff_proba,
                decision=decision,
                confidence=confidence,
                model_version=champion.meta["run_id"][:8],
                rule_override_applied=override,
                latency_ms=latency_ms,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.warning(f"[simulator] DB write failed (continuing without persistence): {e}")
            session.rollback()

    session.close()
    _state["running"] = False
    logger.info("[simulator] replay finished")


def start_simulation(loop: asyncio.AbstractEventLoop, champion_model_getter,
                      speed_multiplier: float = 1000.0, max_rows: int = 3000):
    if _state["running"]:
        return False
    _state["running"] = True
    _state["speed_multiplier"] = speed_multiplier
    _state["max_rows"] = max_rows
    _state["task"] = loop.create_task(run_simulation(champion_model_getter))
    return True


def stop_simulation():
    _state["running"] = False


def simulation_status():
    return {"running": _state["running"], "speed_multiplier": _state["speed_multiplier"]}
