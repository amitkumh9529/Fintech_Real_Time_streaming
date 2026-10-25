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
import sys
import time
from pathlib import Path

import pandas as pd

from . import monitoring
from .db import PredictionLog, get_session, init_db
from .predict import FEATURE_COLUMNS, apply_rule_overrides, decide

logger = logging.getLogger("fraud-simulator")

FEATURES_PATH = Path(__file__).parent.parent.parent / "data" / "creditcard_features.parquet"
sys.path.append(str(Path(__file__).parent.parent.parent / "data_pipeline"))
from feature_engineering import stratified_split  # noqa: E402

_state = {
    "running": False,
    "rows_per_second": 1000.0,
    "max_rows": 3000,
    "task": None,
}


def _load_replay_data(max_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    _, _, test_df = stratified_split(df)
    replay = test_df.sample(n=min(max_rows, len(test_df)), random_state=42)
    return replay.sort_values("id").reset_index(drop=True)


async def run_simulation(champion_model_getter):
    """Async loop: replays transactions, scores them, writes to DB."""
    init_db()
    df = _load_replay_data(_state["max_rows"])
    logger.info(f"[simulator] replaying {len(df)} transactions "
                f"({(df['Class'] == 1).sum()} fraud) at {_state['rows_per_second']} rows/second")

    session = get_session()
    for _, row in df.iterrows():
        if not _state["running"]:
            break
        await asyncio.sleep(1 / max(_state["rows_per_second"], 1.0))

        champion = champion_model_getter()
        if champion is None:
            await asyncio.sleep(0.5)
            continue

        features = {c: float(row[c]) for c in FEATURE_COLUMNS}
        features["is_new_device"] = False

        start = time.perf_counter()
        ml_proba = champion.predict_proba(features)
        eff_proba, override, _ = apply_rule_overrides(ml_proba, features)
        decision, confidence = decide(eff_proba)
        latency_ms = (time.perf_counter() - start) * 1000

        monitoring.record_prediction(decision, eff_proba, latency_ms / 1000, override)

        try:
            log = PredictionLog(
                transaction_id=int(row["id"]),
                amount=float(row["Amount"]),
                actual_class=int(row["Class"]),
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
                      rows_per_second: float = 1000.0, max_rows: int = 3000):
    if _state["running"]:
        return False
    _state["running"] = True
    _state["rows_per_second"] = rows_per_second
    _state["max_rows"] = max_rows
    _state["task"] = loop.create_task(run_simulation(champion_model_getter))
    return True


def stop_simulation():
    _state["running"] = False


def simulation_status():
    return {"running": _state["running"], "rows_per_second": _state["rows_per_second"]}
