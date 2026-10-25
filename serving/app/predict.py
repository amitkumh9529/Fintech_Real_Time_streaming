"""
Loads the champion model artifact and serves predictions.

Implements a simple rule-based override layer on top of the ML model output --
documented here as a production best practice: pure ML models can be blind to
hard business rules (e.g. compliance limits, known fraud rings) and a thin
deterministic layer on top gives ops teams an auditable, instantly-tunable
safety net that doesn't require retraining the model.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data_pipeline.feature_engineering import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent.parent.parent / "ml_pipeline" / "models"
DATASET_VERSION = "transaction_simulator_2018_v1"

# Rule-based override thresholds (production best practice: keep these as
# tunable config, not buried in code, so ops/risk teams can adjust without a
# model redeploy)
RULE_LARGE_NEW_TERMINAL_AMOUNT = 10_000.0
RULE_HIGH_PROB_BLOCK_THRESHOLD = 0.85
RULE_LOW_PROB_ALLOW_THRESHOLD = 0.20


class ChampionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = None
        self.loaded_at = None
        self._load()

    def _load(self):
        meta_path = MODELS_DIR / "champion_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No champion model found at {meta_path}. Run ml_pipeline/train.py "
                "then ml_pipeline/select_champion.py first."
            )
        self.meta = json.loads(meta_path.read_text())
        if self.meta.get("dataset_version") != DATASET_VERSION:
            raise ValueError(
                "Champion artifact was not trained for the current transaction dataset. "
                "Retrain and run ml_pipeline/select_champion.py."
            )
        if self.meta.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError(
                "Champion artifact feature schema does not match the serving API. "
                "Retrain and run ml_pipeline/select_champion.py."
            )
        model_type = self.meta["model_type"]

        artifact = joblib.load(MODELS_DIR / "champion.joblib")
        if model_type == "logistic_regression":
            self.model = artifact["model"]
            self.scaler = artifact["scaler"]
        else:
            self.model = artifact
            self.scaler = None
        self.loaded_at = time.time()

    def predict_proba(self, features: dict) -> float:
        x = np.array([[features[c] for c in FEATURE_COLUMNS]])
        if self.scaler is not None:
            x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)


def apply_rule_overrides(ml_probability: float, features: dict) -> tuple[float, bool, str | None]:
    """
    Returns (effective_probability, override_applied, reason).
    Rules can only push the decision toward "block" (conservative safety net);
    they never silently downgrade a high ML score, only escalate clear-cut cases.
    """
    amount = features.get("TX_AMOUNT", 0)
    is_new_terminal = features.get("is_new_terminal", False)

    if is_new_terminal and amount > RULE_LARGE_NEW_TERMINAL_AMOUNT:
        return (
            1.0,
            True,
            f"Rule: transaction amount > {RULE_LARGE_NEW_TERMINAL_AMOUNT:,.0f} "
            "dataset units from a new/unseen terminal",
        )

    return ml_probability, False, None


def decide(probability: float) -> tuple[str, float]:
    """Maps a fraud probability to a business decision + confidence score."""
    if probability >= RULE_HIGH_PROB_BLOCK_THRESHOLD:
        decision = "block"
        confidence = probability
    elif probability <= RULE_LOW_PROB_ALLOW_THRESHOLD:
        decision = "allow"
        confidence = 1 - probability
    else:
        decision = "review"
        confidence = 1 - abs(probability - 0.5) * 2  # lowest confidence near 0.5
    return decision, float(confidence)
