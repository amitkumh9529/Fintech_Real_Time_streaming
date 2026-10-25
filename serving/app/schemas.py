from typing import Optional

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Anonymized transaction features accepted by the fraud scoring API."""
    id: Optional[int] = Field(None, description="Optional source-row identifier; not a timestamp")
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in source dataset units")

    # Optional operational context used only by the rule-based override layer.
    is_new_device: Optional[bool] = Field(
        False,
        description="Whether the caller's identity system considers this device/IP new",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str = Field(..., description="block | review | allow")
    confidence: float
    model_version: str
    rule_override_applied: bool = False
    rule_override_reason: Optional[str] = None
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_type: str
    run_id: str
    metrics: dict
    serving_since: str
