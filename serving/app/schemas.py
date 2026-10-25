from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Transaction metadata and causal features produced by the feature pipeline."""
    TRANSACTION_ID: Optional[int] = None
    TX_DATETIME: Optional[datetime] = None
    CUSTOMER_ID: Optional[str] = None
    TERMINAL_ID: Optional[str] = None
    TX_AMOUNT: float = Field(..., ge=0, description="Transaction amount in source dataset units")
    HOUR_SIN: float
    HOUR_COS: float
    CUSTOMER_TX_COUNT_1H: float = Field(..., ge=0)
    CUSTOMER_TX_COUNT_24H: float = Field(..., ge=0)
    CUSTOMER_AMOUNT_SUM_1H: float = Field(..., ge=0)
    CUSTOMER_AMOUNT_ZSCORE: float
    CUSTOMER_AMOUNT_RATIO: float = Field(..., ge=0)
    TERMINAL_TX_COUNT_1H: float = Field(..., ge=0)
    TERMINAL_TX_COUNT_24H: float = Field(..., ge=0)

    # Caller-supplied operational context; it is not a model feature.
    is_new_terminal: bool = Field(
        False,
        description="Whether the caller considers this terminal new for the customer",
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
