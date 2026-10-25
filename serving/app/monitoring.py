"""Prometheus metrics for the fraud-detection serving API."""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

REGISTRY = CollectorRegistry()

PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of predictions served, labeled by decision",
    ["decision"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Latency of /predict requests in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

FRAUD_PROBABILITY_HIST = Histogram(
    "fraud_probability_distribution",
    "Distribution of predicted fraud probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

RULE_OVERRIDE_COUNTER = Counter(
    "fraud_rule_overrides_total",
    "Number of times the rule-based override layer changed the ML decision",
    registry=REGISTRY,
)

MODEL_INFO_GAUGE = Gauge(
    "fraud_model_info",
    "Static info about the currently serving model (value is always 1)",
    ["model_type", "run_id"],
    registry=REGISTRY,
)


def record_prediction(decision: str, probability: float, latency_seconds: float, rule_override: bool):
    PREDICTION_COUNTER.labels(decision=decision).inc()
    PREDICTION_LATENCY.observe(latency_seconds)
    FRAUD_PROBABILITY_HIST.observe(probability)
    if rule_override:
        RULE_OVERRIDE_COUNTER.inc()


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)
