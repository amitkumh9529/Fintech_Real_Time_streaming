"""Compare earlier and later transaction feature distributions using KS and PSI."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES_PATH = Path(__file__).parent.parent.parent / "data" / "creditcard_features.parquet"
PSI_FLAG_THRESHOLD = 0.2  # common industry rule-of-thumb: PSI > 0.2 = significant shift
KS_PVALUE_FLAG_THRESHOLD = 0.05

MONITORED_FEATURES = [
    "TX_AMOUNT",
    "CUSTOMER_TX_COUNT_1H",
    "CUSTOMER_AMOUNT_ZSCORE",
    "TERMINAL_TX_COUNT_1H",
]


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(sample_size: int = 5000) -> dict:
    from data_pipeline.feature_engineering import time_based_split

    df = pd.read_parquet(FEATURES_PATH)
    train_df, _, test_df = time_based_split(df)

    train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    results = []
    for feature in MONITORED_FEATURES:
        ref = train_sample[feature].dropna().to_numpy()
        cur = test_sample[feature].dropna().to_numpy()
        ks_stat, ks_pvalue = stats.ks_2samp(ref, cur)
        psi_value = _psi(ref, cur)
        flagged = (ks_pvalue < KS_PVALUE_FLAG_THRESHOLD) or (psi_value > PSI_FLAG_THRESHOLD)
        results.append({
            "feature": feature,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "psi": round(psi_value, 4),
            "flagged": bool(flagged),
        })

    return {
        "interpretation": (
            "Offline comparison of earlier training data with the later held-out test period; "
            "it is not a continuously updated production drift monitor."
        ),
        "psi_threshold": PSI_FLAG_THRESHOLD,
        "ks_pvalue_threshold": KS_PVALUE_FLAG_THRESHOLD,
        "features": results,
        "any_drift_detected": any(r["flagged"] for r in results),
    }
