"""Quick smoke test for the serving API once the champion model is trained."""
import json

import pandas as pd
import requests

from data_pipeline.feature_engineering import FEATURE_COLUMNS

BASE = "http://localhost:8000"


def main():
    df = pd.read_parquet("data/creditcard_features.parquet")
    sample = df[df["TX_FRAUD"] == 1].iloc[0]
    legit = df[df["TX_FRAUD"] == 0].iloc[0]
    for label, row in [("fraud", sample), ("legit", legit)]:
        payload = {c: float(row[c]) for c in FEATURE_COLUMNS}
        payload["TRANSACTION_ID"] = int(row["TRANSACTION_ID"])
        payload["TX_DATETIME"] = row["TX_DATETIME"].isoformat()
        payload["CUSTOMER_ID"] = str(row["CUSTOMER_ID"])
        payload["TERMINAL_ID"] = str(row["TERMINAL_ID"])
        payload["is_new_terminal"] = bool(row["IS_NEW_TERMINAL"])
        resp = requests.post(f"{BASE}/predict", json=payload)
        resp.raise_for_status()
        print(f"--- {label} transaction ---")
        print(json.dumps(resp.json(), indent=2))

    print("--- /model/info ---")
    print(json.dumps(requests.get(f"{BASE}/model/info").json(), indent=2))


if __name__ == "__main__":
    main()
