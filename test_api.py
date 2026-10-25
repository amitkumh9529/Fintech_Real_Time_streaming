"""Quick smoke test for the serving API once the champion model is trained."""
import json

import pandas as pd
import requests

BASE = "http://localhost:8000"


def main():
    df = pd.read_parquet("data/creditcard_features.parquet")
    sample = df[df["Class"] == 1].iloc[0]  # a known fraud case
    legit = df[df["Class"] == 0].iloc[0]

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    for label, row in [("fraud", sample), ("legit", legit)]:
        payload = {c: float(row[c]) for c in feature_cols}
        payload["id"] = int(row["id"])
        payload["is_new_device"] = False
        resp = requests.post(f"{BASE}/predict", json=payload)
        print(f"--- {label} transaction ---")
        print(json.dumps(resp.json(), indent=2))

    print("--- /model/info ---")
    print(json.dumps(requests.get(f"{BASE}/model/info").json(), indent=2))


if __name__ == "__main__":
    main()
